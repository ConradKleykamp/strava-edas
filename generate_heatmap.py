"""
Strava Heatmap Generator
Produces three interactive HTML outputs from any Strava bulk data export:
  - strava_heatmap.html   : density heatmap of all GPS points
  - strava_calendar.html  : GitHub-style calendar heatmap of daily mileage
  - strava_animated.html  : chronological replay — runs accumulate on the map over time

Works out of the box with any user's exported strava/ folder — no code changes
required. All tuning knobs (activity type, region filtering, pace thresholds,
heatmap styling) are optional constructor arguments on StravaHeatmapGenerator;
see README.md for details. See run() at the bottom for the default entry point.
"""

import csv
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from folium.plugins import HeatMap

GPX_NS = "http://www.topografix.com/GPX/1/1"
DATA_DIR = Path("strava")
ACTIVITIES_CSV = DATA_DIR / "activities.csv"
DEFAULT_TYPES = {"run"}

# Example bounding box (min_lat, max_lat, min_lon, max_lon), included only as a
# sample for the author's own data (Massachusetts). Bounds filtering is OFF by
# default — pass `bounds=MA_BOUNDS`, or your own (min_lat, max_lat, min_lon,
# max_lon) tuple, to StravaHeatmapGenerator() if you want to restrict routes
# to a specific region.
MA_BOUNDS = (41.239, 42.886, -73.508, -69.928)

# PR distances: (display label, element id suffix, meters)
PR_DISTANCES = [
    ("400M",   "400m",  400.0),
    ("1 MILE", "1mi",   1609.344),
    ("2 MILE", "2mi",   3218.688),
    ("5K",     "5k",    5000.0),
    ("10K",    "10k",   10000.0),
    ("HALF",   "half",  21097.5),
]

# (lat, lon, timestamp)
Point = tuple[float, float, Optional[datetime]]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS coordinates."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _pace_to_hex(pace_sec_per_mile: float, fast: float, slow: float) -> str:
    """Map pace (sec/mile) to a hex color: green → yellow → red."""
    t = max(0.0, min(1.0, (pace_sec_per_mile - fast) / (slow - fast)))
    if t <= 0.5:
        s = t * 2
        r, g, b = int(s * 255), int(168 + s * 47), int(107 - s * 107)
    else:
        s = (t - 0.5) * 2
        r, g, b = int(255 - s * 63), int(215 - s * 158), int(s * 43)
    return f"#{r:02x}{g:02x}{b:02x}"


def _best_segment_time(
    cum: list[float], timed: list[Point], target_m: float
) -> Optional[float]:
    """Minimum elapsed seconds to cover target_m using a two-pointer sliding window.

    Accepts a precomputed cumulative-distance array so the caller can reuse it
    across multiple target distances without redundant haversine calls.
    Segments implying a pace faster than 5:00/mile are rejected as GPS noise.
    """
    if not cum or cum[-1] < target_m * 0.9:
        return None
    best: Optional[float] = None
    j = 0
    for i in range(len(timed)):
        if cum[-1] - cum[i] < target_m:
            break
        if j <= i:
            j = i + 1
        while j < len(timed) and cum[j] - cum[i] < target_m:
            j += 1
        if j >= len(timed):
            break
        elapsed = (timed[j][2] - timed[i][2]).total_seconds()  # type: ignore[operator]
        dist = cum[j] - cum[i]
        if elapsed > 0 and (elapsed / dist * 1609.344) >= 300:  # >= 5:00/mile
            if best is None or elapsed < best:
                best = elapsed
    return best


class StravaHeatmapGenerator:
    def __init__(
        self,
        activity_types: set[str] = DEFAULT_TYPES,
        heatmap_radius: int = 8,
        heatmap_blur: int = 10,
        heatmap_min_opacity: float = 0.3,
        default_zoom: int = 12,
        fast_pace_sec_per_mile: float = 360.0,   # 6:00/mi
        slow_pace_sec_per_mile: float = 540.0,   # 9:00/mi
        bounds: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        self.activity_types = {t.lower() for t in activity_types}
        self.heatmap_radius = heatmap_radius
        self.heatmap_blur = heatmap_blur
        self.heatmap_min_opacity = heatmap_min_opacity
        self.default_zoom = default_zoom
        self.fast_pace = fast_pace_sec_per_mile
        self.slow_pace = slow_pace_sec_per_mile
        self.bounds = bounds

        self._activities: list[dict] = []
        self._all_coords: list[tuple[float, float]] = []
        self._routes: list[list[Point]] = []
        self._included_filenames: set[str] = set()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "StravaHeatmapGenerator":
        self._activities = self._read_csv()
        print(f"Found {len(self._activities)} matching activities in CSV")
        self._routes, self._all_coords = self._parse_gpx_files()
        print(f"Parsed {len(self._routes)} routes, {len(self._all_coords):,} total GPS points")
        return self

    def _read_csv(self) -> list[dict]:
        rows = []
        with open(ACTIVITIES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["Activity Type"].lower() in self.activity_types:
                    rows.append(row)
        return rows

    def _in_bounds(self, points: list[Point]) -> bool:
        if not points or self.bounds is None:
            return True
        lat, lon = points[0][0], points[0][1]
        min_lat, max_lat, min_lon, max_lon = self.bounds
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

    def _parse_gpx_files(self) -> tuple[list[list[Point]], list[tuple[float, float]]]:
        routes: list[list[Point]] = []
        all_coords: list[tuple[float, float]] = []
        self._included_filenames = set()
        skipped = 0
        for activity in self._activities:
            filename = activity.get("Filename", "").strip()
            if not filename:
                continue
            points = self._parse_gpx(DATA_DIR / filename)
            if not points:
                continue
            if not self._in_bounds(points):
                skipped += 1
                continue
            routes.append(points)
            all_coords.extend((p[0], p[1]) for p in points)
            self._included_filenames.add(filename)
        if skipped:
            print(f"  Excluded {skipped} routes outside bounds")
        return routes, all_coords

    def _parse_gpx(self, path: Path) -> list[Point]:
        if not path.exists():
            return []
        try:
            root = ET.parse(path).getroot()
            points: list[Point] = []
            for trkpt in root.iter(f"{{{GPX_NS}}}trkpt"):
                lat, lon = trkpt.get("lat"), trkpt.get("lon")
                if lat is None or lon is None:
                    continue
                ts: Optional[datetime] = None
                time_el = trkpt.find(f"{{{GPX_NS}}}time")
                if time_el is not None and time_el.text:
                    try:
                        ts = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
                    except ValueError:
                        pass
                points.append((float(lat), float(lon), ts))
            return points
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _center(self) -> tuple[float, float]:
        lats = [c[0] for c in self._all_coords]
        lons = [c[1] for c in self._all_coords]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    def _base_map(self, center: tuple[float, float]) -> folium.Map:
        return folium.Map(location=center, zoom_start=self.default_zoom, tiles="CartoDB dark_matter")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def build_heatmap(self, output: str = "strava_heatmap.html") -> None:
        if not self._all_coords:
            print("No coordinates loaded — run .load() first")
            return
        m = self._base_map(self._center())
        HeatMap(
            self._all_coords,
            radius=self.heatmap_radius,
            blur=self.heatmap_blur,
            min_opacity=self.heatmap_min_opacity,
        ).add_to(m)
        m.save(output)
        print(f"Density heatmap saved → {output}")

    def _pace_legend_html(self) -> str:
        def fmt(sec: float) -> str:
            return f"{int(sec) // 60}:{int(sec) % 60:02d}/mi"

        mid = (self.fast_pace + self.slow_pace) / 2
        g = _pace_to_hex(self.fast_pace, self.fast_pace, self.slow_pace)
        y = _pace_to_hex(mid, self.fast_pace, self.slow_pace)
        r = _pace_to_hex(self.slow_pace, self.fast_pace, self.slow_pace)
        return f"""
        <div style="position:fixed;bottom:30px;right:10px;z-index:9999;
                    background:rgba(25,25,25,0.88);color:#ddd;
                    font-family:sans-serif;font-size:12px;
                    padding:10px 14px;border-radius:6px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.6)">
            <b style="display:block;margin-bottom:8px;letter-spacing:.5px">Pace</b>
            <div style="display:flex;align-items:center;gap:8px">
                <span>{fmt(self.fast_pace)}</span>
                <div style="width:90px;height:6px;border-radius:3px;
                    background:linear-gradient(to right,{g},{y},{r})"></div>
                <span>{fmt(self.slow_pace)}</span>
            </div>
        </div>"""

    def build_calendar_heatmap(self, output: str = "strava_calendar.html") -> None:
        """Year-per-row calendar heatmap + monthly bar chart + weekly trend line."""
        df = pd.read_csv(ACTIVITIES_CSV)
        df = df[df["Activity Type"].str.lower().isin(self.activity_types)].copy()
        if self._included_filenames:
            df = df[df["Filename"].isin(self._included_filenames)]

        df["date"] = pd.to_datetime(df["Activity Date"], format="mixed")
        df["miles"] = pd.to_numeric(df["Distance"], errors="coerce") * 0.621371
        df["date_only"] = df["date"].dt.normalize()

        def _parse_mt(val: object) -> float:
            if pd.isna(val):
                return 0.0
            parts = str(val).strip().split(":")
            try:
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                if len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                return float(parts[0])
            except (ValueError, IndexError):
                return 0.0

        df["moving_secs"] = df["Moving Time"].apply(_parse_mt)
        if "Activity Name" in df.columns:
            df["run_name"] = df["Activity Name"].fillna("")
        else:
            df["run_name"] = ""

        def _fmt_pace(secs: float, mi: float) -> str:
            if not mi or secs <= 0:
                return "—"
            pace = secs / mi
            return f"{int(pace) // 60}:{int(round(pace % 60)):02d}/mi"

        records = []
        for dt, grp in df.groupby("date_only"):
            mi = grp["miles"].sum()
            sc = grp["moving_secs"].sum()
            cnt = len(grp)
            name = grp["run_name"].iloc[0] if cnt == 1 else f"{cnt} runs"
            records.append({
                "date": dt, "miles": mi, "secs": sc,
                "pace": _fmt_pace(sc, mi), "name": str(name),
            })

        daily = pd.DataFrame(records)
        daily["date"] = pd.to_datetime(daily["date"])

        full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = (
            daily.set_index("date")
            .reindex(full_range)
            .reset_index()
            .rename(columns={"index": "date"})
        )
        daily["miles"] = daily["miles"].fillna(0.0)
        daily["pace"] = daily["pace"].fillna("—")
        daily["name"] = daily["name"].fillna("")

        years = sorted(daily["date"].dt.year.unique())
        n_years = len(years)
        global_max = max(daily["miles"].max(), 0.1)

        COLORSCALE = [
            [0.0,  "#161b22"],
            [0.01, "#3d1f00"],
            [0.2,  "#8a3a00"],
            [0.5,  "#fc4c02"],
            [1.0,  "#ff9a5c"],
        ]
        DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        row_heights = [1.0] * n_years + [2.5, 2.5]
        subtitles = [str(y) for y in years] + ["Monthly Mileage", "Weekly Mileage"]

        fig = make_subplots(
            rows=n_years + 2,
            cols=1,
            shared_xaxes=False,
            row_heights=row_heights,
            vertical_spacing=0.03,
            subplot_titles=subtitles,
        )

        for row_idx, year in enumerate(years, start=1):
            jan1 = pd.Timestamp(year, 1, 1)
            dec31 = pd.Timestamp(year, 12, 31)
            grid_start = jan1 - pd.Timedelta(days=jan1.dayofweek)
            n_weeks = (dec31 - grid_start).days // 7 + 1

            z = np.full((7, n_weeks), np.nan)
            cd_date = np.full((7, n_weeks), "", dtype=object)
            cd_detail = np.full((7, n_weeks), "", dtype=object)

            # Seed every in-year cell as a rest day
            for w in range(n_weeks):
                for d in range(7):
                    dt = grid_start + pd.Timedelta(days=w * 7 + d)
                    if dt.year == year:
                        z[d, w] = 0.0
                        cd_date[d, w] = dt.strftime("%b %d, %Y")
                        cd_detail[d, w] = "Rest"

            # Overlay actual run days
            year_data = daily[daily["date"].dt.year == year]
            for _, row in year_data.iterrows():
                dt = row["date"]
                offset = (dt - grid_start).days
                w_i, d_i = offset // 7, offset % 7
                if 0 <= w_i < n_weeks:
                    z[d_i, w_i] = row["miles"]
                    cd_date[d_i, w_i] = dt.strftime("%b %d, %Y")
                    if row["miles"] > 0:
                        detail = f"{row['miles']:.1f} mi · {row['pace']}"
                        if row["name"]:
                            detail += f"<br>{row['name']}"
                        cd_detail[d_i, w_i] = detail

            customdata = np.stack([cd_date, cd_detail], axis=-1)

            # Month tick positions: first week each month appears in the year
            tick_vals: list[int] = []
            tick_text: list[str] = []
            seen_months: set[int] = set()
            for w in range(n_weeks):
                for d in range(7):
                    dt = grid_start + pd.Timedelta(days=w * 7 + d)
                    if dt.year == year and dt.month not in seen_months:
                        tick_vals.append(w)
                        tick_text.append(dt.strftime("%b"))
                        seen_months.add(dt.month)
                        break

            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=list(range(n_weeks)),
                    y=DAY_LABELS,
                    colorscale=COLORSCALE,
                    zmin=0,
                    zmax=global_max,
                    showscale=(row_idx == 1),
                    colorbar=dict(
                        title="mi", thickness=10, len=0.12,
                        tickfont=dict(color="#ccc"),
                        title_font=dict(color="#ccc"),
                    ),
                    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
                    customdata=customdata,
                    xgap=2,
                    ygap=2,
                ),
                row=row_idx, col=1,
            )
            fig.update_xaxes(
                tickmode="array", tickvals=tick_vals, ticktext=tick_text,
                showgrid=False, zeroline=False,
                tickfont=dict(size=10, color="#aaa"),
                row=row_idx, col=1,
            )
            fig.update_yaxes(
                showgrid=False, zeroline=False, autorange="reversed",
                tickfont=dict(size=9, color="#888"),
                row=row_idx, col=1,
            )

        # Monthly bar chart
        daily["ym"] = daily["date"].dt.to_period("M")
        monthly = daily.groupby("ym")["miles"].sum().reset_index()
        monthly["month_dt"] = monthly["ym"].dt.to_timestamp()

        bar_row = n_years + 1
        fig.add_trace(
            go.Bar(
                x=monthly["month_dt"],
                y=monthly["miles"].round(1),
                marker_color="#fc4c02",
                marker_line_width=0,
                opacity=0.85,
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f} mi<extra></extra>",
                showlegend=False,
            ),
            row=bar_row, col=1,
        )
        fig.update_xaxes(
            showgrid=False, zeroline=False,
            tickformat="%b '%y", tickangle=-45,
            tickfont=dict(color="#aaa", size=10),
            row=bar_row, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="#1c2330", zeroline=False,
            tickfont=dict(color="#aaa"),
            title_text="Miles", title_font=dict(color="#888", size=11),
            row=bar_row, col=1,
        )

        # Weekly trend with 4-week rolling average
        daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.dayofweek, unit="D")
        weekly = daily.groupby("week_start")["miles"].sum().reset_index().sort_values("week_start")
        weekly["rolling4"] = weekly["miles"].rolling(4, min_periods=1).mean()

        trend_row = n_years + 2
        fig.add_trace(
            go.Scatter(
                x=weekly["week_start"], y=weekly["miles"].round(1),
                mode="lines",
                line=dict(color="#fc4c02", width=1),
                opacity=0.35,
                hovertemplate="Week of %{x|%b %d, %Y}<br>%{y:.1f} mi<extra></extra>",
                showlegend=False, name="Weekly",
            ),
            row=trend_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=weekly["week_start"], y=weekly["rolling4"].round(1),
                mode="lines",
                line=dict(color="#ff8c69", width=2.5),
                hovertemplate="4-wk avg (week of %{x|%b %d, %Y})<br>%{y:.1f} mi<extra></extra>",
                showlegend=False, name="4-wk avg",
            ),
            row=trend_row, col=1,
        )
        fig.update_xaxes(
            showgrid=False, zeroline=False,
            tickfont=dict(color="#aaa", size=10),
            row=trend_row, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="#1c2330", zeroline=False,
            tickfont=dict(color="#aaa"),
            title_text="Mi/wk", title_font=dict(color="#888", size=11),
            row=trend_row, col=1,
        )

        height = n_years * 130 + 560 + 100

        fig.update_layout(
            title=dict(text="Running Dashboard", font=dict(size=22, color="#eee"), x=0.5),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#ccc", family="sans-serif"),
            height=height,
            margin=dict(l=60, r=80, t=80, b=60),
            bargap=0.12,
            hoverlabel=dict(bgcolor="#1e2530", font_color="#eee", bordercolor="#333"),
        )

        for ann in fig.layout.annotations:
            ann.update(font=dict(color="#aaa", size=13))

        fig.write_html(output, include_plotlyjs="cdn")
        print(f"Calendar heatmap saved → {output}")

    def build_animated_routes(
        self, output: str = "strava_animated.html", coord_stride: int = 3
    ) -> None:
        """Chronological replay with route tracing, recency fading, stats ticker, and PR panel.

        For each run the best segment time is pre-computed in Python (two-pointer sliding
        window over cumulative GPS distances) and shipped in the JSON payload so the
        JavaScript only needs to compare values — no heavy computation in the browser.
        """
        if not self._routes:
            print("No routes loaded — run .load() first")
            return

        timed_routes = [(r, r[0][2]) for r in self._routes if r[0][2] is not None]
        timed_routes.sort(key=lambda x: x[1])

        run_data = []
        for route, start_ts in timed_routes:
            timed = [p for p in route if p[2] is not None]
            if len(timed) < 2:
                continue

            # Build cumulative distance array once; reuse for total_dist and all PRs.
            cum = [0.0]
            for i in range(1, len(timed)):
                cum.append(cum[-1] + _haversine(
                    timed[i - 1][0], timed[i - 1][1], timed[i][0], timed[i][1]
                ))
            total_dist = cum[-1]
            total_time = (timed[-1][2] - timed[0][2]).total_seconds()  # type: ignore[operator]
            if total_dist < 10 or total_time <= 0:
                continue

            avg_pace = (total_time / total_dist) * 1609.344
            color = _pace_to_hex(avg_pace, self.fast_pace, self.slow_pace)
            sampled = route[::coord_stride] or route

            prs = {
                key: (round(t, 1) if (t := _best_segment_time(cum, timed, dist_m)) else None)
                for _, key, dist_m in PR_DISTANCES
            }

            run_data.append({
                "coords": [[p[0], p[1]] for p in sampled],
                "color": color,
                "date": start_ts.strftime("%b %d, %Y"),
                "miles": round(total_dist / 1609.344, 2),
                "pace_sec": round(avg_pace),
                "prs": prs,
            })

        if not run_data:
            print("No routes with sufficient timestamp data")
            return

        m = self._base_map(self._center())
        m.get_root().html.add_child(folium.Element(self._pace_legend_html()))

        map_var = m.get_name()
        total = len(run_data)

        pr_cells = "".join(
            f"""<div>
                    <div style="font-size:10px;color:#888;letter-spacing:.5px;margin-bottom:2px;">{label}</div>
                    <div id="_pr_{key}" style="font-size:15px;font-weight:600;transition:color .4s;">—</div>
                </div>"""
            for label, key, _ in PR_DISTANCES
        )

        overlay = f"""
        <div id="_ctrl" style="position:fixed;top:20px;left:50%;transform:translateX(-50%);
            z-index:9999;background:rgba(25,25,25,0.88);color:#ddd;
            font-family:sans-serif;font-size:13px;padding:10px 18px;border-radius:8px;
            box-shadow:0 2px 10px rgba(0,0,0,0.6);display:flex;align-items:center;
            gap:14px;user-select:none;">
            <button id="_btn" onclick="_toggle()"
                style="background:#fc4c02;border:none;color:#fff;padding:5px 14px;
                       border-radius:4px;cursor:pointer;font-size:13px;min-width:60px;">Play</button>
            <span id="_date" style="min-width:110px;text-align:center;">—</span>
            <span id="_count" style="color:#aaa;font-size:11px;min-width:72px;">0 / {total}</span>
            <label style="display:flex;align-items:center;gap:6px;font-size:11px;color:#aaa;">
                Speed
                <input id="_spd" type="range" min="0" max="700" value="500" step="50"
                    style="width:70px;cursor:pointer;" oninput="_setSpeed(this.value);">
            </label>
        </div>

        <div id="_stats" style="position:fixed;bottom:30px;left:10px;z-index:9999;
            background:rgba(25,25,25,0.88);color:#ddd;font-family:sans-serif;
            padding:12px 20px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.6);
            min-width:260px;">
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;text-align:center;">
                <div>
                    <div style="font-size:10px;color:#888;letter-spacing:.6px;margin-bottom:3px;">TOTAL MILES</div>
                    <div id="_smiles" style="font-size:20px;font-weight:600;">0.0</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#888;letter-spacing:.6px;margin-bottom:3px;">RUNS</div>
                    <div id="_sruns" style="font-size:20px;font-weight:600;">0</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#888;letter-spacing:.6px;margin-bottom:3px;">AVG PACE</div>
                    <div id="_space" style="font-size:20px;font-weight:600;">—</div>
                </div>
            </div>
            <div style="border-top:1px solid #333;margin-top:10px;padding-top:8px;">
                <div style="font-size:10px;color:#888;letter-spacing:.6px;margin-bottom:6px;">SEGMENT PRs</div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px 16px;text-align:center;">
                    {pr_cells}
                </div>
            </div>
        </div>

        <script>
        const _routes = {json.dumps(run_data)};
        let _idx = 0, _playing = false, _speed = 200;
        let _group = null, _layers = [], _totalMiles = 0, _paceSum = 0;
        let _prs = {{}};

        const _OP = [0.85, 0.65, 0.45, 0.30];
        const _OP_FLOOR = 0.18;

        function _init() {{
            _group = L.layerGroup().addTo({map_var});
        }}

        function _fmt(sec) {{
            sec = Math.round(sec);
            const h = Math.floor(sec / 3600);
            const m = Math.floor((sec % 3600) / 60);
            const s = sec % 60;
            if (h > 0) return h + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
            return m + ':' + String(s).padStart(2,'0');
        }}

        function _fmtPace(sec) {{
            const m = Math.floor(sec / 60);
            const s = Math.round(sec % 60);
            return m + ':' + String(s).padStart(2,'0') + '/mi';
        }}

        function _updateOpacities() {{
            const n = _layers.length;
            const upd = Math.min(n, _OP.length + 1);
            for (let i = 0; i < upd; i++) {{
                _layers[n - 1 - i].setStyle({{opacity: i < _OP.length ? _OP[i] : _OP_FLOOR}});
            }}
        }}

        function _updateStats() {{
            document.getElementById('_smiles').textContent = _totalMiles.toFixed(1);
            document.getElementById('_sruns').textContent = _idx;
            if (_idx > 0)
                document.getElementById('_space').textContent = _fmtPace(_paceSum / _idx);
        }}

        function _updatePRs(runPRs) {{
            for (const [key, t] of Object.entries(runPRs)) {{
                if (t === null) continue;
                if (_prs[key] === undefined || t < _prs[key]) {{
                    _prs[key] = t;
                    const el = document.getElementById('_pr_' + key);
                    el.textContent = _fmt(t);
                    el.style.color = '#4ade80';
                    setTimeout(() => {{ el.style.color = '#ddd'; }}, 1200);
                }}
            }}
        }}

        function _resetPRs() {{
            _prs = {{}};
            {json.dumps([key for _, key, _ in PR_DISTANCES])}.forEach(k => {{
                const el = document.getElementById('_pr_' + k);
                el.textContent = '—';
                el.style.color = '#ddd';
            }});
        }}

        function _traceRun(r, onDone) {{
            const poly = L.polyline([], {{color: r.color, weight: 2, opacity: _OP[0]}}).addTo(_group);
            _layers.push(poly);
            _updateOpacities();
            const pts = r.coords;
            const ppf = Math.max(1, Math.ceil(pts.length / 15));
            let i = 0;
            (function frame() {{
                for (let k = 0; k < ppf && i < pts.length; k++, i++) poly.addLatLng(pts[i]);
                i < pts.length ? requestAnimationFrame(frame) : onDone();
            }})();
        }}

        function _nextRun() {{
            if (!_playing || _idx >= _routes.length) {{
                if (_idx >= _routes.length) {{
                    _playing = false;
                    document.getElementById('_btn').textContent = 'Replay';
                }}
                return;
            }}
            const r = _routes[_idx];
            _totalMiles += r.miles;
            _paceSum += r.pace_sec;
            _idx++;
            document.getElementById('_date').textContent = r.date;
            document.getElementById('_count').textContent = _idx + ' / ' + _routes.length;
            _updateStats();
            _updatePRs(r.prs);
            _traceRun(r, function() {{
                if (_playing) setTimeout(_nextRun, _speed);
            }});
        }}

        function _toggle() {{
            const btn = document.getElementById('_btn');
            if (btn.textContent === 'Replay') {{
                _idx = 0; _totalMiles = 0; _paceSum = 0; _layers = [];
                _group.clearLayers();
                _resetPRs();
                document.getElementById('_date').textContent = '—';
                document.getElementById('_count').textContent = '0 / ' + _routes.length;
                document.getElementById('_smiles').textContent = '0.0';
                document.getElementById('_sruns').textContent = '0';
                document.getElementById('_space').textContent = '—';
                btn.textContent = 'Play';
                return;
            }}
            if (_playing) {{
                _playing = false;
                btn.textContent = 'Play';
            }} else {{
                _playing = true;
                btn.textContent = 'Pause';
                _nextRun();
            }}
        }}

        function _setSpeed(v) {{ _speed = 700 - parseInt(v); }}

        window.addEventListener('load', _init);
        </script>
        """

        m.get_root().html.add_child(folium.Element(overlay))
        m.save(output)
        print(f"Animated routes saved → {output} ({total} runs)")

    def run(self) -> None:
        self.load()
        if not self._all_coords:
            print("No GPS data found. Check that GPX files exist in strava/activities/")
            return
        self.build_heatmap()
        self.build_calendar_heatmap()
        self.build_animated_routes()


if __name__ == "__main__":
    # Default: uses every activity type in DEFAULT_TYPES, no geographic filtering.
    StravaHeatmapGenerator().run()

    # To restrict to a region (e.g. the author's own Massachusetts example data),
    # pass a bounds tuple instead:
    #   StravaHeatmapGenerator(bounds=MA_BOUNDS).run()
