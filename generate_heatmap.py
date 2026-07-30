"""
Strava Heatmap Generator
Produces four interactive HTML outputs from a Strava data export:
  - strava_heatmap.html       : density heatmap of all GPS points
  - strava_pace_routes.html   : runs color-coded by pace (green=fast, red=slow)
  - strava_calendar.html      : GitHub-style calendar heatmap of daily mileage
  - strava_animated.html      : chronological replay — runs accumulate on the map over time
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
from folium.plugins import HeatMap

GPX_NS = "http://www.topografix.com/GPX/1/1"
DATA_DIR = Path("strava")
ACTIVITIES_CSV = DATA_DIR / "activities.csv"
DEFAULT_TYPES = {"run"}

# Massachusetts bounding box (min_lat, max_lat, min_lon, max_lon)
MA_BOUNDS = (41.239, 42.886, -73.508, -69.928)

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


class StravaHeatmapGenerator:
    def __init__(
        self,
        activity_types: set[str] = DEFAULT_TYPES,
        heatmap_radius: int = 8,
        heatmap_blur: int = 10,
        heatmap_min_opacity: float = 0.3,
        default_zoom: int = 12,
        pace_stride: int = 10,
        pace_buckets: int = 10,
        fast_pace_sec_per_mile: float = 360.0,   # 6:00/mi
        slow_pace_sec_per_mile: float = 540.0,   # 9:00/mi
        bounds: Optional[tuple[float, float, float, float]] = MA_BOUNDS,
    ) -> None:
        self.activity_types = {t.lower() for t in activity_types}
        self.heatmap_radius = heatmap_radius
        self.heatmap_blur = heatmap_blur
        self.heatmap_min_opacity = heatmap_min_opacity
        self.default_zoom = default_zoom
        self.pace_stride = pace_stride
        self.pace_buckets = pace_buckets
        self.fast_pace = fast_pace_sec_per_mile
        self.slow_pace = slow_pace_sec_per_mile
        self.bounds = bounds
        self._activities: list[dict] = []
        self._all_coords: list[tuple[float, float]] = []
        self._routes: list[list[Point]] = []

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

    def _bucket_color(self, bucket: int) -> str:
        t = bucket / max(self.pace_buckets - 1, 1)
        pace = self.fast_pace + t * (self.slow_pace - self.fast_pace)
        return _pace_to_hex(pace, self.fast_pace, self.slow_pace)

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

    def build_pace_routes(self, output: str = "strava_pace_routes.html") -> None:
        """Draw each run as pace-colored polyline segments (green=fast, red=slow).

        Consecutive segments that fall in the same pace bucket are merged into a
        single PolyLine to keep the map performant (~10 buckets vs thousands of
        individual 2-point lines).
        """
        if not self._routes:
            print("No routes loaded — run .load() first")
            return

        m = self._base_map(self._center())
        m.get_root().html.add_child(folium.Element(self._pace_legend_html()))

        routes_drawn = 0
        for route in self._routes:
            timed = [p for p in route if p[2] is not None]
            if len(timed) < 2:
                folium.PolyLine(
                    [(p[0], p[1]) for p in route],
                    color="#888888", weight=1.5, opacity=0.35,
                ).add_to(m)
                continue

            strided = timed[:: self.pace_stride] or timed

            # Calculate pace bucket per segment, then merge consecutive same-bucket segments
            run_segments: list[tuple[int, list[tuple[float, float]]]] = []
            for i in range(len(strided) - 1):
                a, b = strided[i], strided[i + 1]
                dist = _haversine(a[0], a[1], b[0], b[1])
                dt = (b[2] - a[2]).total_seconds()  # type: ignore[operator]
                if dist < 1 or dt <= 0:
                    continue
                pace = (dt / dist) * 1609.344
                bucket = min(
                    self.pace_buckets - 1,
                    int(max(0.0, min(1.0, (pace - self.fast_pace) / (self.slow_pace - self.fast_pace))) * self.pace_buckets),
                )
                coord = (a[0], a[1])
                next_coord = (b[0], b[1])
                if run_segments and run_segments[-1][0] == bucket:
                    run_segments[-1][1].append(next_coord)
                else:
                    run_segments.append((bucket, [coord, next_coord]))

            for bucket, coords in run_segments:
                folium.PolyLine(
                    coords,
                    color=self._bucket_color(bucket),
                    weight=2,
                    opacity=0.75,
                ).add_to(m)

            routes_drawn += 1

        m.save(output)
        print(f"Pace-colored routes saved → {output} ({routes_drawn} routes with timestamps)")

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
        """GitHub-style calendar heatmap of daily mileage using Plotly."""
        df = pd.read_csv(ACTIVITIES_CSV)
        df = df[df["Activity Type"].str.lower().isin(self.activity_types)].copy()
        df["date"] = pd.to_datetime(df["Activity Date"], format="mixed")
        df["miles"] = pd.to_numeric(df["Distance"], errors="coerce") * 0.621371
        df["date_only"] = df["date"].dt.normalize()

        daily = df.groupby("date_only")["miles"].sum().reset_index()
        daily.columns = ["date", "miles"]
        daily["date"] = pd.to_datetime(daily["date"])

        full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = (
            daily.set_index("date")
            .reindex(full_range, fill_value=0.0)
            .reset_index()
            .rename(columns={"index": "date"})
        )

        daily["week_idx"] = (daily["date"] - daily["date"].min()).dt.days // 7
        daily["dow"] = daily["date"].dt.dayofweek  # 0=Mon, 6=Sun
        daily["label"] = daily["date"].dt.strftime("%b %d, %Y")

        pivot_miles = daily.pivot(index="dow", columns="week_idx", values="miles").fillna(0.0)
        pivot_labels = daily.pivot(index="dow", columns="week_idx", values="label").fillna("")

        # Stack label + miles into customdata so hover doesn't rely on %{z}
        customdata = np.stack([
            pivot_labels.values,
            pivot_miles.round(1).astype(str).values,
        ], axis=-1)

        # X-axis: show month name at first week of each month
        week_starts = daily.groupby("week_idx")["date"].min()
        x_ticks = [d.strftime("%b '%y") if d.day <= 7 else "" for d in week_starts]

        fig = go.Figure(go.Heatmap(
            z=pivot_miles.values,
            x=list(range(len(x_ticks))),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale=[
                [0.0,  "#161b22"],
                [0.01, "#3d1f00"],
                [0.2,  "#8a3a00"],
                [0.5,  "#fc4c02"],
                [1.0,  "#ff9a5c"],
            ],
            zmin=0,
            showscale=True,
            colorbar=dict(title="Miles", thickness=12, len=0.6, tickfont=dict(color="#ccc")),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} miles<extra></extra>",
            customdata=customdata,
        ))

        fig.update_layout(
            title=dict(text="Running Calendar", font=dict(size=22, color="#eee"), x=0.5),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#ccc", family="sans-serif"),
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(x_ticks))),
                ticktext=x_ticks,
                tickangle=-45,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=10),
            ),
            yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"),
            height=260,
            margin=dict(l=50, r=80, t=60, b=80),
        )

        fig.write_html(output, include_plotlyjs="cdn")
        print(f"Calendar heatmap saved → {output}")

    def build_animated_routes(
        self, output: str = "strava_animated.html", coord_stride: int = 3
    ) -> None:
        """Chronological replay with three enhancements over a simple pop-in:

        - Route trace: each run draws itself via requestAnimationFrame (~15 frames)
        - Recency fading: newest run is brightest; older runs dim toward a floor opacity
        - Stats ticker: cumulative miles, run count, and rolling avg pace update live
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
            total_dist = sum(
                _haversine(timed[i][0], timed[i][1], timed[i + 1][0], timed[i + 1][1])
                for i in range(len(timed) - 1)
            )
            total_time = (timed[-1][2] - timed[0][2]).total_seconds()
            if total_dist < 10 or total_time <= 0:
                continue
            avg_pace = (total_time / total_dist) * 1609.344
            color = _pace_to_hex(avg_pace, self.fast_pace, self.slow_pace)
            sampled = route[::coord_stride] or route
            run_data.append({
                "coords": [[p[0], p[1]] for p in sampled],
                "color": color,
                "date": start_ts.strftime("%b %d, %Y"),
                "miles": round(total_dist / 1609.344, 2),
                "pace_sec": round(avg_pace),
            })

        if not run_data:
            print("No routes with sufficient timestamp data")
            return

        m = self._base_map(self._center())
        m.get_root().html.add_child(folium.Element(self._pace_legend_html()))

        map_var = m.get_name()
        total = len(run_data)

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
            padding:12px 20px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.6);">
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
        </div>

        <script>
        const _routes = {json.dumps(run_data)};
        let _idx = 0, _playing = false, _speed = 200;
        let _group = null, _layers = [], _totalMiles = 0, _paceSum = 0;

        // Opacity schedule: index 0 = newest run, floor applied to all older ones.
        // Only the last OP.length+1 polylines are ever updated per run — O(1).
        const _OP = [0.85, 0.65, 0.45, 0.30];
        const _OP_FLOOR = 0.18;

        function _init() {{
            _group = L.layerGroup().addTo({map_var});
        }}

        function _fmtPace(sec) {{
            const m = Math.floor(sec / 60);
            const s = Math.round(sec % 60);
            return m + ':' + String(s).padStart(2, '0') + '/mi';
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
            _traceRun(r, function() {{
                if (_playing) setTimeout(_nextRun, _speed);
            }});
        }}

        function _toggle() {{
            const btn = document.getElementById('_btn');
            if (btn.textContent === 'Replay') {{
                _idx = 0; _totalMiles = 0; _paceSum = 0; _layers = [];
                _group.clearLayers();
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

        function _setSpeed(v) {{
            _speed = 700 - parseInt(v);
        }}

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
        self.build_pace_routes()
        self.build_calendar_heatmap()
        self.build_animated_routes()


if __name__ == "__main__":
    StravaHeatmapGenerator().run()
