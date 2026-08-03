"""
Strava EDA Generator
Produces four interactive HTML outputs from any Strava bulk data export:
  - strava_heatmap.html   : density heatmap of all GPS points, plus a
                            most-repeated-routes leaderboard
  - strava_calendar.html  : GitHub-style calendar heatmap of daily mileage
  - strava_animated.html  : chronological replay — runs accumulate on the map over time
  - strava_wrapped.html   : shareable year-in-review "Wrapped" summary

Works out of the box with any user's exported strava/ folder — no code changes
required. All tuning knobs (activity type, region filtering, pace thresholds,
heatmap styling) are optional constructor arguments on StravaHeatmapGenerator;
see README.md for details. See run() at the bottom for the default entry point.
"""

import csv
import json
import math
import statistics
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
PR_LABELS = {key: label for label, key, _ in PR_DISTANCES}

# Categorical colors for highlighting individual clustered routes on the density
# heatmap. Deliberately skips warm hues (orange/red) since those are already the
# "hot" end of the heatmap's own colorscale — a warm route line would disappear
# into a dense blob instead of standing out. Validated colorblind-safe adjacent
# ordering against the map's dark background (see dataviz skill's palette
# validator); cycles past 6 simultaneously-toggled routes, which is rare enough
# in practice not to warrant a larger validated set.
ROUTE_PALETTE = ["#3987e5", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"]

EVEREST_M = 8849.0  # height of Mt. Everest, for "climbed X Everests" framing

# (lat, lon, timestamp, heart_rate_bpm)
Point = tuple[float, float, Optional[datetime], Optional[float]]

# Localname (namespace-agnostic) of the GPX extension tag various devices/apps
# use for per-point heart rate: Garmin TrackPointExtension (<gpxtpx:hr>), some
# older exports (<gpxdata:hr>), etc. We match on tag name alone since the
# namespace prefix/URI varies by exporter.
_HR_TAG_NAMES = {"hr", "heartrate"}


def _parse_moving_time(val: object) -> float:
    """Strava's "Moving Time" column as seconds — accepts H:MM:SS, M:SS, or a
    bare seconds value, since exports have used all three depending on the
    row's activity type / duration."""
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


def _fmt_pace(sec_per_mile: Optional[float]) -> Optional[str]:
    """"M:SS/mi" from a seconds-per-mile pace, or None if there's nothing
    valid to show (no pace data, or a non-positive value) — every call site
    that has its own no-data placeholder (a dash, `None`) applies it on top."""
    if sec_per_mile is None or pd.isna(sec_per_mile) or sec_per_mile <= 0:
        return None
    return f"{int(sec_per_mile) // 60}:{int(round(sec_per_mile % 60)):02d}/mi"


def _streak_stats(active: np.ndarray) -> tuple[int, int]:
    """(longest, current) consecutive-active-day streaks from a boolean array
    ordered oldest→newest. "Current" is the run of active days ending on the
    array's last entry (0 if the most recent day was inactive)."""
    longest = cur = 0
    for a in active:
        cur = cur + 1 if a else 0
        longest = max(longest, cur)
    current = 0
    for a in active[::-1]:
        if not a:
            break
        current += 1
    return longest, current


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS coordinates."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _value_to_hex(value: float, good: float, bad: float) -> str:
    """Map a value between `good` and `bad` to a hex color: green → yellow → red.

    Used for both pace (low seconds/mile = good) and heart rate (low bpm = good),
    since in both cases a smaller value represents lower effort.
    """
    t = max(0.0, min(1.0, (value - good) / (bad - good)))
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
        fast_pace_sec_per_mile: Optional[float] = None,  # None = auto (5th percentile of your paces)
        slow_pace_sec_per_mile: Optional[float] = None,  # None = auto (95th percentile of your paces)
        hr_low_bpm: Optional[float] = None,   # None = auto (5th percentile of your heart rate)
        hr_high_bpm: Optional[float] = None,  # None = auto (95th percentile of your heart rate)
        bounds: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        self.activity_types = {t.lower() for t in activity_types}
        self.heatmap_radius = heatmap_radius
        self.heatmap_blur = heatmap_blur
        self.heatmap_min_opacity = heatmap_min_opacity
        self.default_zoom = default_zoom
        self.fast_pace = fast_pace_sec_per_mile
        self.slow_pace = slow_pace_sec_per_mile
        self.hr_low = hr_low_bpm
        self.hr_high = hr_high_bpm
        self.bounds = bounds

        self._activities: list[dict] = []
        self._all_coords: list[tuple[float, float]] = []
        self._routes: list[list[Point]] = []
        self._route_csv_hr: list[Optional[float]] = []
        self._included_filenames: set[str] = set()
        self._metrics_cache: Optional[list[dict]] = None

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
        self._route_csv_hr = []
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
            self._route_csv_hr.append(self._csv_avg_hr(activity))
        if skipped:
            print(f"  Excluded {skipped} routes outside bounds")
        return routes, all_coords

    @staticmethod
    def _csv_avg_hr(activity: dict) -> Optional[float]:
        """Per-activity average heart rate from the CSV, used as a fallback when
        the GPX file itself has no per-point heart rate extension."""
        raw = activity.get("Average Heart Rate", "")
        try:
            return float(raw) if raw not in (None, "") else None
        except ValueError:
            return None

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
                hr = self._parse_hr_extension(trkpt)
                points.append((float(lat), float(lon), ts, hr))
            return points
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
            return []

    @staticmethod
    def _parse_hr_extension(trkpt: ET.Element) -> Optional[float]:
        """Namespace-agnostic search for a heart-rate value under <extensions>.

        Different devices/exporters use different extension namespaces
        (e.g. Garmin TrackPointExtension), so we match on the tag's local
        name rather than requiring a specific schema.
        """
        ext = trkpt.find(f"{{{GPX_NS}}}extensions")
        if ext is None:
            return None
        for el in ext.iter():
            local = el.tag.rsplit("}", 1)[-1].lower()
            if local in _HR_TAG_NAMES and el.text:
                try:
                    return float(el.text)
                except ValueError:
                    return None
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _center(self) -> tuple[float, float]:
        lats = [c[0] for c in self._all_coords]
        lons = [c[1] for c in self._all_coords]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    def _base_map(self, center: tuple[float, float]) -> folium.Map:
        return folium.Map(location=center, zoom_start=self.default_zoom, tiles="CartoDB dark_matter")

    def _load_activity_df(self) -> pd.DataFrame:
        """Activities from the CSV, filtered to the configured activity types and
        to the routes that actually parsed successfully, with `date`/`miles` columns
        added. Shared by every output that needs per-activity stats."""
        df = pd.read_csv(ACTIVITIES_CSV)
        df = df[df["Activity Type"].str.lower().isin(self.activity_types)].copy()
        if self._included_filenames:
            df = df[df["Filename"].isin(self._included_filenames)]
        df["date"] = pd.to_datetime(df["Activity Date"], format="mixed")
        df["miles"] = pd.to_numeric(df["Distance"], errors="coerce") * 0.621371
        return df

    def _route_metrics(self) -> list[dict]:
        """Chronologically sorted per-route stats: distance, avg pace, avg heart
        rate, segment PR splits, and which PRs were newly set (all-time-best as
        of that route). Heart rate prefers the GPX per-point extension (averaged
        over the route) and falls back to the CSV's per-activity average.

        Cached after the first call — shared by build_animated_routes and the
        calendar's PR-day overlay so the PR computation only runs once.
        """
        if self._metrics_cache is not None:
            return self._metrics_cache

        paired = zip(self._routes, self._route_csv_hr)
        timed_routes = [(route, csv_hr, route[0][2]) for route, csv_hr in paired if route[0][2] is not None]
        timed_routes.sort(key=lambda x: x[2])

        bests: dict[str, float] = {}
        metrics: list[dict] = []
        for route, csv_hr, start_ts in timed_routes:
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

            gpx_hrs = [p[3] for p in timed if p[3] is not None]
            avg_hr = (sum(gpx_hrs) / len(gpx_hrs)) if gpx_hrs else csv_hr

            prs = {
                key: (round(t, 1) if (t := _best_segment_time(cum, timed, dist_m)) else None)
                for _, key, dist_m in PR_DISTANCES
            }
            new_prs = []
            for key, t in prs.items():
                if t is not None and (key not in bests or t < bests[key]):
                    bests[key] = t
                    new_prs.append(key)

            metrics.append({
                "route": route,
                "start_ts": start_ts,
                "total_dist": total_dist,
                "avg_pace": avg_pace,
                "avg_hr": avg_hr,
                "prs": prs,
                "new_prs": new_prs,
            })

        self._metrics_cache = metrics
        return metrics

    def _auto_pace_range(self) -> tuple[float, float]:
        """Fast/slow pace thresholds (sec/mile) for color scaling. Uses
        `fast_pace`/`slow_pace` if the constructor was given explicit values;
        otherwise auto-computed from the 5th/95th percentile of this runner's
        own per-route average pace, so the color scale always spans whoever's
        data is loaded rather than assuming a specific pace band. Percentiles
        (not raw min/max) keep one GPS glitch or an easy walk break from
        blowing out the whole scale."""
        if self.fast_pace is not None and self.slow_pace is not None:
            return self.fast_pace, self.slow_pace
        paces = [m["avg_pace"] for m in self._route_metrics()]
        if not paces:
            return self.fast_pace or 360.0, self.slow_pace or 540.0
        fast = self.fast_pace if self.fast_pace is not None else float(np.percentile(paces, 5))
        slow = self.slow_pace if self.slow_pace is not None else float(np.percentile(paces, 95))
        if slow <= fast:
            slow = fast + 30.0  # keep a visible band even if every run is nearly the same pace
        return fast, slow

    def _auto_hr_range(self) -> tuple[float, float]:
        """Same auto-scaling as `_auto_pace_range`, for heart rate (bpm)."""
        if self.hr_low is not None and self.hr_high is not None:
            return self.hr_low, self.hr_high
        hrs = [m["avg_hr"] for m in self._route_metrics() if m["avg_hr"] is not None]
        if not hrs:
            return self.hr_low or 130.0, self.hr_high or 175.0
        low = self.hr_low if self.hr_low is not None else float(np.percentile(hrs, 5))
        high = self.hr_high if self.hr_high is not None else float(np.percentile(hrs, 95))
        if high <= low:
            high = low + 10.0
        return low, high

    def _cluster_routes(
        self, grid_deg: float = 0.0009, jaccard_threshold: float = 0.55, dist_tolerance: float = 0.25,
    ) -> list[dict]:
        """Group routes that retrace (roughly) the same path, so a loop you run
        every week surfaces as one entry with a run count instead of N unrelated
        traces on the map.

        Each route is fingerprinted as the set of ~100m grid cells its GPS
        points fall into; a route joins the first existing cluster whose
        fingerprint has high overlap (Jaccard similarity) AND a similar total
        distance (guards against, e.g., a 2-mile out-and-back nested inside a
        10-mile route that happens to share the same starting blocks).

        Clusters compare against their first member (not a running-average
        fingerprint), so a cluster's identity stays anchored to one concrete
        route rather than drifting as slightly different variants get added.
        `grid_deg=0.0009` is roughly 100m at mid-latitudes — coarse enough to
        absorb GPS jitter and lane-to-lane noise, fine enough not to conflate
        genuinely different streets.
        """
        metrics = self._route_metrics()

        def _cells(route: list[Point]) -> set[tuple[int, int]]:
            return {(round(lat / grid_deg), round(lon / grid_deg)) for lat, lon, _, _ in route}

        clusters: list[dict] = []
        for m in metrics:
            cell_set = _cells(m["route"])
            dist_mi = m["total_dist"] / 1609.344
            best_cluster, best_sim = None, 0.0
            for cl in clusters:
                union = len(cell_set | cl["cells"])
                if not union:
                    continue
                sim = len(cell_set & cl["cells"]) / union
                dist_ratio = abs(dist_mi - cl["dist_mi"]) / max(dist_mi, cl["dist_mi"])
                if sim >= jaccard_threshold and dist_ratio <= dist_tolerance and sim > best_sim:
                    best_cluster, best_sim = cl, sim
            if best_cluster is not None:
                best_cluster["members"].append(m)
            else:
                clusters.append({"cells": cell_set, "dist_mi": dist_mi, "route": m["route"], "members": [m]})

        return clusters

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def _top_route_clusters(self, limit: int = 10, min_repeats: int = 2) -> list[dict]:
        """Repeat routes (run `min_repeats`+ times), ranked by run count then
        by distance — the candidates for the heatmap's route leaderboard.
        Onetime routes are excluded; a "leaderboard" of routes you've only run
        once isn't a meaningful ranking."""
        clusters = self._cluster_routes()
        repeats = [cl for cl in clusters if len(cl["members"]) >= min_repeats]
        repeats.sort(
            key=lambda cl: (-len(cl["members"]), -statistics.median(m["total_dist"] for m in cl["members"]))
        )
        return repeats[:limit]

    def build_heatmap(self, output: str = "strava_heatmap.html") -> None:
        """Density heatmap with a per-year layer toggle, a stats overlay, and a
        most-repeated-routes leaderboard.

        Each year gets its own togglable heat layer (via folium's LayerControl)
        so you can isolate a single year or compare several, in addition to the
        default "All Years" combined view.

        Routes you've run more than once are clustered together (see
        `_cluster_routes`) and ranked by frequency into a leaderboard panel;
        each entry can toggle a highlighted trace of that specific route on
        the map and zoom to it, turning "where do I run" into "which loop do I
        actually run, and how fast."
        """
        if not self._all_coords:
            print("No coordinates loaded — run .load() first")
            return
        m = self._base_map(self._center())

        year_coords: dict[str, list[tuple[float, float]]] = {}
        for route in self._routes:
            ts = next((p[2] for p in route if p[2] is not None), None)
            year = str(ts.year) if ts else "Unknown date"
            year_coords.setdefault(year, []).extend((p[0], p[1]) for p in route)

        def _add_heat_layer(name: str, coords: list[tuple[float, float]], show: bool) -> None:
            fg = folium.FeatureGroup(name=name, show=show)
            HeatMap(
                coords,
                radius=self.heatmap_radius,
                blur=self.heatmap_blur,
                min_opacity=self.heatmap_min_opacity,
            ).add_to(fg)
            fg.add_to(m)

        _add_heat_layer("All Years", self._all_coords, show=True)
        for year in sorted(year_coords, reverse=True):
            _add_heat_layer(year, year_coords[year], show=False)

        top_routes = self._top_route_clusters()
        map_var = m.get_name()
        layer_vars = []
        for rank, cl in enumerate(top_routes, start=1):
            color = ROUTE_PALETTE[(rank - 1) % len(ROUTE_PALETTE)]
            sampled = cl["route"][::3] or cl["route"]
            fg = folium.FeatureGroup(name=f"Route #{rank}", show=False)
            folium.PolyLine(
                [(p[0], p[1]) for p in sampled], color=color, weight=3, opacity=0.9,
            ).add_to(fg)
            fg.add_to(m)
            layer_vars.append(fg.get_name())

        folium.LayerControl(collapsed=False).add_to(m)
        m.get_root().html.add_child(folium.Element(self._heatmap_stats_html()))
        if top_routes:
            m.get_root().html.add_child(
                folium.Element(self._route_leaderboard_html(top_routes, layer_vars, map_var))
            )

        m.save(output)
        print(f"Density heatmap saved → {output}")

    def _route_leaderboard_html(self, clusters: list[dict], layer_vars: list[str], map_var: str) -> str:
        """Ranked panel of repeat routes, each row toggling a highlighted trace
        of that route (via `layer_vars`, the JS variable names of the
        FeatureGroups already added to the map in `build_heatmap`) and
        zooming to its bounds."""

        rows, bounds_list = [], []
        for rank, cl in enumerate(clusters, start=1):
            members = cl["members"]
            distances = [mm["total_dist"] / 1609.344 for mm in members]
            med_dist = statistics.median(distances)
            best_pace = min(mm["avg_pace"] for mm in members)
            color = ROUTE_PALETTE[(rank - 1) % len(ROUTE_PALETTE)]

            lats = [p[0] for mm in members for p in mm["route"]]
            lons = [p[1] for mm in members for p in mm["route"]]
            bounds_list.append([[min(lats), min(lons)], [max(lats), max(lons)]])

            btn_id = f"_rbtn{rank}"
            rows.append(f"""
            <div style="display:flex;align-items:stretch;gap:10px;padding:7px 0;
                border-bottom:1px solid #2a2a2a;">
                <div style="width:4px;border-radius:2px;background:{color};"></div>
                <div style="flex:1;min-width:0;">
                    <div style="font-size:12.5px;color:#ddd;">Route #{rank} · {med_dist:.1f} mi</div>
                    <div style="font-size:10.5px;color:#888;">×{len(members)} · best {_fmt_pace(best_pace)}</div>
                </div>
                <button id="{btn_id}" onclick="_toggleRoute({rank - 1},'{btn_id}')"
                    style="background:#333;border:none;color:#ddd;padding:4px 10px;
                           border-radius:4px;cursor:pointer;font-size:11px;
                           white-space:nowrap;align-self:center;">Show</button>
            </div>""")

        return f"""
        <div style="position:fixed;top:12px;left:10px;z-index:9999;
            background:rgba(25,25,25,0.9);color:#ddd;font-family:sans-serif;
            padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.6);
            width:230px;max-height:70vh;overflow-y:auto;">
            <div style="font-size:11px;color:#888;letter-spacing:.6px;margin-bottom:2px;">
                MOST-RUN ROUTES
            </div>
            {"".join(rows)}
        </div>
        <script>
        // This overlay is injected as HTML content, which folium/branca renders
        // *before* the big inline <script> further down the page that actually
        // declares the feature_group_* variables for each route layer. Building
        // the _routeLayers array has to wait for the 'load' event so those
        // variables exist by the time we reference them (same pattern the
        // animated-replay page uses for its own post-load init).
        let _routeLayers = [];
        const _routeBounds = {json.dumps(bounds_list)};
        let _routeShown = [];
        function _initRoutes() {{
            _routeLayers = [{", ".join(layer_vars)}];
            _routeShown = new Array(_routeLayers.length).fill(false);
        }}
        function _toggleRoute(i, btnId) {{
            const btn = document.getElementById(btnId);
            if (_routeShown[i]) {{
                {map_var}.removeLayer(_routeLayers[i]);
                btn.textContent = 'Show';
            }} else {{
                _routeLayers[i].addTo({map_var});
                {map_var}.fitBounds(_routeBounds[i]);
                btn.textContent = 'Hide';
            }}
            _routeShown[i] = !_routeShown[i];
        }}
        window.addEventListener('load', _initRoutes);
        </script>"""

    def _heatmap_stats_html(self) -> str:
        df = self._load_activity_df()
        total_miles = df["miles"].sum()
        total_activities = len(df)
        date_min, date_max = df["date"].min(), df["date"].max()
        date_range = f"{date_min.strftime('%b %Y')} – {date_max.strftime('%b %Y')}"
        return f"""
        <div style="position:fixed;bottom:30px;left:10px;z-index:9999;
            background:rgba(25,25,25,0.88);color:#ddd;font-family:sans-serif;
            padding:12px 20px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.6);
            min-width:220px;">
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;text-align:center;">
                <div>
                    <div style="font-size:10px;color:#888;letter-spacing:.6px;margin-bottom:3px;">TOTAL MILES</div>
                    <div style="font-size:20px;font-weight:600;">{total_miles:.1f}</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#888;letter-spacing:.6px;margin-bottom:3px;">ACTIVITIES</div>
                    <div style="font-size:20px;font-weight:600;">{total_activities}</div>
                </div>
            </div>
            <div style="text-align:center;margin-top:8px;padding-top:8px;border-top:1px solid #333;
                font-size:11px;color:#aaa;">{date_range}</div>
        </div>"""

    def _gradient_legend_html(
        self, dom_id: str, label: str, good: float, bad: float, fmt, visible: bool
    ) -> str:
        """Build a fixed-position gradient legend (green→yellow→red) for a
        value range. Shared by the pace legend and the optional HR legend so
        both look and behave identically, just toggled by `visible`."""
        mid = (good + bad) / 2
        g = _value_to_hex(good, good, bad)
        y = _value_to_hex(mid, good, bad)
        r = _value_to_hex(bad, good, bad)
        display = "block" if visible else "none"
        return f"""
        <div id="{dom_id}" style="display:{display};position:fixed;bottom:30px;right:10px;z-index:9999;
                    background:rgba(25,25,25,0.88);color:#ddd;
                    font-family:sans-serif;font-size:12px;
                    padding:10px 14px;border-radius:6px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.6)">
            <b style="display:block;margin-bottom:8px;letter-spacing:.5px">{label}</b>
            <div style="display:flex;align-items:center;gap:8px">
                <span>{fmt(good)}</span>
                <div style="width:90px;height:6px;border-radius:3px;
                    background:linear-gradient(to right,{g},{y},{r})"></div>
                <span>{fmt(bad)}</span>
            </div>
        </div>"""

    def _pace_legend_html(self, visible: bool = True) -> str:
        def fmt(sec: float) -> str:
            return f"{int(sec) // 60}:{int(sec) % 60:02d}/mi"

        fast, slow = self._auto_pace_range()
        return self._gradient_legend_html("_legendPace", "Pace", fast, slow, fmt, visible)

    def _hr_legend_html(self, visible: bool = False) -> str:
        def fmt(bpm: float) -> str:
            return f"{int(round(bpm))} bpm"

        low, high = self._auto_hr_range()
        return self._gradient_legend_html("_legendHR", "Heart Rate", low, high, fmt, visible)

    def _daily_mileage_df(self) -> tuple[pd.DataFrame, list[int]]:
        """Per-day mileage/time/pace, reindexed to a continuous daily range so
        gaps become explicit rest days (needed for streak math and the weekly
        rolling averages). Shared by the calendar heatmap and the wrapped
        summary — both need day-level stats, not just per-activity rows.

        Returns `(daily, active_years)`. `active_years` is captured *before*
        the reindex and only lists years with logged activity — the reindex
        fills every day between the first and last activity, so sourcing the
        year list from it would pull in fully blank years for any hiatus (or
        one old stray activity years before a training block).
        """
        df = self._load_activity_df()
        df["date_only"] = df["date"].dt.normalize()
        df["moving_secs"] = df["Moving Time"].apply(_parse_moving_time)
        if "Activity Name" in df.columns:
            df["run_name"] = df["Activity Name"].fillna("")
        else:
            df["run_name"] = ""

        records = []
        for dt, grp in df.groupby("date_only"):
            mi = grp["miles"].sum()
            sc = grp["moving_secs"].sum()
            cnt = len(grp)
            name = grp["run_name"].iloc[0] if cnt == 1 else f"{cnt} runs"
            records.append({
                "date": dt, "miles": mi, "secs": sc,
                "pace": _fmt_pace(sc / mi if mi else None) or "—", "name": str(name),
            })

        daily = pd.DataFrame(records)
        daily["date"] = pd.to_datetime(daily["date"])
        active_years = sorted(daily["date"].dt.year.unique())

        full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = (
            daily.set_index("date")
            .reindex(full_range)
            .reset_index()
            .rename(columns={"index": "date"})
        )
        daily["miles"] = daily["miles"].fillna(0.0)
        daily["secs"] = daily["secs"].fillna(0.0)
        daily["pace"] = daily["pace"].fillna("—")
        daily["name"] = daily["name"].fillna("")

        return daily, active_years

    def build_calendar_heatmap(self, output: str = "strava_calendar.html") -> None:
        """Year-per-row calendar heatmap + monthly bar chart + weekly mileage and
        pace trends, with streak callouts and PR-day markers."""
        daily, years = self._daily_mileage_df()
        n_years = len(years)
        global_max = max(daily["miles"].max(), 0.1)

        # PR-day lookup: date -> labels of any all-time-best segment set that day,
        # reusing the same PR computation build_animated_routes relies on.
        pr_days: dict[pd.Timestamp, list[str]] = {}
        for rm in self._route_metrics():
            if rm["new_prs"]:
                # GPX timestamps are tz-aware (UTC); the calendar's dates are tz-naive
                # (parsed from the CSV's local-time "Activity Date"), so drop tzinfo here
                # rather than converting — matches the date build_animated_routes shows.
                d = pd.Timestamp(rm["start_ts"].replace(tzinfo=None)).normalize()
                pr_days.setdefault(d, []).extend(PR_LABELS[k] for k in rm["new_prs"])

        # Streak stats: longest run of active days, and the streak ending on the
        # most recent day present in the export.
        longest_streak, current_streak = _streak_stats((daily["miles"] > 0).to_numpy())

        COLORSCALE = [
            [0.0,  "#161b22"],
            [0.01, "#3d1f00"],
            [0.2,  "#8a3a00"],
            [0.5,  "#fc4c02"],
            [1.0,  "#ff9a5c"],
        ]
        DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        row_heights = [1.0] * n_years + [2.5, 2.5, 2.5]
        subtitles = [str(y) for y in years] + ["Monthly Mileage", "Weekly Mileage", "Weekly Pace"]

        fig = make_subplots(
            rows=n_years + 3,
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

            # PR-day markers: star at any cell where an all-time-best was set
            pr_x, pr_y, pr_text = [], [], []
            for pr_date, labels in pr_days.items():
                if pr_date.year != year:
                    continue
                offset = (pr_date - grid_start).days
                w_i, d_i = offset // 7, offset % 7
                if 0 <= w_i < n_weeks:
                    pr_x.append(w_i)
                    pr_y.append(DAY_LABELS[d_i])
                    pr_text.append(", ".join(labels) + " PR")

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
            if pr_x:
                fig.add_trace(
                    go.Scatter(
                        x=pr_x, y=pr_y,
                        mode="markers",
                        marker=dict(symbol="star", size=9, color="#ffd700",
                                    line=dict(color="#0d1117", width=1)),
                        text=pr_text,
                        hovertemplate="<b>%{text}</b><extra></extra>",
                        showlegend=False,
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
        weekly = daily.groupby("week_start")[["miles", "secs"]].sum().reset_index().sort_values("week_start")
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

        # Weekly pace trend with 4-week rolling average (min/mi, lower = faster)
        weekly["pace"] = np.where(weekly["miles"] > 0, weekly["secs"] / weekly["miles"], np.nan)
        weekly["pace_rolling4"] = weekly["pace"].rolling(4, min_periods=1).mean()
        weekly["pace_fmt"] = weekly["pace"].apply(lambda s: _fmt_pace(s) or "—")
        weekly["pace_rolling4_fmt"] = weekly["pace_rolling4"].apply(lambda s: _fmt_pace(s) or "—")

        pace_row = n_years + 3
        fig.add_trace(
            go.Scatter(
                x=weekly["week_start"], y=(weekly["pace"] / 60).round(2),
                mode="lines",
                line=dict(color="#4ade80", width=1),
                opacity=0.35,
                customdata=weekly["pace_fmt"],
                hovertemplate="Week of %{x|%b %d, %Y}<br>%{customdata}<extra></extra>",
                showlegend=False, name="Weekly Pace",
            ),
            row=pace_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=weekly["week_start"], y=(weekly["pace_rolling4"] / 60).round(2),
                mode="lines",
                line=dict(color="#22c55e", width=2.5),
                customdata=weekly["pace_rolling4_fmt"],
                hovertemplate="4-wk avg (week of %{x|%b %d, %Y})<br>%{customdata}<extra></extra>",
                showlegend=False, name="4-wk avg pace",
            ),
            row=pace_row, col=1,
        )
        fig.update_xaxes(
            showgrid=False, zeroline=False,
            tickfont=dict(color="#aaa", size=10),
            row=pace_row, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="#1c2330", zeroline=False,
            tickfont=dict(color="#aaa"),
            title_text="min/mi", title_font=dict(color="#888", size=11),
            row=pace_row, col=1,
        )

        height = n_years * 130 + 560 + 100 + 240

        streak_note = f"Current streak: {current_streak}d · Longest streak: {longest_streak}d"
        if pr_days:
            streak_note += " · ★ = new all-time PR"
        title_text = (
            f"Running Dashboard<br>"
            f"<span style='font-size:13px;color:#888'>{streak_note}</span>"
        )

        fig.update_layout(
            title=dict(text=title_text, font=dict(size=22, color="#eee"), x=0.5),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#ccc", family="sans-serif"),
            height=height,
            margin=dict(l=60, r=80, t=100, b=60),
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

        If heart rate data is available (per-point GPX extension, or a per-activity
        average from the CSV), a Pace/HR color toggle is included; otherwise it's
        omitted entirely rather than shown non-functional. A "Hide Map" toggle is
        always included so the replay can be screenshotted/recorded for sharing
        without exposing the underlying map — note this only hides the on-screen
        basemap, the raw route coordinates still live in the page's source.
        """
        if not self._routes:
            print("No routes loaded — run .load() first")
            return

        fast_pace, slow_pace = self._auto_pace_range()
        hr_low, hr_high = self._auto_hr_range()

        run_data = []
        for rm in self._route_metrics():
            sampled = rm["route"][::coord_stride] or rm["route"]
            pace_color = _value_to_hex(rm["avg_pace"], fast_pace, slow_pace)
            hr_color = (
                _value_to_hex(rm["avg_hr"], hr_low, hr_high)
                if rm["avg_hr"] is not None else None
            )
            run_data.append({
                "coords": [[p[0], p[1]] for p in sampled],
                "color": pace_color,
                "hr_color": hr_color,
                "hr": round(rm["avg_hr"]) if rm["avg_hr"] is not None else None,
                "date": rm["start_ts"].strftime("%b %d, %Y"),
                "miles": round(rm["total_dist"] / 1609.344, 2),
                "pace_sec": round(rm["avg_pace"]),
                "prs": rm["prs"],
            })

        if not run_data:
            print("No routes with sufficient timestamp data")
            return

        has_hr = any(r["hr"] is not None for r in run_data)

        m = self._base_map(self._center())
        m.get_root().html.add_child(folium.Element(self._pace_legend_html(visible=True)))
        if has_hr:
            m.get_root().html.add_child(folium.Element(self._hr_legend_html(visible=False)))

        map_var = m.get_name()
        total = len(run_data)

        pr_cells = "".join(
            f"""<div>
                    <div style="font-size:10px;color:#888;letter-spacing:.5px;margin-bottom:2px;">{label}</div>
                    <div id="_pr_{key}" style="font-size:15px;font-weight:600;transition:color .4s;">—</div>
                </div>"""
            for label, key, _ in PR_DISTANCES
        )

        hr_btn = """<button id="_hrbtn" onclick="_toggleColorMode()"
                style="background:#333;border:none;color:#ddd;padding:5px 12px;
                       border-radius:4px;cursor:pointer;font-size:12px;">Color: Pace</button>""" if has_hr else ""

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
            {hr_btn}
            <button id="_mapbtn" onclick="_toggleMap()"
                style="background:#333;border:none;color:#ddd;padding:5px 12px;
                       border-radius:4px;cursor:pointer;font-size:12px;">Hide Map</button>
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
                    <div style="font-size:10px;color:#888;letter-spacing:.6px;margin-bottom:3px;" id="_spaceLabel">AVG PACE</div>
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
        let _group = null, _layers = [], _totalMiles = 0, _paceSum = 0, _hrSum = 0, _hrCount = 0;
        let _prs = {{}};
        let _colorMode = 'pace';
        let _mapHidden = false;

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

        function _colorFor(r) {{
            if (_colorMode === 'hr') return r.hr_color || '#555';
            return r.color;
        }}

        function _toggleColorMode() {{
            _colorMode = (_colorMode === 'pace') ? 'hr' : 'pace';
            document.getElementById('_hrbtn').textContent = 'Color: ' + (_colorMode === 'hr' ? 'HR' : 'Pace');
            document.getElementById('_legendPace').style.display = (_colorMode === 'pace') ? 'block' : 'none';
            document.getElementById('_legendHR').style.display = (_colorMode === 'hr') ? 'block' : 'none';
            for (let i = 0; i < _layers.length; i++) {{
                _layers[i].setStyle({{color: _colorFor(_routes[i])}});
            }}
            _updateStats();
        }}

        function _toggleMap() {{
            _mapHidden = !_mapHidden;
            const tilePane = document.querySelector('.leaflet-tile-pane');
            if (tilePane) tilePane.style.display = _mapHidden ? 'none' : '';
            {map_var}.getContainer().style.background = _mapHidden ? '#000' : '';
            document.getElementById('_mapbtn').textContent = _mapHidden ? 'Show Map' : 'Hide Map';
        }}

        function _updateOpacities() {{
            const n = _layers.length;
            const upd = Math.min(n, _OP.length + 1);
            for (let i = 0; i < upd; i++) {{
                _layers[n - 1 - i].setStyle({{opacity: i < _OP.length ? _OP[i] : _OP_FLOOR}});
            }}
        }}

        function _fmtHR(bpm) {{ return Math.round(bpm) + ' bpm'; }}

        function _updateStats() {{
            document.getElementById('_smiles').textContent = _totalMiles.toFixed(1);
            document.getElementById('_sruns').textContent = _idx;
            const label = document.getElementById('_spaceLabel');
            const value = document.getElementById('_space');
            if (_colorMode === 'hr') {{
                label.textContent = 'AVG HR';
                value.textContent = _hrCount > 0 ? _fmtHR(_hrSum / _hrCount) : '—';
            }} else {{
                label.textContent = 'AVG PACE';
                value.textContent = _idx > 0 ? _fmtPace(_paceSum / _idx) : '—';
            }}
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
            const poly = L.polyline([], {{color: _colorFor(r), weight: 2, opacity: _OP[0]}}).addTo(_group);
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
            if (r.hr !== null) {{ _hrSum += r.hr; _hrCount++; }}
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
                _idx = 0; _totalMiles = 0; _paceSum = 0; _hrSum = 0; _hrCount = 0; _layers = [];
                _group.clearLayers();
                _resetPRs();
                document.getElementById('_date').textContent = '—';
                document.getElementById('_count').textContent = '0 / ' + _routes.length;
                document.getElementById('_smiles').textContent = '0.0';
                document.getElementById('_sruns').textContent = '0';
                _updateStats();
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

    def _wrapped_stats(self) -> dict:
        """Per-scope stats for the Wrapped summary — one entry per calendar
        year plus an "all" entry for the full history, keyed so the page's JS
        can swap scopes client-side without recomputing anything.

        PR counts reuse `_route_metrics`'s chronological `new_prs` (each an
        all-time best *as of that run*), bucketed by the year it happened in —
        so "PRs this year" reflects real all-time bests, not merely that
        year's own best. "Favorite route" reuses `_cluster_routes`, restricted
        to members whose start date falls in the scope, so a route only
        qualifies as a given year's favorite if you actually repeated it that
        year (not just historically).
        """
        df = self._load_activity_df()
        df["moving_secs"] = df["Moving Time"].apply(_parse_moving_time)
        if "Elevation Gain" in df.columns:
            df["elev_gain_m"] = pd.to_numeric(df["Elevation Gain"], errors="coerce").fillna(0.0)
        else:
            df["elev_gain_m"] = 0.0
        df["year"] = df["date"].dt.year

        daily, active_years = self._daily_mileage_df()
        daily["year"] = daily["date"].dt.year
        _, current_streak = _streak_stats((daily["miles"] > 0).to_numpy())
        latest_year = max(active_years) if active_years else None

        metrics = self._route_metrics()
        clusters = self._cluster_routes()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        def _scope_stats(year: Optional[int]) -> dict:
            sub = df if year is None else df[df["year"] == year]
            sub_daily = daily if year is None else daily[daily["year"] == year]
            sub_metrics = metrics if year is None else [m for m in metrics if m["start_ts"].year == year]

            total_miles = float(sub["miles"].sum())
            total_secs = float(sub["moving_secs"].sum())
            elev_gain_m = float(sub["elev_gain_m"].sum())
            longest_streak, _ = _streak_stats((sub_daily["miles"] > 0).to_numpy())

            month_totals = sub_daily.groupby(sub_daily["date"].dt.to_period("M"))["miles"].sum()
            if len(month_totals):
                best_period = month_totals.idxmax()
                most_active_month = {
                    "label": best_period.strftime("%b %Y"),
                    "miles": round(float(month_totals.max()), 1),
                }
            else:
                most_active_month = None

            if year is None:
                trend_src = sub_daily.groupby(sub_daily["date"].dt.year)["miles"].sum()
                trend = [{"label": str(y), "miles": round(float(v), 1)} for y, v in trend_src.items()]
            else:
                trend_src = sub_daily.groupby(sub_daily["date"].dt.month)["miles"].sum()
                trend = [{"label": month_names[mo - 1], "miles": round(float(v), 1)} for mo, v in trend_src.items()]

            new_pr_events = [k for m in sub_metrics for k in m["new_prs"]]
            pr_counts: dict[str, int] = {}
            for k in new_pr_events:
                pr_counts[k] = pr_counts.get(k, 0) + 1

            fastest_pace = min((m["avg_pace"] for m in sub_metrics), default=None)

            favorite, best_count = None, 1  # must beat 1 to count as a repeat
            for cl in clusters:
                in_scope = [m for m in cl["members"] if year is None or m["start_ts"].year == year]
                if len(in_scope) > best_count:
                    best_count = len(in_scope)
                    distances = [m["total_dist"] / 1609.344 for m in in_scope]
                    favorite = {
                        "distanceMi": round(statistics.median(distances), 1),
                        "count": len(in_scope),
                        "bestPace": _fmt_pace(min(m["avg_pace"] for m in in_scope)),
                    }

            date_min, date_max = sub["date"].min(), sub["date"].max()
            return {
                "label": "All Time" if year is None else str(year),
                "dateRange": f"{date_min:%b %Y} – {date_max:%b %Y}" if pd.notna(date_min) else "",
                "totalMiles": round(total_miles, 1),
                "totalRuns": int(len(sub)),
                "totalHours": round(total_secs / 3600, 1),
                "elevGainFt": round(elev_gain_m * 3.28084),
                "everests": round(elev_gain_m / EVEREST_M, 2),
                "avgPace": _fmt_pace(total_secs / total_miles) if total_miles else None,
                "fastestPace": _fmt_pace(fastest_pace),
                "longestStreak": int(longest_streak),
                "currentStreak": int(current_streak) if (year is None or year == latest_year) else None,
                "mostActiveMonth": most_active_month,
                "prCount": len(new_pr_events),
                "prBreakdown": {key: pr_counts.get(key, 0) for _, key, _ in PR_DISTANCES},
                "favoriteRoute": favorite,
                "marathons": round(total_miles / 26.2, 1),
                "trend": trend,
            }

        year_keys = [str(y) for y in reversed(active_years)]
        scopes = {k: _scope_stats(int(k)) for k in year_keys}
        scopes["all"] = _scope_stats(None)
        return {
            "scopes": scopes,
            "order": year_keys + ["all"],
            "defaultScope": str(latest_year) if latest_year is not None else "all",
        }

    def build_wrapped_summary(self, output: str = "strava_wrapped.html") -> None:
        """Shareable, story-card "Wrapped" summary — one set of slides per
        calendar year (plus an all-time set), selectable via a pill bar and
        navigated like a story: arrows, dot nav, keyboard, or swipe.

        All scopes' stats are computed once in Python and shipped as a single
        JSON payload; switching scopes/slides is pure client-side rendering,
        no regeneration needed.
        """
        payload = self._wrapped_stats()
        if not payload["scopes"]:
            print("No activity data found — run .load() first")
            return

        pr_cells = "".join(
            f'<div class="prcell"><div class="prcell-label">{label}</div>'
            f'<div class="prcell-val" data-pr="{key}">—</div></div>'
            for label, key, _ in PR_DISTANCES
        )
        pill_html = "".join(
            f'<button class="pill" data-scope="{k}">{"All Time" if k == "all" else k}</button>'
            for k in payload["order"]
        )

        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Strava Wrapped</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --ink: #f0f0f0; --ink-dim: #8b949e;
    --accent: #fc4c02; --accent-dim: rgba(252,76,2,0.18);
    --gold: #ffd700;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    overflow: hidden;
  }}
  .wrap {{
    position: relative; height: 100%; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 18px; padding: 24px 12px;
  }}
  .bg-glow {{
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: radial-gradient(circle at 50% 30%, var(--accent-dim), transparent 60%);
  }}
  .scope-bar {{
    position: relative; z-index: 1; display: flex; gap: 8px; flex-wrap: wrap;
    justify-content: center; max-width: 90vw;
  }}
  .pill {{
    background: var(--surface); border: 1px solid #2a2f38; color: var(--ink-dim);
    padding: 6px 14px; border-radius: 999px; font-size: 13px; cursor: pointer;
    font-family: inherit; transition: background .2s, color .2s, border-color .2s;
  }}
  .pill.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  .stage {{
    position: relative; z-index: 1; display: flex; align-items: center; gap: 10px;
  }}
  .nav {{
    background: var(--surface); border: 1px solid #2a2f38; color: var(--ink);
    width: 38px; height: 38px; border-radius: 50%; font-size: 18px; cursor: pointer;
    flex: none; transition: opacity .2s;
  }}
  .nav:disabled {{ opacity: 0.25; cursor: default; }}
  .card-viewport {{
    position: relative; width: min(380px, 82vw); height: min(640px, 78vh);
    border-radius: 22px; overflow: hidden; background: var(--surface);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5); border: 1px solid #262c36;
  }}
  .card {{
    position: absolute; inset: 0; display: flex; flex-direction: column;
    padding: 34px 26px; opacity: 0; transform: translateX(24px);
    transition: opacity .35s ease, transform .35s ease; pointer-events: none;
  }}
  .card.active {{ opacity: 1; transform: translateX(0); pointer-events: auto; }}
  .card-head {{ flex: none; }}
  .card-body {{
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
  }}
  .eyebrow {{
    font-size: 11px; letter-spacing: 1.6px; color: var(--ink-dim); font-weight: 600;
  }}
  .card-title {{ font-size: 22px; font-weight: 700; margin: 4px 0 2px; }}
  .card-sub {{ font-size: 12.5px; color: var(--ink-dim); }}
  .hero-block {{ text-align: center; }}
  .hero-num {{
    font-size: 58px; font-weight: 800; color: var(--accent); line-height: 1;
    letter-spacing: -1px;
  }}
  .hero-cap {{
    font-size: 12px; letter-spacing: 1.2px; color: var(--ink-dim); margin-top: 8px;
  }}
  .stat-row {{ display: flex; justify-content: center; gap: 28px; margin-top: 22px; }}
  .stat {{ text-align: center; }}
  .stat-val {{ font-size: 20px; font-weight: 700; }}
  .stat-cap {{ font-size: 10px; letter-spacing: .8px; color: var(--ink-dim); margin-top: 3px; }}
  .fun-line {{ text-align: center; font-size: 13px; color: var(--ink-dim); margin-top: 20px; }}
  .fun-line b {{ color: var(--ink); }}
  .badge {{
    display: inline-block; margin-top: 14px; font-size: 12px; color: var(--gold);
    background: rgba(255,215,0,0.1); border: 1px solid rgba(255,215,0,0.3);
    padding: 5px 12px; border-radius: 999px;
  }}
  .empty-state {{ text-align: center; color: var(--ink-dim); font-size: 14px; padding: 0 10px; }}
  .pr-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px 10px;
    text-align: center;
  }}
  .prcell-label {{ font-size: 10px; letter-spacing: .6px; color: var(--ink-dim); }}
  .prcell-val {{ font-size: 17px; font-weight: 700; margin-top: 4px; }}
  .prcell-val.hit {{ color: var(--gold); }}
  .trend {{ display: flex; align-items: flex-end; gap: 4px; height: 90px; width: 100%; margin-top: 18px; }}
  .trend-bar {{ flex: 1; background: var(--accent); border-radius: 3px 3px 0 0; min-height: 2px; opacity: .85; }}
  .trend-labels {{ display: flex; gap: 4px; width: 100%; margin-top: 6px; }}
  .trend-labels span {{ flex: 1; text-align: center; font-size: 8.5px; color: var(--ink-dim); }}
  .dots {{ position: relative; z-index: 1; display: flex; gap: 7px; }}
  .dot {{ width: 7px; height: 7px; border-radius: 50%; background: #2a2f38; cursor: pointer; border: none; }}
  .dot.active {{ background: var(--accent); }}
  .footer-line {{ text-align: center; font-size: 11px; color: var(--ink-dim); margin-top: 16px; }}
</style></head>
<body>
<div class="bg-glow"></div>
<div class="wrap">
  <div class="scope-bar" id="scopeBar">{pill_html}</div>
  <div class="stage">
    <button class="nav" id="prevBtn">‹</button>
    <div class="card-viewport" id="cardViewport">
      <section class="card" data-slide="0">
        <div class="card-head">
          <div class="eyebrow" data-f="eyebrow0">YEAR IN REVIEW</div>
          <div class="card-title" data-f="label"></div>
          <div class="card-sub" data-f="dateRange"></div>
        </div>
        <div class="card-body">
          <div class="hero-block">
            <div class="hero-num" data-num="totalMiles" data-decimals="1">0</div>
            <div class="hero-cap">MILES RUN</div>
          </div>
          <div class="stat-row">
            <div class="stat"><div class="stat-val" data-num="totalRuns" data-decimals="0">0</div><div class="stat-cap">RUNS</div></div>
            <div class="stat"><div class="stat-val" data-num="totalHours" data-decimals="1">0</div><div class="stat-cap">HOURS MOVING</div></div>
          </div>
        </div>
      </section>

      <section class="card" data-slide="1">
        <div class="card-head">
          <div class="eyebrow">PACE &amp; EFFORT</div>
          <div class="card-title">How you moved</div>
          <div class="card-sub" data-f="label"></div>
        </div>
        <div class="card-body">
          <div class="hero-block">
            <div class="hero-num" data-f="avgPace" style="font-size:44px;">—</div>
            <div class="hero-cap">AVERAGE PACE</div>
          </div>
          <div class="stat-row">
            <div class="stat"><div class="stat-val" data-f="fastestPace">—</div><div class="stat-cap">FASTEST PACE</div></div>
            <div class="stat"><div class="stat-val" data-num="elevGainFt" data-decimals="0">0</div><div class="stat-cap">FT CLIMBED</div></div>
          </div>
          <div class="fun-line" data-show="everests">That's <b data-f="everests"></b>&#215; up Mt. Everest</div>
        </div>
      </section>

      <section class="card" data-slide="2">
        <div class="card-head">
          <div class="eyebrow">CONSISTENCY</div>
          <div class="card-title">Showing up</div>
          <div class="card-sub" data-f="label"></div>
        </div>
        <div class="card-body">
          <div class="hero-block">
            <div class="hero-num" data-num="longestStreak" data-decimals="0">0</div>
            <div class="hero-cap">LONGEST STREAK (DAYS)</div>
          </div>
          <div class="stat-row">
            <div class="stat"><div class="stat-val" data-f="mostActiveMonthLabel">—</div><div class="stat-cap">BEST MONTH</div></div>
            <div class="stat"><div class="stat-val" data-f="mostActiveMonthMiles">—</div><div class="stat-cap">MI THAT MONTH</div></div>
          </div>
          <div class="badge" data-show="currentStreak">On a <b data-f="currentStreak"></b>-day streak right now</div>
        </div>
      </section>

      <section class="card" data-slide="3">
        <div class="card-head">
          <div class="eyebrow">PERSONAL RECORDS</div>
          <div class="card-title" data-f="prTitle">New all-time bests</div>
          <div class="card-sub" data-f="label"></div>
        </div>
        <div class="card-body">
          <div class="pr-grid">{pr_cells}</div>
          <div class="empty-state" data-show="noPr">No new all-time bests in this stretch — the next one's coming.</div>
        </div>
      </section>

      <section class="card" data-slide="4">
        <div class="card-head">
          <div class="eyebrow">FAVORITE ROUTE</div>
          <div class="card-title">Your go-to loop</div>
          <div class="card-sub" data-f="label"></div>
        </div>
        <div class="card-body">
          <div class="hero-block" data-show="hasFavorite">
            <div class="hero-num" data-num="favCount" data-decimals="0">0</div>
            <div class="hero-cap">TIMES RUN</div>
          </div>
          <div class="stat-row" data-show="hasFavorite">
            <div class="stat"><div class="stat-val" data-f="favDistance">—</div><div class="stat-cap">LOOP DISTANCE</div></div>
            <div class="stat"><div class="stat-val" data-f="favPace">—</div><div class="stat-cap">BEST PACE</div></div>
          </div>
          <div class="empty-state" data-show="noFavorite">No repeat routes yet in this stretch — variety's your style.</div>
        </div>
      </section>

      <section class="card" data-slide="5">
        <div class="card-head">
          <div class="eyebrow">WRAPPED UP</div>
          <div class="card-title">The big picture</div>
          <div class="card-sub" data-f="label"></div>
        </div>
        <div class="card-body">
          <div class="hero-block">
            <div class="hero-num" data-num="marathons" data-decimals="1" style="font-size:48px;">0</div>
            <div class="hero-cap">MARATHONS WORTH OF RUNNING</div>
          </div>
          <div class="trend" id="trendBars"></div>
          <div class="trend-labels" id="trendLabels"></div>
          <div class="footer-line">strava-edas — generated from your own export</div>
        </div>
      </section>
    </div>
    <button class="nav" id="nextBtn">›</button>
  </div>
  <div class="dots" id="dots"></div>
</div>
<script>
const DATA = {json.dumps(payload)};
let _scope = DATA.defaultScope;
let _slide = 0;
const N_SLIDES = 6;

function _fmtNum(v, decimals) {{
  if (v === null || v === undefined) return '—';
  return Number(v).toLocaleString(undefined, {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}});
}}

function _countUp(el, target, decimals) {{
  const start = 0;
  const dur = 700;
  const t0 = performance.now();
  function frame(t) {{
    const p = Math.min(1, (t - t0) / dur);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = _fmtNum(start + (target - start) * eased, decimals);
    if (p < 1) requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
}}

function _renderTrend(trend) {{
  const bars = document.getElementById('trendBars');
  const labels = document.getElementById('trendLabels');
  bars.innerHTML = ''; labels.innerHTML = '';
  const max = Math.max(1, ...trend.map(p => p.miles));
  trend.forEach(p => {{
    const bar = document.createElement('div');
    bar.className = 'trend-bar';
    bar.style.height = Math.max(2, (p.miles / max) * 100) + '%';
    bar.title = p.label + ': ' + p.miles + ' mi';
    bars.appendChild(bar);
    const lab = document.createElement('span');
    lab.textContent = p.label;
    labels.appendChild(lab);
  }});
}}

function _render() {{
  const s = DATA.scopes[_scope];

  document.querySelectorAll('[data-f="label"]').forEach(el => el.textContent = s.label);
  document.querySelectorAll('[data-f="dateRange"]').forEach(el => el.textContent = s.dateRange);
  document.querySelector('[data-f="eyebrow0"]').textContent = _scope === 'all' ? 'ALL-TIME REVIEW' : 'YEAR IN REVIEW';

  document.querySelectorAll('[data-num]').forEach(el => {{
    const key = el.getAttribute('data-num');
    const decimals = parseInt(el.getAttribute('data-decimals') || '0', 10);
    let val = s[key];
    if (key === 'favCount') val = s.favoriteRoute ? s.favoriteRoute.count : 0;
    _countUp(el, val || 0, decimals);
  }});

  const setText = (sel, val) => {{ const el = document.querySelector(sel); if (el) el.textContent = (val === null || val === undefined) ? '—' : val; }};
  setText('[data-f="avgPace"]', s.avgPace);
  setText('[data-f="fastestPace"]', s.fastestPace);
  setText('[data-f="everests"]', s.everests);
  setText('[data-f="currentStreak"]', s.currentStreak);
  setText('[data-f="mostActiveMonthLabel"]', s.mostActiveMonth ? s.mostActiveMonth.label : '—');
  setText('[data-f="mostActiveMonthMiles"]', s.mostActiveMonth ? s.mostActiveMonth.miles + ' mi' : '—');
  setText('[data-f="prTitle"]', s.prCount > 0 ? s.prCount + ' new all-time best' + (s.prCount === 1 ? '' : 's') : 'New all-time bests');
  setText('[data-f="favDistance"]', s.favoriteRoute ? s.favoriteRoute.distanceMi + ' mi' : '—');
  setText('[data-f="favPace"]', s.favoriteRoute ? s.favoriteRoute.bestPace : '—');

  document.querySelectorAll('[data-pr]').forEach(el => {{
    const key = el.getAttribute('data-pr');
    const n = s.prBreakdown[key] || 0;
    el.textContent = n > 0 ? '★ ' + n : '—';
    el.classList.toggle('hit', n > 0);
  }});

  document.querySelectorAll('[data-show="everests"]').forEach(el => el.style.display = (s.everests >= 0.1) ? '' : 'none');
  document.querySelectorAll('[data-show="currentStreak"]').forEach(el => el.style.display = (s.currentStreak !== null && s.currentStreak !== undefined && s.currentStreak > 0) ? '' : 'none');
  document.querySelectorAll('[data-show="noPr"]').forEach(el => el.style.display = (s.prCount === 0) ? '' : 'none');
  document.querySelectorAll('[data-show="hasFavorite"]').forEach(el => el.style.display = s.favoriteRoute ? '' : 'none');
  document.querySelectorAll('[data-show="noFavorite"]').forEach(el => el.style.display = s.favoriteRoute ? 'none' : '');

  _renderTrend(s.trend);

  document.querySelectorAll('.pill').forEach(p => p.classList.toggle('active', p.getAttribute('data-scope') === _scope));
}}

function _goToSlide(i) {{
  _slide = Math.max(0, Math.min(N_SLIDES - 1, i));
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('active', parseInt(c.getAttribute('data-slide'), 10) === _slide));
  document.querySelectorAll('.dot').forEach(d => d.classList.toggle('active', parseInt(d.getAttribute('data-i'), 10) === _slide));
  document.getElementById('prevBtn').disabled = _slide === 0;
  document.getElementById('nextBtn').disabled = _slide === N_SLIDES - 1;
}}

document.getElementById('scopeBar').addEventListener('click', e => {{
  if (!e.target.classList.contains('pill')) return;
  _scope = e.target.getAttribute('data-scope');
  _render();
}});
document.getElementById('prevBtn').addEventListener('click', () => _goToSlide(_slide - 1));
document.getElementById('nextBtn').addEventListener('click', () => _goToSlide(_slide + 1));
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') _goToSlide(_slide + 1);
  if (e.key === 'ArrowLeft') _goToSlide(_slide - 1);
}});

let _touchX = null;
const viewport = document.getElementById('cardViewport');
viewport.addEventListener('touchstart', e => {{ _touchX = e.touches[0].clientX; }});
viewport.addEventListener('touchend', e => {{
  if (_touchX === null) return;
  const dx = e.changedTouches[0].clientX - _touchX;
  if (Math.abs(dx) > 40) _goToSlide(_slide + (dx < 0 ? 1 : -1));
  _touchX = null;
}});

const dotsEl = document.getElementById('dots');
for (let i = 0; i < N_SLIDES; i++) {{
  const d = document.createElement('button');
  d.className = 'dot'; d.setAttribute('data-i', i);
  d.addEventListener('click', () => _goToSlide(i));
  dotsEl.appendChild(d);
}}

_render();
_goToSlide(0);
</script>
</body></html>"""

        Path(output).write_text(html, encoding="utf-8")
        print(f"Wrapped summary saved → {output}")

    def run(self) -> None:
        self.load()
        if not self._all_coords:
            print("No GPS data found. Check that GPX files exist in strava/activities/")
            return
        self.build_heatmap()
        self.build_calendar_heatmap()
        self.build_animated_routes()
        self.build_wrapped_summary()


if __name__ == "__main__":
    # Default: uses every activity type in DEFAULT_TYPES, no geographic filtering.
    StravaHeatmapGenerator().run()

    # To restrict to a region (e.g. the author's own Massachusetts example data),
    # pass a bounds tuple instead:
    #   StravaHeatmapGenerator(bounds=MA_BOUNDS).run()
