"""
Strava Heatmap Generator
Produces two interactive HTML maps from a Strava data export:
  - strava_heatmap.html  : density heatmap of all GPS points
  - strava_routes.html   : individual run paths as polylines
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import folium
from folium.plugins import HeatMap

GPX_NS = "http://www.topografix.com/GPX/1/1"
DATA_DIR = Path("strava")
ACTIVITIES_CSV = DATA_DIR / "activities.csv"
ACTIVITIES_DIR = DATA_DIR / "activities"

# Activity types to include (case-insensitive)
DEFAULT_TYPES = {"run"}


class StravaHeatmapGenerator:
    def __init__(
        self,
        activity_types: set[str] = DEFAULT_TYPES,
        heatmap_radius: int = 8,
        heatmap_blur: int = 10,
        heatmap_min_opacity: float = 0.3,
        default_zoom: int = 12,
    ) -> None:
        self.activity_types = {t.lower() for t in activity_types}
        self.heatmap_radius = heatmap_radius
        self.heatmap_blur = heatmap_blur
        self.heatmap_min_opacity = heatmap_min_opacity
        self.default_zoom = default_zoom
        self._activities: list[dict] = []
        self._all_coords: list[tuple[float, float]] = []
        self._routes: list[list[tuple[float, float]]] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "StravaHeatmapGenerator":
        self._activities = self._read_csv()
        print(f"Found {len(self._activities)} matching activities in CSV")
        self._routes, self._all_coords = self._parse_gpx_files()
        print(
            f"Parsed {len(self._routes)} routes, "
            f"{len(self._all_coords):,} total GPS points"
        )
        return self

    def _read_csv(self) -> list[dict]:
        rows = []
        with open(ACTIVITIES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["Activity Type"].lower() in self.activity_types:
                    rows.append(row)
        return rows

    def _parse_gpx_files(
        self,
    ) -> tuple[list[list[tuple[float, float]]], list[tuple[float, float]]]:
        routes: list[list[tuple[float, float]]] = []
        all_coords: list[tuple[float, float]] = []

        for activity in self._activities:
            filename = activity.get("Filename", "").strip()
            if not filename:
                continue
            gpx_path = DATA_DIR / filename
            coords = self._parse_gpx(gpx_path)
            if coords:
                routes.append(coords)
                all_coords.extend(coords)

        return routes, all_coords

    def _parse_gpx(self, path: Path) -> list[tuple[float, float]]:
        if not path.exists():
            return []
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            coords = []
            for trkpt in root.iter(f"{{{GPX_NS}}}trkpt"):
                lat = trkpt.get("lat")
                lon = trkpt.get("lon")
                if lat is not None and lon is not None:
                    coords.append((float(lat), float(lon)))
            return coords
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
            return []

    # ------------------------------------------------------------------
    # Map helpers
    # ------------------------------------------------------------------

    def _center(self) -> tuple[float, float]:
        lats = [c[0] for c in self._all_coords]
        lons = [c[1] for c in self._all_coords]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    def _base_map(self, center: tuple[float, float]) -> folium.Map:
        return folium.Map(
            location=center,
            zoom_start=self.default_zoom,
            tiles="CartoDB dark_matter",
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def build_heatmap(self, output: str = "strava_heatmap.html") -> None:
        if not self._all_coords:
            print("No coordinates loaded — run .load() first")
            return
        center = self._center()
        m = self._base_map(center)
        HeatMap(
            self._all_coords,
            radius=self.heatmap_radius,
            blur=self.heatmap_blur,
            min_opacity=self.heatmap_min_opacity,
        ).add_to(m)
        m.save(output)
        print(f"Density heatmap saved → {output}")

    def build_routes(
        self,
        output: str = "strava_routes.html",
        line_color: str = "#FC4C02",
        line_weight: float = 1.5,
        line_opacity: float = 0.5,
        max_routes: Optional[int] = None,
    ) -> None:
        if not self._routes:
            print("No routes loaded — run .load() first")
            return
        center = self._center()
        m = self._base_map(center)
        routes = self._routes if max_routes is None else self._routes[:max_routes]
        for route in routes:
            folium.PolyLine(
                route,
                color=line_color,
                weight=line_weight,
                opacity=line_opacity,
            ).add_to(m)
        m.save(output)
        print(f"Route overlay saved → {output}")

    def run(self) -> None:
        self.load()
        if not self._all_coords:
            print("No GPS data found. Check that GPX files exist in strava/activities/")
            return
        self.build_heatmap()
        self.build_routes()


if __name__ == "__main__":
    StravaHeatmapGenerator().run()
