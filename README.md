# Strava Heatmap

Turn your Strava bulk data export into three interactive HTML visualizations —
no Strava API access, account linking, or server required. Everything runs
locally against the export you already have.

- `strava_heatmap.html` — density heatmap of every GPS point you've recorded
- `strava_calendar.html` — GitHub-style calendar heatmap of daily mileage, plus
  monthly mileage bars and a weekly mileage trend line
- `strava_animated.html` — chronological replay of your routes accumulating on
  the map over time, with a pace legend, stats ticker, and segment PRs

This works with anyone's Strava data — just drop in your own export.

## Scope

Built for **running** data, and defaults to only including `Run` activities
(`DEFAULT_TYPES = {"run"}`). The pace coloring, GPS-noise filtering, and PR
distances throughout `generate_heatmap.py` are all tuned around running paces.

The density heatmap and calendar heatmap are activity-agnostic — pointing
`activity_types` at `{"ride"}`, `{"walk"}`, `{"hike"}`, etc. (see
[Configuration](#configuration)) works fine for those two outputs, since they
just aggregate GPS points and distance/time. The animated replay
(`strava_animated.html`) will not: it discards any segment faster than
5:00/mile as GPS noise, which throws out real cycling (or any faster) data,
and its pace-based color scale and PR distances (400m through half marathon)
are calibrated for running paces and race distances. Using it for a faster
activity type requires overriding `fast_pace_sec_per_mile` /
`slow_pace_sec_per_mile` at minimum, and the PR panel will still be
meaningless for anything other than running.

## Setup

```bash
pip install -r requirements.txt
```

### Get your Strava export

1. Go to [strava.com/athlete/delete_your_account](https://www.strava.com/athlete/delete_your_account)
   (this page also offers a data export, and does **not** delete your account).
2. Request your archive and wait for the email — it can take a few hours.
3. Unzip the archive and copy the relevant pieces into this repo as:

```
strava/
  activities.csv
  activities/
    *.gpx
```

`activities.csv` is the top-level activity list from the export. `activities/`
holds the per-activity GPS files it references via its `Filename` column.

> **Note:** this project only parses `.gpx` files. Strava exports increasingly
> include some activities as `.fit.gz` instead — those are skipped silently
> (they just won't appear on the map). If you need them, convert them to GPX
> first with a tool like [gpsbabel](https://www.gpsbabel.org/) or
> [fitdecode](https://pypi.org/project/fitdecode/).

## Usage

```bash
python3 generate_heatmap.py
```

Open any of the three generated HTML files in a browser.

## Configuration

All tuning is done by passing arguments to `StravaHeatmapGenerator` in
`generate_heatmap.py` — no other code changes needed. Everything below is
optional; sensible defaults are used if you omit it.

```python
from generate_heatmap import StravaHeatmapGenerator, MA_BOUNDS

StravaHeatmapGenerator(
    activity_types={"run"},          # which "Activity Type" values (from the CSV) to include
    heatmap_radius=8,                # density heatmap point radius, in pixels
    heatmap_blur=10,                 # density heatmap blur, in pixels
    heatmap_min_opacity=0.3,         # density heatmap minimum opacity
    default_zoom=12,                 # initial map zoom level
    fast_pace_sec_per_mile=360.0,    # pace (sec/mi) treated as "fast" for color scaling — 6:00/mi
    slow_pace_sec_per_mile=540.0,    # pace (sec/mi) treated as "slow" for color scaling — 9:00/mi
    bounds=None,                     # optional (min_lat, max_lat, min_lon, max_lon) filter
).run()
```

`bounds` is off by default, so every route in your export is included. It
exists so you can restrict the maps to a specific region — useful if your
export spans multiple places you've lived and you only want one shown. The
repo ships `MA_BOUNDS`, a Massachusetts bounding box used only for the
author's own example data; pass your own tuple (or `MA_BOUNDS`) if you want
that behavior:

```python
StravaHeatmapGenerator(bounds=MA_BOUNDS).run()
```

## Privacy

Everything runs locally — your activity data never leaves your machine. The
generated HTML files embed your coordinates directly, so treat them like any
other file containing your location history (e.g. don't publish
`strava_animated.html` publicly if you'd rather keep your routes private).
