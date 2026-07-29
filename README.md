# Strava Heatmap

Generates two interactive HTML maps from a Strava data export.

## Setup

```bash
pip install -r requirements.txt
```

Place your Strava export under `strava/`:
```
strava/
  activities.csv
  activities/
    *.gpx
```

## Usage

```bash
python3 generate_heatmap.py
```

Outputs:
- `strava_heatmap.html` — density heatmap of GPS points
- `strava_routes.html` — individual runs as polylines

Open either file in a browser.
