# C3DIR forecasting data, step 2: C3DIR fields (model-deploy)

Runs a trained C3DIR model on every GOES crop saved by step 1,
[`data-processing/c3dir_forecasting`](../../data-processing/c3dir_forecasting/README.md), and
saves the 3D cloud field around each site: the time series the forecaster learns from. Separate
from `c3dir_nowcasting`.

| File | What it does |
|---|---|
| `predict.py` | C3DIR on every crop without a field yet (settings at the top) |

## Run

From the project root, after step 1:

```bash
uv run --project src/model-deploy python -m c3dir_forecasting.predict
```

Needs a trained C3DIR model in `artifacts/models/c3dir/<version>/` (default `v002`). Cached:
crops that already have a field are skipped, so run it again after collecting more days.

## Outputs

In `data/processed/c3dir_forecast/c3dir/<site>/<YYYYmmddTHHMM>.nc` (git-ignored), one per crop:

| | |
|---|---|
| grid | ±2° latitude (wider in longitude by 1 / cos lat) at 0.04° (~4 km): 101 × 101–133 cells, ~440 km across |
| levels | all 80 C3DIR levels, 0.125–19.875 km |
| variables | `prob_any / ice / liquid / rain`, `ice / liquid / rain_water_g_m3`, plus GOES bands 1, 2, 3 and 13 on the same grid |
| attrs | model version, site and its lat / lon, the actual scan start, the sun zenith angle at the site |

~16–19 MB per field. Columns are normal to the surface: each cell at each height takes the voxel
of the GOES pixel whose line of sight crosses that height above it (see `c3dir.inference.predict_region`).
