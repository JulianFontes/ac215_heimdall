# C3DIR (model-deploy)

Predictions from a trained C3DIR model on any GOES-19 scan: the column above a point and the 3D
field around it.

| File | What it does |
|---|---|
| `inference.py` | `goes_scene`, `predict_column`, `predict_region` |
| `plots.py` | column and region figures |
| `predict.py` | a prediction run (settings at the top) |

## Run

From the project root:

```bash
uv run --project src/model-deploy python -m c3dir.predict
```

Edit `SITE` (or `LAT` / `LON` / `NAME`), `TIME` and `MODEL_VERSION` at the top of `predict.py`
first. Outputs: `artifacts/figures/c3dir_predict/<site>_<time>.png` and `.csv` (the column), and
`data/predictions/c3dir/<site>_<time>_region.nc` (the 3D field). The model was trained on daytime
passes only, so the script warns when the sun is down.

See [`docs/c3dir.md`](../../../docs/c3dir.md) for the whole pipeline.
