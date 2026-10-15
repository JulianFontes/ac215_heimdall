# heimdall

## C3DIR forecasting (branch `ruhi-forecasting`)

A ConvLSTM that forecasts the next C3DIR 3D cloud fields from the previous ones. C3DIR (in
`src/heimdall/c3dir/`, brought over from the `alessandra` branch) predicts 3D cloud structure
from GOES-19 imagery. This branch runs it on every daytime 10-min scan of the month it was
trained on, around a site, and trains a ConvLSTM on that time series.

### Setup

```bash
uv sync
```

Python 3.14 (pinned in `.python-version`; torch has no 3.15 wheels yet).

### Run

1. Put a trained C3DIR model in `artifacts/models/c3dir/<version>/` (`model.pt`, `config.json`).
2. `uv run python scripts/forecast_build_sequences.py`: C3DIR output every 10 min → `data/c3dir_forecast/`
3. `uv run python scripts/forecast_train.py`: train, evaluate against persistence → `artifacts/models/c3dir_forecast/<version>/`

See [`src/heimdall/c3dir_forecast/README.md`](src/heimdall/c3dir_forecast/README.md) for
how it works, and [`src/heimdall/c3dir/README.md`](src/heimdall/c3dir/README.md) for C3DIR.
