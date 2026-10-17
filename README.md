# heimdall

3D cloud structure from GOES-19 imagery, trained on EarthCARE cloud profiles (following
White et al., 2026).

## Services

Each service in `src/` has its own `pyproject.toml` (managed with uv) and is run from its
own folder.

| Service | What it does |
|---|---|
| [`src/c3dir/`](src/c3dir/README.md) | scrape and collocate EarthCARE with GOES-19, train C3DIR, predict 3D clouds |
| `src/solar-forecasting/` | SURFRAD station exploration for solar applications |
| [`src/nowcasting/`](src/nowcasting/README.md) | placeholder: GOES nowcasting and cloud forecasting |
| [`src/frontend-app/`](src/frontend-app/README.md) | placeholder: web front end |

Data and outputs go to `data/` and `artifacts/` at the repo root (git-ignored), shared by
every service.

## C3DIR quick start

```bash
cd src/c3dir
uv sync
uv run python scripts/c3dir_runner.py
```

Edit the settings at the top of `scripts/c3dir_runner.py` (time window, etc.) first.
EarthCARE downloads need a one-time ESA MAAP login setup:
https://tropos-rsd.github.io/earthcarekit/setup/

See [`src/c3dir/README.md`](src/c3dir/README.md) for training, prediction, how the code
works, data formats, and assumptions.
