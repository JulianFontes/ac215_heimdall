# heimdall

## C3DIR

Predicts 3D cloud structure from GOES-19 imagery, trained on EarthCARE cloud profiles
(following White et al., 2026). So far: scraping, collocating EarthCARE with GOES-19,
plotting, and building training samples.

### Setup

```bash
uv sync
```

EarthCARE downloads need a one-time ESA MAAP login setup:
https://tropos-rsd.github.io/earthcarekit/setup/

### Run

Edit the settings at the top of `scripts/c3dir_runner.py` (time window, etc.) and run it:

```bash
uv run python scripts/c3dir_runner.py
```

Data is saved to `data/` and example figures to `artifacts/figures/c3dir_example/`
(both git-ignored).

See [`src/heimdall/c3dir/README.md`](src/heimdall/c3dir/README.md) for how the code
works, data formats and assumptions.
