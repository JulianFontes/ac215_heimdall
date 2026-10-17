# C3DIR (data-processing)

Builds C3DIR training data from EarthCARE and GOES-19 in two runs.

| File | What it does |
|---|---|
| `earthcare.py` | `EarthCareFrames`: finds the ACM_CAP_2B frames GOES-19 can see and downloads them → `data/raw/earthcare/` |
| `goes.py` | `GOESStrips`: downloads the GOES-19 scans within 2.5 min of each pass and keeps the strip where the track can appear → `data/raw/goes/` |
| `labels.py` | `EarthCareLabels`: averages each frame onto 80 levels of 0.25 km |
| `collocation.py` | `Collocation`: matches EarthCARE voxels to GOES lines of sight (within 5 km and 2.5 min) |
| `samples.py` | `Samples`: training samples, train / val / test split by day, band statistics |
| `plots.py` | collocation figures (map, curtain, line-of-sight voxels) |
| `collect.py` | download run for a time window (settings at the top) |
| `process.py` | processing run: labels, collocation, an example figure, samples and the split |

## Run

From the project root:

```bash
uv run --project src/data-processing python -m c3dir.collect
uv run --project src/data-processing python -m c3dir.process
```

Edit `START` / `END` at the top of each script first (and `VAL_AFTER` / `TEST_AFTER` in
`process.py` for the split). Both runs are cached, so re-running a window only does what's new.
EarthCARE downloads need a one-time ESA MAAP login setup:
https://tropos-rsd.github.io/earthcarekit/setup/

See [`docs/c3dir.md`](../../../docs/c3dir.md) for the whole pipeline.
