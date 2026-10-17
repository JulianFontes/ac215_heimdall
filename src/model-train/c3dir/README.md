# C3DIR (model-train)

Training and evaluation of C3DIR on the samples from `data-processing`. Each run saves the next
version (`v001`, `v002`, ...) to `artifacts/models/c3dir/` with its settings, loss history, test
scores and figures.

| File | What it does |
|---|---|
| `dataset.py` | `C3DIRDataset` and `collate`: 64×64 patches along the EarthCARE track |
| `training.py` | `fit`: the loss (paper eq. 1-2), random flips, and the training loop |
| `evaluation.py` | test predictions and scores (`detection_scores`, `water_scores`) |
| `plots.py` | training history, scores, coverage maps, EarthCARE vs C3DIR curtains and profiles |
| `train.py` | the training run (settings at the top) |

## Run

From the project root:

```bash
uv run --project src/model-train python -m c3dir.train
```

Edit `MODEL`, `TRAINING` and `VERSION` at the top of `train.py` first. Training uses CUDA or
Apple MPS when available; ~30 days of data and 3,000 steps take about an hour on a laptop.

See [`docs/c3dir.md`](../../../docs/c3dir.md) for the whole pipeline.
