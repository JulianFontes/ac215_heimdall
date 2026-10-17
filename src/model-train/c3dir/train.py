"""train and evaluate a C3DIR model on data-processing's samples (PROCESSED_DIR/c3dir/):

  1. load the train / val / test split made by data-processing's c3dir.process
  2. train, keeping the weights with the lowest validation loss
  3. save as the next version in MODELS_DIR/c3dir/ (v001, v002, ...)
  4. score on the test days (paper section 5.2) and save metrics and figures with the model

collect and process data first (data-processing's c3dir.collect, then c3dir.process). change
the settings below and run from the repo root:

  uv run --project src/model-train python -m c3dir.train
"""
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from c3dir.dataset import C3DIRDataset
from c3dir.evaluation import detection_scores, predict_curtain, predict_samples, water_scores
from c3dir.model import C3DIR
from c3dir.network import ModelConfig
from c3dir.paths import PROCESSED_DIR
from c3dir.plots import (plot_coverage, plot_coverage_disk, plot_curtain, plot_detection,
                         plot_history, plot_profiles, plot_species_curtain, plot_water)
from c3dir.training import TrainConfig, fit

# settings: change these
MODEL = ModelConfig(width=32)  # paper: width 64
TRAINING = TrainConfig(steps=3000, batch_size=16, eval_every=250)  # paper: 200,000 steps, batch 1,024
VERSION = None  # None saves as the next v001, v002, ...; or a name like "v003_wide"

COLLOCATED_DIR = PROCESSED_DIR / "c3dir" / "collocated"

# 1. the split
train, val = C3DIRDataset("train"), C3DIRDataset("val")
print(f"{len(train)} train patches, {len(val)} val patches")

# 2-3. train and save
model = fit(C3DIR(MODEL), train, val, TRAINING)
out = model.save(VERSION)
print(f"saved {model.version} to {out}")

# 4. evaluate on the test days
test_files = C3DIRDataset("test").files
table = predict_samples(model, test_files)
detection, water = detection_scores(table), water_scores(table)
print(detection.round(3).to_string(index=False))
print(water.round(3).to_string(index=False))
pd.concat([detection.assign(kind="detection"), water.assign(kind="water")]).to_csv(out / "metrics.csv", index=False)

figures = out / "figures"
figures.mkdir(exist_ok=True)
splits = {name: [COLLOCATED_DIR / path.name for path in files]
          for name, files in [("Train", train.files), ("Validation", val.files), ("Test", test_files)]}
for name, fig in [("history", plot_history(model.history)),
                  ("detection", plot_detection(detection_scores(table, by_level=True))),
                  ("water", plot_water(table, water)),
                  ("coverage", plot_coverage(splits)),
                  ("coverage_disk", plot_coverage_disk(splits))]:
    fig.savefig(figures / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# along-track comparisons for every test pair
for path in test_files:
    pair = xr.load_dataset(COLLOCATED_DIR / path.name)
    curtain = predict_curtain(model, pair, xr.load_dataset(path))
    for name, fig in [("curtain", plot_curtain(pair, curtain)), ("species", plot_species_curtain(pair, curtain)),
                      ("profiles", plot_profiles(pair, curtain))]:
        fig.savefig(figures / f"{path.stem}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
print(f"metrics and figures in {out}")
