"""train and evaluate a C3DIR model on data-processing's samples (PROCESSED_DIR/c3dir/):

  1. split every sample into train / val / test by date and compute the band normalization
     from train (c3dir.splits)
  2. train, keeping the weights with the lowest validation loss
  3. save as the next version in MODELS_DIR/c3dir/ (v001, v002, ...), with the split and band
     statistics it used (splits.csv, band_stats.csv)
  4. score on the test days (paper section 5.2) and save metrics and figures with the model

collect and process data first (data-processing's c3dir.collect, then c3dir.process). with PULL,
processed/c3dir is first copied from the shared bucket; with PUSH, the new model version is copied
to it at the end. change the settings below and run from the repo root (.env names the bucket):

  uv run --project src/model-train --env-file .env python -m c3dir.train
"""
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from c3dir.dataset import C3DIRDataset
from c3dir.evaluation import detection_scores, predict_curtain, predict_samples, water_scores
from c3dir.model import C3DIR
from c3dir.network import ModelConfig
from c3dir.plots import (plot_coverage, plot_coverage_disk, plot_curtain, plot_detection,
                         plot_history, plot_profiles, plot_species_curtain, plot_water)
from c3dir.splits import band_stats, split_samples, summary
from c3dir.training import TrainConfig, fit
from utilities.bucket import pull, push
from utilities.paths import PROCESSED_DIR

# settings: change these
MODEL = ModelConfig(width=32)  # paper: width 64
TRAINING = TrainConfig(steps=3000, batch_size=16, eval_every=250)  # paper: 200,000 steps, batch 1,024
VERSION = None  # None saves as the next v001, v002, ...; or a name like "v003_wide"
VAL_AFTER = None  # date: days from here go to validation; None splits ~80/10/10 by date
TEST_AFTER = None  # date: days from here go to test
PULL = False  # bucket setting: copy processed/c3dir from the shared bucket first; replaces local files that differ
PUSH = True  # bucket setting: copy the new model version to the shared bucket at the end

COLLOCATED_DIR = PROCESSED_DIR / "c3dir" / "collocated"

# 1. the split and the band normalization, over every sample present
if PULL:
    pull("processed/c3dir")
table = split_samples(val_after=VAL_AFTER, test_after=TEST_AFTER)
stats = band_stats(table)
print(summary(table).to_string())
train, val = C3DIRDataset("train", table, stats), C3DIRDataset("val", table, stats)
print(f"{len(train)} train patches, {len(val)} val patches")

# 2-3. train and save, with the split and band statistics used
model = fit(C3DIR(MODEL), train, val, TRAINING)
out = model.save(VERSION)
table.to_csv(out / "splits.csv", index=False)
stats.to_csv(out / "band_stats.csv", index=False)
print(f"saved {model.version} to {out}")

# 4. evaluate on the test days
test_files = C3DIRDataset("test", table).files
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

# share the model
if PUSH:
    push(f"models/c3dir/{model.version}")
