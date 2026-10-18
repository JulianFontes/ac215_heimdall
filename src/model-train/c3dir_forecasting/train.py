"""train a ConvLSTM to forecast C3DIR cloud fields: from the INPUT_STEPS frames up to t, the next
OUTPUT_STEPS frames (t+1 .. t+m), on the fields made by model-deploy's c3dir_forecasting.predict.

  1. (PULL) copy processed/c3dir_forecast from the shared bucket
  2. load every site's fields as frames; whole days go to train / val / test
  3. train, keeping the weights with the lowest validation loss; save as the next version in
     MODELS_DIR/c3dir_forecasting/ (v001, v002, ...)
  4. score the test days by lead time next to persistence; save metrics and figures with the model
  5. (PUSH) copy the new version to the shared bucket

every setting below can be overridden by an environment variable of the same name (e.g.
INPUT_STEPS=6 STEPS=1000), so a container can change them without editing code. run from the repo
root (add --env-file .env when PULL / PUSH need the bucket):

  uv run --project src/model-train python -m c3dir_forecasting.train
"""
import matplotlib.pyplot as plt

from c3dir_forecasting.dataset import ForecastDataset, load_series, split_days
from c3dir_forecasting.frames import FrameConfig
from c3dir_forecasting.model import Forecaster
from c3dir_forecasting.network import NetConfig
from c3dir_forecasting.plots import plot_example, plot_history, plot_scores
from c3dir_forecasting.training import TrainConfig, evaluate, fit
from utilities.settings import setting

# settings: change these, or override any of them with an environment variable
INPUT_STEPS = setting("INPUT_STEPS", 3)  # n: frames up to t (3 x 10 min = 30 min)
OUTPUT_STEPS = setting("OUTPUT_STEPS", 3)  # m: frames forecast, t+1 .. t+3 (30 min)
VAL_DAYS = setting("VAL_DAYS", 0)  # days before the test days held out for picking the best weights
TEST_DAYS = setting("TEST_DAYS", 1)  # the last days, held out for scoring
STEPS = setting("STEPS", 300)  # training steps
BATCH_SIZE = setting("BATCH_SIZE", 8)
WIDTH = setting("WIDTH", 32)  # network channels
VERSION = setting("VERSION", "")  # empty: the next v001, v002, ...; or a name like "v002_6in"
PULL = setting("PULL", False)  # bucket: copy processed/c3dir_forecast from the shared bucket first
PUSH = setting("PUSH", False)  # bucket: copy the new model version to the shared bucket at the end

FRAMES = FrameConfig()  # prob_any, 80 levels averaged 4 at a time, central 100 x 100 cells

# 1-2. data
if PULL:
    from utilities.bucket import pull
    pull("processed/c3dir_forecast/c3dir")
series = load_series(frame_config=FRAMES)
days = split_days({t.normalize() for times, _ in series.values() for t in times}, VAL_DAYS, TEST_DAYS)
data = {split: ForecastDataset(series, [d for d, s in days.items() if s == split], INPUT_STEPS, OUTPUT_STEPS)
        for split in ("train", "val", "test")}
print(f"{len(series)} sites, {sum(len(t) for t, _ in series.values())} frames of {FRAMES.levels} levels x "
      f"{FRAMES.size} x {FRAMES.size}; " + ", ".join(f"{len(d)} {s}" for s, d in data.items()) + " examples")

# 3. train and save
training = TrainConfig(steps=STEPS, batch_size=BATCH_SIZE)
model = Forecaster(NetConfig(channels=FRAMES.levels, width=WIDTH, hidden=[WIDTH, WIDTH]), FRAMES,
                   INPUT_STEPS, OUTPUT_STEPS)
model.info = {"training": training.as_dict(), "sites": list(series),
              "days": {s: sorted(f"{d:%Y-%m-%d}" for d, v in days.items() if v == s) for s in data},
              "examples": {s: len(d) for s, d in data.items()}}
fit(model, data["train"], data["val"], training)
out = model.save(VERSION or None)
print(f"saved {model.version} to {out}")

# 4. test scores and figures
table, forecasts = evaluate(model, data["test"])
print(table.round(3).to_string(index=False))
table.to_csv(out / "metrics.csv", index=False)
figures = out / "figures"
figures.mkdir(exist_ok=True)
k = max(range(len(data["test"])), key=lambda i: data["test"][i][1].sum())  # the cloudiest test example
x, y = data["test"][k]
site, t = data["test"].describe(k)
for name, fig in [("history", plot_history(model.history)), ("scores", plot_scores(table)),
                  ("example", plot_example(x.numpy(), y.numpy(), forecasts[k], FRAMES.heights_km(),
                                           f"{site}, t = {t:%Y-%m-%d %H:%M} UTC, ConvLSTM {model.version}"))]:
    fig.savefig(figures / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
print(f"metrics and figures in {out}")

# 5. share the model
if PUSH:
    from utilities.bucket import push
    push(f"models/c3dir_forecasting/{model.version}")
