"""train a ConvLSTM to forecast the next OUTPUT_STEPS C3DIR frames from the previous INPUT_STEPS.

  1. load the C3DIR sequence made by scripts/forecast_build_sequences.py
  2. split by day (the last days for val and test) and cut (x frames -> next frames) examples
  3. train, keeping the weights with the lowest validation loss, and save the next version to
     artifacts/models/c3dir_forecast/ (v001, v002, ...)
  4. score on the test days by lead time next to persistence; save metrics and figures with the model

change the settings below and run the file.
"""
import matplotlib.pyplot as plt

from heimdall.c3dir_forecast.dataset import SequenceDataset, channel_names, split_days, to_array
from heimdall.c3dir_forecast.evaluation import compare
from heimdall.c3dir_forecast.forecaster import Forecaster, TrainConfig
from heimdall.c3dir_forecast.network import ForecastConfig
from heimdall.c3dir_forecast.plots import plot_example, plot_history, plot_scores
from heimdall.c3dir_forecast.sequences import FIELDS, load_sequence

# settings: change these
SITE = "psu"  # the sequence to train on (see forecast_build_sequences.py)
INPUT_STEPS = 6  # x: previous frames the model sees (6 x 10 min = 1 hour)
OUTPUT_STEPS = 3  # frames forecast: t+1 .. t+3 (30 min)
VAL_DAYS, TEST_DAYS = 3, 3  # the last days, held out
CADENCE = "10min"
NETWORK = dict(width=64, hidden=[64, 64])
TRAINING = TrainConfig(steps=3000, batch_size=8, eval_every=200)
VERSION = None  # None saves as the next v001, v002, ...; or a name like "v002_2h_input"

# 1-2. data
sequence = load_sequence(SITE)
array, valid = to_array(sequence)
times = sequence.time.values
labels = split_days(times, VAL_DAYS, TEST_DAYS)
data = {s: SequenceDataset(array, times, labels, s, INPUT_STEPS, OUTPUT_STEPS, CADENCE) for s in ["train", "val", "test"]}
print(f"{array.shape[0]} frames of {array.shape[1]} channels x {array.shape[2]} x {array.shape[3]}; "
      + ", ".join(f"{len(d)} {s}" for s, d in data.items()) + " examples")

# 3. train and save
levels = sequence.sizes["height_km"]
config = ForecastConfig(channels=array.shape[1], n_prob=levels, input_steps=INPUT_STEPS, output_steps=OUTPUT_STEPS,
                        **NETWORK)
model = Forecaster(config)
model.info = {"c3dir_version": sequence.attrs.get("model"), "site": SITE, "fields": FIELDS,
              "channels": channel_names(sequence), "first_time": str(times[0]), "last_time": str(times[-1]),
              "days": {s: sorted({str(t)[:10] for t, l in zip(times, labels) if l == s}) for s in data}}
model.fit(data["train"], data["val"], TRAINING, valid=valid)
out = model.save(VERSION)
print(f"saved {model.version} to {out}")

# 4. test scores and figures
table, forecast, target, inputs = compare(model, data["test"], valid, levels)
print(table.round(3).to_string(index=False))
table.to_csv(out / "metrics.csv", index=False)
figures = out / "figures"
figures.mkdir(exist_ok=True)
heights = sequence.height_km.values
cloudiest = int((target[:, 0, :levels] >= 0.5).sum(axis=(1, 2, 3)).argmax())
example_times = [data["test"].times[data["test"].starts[cloudiest] + INPUT_STEPS - 1], *data["test"].target_times(cloudiest)]
for name, fig in [("history", plot_history(model.history)), ("scores", plot_scores(table)),
                  ("example", plot_example(inputs[cloudiest], target[cloudiest], forecast[cloudiest], heights, valid,
                                           example_times, f"{SITE}: ConvLSTM {model.version}"))]:
    fig.savefig(figures / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
print(f"metrics and figures in {out}")
