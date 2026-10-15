"""turn a C3DIR sequence into (previous x frames -> next k frames) training examples.

each frame becomes a (channels, lat, lon) image: for every field in FIELDS, one channel per
level. probabilities stay in [0, 1]; water contents are scaled as in C3DIR,
log10(1 + w / 5e-5) / 4, so most values fall in [0, 1].

examples never cross a gap (missing scan, night) and every day goes wholly to one of
train / val / test, the later days held out, as in the C3DIR split.
"""
import numpy as np
import pandas as pd
import torch

from heimdall.c3dir.model.network import WATER_SCALE
from heimdall.c3dir_forecast.sequences import FIELDS

WATER_DIVISOR = 4.0  # log10(1 + 1 g m^-3 / 5e-5) ~ 4.3


def scale(name, values):
    return values if name.startswith("prob") else np.log10(1 + np.clip(values, 0, None) / WATER_SCALE) / WATER_DIVISOR


def unscale(name, values):
    return values if name.startswith("prob") else WATER_SCALE * (10 ** (values * WATER_DIVISOR) - 1)


def to_array(sequence, fields=FIELDS):
    """(time, channels, lat, lon) float32 with channels = fields x levels, and a (lat, lon) mask
    of cells inside the GOES crop at every level and time (False ones are left out of the loss)."""
    parts = [scale(f, sequence[f].values) for f in fields]  # each (time, level, lat, lon)
    array = np.concatenate(parts, axis=1).astype(np.float32)
    valid = np.isfinite(array).all(axis=(0, 1))
    return np.nan_to_num(array), valid


def channel_names(sequence, fields=FIELDS):
    return [f"{f}@{h:.2f}km" for f in fields for h in sequence.height_km.values]


def windows(times, input_steps, output_steps, cadence="10min"):
    """start indices i whose frames i .. i + input_steps + output_steps - 1 are all one cadence apart."""
    times = pd.DatetimeIndex(times)
    span = input_steps + output_steps
    step = np.diff(times.values) == pd.Timedelta(cadence).to_timedelta64()
    run = np.convolve(step.astype(int), np.ones(span - 1, int), mode="valid") if span > 1 else np.ones(len(times))
    return np.flatnonzero(run == span - 1) if span > 1 else np.arange(len(times))


def split_days(times, val_days=3, test_days=3):
    """'train' / 'val' / 'test' for each time: the last test_days days are test, the val_days before
    them val."""
    days = pd.DatetimeIndex(times).normalize()
    unique = days.unique().sort_values()
    if len(unique) < val_days + test_days + 1:
        raise ValueError(f"only {len(unique)} days; need more than val_days + test_days")
    n = len(unique)
    test, val = set(unique[n - test_days:]), set(unique[n - test_days - val_days:n - test_days])
    return np.array(["test" if d in test else "val" if d in val else "train" for d in days])


class SequenceDataset(torch.utils.data.Dataset):
    """(x, y) pairs: x (input_steps, C, H, W) previous frames, y (output_steps, C, H, W) next frames."""

    def __init__(self, array, times, split_labels, split, input_steps=6, output_steps=3, cadence="10min"):
        self.array, self.input_steps, self.output_steps = array, input_steps, output_steps
        starts = windows(times, input_steps, output_steps, cadence)
        span = input_steps + output_steps
        same = np.array([(split_labels[i:i + span] == split).all() for i in starts], bool)
        self.starts = starts[same] if len(starts) else starts
        self.times = pd.DatetimeIndex(times)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        i, n = self.starts[k], self.input_steps
        return (torch.from_numpy(self.array[i:i + n]),
                torch.from_numpy(self.array[i + n:i + n + self.output_steps]))

    def target_times(self, k):
        i = self.starts[k] + self.input_steps
        return self.times[i:i + self.output_steps]
