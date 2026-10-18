"""C3DIR fields -> (n frames up to t -> next m frames) training examples.

every site's fields are loaded as one time series of frames (frames.py). an example is a run of
n + m frames one scan apart (default 10 min) at one site, so it never crosses a gap (overnight, a
missing scan). whole days go to train / val / test, the same days at every site.
"""
import numpy as np
import pandas as pd
import torch
import xarray as xr

from c3dir_forecasting.frames import DATA_DIR, FrameConfig, to_frame


def load_series(sites=None, frame_config=None, data_dir=DATA_DIR):
    """{site: (times, frames (T, levels, size, size))} from data_dir/c3dir/<site>/<YYYYmmddTHHMM>.nc."""
    sites = sites or sorted(p.name for p in (data_dir / "c3dir").iterdir() if p.is_dir())
    series = {}
    for site in sites:
        paths = sorted((data_dir / "c3dir" / site).glob("*.nc"))
        if not paths:
            raise FileNotFoundError(f"no C3DIR fields in {data_dir / 'c3dir' / site}")
        times = pd.DatetimeIndex([pd.Timestamp(p.stem) for p in paths])
        series[site] = times, np.stack([to_frame(xr.load_dataset(p), frame_config) for p in paths])
    return series


def split_days(days, val_days=0, test_days=1):
    """'train' / 'val' / 'test' per day: the last test_days days are test, the val_days before them val."""
    days = sorted(set(days))
    if len(days) <= val_days + test_days:
        raise ValueError(f"only {len(days)} days for {val_days} val + {test_days} test days")
    return {d: "test" if k >= len(days) - test_days else "val" if k >= len(days) - test_days - val_days else "train"
            for k, d in enumerate(days)}


class ForecastDataset(torch.utils.data.Dataset):
    """(x, y): x (input_steps, levels, size, size) frames up to t, y (output_steps, ...) the frames after."""

    def __init__(self, series, days, input_steps=3, output_steps=3, cadence="10min"):
        self.series, self.input_steps, self.output_steps = series, input_steps, output_steps
        span, step = input_steps + output_steps, pd.Timedelta(cadence)
        days = {pd.Timestamp(d).normalize() for d in days}
        self.examples = []  # (site, index of the first input frame)
        for site, (times, _) in series.items():
            for i in range(len(times) - span + 1):
                window = times[i:i + span]
                if window[0].normalize() in days and (np.diff(window) == step).all():
                    self.examples.append((site, i))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, k):
        site, i = self.examples[k]
        frames = self.series[site][1][i:i + self.input_steps + self.output_steps]
        return torch.from_numpy(frames[:self.input_steps]), torch.from_numpy(frames[self.input_steps:])

    def describe(self, k):
        """site and time of the last input frame (t) of example k."""
        site, i = self.examples[k]
        return site, self.series[site][0][i + self.input_steps - 1]
