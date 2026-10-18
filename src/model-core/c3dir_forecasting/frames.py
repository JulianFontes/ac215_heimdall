"""C3DIR fields -> forecaster frames, the same way for training and prediction.

a C3DIR field (model-deploy's c3dir_forecasting.predict) is (height_km 80, lat, lon) around a
site. a frame keeps one variable, averages the 80 levels a few at a time and cuts the central
size x size cells, so every site gives the same shape whatever its latitude:

  frame = to_frame(xr.load_dataset(path), FrameConfig())     # (levels, size, size) float32
"""
from dataclasses import asdict, dataclass

import numpy as np

from c3dir.constants import LEVELS_KM
from utilities.paths import PROCESSED_DIR

DATA_DIR = PROCESSED_DIR / "c3dir_forecast"  # goes/<site>/ and c3dir/<site>/ <YYYYmmddTHHMM>.nc


@dataclass
class FrameConfig:
    variable: str = "prob_any"  # P(any cloud), in [0, 1]
    levels_every: int = 4  # average every 4 of the 80 levels: 20 levels of 1 km
    size: int = 100  # central size x size cells (~400 km at 0.04°)

    @property
    def levels(self):
        return len(LEVELS_KM) // self.levels_every

    def heights_km(self):
        return LEVELS_KM[: self.levels * self.levels_every].reshape(self.levels, -1).mean(axis=1)

    def as_dict(self):
        return asdict(self)


def to_frame(field, config=None):
    """(levels, size, size) float32 from one C3DIR field; cells outside the GOES crop become 0."""
    config = config or FrameConfig()
    values = field[config.variable].values  # (80, lat, lon)
    values = values[: config.levels * config.levels_every].reshape(config.levels, config.levels_every,
                                                                    *values.shape[1:]).mean(axis=1)
    _, n_lat, n_lon = values.shape
    if min(n_lat, n_lon) < config.size:
        raise ValueError(f"field is {n_lat} x {n_lon} cells, smaller than size {config.size}")
    r0, c0 = (n_lat - config.size) // 2, (n_lon - config.size) // 2
    return np.nan_to_num(values[:, r0:r0 + config.size, c0:c0 + config.size]).astype(np.float32)
