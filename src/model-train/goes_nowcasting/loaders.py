"""Split, normalize and build sliding-window loaders over the channels from preprocess.load_channels (model-core).
`prepare_data` runs every step; the functions below it are those steps."""
from dataclasses import dataclass

import numpy as np
import xbatcher
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader
from xbatcher.loaders.torch import MapDataset

from tqdm import tqdm

from .config import BANDS, N_CH, STEP_MIN, TARGET_IDX, T_IN, T_OUT, VAL_FRAC, TEST_FRAC
from .preprocess import load_channels, scale

MAX_GAP_S = 1.5 * STEP_MIN * 60                     # time gap (s) that breaks a sequence, so windows never span a data outage

@dataclass
class Data:
    """Everything training needs from the data."""
    train: DataLoader
    val: DataLoader
    test: DataLoader | None # None if the test split is too short for one window
    mean: np.ndarray        # per band (BANDS order), physical units, float32
    std: np.ndarray
    units: dict             # {band: unit} as read from the files, e.g. {13: "K"}

def prepare_data(path, batch_size):
    """Load the files, split chronologically, normalize with the training statistics, and build the sliding-window loaders."""
    x = load_channels(path)
    train, val, test = split_time(x)
    train, val, test, scaler = normalize(train, val, test)
    train_loader = make_loader(train, batch_size, shuffle=True)
    val_loader = make_loader(val, batch_size, shuffle=False)
    test_loader = make_loader(test, batch_size, shuffle=False)
    if train_loader is None or val_loader is None:
        raise ValueError("Not enough frames for the train / validation split: need more data or a smaller T_IN + T_OUT in config.py")
    describe_splits({"train": (train, train_loader), "val": (val, val_loader), "test": (test, test_loader)})
    return Data(train_loader, val_loader, test_loader,
                mean=scaler.mean_.astype("float32"), std=scaler.scale_.astype("float32"), # float32: MPS has no float64
                units=dict(zip(BANDS, x["units"].values.tolist())))

def describe_splits(splits):
    """Print the time range, frames and windows (samples) of every split, and the totals. splits: {name: (frames, loader)}."""
    minute = lambda t: str(np.datetime64(t, "m")).replace("T", " ")
    print("  split  time (UTC)                               frames  windows")
    frames = windows = 0
    for name, (x, loader) in splits.items():
        n, w = x.sizes["time"], len(loader.dataset) if loader is not None else 0
        span = f"{minute(x.time.values[0])} -> {minute(x.time.values[-1])}" if n else "none"
        print(f"  {name:<6} {span:<40} {n:>6}  {w:>7}")
        frames, windows = frames + n, windows + w
    print(f"  {'total':<6} {'':<40} {frames:>6}  {windows:>7}")

def split_time(x):
    """Chronological train / validation / test split, so windows never straddle two splits."""
    n = x.sizes["time"]
    c1, c2 = int(n * (1 - VAL_FRAC - TEST_FRAC)), int(n * (1 - TEST_FRAC))
    return x.isel(time=slice(0, c1)), x.isel(time=slice(c1, c2)), x.isel(time=slice(c2, n))

def normalize(train, val, test, batch_size=16):
    """Z-score bands using training statistics, ignoring NaNs."""
    print("Normalizing the splits")

    bands = train.isel(channel=slice(0, N_CH)).transpose("time", "y", "x", "channel")
    scaler = StandardScaler()

    # create partial fits in batches to avoid allocating large arrays
    # when collecting statistics
    for i in tqdm(range(0, bands.sizes["time"], batch_size), desc='Fitting Scales in Batches'):
        batch = bands.isel(time=slice(i, i + batch_size))
        scaler.partial_fit(batch.values.reshape(-1, N_CH))

    m, s = scaler.mean_, scaler.scale_
    return scale(train, m, s), scale(val, m, s), scale(test, m, s), scaler

def contiguous(x):
    """Split at time gaps larger than MAX_GAP_S."""
    dt = np.diff(x.time.values).astype("timedelta64[s]").astype(int)
    edges = [0, *(np.where(dt > MAX_GAP_S)[0] + 1), x.sizes["time"]]
    return [x.isel(time=slice(a, b)) for a, b in zip(edges[:-1], edges[1:])]

def make_loader(x, batch_size, shuffle):
    """Sliding windows of T_IN input frames + T_OUT target frames, one frame apart, using xbatcher."""
    n = T_IN + T_OUT
    datasets = []
    for part in contiguous(x):
        if part.sizes["time"] < n:
            continue
        dims = {"time": n, **{d: part.sizes[d] for d in part.dims if d != "time"}} # grabs n frames, and all other dimensions
        bgen = xbatcher.BatchGenerator(part, input_dims=dims, input_overlap={"time": n - 1}) # generate sliding windows of the input data
        datasets.append(MapDataset(bgen))
    if not datasets:
        return None
    return DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=shuffle)

def split_batch(w):
    """w: (B, T_IN + T_OUT, N_CH + 1, H, W) -> inputs, target frames, validity of the target frames."""
    return w[:, :T_IN, :N_CH], w[:, T_IN:, TARGET_IDX], w[:, T_IN:, N_CH:N_CH + 1]
