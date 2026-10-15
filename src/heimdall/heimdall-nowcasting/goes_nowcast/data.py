"""Load GOES-19 ABI MCMIP data and build sliding-window loaders. `prepare_data` runs every step; the functions below it are those steps."""
import glob
import os
from dataclasses import dataclass

import numpy as np
import xarray as xr
import xbatcher
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader
from xbatcher.loaders.torch import MapDataset

from .config import BANDS, STEP_MIN, TARGET_BANDS, T_IN, T_OUT, VAL_FRAC, TEST_FRAC

N_CH = len(BANDS)                                   # model input channels
TARGET_IDX = [BANDS.index(b) for b in TARGET_BANDS] # positions of the target bands among the input channels
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

def find_files(path):
    """A single file, a folder of .nc files, or a glob pattern."""
    files = sorted(glob.glob(os.path.join(path, "*.nc"))) if os.path.isdir(path) else sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"No files found for {path!r}")
    return files

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

def load_file(f):
    with xr.open_dataset(f) as ds:
        ds = ds.rename(t="time")
        cmi = xr.concat([ds[f"CMI_C{b:02d}"].reset_coords(drop=True) for b in BANDS], dim="channel").assign_coords(channel=[f"C{b:02d}" for b in BANDS])
        valid = cmi.notnull().all("channel") # a pixel is valid only if every band has a value; invalid pixels are ignored by loss and metrics
        cmi = cmi.where(valid) # bad pixels become NaN (ignored in the statistics, filled after normalizing)
        units = [ds[f"CMI_C{b:02d}"].attrs["units"] for b in BANDS] + [""] # as stored in the file: "K", or "1" (dimensionless) for reflectance
        return xr.concat([cmi, valid.expand_dims(channel=["valid"])], dim="channel").assign_coords(units=("channel", units)).load()

def load_channels(path):
    """DataArray (time, channel, y, x): len(BANDS) bands in physical units, 1 validity flag (last, not a model input).
    The unit of every channel, read from the files, is in the `units` coordinate."""
    files = find_files(path)
    print(f"Loading {len(files)} file(s), bands {list(BANDS)}")
    x = xr.concat([load_file(f) for f in files], dim="time").sortby("time").drop_duplicates("time")
    x = x.transpose("time", "channel", "y", "x").astype("float32")
    print(f"  {x.sizes['time']} frames, {x.time.values[0]} -> {x.time.values[-1]}, grid {x.sizes['y']}x{x.sizes['x']}")
    return x

def split_time(x):
    """Chronological train / validation / test split, so windows never straddle two splits."""
    n = x.sizes["time"]
    c1, c2 = int(n * (1 - VAL_FRAC - TEST_FRAC)), int(n * (1 - TEST_FRAC))
    return x.isel(time=slice(0, c1)), x.isel(time=slice(c1, c2)), x.isel(time=slice(c2, n))

def normalize(train, val, test):
    """z-score every band with the per-band mean / std of the training frames only (StandardScaler, NaN pixels ignored).
    Invalid pixels (NaN) become 0, i.e. the mean; the validity flag is kept as is.
    Returns the three normalized splits and the fitted scaler (scaler.mean_, scaler.scale_)."""
    rows = lambda a: a.isel(channel=slice(0, N_CH)).transpose("time", "y", "x", "channel") # flattened: one row per pixel, one column per band
    scaler = StandardScaler().fit(rows(train).values.reshape(-1, N_CH))
    out = []
    for a in (train, val, test):
        if a.sizes["time"] == 0: # e.g. TEST_FRAC = 0: nothing to normalize, and StandardScaler rejects empty arrays
            out.append(a)
            continue
        bands = rows(a)
        z = bands.copy(data=scaler.transform(bands.values.reshape(-1, N_CH)).reshape(bands.shape))
        z = z.transpose("time", "channel", "y", "x").fillna(0)
        out.append(xr.concat([z, a.isel(channel=slice(N_CH, None))], dim="channel").astype("float32"))
    return (*out, scaler)

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
