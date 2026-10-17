"""Load GOES-19 ABI MCMIP files into model channels and normalize them. Shared by training and prediction, so both see the same inputs."""
import glob
import os

import numpy as np
import xarray as xr

from .config import BANDS, N_CH

def find_files(path):
    """A single file, a folder of .nc files, or a glob pattern."""
    files = sorted(glob.glob(os.path.join(path, "*.nc"))) if os.path.isdir(path) else sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"No files found for {path!r}")
    return files

def load_file(f):
    print('Loading + Formatting ...')
    with xr.open_dataset(f) as ds:
        ds = ds.rename(t="time")
        cmi = xr.concat([ds[f"CMI_C{b:02d}"].reset_coords(drop=True) for b in BANDS], dim="channel").assign_coords(channel=[f"C{b:02d}" for b in BANDS])
        valid = cmi.notnull().all("channel") # a pixel is valid only if every band has a value; invalid pixels are ignored by loss and metrics
        cmi.values[:, ~valid.values] = np.nan # in place, no copy: bad pixels become NaN in every band (ignored in the statistics, filled after normalizing)
        units = [ds[f"CMI_C{b:02d}"].attrs["units"] for b in BANDS] + [""] # as stored in the file: "K", or "1" (dimensionless) for reflectance
        return xr.concat([cmi, valid.expand_dims(channel=["valid"])], dim="channel").assign_coords(units=("channel", units))

def load_channels(path):
    """DataArray (time, channel, y, x): len(BANDS) bands in physical units, 1 validity flag (last, not a model input).
    The unit of every channel, read from the files, is in the `units` coordinate."""
    files = find_files(path)
    print(f"Loading {len(files)} file(s), bands {list(BANDS)}")
    if len(files) == 1:
        x = load_file(files[0]) # a single file is already in time order, so skip the concat/sort
    else:
        x = xr.concat([load_file(f) for f in files], dim="time")

        # sorting and dropping duplicates create copies of the whole array
        # so only do it when the files are out of order or there's overlap
        if not x.indexes["time"].is_monotonic_increasing:
            x = x.sortby("time")
        if not x.indexes["time"].is_unique:
            x = x.drop_duplicates("time")

    x = x.transpose("time", "channel", "y", "x")
    print(f"  {x.sizes['time']} frames, {x.time.values[0]} -> {x.time.values[-1]}, grid {x.sizes['y']}x{x.sizes['x']}")
    return x

def scale(a, mean, std):
    """Normalize bands with the training mean / std (per band, BANDS order) while preserving the validity channel."""
    if a.sizes["time"] == 0:
        return a

    # scale each band to z-scores in place to avoid allocating new arrays
    for c in range(N_CH):
        band = a.isel(channel=c)

        # Sample ~10 valid pixels before normalization to check if it succeeds
        # This is a sanity check to ensure that the normalization is applied correctly
        flat = band.data.flat
        idx = np.linspace(0, band.data.size - 1, min(1000, band.data.size), dtype=int)
        idx = idx[np.isfinite(flat[idx])][:10]
        original = np.asarray(flat[idx]).copy()

        band.data -= mean[c] # subtract the training mean (center to 0)
        band.data /= std[c] # divide by the training std (scale to unit variance)
        band.data[np.isnan(band.data)] = 0.0 # replace nan values without allocating a new array

        assert np.allclose(
            flat[idx], (original - mean[c]) / std[c], atol=1e-5
        ), f"Normalization failed for channel {c}"

    return a
