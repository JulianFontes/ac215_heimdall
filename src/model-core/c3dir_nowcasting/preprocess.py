"""Load the C3DIR-like data and build the model channels. Shared by training and prediction, so both see the same inputs."""
import numpy as np
import xarray as xr

from .config import S0, CLOUD_PROPERTIES

def load_channels(path, alt_max):
    """DataArray (time, channel, alt, lat, lon): 3 scaled contents, 3 masks, 1 validity flag (last, not a model input)."""
    ds = xr.open_dataset(path)
    ds = ds.sel(alt=ds.alt < alt_max)

    valid = ds.IWC.notnull().astype("float32") # handle bins outside the ERA5 column that are NaN (used to ignore values during evaluation)
    contents = [np.log10(1 + ds[v].fillna(0) / S0) / 4 for v in ("IWC", "LWC", "RWC")] # log scaling of water content as in the C3DIR paper
    masks = [ds[f"{prop}_occ"].astype("float32") for prop in CLOUD_PROPERTIES] # assigning masks

    # Channel Contents
    # Channels 0–5 (N_CH = 6) are the model inputs
    # Channels 3–5 (the masks) are the response variables
    # Channel 6 (validity) is never fed to the model. It is only used to mask the loss and the metrics.
    x = xr.concat(contents + masks + [valid], dim="channel") # concatenate all channels into a single channel dimension
    return x.transpose("time", "channel", "alt", "latitude", "longitude").astype("float32")
