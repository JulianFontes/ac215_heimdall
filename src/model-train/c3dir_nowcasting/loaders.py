"""Sliding-window loaders over the channels from preprocess.load_channels (model-core)."""
import xbatcher
from torch.utils.data import DataLoader
from xbatcher.loaders.torch import MapDataset

from .config import N_CH

def make_loader(x, t_in, batch_size, shuffle):
    """Sliding windows of t_in input hours + 1 target hour, one hour apart, using xbatcher."""
    dims = {"time": t_in + 1, **{d: x.sizes[d] for d in x.dims if d != "time"}} # grabs t_in + 1 hours, and all other dimensions
    bgen = xbatcher.BatchGenerator(x, input_dims=dims, input_overlap={"time": t_in}) # generate sliding windows of the input data
    return DataLoader(MapDataset(bgen), batch_size=batch_size, shuffle=shuffle)

def split_batch(w):
    """w: (B, t_in + 1, 7, D, H, W) -> inputs, target masks, validity mask."""
    return w[:, :-1, :N_CH], w[:, -1, 3:N_CH], w[:, -1, N_CH:N_CH + 1]
