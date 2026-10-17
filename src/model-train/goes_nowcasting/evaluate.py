"""Evaluate a saved checkpoint on the validation split (or the test split) and write the plots train.py makes at the end of a run.
The forecast figures are made for one window per day of the split (forecasts/<date>/), the one starting closest to --hour UTC.
The frames are those in the run's splits.json (written by train.py), or split with config.VAL_FRAC / TEST_FRAC for older runs,
and they are normalized with the training mean / std stored in the checkpoint.
Example (from the project root):
    python -m goes_nowcasting.evaluate -p data/noaa-goes19/aug2025/MCMIPC_G19_20250801T0000_20250808T0000_psu.nc \
        -c training_runs/pcu_week1/convlstm_best.pt
"""
import argparse
import json
import os

import numpy as np
import torch

from goes_nowcasting.config import BANDS, TARGET_IDX, T_IN
from goes_nowcasting.loaders import make_loader, split_time
from goes_nowcasting.preprocess import load_channels, scale
from goes_nowcasting.evaluation import evaluate
from goes_nowcasting.model import ConvLSTM
from goes_nowcasting.plots import plot_forecast, plot_vs_persistence
from goes_nowcasting.train import get_device

def daily_windows(loader, hour):
    """{date: window index}: for each day of the split, the window whose forecast starts (last input frame) closest to `hour` UTC."""
    best, index = {}, 0
    for d in loader.dataset.datasets:
        t = d.X_generator.ds.time.values
        for i in range(len(d)): # window i starts at frame i, its last input frame is i + T_IN - 1
            t0 = t[i + T_IN - 1]
            day = np.datetime_as_string(t0, unit="D")
            gap = abs((t0 - np.datetime64(f"{day}T{hour:02d}:00")) / np.timedelta64(1, "m"))
            if day not in best or gap < best[day][0]:
                best[day] = (gap, index)
            index += 1
    return {day: i for day, (_, i) in best.items()}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--path", required=True, help="MCMIP .nc file, folder of .nc files, or glob pattern (the data the model was trained on)")
    ap.add_argument("-c", "--ckpt", required=True, help="checkpoint written by train.py, e.g. training_runs/<run>/convlstm_best.pt")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--hour", type=int, default=18, help="UTC hour the daily forecast figures start closest to (default 18, early afternoon in the eastern US)")
    ap.add_argument("--out", default=None, help="output folder (default: the checkpoint's folder)")
    args = ap.parse_args()
    out = args.out or os.path.dirname(args.ckpt)
    os.makedirs(out, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = ConvLSTM().to(device)
    model.load_state_dict(ckpt["model"])
    print(f"checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f}")
    mean, std = ckpt["mean"].cpu().numpy(), ckpt["std"].cpu().numpy()
    stats = (torch.tensor(mean[TARGET_IDX], device=device), torch.tensor(std[TARGET_IDX], device=device))

    x = load_channels(args.path)
    units = dict(zip(BANDS, x["units"].values.tolist())) # as in train.py: {band: unit}
    splits_file = os.path.join(os.path.dirname(args.ckpt), "splits.json") # written by train.py: timestamps of the frames in each split
    if os.path.exists(splits_file):
        with open(splits_file) as f:
            times = np.array(json.load(f)[args.split], dtype="datetime64[ns]")
        if len(times) == 0:
            raise ValueError(f"The run has no {args.split} frames ({splits_file})")
        split = x.sel(time=times) # exactly the frames the run used
        print(f"using the {args.split} timestamps from {splits_file}")
    else: # runs from before train.py saved splits.json
        _, val, test = split_time(x)
        split = val if args.split == "val" else test
        print(f"no splits.json next to the checkpoint, splitting with config.VAL_FRAC / TEST_FRAC")
    print(f"{args.split}: {split.sizes['time']} frames, {split.time.values[0]} -> {split.time.values[-1]}")
    loader = make_loader(scale(split, mean, std), args.batch, shuffle=False) # the training statistics, not refitted
    del x, split
    if loader is None:
        raise ValueError(f"The {args.split} split is too short for one window")

    name = "validation" if args.split == "val" else "test"
    loss, r, base = evaluate(model, loader, stats, device, with_baseline=True, desc=name)
    print(f"{name} loss {loss:.5f}")
    with open(os.path.join(out, f"{name}_scores.json"), "w") as f:
        json.dump({"convlstm": r, "persistence": base}, f, indent=2)
    plot_vs_persistence(r, base, units, os.path.join(out, "vs_persistence.png"))
    for day, i in daily_windows(loader, args.hour).items():
        plot_forecast(model, loader, stats, units, device, os.path.join(out, "forecasts", day), index=i)
        print(f"forecast figures for {day} written to {os.path.join(out, 'forecasts', day)}")
    print(f"outputs written to {out}")

if __name__ == "__main__":
    main()
