# Forecasting Training Pipeline
"""
Small-scale 3D ConvLSTM nowcasting on the C3DIR-like ERA5 volumes (c3dir_like_*.nc).

TODO: This currently meant for small-scale experimentation until C3DIR data is available.

Task: from the previous T_IN hours of (ice, liquid, rain) water content + occurrence masks on the
(altitude, latitude, longitude) grid, predict the three occurrence masks one hour ahead.

Files:  config.py      constants and fixed settings
        data.py        loading, channels, sliding-window loaders
        model.py       ConvLSTM3d
        evaluation.py  loss, CSI / F1, validation pass
        plots.py       training curves, best model vs persistence, height-longitude forecast sections
        forecasting.py this script: the training loop

Outputs (in the --out folder, default runs/):
    convlstm3d_best.pt   best checkpoint
    history.csv          per-epoch losses and validation scores
    curves.png           train/val loss + validation CSI per epoch, persistence CSI as dashed lines
    vs_persistence.png   validation CSI: best model vs persistence
    forecast_height_lon/ one height-longitude forecast plot per validation stretch

Example: python forecasting.py -p c3dir_like_era5_20260601_20260603_N45_5_W74_S39_5_E68.nc --epochs 20
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import ALT_MAX, HIDDEN, N_CH, SEED, CLOUD_PROPERTIES, T_IN, VAL_FRAC, K
from data import load_channels, make_loader, split_batch
from evaluation import evaluate, weighted_bce
from model import ConvLSTM3d
from plots import plot_curves, plot_vs_persistence, plot_height_lon_sequence

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--path", required=True, help="c3dir_like_*.nc")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="runs", help="output folder (created if missing)")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "convlstm3d_best.pt"

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    #* Data: chronological split, so validation windows only use hours the model has never seen
    x = load_channels(a.path, ALT_MAX)
    split = int(x.sizes["time"] * (1 - VAL_FRAC))
    train_dl = make_loader(x.isel(time=slice(0, split)), T_IN, a.batch, shuffle=True) # split - 1 the last training hour
    val_dl = make_loader(x.isel(time=slice(split, None)), T_IN, a.batch, shuffle=False) # split is the first validation hour
    print(f"device {device} | volume {dict(x.sizes)} | train windows {len(train_dl.dataset)}, val windows {len(val_dl.dataset)}")

    #* Print class weights from the training period to see if there's imbalances
    tr = x.isel(time=slice(0, split))
    valid_tr = tr.sel(channel=N_CH) # use this so we ignore the nan bins from the proportion
    frac = torch.tensor([float((tr.sel(channel=3 + i) * valid_tr).sum() / valid_tr.sum()) for i in range(3)]) # fraction of each cloud property in the training set
    
    # Handle if a cloud property is rarer than others, 
    # so upweight positive bins in the loss: pos_w = sqrt(negatives / positives) per cloud property,
    # from valid training bins only. The sqrt softens the ratio; the clamp avoids dividing by zero.
    pos_w = ((1 - frac) / frac.clamp(min=1e-4)).sqrt().to(device)
    print("Cloud property balance for training:", {prop: round(f, 3) for prop, f in zip(CLOUD_PROPERTIES, frac.tolist())})

    #* Model training and optimizer setup
    model = ConvLSTM3d(N_CH, HIDDEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    history, best, base = [], float("inf"), None # tracking for training history, best validation loss and persistence scores
    epochs = tqdm(range(1, a.epochs + 1), desc="epochs", unit="ep", dynamic_ncols=True)

    #* Begin training loop
    for epoch in epochs:
        model.train()
        run, n = 0.0, 0
        batches = tqdm(train_dl, desc=f"train {epoch}/{a.epochs}", leave=False, dynamic_ncols=True)
        for w in batches:
            x_in, y, valid = (t.to(device) for t in split_batch(w))
            loss = weighted_bce(model(x_in), y, valid, pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run, n = run + loss.item(), n + 1
            batches.set_postfix(loss=f"{run / n:.4f}")

        val_loss, val_scores, b = evaluate(model, val_dl, pos_w, device, with_baseline=base is None) # run evaluation
        base = base or b # persistence does not depend on the model, so it is only computed on the first pass

        flag = ""
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "args": vars(a),
                        "hidden": HIDDEN, "t_in": T_IN, "alt_max": ALT_MAX}, ckpt_path)
            flag = "  (saved)" # set flag to indicate that the model was saved
        mean_csi = np.mean([val_scores[prop]["CSI"] for prop in CLOUD_PROPERTIES])

        # track one row per epoch, rewritten every epoch so a stopped run keeps its curves
        row = {"epoch": epoch, "train_loss": run / n, "val_loss": val_loss, "val_mean_CSI": mean_csi}
        row |= {f"val_{k}_{prop}": val_scores[prop][k] for prop in CLOUD_PROPERTIES for k in ("CSI", "F1")}

        # store history and save plots
        history.append(row)
        hist = pd.DataFrame(history)
        hist.to_csv(out / "history.csv", index=False)
        plot_curves(hist, base, out / "curves.png")

        epochs.set_postfix(train=f"{run / n:.4f}", val=f"{val_loss:.4f}", csi=f"{mean_csi:.3f}", best=f"{best:.4f}")
        tqdm.write(f"epoch {epoch:3d} | train loss {run / n:.4f} | val loss {val_loss:.4f} | val mean CSI {mean_csi:.3f}{flag}")

    epochs.close()

    #* Evaluation on the validation set
    # Load the best model and evaluate on the validation set
    # create plotting against the persistence baseline
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])

    print("Running evaluation on the validation set...")
    _, final, _ = evaluate(model, val_dl, pos_w, device, desc="final")

    # forecasts over time: k one-hour-ahead forecasts in a row, on every validation stretch
    k = min(K, len(val_dl.dataset))
    val = x.isel(time=slice(split, None))
    w = torch.from_numpy(val.values) # (hours, channel, alt, lat, lon)
    plot_dir = out / "forecast_height_lon"
    plot_dir.mkdir(exist_ok=True)

    model.eval()
    starts = range(0, len(w) - T_IN - k + 1, k) # next stretch forecasts the next k hours
    for start in tqdm(starts, desc="forecast plots"):
        seq = w[start:start + T_IN + k] # T_IN input hours + k forecasted hours
        x_in = torch.stack([seq[j:j + T_IN, :N_CH] for j in range(k)]) # window j forecasts hour T_IN + j
        with torch.no_grad():
            prob = torch.sigmoid(model(x_in.to(device))).cpu().numpy()
        times = pd.to_datetime(val.time.values[start:start + T_IN + k])
        plot_height_lon_sequence(seq[:, 3:N_CH].numpy(), prob, seq[:, N_CH].numpy(), x.alt.values, x.longitude.values,
                                 times, plot_dir / f"{times[T_IN]:%Y-%m-%d_%H}h.png")

    plot_vs_persistence(final, base, out / "vs_persistence.png")

    print(f"\nsaved checkpoint, history and plots to {out}/")

if __name__ == "__main__":
    main()