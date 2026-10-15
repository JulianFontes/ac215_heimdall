"""Train the ConvLSTM on GOES-19 MCMIP files and compare it against persistence.

    python -m goes_nowcast.scripts.train -p data/goes19 --epochs 20 --out runs/baseline
"""
import argparse
import csv
import json
import os

import torch
from tqdm.auto import tqdm

from ..data import TARGET_IDX, prepare_data, split_batch
from ..evaluation import evaluate, masked_mse, summary
from ..model import ConvLSTM
from ..plots import plot_curves, plot_forecast, plot_vs_persistence

CLIP = 1.0 # gradient-norm clipping
SEED = 0

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--path", required=True, help="MCMIP .nc file, folder of .nc files, or glob pattern")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    #* Data Loading + Conditioning
    # chronological split, normalization from the training frames only
    data = prepare_data(args.path, args.batch)
    stats = (torch.tensor(data.mean[TARGET_IDX], device=device), torch.tensor(data.std[TARGET_IDX], device=device))

    #* Model definition and training loop
    model = ConvLSTM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history, best = [], float("inf")
    ckpt = os.path.join(args.out, "convlstm_best.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for w in tqdm(data.train, desc=f"epoch {epoch}/{args.epochs}", leave=False, dynamic_ncols=True):
            xb, y, valid = (a.to(device) for a in split_batch(w))
            loss = masked_mse(model(xb), y, valid)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total, n = total + loss.item(), n + 1
        val_loss, val_r, base_r = evaluate(model, data.val, stats, device, with_baseline=True)
        row = {"epoch": epoch, "train_loss": total / n, "val_loss": val_loss}
        row.update({f"val_{k}": v for k, v in summary(val_r).items()})
        row.update({f"base_{k}": v for k, v in summary(base_r).items()})
        history.append(row)
        with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            wr.writeheader()
            wr.writerows(history)
        saved = val_loss < best
        if saved:
            best = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "mean": torch.tensor(data.mean), "std": torch.tensor(data.std)}, ckpt)
        print(f"epoch {epoch:3d} | train {row['train_loss']:.5f} | val {val_loss:.5f}" + (" (saved)" if saved else ""))
        plot_curves(history, data.units, os.path.join(args.out, "curves.png")) # redrawn every epoch, so it can be watched during the run

    #* Model Evaluation
    # final evaluation of the best checkpoint, on the test split (validation if there is no test window)
    print("Running evaluation on the validation set...")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    loader, name = (data.test, "test") if data.test is not None else (data.val, "validation")
    _, r, base = evaluate(model, loader, stats, device, with_baseline=True, desc=name)
    print(f"\nbest checkpoint, {name} split")

    with open(os.path.join(args.out, f"{name}_scores.json"), "w") as f:
        json.dump({"convlstm": r, "persistence": base}, f, indent=2)
    plot_vs_persistence(r, base, data.units, os.path.join(args.out, "vs_persistence.png"))
    plot_forecast(model, loader, stats, data.units, device, os.path.join(args.out, "forecasts"))
    print(f"outputs written to {args.out}")

if __name__ == "__main__":
    main()
