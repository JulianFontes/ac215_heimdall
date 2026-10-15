"""Loss, RMSE / MAE scores, validation pass and score printing."""
import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm.auto import tqdm

from .config import STEP_MIN, TARGET_BANDS, T_OUT
from .data import TARGET_IDX, split_batch

def masked_mse(pred, y, valid):
    return (((pred - y) ** 2) * valid).sum() / valid.expand_as(y).sum().clamp(min=1)

def persistence(x):
    """Baseline: every forecast frame is the last observed frame."""
    return x[:, -1, TARGET_IDX].unsqueeze(1).expand(-1, T_OUT, -1, -1, -1)

def make_scores(device):
    scores = {f"t{k}_C{b:02d}": nn.ModuleDict({"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()})
              for k in range(T_OUT) for b in TARGET_BANDS}
    return nn.ModuleDict(scores).to(device)

def update_scores(scores, pred, y, valid, std):
    """pred, y: normalized (B, T_OUT, n_out, H, W). RMSE / MAE are computed in physical units (K or reflectance)."""
    ok = valid[:, :, 0].bool() # (B, T_OUT, H, W)
    for k in range(T_OUT):
        for i, b in enumerate(TARGET_BANDS):
            p, t = pred[:, k, i][ok[:, k]] * std[i], y[:, k, i][ok[:, k]] * std[i]
            for m in scores[f"t{k}_C{b:02d}"].values():
                m.update(p, t)

def report(scores):
    out = {name: {k: m.compute().item() for k, m in mods.items()} for name, mods in scores.items()}
    for mods in scores.values():
        for m in mods.values():
            m.reset()
    return out

@torch.no_grad()
def evaluate(model, loader, stats, device, with_baseline=False, desc="val"):
    """stats = (mean, std) of the target bands. Returns loss, scores of the model and (optionally) of persistence."""
    model.eval()
    _, std = stats
    scores, base = make_scores(device), make_scores(device)
    total, n = 0.0, 0
    for w in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        x, y, valid = (a.to(device) for a in split_batch(w))
        pred = model(x)
        total, n = total + masked_mse(pred, y, valid).item(), n + 1
        update_scores(scores, pred, y, valid, std)
        if with_baseline:
            update_scores(base, persistence(x), y, valid, std)
    return total / n, report(scores), (report(base) if with_baseline else None)

def summary(r):
    """Scores averaged over the lead times: {"RMSE_C13": ..., "MAE_C13": ..., ...}."""
    return {f"{s}_C{b:02d}": sum(r[f"t{k}_C{b:02d}"][s] for k in range(T_OUT)) / T_OUT for b in TARGET_BANDS for s in ("RMSE", "MAE")}

def print_scores(r, title):
    print(f"--- {title} ---")
    print("lead(min) " + " ".join(f"RMSE_C{b:02d}" for b in TARGET_BANDS))
    for k in range(T_OUT):
        print(f"{(k + 1) * STEP_MIN:>9d} " + " ".join(f"{r[f't{k}_C{b:02d}']['RMSE']:9.3f}" for b in TARGET_BANDS))
    s = summary(r)
    print("mean RMSE: " + ", ".join(f"C{b:02d}={s[f'RMSE_C{b:02d}']:.3f}" for b in TARGET_BANDS))
