"""train a Forecaster and score it against persistence (the last input frame repeated).

  loss       binary cross-entropy against C3DIR's probabilities (soft targets), averaged over lead times
  scores     per lead time: CSI of cloudy voxels (P >= 0.5), mean absolute error of P, and the skill
             over persistence, (score - persistence) / (perfect - persistence): 0 = no better
             than persistence, 1 = perfect
"""
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    eval_every: int = 50
    seed: int = 0

    def as_dict(self):
        return asdict(self)


def _loss(model, x, y):
    forecast = model.net(x, y.shape[1])
    return F.binary_cross_entropy(forecast.clamp(1e-6, 1 - 1e-6), y)


def fit(model, train, val, cfg=None):
    """train model.net on train, keeping the weights with the lowest loss on val (or on train
    when there is no val set). sets model.history."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    sampler = RandomSampler(train, replacement=True, num_samples=cfg.steps * cfg.batch_size,
                            generator=torch.Generator().manual_seed(cfg.seed))
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps)
    check = val if len(val) else train
    best, best_state, rows, running, start = np.inf, None, [], [], time.time()
    for step, (x, y) in enumerate(DataLoader(train, batch_size=cfg.batch_size, sampler=sampler), 1):
        model.net.train()
        x, y = x.to(model.device), y.to(model.device)
        if torch.rand(()) < 0.5:  # random east-west flip, the same for inputs and targets
            x, y = x.flip(-1), y.flip(-1)
        loss = _loss(model, x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        running.append(loss.item())
        if step % cfg.eval_every == 0 or step == cfg.steps:
            check_loss = mean_loss(model, check, cfg.batch_size)
            rows.append({"step": step, "train_loss": np.mean(running), "val_loss": check_loss,
                         "minutes": (time.time() - start) / 60})
            running = []
            print(f"  step {step:>5}  train {rows[-1]['train_loss']:.4f}  val {check_loss:.4f}  "
                  f"{rows[-1]['minutes']:.1f} min", flush=True)
            if check_loss < best:
                best, best_state = check_loss, {k: v.detach().clone() for k, v in model.net.state_dict().items()}
    model.net.load_state_dict(best_state)
    model.history = pd.DataFrame(rows)
    return model


@torch.no_grad()
def mean_loss(model, dataset, batch_size=8):
    model.net.eval()
    total = sum(_loss(model, x.to(model.device), y.to(model.device)).item() * len(x)
                for x, y in DataLoader(dataset, batch_size=batch_size))
    return total / len(dataset)


def evaluate(model, dataset, cadence_minutes=10, batch_size=8):
    """scores by lead time for the model and persistence, and every forecast (for figures)."""
    sums = {m: np.zeros((dataset.output_steps, 5)) for m in ("convlstm", "persistence")}  # tp, fp, fn, |err|, n
    forecasts = []
    for x, y in DataLoader(dataset, batch_size=batch_size):
        x, y = x.numpy(), y.numpy()
        forecast = model.predict(x, dataset.output_steps)
        forecasts.append(forecast)
        for method, f in (("convlstm", forecast), ("persistence", np.repeat(x[:, -1:], y.shape[1], axis=1))):
            for lead in range(y.shape[1]):
                pf, pt = f[:, lead] >= 0.5, y[:, lead] >= 0.5
                sums[method][lead] += [(pf & pt).sum(), (pf & ~pt).sum(), (~pf & pt).sum(),
                                       np.abs(f[:, lead] - y[:, lead]).sum(), y[:, lead].size]
    rows = []
    for method, s in sums.items():
        for lead, (tp, fp, fn, err, n) in enumerate(s):
            rows.append({"method": method, "lead_minutes": (lead + 1) * cadence_minutes,
                         "csi": tp / max(tp + fp + fn, 1), "prob_mae": err / n})
    table = pd.DataFrame(rows)
    ref = table[table["method"] == "persistence"].set_index("lead_minutes")
    for score, perfect in (("csi", 1.0), ("prob_mae", 0.0)):
        base = table["lead_minutes"].map(ref[score])
        table[f"{score}_skill"] = (table[score] - base) / (perfect - base).where((perfect - base).abs() > 1e-12)
    return table, np.concatenate(forecasts)
