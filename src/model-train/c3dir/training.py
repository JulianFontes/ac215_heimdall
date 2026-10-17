"""training C3DIR (paper section 4.4): the loss (paper eq. 1-2), random flips, and the loop that
keeps the weights with the lowest validation loss. the model itself (predict, save, load) is
c3dir.model in model-core.

  model = C3DIR(ModelConfig())
  fit(model, C3DIRDataset("train"), C3DIRDataset("val"), TrainConfig())
  model.save()
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from c3dir.dataset import collate
from c3dir.model import MASK_COLUMNS, RADIUS_COLUMNS, WATER_COLUMNS
from c3dir.network import LEVELS, mask_logits, scale_radius, scale_water


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 16  # paper: 1024
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    gamma_mask: float = 0.25  # paper eq. 1
    lam: float = 0.25  # paper eq. 2
    eval_every: int = 200
    seed: int = 0


def label_occurrence(dataset):
    """fraction of labeled voxels at each level holding each species, (4, 80) (paper eq. 2)."""
    positive, total = np.zeros((4, LEVELS)), np.zeros(LEVELS)
    for path in dataset.files:
        with xr.open_dataset(path) as s:
            level, labels = s["voxel_level"].values, s["voxel_labels"].values
        np.add.at(total, level, 1)
        for k, col in enumerate(MASK_COLUMNS):
            np.add.at(positive[k], level, labels[:, col] == 1)
    return (positive / np.maximum(total, 1)).astype(np.float32)


def flip(batch, rng):
    """random horizontal and vertical flips of a batch (paper section 4.4)."""
    size = batch["x"].shape[-1]
    for axis, column in [(-1, 2), (-2, 1)]:
        if rng.random() < 0.5:
            batch["x"] = batch["x"].flip(axis)
            batch["index"][..., column] = size - 1 - batch["index"][..., column]
    return batch


def loss(model, raw, labels, level, cfg):
    """paper eq. 1 averaged over labeled voxels; raw (V, 8), labels (V, 9) physical units."""
    target = labels[:, MASK_COLUMNS]
    occurrence = torch.as_tensor(model.occurrence, device=raw.device)[:, level].T
    weight = torch.where(target > 0, 1.0, (1 - cfg.lam) + cfg.lam * occurrence.clamp_min(0.1))
    masks = (weight * F.binary_cross_entropy_with_logits(mask_logits(raw), target, reduction="none")).sum(1)
    water = ((raw[:, 3:6] - scale_water(labels[:, WATER_COLUMNS])) ** 2).sum(1)
    radius_target = labels[:, RADIUS_COLUMNS]
    has_radius = torch.isfinite(radius_target)
    radius = (((raw[:, 6:8] - scale_radius(radius_target.nan_to_num())) ** 2) * has_radius).sum(1)
    return (cfg.gamma_mask * masks + water + radius).mean()


def gather(model, batch):
    """network outputs and labels at the labeled voxels of a batch."""
    x = torch.as_tensor(batch["x"], device=model.device)
    index = torch.as_tensor(batch["index"], device=model.device)
    mask = torch.as_tensor(batch["mask"], device=model.device)
    b = torch.arange(len(x), device=model.device)[:, None].expand_as(mask)[mask]
    level, row, col = index[mask].T
    raw = model.net(x)[b, :, level, row, col]  # (V, 8)
    labels = torch.as_tensor(batch["y"], device=model.device)[mask]
    return raw, labels, level


def fit(model, train, val, cfg=None):
    """train on C3DIRDataset train, keeping the weights with the lowest val loss."""
    cfg = model.train_config = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model.mean, model.std = train.mean.ravel(), train.std.ravel()
    model.occurrence = label_occurrence(train)
    model.data = {"train": [p.name for p in train.files], "val": [p.name for p in val.files]}
    loader = DataLoader(train, batch_size=cfg.batch_size, collate_fn=to_tensors,
                        sampler=RandomSampler(train, replacement=True, num_samples=cfg.steps * cfg.batch_size))
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps, eta_min=cfg.min_lr)
    rows, running, best, best_state, start = [], [], np.inf, None, time.time()
    model.net.train()
    for step, batch in enumerate(loader, 1):
        raw, labels, level = gather(model, flip(batch, rng))
        batch_loss = loss(model, raw, labels, level, cfg)
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 5.0)
        optimizer.step()
        schedule.step()
        running.append(batch_loss.item())
        if step % cfg.eval_every == 0 or step == cfg.steps:
            current = val_loss(model, val, cfg)
            rows.append({"step": step, "train_loss": np.mean(running), "val_loss": current,
                         "lr": schedule.get_last_lr()[0], "minutes": (time.time() - start) / 60})
            running = []
            if current < best:
                best, best_state = current, {k: v.detach().clone() for k, v in model.net.state_dict().items()}
            print(f"step {step}/{cfg.steps}  train {rows[-1]['train_loss']:.4f}  val {current:.4f}  "
                  f"[{rows[-1]['minutes']:.1f} min]", flush=True)
            model.net.train()
    model.net.load_state_dict(best_state)
    model.history = pd.DataFrame(rows)
    return model


def to_tensors(items):
    batch = collate(items)
    return {k: torch.as_tensor(v) for k, v in batch.items()}


@torch.no_grad()
def val_loss(model, dataset, cfg):
    model.net.eval()
    total, count = 0.0, 0
    for batch in DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=to_tensors):
        raw, labels, level = gather(model, batch)
        if len(raw):
            total += loss(model, raw, labels, level, cfg).item() * len(raw)
            count += len(raw)
    return total / max(count, 1)
