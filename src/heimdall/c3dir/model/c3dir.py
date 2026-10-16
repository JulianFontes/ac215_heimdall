"""train, evaluate, save and load C3DIR models.

models are versioned under artifacts/models/c3dir/<version>/:
  model.pt       weights, model config, input normalization, label occurrence
  config.json    model and training settings, data split, git commit, time
  history.csv    train / val loss over training
  metrics.csv    test scores (see evaluation.py); figures/ holds the plots

  C3DIR(ModelConfig()).fit(train, val, TrainConfig())  ->  .save()  -> "v001", "v002", ...
  C3DIR.load("latest")
"""
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from heimdall.c3dir.constants import BANDS, TARGETS
from heimdall.c3dir.data_scraping.dataset import C3DIRDataset, collate
from heimdall.c3dir.model.network import LEVELS, C3DIRNet, ModelConfig, decode, mask_logits, scale_radius, scale_water
from heimdall.paths import MODELS_DIR, ROOT

MODEL_ROOT = MODELS_DIR / "c3dir"
MASK_COLUMNS = [TARGETS.index(m) for m in ["hydrometeor_mask", "ice_mask", "liquid_mask", "rain_mask"]]
WATER_COLUMNS = [TARGETS.index(w) for w in ["iwc", "lwc", "rwc"]]
RADIUS_COLUMNS = [TARGETS.index(r) for r in ["ice_radius", "liquid_radius"]]


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


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


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


class C3DIR:
    def __init__(self, config=None, device=None):
        self.config = config or ModelConfig()
        self.device = device or default_device()
        self.net = C3DIRNet(self.config).to(self.device)
        self.mean = self.std = self.occurrence = None
        self.history = pd.DataFrame()
        self.train_config = None
        self.data = {}
        self.version = None

    # ---- training
    def loss(self, raw, labels, level, cfg):
        """paper eq. 1 averaged over labeled voxels; raw (V, 8), labels (V, 9) physical units."""
        target = labels[:, MASK_COLUMNS]
        occurrence = torch.as_tensor(self.occurrence, device=raw.device)[:, level].T
        weight = torch.where(target > 0, 1.0, (1 - cfg.lam) + cfg.lam * occurrence.clamp_min(0.1))
        masks = (weight * F.binary_cross_entropy_with_logits(mask_logits(raw), target, reduction="none")).sum(1)
        water = ((raw[:, 3:6] - scale_water(labels[:, WATER_COLUMNS])) ** 2).sum(1)
        radius_target = labels[:, RADIUS_COLUMNS]
        has_radius = torch.isfinite(radius_target)
        radius = (((raw[:, 6:8] - scale_radius(radius_target.nan_to_num())) ** 2) * has_radius).sum(1)
        return (cfg.gamma_mask * masks + water + radius).mean()

    def _gather(self, batch):
        """network outputs and labels at the labeled voxels of a batch."""
        x = torch.as_tensor(batch["x"], device=self.device)
        index = torch.as_tensor(batch["index"], device=self.device)
        mask = torch.as_tensor(batch["mask"], device=self.device)
        b = torch.arange(len(x), device=self.device)[:, None].expand_as(mask)[mask]
        level, row, col = index[mask].T
        raw = self.net(x)[b, :, level, row, col]  # (V, 8)
        labels = torch.as_tensor(batch["y"], device=self.device)[mask]
        return raw, labels, level

    def fit(self, train, val, cfg=None):
        """train on C3DIRDataset train, keeping the weights with the lowest val loss."""
        cfg = self.train_config = cfg or TrainConfig()
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.mean, self.std = train.mean.ravel(), train.std.ravel()
        self.occurrence = label_occurrence(train)
        self.data = {"train": [p.name for p in train.files], "val": [p.name for p in val.files]}
        loader = DataLoader(train, batch_size=cfg.batch_size, collate_fn=self._to_tensors,
                            sampler=RandomSampler(train, replacement=True, num_samples=cfg.steps * cfg.batch_size))
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps, eta_min=cfg.min_lr)
        rows, running, best, best_state, start = [], [], np.inf, None, time.time()
        self.net.train()
        for step, batch in enumerate(loader, 1):
            raw, labels, level = self._gather(flip(batch, rng))
            loss = self.loss(raw, labels, level, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            optimizer.step()
            schedule.step()
            running.append(loss.item())
            if step % cfg.eval_every == 0 or step == cfg.steps:
                val_loss = self.val_loss(val, cfg)
                rows.append({"step": step, "train_loss": np.mean(running), "val_loss": val_loss,
                             "lr": schedule.get_last_lr()[0], "minutes": (time.time() - start) / 60})
                running = []
                if val_loss < best:
                    best, best_state = val_loss, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                print(f"step {step}/{cfg.steps}  train {rows[-1]['train_loss']:.4f}  val {val_loss:.4f}  "
                      f"[{rows[-1]['minutes']:.1f} min]", flush=True)
                self.net.train()
        self.net.load_state_dict(best_state)
        self.history = pd.DataFrame(rows)
        return self

    @staticmethod
    def _to_tensors(items):
        batch = collate(items)
        return {k: torch.as_tensor(v) for k, v in batch.items()}

    @torch.no_grad()
    def val_loss(self, dataset, cfg):
        self.net.eval()
        total, count = 0.0, 0
        for batch in DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=self._to_tensors):
            raw, labels, level = self._gather(batch)
            if len(raw):
                total += self.loss(raw, labels, level, cfg).item() * len(raw)
                count += len(raw)
        return total / max(count, 1)

    # ---- prediction
    @torch.no_grad()
    def predict_raw(self, x):
        """raw outputs (B, 8, 80, H, W) for normalized patches x (B, 16, H, W)."""
        self.net.eval()
        return self.net(torch.as_tensor(x, device=self.device))

    def predict_voxels(self, sample, voxels=None, window=128, margin=32):
        """raw outputs (V, 8) at voxels (level, row, col) of a sample's goes crop (default: its
        matched voxels), predicting in overlapping windows so each voxel has context on all
        sides."""
        image = (sample["goes"].values - self.mean[:, None, None]) / self.std[:, None, None]
        image = np.nan_to_num(image.astype(np.float32))
        _, height, width = image.shape
        image = np.pad(image, ((0, 0), (0, max(0, window - height)), (0, max(0, window - width))))
        if voxels is None:
            voxels = [sample[f"voxel_{k}"].values for k in ["level", "row", "col"]]
        level, row, col = (np.asarray(v) for v in voxels)
        raw = np.full((len(level), 8), np.nan, np.float32)
        step = window - 2 * margin
        starts = lambda n: sorted({*range(0, max(n - window, 0), step), max(n - window, 0)})  # noqa: E731
        for r0 in starts(image.shape[1]):
            for c0 in starts(image.shape[2]):
                inside = ((row >= r0 + (margin if r0 else 0)) & (row < r0 + window) &
                          (col >= c0 + (margin if c0 else 0)) & (col < c0 + window) & np.isnan(raw[:, 0]))
                if not inside.any():
                    continue
                out = self.predict_raw(image[None, :, r0:r0 + window, c0:c0 + window])
                raw[inside] = out[0, :, level[inside], row[inside] - r0, col[inside] - c0].T.cpu().numpy()
        return raw

    # ---- saving
    def save(self, version=None, root=MODEL_ROOT):
        """save to root/<version>; version defaults to the next v001, v002, ..."""
        version = self.version = version or f"v{len(self.versions(root)) + 1:03d}"
        out = Path(root) / version
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "config": self.config.as_dict(), "mean": self.mean,
                    "std": self.std, "occurrence": self.occurrence}, out / "model.pt")
        self.history.to_csv(out / "history.csv", index=False)
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip()
        except OSError:
            commit = None
        info = {"version": version, "created": pd.Timestamp.now().isoformat(timespec="seconds"),
                "git_commit": commit, "device": self.device, "bands": BANDS, "model": self.config.as_dict(),
                "training": asdict(self.train_config) if self.train_config else None,
                "parameters_m": round(sum(p.numel() for p in self.net.parameters()) / 1e6, 3), "data": self.data}
        (out / "config.json").write_text(json.dumps(info, indent=2))
        return out

    @staticmethod
    def versions(root=MODEL_ROOT):
        return sorted(p.name for p in Path(root).glob("v*") if (p / "model.pt").exists())

    @classmethod
    def load(cls, version="latest", root=MODEL_ROOT, device=None):
        if version == "latest":
            version = cls.versions(root)[-1]
        out = Path(root) / version
        saved = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
        model = cls(ModelConfig(**saved["config"]), device)
        model.net.load_state_dict(saved["state"])
        model.mean, model.std, model.occurrence = saved["mean"], saved["std"], saved["occurrence"]
        if (out / "history.csv").exists():
            model.history = pd.read_csv(out / "history.csv")
        model.version = version
        return model

    def directory(self, root=MODEL_ROOT):
        return Path(root) / self.version


def decode_numpy(raw):
    """numpy version of model.decode: probability (V, 4), water (V, 3), radius (V, 2)."""
    return tuple(a.numpy() for a in decode(torch.as_tensor(raw)))

