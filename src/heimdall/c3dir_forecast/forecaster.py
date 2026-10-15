"""train, save and load ConvLSTM forecasters of C3DIR fields.

versions live in artifacts/models/c3dir_forecast/<version>/ (v001, v002, ...):
  model.pt      weights and ForecastConfig
  config.json   settings, the C3DIR version and site the frames came from, data split, git commit
  history.csv   train / val loss; metrics.csv and figures/ are added by the training script

  Forecaster(ForecastConfig(...)).fit(train, val, TrainConfig())  ->  .save()
  Forecaster.load("latest").predict(x)
"""
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from heimdall.c3dir.model.c3dir import default_device
from heimdall.c3dir_forecast.network import ConvLSTMForecaster, ForecastConfig
from heimdall.paths import MODELS_DIR, ROOT

MODEL_ROOT = MODELS_DIR / "c3dir_forecast"


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 8
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    water_weight: float = 1.0  # weight of the water MSE next to the probability cross-entropy
    eval_every: int = 200
    seed: int = 0


class Forecaster:
    def __init__(self, config=None, device=None):
        self.config = config or ForecastConfig()
        self.device = device or default_device()
        self.net = ConvLSTMForecaster(self.config).to(self.device)
        self.valid = None  # (H, W) cells inside the region, from the data
        self.history = pd.DataFrame()
        self.train_config = None
        self.info = {}  # where the frames came from: c3dir version, site, days, split
        self.version = None

    def loss(self, forecast, target, cfg):
        """cross-entropy on the probability channels (soft C3DIR targets) + mse on scaled water,
        over the cells inside the region."""
        n = self.config.n_prob
        valid = torch.as_tensor(self.valid, device=forecast.device)
        prob = F.binary_cross_entropy(forecast[:, :, :n].clamp(1e-6, 1 - 1e-6), target[:, :, :n], reduction="none")
        water = (forecast[:, :, n:] - target[:, :, n:]) ** 2
        mean = lambda a: a.mean(dim=(0, 1, 2))[valid].mean()  # noqa: E731
        return mean(prob) + cfg.water_weight * mean(water)

    def fit(self, train, val, cfg=None, valid=None):
        cfg = self.train_config = cfg or TrainConfig()
        torch.manual_seed(cfg.seed)
        self.valid = np.ones(train.array.shape[-2:], bool) if valid is None else valid
        loader = DataLoader(train, batch_size=cfg.batch_size, drop_last=False,
                            sampler=RandomSampler(train, replacement=True, num_samples=cfg.steps * cfg.batch_size))
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps, eta_min=cfg.min_lr)
        best, best_state, rows, running, start = np.inf, None, [], [], time.time()
        for step, (x, y) in enumerate(loader, 1):
            x, y = self.flip(x.to(self.device), y.to(self.device))
            self.net.train()
            loss = self.loss(self.net(x, y.shape[1]), y, cfg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            running.append(loss.item())
            if step % cfg.eval_every == 0 or step == cfg.steps:
                val_loss = self.val_loss(val, cfg)
                rows.append({"step": step, "train_loss": np.mean(running), "val_loss": val_loss,
                             "lr": schedule.get_last_lr()[0], "minutes": (time.time() - start) / 60})
                running = []
                print(f"step {step:>6}  train {rows[-1]['train_loss']:.4f}  val {val_loss:.4f}  "
                      f"{rows[-1]['minutes']:.1f} min")
                if val_loss < best:
                    best = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.history = pd.DataFrame(rows)
        return self

    @staticmethod
    def flip(x, y):
        """the same random lat / lon flips on inputs and targets."""
        for dim in (-1, -2):
            if torch.rand(()) < 0.5:
                x, y = x.flip(dim), y.flip(dim)
        return x, y

    @torch.no_grad()
    def val_loss(self, dataset, cfg):
        if not len(dataset):
            return np.nan
        self.net.eval()
        total = 0.0
        for x, y in DataLoader(dataset, batch_size=cfg.batch_size):
            total += self.loss(self.net(x.to(self.device), y.shape[1]), y.to(self.device), cfg).item() * len(x)
        return total / len(dataset)

    @torch.no_grad()
    def predict(self, x, output_steps=None, batch_size=8):
        """forecast frames (N, output_steps, C, H, W) for previous frames x (N, input_steps, C, H, W).
        output_steps beyond the trained number roll the model forward on its own forecasts."""
        self.net.eval()
        x = torch.as_tensor(np.asarray(x))
        out = [self.net(x[i:i + batch_size].to(self.device), output_steps).cpu()
               for i in range(0, len(x), batch_size)]
        return torch.cat(out).numpy()

    # ---- saving
    def save(self, version=None, root=MODEL_ROOT):
        version = self.version = version or f"v{len(self.versions(root)) + 1:03d}"
        out = Path(root) / version
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "config": self.config.as_dict(), "valid": self.valid},
                   out / "model.pt")
        self.history.to_csv(out / "history.csv", index=False)
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip()
        except OSError:
            commit = None
        info = {"version": version, "created": pd.Timestamp.now().isoformat(timespec="seconds"),
                "git_commit": commit, "device": self.device, "model": self.config.as_dict(),
                "training": asdict(self.train_config) if self.train_config else None,
                "parameters_m": round(sum(p.numel() for p in self.net.parameters()) / 1e6, 3), "data": self.info}
        (out / "config.json").write_text(json.dumps(info, indent=2, default=str))
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
        model = cls(ForecastConfig(**saved["config"]), device)
        model.net.load_state_dict(saved["state"])
        model.valid = saved["valid"]
        if (out / "history.csv").exists():
            model.history = pd.read_csv(out / "history.csv")
        model.info = json.loads((out / "config.json").read_text()).get("data", {})
        model.version = version
        return model
