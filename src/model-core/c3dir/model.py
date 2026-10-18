"""C3DIR models: predict, save and load (training is c3dir.training in model-train).

models are versioned under MODELS_DIR/c3dir/<version>/:
  model.pt       weights, model config, input normalization, label occurrence
  config.json    model and training settings, data split, git commit, time
  history.csv    train / val loss over training
  metrics.csv    test scores (model-train's evaluation); figures/ holds the plots

  model = C3DIR(ModelConfig()); fit(model, train, val, TrainConfig()); model.save()  -> "v001", ...
  C3DIR.load("latest")
"""
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from c3dir.constants import BANDS, TARGETS
from c3dir.network import C3DIRNet, ModelConfig, decode
from utilities.paths import MODELS_DIR, ROOT

MODEL_ROOT = MODELS_DIR / "c3dir"
MASK_COLUMNS = [TARGETS.index(m) for m in ["hydrometeor_mask", "ice_mask", "liquid_mask", "rain_mask"]]
WATER_COLUMNS = [TARGETS.index(w) for w in ["iwc", "lwc", "rwc"]]
RADIUS_COLUMNS = [TARGETS.index(r) for r in ["ice_radius", "liquid_radius"]]


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


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

