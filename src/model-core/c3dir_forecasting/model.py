"""save, load and run ConvLSTM forecasters of C3DIR fields.

versions live in MODELS_DIR/c3dir_forecasting/<version>/ (v001, v002, ...):
  model.pt       weights, NetConfig, FrameConfig, input / output steps
  config.json    the same settings plus training settings, data (sites, days), git commit, time
  history.csv    train / val loss; metrics.csv and figures/ are added by model-train's train.py

  model = Forecaster(NetConfig(...), FrameConfig(...), input_steps=3, output_steps=3)
  model.save()  ->  "v001", "v002", ...
  Forecaster.load("latest").predict(frames)      frames (B, input_steps, levels, size, size)
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from c3dir_forecasting.frames import FrameConfig
from c3dir_forecasting.network import ConvLSTMForecaster, NetConfig
from utilities.paths import MODELS_DIR, ROOT

MODEL_ROOT = MODELS_DIR / "c3dir_forecasting"


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Forecaster:
    def __init__(self, net_config=None, frame_config=None, input_steps=3, output_steps=3, device=None):
        self.frame_config = frame_config or FrameConfig()
        self.net_config = net_config or NetConfig(channels=self.frame_config.levels)
        self.input_steps, self.output_steps = input_steps, output_steps
        self.device = device or default_device()
        self.net = ConvLSTMForecaster(self.net_config).to(self.device)
        self.history = pd.DataFrame()
        self.info = {}  # training settings and data, written to config.json
        self.version = None

    @torch.no_grad()
    def predict(self, frames, output_steps=None):
        """forecast (B, output_steps, levels, size, size) from frames (B, input_steps, levels, size, size)."""
        self.net.eval()
        x = torch.as_tensor(np.asarray(frames, np.float32), device=self.device)
        return self.net(x, output_steps or self.output_steps).cpu().numpy()

    def save(self, version=None, root=MODEL_ROOT):
        """save to root/<version>; version defaults to the next v001, v002, ..."""
        version = self.version = version or f"v{len(self.versions(root)) + 1:03d}"
        out = Path(root) / version
        out.mkdir(parents=True, exist_ok=True)
        settings = {"net": self.net_config.as_dict(), "frames": self.frame_config.as_dict(),
                    "input_steps": self.input_steps, "output_steps": self.output_steps}
        torch.save({"state": self.net.state_dict(), **settings}, out / "model.pt")
        self.history.to_csv(out / "history.csv", index=False)
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip() or None
        except OSError:
            commit = None
        info = {"version": version, "created": pd.Timestamp.now().isoformat(timespec="seconds"),
                "git_commit": commit, "device": self.device, **settings,
                "parameters_m": round(sum(p.numel() for p in self.net.parameters()) / 1e6, 3), **self.info}
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
        model = cls(NetConfig(**saved["net"]), FrameConfig(**saved["frames"]), saved["input_steps"],
                    saved["output_steps"], device)
        model.net.load_state_dict(saved["state"])
        if (out / "history.csv").exists():
            model.history = pd.read_csv(out / "history.csv")
        model.version = version
        return model
