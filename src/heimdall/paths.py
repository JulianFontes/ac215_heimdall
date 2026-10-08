"""project folders, worked out from where this file lives so they work from any directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/heimdall/paths.py -> repo root
DATA_DIR = ROOT / "data"  # scraped satellite data
CONFIG_DIR = ROOT / "config"
ARTIFACTS_DIR = ROOT / "artifacts"  # generated outputs: models and figures
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
