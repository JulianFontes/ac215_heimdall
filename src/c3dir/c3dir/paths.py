"""project folders, worked out from where this file lives so they work from any directory.

data/ and artifacts/ sit at the repo root and are shared by every service. HEIMDALL_DATA_DIR and
HEIMDALL_ARTIFACTS_DIR override them, e.g. to point a container at mounted volumes.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # src/c3dir/c3dir/paths.py -> repo root
DATA_DIR = Path(os.environ.get("HEIMDALL_DATA_DIR", ROOT / "data"))  # scraped satellite data
ARTIFACTS_DIR = Path(os.environ.get("HEIMDALL_ARTIFACTS_DIR", ROOT / "artifacts"))  # models and figures
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
