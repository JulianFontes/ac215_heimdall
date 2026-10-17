"""where every service reads and writes, the data contract between services.

services only hand data to each other through these folders. locally they sit under data/ and
artifacts/ at the repo root (git-ignored); on GCP the same layout becomes prefixes in a bucket,
copied to local disk at the start of a job and back at the end. HEIMDALL_DATA_DIR and
HEIMDALL_ARTIFACTS_DIR move the roots, e.g. to mounted volumes in a container.

  RAW_DIR / <source>/          data-processing: downloads as they come (earthcare, goes, ...)
  PROCESSED_DIR / <model>/     data-processing: training-ready data for one model (c3dir, ...)
  CACHE_DIR /                  temporary downloads (e.g. full GOES scans), safe to delete
  PREDICTIONS_DIR / <model>/   model-deploy: predicted fields
  MODELS_DIR / <model>/<version>/   model-train: versioned models with metrics and figures
  FIGURES_DIR /                example figures from any service
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # src/heimdall-core/c3dir/paths.py -> repo root
DATA_DIR = Path(os.environ.get("HEIMDALL_DATA_DIR", ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("HEIMDALL_ARTIFACTS_DIR", ROOT / "artifacts"))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
