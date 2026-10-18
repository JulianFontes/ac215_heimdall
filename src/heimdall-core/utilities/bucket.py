"""the shared GCS bucket: one place where every service reads and writes data and models.

the bucket has the same layout as paths.py, so a bucket folder and its local folder always match:

  gs://<bucket>/raw/<source>/              <->  RAW_DIR / <source>
  gs://<bucket>/processed/<model>/         <->  PROCESSED_DIR / <model>
  gs://<bucket>/predictions/<model>/       <->  PREDICTIONS_DIR / <model>
  gs://<bucket>/models/<model>/<version>/  <->  MODELS_DIR / <model> / <version>
  gs://<bucket>/figures/                   <->  FIGURES_DIR
the cache stays local. jobs pull what they need to local disk at the start and push their outputs at
the end, so the code itself only ever reads local files.

HEIMDALL_BUCKET names the bucket. credentials come from Application Default Credentials: locally
`gcloud auth application-default login`, on GCP the service account of the job. needs the gcs extra
(heimdall-core[gcs]).

  pull("models/c3dir/v002")      bucket -> local, only files that are new or changed
  push("processed/c3dir")        local -> bucket, only files that are new or changed
  folders("models/c3dir")        what is in the bucket, e.g. ["v001", "v002"]
neither direction deletes files.

use in each service (any model, e.g. c3dir or goes_nowcasting; needs heimdall-core[gcs]):

  from utilities.bucket import pull, push
  data-processing   push("raw/<source>") after downloading and push("processed/<model>") after
                    processing; pull("processed/<model>") first when new data is added to existing
                    data (c3dir.collect and c3dir.process do this with PUSH = True)
  model-train       pull("processed/<model>") before training, push("models/<model>/<version>")
                    after; never push into an existing version
  model-deploy      pull("models/<model>/<version>") before loading the model,
                    push("predictions/<model>") after predicting
"""
import base64
import hashlib
import os
from functools import cache

from google.cloud import storage

from utilities.paths import FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DIR, RAW_DIR

FOLDERS = {"raw": RAW_DIR, "processed": PROCESSED_DIR, "predictions": PREDICTIONS_DIR,
           "models": MODELS_DIR, "figures": FIGURES_DIR}
SKIP = {".DS_Store"}  # local files never pushed


@cache
def bucket():
    name = os.environ.get("HEIMDALL_BUCKET")
    if not name:
        raise RuntimeError("HEIMDALL_BUCKET is not set; add it to .env (see .env.example)")
    return storage.Client().bucket(name)


def local_path(prefix):
    """the local folder for a bucket folder, e.g. models/c3dir/v002 -> artifacts/models/c3dir/v002."""
    top, _, rest = prefix.strip("/").partition("/")
    if top not in FOLDERS:
        raise ValueError(f"{prefix!r} must start with one of {list(FOLDERS)}")
    return FOLDERS[top] / rest


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def _same(path, blob):
    """true if the local file matches the bucket file (size, then md5 when the bucket has one)."""
    if not path.is_file() or path.stat().st_size != blob.size:
        return False
    return blob.md5_hash is None or _md5(path) == blob.md5_hash


def files(prefix):
    """the files in a bucket folder, as {path inside the folder: blob}."""
    prefix = prefix.strip("/") + "/"
    return {blob.name.removeprefix(prefix): blob for blob in bucket().list_blobs(prefix=prefix)
            if not blob.name.endswith("/")}


def folders(prefix=""):
    """the folders directly inside a bucket folder, e.g. folders("models/c3dir") -> ["v001", "v002"]."""
    prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
    blobs = bucket().list_blobs(prefix=prefix, delimiter="/")
    list(blobs)  # the folder names are only filled in once the listing is read
    return sorted(p.removeprefix(prefix).rstrip("/") for p in blobs.prefixes)


def pull(prefix):
    """copy a bucket folder to its local folder; returns the local folder."""
    target = local_path(prefix)
    remote = files(prefix)
    if not remote:
        raise FileNotFoundError(f"gs://{bucket().name}/{prefix.strip('/')}/ is empty or missing")
    copied = 0
    for name, blob in remote.items():
        path = target / name
        if _same(path, blob):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(path)
        copied += 1
    print(f"pulled {copied} of {len(remote)} files: gs://{bucket().name}/{prefix.strip('/')}/ -> {target}")
    return target


def push(prefix):
    """copy a local folder to its bucket folder."""
    source = local_path(prefix)
    if not source.is_dir():
        raise FileNotFoundError(source)
    remote = files(prefix)
    local = sorted(p for p in source.rglob("*") if p.is_file() and p.name not in SKIP)
    copied = 0
    for path in local:
        name = path.relative_to(source).as_posix()
        if name in remote and _same(path, remote[name]):
            continue
        bucket().blob(f"{prefix.strip('/')}/{name}").upload_from_filename(path)
        copied += 1
    print(f"pushed {copied} of {len(local)} files: {source} -> gs://{bucket().name}/{prefix.strip('/')}/")
