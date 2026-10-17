"""copy folders between the shared bucket and local disk (see bucket.py for the layout).

edit the settings and run from the repo root:
  uv run --project src/heimdall-core --extra gcs --env-file .env python -m c3dir.sync
"""
from c3dir.bucket import folders, pull, push

# settings: bucket folders, named like the local folders (raw/..., processed/..., models/..., predictions/...)
SHOW = ["models/c3dir", "processed"]  # print what is in these bucket folders
PULL = ["models/c3dir/v002"]  # bucket -> local
PUSH = []  # local -> bucket, e.g. "processed/c3dir", "models/c3dir/v003"

for prefix in SHOW:
    print(f"{prefix}/: {', '.join(folders(prefix)) or '(empty)'}")
for prefix in PULL:
    pull(prefix)
for prefix in PUSH:
    push(prefix)
