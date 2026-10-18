# Data Processing

Download and preprocessing of the raw data into the files that `model-train` and `model-deploy`
read. Raw downloads go to `data/raw/<source>/`, training-ready data to `data/processed/<model>/`.

| Model | Folder | Status |
|---|---|---|
| C3DIR | [`c3dir/`](c3dir/README.md) | EarthCARE + GOES-19 download, labels, collocation, samples |
| GOES / C3DIR nowcasting | | not set up yet (GOES-19 ABI, ERA5) |

## Container

The `Dockerfile` builds an image (no torch) that collects a time window, processes everything
collected, and pushes the results to the shared bucket. Settings are environment variables with
the same names as the script settings (`START`, `END`, `PUSH`, ...). From the repo root:

```bash
docker compose build data-processing
START=2025-08-01T00:00 END=2025-08-02T00:00 docker compose run --rm data-processing
```

`.env` needs `HEIMDALL_BUCKET` and `GOOGLE_CLOUD_PROJECT`. Locally, `docker-compose.yml` mounts
`data/`, `artifacts/` and the GCS and ESA MAAP logins (read-only, never in the image).

Each run only needs its own window on disk: samples stand alone, and model-train makes the
train / val / test split from everything in the bucket. After a run has pushed, its local files in
`data/raw`, `data/processed` and `data/cache` can be deleted.
