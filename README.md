## Running Heimdall

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

All services are defined in [`docker-compose.yml`](docker-compose.yml). Run every command below from this (repo root) folder.

### Setup (once)
Copy the environment template and fill in any values you need (`.env` is git-ignored; see the comments in `.env.example`):
```bash
cp .env.example .env
```

### Build, start, stop
Everything goes through [`heimdall_docker.sh`](heimdall_docker.sh), a thin wrapper around `docker compose`.
Leave out the service name to act on all services in `docker-compose.yml`.
```bash
./heimdall_docker.sh build [service...]   # 1. build images (again after code or .env changes; add --no-cache for a clean rebuild)
./heimdall_docker.sh up    [service...]   # 2. start containers in the background
./heimdall_docker.sh down                 # 3. stop and remove containers (images are kept)
```

### Services
| Service | URL | Details |
|---|---|---|
| `frontend` | http://localhost:8080 | [frontend README](src/heimdall-frontend/README.md) |

Each service's README covers its own data, configuration and local development setup.

## C3DIR: 3D clouds from GOES-19

C3DIR predicts 3D cloud structure from GOES-19 imagery, trained on EarthCARE cloud profiles
(following White et al., 2026). Its code is the `c3dir` package in `data-processing`,
`model-core`, `model-train` and `model-deploy`, with torch-free shared code in `heimdall-core`. Run each step from this (repo root) folder:

```bash
uv run --project src/data-processing --env-file .env python -m c3dir.collect   # download EarthCARE frames and GOES-19 strips
uv run --project src/data-processing --env-file .env python -m c3dir.process   # labels, collocation, training samples
uv run --project src/model-train python -m c3dir.train         # train and evaluate a new version
uv run --project src/model-deploy python -m c3dir.predict      # 3D clouds on any GOES-19 scan
```

Edit the settings at the top of each file first. Data goes to `data/` and models and figures to
`artifacts/` (both git-ignored); `collect` and `process` also push their new files to the shared
bucket (`PUSH` setting, see below). See [`docs/c3dir.md`](docs/c3dir.md) for how the pipeline works,
data formats and assumptions.

## Shared data and models (GCS bucket)

Data and models are stored in one Google Cloud Storage bucket with the same layout as the local
folders (`raw/`, `processed/`, `predictions/`, `models/<model>/<version>/`, `figures/`). Set up once:

1. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install), then run `gcloud auth application-default login`
   with a Google account that has access to the bucket (Storage Object Admin role).
2. Copy `.env.example` to `.env` and set `HEIMDALL_BUCKET`.

To pull or push folders, edit the settings in
[`src/heimdall-core/utilities/sync.py`](src/heimdall-core/utilities/sync.py) and running:

```bash
uv run --project src/heimdall-core --extra gcs --env-file .env python -m utilities.sync
```

Files only copy when new or changed, and nothing is ever deleted. Models are never overwritten: push a
new version folder instead.
