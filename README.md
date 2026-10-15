# heimdall

## Running Heimdall

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

All services are defined in [`docker-compose.yml`](docker-compose.yml). Run every command below from this (repo root) folder.

### Frontend
3D cloud visualisation (Kepler.gl), served as a static site by nginx at http://localhost:8080.

**1. Make sure there is data to show.** The frontend reads `points.csv`, `voxels.geojson` and (optionally) `meta.json`
from `src/heimdall/heimdall-frontend/public/data/`. That folder is bind-mounted into the container, not built into the image,
so it must be filled first, either by:
- model forecasts: run `heimdall-nowcasting/cloud_forecasting/scripts/predict.py` (see its [README](src/heimdall/heimdall-nowcasting/cloud_forecasting/README.md)), or
- sample data: `cd src/heimdall/heimdall-frontend && npm install && npm run data` (needs Node 18+)

**2. (Optional) Mapbox token.** Only needed for the Mapbox basemap styles; the default basemap works without one.
Create a `.env` file in the repo root (it is git-ignored):
```bash
VITE_MAPBOX_TOKEN=pk....
```
The token is baked in at build time, so rebuild (`--build`) after changing it.

**3. Start it.**
```bash
docker compose up --build frontend      # first run, or after changing frontend code / token
docker compose up -d frontend           # later runs, in the background
docker compose down                     # stop
```
Open http://localhost:8080. When new data is written to `public/data/`, just refresh the browser; no rebuild needed.

For local development without Docker (hot reload), see the [frontend README](src/heimdall/heimdall-frontend/README.md).
