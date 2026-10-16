## Running Heimdall

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

All services are defined in [`docker-compose.yml`](docker-compose.yml). Run every command below from this (repo root) folder.

### Commands
Everything goes through [`heimdall_docker.sh`](heimdall_docker.sh), a thin wrapper around `docker compose`.
Leave out the service name to build all services included in `docker-compose.yaml`.
```bash
./heimdall_docker.sh build [service...]   # build images (add --no-cache for a clean rebuild)
./heimdall_docker.sh up    [service...]   # build anything that changed, start in the background
./heimdall_docker.sh down                 # stop and remove containers (images are kept)
```

### Frontend
3D cloud visualisation (Kepler.gl), served as a static site by nginx at http://localhost:8080.

**1. Make sure there is data to show.** The frontend reads `points.csv`, `voxels.geojson` and (optionally) `meta.json`
from `src/heimdall/heimdall-frontend/public/data/`. That folder is bind-mounted into the container, not built into the image,
so it must be filled first, either by:
- model forecasts: run `heimdall-nowcasting/cloud_forecasting/scripts/predict.py` (see its [README](src/heimdall/heimdall-nowcasting/cloud_forecasting/README.md)), or
- sample data: `cd src/heimdall/heimdall-frontend && npm install && npm run data` (needs Node 18+)

**2. (Optional) Mapbox token.** Only needed for the Mapbox basemap styles; the default basemap works without one.
Copy the template and fill in the token (`.env` is git-ignored):
```bash
cp .env.example .env      # then set VITE_MAPBOX_TOKEN=pk....
```
The token is baked in at build time; `./heimdall_docker.sh up` rebuilds automatically after you change it.

**3. Start it.**
```bash
./heimdall_docker.sh up frontend
```
Open http://localhost:8080. When new data is written to `public/data/`, just refresh the browser; no rebuild needed.

**Developing the frontend?** Use `src/heimdall/heimdall-frontend/start_frontend.sh` instead of Docker: it runs the Vite
dev server with hot reload, so code changes show up instantly without rebuilding. See the [frontend README](src/heimdall/heimdall-frontend/README.md).
