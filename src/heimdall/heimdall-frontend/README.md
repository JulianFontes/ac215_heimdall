# Kepler.gl – 3D voxel cloud PoC

Kepler.gl 3.1 (React + Vite) showing a synthetic 3D cloud as (a) extruded voxels and (b) a raw 3D point cloud,
with **confidence** and **density** as built-in Kepler filters (histogram sliders) and colour channels.

### Mapbox token
**Not required.** The configured basemap is `dark-matter`, which Kepler 3.1 serves from Carto over MapLibre -
no account, no token. Set `VITE_MAPBOX_TOKEN=pk....` in `.env` only if you want the Mapbox-hosted styles
(Dark / Light / Muted / Satellite) in Kepler's basemap picker; those render blank without one.

## Run
There are two independent ways to run the frontend. Pick one; they don't depend on each other.

> **Developing the frontend? Use `./start_frontend.sh`.** Code changes show up in the browser instantly, with no rebuilds.
> Use Docker when you just want to run the app, or to check that the containerised build works before sharing it.

| | Local dev server (`start_frontend.sh`) | Docker (`heimdall_docker.sh`) |
|---|---|---|
| **Runs on** | Your Mac directly | Inside a container |
| **Needs** | Node 18+ | Only Docker Desktop |
| **Served by** | Vite dev server, http://localhost:5173 | nginx, http://localhost:8080 |
| **Code changes** | Show up instantly (hot reload) | Need a rebuild |
| **Use it for** | Working on the frontend code | Running the app / sharing it with teammates |

### Local dev server (recommended for development)
```bash
./start_frontend.sh             # checks Node, installs deps, generates data, starts the server
```

### Docker
From the repo root (see the [top-level README](../../../README.md); put `VITE_MAPBOX_TOKEN=pk....` in the root `.env` if you want the token):
```bash
./heimdall_docker.sh build frontend   # build the image (again after code or token changes)
./heimdall_docker.sh up frontend      # start it, then open http://localhost:8080
./heimdall_docker.sh down             # stop it
```

How the container works:
- **Build time:** the [Dockerfile](Dockerfile)'s first stage installs the packages and runs `npm run build`. Only the
  resulting `dist/` folder is copied into a small nginx image; Node is not in the final image.
- **Start-up:** the Dockerfile's `CMD ["nginx", "-g", "daemon off;"]` is what runs when the container starts. nginx serves
  the built app, and the app itself then runs in your browser. `start_frontend.sh` is **not** used by Docker (it is in
  `.dockerignore`): its Node checks, iCloud workaround, installs and dev server only make sense on your Mac.
- **Data:** not built into the image. `public/data` is bind-mounted into the container (see `docker-compose.yml`), so new
  `predict.py` output shows up on a browser refresh with no rebuild. Generate data first (`npm run data` or `predict.py`).

Without the script, by hand from this folder:
```bash
docker build -t heimdall-frontend .                                  # no Mapbox token
docker build -t heimdall-frontend --build-arg VITE_MAPBOX_TOKEN=pk.... .   # with token (baked in at build time)
docker run --rm -p 8080:80 -v "$(pwd)/public/data:/usr/share/nginx/html/data:ro" heimdall-frontend
```

## Our Own Data
This will be updated once we serve data from our models to the front end. As of now the current frontend generates and loads in default cloud points and voxels. Note that this data was AI generated with the minimum values required to visualize and demonstrate the filtering.
- `public/data/points.csv`: `lat, lon, alt, density, confidence, timestamp` (alt in metres, timestamp as `YYYY-MM-DD HH:mm:ss` UTC)
- `public/data/voxels.geojson`: square Polygons with z = base altitude and properties `density, confidence, points, base_alt_m, height_m, timestamp`
Edit `src/settings.js` to change the defaults (playback speed, filter ranges, layer colours and opacity, camera, basemap).
`src/config.js` turns those settings into Kepler's config format; edit it only for things `settings.js` doesn't cover.

To show model forecasts instead, run `heimdall-nowcasting/cloud_forecasting/scripts/predict.py` (see `heimdall-nowcasting/cloud_forecasting/README.md`). It writes both files with the same columns plus a `cloud_type` label (ice / liquid / rain), and a `meta.json` that sets the timeline start and map centre. Delete the three files to go back to the sample data on the next `./start_frontend.sh`.

### Time
Each row is one snapshot (sample data: 12 hourly frames from 2026-10-01 00:00 UTC). The `f-time` filter in `src/config.js`
drives Kepler's timeline at the bottom of the map (press play to animate) and is shared by both layers.
If you change the time range of the sample data, update `sample.t0_ms` / `sample.step_ms` in `src/settings.js`, or just drag the timeline handles
(forecasts take theirs from `meta.json`).

## Working inside iCloud Drive (macOS)
`node_modules` contains tens of thousands of files and iCloud will try to sync them, which slows installs and can corrupt them.
Any folder ending in `.nosync` is not synced, so `node_modules` is kept as a symlink into `node_modules.nosync`.
`./start_frontend.sh` sets this up for you and converts an existing real `node_modules` directory if you already installed. Manually:
```bash
mv node_modules node_modules.nosync   # or: mkdir node_modules.nosync
ln -s node_modules.nosync node_modules
```
