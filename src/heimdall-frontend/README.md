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

Without the script, by hand from this folder:
```bash
docker build -t heimdall-frontend .                                  # no Mapbox token
docker build -t heimdall-frontend --build-arg VITE_MAPBOX_TOKEN=pk.... .   # with token (baked in at build time)
docker run --rm -p 8080:80 -v "$(pwd)/public/data:/usr/share/nginx/html/data:ro" heimdall-frontend
```
