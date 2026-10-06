# Kepler.gl – 3D voxel cloud PoC

Kepler.gl 3.1 (React + Vite) showing a synthetic 3D cloud as (a) extruded voxels and (b) a raw 3D point cloud,
with **confidence** and **density** as built-in Kepler filters (histogram sliders) and colour channels.

### Mapbox token
**Not required.** The configured basemap is `dark-matter`, which Kepler 3.1 serves from Carto over MapLibre -
no account, no token. Set `VITE_MAPBOX_TOKEN=pk....` in `.env` only if you want the Mapbox-hosted styles
(Dark / Light / Muted / Satellite) in Kepler's basemap picker; those render blank without one.

## Run
```bash
./start.sh             # checks Node, installs deps, generates data, starts the server
```

## Our Own Data
This will be updated once we serve data from our models to the front end. As of now the current frontend generates and loads in default cloud points and voxels.
- `public/data/points.csv`: `lat, lon, alt, density, confidence, timestamp` (alt in metres, timestamp as `YYYY-MM-DD HH:mm:ss` UTC)
- `public/data/voxels.geojson`: square Polygons with z = base altitude and properties `density, confidence, points, base_alt_m, height_m, timestamp`
Edit `src/config.js` to change layers, colors and filters.

### Time
Each row is one snapshot (sample data: 12 hourly frames from 2026-10-01 00:00 UTC). The `f-time` filter in `src/config.js`
drives Kepler's timeline at the bottom of the map (press play to animate) and is shared by both layers.
If you change the time range of your data, update `T0` / `STEP_MS` in `src/config.js`, or just drag the timeline handles.

## Working inside iCloud Drive (macOS)
`node_modules` contains tens of thousands of files and iCloud will try to sync them, which slows installs and can corrupt them.
Any folder ending in `.nosync` is not synced, so `node_modules` is kept as a symlink into `node_modules.nosync`.
`./start_frontend.sh` sets this up for you and converts an existing real `node_modules` directory if you already installed. Manually:
```bash
mv node_modules node_modules.nosync   # or: mkdir node_modules.nosync
ln -s node_modules.nosync node_modules
```