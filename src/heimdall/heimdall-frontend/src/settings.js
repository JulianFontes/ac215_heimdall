// Frontend defaults: edit the values here, save, and the page reloads with them (`npm run dev` hot-reloads).
// Everything is still adjustable in the Kepler UI afterwards; these are only the starting state.
// Settings that differ between the forecasting model's output (public/data/meta.json present)
// and the sample point cloud (scripts/generate-data.mjs) have a `forecast` and a `sample` value.

export const SETTINGS = {
  // Timeline at the bottom of the map
  playback: {
    speed: 6                    // animation speed multiplier of the play button
  },

  // Starting ranges of the filter sliders (voxel layer)
  filters: {
    confidence: [0.3, 1],       // probability of the predicted cloud type, 0-1
    density: {                  // P(any cloud) x 100, 0-100
      forecast: [60, 100],      // forecasts export every bin, so hide the unlikely ones
      sample: [0, 100]
    }
  },

  // Extruded 3D voxels (shown on load)
  voxels: {
    visible: true,
    opacity: 0.25,              // 0-1; low values let stacked layers build up like cloud
    colorBy: 'density',         // any voxel property: 'density', 'confidence', 'cloud_type' (forecasts only), ...
    colorScale: 'quantize',     // 'quantize' = equal-width steps, 'quantile' = equal number of voxels per colour
    colors: ['#4a5560', '#6b7682', '#8f99a3', '#b4bcc4', '#d9dee2', '#ffffff'] // low -> high, grey to white
  },

  // Raw 3D point cloud (hidden on load)
  points: {
    visible: false,
    opacity: 0.8,
    radius: 3,
    colorBy: 'confidence',
    colorScale: 'quantile',
    colors: ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'] // low -> high, red to green
  },

  // Outline around the whole lat/lon region of the data (not on the timeline, so always drawn)
  region: {
    visible: true,
    color: [186, 90, 90],       // RGB
    thickness: 2,               // line width in pixels
    opacity: 1
  },

  // Floating details card, top right (forecasts only; filled from meta.json). Click it to collapse / expand.
  info: {
    visible: true,
    title: 'Info Box',       // region name shown as the card's heading
    collapsed: false            // true = start as just the title line
  },

  // Starting camera. Forecast centre comes from meta.json; the sample one is below.
  camera: {
    view3d: false,              // false = flat 2D map, north up; the 3D button in the Kepler map controls switches it on
    zoom: {forecast: 4, sample: 13.6}, // starting zoom level (higher = closer in)
    pitch: 55,                  // 3D only: tilt in degrees, 0 = straight down
    bearing: -25                // 3D only: rotation in degrees, 0 = north up
  },

  // 'dark-matter', 'positron' and 'voyager' are Kepler's MapLibre/Carto basemaps (no token needed).
  // 'dark' / 'light' / 'muted' are Mapbox styles and render blank without VITE_MAPBOX_TOKEN.
  basemap: 'dark-matter',
  // Map overlay blending: how layers mix with the basemap so both stay visible.
  // 'screen' = "dark base map" in the Kepler UI (lightens, good for white clouds on dark-matter),
  // 'darken' = "light base map" (for positron / voyager), 'normal' = no blending.
  overlayBlending: 'screen',

  // Time range and map centre for the sample data; must match T0 / STEP_MS in scripts/generate-data.mjs.
  // Forecasts take these from public/data/meta.json (written by cloud_forecasting/scripts/predict.py).
  sample: {
    t0_ms: Date.UTC(2026, 9, 1),
    step_ms: 3600e3,
    center: {latitude: 37.7735, longitude: -122.4235}
  }
};
