// Kepler.gl map config: two layers (voxels as extruded 3D cells, raw 3D point cloud) + confidence/density filters.
// Tweak anything here, or in the Kepler UI, then use "Share > Export Map" to get an updated config.
const warm = {
  name: 'Global Warming', type: 'sequential', category: 'Uber',
  colors: ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']
};
const conf = {
  name: 'Confidence', type: 'diverging', category: 'ColorBrewer',
  colors: ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
};
// Must match T0 / STEP_MS in scripts/generate-data.mjs. The time window is just under one step wide,
// so it shows one snapshot at a time while playing.
const T0 = Date.UTC(2026, 9, 1), STEP_MS = 3600e3;

export function buildConfig() {
  return {
    version: 'v1',
    config: {
      visState: {
        filters: [
          // Kepler 3.x: `view` replaced `enlarged`, and `plotType` is an object, not the string 'histogram'.
          {dataId: ['voxels'], id: 'f-conf', name: ['confidence'], type: 'range', value: [0.3, 1], view: 'side', plotType: {type: 'histogram'}, animationWindow: 'free', yAxis: null, speed: 1},
          {dataId: ['voxels'], id: 'f-dens', name: ['density'], type: 'range', value: [0, 100], view: 'side', plotType: {type: 'histogram'}, animationWindow: 'free', yAxis: null, speed: 1},
          // Time filter synced across both datasets; view 'enlarged' = the timeline with play button at the bottom.
          {dataId: ['voxels', 'points'], id: 'f-time', name: ['timestamp', 'timestamp'], type: 'timeRange', value: [T0, T0 + STEP_MS - 1000], view: 'enlarged', plotType: {type: 'histogram'}, animationWindow: 'free', yAxis: null, speed: 1}
        ],
        layers: [
          {
            id: 'voxel-layer', type: 'geojson',
            config: {
              dataId: 'voxels', label: 'Voxels (extruded)', color: [255, 195, 0], columns: {geojson: '_geojson'}, isVisible: true,
              visConfig: {
                opacity: 0.9, strokeOpacity: 0.8, thickness: 0.5, strokeColor: null, colorRange: warm, strokeColorRange: warm,
                radius: 10, sizeRange: [0, 10], radiusRange: [0, 50], heightRange: [0, 100],
                elevationScale: 1, enableElevationZoomFactor: false, stroked: false, filled: true, enable3d: true, wireframe: false
              },
              hidden: false, textLabel: []
            },
            visualChannels: {
              colorField: {name: 'density', type: 'real'}, colorScale: 'quantile',
              strokeColorField: null, strokeColorScale: 'quantile',
              sizeField: null, sizeScale: 'linear',
              heightField: {name: 'height_m', type: 'integer'}, heightScale: 'linear',
              radiusField: null, radiusScale: 'linear'
            }
          },
          {
            id: 'point-layer', type: 'point',
            config: {
              dataId: 'points', label: 'Point cloud (3D)', color: [18, 147, 154], columns: {lat: 'lat', lng: 'lon', altitude: 'alt'}, isVisible: false,
              visConfig: {
                radius: 3, fixedRadius: false, opacity: 0.8, outline: false, thickness: 2, strokeColor: null,
                colorRange: conf, strokeColorRange: conf, radiusRange: [0, 50], filled: true
              },
              hidden: false, textLabel: []
            },
            visualChannels: {
              colorField: {name: 'confidence', type: 'real'}, colorScale: 'quantile',
              strokeColorField: null, strokeColorScale: 'quantile', sizeField: null, sizeScale: 'linear'
            }
          }
        ],
        interactionConfig: {
          tooltip: {
            fieldsToShow: {
              voxels: [{name: 'density', format: null}, {name: 'confidence', format: null}, {name: 'points', format: null}, {name: 'base_alt_m', format: null}, {name: 'timestamp', format: null}],
              points: [{name: 'density', format: null}, {name: 'confidence', format: null}, {name: 'alt', format: null}, {name: 'timestamp', format: null}]
            },
            compareMode: false, compareType: 'absolute', enabled: true
          },
          brush: {size: 0.5, enabled: false}, geocoder: {enabled: false}, coordinate: {enabled: false}
        }
      },
      mapState: {latitude: 37.7735, longitude: -122.4235, zoom: 13.6, pitch: 55, bearing: -25, dragRotate: true, isSplit: false},
      // 'dark-matter' is Kepler's MapLibre/Carto basemap: no Mapbox token needed. The 'dark'/'light'/'muted'
      // styles are mapbox:// URLs and render blank without VITE_MAPBOX_TOKEN.
      mapStyle: {styleType: 'dark-matter', topLayerGroups: {}, visibleLayerGroups: {label: true, road: true, border: false, building: true, water: true, land: true, '3d building': false}}
    }
  };
}
