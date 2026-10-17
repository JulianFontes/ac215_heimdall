// Kepler.gl map config: voxels as extruded 3D cells, raw 3D point cloud and an outline of the data region,
// plus confidence/density filters. The info card is an HTML overlay (InfoCard.jsx), not a Kepler layer.
// The adjustable defaults (speeds, filter ranges, colours, camera, basemap) live in settings.js; this file only
// assembles them into Kepler's config format. For anything not in settings.js, tweak it in the Kepler UI and use
// "Share > Export Map" to see the matching config.
import {SETTINGS} from './settings';

const palette = (name, colors) => ({name, type: 'sequential', category: 'Custom', colors});
// Kepler needs each colour field's type; text fields such as cloud_type need an ordinal (one colour per value) scale.
const FIELD_TYPES = {density: 'real', confidence: 'real', points: 'integer', base_alt_m: 'integer', alt: 'real', cloud_type: 'string'};
const colorChannel = ({colorBy, colorScale}) => {
  const type = FIELD_TYPES[colorBy] ?? 'real';
  return {colorField: {name: colorBy, type}, colorScale: type === 'string' ? 'ordinal' : colorScale};
};

export function buildConfig(meta = null) {
  const S = SETTINGS, mode = meta ? 'forecast' : 'sample'; // meta.json is only written for model forecasts
  const T0 = meta?.t0_ms ?? S.sample.t0_ms, STEP_MS = meta?.step_ms ?? S.sample.step_ms;
  const {latitude, longitude} = meta?.center ?? S.sample.center;
  const voxelColors = palette('Voxels', S.voxels.colors), pointColors = palette('Points', S.points.colors);
  return {
    version: 'v1',
    config: {
      visState: {
        filters: [
          // Kepler 3.x: `view` replaced `enlarged`, and `plotType` is an object, not the string 'histogram'.
          {dataId: ['voxels'], id: 'f-conf', name: ['confidence'], type: 'range', value: S.filters.confidence, view: 'side', plotType: {type: 'histogram'}, animationWindow: 'free', yAxis: null, speed: 1},
          {dataId: ['voxels'], id: 'f-dens', name: ['density'], type: 'range', value: S.filters.density[mode], view: 'side', plotType: {type: 'histogram'}, animationWindow: 'free', yAxis: null, speed: 1},
          // Time filter synced across both datasets; view 'enlarged' = the timeline with play button at the bottom.
          // The window is just under one step wide, so it shows one snapshot at a time while playing.
          {dataId: ['voxels', 'points'], id: 'f-time', name: ['timestamp', 'timestamp'], type: 'timeRange', value: [T0, T0 + STEP_MS - 1000], view: 'enlarged', plotType: {type: 'histogram'}, animationWindow: 'free', yAxis: null, speed: S.playback.speed}
        ],
        layers: [
          // first in the list = drawn on top of the voxels
          {
            id: 'region-layer', type: 'geojson',
            config: {
              dataId: 'region', label: 'Region outline', color: S.region.color, columns: {geojson: '_geojson'}, isVisible: S.region.visible,
              visConfig: {
                opacity: S.region.opacity, strokeOpacity: S.region.opacity, thickness: S.region.thickness, strokeColor: S.region.color,
                stroked: true, filled: false, enable3d: false, wireframe: false
              },
              hidden: false, textLabel: []
            },
            visualChannels: {colorField: null, colorScale: 'quantile', strokeColorField: null, strokeColorScale: 'quantile', sizeField: null, sizeScale: 'linear'}
          },
          {
            id: 'voxel-layer', type: 'geojson',
            config: {
              dataId: 'voxels', label: 'Voxels (extruded)', color: [255, 195, 0], columns: {geojson: '_geojson'}, isVisible: S.voxels.visible,
              visConfig: {
                opacity: S.voxels.opacity, strokeOpacity: 0.8, thickness: 0.5, strokeColor: null, colorRange: voxelColors, strokeColorRange: voxelColors,
                radius: 10, sizeRange: [0, 10], radiusRange: [0, 50], heightRange: [0, 100],
                elevationScale: 1, enableElevationZoomFactor: false, stroked: false, filled: true, enable3d: true, wireframe: false
              },
              hidden: false, textLabel: []
            },
            visualChannels: {
              ...colorChannel(S.voxels),
              strokeColorField: null, strokeColorScale: 'quantile',
              sizeField: null, sizeScale: 'linear',
              heightField: {name: 'height_m', type: 'integer'}, heightScale: 'linear',
              radiusField: null, radiusScale: 'linear'
            }
          },
          {
            id: 'point-layer', type: 'point',
            config: {
              dataId: 'points', label: 'Point cloud (3D)', color: [18, 147, 154], columns: {lat: 'lat', lng: 'lon', altitude: 'alt'}, isVisible: S.points.visible,
              visConfig: {
                radius: S.points.radius, fixedRadius: false, opacity: S.points.opacity, outline: false, thickness: 2, strokeColor: null,
                colorRange: pointColors, strokeColorRange: pointColors, radiusRange: [0, 50], filled: true
              },
              hidden: false, textLabel: []
            },
            visualChannels: {
              ...colorChannel(S.points),
              strokeColorField: null, strokeColorScale: 'quantile', sizeField: null, sizeScale: 'linear'
            }
          }
        ],
        overlayBlending: S.overlayBlending,
        interactionConfig: {
          tooltip: {
            fieldsToShow: {
              // model forecasts (meta.json present) also carry a cloud_type label and an observed / forecast source
              voxels: [...(meta ? [{name: 'source', format: null}, {name: 'cloud_type', format: null}] : []), {name: 'density', format: null}, {name: 'confidence', format: null}, {name: 'points', format: null}, {name: 'base_alt_m', format: null}, {name: 'timestamp', format: null}],
              points: [...(meta ? [{name: 'source', format: null}, {name: 'cloud_type', format: null}] : []), {name: 'density', format: null}, {name: 'confidence', format: null}, {name: 'alt', format: null}, {name: 'timestamp', format: null}]
            },
            compareMode: false, compareType: 'absolute', enabled: true
          },
          brush: {size: 0.5, enabled: false}, geocoder: {enabled: false}, coordinate: {enabled: false}
        }
      },
      mapState: {latitude, longitude, zoom: S.camera.zoom[mode], pitch: S.camera.view3d ? S.camera.pitch : 0, bearing: S.camera.view3d ? S.camera.bearing : 0,
                 dragRotate: S.camera.view3d, isSplit: false}, // dragRotate = Kepler's 3D map toggle
      mapStyle: {styleType: S.basemap, topLayerGroups: {}, visibleLayerGroups: {label: true, road: true, border: false, building: true, water: true, land: true, '3d building': false}}
    }
  };
}
