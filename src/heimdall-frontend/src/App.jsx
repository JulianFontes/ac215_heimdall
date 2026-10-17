import React, {useEffect, useState} from 'react';
import {useDispatch} from 'react-redux';
import KeplerGl from '@kepler.gl/components';
import {addDataToMap} from '@kepler.gl/actions';
import {processCsvData, processGeojson} from '@kepler.gl/processors';
import {buildConfig} from './config';
import InfoCard from './InfoCard';
import {SETTINGS} from './settings';

// Kepler's default basemap needs a Mapbox token: put it in .env as VITE_MAPBOX_TOKEN=pk...
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || '';

// Outline of the region: the outer edges of the grid, cut into dashes
// (Kepler's GeoJSON layer can't draw dashed lines, so each dash is its own short line)
function regionOutline({w, s, e, n}, {dashDeg, gapDeg}) {
  const corners = [[w, s], [e, s], [e, n], [w, n], [w, s]], dashes = [];
  for (let i = 0; i < 4; i++) {
    const [[x0, y0], [x1, y1]] = [corners[i], corners[i + 1]], len = Math.hypot(x1 - x0, y1 - y0);
    const at = d => [x0 + (x1 - x0) * d / len, y0 + (y1 - y0) * d / len];
    if (!dashDeg) { dashes.push([corners[i], corners[i + 1]]); continue; }
    for (let d = 0; d < len; d += dashDeg + gapDeg) dashes.push([at(d), at(Math.min(d + dashDeg, len))]);
  }
  return {type: 'FeatureCollection', features: [{type: 'Feature', geometry: {type: 'MultiLineString', coordinates: dashes}, properties: {name: 'region'}}]};
}

// Outer edges of the voxel grid, plus the cell size (degrees) and level height (m) of its voxels
function regionBounds(geo) {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const f of geo.features) {
    for (const [lon, lat] of f.geometry.coordinates[0]) {
      w = Math.min(w, lon); e = Math.max(e, lon); s = Math.min(s, lat); n = Math.max(n, lat);
    }
  }
  const [[x0], [x1]] = geo.features[0].geometry.coordinates[0];
  return {w, s, e, n, cellDeg: Math.abs(x1 - x0), levelM: geo.features[0].properties.height_m};
}

function useWindowSize() {
  const [size, setSize] = useState({w: window.innerWidth, h: window.innerHeight});
  useEffect(() => {
    const on = () => setSize({w: window.innerWidth, h: window.innerHeight});
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);
  return size;
}

export default function App() {
  const dispatch = useDispatch();
  const {w, h} = useWindowSize();
  const [error, setError] = useState('');
  const [info, setInfo] = useState(null); // {meta, region} for the info card (forecasts only)

  useEffect(() => {
    (async () => {
      try {
        const [csv, geo, meta] = await Promise.all([
          fetch('/data/points.csv', {cache: 'no-cache'}).then(r => r.text()),
          fetch('/data/voxels.geojson', {cache: 'no-cache'}).then(r => r.json()),
          // optional: written alongside model forecasts by heimdall-nowcasting/cloud_forecasting/scripts/predict.py
          fetch('/data/meta.json', {cache: 'no-cache'}).then(r => (r.ok ? r.json() : null)).catch(() => null)
        ]);
        const bounds = regionBounds(geo);
        // The card shows the grid's cell centres (39.5-45.5 N), not its outer edges (39.375-45.625 N)
        const half = bounds.cellDeg / 2;
        if (meta) setInfo({meta, region: {...bounds, w: bounds.w + half, e: bounds.e - half, s: bounds.s + half, n: bounds.n - half}});
        // Add data to the Kepler map
        dispatch(
          addDataToMap({
            datasets: [
              {info: {id: 'points', label: 'Point cloud'}, data: processCsvData(csv)},
              {info: {id: 'voxels', label: 'Voxels'}, data: processGeojson(geo)},
              {info: {id: 'region', label: 'Region'}, data: processGeojson(regionOutline(bounds, SETTINGS.region))}
            ],
            options: {centerMap: false, readOnly: false},
            config: buildConfig(meta)
          })
        );
      } catch (e) {
        console.error(e);
        setError(String(e.message || e));
      }
    })();
  }, [dispatch]);

  return (
    <div style={{position: 'absolute', inset: 0}}>
      {error && <div style={{position: 'absolute', zIndex: 10, top: 8, left: 8, color: '#f66'}}>{error}</div>}
      <KeplerGl id="keplerGl" mapboxApiAccessToken={MAPBOX_TOKEN} width={w} height={h} />
      {info && SETTINGS.info.visible && <InfoCard {...info} settings={SETTINGS.info} />}
    </div>
  );
}
