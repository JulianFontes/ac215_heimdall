import React, {useEffect, useState} from 'react';
import {useDispatch} from 'react-redux';
import KeplerGl from '@kepler.gl/components';
import {addDataToMap} from '@kepler.gl/actions';
import {processCsvData, processGeojson} from '@kepler.gl/processors';
import {buildConfig} from './config';

// Kepler's default basemap needs a Mapbox token: put it in .env as VITE_MAPBOX_TOKEN=pk...
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || '';

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

  useEffect(() => {
    (async () => {
      try {
        const [csv, geo] = await Promise.all([
          fetch('/data/points.csv').then(r => r.text()),
          fetch('/data/voxels.geojson').then(r => r.json())
        ]);
        // Add data to the Kepler map
        dispatch(
          addDataToMap({
            datasets: [
              {info: {id: 'points', label: 'Point cloud'}, data: processCsvData(csv)},
              {info: {id: 'voxels', label: 'Voxels'}, data: processGeojson(geo)}
            ],
            options: {centerMap: false, readOnly: false},
            config: buildConfig()
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
    </div>
  );
}
