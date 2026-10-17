// Generates a synthetic 3D cloud (points) and a voxelised version (GeoJSON, one extruded square per voxel),
// as a time series: the blobs drift and grow/decay over FRAMES hourly snapshots.
// Replace this with your own pipeline: you only need lat, lon, alt (m), density, confidence, timestamp.
import {writeFileSync, mkdirSync} from 'node:fs';

const M = 111320, LAT0 = 37.77, LON0 = -122.42, CL = Math.cos(LAT0 * Math.PI / 180);
const VOXEL = Number(process.argv[2] || 100); // meters

// Time: FRAMES snapshots, STEP_MS apart, starting at T0. Keep T0/STEP_MS in sync with src/config.js.
const FRAMES = 12, T0 = Date.UTC(2026, 9, 1), STEP_MS = 3600e3;
const fmtTime = ms => new Date(ms).toISOString().slice(0, 19).replace('T', ' '); // 'YYYY-MM-DD HH:mm:ss', auto-detected by Kepler

function rng(s) { return () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const r = rng(7);
const gauss = () => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// n: points per frame, vx/vy: drift in m per frame, grow: relative amplitude change per frame
const blobs = [
  {x: -300, y: 200, z: 500, sx: 350, sy: 300, sz: 130, amp: 100, n: 1400, vx: 60, vy: 25, grow: -0.04},
  {x: 500, y: -250, z: 800, sx: 280, sy: 380, sz: 170, amp: 75, n: 1000, vx: -35, vy: 40, grow: 0.03},
  {x: 100, y: -600, z: 300, sx: 200, sy: 200, sz: 90, amp: 55, n: 600, vx: 20, vy: 55, grow: 0.06}
];

const pts = [];
for (const b of blobs) {
  const bx = b.x + (r() - .5) * 200, by = b.y + (r() - .5) * 200, cz = b.z + (r() - .5) * 150;
  for (let f = 0; f < FRAMES; f++) {
    const cx = bx + b.vx * f, cy = by + b.vy * f, amp = b.amp * Math.max(0.2, 1 + b.grow * f), t = fmtTime(T0 + f * STEP_MS);
    for (let i = 0; i < b.n; i++) {
      const x = cx + gauss() * b.sx, y = cy + gauss() * b.sy, z = Math.max(15, cz + gauss() * b.sz);
      const d2 = ((x - cx) / b.sx) ** 2 + ((y - cy) / b.sy) ** 2 + ((z - cz) / b.sz) ** 2;
      const density = amp * Math.exp(-d2 / 2) * (0.85 + 0.3 * r());
      const dist = Math.hypot(x, y + 1000, z); // simulated sensor at (0,-1000,0): confidence decays with range
      const confidence = clamp(0.98 - dist / 4200 + (r() - .5) * 0.2, 0.05, 0.99);
      pts.push({lat: LAT0 + y / M, lon: LON0 + x / (M * CL), alt: z, density, confidence, t});
    }
  }
}

mkdirSync('public/data', {recursive: true});
writeFileSync('public/data/points.csv',
  'lat,lon,alt,density,confidence,timestamp\n' +
  pts.map(p => `${p.lat.toFixed(6)},${p.lon.toFixed(6)},${p.alt.toFixed(1)},${p.density.toFixed(2)},${p.confidence.toFixed(3)},${p.t}`).join('\n'));

// voxelise per frame (one shared grid, so a voxel stays put between frames)
const dLat = VOXEL / M, dLon = VOXEL / (M * CL);
const lat0 = Math.min(...pts.map(p => p.lat)), lon0 = Math.min(...pts.map(p => p.lon));
const cells = new Map();
for (const p of pts) {
  const ix = Math.floor((p.lon - lon0) / dLon), iy = Math.floor((p.lat - lat0) / dLat), iz = Math.floor(p.alt / VOXEL);
  const k = `${p.t},${ix},${iy},${iz}`;
  const c = cells.get(k) || {ix, iy, iz, t: p.t, d: 0, c: 0, n: 0};
  c.d += p.density; c.c += p.confidence; c.n++; cells.set(k, c);
}
const cov = 0.92;
const features = [...cells.values()].map(c => {
  const x0 = lon0 + (c.ix + (1 - cov) / 2) * dLon, x1 = lon0 + (c.ix + 1 - (1 - cov) / 2) * dLon;
  const y0 = lat0 + (c.iy + (1 - cov) / 2) * dLat, y1 = lat0 + (c.iy + 1 - (1 - cov) / 2) * dLat;
  const z = c.iz * VOXEL;
  return {
    type: 'Feature',
    properties: {density: +(c.d / c.n).toFixed(2), confidence: +(c.c / c.n).toFixed(3), points: c.n, base_alt_m: z, height_m: VOXEL, timestamp: c.t},
    geometry: {type: 'Polygon', coordinates: [[[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]]]}
  };
});
writeFileSync('public/data/voxels.geojson', JSON.stringify({type: 'FeatureCollection', features}));
console.log(`points: ${pts.length}, voxels: ${features.length} (${VOXEL} m), frames: ${FRAMES}`);
