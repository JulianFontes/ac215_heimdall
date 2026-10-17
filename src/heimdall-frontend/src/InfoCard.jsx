// Floating "glass" card with details about the forecast region (Apple Maps / Weather style).
// Forecast details come from public/data/meta.json; the region and grid size from the voxels. Click to collapse.
import React, {useState} from 'react';

const fmtTime = ms => new Date(ms).toLocaleString('en-US', {timeZone: 'UTC', month: 'short', day: 'numeric', hour: 'numeric'});
const fmtHour = ms => new Date(ms).toLocaleString('en-US', {timeZone: 'UTC', hour: 'numeric'});
const fmtDeg = v => `${+Math.abs(v).toFixed(2)}`;

export default function InfoCard({meta, region, settings}) {
  const [open, setOpen] = useState(!settings.collapsed);
  const run = meta.run ?? {};
  const t1 = meta.t0_ms + (meta.frames - 1) * meta.step_ms;
  // forecast_t0_ms: first forecast hour; the hours before it are observations (older meta.json files have none)
  const f0 = meta.forecast_t0_ms ?? meta.t0_ms, nObs = Math.round((f0 - meta.t0_ms) / meta.step_ms);
  const nFc = meta.frames - nObs;
  const {w, s, e, n, cellDeg, levelM} = region;
  const ns = v => (v >= 0 ? 'N' : 'S'), ew = v => (v >= 0 ? 'E' : 'W');
  const model = run.ckpt ? run.ckpt.split(/[\\/]/).slice(-2).join('/') : null; // e.g. june2026_epoch100/convlstm3d_best.pt
  const rows = [
    ['Observed', nObs ? `${fmtTime(meta.t0_ms)}–${fmtHour(f0 - meta.step_ms)} UTC (${nObs} h)` : null],
    ['Area', `${fmtDeg(s)}–${fmtDeg(n)}°${ns(n)} · ${fmtDeg(w)}–${fmtDeg(e)}°${ew(e)}`],
    ['Height', run.alt_max ? `Surface to ${run.alt_max} km` : null],
    ['Grid', `${+cellDeg.toFixed(3)}° · ${levelM} m levels`],
    ['Model', model],
    ['Data', run.in_training_period == null ? null : run.in_training_period ? 'Seen in training' : 'Not seen in training']
  ].filter(([, v]) => v);

  return (
    <div style={css.card} onClick={() => setOpen(o => !o)} title={open ? 'Collapse' : 'Show details'}>
      <div style={css.header}>
        <div style={css.title}>{settings.title}</div>
        <div style={{...css.chevron, transform: open ? 'rotate(180deg)' : 'none'}}>⌄</div>
      </div>
      <div style={css.subtitle}>
        Cloud forecast · {fmtTime(f0)}–{fmtHour(t1)} UTC ({nFc} h)
      </div>
      {open && (
        <div style={css.rows}>
          {rows.map(([k, v]) => (
            <div key={k} style={css.row}><span style={css.label}>{k}</span><span style={css.value}>{v}</span></div>
          ))}
        </div>
      )}
    </div>
  );
}

const css = {
  card: {
    position: 'absolute', zIndex: 10, top: 12, right: 56, width: 230, padding: '10px 12px', cursor: 'pointer',
    borderRadius: 12, color: '#fff', userSelect: 'none',
    background: 'rgba(28, 28, 30, 0.72)', backdropFilter: 'blur(20px) saturate(180%)', WebkitBackdropFilter: 'blur(20px) saturate(180%)',
    border: '0.5px solid rgba(255, 255, 255, 0.12)', boxShadow: '0 8px 32px rgba(0, 0, 0, 0.35)',
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", system-ui, sans-serif'
  },
  header: {display: 'flex', justifyContent: 'space-between', alignItems: 'center'},
  title: {fontSize: 14, fontWeight: 600, letterSpacing: -0.2},
  chevron: {fontSize: 12, color: 'rgba(235, 235, 245, 0.6)', transition: 'transform 0.2s', lineHeight: 1},
  subtitle: {fontSize: 11, color: 'rgba(235, 235, 245, 0.6)', marginTop: 2},
  rows: {marginTop: 8, paddingTop: 8, borderTop: '0.5px solid rgba(255, 255, 255, 0.15)', display: 'grid', rowGap: 4},
  row: {display: 'flex', justifyContent: 'space-between', gap: 10, fontSize: 11},
  label: {color: 'rgba(235, 235, 245, 0.6)'},
  value: {fontVariantNumeric: 'tabular-nums', textAlign: 'right', overflowWrap: 'anywhere'} // long paths wrap instead of overflowing
};
