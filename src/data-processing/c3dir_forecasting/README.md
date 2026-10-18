# C3DIR forecasting data, step 1: GOES crops (data-processing)

GOES-19 full-disk scans over a month, cropped around several climate-diverse sites. Step 2,
[`model-deploy/c3dir_forecasting`](../../model-deploy/c3dir_forecasting/README.md), runs C3DIR on
the crops. Separate from `c3dir_nowcasting`.

| File | What it does |
|---|---|
| `collect.py` | download each scan once, save every site's crop, delete the scan (settings at the top) |

## Run

From the project root:

```bash
uv run --project src/data-processing python -m c3dir_forecasting.collect
```

No login needed. Cached: times already cropped for every site are skipped, so an interrupted run
picks up where it stopped, and changing `DAYS` adds days without redoing the old ones.

## What it collects

| Setting | Default | |
|---|---|---|
| sites | Penn State (40.72, −77.93), San José, Costa Rica (9.93, −84.08), Manaus (−3.12, −60.02), Buenos Aires (−34.60, −58.38), off Antofagasta, Chile (−23.50, −71.50) | mid-latitude summer, tropical convection, Amazon, southern-hemisphere winter, marine stratocumulus |
| days | 2025-08-01 to 2025-08-31 | C3DIR v002 was trained almost entirely on August 2025 |
| times | every full-disk scan from 17:40 to 20:00 UTC: 15 a day | the span of C3DIR's training scans (EarthCARE's early-afternoon passes) |
| GOES domain | full disk (`"F"`), ~340 MB a scan, downloaded once for all sites | covers every site; what C3DIR was trained on |
| crop | 320 × 320 pixels, all 16 bands | ±2° around the site plus slanted lines of sight |

About 30–60 s per scan time (almost all of it the download). The full-disk scans are deleted
once cropped; the crops are 3–4 MB each.

## Outputs

In `data/processed/c3dir_forecast/` (git-ignored):

| Path | Contents |
|---|---|
| `goes/<site>/<YYYYmmddTHHMM>.nc` | the crop: 16 bands `(y, x)` and `goes_imager_projection`; `attrs` hold the site, its lat / lon and the actual scan start |
| `manifest.csv` | one row per scan time attempted: scan start, status, seconds |

File names are the nominal scan time (UTC). The crop is cut exactly as model-deploy's
`c3dir.inference.crop` cuts it, so C3DIR on a crop gives the same field as C3DIR on the whole scan.
