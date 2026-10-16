# C3DIR: 3D clouds from GOES-19, trained on EarthCARE

The goal is a model that takes a GOES-19 image and predicts, for every pixel, an
80-level column of cloud properties (ice, liquid and rain water content, particle
size, and cloud masks), with uncertainty. It follows the C3DIR paper (White et al.,
2026); section numbers below refer to that paper.

EarthCARE measures these properties directly, but only along its narrow ground track.
GOES-19 sees the whole disk every 10 minutes, but at a slant: each pixel looks down a
tilted line of sight, so a cloud appears shifted from its ground position (~2 km for a
10 km cloud in the tropics, ~30 km at 60°N). Training data is built by matching each
EarthCARE voxel (profile, height) to the GOES pixel whose line of sight passes within
5 km of it at that height, in a scan taken within 2.5 minutes (paper §3).

## Pipeline

```
data_scraping/  [scrape, core]
  EarthCareLabels      earthcare.py   find, download and bin EarthCARE frames
        │                              → data/earthcare/labels/<frame>.nc
        ▼
  GOES19.collocate()   goes.py        download GOES scans, match voxels along lines of sight
        │                              → data/collocated/<frame>_<scan>_<domain>.nc
        ▼
  Samples, C3DIRDataset  dataset.py    training samples, train / val / test split, 64×64 patches
        │                              → data/samples/
        ▼
model/  [ml]
  C3DIR                c3dir.py       train, predict, save versioned models
        │                              → artifacts/models/c3dir/<version>/
        ▼
  evaluation, inference               test scores; predictions on new GOES scans
```

See [Quick start](#quick-start) for the scripts that run each step.

Each folder matches a dependency group in `pyproject.toml`; the top-level files and
`data_scraping/dataset.py` need only the core libraries.

| File | What it does | Needs |
|---|---|---|
| `constants.py` | levels, bands, target names and order shared by every part | core |
| `geometry.py` | GOES-19 line-of-sight geometry (`scan_angles`, `ray_point`, `ray_latlon`, `closest_ray`) | core |
| `data_scraping/dataset.py` | `Samples` builds and splits training samples; `C3DIRDataset` and `collate` serve batches | core |
| `data_scraping/earthcare.py` | `EarthCareLabels`: finds EarthCARE ACM_CAP_2B frames GOES-19 can see, downloads them and averages them onto 80 levels of 0.25 km | scrape |
| `data_scraping/goes.py` | `GOES19.collocate()`: downloads GOES-19 scans and matches EarthCARE voxels to their lines of sight | scrape |
| `data_scraping/scrape.py` | `scrape(start, end)`: runs the two steps above for a time window | scrape |
| `model/network.py` | `C3DIRNet`: the network | ml |
| `model/c3dir.py` | `C3DIR`: training (`fit`), prediction (`predict_voxels`), `save` and `load` | ml |
| `model/evaluation.py` | test predictions and scores (`detection_scores`, `water_scores`) | ml |
| `model/inference.py` | predictions on any GOES-19 scan: the column above a point (`predict_column`) and the 3D field around it (`predict_region`) | ml, scrape |
| `plots/collocation_plots.py` | `Plotting`: collocation figures (`track_on_goes`, `curtain`, `voxels`) | plots |
| `plots/model_plots.py` | model figures: training history, scores, EarthCARE vs C3DIR curtains and profiles, column and region predictions | ml, plots |

"Needs" is the dependency group in `pyproject.toml` (`uv sync --extra scrape`, `--extra ml`,
`--extra plots`); `uv sync` alone installs everything for local work.

## Quick start

Each script has its settings at the top: edit them and run the file (from an IDE or
`uv run python scripts/<name>.py`). EarthCARE downloads need the one-time ESA MAAP login
setup (https://tropos-rsd.github.io/earthcarekit/setup/).

**1. Scrape and collocate:** `scripts/c3dir_runner.py`

Set `START` / `END` (UTC) to a time window. It finds EarthCARE frames GOES-19 can see,
downloads and matches them, and saves pairs to `data/collocated/`, plus example figures.
Cached, so re-running a window only fetches what's new. Run it for as many windows as you
want training data (a day of 18–20 UTC passes takes a few minutes).

**2. Train and evaluate:** `scripts/c3dir_train.py`

Builds samples from everything in `data/collocated/`, splits them by date, trains, and
saves the next version to `artifacts/models/c3dir/` (v001, v002, ...) with test scores and
figures. Main settings:

| Setting | Meaning |
|---|---|
| `VAL_AFTER`, `TEST_AFTER` | dates from which days go to validation / test; `None` splits ~80/10/10 by date |
| `MODEL` | network size, e.g. `ModelConfig(width=32)` (paper: 64) |
| `TRAINING` | steps, batch size, how often to check validation, e.g. `TrainConfig(steps=3000)` |
| `VERSION` | `None` for the next number, or a name like `"v003_wide"` |

Training runs on the GPU if there is one (CUDA, or Apple MPS); ~30 days of data and 3,000
steps take about an hour on a laptop.

**3. Predict:** `scripts/c3dir_predict.py`

Runs a saved model on a GOES-19 scan around a point:

| Setting | Meaning |
|---|---|
| `SITE` | a SURFRAD station (`bon`, `dra`, `fpk`, `gwn`, `psu`, `sxf`, `tbl`), or `None` to use `LAT` / `LON` |
| `TIME` | `"latest"` or a UTC time like `"2025-08-15T18:30"` |
| `MODEL_VERSION` | `"latest"` or e.g. `"v002"` |
| `REGION`, `HALF_WIDTH_DEG` | also predict the 3D field within ± this many degrees |

Outputs:
- `artifacts/figures/c3dir_predict/<site>_<time>.png` and `.csv`: the column above the
  point, per level: probabilities, water content (g m⁻³), particle size (µm), and voxel
  lat / lon.
- `<site>_<time>_region.png`: GOES true color and band 13, C3DIR cloud-top height, a 3D
  view, and a west–east cross section.
- `data/c3dir_predictions/<site>_<time>_region.nc`: the 3D field on a regular
  (height, lat, lon) grid.

The model was trained on daytime passes only, so the script warns when the sun is down.

From Python, the same steps are:

```python
from heimdall.c3dir.model.c3dir import C3DIR
from heimdall.c3dir.model.inference import goes_scene, predict_column, predict_region

model = C3DIR.load("latest")
scene = goes_scene("2025-08-17T18:30")
column = predict_column(model, scene, 40.72, -77.93)   # DataFrame over 80 levels
region = predict_region(model, scene, 40.72, -77.93)   # xarray (height, lat, lon)
```

## How matching works

For every EarthCARE voxel with a valid retrieval:
1. Find the GOES pixel whose line of sight passes closest to the voxel at its height.
2. Keep the match if that distance is ≤ 5 km and GOES scanned that pixel within
   2.5 minutes of the EarthCARE measurement.
3. If several EarthCARE voxels match the same GOES voxel, keep the closest.

So each GOES pixel gets labels at the heights where its line of sight crosses the
EarthCARE track. The model's vertical axis follows those slanted lines of sight.

## Key assumptions

| What | Value | Source |
|---|---|---|
| EarthCARE product | ACM_CAP_2B | paper |
| Targets | ice, liquid, rain water content; ice and liquid effective radius; ice, liquid, rain, any-hydrometeor masks | paper §2.2 |
| Vertical grid | 80 levels × 0.25 km from 0 to 20 km | paper §2.3 (the paper writes −0.125 to 19.875 km, but 80 levels only fit 0–20 km) |
| Cloud threshold | water content > 1e-5 g m⁻³ | paper §2.2 |
| GOES product | all 16 ABI bands at 2 km, full disk every 10 min | paper §2.1 |
| Matching | within 5 km of the line of sight at the same height, closest EarthCARE voxel per GOES voxel | paper §3 |
| Time window | ≤ 2.5 min between EarthCARE and the GOES scan of that pixel | paper §3.2 |
| Patch size | 64 × 64 pixels | paper §2.3 |
| Absent liquid/rain | treated as 0, not missing | an assumption, not directly from the paper |
| Bad retrievals | below ground or `quality_status >= 8` set to missing | an assumption, not directly from the paper |
| GOES view limit | within 65° of the point under GOES-19 | an assumption, not directly from the paper |
| Train / val / test split | by whole days: later date ranges held out for val and test (`val_after`, `test_after`), otherwise ~80/10/10 by date | the paper also holds out date ranges (§3.2) |

## Data

| Path | Contents | Size |
|---|---|---|
| `data/earthcare/labels/<frame>.nc` | EarthCARE labels for one frame: `(profile, level)` | ~10–15 MB |
| `data/collocated/<frame>_<scan>_<domain>.nc` | GOES crop `(y, x)`, matched profiles `(profile, level)` and matched voxels `(voxel)` | ~1–20 MB |
| `data/samples/<pair>.nc` | GOES image `(band, y, x)` plus the matched voxels | ~1–30 MB |
| `data/samples/splits.csv`, `band_stats.csv` | split per sample; per-band mean and std | tiny |

Raw EarthCARE frames (~50 MB) and GOES scans (~250 MB) are deleted after processing
unless `keep_raw` is set.

Each matched voxel stores its position (`voxel_level`, `voxel_row`, `voxel_col`), the
EarthCARE profile it came from, its distance to the line of sight, its time gap, and
its 9 labels (`voxel_labels`, in `TARGETS` order).

### Batches for the model

| Key | Shape | Meaning |
|---|---|---|
| `x` | (B, 16, 64, 64) | normalized GOES bands |
| `index` | (B, n, 3) | (level, row, col) of each labeled voxel |
| `y` | (B, n, 9) | labels for those voxels |
| `mask` | (B, n) | False for padding |

`B` is the batch size and `n` the most labeled voxels in any patch of the batch.

## Model

Following the paper (§4):
- **Network** (`model/network.py`): a U-Net of ConvNeXt blocks (5×5 kernels, 12.5% dropout,
  stochastic depth up to 25%) with a decoder that keeps a constant width, ending in a
  1×1 conv folded into 8 outputs × 80 levels per pixel: ice / liquid / rain masks,
  water contents and two particle sizes. Width 64 in the paper; set in `ModelConfig`.
- **Loss** (`C3DIR.loss`, paper eq. 1–2): weighted cross-entropy for the four masks plus
  squared error for water content (scaled as log10(1 + w / 5e-5)) and particle size,
  only at the labeled voxels.
- **Training** (`C3DIR.fit`): AdamW, cosine learning rate 1e-3 → 1e-6, random flips;
  keeps the weights with the lowest validation loss. Settings in `TrainConfig`.

### Versions

Each trained model is saved to `artifacts/models/c3dir/<version>/` (v001, v002, ... by
default, or a name you choose):

| File | Contents |
|---|---|
| `model.pt` | weights, model settings, input normalization |
| `config.json` | model and training settings, train / val files, git commit, date |
| `history.csv` | train and validation loss during training |
| `metrics.csv` | test scores |
| `figures/` | training history, detection by height, water content, curtain |

Load one with `C3DIR.load("v001")` or `C3DIR.load("latest")`. Models and figures are not
committed to git (`artifacts/` is ignored).

### Evaluation (paper §5.2)

- **Detection** of each species by voxel: F1, accuracy, true positive and true negative
  rates and balanced accuracy, overall and by height.
- **Water content:** mean absolute log error (MALE) and correlation, where both
  EarthCARE and C3DIR see cloud.
- **Curtain:** EarthCARE vs C3DIR along the track, each EarthCARE voxel compared with
  the GOES voxel whose line of sight passes closest.

## Open items

- **More data:** weeks to months of passes, day and night. With a few days, scores
  mostly show the pipeline works, not how good the model can be.
- **Scale:** the paper trains for 200,000 steps with batches of 1,024 on 4 GPUs.
- **Night:** the visible bands (1–6) are empty at night; zero them and add a sun-angle
  input, or train on infrared bands only.
- **Extra inputs from the paper:** weather-model temperature and humidity profiles and
  surface elevation (§2.1).
- **Extra labels from the paper:** NOAA cloud mask, cloud-top height and phase (§2.3).
