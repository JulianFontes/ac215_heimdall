# C3DIR: 3D clouds from GOES-19, trained on EarthCARE

Goal: a model that takes a GOES-19 ABI image and predicts, for every pixel, an
80-level column of cloud properties (ice/liquid/rain water content, effective radii
and hydrometeor masks), with uncertainty. It follows the C3DIR setup of White et al.
(2026); section numbers below refer to that paper.

EarthCARE's ACM-CAP product measures those properties, but only along its ground
track, a ~1 km wide line. GOES-19 sees the whole disk every 10 min, but at a slant:
each pixel looks through the atmosphere along a tilted line of sight, so a cloud at
height z shows up z · tan(viewing angle) from its ground position (~2 km for a 10 km
cloud in the tropics, ~30 km at 60°N). Training therefore matches voxel to voxel
(paper §3.1): every EarthCARE voxel (profile, height) is paired with the GOES pixel
whose line of sight passes within 5 km of it at that height, in the scan taken within
2.5 min. The model's vertical axis follows those slanted lines of sight.

## How the pieces fit together

```
EarthCareLabels                                         earthcare_labels.py
  search frames → download .h5 → bin to 0.25 km  ─▶  data/earthcare/labels/<frame>.nc
                 │
                 ▼
GOES19.collocate()                                      goes19.py
  list scans → download → match voxels along lines of sight → crop
                                                    ─▶  data/collocated/<frame>_<scan>_<domain>.nc
                 │
                 ▼
Samples.build()  GOES image + matched voxel list       dataset.py
Samples.split()  train / val / test by date          ─▶ data/samples/<pair>.nc, splits.csv, band_stats.csv
                 │
                 ▼
C3DIRDataset(split) → collate                           dataset.py
  64×64 patches along the track, batched             ─▶ arrays for the model
                 │
                 ▼
C3DIR: fit / validate / predict                         planned: model.py, c3dir.py
```

`scrape(start, end)` in `scrape.py` runs the first two steps for a time window, and
`scripts/c3dir_runner.py` wraps that with example plots. Every step caches its output
on disk and skips work it has already done, so re-running only fetches what's new.

### `earthcare_labels.py`: `EarthCareLabels`

1. `search(start, end)` asks the ESA MAAP catalogue (through earthcarekit) for
   ACM_CAP_2B frames whose footprint crosses a circle around the point under GOES-19.
   It lists frames only, no download.
2. `load(frame)` downloads the frame's `.h5` the first time, then `_read` turns it into labels:
   - drops profiles GOES can't see (`visible`);
   - converts units: water content kg m⁻³ → g m⁻³, radii m → µm;
   - sets absent liquid/rain to 0 and invalid retrievals to NaN;
   - averages the native ~100 m levels into the 80 C3DIR levels (`_bin_mean`);
   - derives the four masks from the water content threshold.
3. The result is cached as `labels/<frame>.nc`. The raw `.h5` is deleted unless `keep_raw`.

`TARGETS` fixes the order of the 9 targets used everywhere downstream.

### `goes19.py`: `GOES19` and line-of-sight geometry

The geometry follows the ABI fixed-grid navigation in the GOES-R Product User Guide:
- `scan_angles(lat, lon, height)`: the GOES scan angles at which a point at that
  height is seen (matches `goes_proj` at height 0);
- `ray_point(x, y, height)` / `ray_latlon(...)`: where a pixel's line of sight
  crosses a given height;
- `satellite_frame(lat, lon, height)`: positions in the frame used for distances.

1. `scans(start, end)` lists GOES-19 MCMIP scans (via goes2go) that overlap a time window.
2. `collocate(labels)` takes one frame's labels and, for each scan that could overlap it:
   - downloads the scan and runs `_match`, which for every labelled EarthCARE voxel:
     - finds the GOES pixel whose line of sight passes closest at the voxel's height
       (`scan_angles` for a first guess, then the closest of the 5×5 pixels around it,
       since footprints are stretched diagonally at oblique views; `_closest_ray`);
     - keeps it if that distance is ≤ `max_km` (5 km), the pixel has data (band 13
       finite), and the pixel's row was scanned within 150 s of the EarthCARE time
       (`row_times`);
     - keeps a single EarthCARE voxel per GOES voxel, the closest;
   - crops the scan around every matched pixel, padded by `pad_px`.
3. Each pair is cached as `collocated/<frame>_<scan>_<domain>.nc`. The raw scan is
   deleted unless `keep_raw`. A scan with no matching voxels gets a `.skip` marker,
   so it isn't downloaded again.

### `scrape.py`: `scrape(start, end)`

Searches frames, then calls `EarthCareLabels.load` and `GOES19.collocate` for each
one. Returns the list of collocated pairs.

### `dataset.py`: `Samples`, `C3DIRDataset`, `collate`

**`Samples.build()`** turns each collocated pair into a training sample: the 16 bands
stacked into one `goes (band, y, x)` image, plus the matched voxel list.

**`Samples.split()`** assigns each sample to train / val / test by the date of its GOES
scan, then writes the per-band train mean/std used to normalize the inputs.

**`C3DIRDataset(split)`** reads `splits.csv`, keeps one split, and cuts 64×64 patches
centered at regular steps along each track. Each patch comes with every labelled voxel
inside it, as (level, row, col) coordinates in the model's output volume (paper §4).
It only needs numpy; it works with `torch.utils.data.DataLoader` as-is.

**`collate`** pads the voxel lists so patches can be stacked into a batch.

### `plotting.py`: `Plotting`

Figures for a collocated pair, saved to `out_dir` if one is given. Every heimdall figure
uses the shared style in `src/heimdall/plot_style.py` (Arial bold, 12 pt tick labels
and legends, 2 pt lines).
- `track_on_goes`: map of the EarthCARE track over GOES true color and band 13,
  colored by ACM-CAP cloud-top height.
- `curtain`: along-track view: what GOES saw of each profile (from the pixel matched
  to its cloud top) stacked over the ice/liquid/rain curtains.
- `voxels`: zoom on a stretch of matched pixels. Left: band 13 pixels with the
  EarthCARE ground track and the matched pixels outlined. Right (default): the
  EarthCARE curtain with those pixels' lines of sight drawn through it and the
  EarthCARE voxels matched to them; `rays=False` shows a numbered pixel × level grid
  instead. Top: the viewing geometry (Earth, GOES-19, line of sight, track) to scale.

## Key assumptions

| What | Value | Where from |
|---|---|---|
| EarthCARE product | ACM_CAP_2B (combined cloud/aerosol/precip) | paper |
| Targets | ice, liquid, rain water content; ice and liquid effective radius; ice, liquid, rain, any-hydrometeor masks | paper §2.2 |
| Vertical grid | 80 levels × 0.25 km, 0–20 km, centers 0.125…19.875 km; native ~100 m levels averaged into them | paper §2.3. The paper writes "−0.125 to 19.875 km", but its 80 levels and figure 2 only work with 0–20 km bins |
| Mask threshold | water content > 1e-5 g m⁻³ | paper §2.2 |
| No liquid/rain in ACM-CAP | fill values → 0 (absent, not missing) | an assumption, not directly from the paper |
| Invalid retrievals | `liquid_classification < 0` (e.g. below ground) or `quality_status >= 8` → NaN | an assumption, not directly from the paper |
| GOES-19 position | sub-satellite longitude −75.2° | GOES-East |
| What GOES can see | within 65° of Earth-center angle from the sub-satellite point (GOES viewing at ~73° from straight down at the edge) | an assumption, not directly from the paper |
| GOES product | ABI L2 MCMIP, all 16 bands on the 2 km grid; full disk (every 10 min) by default | paper §2.1 uses full disk at 2 km |
| Collocation | voxel to voxel along GOES lines of sight; EarthCARE voxel within 5 km horizontally of the line of sight at the same height | paper §3.1–3.2 |
| One label per GOES voxel | the closest EarthCARE voxel, no averaging | paper §3.2 |
| Time match | \|EarthCARE time − GOES pixel time\| ≤ 150 s, per matched pixel | paper §3.2 |
| GOES pixel time | rows spread evenly between scan start (north edge) and end (south edge) | approximates the north→south sweep of §3.2 |
| Pixel has data | band 13 finite (not space or a gap) | an assumption, not directly from the paper |
| Line-of-sight geometry | GOES-R Product User Guide navigation; a height h is handled by inflating the ellipsoid by h (centimetre-level error at cloud heights) | an assumption, not directly from the paper |
| Crop padding | `pad_px = 64` GOES pixels around the matched pixels | an assumption, not directly from the paper |
| Patch size | 64 × 64 GOES pixels | paper §2.3 |
| Splits | whole days to train / val / test, ~80/10/10 by an md5 hash of the date; optional `test_after` date for a held-out later period | an assumption, not directly from the paper (the paper holds out chosen date ranges, §3.2) |
| Input normalization | per-band mean/std over train; NaN (space, padding) → 0 after normalizing | an assumption, not directly from the paper |

## Data formats and shapes

### On disk

| Path | Format | Contents | Typical size |
|---|---|---|---|
| `data/earthcare/raw/` | ACM_CAP_2B `.h5` | raw frames; deleted after labelling unless `keep_raw` | ~50 MB per frame, transient |
| `data/goes/` | MCMIP `.nc` | raw scans; deleted after cropping unless `keep_raw` | ~250 MB per full-disk scan, transient |
| `data/earthcare/labels/<frame>.nc` | NetCDF | labels for every profile GOES can see in a frame | ~10–15 MB per frame |
| `data/collocated/<frame>_<scan>_<domain>.nc` | NetCDF | GOES crop + matched profiles + matched voxels | ~1–20 MB per pair |
| `data/collocated/*.skip` | text | scans checked and found to have no match | tiny |
| `data/samples/<pair>.nc` | NetCDF, image zlib-compressed | training sample | ~1–30 MB per pair |
| `data/samples/splits.csv` | CSV | file, date, frame, scan_start, n_voxels, split | tiny |
| `data/samples/band_stats.csv` | CSV | per-band train mean and std | tiny |

An EarthCARE frame is ~1/8 of an orbit. One frame usually yields one or two
collocated pairs, because the track spans the 10-min boundary between GOES scans.

### Labels: `data/earthcare/labels/<frame>.nc`

Dimensions: `profile` (one every ~1 km along the visible part of the frame, a few
thousand per frame) × `level` (80).

| Variable | Dims | Units / meaning |
|---|---|---|
| `iwc`, `lwc`, `rwc` | (profile, level) | g m⁻³; 0 = none, NaN = no valid retrieval |
| `ice_radius`, `liquid_radius` | (profile, level) | µm; NaN where that phase isn't present (most voxels) |
| `ice_mask`, `liquid_mask`, `rain_mask`, `hydrometeor_mask` | (profile, level) | 0/1; NaN = no valid retrieval |
| `time`, `lat`, `lon` | (profile,) | |
| `level` coordinate | (level,) | km, 0.125 … 19.875 |

### Collocated pair: `data/collocated/<frame>_<scan>_<domain>.nc`

Three independent sets of dimensions share one file: the GOES crop `(y, x)`, the full
label columns of the matched profiles `(profile, level)`, and the matched voxels
`(voxel)`.

- **Crop shape:** covers every matched pixel at every height, plus `pad_px` on each
  side. Crops are tall and narrow: up to ~1,000+ px north–south, ~130–300 px east–west.
- **Profiles:** those with at least one matched voxel; from a handful, when only the
  end of a frame falls in the scan's time window, up to a few thousand.
- **Voxels:** a GOES pixel can be matched at several heights, and one EarthCARE profile
  spreads over several pixels as height increases (more at steep viewing angles).
  Tens of thousands per pair are typical.

| Variable | Dims | Units / meaning |
|---|---|---|
| `CMI_C01`…`CMI_C06` | (y, x) | reflectance 0–1, meaningful in daylight only |
| `CMI_C07`…`CMI_C16` | (y, x) | brightness temperature, K |
| `pixel_time` | (y,) | estimated time each row was scanned |
| `x`, `y` coordinates | (x,), (y,) | GOES scan angles in radians, not lat/lon. `goes19.ray_latlon` / `scan_angles` convert |
| `goes_imager_projection` | () | projection attributes (satellite height, longitude, Earth shape) |
| all label variables above | (profile, …) | full columns of the matched profiles (for plots) |
| `voxel_level`, `voxel_row`, `voxel_col` | (voxel,) | where the voxel sits: level index, and the pixel in the crop |
| `voxel_profile` | (voxel,) | which matched profile it came from |
| `voxel_distance_km` | (voxel,) | horizontal distance to the line of sight, ≤ 5 km |
| `voxel_dt_seconds` | (voxel,) | EarthCARE time − GOES pixel time |
| `voxel_labels` | (voxel, target = 9) | the targets of the matched EarthCARE voxel, in `TARGETS` order |
| attributes | | `orbit_and_frame`, `scan_file`, `scan_start`, `scan_end`, `max_seconds`, `max_km` |

### Training sample: `data/samples/<pair>.nc`

The same crop as one image, plus the voxel variables above.

| Variable | Dims | Units / meaning |
|---|---|---|
| `goes` | (band = 16, y, x) | the GOES crop, bands `CMI_C01`…`CMI_C16` |
| `voxel_*` | (voxel, …) | as in the collocated pair |

### Batches from `C3DIRDataset` + `collate`

`C3DIRDataset(split, patch=64, stride=32)` centers one patch on every `stride`-th
matched pixel along the track. `B` is the batch size. `n` is the largest number of
labelled voxels in any patch of the batch; shorter lists are padded.

| Key | Shape | Units / meaning |
|---|---|---|
| `x` | (B, 16, P, P) | normalized bands; space and padding are 0 |
| `valid` | (B, P, P) | False for space / padding |
| `index` | (B, n, 3) | (level, row, col) of each labelled voxel in the output volume |
| `y` | (B, n, 9) | labels in physical units, `TARGETS` order; radii NaN where that phase is absent |
| `mask` | (B, n) | False for padding |

The arrays are numpy; wrap them with `torch.as_tensor` in the training loop. Target
transforms (e.g. log10 of water content) belong in the model or loss, not the dataset.

## Model plan (not written yet)

### Network: image in → slanted voxel column per pixel out

```
x (B, 16, P, P)
  → encoder: ConvNeXt stages, downsampling P → P/2 → … → P/16
  → decoder: upsample back to P with skip connections
  → head: 1×1 conv → (B, 14·80, P, P) → reshape (B, 14, 80, P, P)
```

The paper's C3DIR is an asymmetric U-Net of ConvNeXt blocks with a 1×1 conv and a
reshape to 80 levels (§4). Level k of pixel (row, col) means "height k along that
pixel's line of sight", matching how the labels were collocated. The head makes
14 × 80 numbers per pixel; they take on meaning only because the loss compares channel
`v·80 + k` with "variable v at level k", so the head size, the reshape order and the
label order must agree.

### Outputs with uncertainty: hurdle heads

The paper predicts values only. This plan adds uncertainty by splitting each variable
into "is it there" and "how much if so":

| Head | Per voxel | Loss | Trained on |
|---|---|---|---|
| presence: ice, liquid, rain, any | p = P(present), sigmoid | BCE with the 0/1 mask labels; weight rain up (rare) | all labelled voxels |
| amount: iwc, lwc, rwc | μ, log σ² of log10(content) | Gaussian NLL | voxels where that phase is present |
| radius: ice, liquid | μ, log σ² | Gaussian NLL | voxels where the radius isn't NaN |

That's 4 + 3×2 + 2×2 = 14 values per level. The expected water content is about
p · 10^μ, with p and σ as its uncertainty. Gaussian NLL learns σ without extra labels,
by raising it where the model tends to be wrong (aleatoric uncertainty). For model
uncertainty, train a deep ensemble (~5 seeds), or use MC dropout as a cheaper option.

### Training loop

```python
for batch in DataLoader(C3DIRDataset("train"), batch_size=B, shuffle=True, collate_fn=collate):
    out = model(x)                                   # (B, 14, 80, P, P): every voxel
    level, row, col = index.unbind(-1)               # (B, n) each
    pred = out[b_idx, :, level, row, col]            # (B, n, 14): only the labelled voxels
    loss = hurdle_loss(pred[mask], y[mask])          # padding and NaN labels ignored
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

The forward pass predicts every voxel, but only the labelled ones are scored (typically
a few percent of a patch, paper §2.3). Their gradients update the shared convolution
weights, which is how the model learns to fill in voxels EarthCARE never flew through.
At inference, run it on a full GOES image and keep `out`; columns normal to the surface
can be built afterwards by sampling each height from the pixel whose line of sight
crosses it there (paper §3.1).

### Planned code

| File | Class | Job |
|---|---|---|
| `model.py` | network | the U-Net + hurdle heads |
| `c3dir.py` | `C3DIR` | `fit(train, val)`, `validate(split)`, `predict(goes_scene)` → 3D field + uncertainty, `save`, `load`. No scraping or splitting |
| `scripts/c3dir_train.py` | | build samples → split → `C3DIR.fit` → validate |

Needs `torch` added to `pyproject.toml`.

## Open items

- **Data volume.** Training needs weeks to months of scraped passes, both day and
  night. Splits by date only become meaningful with many days.
- **Night.** Bands 1–6 carry no information at night. Either zero them and add a
  solar-zenith channel, or train an IR-only (bands 7–16) model.
- **Extra inputs from the paper (§2.1):** GFS 6-hour forecast temperature and humidity
  profiles, surface and tropopause fields, and surface elevation.
- **Auxiliary labels from the paper (§2.3):** NOAA cloud mask, cloud-top height and
  cloud-top phase, trained everywhere in the patch as weak extra supervision.
