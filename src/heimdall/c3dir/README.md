# C3DIR: 3D clouds from GOES-19, trained on EarthCARE

Goal: a model that takes a GOES-19 ABI image and predicts, for every pixel, an
80-level vertical column of cloud properties (ice/liquid/rain water content,
effective radii and hydrometeor masks), with uncertainty. It follows the C3DIR
setup of White et al. (2026); section numbers below refer to that paper.

EarthCARE's ACM-CAP product measures those columns, but only along its ground
track, a ~1 km wide line. GOES-19 sees the whole disk every 10 min, but only from
above. Training pairs the two: each EarthCARE profile is matched to the GOES pixel
above it, in the scan taken within 2.5 min. The model learns from those matched
pixels and is then applied to every pixel.

## How the pieces fit together

```
EarthCareLabels                                         earthcare_labels.py
  search frames → download .h5 → bin to 0.25 km  ─▶  data/earthcare/labels/<frame>.nc
                 │
                 ▼
GOES19.collocate()                                      goes19.py
  list scans → download → match space + time → crop ─▶ data/collocated/<frame>_<scan>_<domain>.nc
                 │
                 ▼
Samples.build()  profiles → one column per GOES pixel   dataset.py
Samples.split()  train / val / test by date          ─▶ data/samples/<pair>.nc, splits.csv, band_stats.csv
                 │
                 ▼
C3DIRDataset(split) → collate                           dataset.py
  128×128 patches along the track, batched           ─▶ arrays for the model
                 │
                 ▼
C3DIR (UNeXt): fit / validate / predict                 planned: model.py, c3dir.py
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

### `goes19.py`: `GOES19`

1. `scans(start, end)` lists GOES-19 MCMIP scans (via goes2go) that overlap a time window.
2. `collocate(labels)` takes one frame's labels and, for each scan that could overlap it:
   - downloads the scan and runs `_match`;
   - `_match` projects each profile's lat/lon into GOES x/y scan angles (`goes_proj`);
   - keeps profiles that land on the grid on a pixel with real data (band 13 finite);
   - estimates when that pixel's row was scanned (`row_times`) and keeps the profile
     only if that's within 150 s of the EarthCARE time;
   - crops the scan to a box around the kept profiles, padded by `pad_px`.
3. Each pair is cached as `collocated/<frame>_<scan>_<domain>.nc`. The raw scan is
   deleted unless `keep_raw`. A scan with no matching profiles gets a `.skip` marker,
   so it isn't downloaded again.

### `scrape.py`: `scrape(start, end)`

Searches frames, then calls `EarthCareLabels.load` and `GOES19.collocate` for each
one. Returns the list of collocated pairs.

### `dataset.py`: `Samples`, `C3DIRDataset`, `collate`

**`Samples.build()`** turns each collocated pair into a training sample:
- `track_pixels` finds the GOES pixel under each profile, with the same nearest-pixel
  rule as `collocate`.
- `pixel_columns` averages the profiles that share a pixel into one label column, and
  keeps their spread (`*_std`).
- The 16 bands are stacked into one `goes (band, y, x)` image.

**`Samples.split()`** assigns each sample to train / val / test by the date of its GOES
scan, then writes the per-band train mean/std used to normalize the inputs.

**`C3DIRDataset(split)`** reads `splits.csv`, keeps one split, and cuts fixed-size
patches centered at regular steps along each track. Each patch comes with every
labelled pixel inside it. It only needs numpy; it works with
`torch.utils.data.DataLoader` as-is.

**`collate`** pads the labelled pixels so patches can be stacked into a batch.

### `plotting.py`: `Plotting`

Figures for a collocated pair, saved to `out_dir` if one is given:
- `track_on_goes`: map of the track over GOES true color and band 13.
- `curtain`: along-track view, what GOES saw above each profile stacked over the
  ice/liquid/rain curtains.
- `voxels`: zoom on a stretch of matched pixels, showing the GOES pixel footprints
  and the 0.25 km levels.

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
| GOES product | ABI L2 MCMIP, all 16 bands on the 2 km grid; full disk (every 10 min) by default | paper uses full disk |
| Time match | \|EarthCARE time − GOES pixel time\| ≤ 150 s | paper §3.2 |
| GOES pixel time | rows spread evenly between scan start (north edge) and end (south edge) | approximates the north→south sweep of §3.2 |
| Space match | nearest GOES pixel to the profile's ground lat/lon; band 13 must be finite (not space) | an assumption, not directly from the paper |
| Parallax | not corrected; GOES sees a 20 km cloud top up to ~9 px from its ground position at the edge of the view. The model sees neighbouring pixels, so it can learn the shift | an assumption, not directly from the paper |
| Crop padding | `pad_px = 64` GOES pixels each side of the matched track | an assumption, not directly from the paper; limits patches to ~128 px |
| One label column per GOES pixel | profiles in the same pixel are averaged; their spread is kept as `*_std` | an assumption, not directly from the paper |
| Mask labels in samples | fraction of the pixel's profiles with that hydrometeor (soft 0–1 label) | an assumption, not directly from the paper |
| Splits | whole days to train / val / test, ~80/10/10 by an md5 hash of the date; optional `test_after` date for a held-out later period | an assumption, not directly from the paper; check if the paper defines one |
| Input normalization | per-band mean/std over train; NaN (space, padding) → 0 after normalizing | an assumption, not directly from the paper |

## Data formats and shapes

### On disk

| Path | Format | Contents | Typical size |
|---|---|---|---|
| `data/earthcare/raw/` | ACM_CAP_2B `.h5` | raw frames; deleted after labelling unless `keep_raw` | ~50 MB per frame, transient |
| `data/goes/` | MCMIP `.nc` | raw scans; deleted after cropping unless `keep_raw` | ~250 MB per full-disk scan, transient |
| `data/earthcare/labels/<frame>.nc` | NetCDF | labels for every profile GOES can see in a frame | ~10–15 MB per frame |
| `data/collocated/<frame>_<scan>_<domain>.nc` | NetCDF | GOES crop + the profiles matched to it | ~1–15 MB per pair |
| `data/collocated/*.skip` | text | scans checked and found to have no match | tiny |
| `data/samples/<pair>.nc` | NetCDF, image zlib-compressed | training sample | ~1–30 MB per pair |
| `data/samples/splits.csv` | CSV | file, date, frame, scan_start, n_pixels, split | tiny |
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

Two independent sets of dimensions share one file: the GOES crop `(y, x)` and the
matched labels `(profile, level)`.

- **Crop shape:** `y` covers the north–south extent of the matched track plus padding
  (up to ~1,000+ px). `x` covers its east–west extent plus 2 × `pad_px` (~130–300 px).
  So crops are tall and narrow.
- **Profiles:** only those that passed the time and space match. That ranges from a
  handful, when only the end of a frame falls in the scan's time window, up to a few
  thousand.

| Variable | Dims | Units / meaning |
|---|---|---|
| `CMI_C01`…`CMI_C06` | (y, x) | reflectance 0–1, meaningful in daylight only |
| `CMI_C07`…`CMI_C16` | (y, x) | brightness temperature, K |
| `pixel_time` | (y,) | estimated time each row was scanned |
| `x`, `y` coordinates | (x,), (y,) | GOES scan angles in radians, not lat/lon. `goes19.goes_proj` converts to/from lat/lon |
| `goes_imager_projection` | () | projection attributes (satellite height, longitude, Earth shape) |
| all label variables above | (profile, …) | matched profiles only |
| `dt_seconds` | (profile,) | EarthCARE time − GOES pixel time |
| attributes | | `orbit_and_frame`, `scan_file`, `scan_start`, `scan_end`, `max_seconds` |

### Training sample: `data/samples/<pair>.nc`

Same crop, with the profiles averaged into one column per GOES pixel. `pixel` counts
the distinct GOES pixels the track crosses, in track order. Each pixel holds ~2
profiles near the equator and ~5–6 near 60°, where GOES pixels are larger.

| Variable | Dims | Units / meaning |
|---|---|---|
| `goes` | (band = 16, y, x) | the GOES crop as one image, bands `CMI_C01`…`CMI_C16` |
| `row`, `col` | (pixel,) | position of each labelled pixel in the crop |
| `n_profiles` | (pixel,) | profiles averaged into it |
| `lat`, `lon`, `time`, `dt_seconds` | (pixel,) | averaged over those profiles |
| `iwc`, `lwc`, `rwc`, `ice_radius`, `liquid_radius` | (pixel, level) | mean over the pixel's profiles |
| same with `_std` | (pixel, level) | spread over the pixel's profiles; NaN if fewer than 2 had a value |
| `ice_mask`, `liquid_mask`, `rain_mask`, `hydrometeor_mask` | (pixel, level) | fraction of the pixel's profiles, 0–1 |

### Batches from `C3DIRDataset` + `collate`

`C3DIRDataset(split, patch=128, stride=64)` centers one patch on every `stride`-th
labelled pixel. `B` is the batch size. `n` is the largest number of labelled pixels in
any patch of the batch (typically ~100–150 for a 128 px patch); shorter ones are padded.

| Key | Shape | Units / meaning |
|---|---|---|
| `x` | (B, 16, P, P) | normalized bands; space and padding are 0 |
| `valid` | (B, P, P) | False for space / padding |
| `track_rc` | (B, n, 2) | row, col of each labelled pixel inside the patch |
| `y` | (B, n, 9, 80) | labels in physical units, order given by `dataset.TARGETS`; NaN = no label (incl. padding) |
| `y_std` | (B, n, 5, 80) | sub-pixel spread for the 3 water contents + 2 radii |
| `n_profiles` | (B, n) | 0 for padding |

The arrays are numpy; wrap them with `torch.as_tensor` in the training loop. Target
transforms (e.g. log10 of water content) belong in the model or loss, not the dataset.

## Model plan (not written yet)

### Network: UNeXt, image in → column per pixel out

```
x (B, 16, P, P)
  → encoder: conv stages, then tokenized-MLP stages, downsampling P → P/2 → … → P/16
  → decoder: upsample back to P with skip connections
  → head: 1×1 conv → (B, 14·80, P, P) → reshape (B, 14, 80, P, P)
```

The head makes 14 × 80 numbers per pixel. They take on meaning only because the loss
compares channel `v·80 + k` with "variable v at level k". So the head size, the reshape
order and the label order must agree. Levels are just channels to the network, so
vertical structure is learned from data. If profiles come out noisy, add a 1D conv
along the level axis or a smoothness penalty.

### Outputs with uncertainty: hurdle heads

Water content is zero in most voxels and spans 1e-5 to 1 g m⁻³ elsewhere, so each
variable is split into "is it there" and "how much if so":

| Head | Per voxel | Loss | Trained on |
|---|---|---|---|
| presence: ice, liquid, rain, any | p = P(present), sigmoid | BCE with the soft mask labels; weight rain up (rare) | all labelled voxels |
| amount: iwc, lwc, rwc | μ, log σ² of log10(content) | Gaussian NLL | voxels where that phase is present |
| radius: ice, liquid | μ, log σ² | Gaussian NLL | voxels where the radius isn't NaN |

That's 4 + 3×2 + 2×2 = 14 values per level. The expected water content is about
p · 10^μ, with p and σ as its uncertainty. Gaussian NLL learns σ without extra labels,
by raising it where the model tends to be wrong (aleatoric uncertainty). For model
uncertainty, train a deep ensemble (~5 seeds), or use MC dropout as a cheaper option.
`y_std` gives a measured sub-pixel spread to check whether σ is calibrated.

### Training loop

```python
for batch in DataLoader(C3DIRDataset("train"), batch_size=B, shuffle=True, collate_fn=collate):
    out = model(x)                          # (B, 14, 80, P, P): a column for every pixel
    pred = out[b_idx, :, :, rows, cols]     # (B, n, 14, 80): only where EarthCARE measured
    loss = hurdle_loss(pred, y)             # NaN labels and padding ignored
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

The forward pass predicts every pixel, but only the labelled pixels are scored. Their
gradients update the shared convolution weights, which is how the model learns to fill
in pixels EarthCARE never flew over. At inference, run it on a full GOES image and keep `out`.

### Planned code

| File | Class | Job |
|---|---|---|
| `model.py` | `UNeXt` | the network + hurdle heads |
| `c3dir.py` | `C3DIR` | `fit(train, val)`, `validate(split)`, `predict(goes_scene)` → 3D field + uncertainty, `save`, `load`. No scraping or splitting |
| `scripts/c3dir_train.py` | | build samples → split → `C3DIR.fit` → validate |

Needs `torch` added to `pyproject.toml`.

## Open items

- **Data volume.** Training needs weeks to months of scraped passes, both day and
  night. Splits by date only become meaningful with many days.
- **Night.** Bands 1–6 carry no information at night. Either zero them and add a
  solar-zenith channel, or train an IR-only (bands 7–16) model.
- **Extra input channels** to consider: lat/lon, satellite viewing angle, solar
  angle, land/sea mask. Check what the paper uses.
- **Patch size.** Patches bigger than ~128 px need a larger `pad_px` in `GOES19`, and
  the collocated pairs rebuilt.
- **Check against the paper:** split strategy, patch size, input channels, loss
  weights, and whether masks and water content are predicted separately.
- `Plotting.voxels` groups profiles by consecutive pixel; `dataset.pixel_columns`
  groups by unique pixel. The plot could reuse `pixel_columns`.
