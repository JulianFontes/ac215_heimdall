# heimdall-core

Light shared code, without torch, so services that don't run models (like `data-processing`)
stay small. It has two packages:

- `utilities`: what every model and service uses (data folders, the shared bucket, figure style)
- `c3dir`: C3DIR constants and geometry, which `model-core`, `model-train`, `model-deploy` and
  `data-processing` extend with their own `c3dir` modules

| File | What it does |
|---|---|
| `c3dir/constants.py` | C3DIR levels, bands, target names and order |
| `c3dir/geometry.py` | GOES-19 line-of-sight geometry (`scan_angles`, `ray_point`, `ray_latlon`, `closest_ray`, `within_view`) |
| `utilities/paths.py` | the data folders every service reads and writes (`data/raw`, `data/processed`, `data/cache`, `data/predictions`, `artifacts/models`, `artifacts/figures`) |
| `utilities/bucket.py` | the shared GCS bucket, same layout as `paths.py` (`pull`, `push`, `folders`); needs the `gcs` extra (`heimdall-core[gcs]`) |
| `utilities/sync.py` | script: pull and push bucket folders (settings at the top) |
| `utilities/plot_style.py` | shared figure style (Arial bold) and colors; needs the `plots` extra (`heimdall-core[plots]`) |

Services install it from `../heimdall-core` (see their `pyproject.toml`) and import e.g.
`from utilities.bucket import pull` or `from utilities.paths import RAW_DIR`. See
[`docs/c3dir.md`](../../docs/c3dir.md) for the whole C3DIR pipeline.
