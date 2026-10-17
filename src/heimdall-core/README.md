# heimdall-core

Light shared code, without torch, so services that don't run models (like `data-processing`)
stay small. It provides these modules of the `c3dir` package, which `model-core`, `model-train`,
`model-deploy` and `data-processing` extend with their own:

| File | What it does |
|---|---|
| `c3dir/constants.py` | C3DIR levels, bands, target names and order |
| `c3dir/geometry.py` | GOES-19 line-of-sight geometry (`scan_angles`, `ray_point`, `ray_latlon`, `closest_ray`, `within_view`) |
| `c3dir/paths.py` | the data folders every service reads and writes (`data/raw`, `data/processed`, `data/cache`, `data/predictions`, `artifacts/models`, `artifacts/figures`) |
| `c3dir/plot_style.py` | shared figure style (Arial bold) and colors; needs the `plots` extra (`heimdall-core[plots]`) |

Services install it from `../heimdall-core` (see their `pyproject.toml`). See
[`docs/c3dir.md`](../../docs/c3dir.md) for the whole C3DIR pipeline.
