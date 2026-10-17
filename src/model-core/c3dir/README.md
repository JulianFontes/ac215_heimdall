# C3DIR (model-core)

The C3DIR model (3D clouds from GOES-19, trained on EarthCARE; White et al., 2026), shared by
`model-train` and `model-deploy`:

| File | What it does |
|---|---|
| `network.py` | `C3DIRNet`: U-Net of ConvNeXt blocks, 8 outputs × 80 levels per pixel |
| `model.py` | `C3DIR`: prediction (`predict_voxels`), `save` and `load` of versioned models |

The torch-free shared modules (constants, geometry, data folders, figure style) are in
[`heimdall-core`](../../heimdall-core/README.md), so `data-processing` can use them without torch.
See [`docs/c3dir.md`](../../../docs/c3dir.md) for the whole pipeline.
