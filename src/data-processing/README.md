# Data Processing

Download and preprocessing of the raw data into the files that `model-train` and `model-deploy`
read. Raw downloads go to `data/raw/<source>/`, training-ready data to `data/processed/<model>/`.

| Model | Folder | Status |
|---|---|---|
| C3DIR | [`c3dir/`](c3dir/README.md) | EarthCARE + GOES-19 download, labels, collocation, samples |
| GOES / C3DIR nowcasting | | not set up yet (GOES-19 ABI, ERA5) |
