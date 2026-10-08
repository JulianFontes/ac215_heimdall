"""run data scraping, collocation, and plotting for c3dir:

  1. find the earthcare ACM_CAP_2B frames goes-19 can see in [START, END]
  2. download them and convert them to C3DIR labels
  3. collocate each frame with the goes-19 scans (2.5 min / nearest pixel rule)
  4. save the collocated pairs and plot one example (map, curtain, voxels)

change the settings below and run the file.

everything is cached, so a second run over the same window skips straight to the plots:
  <DATA_DIR>/earthcare/labels/   C3DIR labels, one .nc per frame
  <DATA_DIR>/collocated/         goes crop + matched labels, one .nc per frame x scan
  <PLOT_DIR>/                    the three example figures

requires the one-time ESA MAAP login setup for earthcarekit:
https://tropos-rsd.github.io/earthcarekit/setup/

note: example script; the training dataset(s) and model framework are in progress.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from heimdall.c3dir.earthcare_labels import EarthCareLabels
from heimdall.c3dir.plotting import Plotting
from heimdall.c3dir.scrape import scrape
from heimdall.paths import DATA_DIR, FIGURES_DIR

# settings: change these
START = "2025-07-09T18:00"  # UTC; the default window has 4 collocated pairs
END = "2025-07-09T20:00"
DOMAIN = "F"  # GOES "F" full disk (what the C3DIR paper uses, ~250 MB a scan) or "C" CONUS
KEEP_RAW = False  # keep the raw EarthCARE frames and GOES scans after processing
EXAMPLE = None  # which pair to plot: None picks the cloudiest, or an index from the printed list
N_VOXEL_COLUMNS = 40  # how many GOES pixels the voxel figure zooms in on
PLOT_DIR = FIGURES_DIR / "c3dir_example"


def cloudy_profiles(pair):
    """how many profiles in the pair have any cloud or rain at all."""
    return int((pair["hydrometeor_mask"].values == 1).any(axis=1).sum())


# steps 1-3: search, download, label and collocate (all cached)
print(f"scraping {START} to {END} (GOES domain {DOMAIN})")
pairs = scrape(START, END, DATA_DIR, DOMAIN, KEEP_RAW)
if not pairs:
    raise SystemExit("no collocated pairs in this window; try a longer one")

print(f"\n{len(pairs)} collocated pairs, {sum(p.sizes['profile'] for p in pairs)} profiles total, "
      f"saved in {DATA_DIR / 'collocated'}")
for i, pair in enumerate(pairs):
    print(f"  [{i}] {pair.attrs['orbit_and_frame']} × {pair.attrs['scan_start'][:19]}: "
          f"{pair.sizes['profile']} profiles, {cloudy_profiles(pair)} cloudy")

# step 4: plot one example pair, save the figures and show them
example = max(pairs, key=cloudy_profiles) if EXAMPLE is None else pairs[EXAMPLE]
frame = EarthCareLabels(DATA_DIR).load(example.attrs["orbit_and_frame"])  # cached, for context on the map
plotting = Plotting(PLOT_DIR)
plotting.track_on_goes(example, frame=frame)
plotting.curtain(example)
plotting.voxels(example, n_columns=N_VOXEL_COLUMNS)

print(f"\nexample figures for {example.attrs['orbit_and_frame']} × "
      f"{example.attrs['scan_start'][:19]} in {PLOT_DIR}:")
stem = f"{example.attrs['orbit_and_frame']}_{Path(example.attrs['scan_file']).stem}"
for path in sorted(PLOT_DIR.glob(f"{stem}*.png")):
    print(f"  {path.name}")
plt.show()
