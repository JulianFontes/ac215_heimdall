"""turn the raw data from c3dir.collect into C3DIR training data:

  1. convert each earthcare frame to C3DIR labels (PROCESSED_DIR/c3dir/labels/)
  2. collocate every new goes strip with its frame, voxel by voxel along goes lines of sight
     (within 5 km and 2.5 min, C3DIR paper section 3) -> PROCESSED_DIR/c3dir/collocated/
  3. plot one example pair from [START, END] (map, curtain, voxels)
  4. build training samples and split them by date into train / val / test
     -> PROCESSED_DIR/c3dir/samples/

everything is cached, so a second run only processes what's new. with PULL, processed/c3dir is first
copied from the shared bucket (replacing local files that differ), so samples are built from all
processed days; with PUSH, the new processed files are copied to the bucket at the end. collect data first with c3dir.collect; train next with
model-train's c3dir.train. change the settings below and run from the repo root (.env names the
bucket):

  uv run --project src/data-processing --env-file .env python -m c3dir.process

each setting can also be set by an environment variable of the same name (e.g. START), which is
how the data-processing container is configured.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from c3dir.collocation import Collocation
from c3dir.labels import EarthCareLabels
from c3dir.plots import Plotting
from c3dir.samples import Samples
from utilities.bucket import pull, push
from utilities.paths import FIGURES_DIR
from utilities.settings import setting

# settings: change these
START = setting("START", "2025-07-09T18:00")  # UTC window to list and plot an example from
END = setting("END", "2025-07-09T20:00")
EXAMPLE = None  # which pair to plot: None picks the cloudiest, or an index from the printed list
N_VOXEL_COLUMNS = 40  # how many GOES pixels the voxel figure zooms in on
VOXEL_RAYS = True  # voxel figure: GOES lines of sight through the curtain (False: pixel x level grid)
PLOT_DIR = FIGURES_DIR / "c3dir_example"
VAL_AFTER = setting("VAL_AFTER", None)  # date: days from here go to validation; None splits ~80/10/10 by date
TEST_AFTER = setting("TEST_AFTER", None)  # date: days from here go to test
PULL = setting("PULL", False)  # bucket setting: copy processed/c3dir from the shared bucket first; replaces local files that differ
PUSH = setting("PUSH", True)  # bucket setting: copy new processed files to the shared bucket at the end (only new or changed files)


def cloudy_profiles(pair):
    """how many profiles in the pair have any cloud or rain at all."""
    return int((pair["hydrometeor_mask"].values == 1).any(axis=1).sum())


# steps 1-2: labels and collocation for everything collected (cached); list the window's pairs
if PULL:
    pull("processed/c3dir")
labels = EarthCareLabels()
collocation = Collocation(labels)
collocation.run()
pairs = [xr.load_dataset(path) for path in collocation.run(START, END)]
print(f"{len(pairs)} collocated pairs from {START} to {END} in {collocation.out_dir}")
for i, pair in enumerate(pairs):
    print(f"  [{i}] {pair.attrs['orbit_and_frame']} × {pair.attrs['scan_start'][:19]}: "
          f"{pair.sizes['profile']} profiles ({cloudy_profiles(pair)} cloudy), {pair.sizes['voxel']} voxels")

# step 3: plot one example pair
if pairs:
    example = max(pairs, key=cloudy_profiles) if EXAMPLE is None else pairs[EXAMPLE]
    frame = labels.load(example.attrs["orbit_and_frame"])  # cached, for context on the map
    plotting = Plotting(PLOT_DIR)
    plt.close(plotting.track_on_goes(example, frame=frame))
    plt.close(plotting.curtain(example))
    plt.close(plotting.voxels(example, n_columns=N_VOXEL_COLUMNS, rays=VOXEL_RAYS))
    stem = f"{example.attrs['orbit_and_frame']}_{Path(example.attrs['scan_file']).stem}"
    print(f"example figures in {PLOT_DIR}: " + ", ".join(p.name for p in sorted(PLOT_DIR.glob(f"{stem}*.png"))))

# step 4: training samples and the split
samples = Samples()
samples.build()
samples.split(val_after=VAL_AFTER, test_after=TEST_AFTER)

# step 5: share the processed data
if PUSH:
    push("processed/c3dir")
