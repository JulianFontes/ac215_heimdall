"""scrape goes-19 / earthcare pairs collocated in time and space for C3DIR.

outputs are cached, so re-running a range only fetches what's new
(<data-dir> defaults to data/ at the repo root):
  <data-dir>/earthcare/raw/      ACM_CAP_2B frames (.h5)
  <data-dir>/earthcare/labels/   C3DIR labels, one .nc per frame
  <data-dir>/collocated/         GOES crop + matched labels, one .nc per frame x scan
  <plot-dir>/                    GOES map + EarthCARE curtain figures with --plot
                                 (defaults to artifacts/figures)

example:
  uv run python -m heimdall.c3dir.scrape --start 2025-07-09T18:00 --end 2025-07-09T20:00 --plot
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt

from heimdall.c3dir.earthcare_labels import EarthCareLabels
from heimdall.c3dir.goes19 import GOES19
from heimdall.c3dir.plotting import Plotting
from heimdall.paths import DATA_DIR, FIGURES_DIR


def scrape(start, end, data_dir=DATA_DIR, domain="F", keep_raw=False, plot_dir=None):
    """collocate every ACM_CAP_2B frame in [start, end] that goes-19 can see; returns the pairs."""
    earthcare = EarthCareLabels(data_dir, keep_raw=keep_raw)
    goes = GOES19(data_dir, domain=domain, keep_raw=keep_raw)
    plotting = Plotting(plot_dir) if plot_dir else None
    # find the earthcare frames, then the goes scans matching each one
    frames = earthcare.search(start, end)
    print(f"{len(frames)} ACM_CAP_2B frames in the GOES-19 view")
    pairs = []
    for frame in frames.get("orbit_and_frame", []):
        labels = earthcare.load(frame)
        collocated = goes.collocate(labels)
        print(f"{frame}: {labels.sizes['profile']} visible profiles, {len(collocated)} collocated scans")
        for pair in collocated:
            if plotting:
                plt.close(plotting.track_on_goes(pair, frame=labels))  # close to free memory
                plt.close(plotting.curtain(pair))
        pairs += collocated
    return pairs


def main():
    matplotlib.use("Agg")  # save figures to file without opening windows
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="UTC start, e.g. 2025-07-09T18:00")
    parser.add_argument("--end", required=True, help="UTC end")
    parser.add_argument("--data-dir", default=DATA_DIR)
    # full disk is what the C3DIR paper uses
    parser.add_argument("--domain", default="F", choices=["F", "C"],
                        help="GOES full disk (C3DIR default) or CONUS")
    parser.add_argument("--keep-raw", action="store_true",
                        help="keep raw EarthCARE frames and GOES scans after processing")
    parser.add_argument("--plot", action="store_true",
                        help="save a GOES map and an EarthCARE curtain figure per pair to --plot-dir")
    parser.add_argument("--plot-dir", default=FIGURES_DIR)
    args = parser.parse_args()
    scrape(args.start, args.end, args.data_dir, args.domain, args.keep_raw,
           args.plot_dir if args.plot else None)


if __name__ == "__main__":
    main()
