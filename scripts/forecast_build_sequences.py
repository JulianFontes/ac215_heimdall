"""run a trained C3DIR model on every daytime GOES-19 scan of the days it was trained on, around a
site, to make the time series the ConvLSTM forecaster learns from.

  1. read the training days from the C3DIR model (its config lists the sample files it saw)
  2. for each 10-min time with the sun up: download the GOES-19 scan, predict the 3D cloud field
     around the site, average it onto coarser levels, save one frame
  3. stack the frames into <DATA_DIR>/c3dir_forecast/<site>_sequence.nc

cached: frames already saved are skipped, so an interrupted run picks up where it stopped.
GOES downloads need no login. change the settings below and run the file.
"""
import pandas as pd

from heimdall.c3dir.model.c3dir import C3DIR
from heimdall.c3dir_forecast.sequences import build_frames, sequence_path, stack_frames, training_days

# settings: change these
C3DIR_VERSION = "latest"  # the C3DIR model whose output is forecast
SITE = "psu"  # a SURFRAD code (bon, dra, fpk, gwn, psu, sxf, tbl)
DAYS = None  # None: every day from the first to the last C3DIR training day; or e.g. ["2025-07-01", ...]
CADENCE = "10min"  # time between frames
DOMAIN = "C"  # GOES "C" CONUS (smaller downloads) or "F" full disk
HALF_WIDTH_DEG = 1.28  # region: +/- this many degrees of latitude around the site
STEP_DEG = 0.04  # grid spacing (~4 km)
LEVELS_EVERY = 4  # average every 4 of the 80 C3DIR levels: 20 levels of 1 km
MAX_ZENITH = 80  # skip times with the sun lower than this (C3DIR saw daytime only)

model = C3DIR.load(C3DIR_VERSION)
days = training_days(model) if DAYS is None else pd.to_datetime(DAYS)
print(f"C3DIR {model.version}: {len(days)} days, {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}")
paths = build_frames(model, SITE, days, cadence=CADENCE, domain=DOMAIN, half_width_deg=HALF_WIDTH_DEG,
                     step_deg=STEP_DEG, levels_every=LEVELS_EVERY, max_zenith=MAX_ZENITH)
sequence = stack_frames(SITE)
print(f"{len(paths)} frames -> {sequence_path(SITE)} {dict(sequence.sizes)}")
