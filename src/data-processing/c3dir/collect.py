"""collect the raw data C3DIR trains on, for a time window:

  1. find the earthcare ACM_CAP_2B frames goes-19 can see in [START, END]
  2. download them to RAW_DIR/earthcare/
  3. for each frame, download the goes-19 scans within 2.5 min of the overpass and keep the strip
     around the track, in RAW_DIR/goes/

everything is cached, so re-running a window only fetches what's new. with PUSH, the new raw files
are copied to the shared bucket at the end. next, run c3dir.process to collocate and build training
samples.

change the settings below and run from the repo root (.env names the bucket):

  uv run --project src/data-processing --env-file .env python -m c3dir.collect

each setting can also be set by an environment variable of the same name (e.g. START), which is
how the data-processing container is configured.

requires the one-time ESA MAAP login setup for
earthcarekit: https://tropos-rsd.github.io/earthcarekit/setup/
"""
from c3dir.earthcare import EarthCareFrames
from c3dir.goes import GOESStrips
from utilities.bucket import push
from utilities.paths import RAW_DIR
from utilities.settings import setting

# settings: change these
START = setting("START", "2025-07-09T18:00")  # UTC
END = setting("END", "2025-07-09T20:00")
DOMAIN = setting("DOMAIN", "F")  # GOES "F" full disk (what the C3DIR paper uses, ~250 MB a scan) or "C" CONUS
KEEP_FULL = setting("KEEP_FULL", False)  # keep the full goes scans in the cache after cutting the strips
PUSH = setting("PUSH", True)  # bucket setting: copy new raw files to the shared bucket at the end (only new or changed files)

earthcare = EarthCareFrames()
goes = GOESStrips(domain=DOMAIN, keep_full=KEEP_FULL)
frames = earthcare.search(START, END)
print(f"collecting {START} to {END}: {len(frames)} ACM_CAP_2B frames in the GOES-19 view")
for frame in frames.get("orbit_and_frame", []):
    path = earthcare.download(frame)
    profiles = earthcare.profiles(path)
    strips = goes.collect(frame, profiles)
    print(f"  {frame}: {len(profiles)} visible profiles, {len(strips)} goes strips")
print(f"raw data in {RAW_DIR}")
if PUSH:
    push("raw/earthcare")
    push("raw/goes")
