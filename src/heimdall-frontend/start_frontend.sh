#!/usr/bin/env bash
# Starts the Kepler.gl voxel PoC frontend. Safe to re-run: it only sets up what is missing.
set -euo pipefail

# Always run from the folder this script lives in (handles spaces in the iCloud path)
cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Node check
if ! command -v node >/dev/null 2>&1; then
  echo "Node.js not found. Install it with:  brew install node" >&2
  exit 1
fi
NODE_MAJOR=$(node -v | sed 's/^v//' | cut -d. -f1)
if [ "$NODE_MAJOR" -lt 18 ]; then
  echo "Node 18+ required (found $(node -v)). Try:  brew upgrade node" >&2
  exit 1
fi

# 2. npm needs legacy-peer-deps (react-palm has stale peers)
[ -f .npmrc ] || echo "legacy-peer-deps=true" > .npmrc

# 3. Keep node_modules out of iCloud sync (any path ending in .nosync is skipped by iCloud).
# Handles all three states: missing, already a symlink, or a real directory from an earlier npm install.
if [ -L node_modules ]; then
  :                                   # already the symlink we want
elif [ -d node_modules ]; then
  echo "Moving node_modules out of iCloud sync (node_modules.nosync)..."
  rmdir node_modules.nosync 2>/dev/null || true
  mv node_modules node_modules.nosync
  ln -s node_modules.nosync node_modules
else
  mkdir -p node_modules.nosync
  ln -s node_modules.nosync node_modules
fi

# 4. Install dependencies if needed
if [ ! -d node_modules/vite ]; then
  echo "Installing dependencies (first run takes about a minute)..."
  npm install --no-audit --no-fund
fi

# 5. Env file + token check
if [ ! -f .env ]; then
  cp .env.example .env 2>/dev/null || echo "VITE_MAPBOX_TOKEN=" > .env
fi
if ! grep -Eq '^VITE_MAPBOX_TOKEN=pk\.' .env; then
  echo "NOTE: no Mapbox token in .env. That is fine - the default basemap (dark-matter,"
  echo "      served by Carto via MapLibre) needs no token. A token only unlocks the"
  echo "      Mapbox-hosted styles (Dark / Light / Muted / Satellite) in the style picker."
fi

# 6. Sample data if missing
if [ ! -f public/data/points.csv ] || [ ! -f public/data/voxels.geojson ]; then
  echo "Generating sample data..."
  npm run data
fi

echo "Starting dev server at http://localhost:5173 (Ctrl+C to stop)"
exec npm run dev -- --open
