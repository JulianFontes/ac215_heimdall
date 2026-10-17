#!/usr/bin/env bash
# Runs Heimdall with Docker (services are defined in docker-compose.yml).
# Leave out the service name to run on all container services.
#
# Usage:
#   ./heimdall_docker.sh build [service...]   # build images
#   ./heimdall_docker.sh up    [service...]   # build if changed, start in the background
#   ./heimdall_docker.sh down                 # stop and remove containers

# Exit on any error (-e), on unset variables (-u), and if any command in a pipe fails (pipefail)
set -euo pipefail

USAGE="Usage: ./heimdall_docker.sh {build|up|down} [service...]"

cd "$(dirname "${BASH_SOURCE[0]}")" # Always run from the repo root (the folder this script lives in)

# Make sure Docker Desktop is running before doing anything
if ! docker info >/dev/null 2>&1; then # docker info fails if the Docker daemon is not reachable
  echo "Docker is not running. Start Docker Desktop and try again." >&2 # >&2 prints to stderr
  exit 1 # non-zero exit code means failure
fi

# A command (build/up/down) is required
if [ $# -eq 0 ]; then # $# is the number of arguments passed to the script
  echo "$USAGE" >&2 # show how to use the script
  exit 1 # stop: nothing to do
fi

# The first argument is the command; whatever follows (service names) is left in "$@"
command="$1" # save the command, e.g. "up"
shift # drop the command so "$@" holds only the service names (empty means all services)

if [ "$command" = "build" ]; then
  docker compose build "$@" # build the images

elif [ "$command" = "up" ]; then
  docker compose up -d "$@" # build if changed if not then bring up the containers (-d runs them in the background)

  # Print the URL of each running service (only those that are actually up)
  running=$(docker compose ps --services --status running) # names of the running services, one per line
  if echo "$running" | grep -qx frontend; then # -x matches the whole line, -q prints nothing
    echo "Frontend running at: http://localhost:8080" # port comes from "8080:80" in docker-compose.yml
  fi

elif [ "$command" = "down" ]; then
  docker compose down "$@" # stop and remove the containers (images are kept)

else
  # error message for unknown command 
  echo "Unknown command: $command" >&2 # the command was not build, up or down
  echo "$USAGE" >&2 # show how to use the script
  exit 1 # non-zero exit code means failure
fi
