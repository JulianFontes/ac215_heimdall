## Running Heimdall

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

All services are defined in [`docker-compose.yml`](docker-compose.yml). Run every command below from this (repo root) folder.

### Setup (once)
Copy the environment template and fill in any values you need (`.env` is git-ignored; see the comments in `.env.example`):
```bash
cp .env.example .env
```

### Build, start, stop
Everything goes through [`heimdall_docker.sh`](heimdall_docker.sh), a thin wrapper around `docker compose`.
Leave out the service name to act on all services in `docker-compose.yml`.
```bash
./heimdall_docker.sh build [service...]   # 1. build images (again after code or .env changes; add --no-cache for a clean rebuild)
./heimdall_docker.sh up    [service...]   # 2. start containers in the background
./heimdall_docker.sh down                 # 3. stop and remove containers (images are kept)
```

### Services
| Service | URL | Details |
|---|---|---|
| `frontend` | http://localhost:8080 | [frontend README](src/heimdall/heimdall-frontend/README.md) |

Each service's README covers its own data, configuration and local development setup.
