# Heimdall data-processing: collects raw data and turns it into training data (no torch).
# Build from src/ so heimdall-core (a path dependency) is in the build context:
#   docker build -f src/data-processing/Dockerfile -t heimdall-data-processing src
# Settings are environment variables (START, END, PUSH, ...; see c3dir/collect.py and c3dir/process.py).
# Credentials are never baked in: locally they are mounted (see docker-compose.yml), on GCP they come
# from the job's service account and Secret Manager.

FROM python:3.14.6-slim-trixie

# uv, the same version used to write the lock files
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /bin/uv

# compile .pyc at install for faster start; copy files instead of linking; use this image's python
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app/src/data-processing

# dependencies first, from the lock file only, so this layer is cached until uv.lock changes
COPY heimdall-core/pyproject.toml heimdall-core/uv.lock /app/src/heimdall-core/
COPY data-processing/pyproject.toml data-processing/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-install-package heimdall-core

# then the code: heimdall-core (utilities, shared c3dir modules) and data-processing's c3dir
COPY heimdall-core /app/src/heimdall-core
COPY data-processing ./
RUN uv sync --frozen --no-dev

# data stays inside the container (or a mounted volume) and is pushed to the bucket at the end
ENV PATH="/app/src/data-processing/.venv/bin:$PATH" \
    HEIMDALL_DATA_DIR=/data \
    HEIMDALL_ARTIFACTS_DIR=/artifacts \
    PUSH=true

# collect the window, then process everything collected
CMD ["sh", "-c", "python -m c3dir.collect && python -m c3dir.process"]
