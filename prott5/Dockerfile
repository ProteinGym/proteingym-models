FROM python:3.12-slim-bookworm

# install git, curl, certificates, and a C compiler (needed by triton to
# JIT-compile its wrapper kernels at runtime)
RUN apt-get update && apt-get install -y git curl ca-certificates build-essential && rm -rf /var/lib/apt/lists/*

# https://docs.astral.sh/uv/guides/integration/docker/#available-images
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/program

# Copy and install dependencies
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
RUN uv sync --no-cache

# The ProtT5-XL encoder is ~2 GB, fetched from the Hub on first use rather than
# baked into the image (same as esm2/esm1b).

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]
