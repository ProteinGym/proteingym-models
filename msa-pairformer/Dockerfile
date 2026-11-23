FROM python:3.12-slim-bookworm

# install git, curl, certificates, and a C compiler (needed by triton to
# JIT-compile its CUDA wrapper kernels at runtime)
RUN apt-get update && apt-get install -y git curl ca-certificates build-essential && rm -rf /var/lib/apt/lists/*

# https://docs.astral.sh/uv/guides/integration/docker/#available-images
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/program

# Copy and install dependencies
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY patch_cuequivariance.py patch_cuequivariance.py
COPY src/ src/
RUN uv sync --no-cache

# upstream 1.0.2 binds CUEQUIVARIANCE_AVAILABLE only when a GPU is present,
# so a CPU-only import raises NameError; bind the default before the try block
RUN uv run --no-sync python patch_cuequivariance.py

# fetch weights in the HF hub cache layout that from_pretrained expects
RUN mkdir -p /opt/program/data && \
    uv run --no-sync hf download yakiyama/MSA-Pairformer --cache-dir /opt/program/data

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]
