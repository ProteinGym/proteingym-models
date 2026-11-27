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
COPY patch_upstream.py patch_upstream.py
COPY src/ src/
RUN uv sync --no-cache

# upstream's Trainer is GPU-only and its on-the-fly embedding path is broken for
# ProteinNPT; both are patched here and fail loudly if upstream changes
RUN uv run --no-sync python patch_upstream.py

# frozen ESM-2 650M embedder. Upstream loads it with its own vendored ESM, so it
# must be the fair-esm .pt rather than a HuggingFace checkpoint
RUN mkdir -p /opt/program/data && \
    curl -fL --retry 3 --retry-delay 5 \
        https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt \
        -o /opt/program/data/esm2_t33_650M_UR50D.pt

# upstream imports wandb unconditionally; args.use_wandb is false, and this
# keeps it from reaching the network even if that changes
ENV WANDB_MODE=disabled \
    WANDB_SILENT=true \
    HF_HUB_OFFLINE=1

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]
