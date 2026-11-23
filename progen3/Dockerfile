# devel image: CUDA extensions build from source with nvcc matching torch's CUDA
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git curl ca-certificates build-essential && rm -rf /var/lib/apt/lists/*

# https://docs.astral.sh/uv/guides/integration/docker/#available-images
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# bound nvcc memory; L40S/H100 archs (no GPU at build time); force CUDA extension builds
ENV MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="8.9;9.0" \
    CUDA_HOME=/usr/local/cuda \
    UV_PYTHON_INSTALL_DIR=/opt/python

WORKDIR /opt/program

COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY vendor/ vendor/
# megablocks/grouped-gemm build without isolation, so torch must be installed first
RUN uv venv --python 3.12 && \
    uv pip install --no-cache setuptools torch==2.13.0 && \
    uv sync --no-cache && \
    uv run --no-sync python -c "import torch, megablocks_ops, grouped_gemm_backend"

RUN mkdir -p /opt/program/data && \
    uv run --no-sync hf download Profluent-Bio/progen3-112m \
        --local-dir /opt/program/data/progen3-112m

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]