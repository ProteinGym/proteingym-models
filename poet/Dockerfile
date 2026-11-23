# devel image for nvcc (flash-attn may build from source); CUDA must match torch in uv.lock
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

# C/C++ host compiler for nvcc
RUN apt-get update && apt-get install -y git curl ca-certificates build-essential && rm -rf /var/lib/apt/lists/*

# https://docs.astral.sh/uv/guides/integration/docker/#available-images
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# base image ships 3.10; pin 3.12 so a prebuilt flash-attn wheel exists
ENV UV_PYTHON=3.12
# keep the managed Python out of /root so non-root users can run it
ENV UV_PYTHON_INSTALL_DIR=/opt/python
# bound flash-attn's from-source build parallelism (each nvcc job uses several GB)
ENV MAX_JOBS=4
# no GPU visible at build time: 8.9 = L40S, 9.0 = H100
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"

WORKDIR /opt/program

COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
RUN uv sync --no-cache

# PoET weights and their license from Zenodo
RUN mkdir -p /opt/program/data && \
    curl -L "https://zenodo.org/records/10061322/files/poet.ckpt?download=1" \
        -o /opt/program/data/poet.ckpt && \
    curl -L "https://zenodo.org/records/10061322/files/LICENSE?download=1" \
        -o /opt/program/data/LICENSE

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]