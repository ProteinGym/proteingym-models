FROM python:3.12-slim-bookworm

# install git, curl, certificates, and a C compiler (needed by triton to
# JIT-compile its wrapper kernels at runtime)
RUN apt-get update && apt-get install -y git curl ca-certificates build-essential && rm -rf /var/lib/apt/lists/*

# https://docs.astral.sh/uv/guides/integration/docker/#available-images
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/program

# Copy and install dependencies
COPY README.md README.md
COPY LICENSE LICENSE
COPY MODEL_LICENSE.md MODEL_LICENSE.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
RUN uv sync --no-cache

# PoET-2 checkpoint and its non-commercial model licence, from Zenodo. -f so a
# refused download fails the build instead of baking an HTML error page as
# weights; flash-attn is deliberately absent (CUDA-only, cp310-only wheel) and
# the vendored package falls back to standard attention without it.
RUN mkdir -p /opt/program/data && \
    curl -fL --retry 3 --retry-delay 5 \
        "https://zenodo.org/records/16815645/files/poet-2.ckpt?download=1" \
        -o /opt/program/data/poet-2.ckpt && \
    curl -fL --retry 3 --retry-delay 5 \
        "https://zenodo.org/records/16815645/files/MODEL_LICENSE?download=1" \
        -o /opt/program/data/MODEL_LICENSE

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]
