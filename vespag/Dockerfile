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

# VespaG's own head is 2.6 MB, so it is baked; the ESM-2 3B embedder it needs
# (~11 GB) is fetched from the Hub on first use instead, as esm2 does.
# Pinned to the commit the wrapper's architecture check was written against.
RUN mkdir -p /opt/program/data/vespag && \
    curl -fL --retry 3 --retry-delay 5 \
        https://raw.githubusercontent.com/jschlensok/vespag/main/model_weights/v2/esm2.pt \
        -o /opt/program/data/vespag/esm2.pt && \
    test -s /opt/program/data/vespag/esm2.pt

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]
