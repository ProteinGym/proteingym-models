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
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
RUN uv sync --no-cache

# official progen2-small checkpoint (561 MB, small enough to bake); -f so a
# refused download fails the build instead of saving an error page as weights
RUN mkdir -p /opt/program/data && \
    curl -fL --retry 3 --retry-delay 5 \
        https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz \
        -o /tmp/progen2-small.tar.gz && \
    mkdir -p /opt/program/data/progen2-small && \
    tar xzf /tmp/progen2-small.tar.gz -C /opt/program/data/progen2-small && \
    rm /tmp/progen2-small.tar.gz && \
    test -s /opt/program/data/progen2-small/pytorch_model.bin

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]
