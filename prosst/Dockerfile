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
COPY vendor/ vendor/
RUN uv sync --no-cache

# structure quantizer weights (GVP encoder + 2048-cluster k-means codebook)
RUN STATIC=/opt/program/src/proteingym/models/prosst/structure/static && \
    URL=https://github.com/ai4protein/ProSST/raw/13bd56495641a84cd5475eaef7aa98b14fdc6ed5/prosst/structure/static && \
    mkdir -p $STATIC && \
    curl -fL $URL/AE.pt -o $STATIC/AE.pt && \
    curl -fL $URL/2048.joblib -o $STATIC/2048.joblib

# ProSST-2048 checkpoint (incl. trust_remote_code modeling files) for offline use
RUN mkdir -p /opt/program/data && \
    uv run --no-sync hf download AI4Protein/ProSST-2048 \
        --revision e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8 \
        --local-dir /opt/program/data/ProSST-2048

# trust_remote_code copies modeling files into HF_MODULES_CACHE, which must be writable
ENV HF_HUB_OFFLINE=1 \
    HF_MODULES_CACHE=/tmp/hf_modules_cache

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]