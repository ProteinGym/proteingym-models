FROM python:3.12-slim-bookworm

# C/C++ compiler for sdists in the lock that build from source
RUN apt-get update && apt-get install -y git curl ca-certificates build-essential && rm -rf /var/lib/apt/lists/*

# https://docs.astral.sh/uv/guides/integration/docker/#available-images
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/program

COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
RUN uv sync --no-cache

# LLM goes to the card's hf_cache_dir; the structure encoder is only found via HF_HOME
RUN mkdir -p /opt/program/data && \
    uv run --no-sync hf download genbio-ai/AIDO.Protein-RAG-16B-proteingym-dms-zeroshot \
        --cache-dir /opt/program/data/hf_cache && \
    HF_HOME=/opt/program/data/hf_home uv run --no-sync hf download genbio-ai/AIDO.StructureEncoder

# structure-encoder cache baked in above
ENV HF_HOME=/opt/program/data/hf_home
# offline: use the baked caches and avoid writing .locks/ to a read-only filesystem
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
# trust_remote_code writes the checkpoint's .py modules here; keep it writable
ENV HF_MODULES_CACHE=/tmp/hf_modules_cache
# deepspeed is only imported (never compiles ops); skip its CUDA toolkit check
ENV DS_IGNORE_CUDA_DETECTION=1

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]