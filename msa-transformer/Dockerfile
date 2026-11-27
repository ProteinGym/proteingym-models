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
COPY .python-version .python-version
COPY src/ src/
RUN uv sync --no-cache

# ESM-MSA-1b checkpoint. fair-esm's local loader requires the contact-regression
# weights alongside it (every model except ESM-1v and ESM-IF), so both are baked.
RUN mkdir -p /opt/program/data && \
    URL=https://dl.fbaipublicfiles.com/fair-esm && \
    curl -fL --retry 3 --retry-delay 5 $URL/models/esm_msa1b_t12_100M_UR50S.pt \
        -o /opt/program/data/esm_msa1b_t12_100M_UR50S.pt && \
    curl -fL --retry 3 --retry-delay 5 $URL/regression/esm_msa1b_t12_100M_UR50S-contact-regression.pt \
        -o /opt/program/data/esm_msa1b_t12_100M_UR50S-contact-regression.pt

RUN uv run --no-sync python -c "from proteingym.models.msa_transformer import model; print('msa_transformer.model import OK')"

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]
