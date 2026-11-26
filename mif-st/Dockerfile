# Nothing compiles against CUDA at build time - torch's own prebuilt wheel
# supplies the runtime
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

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

# MIF-ST is two checkpoints: the inverse-folding GNN and the CARP-640M sequence
# model it transfers from. sequence_models fetches both through torch.hub's
# cache - including CARP unconditionally, even when the GNN is given as a local
# path - so pre-populate that cache and point TORCH_HOME at it to keep the
# model offline at run time.
ENV TORCH_HOME=/opt/program/data/torch
RUN mkdir -p $TORCH_HOME/hub/checkpoints && cd $TORCH_HOME/hub/checkpoints && \
    curl -fL --retry 3 --retry-delay 5 \
        "https://zenodo.org/record/6573779/files/mifst.pt?download=1" -o mifst.pt && \
    curl -fL --retry 3 --retry-delay 5 \
        "https://zenodo.org/record/6564798/files/carp_640M.pt?download=1" -o carp_640M.pt && \
    [ "$(stat -c%s mifst.pt)" -gt 10000000 ] && \
    [ "$(stat -c%s carp_640M.pt)" -gt 2000000000 ]

# proves both checkpoints are intact and resolve from the cache, not the network
RUN uv run --no-sync python -c "\
from sequence_models.pretrained import load_model_and_alphabet; \
from proteingym.models.mif_st.model import MIFST; \
load_model_and_alphabet('mifst'); print('MIF-ST loads from cache OK')"

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]
