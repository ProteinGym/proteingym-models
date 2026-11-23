FROM python:3.12-slim-bookworm

# clustalo is only used for indel + retrieval scoring
RUN apt-get update && apt-get install -y git curl ca-certificates build-essential clustalo && rm -rf /var/lib/apt/lists/*

# https://docs.astral.sh/uv/guides/integration/docker/#available-images
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/program

# Copy and install dependencies
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
RUN uv sync --no-cache

# fetch the default (Small) checkpoint (add sizes here to use another checkpoint_size)
RUN for size in Small; do \
        mkdir -p /opt/program/data/$size && \
        for f in config.json pytorch_model.bin; do \
            curl -fL https://huggingface.co/OATML-Markslab/Tranception_$size/resolve/main/$f \
                -o /opt/program/data/$size/$f || exit 1; \
        done; \
    done

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]