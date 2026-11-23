# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

# build-essential: KeOps JIT-compiles its CPU kernels with g++.
# libxrender1/libxext6: torchdrug imports rdkit.Chem.Draw at load time.
RUN apt-get update && apt-get install -y git curl ca-certificates build-essential libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

# https://docs.astral.sh/uv/guides/integration/docker/#available-images
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/program

# Copy and install dependencies
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
RUN uv sync --no-cache

# ESM-2-650M weights + S3F checkpoint (Zenodo record 14257708)
RUN mkdir -p /opt/program/data/esm-model-weights && \
    curl -fL https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt \
        -o /opt/program/data/esm-model-weights/esm2_t33_650M_UR50D.pt && \
    curl -fL "https://zenodo.org/records/14257708/files/s3f.pth?download=1" \
        -o /opt/program/data/s3f.pth

ENV S3F_DATA_DIR=/opt/program/data

# pre-compile CPU KeOps kernels into a cache model.py copies at runtime
RUN KEOPS_CACHE_FOLDER=/opt/program/keops_cache uv run --no-sync python -W ignore - <<'PYEOF'
import torch
import torch_geometric.data
from proteingym.models.s3f.s3f import dataset, gvp, model, protein, surface, task
from proteingym.models.s3f import model as m
assert m.IMPORT_AVAILABLE, m._IMPORT_ERROR
torch.manual_seed(0)
n = 40
pos = torch.cumsum(torch.randn(n * 3, 3) * 1.2, 0).float()
atype = torch.nn.functional.one_hot(torch.tensor([3, 0, 0]).repeat(n), 6)
pts, nrm, _ = m.surface.atoms_to_points_normals(pos, torch.zeros(len(pos), dtype=torch.long), atomtypes=atype)
m.surface.knn_atoms(pos, pts, k=20)
m.surface.knn_atoms(pts, pos[1::3], k=3, batch_x=torch.zeros(len(pts), dtype=torch.long), batch_y=torch.zeros(n, dtype=torch.long))
m.surface.compute_curvatures(pts, nrm, batch=torch.zeros(len(pts), dtype=torch.long), curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0])
m.S3F(variant='s3f', checkpoint_path='/opt/program/data/s3f.pth', device='cpu')._load_task()
print('s3f import + checkpoint load OK')
PYEOF
RUN chmod -R a+rX /opt/program/keops_cache

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]