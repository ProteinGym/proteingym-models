# ProteinGym2 Models

The model zoo for [ProteinGym](https://github.com/ProteinGym/proteingym-benchmark).
Each top-level folder is one model: a small Python package with a Dockerfile and a
model card. The [benchmark repo](https://github.com/ProteinGym/proteingym-benchmark)
mounts this repo as a git submodule at `models/`, runs the models through its DVC
pipelines and shows the model cards on its website.

Submit a model with a pull request here; see the benchmark's [CONTRIBUTING.md](https://github.com/ProteinGym/proteingym-benchmark/blob/main/CONTRIBUTING.md)
for the submission policy and how CI validates your model.

> [!IMPORTANT]
> The website treats every top-level folder with a `README.md` as a model card.
> Don't add non-model top-level folders that contain a `README.md`.

This document describes how a model is structured so the benchmark can run it.

Models are run through [evedesign](https://github.com/evedesignbio/evedesign),
which we use as the framework to standardize model building and scoring. Every
model shares one generic runner, the `evedesign-proteingym` console script that
`evedesign[proteingym]` installs; what differs between models is the evedesign
model class named in the model card and its hyperparameters.

> [!NOTE]
> The runner is no longer vendored into each model. It lives upstream in
> `evedesign.proteingym` and reaches the image through the
> `evedesign[proteingym]` extra, so a model folder contains no runner code of
> its own. Every model uses it, supervised ones included.

## Layout

```tree
<model-folder>/
├── README.md            # model card: YAML front matter + description
├── Dockerfile           # image the benchmark runs
├── Singularity.def      # optional, for HPC clusters without Docker
├── pyproject.toml       # pinned dependencies
├── uv.lock
├── src/proteingym/models/<package>/
│   └── model.py         # only if you write your own evedesign model class
└── tests/
```

A model whose `model_class` already lives in evedesign (e.g.
`evedesign.models.esm2:ESM2`) has no Python source at all. Drop `src/` and the
`[build-system]`/`[tool.hatch…]` sections, set `package = false` under
`[tool.uv]`, and don't copy `src/` in the container recipe. See [esm2](esm2/),
[eve](eve/), [evmutation](evmutation/), [evmutation2](evmutation2/) and
[proteinmpnn](proteinmpnn/).

## Model card

The `README.md` is the model card. Its YAML front matter is read by the runner
(via `proteingym.base.model.ModelCard`), by the benchmark tooling and by the
website; the Markdown body is shown on the website's model page.

```yaml
---
# Model identifier used in the benchmark and on the website
name: blosum62

# Import string for the evedesign model class, either
# "package.module:ClassName" or "package.module.ClassName"
model_class: "proteingym.models.blosum62.model:SubstitutionMatrixScorer"

# "zero-shot" or "supervised" (which benchmark it runs in);
# add "embeddable" if the model can produce embeddings (see `embed` below)
tags: ["zero-shot"]

# Multi-target support flag
multi_y: false

# Forwarded verbatim as keyword arguments to the model class constructor
hyper_parameters:
    matrix_name: "BLOSUM62"
    gap_penalty: -4.0
---
```

| Field              | Required | Description |
|--------------------|----------|-------------|
| `name`             | Yes      | Model identifier. Usually the folder name, but it may differ (e.g. `huggingface-regressor` is named `ritaregressor`). |
| `model_class`      | Yes      | evedesign model class the runner imports, instantiates and scores with. |
| `tags`             | Yes      | `zero-shot` or `supervised`; optionally `embeddable`. |
| `multi_y`          | No       | Whether the model supports multiple targets. |
| `hyper_parameters` | No       | Constructor keyword arguments for `model_class`. |

Extra fields are allowed. The Markdown body should describe the architecture,
training data and hyperparameters, credit the original paper/code, and state
licences and build-time network access (see the [submission policy](https://github.com/ProteinGym/proteingym-benchmark/blob/main/CONTRIBUTING.md#model-submission-policy)).

## The evedesign runner

The runner is the `evedesign-proteingym` console script, installed by the
`evedesign[proteingym]` extra and implemented in
[`evedesign.proteingym`](https://github.com/evedesignbio/evedesign/blob/main/src/evedesign/proteingym.py).
There is nothing to copy into a model. It exposes three commands:

- **`train`** – what the benchmark calls. It:
  1. loads the dataset (`proteingym.base.Subsets`) and model card,
  2. converts them with `evedesign.proteingym.dataset_to_evedesign` into an
     evedesign `System` plus training/test datasets (for `zero-shot` models the
     whole dataset is scored and `--split`/`--test-fold` are ignored),
  3. instantiates `model_class(**hyper_parameters)`, calls `build(system, data=...)`
     and then `score(...)` on the test instances,
  4. writes `/opt/program/output/<dataset>_predictions.pgdata`.

  Options: `--dataset-file`, `--target`, `--split`, `--test-fold`,
  `--model-card-file` (defaults to `/opt/program/README.md`, i.e. the card baked
  into the image), `--model-file` (load a dill-pickled built model, or save one
  there), `--precomputed-instances-file` (use embeddings produced by `embed`).
- **`embed`** – for `embeddable` models: runs `transform()` (or `score()`) over
  the whole dataset and pickles the instances to `/opt/program/output/embeddings.pkl`
  so a downstream model can consume them.
- **`ping`** – health check.

### Choosing the model class

- If evedesign already wraps the model (e.g. `evedesign.models.esm2:ESM2`),
  point `model_class` at it and set its constructor arguments in
  `hyper_parameters`. See [esm2](esm2/), [evmutation](evmutation/),
  [evmutation2](evmutation2/).
- Otherwise, write a class in `model.py` that subclasses evedesign's `BaseModel`
  (or `SupervisedBaseModel`) and the interfaces it supports, such as `Scorer`
  and `Transformer`, and point `model_class` at it. See
  [blosum62](blosum62/src/proteingym/models/blosum62/model.py) for a minimal
  example, or [esmc](esmc/), [gemme](gemme/), [rsalor](rsalor/).

## Containerization

The benchmark runs one ephemeral container per dataset/model (and fold), so the
image must contain everything needed to run.

```dockerfile
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y git curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/program

# README.md is the model card the runner reads by default
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
RUN uv sync --no-cache

# Bake weights into the image at build time; containers may run without network.
# See esmc/Dockerfile for a pinned Hugging Face download.

ENTRYPOINT ["uv", "run", "evedesign-proteingym"]
```

The entrypoint is the same for every model: `evedesign-proteingym`, installed
into the image's venv by the `evedesign[proteingym]` extra. Models no longer
declare a `[project.scripts]` entry of their own.

Dependencies are pinned to exact PyPI releases
(`proteingym-base==0.2.0`, `evedesign[<extras>,proteingym]==0.0.8`); nothing is
fetched from git except the few upstream packages that still need a patched
revision (see `evmutation`, `tranception`, `aido-rag-16b`). The runner's own
dependencies — `typer`, `polars`, `rich`, `dill` — arrive through the
`proteingym` extra and should not be listed again by a model.

### Build and run

The benchmark expects the image to be tagged `<model-folder>:latest`:

```shell
docker build -t blosum62:latest blosum62

docker run --rm \
  -v $(pwd)/pls/tests/data/Dummy_test_P0DX94.splits.pgdata:/data.pgdata \
  -v $(pwd)/output:/opt/program/output \
  blosum62:latest train \
  --dataset-file /data.pgdata --target charge --split random --test-fold 0
```

### Test

Run the model's unit tests with `uv run pytest` from its folder.
