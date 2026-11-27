# AGENTS.md

ProteinGym2 models: the model zoo benchmarked by
[proteingym-benchmark](https://github.com/ProteinGym/proteingym-benchmark), which mounts
this repo as a git submodule at `models/`. Read these before changing things:

- [README.md](README.md) – model structure, model card fields, the evedesign runner, Dockerfile.
- [proteingym-benchmark CONTRIBUTING.md](https://github.com/ProteinGym/proteingym-benchmark/blob/main/CONTRIBUTING.md) – submission policy, CI, local CI reproduction
  (kept in the benchmark repo).

## Key facts

- Each top-level folder is one model. The benchmark website globs `models/*/README.md`,
  so any top-level folder with a `README.md` shows up as a model card.
- Models run through [evedesign](https://github.com/evedesignbio/evedesign); every model
  uses the `evedesign-proteingym` console script as its entrypoint (shipped by
  `evedesign[proteingym]`, implemented in `evedesign.proteingym`), and the model card's
  `model_class` and `hyper_parameters` select and configure the model. No model vendors
  its own runner — a model folder holds only a model card, a container recipe, pinned
  dependencies, and `model.py` if it needs a model class of its own.
- `evedesign` and `proteingym-base` are pinned to exact PyPI releases in every model's
  `pyproject.toml`; neither is fetched from git. Bumping a release means bumping all of
  them and relocking.
- Container recipes come in two flavours: `Dockerfile` (what the benchmark and CI use) and
  an optional `Singularity.def` for HPC clusters without Docker. Keep them in step. A SIF
  is read-only, so run it with `--writable-tmpfs` if the model writes beside its CWD
  (evcouplings does), and note `/opt/program/output` must already exist in the image —
  Docker creates bind targets, Apptainer cannot.
- Model cards must ship `device: "cpu"`: CI has no GPU. Override it for a local GPU run by
  passing an edited copy via `--model-card-file` rather than editing the committed card.
  `poet` and `progen3` are the exceptions: both are GPU-only by construction, so
  neither can pass CPU-only CI. `poet` has no `device` hyperparameter at all
  (`.cuda()`/`.half()` throughout its wrapper); `progen3` ships `device: "cuda"`
  because its CUDA extensions initialise CUDA on import.
- A few models still take an upstream package from git: `evcouplings` (`evmutation`,
  `tranception`), `progen3`, and `modelgenerator`/`openfold`/`dllogger` (`aido-rag-16b`).
  None is pinned to a revision — only `uv.lock` records the commit — so regenerating a
  lock can silently move them to a newer upstream. Pin a `rev` before relocking those
  models if you need the resolution to be stable.
- `proteingym-base>=0.2` requires `biotite>=2.0`, which requires `numpy>=2.1`, and `numba`
  (via evedesign) refuses NumPy > 2.4. Models whose upstream pins `numpy<2` therefore carry
  `override-dependencies = ["numpy>=2.1,<2.5"]` (`s3f`, `msa-pairformer`, `aido-rag-16b`,
  `progen3`). Drop the override once those upstreams support NumPy 2.
- Model cards are parsed by `proteingym.base.model.ModelCard` from
  [proteingym-base](https://github.com/ProteinGym/proteingym-base).
- Images are tagged `<model-folder>:latest`; the model card is baked into the image at
  `/opt/program/README.md` and predictions are written to `/opt/program/output`.
- Models must run offline: weights are downloaded at Docker build time.
- CI lives in the benchmark repo (reusable `validate-model.yml`); the workflow here only
  calls it. Changing what CI does means changing the benchmark repo.
- The benchmark mounts this repo as a submodule, which records an exact commit, so a merge
  to `main` here does not reach it on its own. `.github/workflows/notify-benchmark.yml`
  dispatches `models-main-updated` to the benchmark, whose `bump-models-pin.yml` bumps the
  pin and opens a pull request there. It needs the `BENCHMARK_DISPATCH_TOKEN` secret.

## Commands

```shell
cd <model> && uv run pytest                 # model unit tests
docker build -t <model>:latest <model>      # build an image
```
