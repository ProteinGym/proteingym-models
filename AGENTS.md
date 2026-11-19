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
  shares the same `__main__.py` runner, and the model card's `model_class` and
  `hyper_parameters` select and configure the model. The supervised models (`pls`,
  `kermut`, `pkermut`, `huggingface-regressor`) still have custom entrypoints; don't
  copy them as templates.
- Model cards are parsed by `proteingym.base.model.ModelCard` from
  [proteingym-base](https://github.com/ProteinGym/proteingym-base).
- Images are tagged `<model-folder>:latest`; the model card is baked into the image at
  `/opt/program/README.md` and predictions are written to `/opt/program/output`.
- Models must run offline: weights are downloaded at Docker build time.
- CI lives in the benchmark repo (reusable `validate-model.yml`); the workflow here only
  calls it. Changing what CI does means changing the benchmark repo.

## Commands

```shell
cd <model> && uv run pytest                 # model unit tests
docker build -t <model>:latest <model>      # build an image
```
