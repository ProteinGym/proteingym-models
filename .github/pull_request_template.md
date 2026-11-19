# Model

Name and one-line description of the model.

## What is novel or interesting

What the field can learn from this model (see the
[submission policy](https://github.com/ProteinGym/proteingym-benchmark/blob/main/CONTRIBUTING.md#model-submission-policy)).

## Build-time network access

Every URL the Dockerfile or code touches during the build, with the HTTP method and why.

# Checklist

- [ ] This PR adds or changes one model
- [ ] The model card has `name`, `model_class` and the right `tags` (`zero-shot` or `supervised`)
- [ ] The model runs offline; weights are baked in at build time
- [ ] Licences of the code and any pretrained weights are stated in the model card
- [ ] The original paper/code is credited
- [ ] Small weights or a fast seeded training recipe are included for the sanity check
- [ ] The image builds and the local benchmark run passes ([how](https://github.com/ProteinGym/proteingym-benchmark/blob/main/CONTRIBUTING.md#how-to-pass-ci))
- [ ] Unit tests pass: `uv run pytest`
