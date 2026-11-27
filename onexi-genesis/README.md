---
# Model identifier used for referencing this model in the benchmark system
name: onexi-genesis

# Import string for the evedesign model class to run.
model_class: "proteingym.models.onexi_genesis.model:OneXIGenesisScorer"

tags: ["zero-shot"]

# Multi-target support flag
multi_y: false

# Hyperparameters forwarded verbatim to OneXIGenesisScorer
hyper_parameters:
    matrix_name: "BLOSUM62"
    context_window: 7
    context_weight: 0.35
    gap_penalty: -4.5
---

# Model Card for ONE XI Genesis (`onexi-genesis`, evedesign-wrapped)

`onexi-genesis` is a zero-shot, zero-weight protein variant fitness and
thermodynamic stability scorer wrapped in the standard
[evedesign](https://github.com/evedesignbio/evedesign) `Scorer` /
`MutationScorer` / `ConditionalMutationScorer` interface for ProteinGym2.

## Overview & Novelty

In single-sequence zero-shot mode (requiring only the wild-type target sequence,
zero trained neural weights, no GPU, and zero runtime network access),
`OneXIGenesisScorer` combines:

1. **Wild-Type-Anchored Substitution Baseline (Attributed Prior Art):** Bundled
   Biopython substitution log-odds (`BLOSUM62` default) anchored strictly so
   every wild-type residue scores `0.0` and every missense substitution scores
   `<= 0.0`.
2. **Local Sequence-Window Biophysical Niche Conditioning (Model Delta):** A
   deterministic sliding window (`context_window = 7`) computing local
   hydrophobic burial density (Kyte-Doolittle hydropathy) together with
   normalized side-chain volume, formal pH 7.4 charge mismatch, and
   proline/secondary-structure disruption penalties. Buried hydrophobic cores
   amplify destabilizing physicochemical deltas while solvent-exposed loops
   dampen them, running in sub-millisecond CPU latency per protein.

## Network & Reproducibility Audit (`CONTRIBUTING.md` Compliance)

- **Runtime Network Access:** **None (`0` external calls)**. Runs 100% offline
  inside `docker run --network none`.
- **Build-Time URLs:**
  - `GET https://pypi.org/simple` (via `uv sync --no-cache` during container
    build to install pinned wheels in `uv.lock`, including
    `proteingym-base==0.2.0` and `evedesign[proteingym]==0.0.8`).
- **Weights:** Zero trained weights (`< 50 KB` embedded biophysical constants
  and Biopython package data; no external model-hub downloads).
- **License:** MIT License (permits unrestricted inclusion, modification, and
  benchmarking by ProteinGym).

## Citation

```bibtex
@misc{younus2026onexigenesis,
  title={ONE XI Genesis: Deterministic Biophysical and Context-Conditioned Variant Effect Scoring},
  author={Younus, Khurram},
  year={2026},
  url={https://onexibio.com/clinical}
}
```
