"""Make MSA-Pairformer importable without a GPU.

Upstream msa-pairformer 1.0.2 defines CUEQUIVARIANCE_AVAILABLE only inside an
``if torch.cuda.is_available():`` branch:

    try:
        if torch.cuda.is_available():
            import cuequivariance_torch as cuex
            CUEQUIVARIANCE_AVAILABLE = True
            ...
    except ImportError:
        CUEQUIVARIANCE_AVAILABLE = False

With no GPU the branch is skipped, no ImportError is raised, and the name is
never bound -- so importing MSA_Pairformer.pairwise_operations dies with
``NameError: name 'CUEQUIVARIANCE_AVAILABLE' is not defined`` while evaluating
a class-body default argument. This binds the default before the try block.

Run at image build time, after `uv sync`. Fails loudly if upstream changes
shape, so the patch can be dropped once msa-pairformer fixes this.
"""

import pathlib
import sys

import MSA_Pairformer

target = pathlib.Path(MSA_Pairformer.__file__).parent / "pairwise_operations.py"
text = target.read_text()

SENTINEL = "CUEQUIVARIANCE_AVAILABLE = False  # patched-in default"
if SENTINEL in text:
    print(f"already patched: {target}")
    sys.exit(0)

NEEDLE = "try:\n    if torch.cuda.is_available():"
if NEEDLE not in text:
    sys.exit(
        f"ERROR: expected cuda-guarded import block not found in {target}. "
        "Upstream has changed; re-check whether this patch is still needed."
    )

text = text.replace(
    NEEDLE,
    f"# Patched for CPU-only use (see patch_cuequivariance.py).\n{SENTINEL}\n\n{NEEDLE}",
    1,
)
target.write_text(text)
print(f"patched {target}")
