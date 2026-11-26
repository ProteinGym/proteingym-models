"""Patch upstream ProteinNPT for CPU support.

Run at image build time, after `uv sync`. The patch fails loudly if upstream
changes shape, so it can be dropped once fixed upstream.

1. Trainer is GPU-only. `utils/model_utils.py` calls `self.model.cuda()`
   unconditionally in train() and eval(), even though ProteinNPTModel itself is
   device-agnostic (no .cuda() calls; set_device() infers from its parameters).

Deliberately NOT patched: `proteinnpt/model.py` passes `self.model_type`
("ProteinNPT") to get_embeddings_ESM, which branches on the *embedder* name and
so leaves `last_layer_index` unbound. That call site is upstream's on-the-fly
embedding path, which this wrapper does not use - it precomputes embeddings to
an .h5 and sets `sequence_embeddings_location`, so forward() never reaches it.
Leaving the bug in place is the safer choice: if that path is ever re-enabled it
raises immediately, rather than silently running the embedder inside forward()
where upstream's optimiser would fine-tune all 651 M of its parameters.
"""

import pathlib
import sys

import proteinnpt.utils.model_utils as mu


def patch(path: pathlib.Path, needle: str, replacement: str, expected: int, what: str):
    text = path.read_text()
    if replacement in text:
        print(f"  already patched ({what}): {path.name}")
        return
    found = text.count(needle)
    if found != expected:
        sys.exit(
            f"ERROR: expected {expected} occurrence(s) of {what} in {path}, found "
            f"{found}. Upstream has changed; re-check whether this patch applies."
        )
    path.write_text(text.replace(needle, replacement, expected))
    print(f"  patched {what} in {path.name} ({found} site(s))")


patch(
    pathlib.Path(mu.__file__),
    "        self.model.cuda()\n",
    "        self.model.to(getattr(self.args, 'device', 'cpu'))  # patched\n",
    2,
    "unconditional model.cuda()",
)
