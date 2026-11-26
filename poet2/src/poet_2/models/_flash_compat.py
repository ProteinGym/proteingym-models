"""Stand-ins for the flash-attn helpers ``poet_2.models.poet_2`` imports.

Upstream imports four names from ``flash_attn`` unconditionally, which forces the
compiled CUDA extension to be installed even to *import* the model. That is a
problem here for two reasons: upstream pins a custom fork of flash-attention
published only as a ``cp310``/cu118 wheel, while this repo runs Python 3.12; and
the benchmark's CI has no GPU at all.

None of the four is needed to score:

* ``rearrange`` is einops, re-exported by ``flash_attn.bert_padding``.
* ``index_first_axis``/``pad_input``/``unpad_input`` are pure-torch packing
  helpers. Upstream's own ``modules.packed_sequence`` has equivalents, but its
  ``unpad_input`` deliberately reports ``max_seqlen = attention_mask.size(-1)``
  rather than the true maximum, so reusing it would quietly change semantics;
  these match flash-attn's behaviour instead.
* ``InferenceParams``/``sample`` are used only by the generation code paths.
  ``score_sequences`` never reaches them, so they raise if actually called.

Attention itself needs no shim: ``attention_flash.is_flash_attn_avail`` already
returns False when the module is absent or the tensors are not on CUDA, and
falls back to standard attention.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange  # noqa: F401  (re-exported, as flash_attn does)


def index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather rows along the first axis. flash-attn's version is the same
    operation wrapped in an autograd.Function to save memory on the backward
    pass, which inference does not need."""
    return tensor[indices]


def pad_input(
    hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int
) -> torch.Tensor:
    """Scatter packed rows back into (batch, seqlen, ...)."""
    dim = hidden_states.shape[-1]
    output = torch.zeros(
        batch * seqlen, dim, device=hidden_states.device, dtype=hidden_states.dtype
    )
    output[indices] = hidden_states
    return output.view(batch, seqlen, dim)


def unpad_input(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack the unmasked positions, returning flash-attn's 4-tuple of
    (packed, indices, cu_seqlens, max_seqlen_in_batch)."""
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    packed = index_first_axis(
        rearrange(hidden_states, "b s ... -> (b s) ..."), indices
    )
    return packed, indices, cu_seqlens, max_seqlen_in_batch


@dataclass
class InferenceParams:
    """Generation-only KV-cache bookkeeping; carried so the module imports."""

    max_seqlen: int = 0
    max_batch_size: int = 0
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Any = None


def sample(*args: Any, **kwargs: Any) -> torch.Tensor:
    raise NotImplementedError(
        "poet_2 generation needs flash-attn's sampler, which is not installed. "
        "This wrapper only scores sequences."
    )
