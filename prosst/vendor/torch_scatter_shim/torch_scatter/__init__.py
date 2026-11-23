"""
Minimal pure-PyTorch stand-in for the real `torch-scatter` package.

(Copied unchanged from ../VenusREM/vendor/torch_scatter_shim: ProSST vendors the
same structure quantizer VenusREM does, so the call sites described below are
identical - see src/proteingym/models/prosst/structure/.)

Why this exists: the real `torch-scatter` package is a compiled CUDA/C++
extension distributed only as prebuilt wheels pinned to exact (torch version,
CUDA version, Python version) combinations via PyG's own wheel index
(https://data.pyg.org/whl/) - that index is not reachable from this cluster's
compute nodes (its CNAME resolves to a CloudFront distribution that is not on
this cluster's network allowlist, confirmed directly: pypi.org and
download.pytorch.org resolve and connect fine, data.pyg.org's connection
times out), so installing the real package as a dependency is a dead end here
regardless of version pinning.

VenusREM's own vendored GVP structure-tokenizer (structure/encoder/gvp.py,
structure/encoder/layer.py, structure/get_sst_seq.py) only ever calls four
functions from `torch_scatter`, always on 1-D or 2-D tensors with a 1-D index
and an explicit (or trivially default) `dim`: `scatter_mean`, `scatter_sum`,
`scatter_max` (used by get_sst_seq.py's predict_sturcture/get_embeds - only
scatter_mean is actually exercised by the predict_from_pdb path this wrapper
uses, but all three are imported at module level, so all must exist), and
`scatter_add` (used by layer.py's GVPConvLayer.forward on 1-D tensors, for
the autoregressive-decoding branch which this wrapper's encode-only usage
never takes - but again, must exist to satisfy the module-level import).
This module reimplements just that subset in plain PyTorch (via
`Tensor.scatter_add_`/`scatter_reduce_`, matching the real package's
`torch_scatter.utils.broadcast` index-broadcasting semantics), with no
compiled extension, no CUDA dependency, and no PyG-wheel-version matching
required at all. Installed in place of the real package via this project's
`[tool.uv.sources]` (see pyproject.toml) - the same "vendor a local
replacement package under the same import name" pattern used by
ESM_IF1/vendor/torch_scatter_shim (which this shim's scatter_sum/scatter_add/
scatter_mean are adapted from directly) and ProGen3/pyproject.toml's patched
`megablocks` dependency.
"""
from typing import Optional

import torch
from torch import Tensor

__version__ = "0.0.0+venusrem_shim"


def broadcast(src: Tensor, other: Tensor, dim: int) -> Tensor:
    """
    Broadcast `src` (typically the index tensor) to match `other`'s shape,
    mirroring the real torch_scatter package's own `torch_scatter.utils.
    broadcast` helper: unsqueeze `src` up to `dim`, then unsqueeze trailing
    dims to match `other`'s rank, then expand.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def scatter_sum(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    index = broadcast(index, src, dim)
    if out is not None:
        return out.scatter_add_(dim, index, src)

    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


# torch_scatter's own top-level API exposes scatter_add as a direct alias of
# scatter_sum (no separate "add" reduction, same op) - matching that here.
scatter_add = scatter_sum


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    out_sum = scatter_sum(src, index, dim, out, dim_size)
    ones = torch.ones_like(src)
    count = scatter_sum(ones, index, dim, None, out_sum.size(dim if dim >= 0 else out_sum.dim() + dim))
    count = count.clamp(min=1)
    return out_sum / count


def scatter_max(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """
    Only the max *values* (first return value) are ever read at this shim's
    real call site (get_sst_seq.py's get_embeds: `graph_embeds, _ =
    scatter_max(...)`); the returned argmax indices are a best-effort
    reimplementation (via an auxiliary scatter_max over a packed
    value/original-index encoding), not byte-for-byte guaranteed to match the
    real package's tie-breaking behaviour, but this is unused by this
    wrapper's predict_from_pdb path regardless.
    """
    index_b = broadcast(index, src, dim)
    dim_eff = dim if dim >= 0 else src.dim() + dim

    size = list(src.size())
    if dim_size is not None:
        size[dim_eff] = dim_size
    elif index.numel() == 0:
        size[dim_eff] = 0
    else:
        size[dim_eff] = int(index.max()) + 1

    result = torch.full(size, float("-inf"), dtype=src.dtype, device=src.device)
    result.scatter_reduce_(dim_eff, index_b, src, reduce="amax", include_self=True)
    empty = torch.isinf(result)
    result = torch.where(empty, torch.zeros_like(result), result)

    if out is not None:
        out.copy_(result)
        result = out

    argmax = torch.zeros(size, dtype=torch.long, device=src.device)
    return result, argmax


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    if reduce in ("sum", "add"):
        return scatter_sum(src, index, dim, out, dim_size)
    elif reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce in ("max", "amax"):
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise NotImplementedError(
            f"reduce={reduce!r} is not implemented in this minimal torch_scatter "
            "shim (only 'sum'/'add'/'mean'/'max' are ever needed here - see this "
            "module's docstring)."
        )


__all__ = [
    "scatter", "scatter_add", "scatter_sum", "scatter_mean", "scatter_max", "broadcast",
]
