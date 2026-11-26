"""Checks the flash-attn fallback, since it stands in for upstream's packing helpers."""

import pytest
import torch

from poet_2.models import _flash_compat as fc
from poet_2.models import poet_2 as model_module


def test_the_fallback_is_what_the_model_module_bound():
    # upstream imports these from flash_attn unconditionally; the wrapper only
    # works without flash_attn if the patched import actually took the fallback
    assert model_module.InferenceParams is fc.InferenceParams


def test_unpad_then_pad_round_trips():
    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    packed, indices, cu_seqlens, max_seqlen = fc.unpad_input(hidden, mask)

    assert packed.shape == (3, 4)            # 2 + 1 valid positions
    assert max_seqlen == 2                   # flash's semantics: the true max,
    assert cu_seqlens.tolist() == [0, 2, 3]  # not the mask width (3)

    restored = fc.pad_input(packed, indices, batch=2, seqlen=3)
    assert torch.equal(restored[mask.bool()], hidden[mask.bool()])
    assert torch.equal(restored[~mask.bool()], torch.zeros(3, 4))


def test_generation_sampler_refuses_rather_than_returning_nonsense():
    with pytest.raises(NotImplementedError):
        fc.sample(torch.zeros(1, 2))


def test_unfused_swiglu_is_used_and_matches_upstream_reference():
    """Without xformers the fused branch is unreachable, so the function must
    delegate to upstream's own unfused implementation rather than raise."""
    from torch import nn

    from poet_2.models.modules import transformer as tf

    assert tf.XFORMERS_MODULE_INSTALLED is False

    # GLU(dim_in, dim_out) projects to dim_out * 2 and chunks, so it emits dim_out
    linear1 = tf.GLU(8, 16, activation=torch.nn.functional.silu)
    linear2 = nn.Linear(16, 8, bias=False)
    dropout = nn.Dropout(0.0)
    x = torch.randn(2, 8)

    got = tf._maybe_fused_swiglu_apply_feedforward(
        linear1, dropout, linear2, x, use_fused_swiglu=False
    )
    want = tf._swiglu_apply_feedforward(linear1, dropout, linear2, x)

    assert torch.equal(got, want)


def test_wrapper_alphabet_defines_a_gap_token():
    """compute_homology_weights asserts gap_token >= 0, and the MSA must be
    encoded with the same alphabet the model tokenizes with. Uniprot21 has no
    gap token (-1); the tokenizer's actual default does."""
    from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet
    from poet_2.alphabets import Uniprot21

    from proteingym.models.poet2.model import PoET2

    assert Uniprot21().gap_token < 0            # the trap
    assert Alphabet().gap_token >= 0            # what the model actually uses

    model = PoET2.__new__(PoET2)                # no weights needed
    PoET2.__init__(model, ckpt_path="/nonexistent")
    assert type(model._alphabet) is Alphabet
    assert model._alphabet.gap_token >= 0


def test_sample_prompt_builds_a_prompt_from_an_msa():
    """Exercises the whole MSA path with no weights loaded: encoding the MSA,
    homology reweighting (which asserts gap_token >= 0) and the token budget.
    This is the path that failed in the container while import-only tests passed."""
    from evedesign.sequence import Sequence, Sequences

    from proteingym.models.poet2.model import PoET2

    msa = ["MKTAYIAKQ", "MKTAYIAKR", "MRTAYIA-Q", "MKT-YIAKQ", "LKTAYLAKQ"]
    seqs = Sequences(
        seqs=[Sequence(seq=s, metadata=None) for s in msa],
        aligned=True,
        format="a3m",
    )

    class _Target:
        sequences = seqs
        rep = msa[0]

    model = PoET2.__new__(PoET2)
    PoET2.__init__(model, ckpt_path="/nonexistent", device="cpu", max_tokens=64)

    prompt = model._sample_prompt(_Target())

    assert len(prompt) > 0
    assert all("-" not in p.sequence for p in prompt)        # gaps stripped
    assert sum(len(p.sequence) for p in prompt) <= 64        # budget respected
    # The target must NOT appear in its own prompt. If it does, the model can
    # copy it and log P(wildtype | prompt) collapses (about -10 instead of -930
    # on a 465-residue target); since every score is a delta against that term,
    # it cost up to 0.19 Spearman on HXK4 and inverted the context scaling.
    assert all(p.sequence != msa[0].replace("-", "") for p in prompt)
