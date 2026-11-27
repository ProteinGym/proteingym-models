"""Checks ProtT5's tokenization contract and the head/embedder wiring.

None of these downloads weights: the tokenization rules and the residue-position
derivation are the two things that silently corrupt embeddings rather than
raising, so they are tested in isolation.
"""

import pytest

from proteingym.models.prott5.model import ProtT5, ProtT5Embedder


class _FakeTokenizer:
    """Stands in for T5Tokenizer: T5 appends </s> and has no start token."""

    all_special_ids = [0, 1, 2]  # pad, eos, unk


def _embedder():
    e = ProtT5Embedder.__new__(ProtT5Embedder)  # no __init__, no imports needed
    e.tokenizer = _FakeTokenizer()
    return e


def test_residues_are_space_separated_and_rare_codes_folded_to_x():
    # U/Z/O/B tokenize fine but are untrained; ProtT5 was trained with them as X
    assert ProtT5Embedder._to_tokenizer_input("MUZOB") == "M X X X X"
    assert ProtT5Embedder._to_tokenizer_input("MKV") == "M K V"


def test_lowercase_insertions_are_upcased_not_dropped():
    # evedesign reps can carry lowercase insert states
    assert ProtT5Embedder._to_tokenizer_input("MkV") == "M K V"


def test_residue_positions_exclude_the_trailing_eos():
    # T5 has no BOS, so the ESM convention of [1:L+1] would shift by one
    ids = [5, 6, 7, 1]  # three residues then </s>
    assert _embedder()._residue_positions(ids) == [0, 1, 2]


def test_residue_positions_drop_padding_in_a_batch():
    ids = [5, 6, 1, 0, 0]  # two residues, </s>, then pad
    assert _embedder()._residue_positions(ids) == [0, 1]


def test_head_is_resolved_by_name_and_embedder_gets_the_checkpoint():
    model = ProtT5(
        predictor="Ridge",
        predictor_kwargs={"alpha": 2.5},
        model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
    )

    assert type(model.predictor).__name__ == "Ridge"
    assert model.predictor.alpha == 2.5
    assert model.embedder.model_name == "Rostlab/prot_t5_xl_half_uniref50-enc"
    assert model.pooling_strategy == "mean"


def test_an_unknown_head_fails_loudly():
    with pytest.raises(ValueError, match="Invalid regressor"):
        ProtT5(predictor="NotARegressor")
