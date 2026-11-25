"""Checks the scoring protocol, which is where ProGen2 is easy to get wrong."""

import math

import pytest
import torch

from proteingym.models.progen2.model import (
    FIRST_AA_TOKEN,
    canonical_log_likelihood,
)


def _logits(rows, vocab=32, favoured=None, strength=10.0):
    """Flat logits, optionally spiking one token id per row."""
    out = torch.zeros(rows, vocab)
    for i, tid in enumerate(favoured or []):
        if tid is not None:
            out[i, tid] = strength
    return out


def test_sums_log_probs_over_the_amino_acid_slice():
    # ids: marker, A, A  -> after shifting, two residue targets
    ids = torch.tensor([3, FIRST_AA_TOKEN, FIRST_AA_TOKEN])
    # flat logits over the 25 amino-acid columns -> log(1/25) per position
    got = canonical_log_likelihood(_logits(3), ids)

    assert got == pytest.approx(2 * math.log(1 / 25), abs=1e-4)


def test_special_token_logits_are_excluded_from_the_normalization():
    # A huge logit on <|pad|> must not change the score: the slice drops it.
    ids = torch.tensor([3, FIRST_AA_TOKEN, FIRST_AA_TOKEN])
    spiked = _logits(3)
    spiked[:, 0] = 50.0

    assert canonical_log_likelihood(spiked, ids) == pytest.approx(
        canonical_log_likelihood(_logits(3), ids), abs=1e-4
    )


def test_trailing_direction_marker_is_dropped():
    # ids ending in the "2" marker (id 4) must score the same as without it
    with_marker = torch.tensor([3, FIRST_AA_TOKEN, FIRST_AA_TOKEN, 4])
    without = torch.tensor([3, FIRST_AA_TOKEN, FIRST_AA_TOKEN])

    assert canonical_log_likelihood(_logits(4), with_marker) == pytest.approx(
        canonical_log_likelihood(_logits(3), without), abs=1e-4
    )


def test_a_confident_model_scores_higher_than_an_unsure_one():
    ids = torch.tensor([3, FIRST_AA_TOKEN, FIRST_AA_TOKEN])
    confident = _logits(3, favoured=[None, FIRST_AA_TOKEN, FIRST_AA_TOKEN])

    assert canonical_log_likelihood(confident, ids) > canonical_log_likelihood(
        _logits(3), ids
    )


def test_a_sequence_with_no_residues_fails_loudly():
    # just the two direction markers, nothing to score
    with pytest.raises(ValueError, match="no scorable amino-acid tokens"):
        canonical_log_likelihood(_logits(2), torch.tensor([3, 4]))
