"""Checks the head's architecture contract and the landscape scoring maths."""

import numpy as np
import pytest
import torch

from evedesign.system import EntityInstance, Protein, System, SystemInstance
from proteingym.models.vespag.model import (
    GEMME_ALPHABET,
    VespaG,
    landscape_by_char,
    load_head,
)

TARGET = "ACDEFG"


def v2_state(out_bias=None):
    """A correctly shaped upstream-v2 state dict (keys net.0 and net.3)."""
    return {
        "net.0.weight": torch.zeros(256, 2560),
        "net.0.bias": torch.zeros(256),
        "net.3.weight": torch.zeros(20, 256),
        "net.3.bias": torch.zeros(20) if out_bias is None else out_bias,
    }


def built_model(landscape):
    """A VespaG with the landscape injected, skipping the 11 GB embedder."""
    model = VespaG.__new__(VespaG)          # no embedder construction
    model._system = System([Protein(rep=TARGET, first_index=1)])
    model._landscape = landscape
    model._landscape_by_char = landscape_by_char(landscape)
    model.normalize = False
    return model


def test_alphabet_matches_upstream_exactly():
    # the 20 output columns are in this order; a permutation silently
    # mis-assigns every single score
    assert GEMME_ALPHABET == "ACDEFGHIKLMNPQRSTVWY"
    assert list(GEMME_ALPHABET) == sorted(GEMME_ALPHABET)
    assert len(GEMME_ALPHABET) == 20


def test_load_head_rejects_a_v1_checkpoint(tmp_path):
    # v1 has no dropout, so its output layer is net.2; loading it into the v2
    # layout would put the output weights on the wrong module
    v1 = {k.replace("net.3", "net.2"): v for k, v in v2_state().items()}
    path = tmp_path / "v1.pt"
    torch.save(v1, path)

    with pytest.raises(ValueError, match="v1 checkpoint is not supported"):
        load_head(path, "cpu")


def test_load_head_builds_the_published_architecture(tmp_path):
    # a zero net with a known output bias must emit exactly that bias, which
    # pins both the layer order and that net.3 really is the output layer
    bias = torch.arange(20, dtype=torch.float32)
    path = tmp_path / "v2.pt"
    torch.save(v2_state(out_bias=bias), path)

    head = load_head(path, "cpu")
    with torch.no_grad():
        out = head(torch.zeros(5, 2560))

    assert out.shape == (5, 20)
    assert torch.allclose(out, bias.expand(5, 20))


def test_landscape_by_char_puts_each_column_at_its_byte():
    landscape = np.arange(len(TARGET) * 20, dtype=float).reshape(len(TARGET), 20)

    by_char = landscape_by_char(landscape)

    for j, aa in enumerate(GEMME_ALPHABET):
        assert np.array_equal(by_char[:, ord(aa)], landscape[:, j])
    # a symbol outside the alphabet contributes nothing rather than erroring
    assert by_char[:, ord("X")].sum() == 0.0


def test_wild_type_scores_zero_and_mutants_sum_their_substitutions():
    # distinct value per (position, aa) so a wrong index cannot coincide
    landscape = np.arange(len(TARGET) * 20, dtype=float).reshape(len(TARGET), 20) + 1.0
    for pos, wt in enumerate(TARGET):
        landscape[pos, GEMME_ALPHABET.index(wt)] = 0.0   # upstream's mask_non_mutations
    model = built_model(landscape)

    wt = SystemInstance([EntityInstance(rep=TARGET)])
    single = SystemInstance([EntityInstance(rep="WCDEFG")])          # A1W
    double = SystemInstance([EntityInstance(rep="WCDEFW")])          # A1W + G6W

    scored = model.score([wt, single, double])
    w = GEMME_ALPHABET.index("W")

    assert scored[0].score == 0.0
    assert scored[1].score == pytest.approx(landscape[0, w])
    # a multiple mutant is the sum of its substitutions, as upstream does
    assert scored[2].score == pytest.approx(landscape[0, w] + landscape[5, w])
