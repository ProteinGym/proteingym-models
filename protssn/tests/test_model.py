"""
Tests for the ProtSSN wrapper.

These cover the places where a wrong implementation still produces
plausible-looking scores: the output vocabulary order, the EGNN config the
checkpoint was trained with, the log-ratio convention, and the residue
filtering the upstream featurizer depends on. None of them downloads weights.
"""
import numpy as np
import pytest

from proteingym.models.protssn.model import (
    AA_ORDER,
    _EGNN_CONFIG,
    ProtSSN,
    _encodable_residues,
    _scatter_to_positions,
)
from proteingym.models.protssn.upstream.models import PLM_model
from proteingym.models.protssn.upstream.mutant_dataset import MutantDataset, one_letter


def test_aa_order_matches_node_features():
    """
    The 20 output logits are indexed by the same list the node one-hot uses.

    MutantDataset lists 37 residue names but one_hot_res() rejects any index
    >= 20, so only the first 20 can appear; if their order differed from
    AA_ORDER every variant would be scored against the wrong column.
    """
    upstream = MutantDataset.allowable_features["possible_amino_acids"][:20]

    assert "".join(one_letter[name] for name in upstream) == AA_ORDER


def test_aa_order_matches_sequence_reconstruction():
    """
    PLM_model rebuilds the sequence from the node one-hot using its own list,
    which must agree with the one the features were built from.
    """
    assert PLM_model.possible_amino_acids == (
        MutantDataset.allowable_features["possible_amino_acids"][:20]
    )
    assert "".join(one_letter[n] for n in PLM_model.possible_amino_acids) == AA_ORDER


def test_aa_order_is_twenty_unique_symbols():
    assert len(AA_ORDER) == 20
    assert len(set(AA_ORDER)) == 20


def test_egnn_config_matches_upstream_checkpoint_config():
    """
    Pin the architecture the published checkpoints were trained with.

    Taken from upstream's config/k20_h512.json; hidden_channels is supplied
    separately from the checkpoint's h, so it is not part of the shared config.
    """
    assert _EGNN_CONFIG == {
        "edge_attr_dim": 93,
        "output_dim": 20,
        "dropout": 0,
        "n_layers": 6,
        "residual": False,
        "embedding": False,
        "embedding_dim": 64,
        "mlp_num": 2,
    }
    assert "hidden_channels" not in _EGNN_CONFIG


def test_checkpoint_name_must_match_k_and_h():
    """A checkpoint whose k/h disagree with the hyperparameters must not load."""
    with pytest.raises(ValueError, match="does not match"):
        ProtSSN(
            gnn_model_path="/data/protssn_k30_h1280.pt",
            c_alpha_max_neighbors=20,
            hidden_channels=512,
        )


def test_checkpoint_name_match_is_accepted():
    model = ProtSSN(
        gnn_model_path="/data/protssn_k10_h768.pt",
        c_alpha_max_neighbors=10,
        hidden_channels=768,
    )

    assert model.c_alpha_max_neighbors == 10
    assert model.hidden_channels == 768


def test_non_upstream_checkpoint_name_is_not_second_guessed():
    """A path that does not follow upstream's naming is left alone."""
    model = ProtSSN(
        gnn_model_path="/data/my_finetuned_model.pt",
        c_alpha_max_neighbors=30,
        hidden_channels=512,
    )

    assert model.gnn_model_path == "/data/my_finetuned_model.pt"


def _model_with_logits(logits: np.ndarray) -> ProtSSN:
    """Build a ProtSSN with a cached log-probability matrix and nothing else."""
    model = object.__new__(ProtSSN)
    model._logits = logits
    return model


def test_wildtype_position_scores_zero():
    """
    The log-ratio convention means an unmutated position contributes exactly
    0.0, so the wild type scores 0.0 overall.
    """
    rng = np.random.default_rng(0)
    logits = np.log(rng.dirichlet(np.ones(20), size=5))
    model = _model_with_logits(logits)

    for pos in range(5):
        wt = AA_ORDER[int(np.argmax(logits[pos]))]
        assert model._position_score(pos, wt, wt) == 0.0


def test_position_score_is_the_log_ratio():
    logits = np.log(np.full((1, 20), 1 / 20))
    logits[0, AA_ORDER.index("W")] = np.log(0.5)
    model = _model_with_logits(logits)

    expected = np.log(0.5) - np.log(1 / 20)

    assert model._position_score(0, "W", "A") == pytest.approx(expected)
    # and the reverse substitution is its negation
    assert model._position_score(0, "A", "W") == pytest.approx(-expected)


def test_uncovered_position_contributes_nothing():
    """A position the structure does not cover is NaN and must score 0.0."""
    logits = np.full((2, 20), np.nan)
    logits[1] = np.log(np.full(20, 1 / 20))
    model = _model_with_logits(logits)

    assert model._position_score(0, "W", "A") == 0.0
    assert not np.isnan(model._position_score(0, "W", "A"))


def test_symbol_outside_the_twenty_raises():
    model = _model_with_logits(np.log(np.full((1, 20), 1 / 20)))

    with pytest.raises(ValueError, match="not one of"):
        model._position_score(0, "X", "A")


def test_encodable_residues_keeps_only_complete_standard_residues():
    """
    Upstream's featurizer rejects the whole structure on one non-standard or
    backbone-incomplete residue, so these must be dropped first.
    """
    struc = pytest.importorskip("biotite.structure")

    def residue(res_id, res_name, atoms, hetero=False):
        array = struc.AtomArray(len(atoms))
        array.coord = np.zeros((len(atoms), 3), dtype=np.float32)
        array.chain_id = np.full(len(atoms), "A")
        array.res_id = np.full(len(atoms), res_id)
        array.res_name = np.full(len(atoms), res_name)
        array.atom_name = np.array(atoms)
        array.element = np.array([a[0] for a in atoms])
        array.hetero = np.full(len(atoms), hetero)
        return array

    full = ("N", "CA", "C", "O")
    atom_array = (
        residue(1, "ALA", full)
        + residue(2, "MSE", full)           # modified residue: not in the 20
        + residue(3, "GLY", ("N", "CA"))    # missing backbone C
        + residue(4, "TRP", full)
        + residue(5, "ALA", full, hetero=True)
    )

    kept = _encodable_residues(atom_array)

    assert sorted(set(kept.res_id.tolist())) == [1, 4]


def test_scatter_maps_by_residue_number_not_node_order():
    """
    A residue dropped before featurization must not shift later positions.

    This is the failure that produces finite, non-degenerate, wrong scores: if
    nodes were written in order, every position after a dropped residue would
    be scored against its neighbour's structural context.
    """
    # residue 3 was dropped, so nodes carry residue numbers 1, 2, 4, 5
    res_ids = np.array([1, 2, 4, 5])
    node_logits = np.array([[float(r)] * 20 for r in res_ids])

    logits = _scatter_to_positions(node_logits, res_ids, length=5, first_index=1)

    # each kept residue lands at its own position, identified by its value
    assert logits[0, 0] == 1.0
    assert logits[1, 0] == 2.0
    assert np.isnan(logits[2, 0])     # the dropped residue 3
    assert logits[3, 0] == 4.0        # NOT shifted up into position 2
    assert logits[4, 0] == 5.0


def test_scatter_respects_first_index():
    """first_index offsets residue numbers, so position 0 is first_index."""
    res_ids = np.array([101, 103])
    node_logits = np.array([[1.0] * 20, [3.0] * 20])

    logits = _scatter_to_positions(node_logits, res_ids, length=3, first_index=101)

    assert logits[0, 0] == 1.0
    assert np.isnan(logits[1, 0])
    assert logits[2, 0] == 3.0


def test_scatter_ignores_residues_outside_the_target():
    """A structure covering more than the target must not write out of range."""
    res_ids = np.array([0, 1, 2, 99])
    node_logits = np.array([[float(r)] * 20 for r in res_ids])

    logits = _scatter_to_positions(node_logits, res_ids, length=2, first_index=1)

    assert logits.shape == (2, 20)
    assert logits[0, 0] == 1.0
    assert logits[1, 0] == 2.0


def test_dropped_residue_scores_zero_and_neighbours_are_unaffected():
    """
    End to end over the two pieces that must agree: a variant at a dropped
    position contributes nothing, while its neighbours score normally.
    """
    res_ids = np.array([1, 2, 4])
    probs = np.full((3, 20), 1 / 20)
    probs[0, AA_ORDER.index("W")] = 0.5     # residue 1
    probs[2, AA_ORDER.index("W")] = 0.25    # residue 4
    node_logits = np.log(probs)

    logits = _scatter_to_positions(node_logits, res_ids, length=4, first_index=1)
    model = _model_with_logits(logits)

    # position 2 (residue 3) was dropped: no contribution either way
    assert model._position_score(2, "W", "A") == 0.0
    # its neighbours still carry their own distinct values
    assert model._position_score(0, "W", "A") == pytest.approx(
        np.log(0.5) - np.log(1 / 20)
    )
    assert model._position_score(3, "W", "A") == pytest.approx(
        np.log(0.25) - np.log(1 / 20)
    )
