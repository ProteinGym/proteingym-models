"""Checks the backbone extraction that conditions the model, and structure selection."""

from types import SimpleNamespace

import biotite.structure as struc
import numpy as np
import pytest
from biotite.sequence import ProteinSequence

from evedesign.structure import Structure
from proteingym.models.mif_st.model import (
    BACKBONE_ATOMS,
    MIFST,
    _extract_backbone_coords,
)

TARGET = "ACDEF"


def chain(sequence=TARGET, first_index=1, missing=(), atom_names=("N", "CA", "C")):
    """Single-chain Structure with one distinguishable coordinate per atom."""
    atoms = []
    for offset, aa in enumerate(sequence):
        res_id = first_index + offset
        if res_id in missing:
            continue
        for axis, name in enumerate(atom_names):
            atoms.append(
                struc.Atom(
                    [float(res_id), float(axis), 0.0],
                    chain_id="A",
                    res_id=res_id,
                    res_name=ProteinSequence.convert_letter_1to3(aa),
                    atom_name=name,
                    element=name[0],
                )
            )
    return Structure(struc.array(atoms))


def test_coords_land_at_rep_offsets_not_residue_ids():
    # a structure numbered from 101 must still align with rep position 0
    coords = _extract_backbone_coords(
        chain(first_index=101), first_index=101, length=len(TARGET), rep=TARGET
    )

    assert coords.shape == (len(TARGET), 3, 3)
    assert np.isfinite(coords).all()
    # res_id 101 -> offset 0, and N/CA/C keep process_coords' expected order
    assert coords[0, 0].tolist() == [101.0, 0.0, 0.0]
    assert coords[0, 1].tolist() == [101.0, 1.0, 0.0]
    assert coords[0, 2].tolist() == [101.0, 2.0, 0.0]


def test_missing_residues_become_nan_rather_than_shifting_the_rest():
    # an unresolved residue must leave a NaN hole, not slide later positions up
    coords = _extract_backbone_coords(
        chain(missing=(3,)), first_index=1, length=len(TARGET), rep=TARGET
    )

    assert np.isnan(coords[2]).all()
    assert np.isfinite(np.delete(coords, 2, axis=0)).all()
    assert coords[3, 0].tolist() == [4.0, 0.0, 0.0]


def test_a_structure_without_backbone_atoms_fails_loudly():
    with pytest.raises(ValueError, match="no peptide backbone"):
        _extract_backbone_coords(
            chain(atom_names=("CB",)), first_index=1, length=len(TARGET), rep=TARGET
        )


def test_a_structure_disagreeing_with_rep_fails_loudly():
    # silently scoring against the wrong backbone is the failure to avoid
    with pytest.raises(ValueError):
        _extract_backbone_coords(
            chain(sequence="ACDEW"), first_index=1, length=len(TARGET), rep=TARGET
        )


def test_multi_chain_structures_are_rejected():
    # MIF-ST graphs one backbone; it has no multi-chain context path
    model = MIFST()
    target = SimpleNamespace(structures={"x": [chain(), chain()]})

    with pytest.raises(ValueError, match="single chain"):
        model._select_structure_chain(target)


def test_structure_name_must_resolve():
    model = MIFST(structure_name="absent")
    target = SimpleNamespace(structures={"present": chain()})

    with pytest.raises(ValueError, match="not found in entity"):
        model._select_structure_chain(target)


def test_several_structures_need_an_explicit_choice():
    model = MIFST()
    target = SimpleNamespace(structures={"a": chain(), "b": chain()})

    with pytest.raises(ValueError, match="more than one structure"):
        model._select_structure_chain(target)


def test_collated_tokens_align_one_to_one_with_residues():
    # score() gathers log-probs at src[:, i] for residue i. An off-by-one here
    # (a BOS token, as MSA Transformer has) would shift every score silently.
    # Built exactly as sequence_models.pretrained builds it, but weight-free.
    import torch
    from sequence_models.collaters import SimpleCollater, StructureCollater
    from sequence_models.constants import PROTEIN_ALPHABET
    from sequence_models.pdb_utils import process_coords

    collater = StructureCollater(
        SimpleCollater(PROTEIN_ALPHABET, pad=True), n_connections=30
    )
    ell = len(TARGET)
    xyz = np.random.default_rng(0).normal(scale=5.0, size=(ell, 3, 3))
    feats = [
        torch.tensor(f, dtype=torch.float)
        for f in process_coords({n: xyz[:, a] for a, n in enumerate(BACKBONE_ATOMS)})
    ]
    src, nodes, _, _, _ = collater([[TARGET, *feats]])

    assert src.shape == (1, ell)
    assert [int(t) for t in src[0]] == [PROTEIN_ALPHABET.index(aa) for aa in TARGET]
    assert nodes.shape[1] == ell
