"""Checks the alignment preparation that conditions the model."""

import pytest

from evedesign.sequence import Sequence, Sequences
from proteingym.models.msa_transformer.model import MSATransformer

TARGET = "ACDEF"


def msa(*rows):
    return Sequences(
        seqs=[Sequence(seq=r, id=None) for r in rows],
        aligned=True,
        type="protein",
        format="a3m",
    )


def model(**kwargs):
    return MSATransformer(checkpoint_path="/nonexistent.pt", **kwargs)


def test_insertions_are_dropped_so_rows_span_the_target():
    # lowercase columns are insertions relative to the target and must go.
    # These rows differ from the target so they stay context rows rather than
    # taking the dedup path the next test covers.
    rows = model()._alignment(TARGET, msa("AgCDEG", "ACDqEH"))

    assert [seq for _, seq in rows] == [TARGET, "ACDEG", "ACDEH"]


def test_the_target_is_the_query_row_without_duplicating_it():
    # the dataset's first row is usually the target already
    rows = model()._alignment(TARGET, msa(TARGET, "ACDEG"))

    assert rows[0] == ("target", TARGET)
    assert [seq for _, seq in rows] == [TARGET, "ACDEG"]


def test_rows_are_subsampled_to_n_seqs():
    rows = model(n_seqs=3)._alignment(TARGET, msa("ACDEG", "ACDEH", "ACDEI", "ACDEK"))

    assert len(rows) == 3
    assert rows[0][1] == TARGET


def test_a_misaligned_row_fails_loudly():
    # a row that does not span the target would silently shift every column
    with pytest.raises(ValueError, match="does not span the target"):
        model()._alignment(TARGET, msa("ACDEF", "ACD"))


def test_the_bos_offset_lines_up_with_the_target():
    # build() masks and reads logits at column pos+1; if that offset were wrong
    # the profile would be shifted and every score silently off
    import esm

    alphabet = esm.Alphabet.from_architecture("msa_transformer")
    rows = model()._alignment(TARGET, msa(TARGET, "ACDEG"))
    _, _, tokens = alphabet.get_batch_converter()([rows])

    assert int(tokens[0, 0, 0]) == alphabet.cls_idx
    assert [int(tokens[0, 0, p + 1]) for p in range(len(TARGET))] == [
        alphabet.get_idx(aa) for aa in TARGET
    ]
