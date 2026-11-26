"""Checks the args namespace, the data shaping and can_model's guards.

Deliberately avoids the embedder checkpoint so these run without the 2.5 GB
download: only _alphabet() and build() need it.
"""

import re
from pathlib import Path

import pytest


from proteingym.models.proteinnpt.model import TARGET_NAME, ProteinNPT

# the modules whose args.<field> reads must all be satisfied
UPSTREAM_MODULES = (
    "proteinnpt/model.py",
    "utils/model_utils.py",
    "proteinnpt/data_processing.py",
    "utils/data_utils.py",
    "utils/embedding_utils.py",
)


def model():
    return ProteinNPT(embedder_checkpoint_path="/nonexistent.pt")


def test_args_cover_every_field_upstream_reads():
    # this caught 20 missing fields while writing the wrapper; upstream reads its
    # config off a flat namespace, so a missing name is an AttributeError at runtime
    import proteinnpt

    root = Path(proteinnpt.__file__).parent
    used = set()
    for rel in UPSTREAM_MODULES:
        used |= set(
            re.findall(r"args\.([a-zA-Z_][a-zA-Z0-9_]*)", (root / rel).read_text())
        )

    provided = set(vars(model()._args()))
    assert not (used - provided), f"args missing: {sorted(used - provided)}"


def test_telemetry_is_off_and_no_zero_shot_scores_claimed():
    args = model()._args()
    # upstream only calls wandb.log when use_wandb is set; the benchmark is offline
    assert args.use_wandb is False
    # the benchmark supplies no zero-shot predictions
    assert args.augmentation == "None"
    assert args.zero_shot_fitness_predictions_location is None
    # >0 dataloader workers fork, and forked pools deadlock in these containers
    assert args.num_data_loaders_workers == 0


def test_dataset_has_the_keys_upstream_indexes():
    m = model()
    ds = m._dataset(["MKTA", "MKTV"], [0.5, -0.5], train=True)

    assert set(ds.column_names) == {"mutant_mutated_seq_pairs", TARGET_NAME}
    assert ds[0]["mutant_mutated_seq_pairs"][1] == "MKTA"
    # training preprocessing must hand back a processing object for the test split
    assert m._target_processing is not None


def test_test_split_reuses_the_training_target_processing():
    m = model()
    m._dataset(["MKTA", "MKTV"], [0.5, -0.5], train=True)
    before = m._target_processing
    m._dataset(["MKTD"], [0.0], train=False)

    assert m._target_processing is before, "test split must not refit the processing"


def test_can_model_rejects_a_sequence_past_the_embedder_window():
    # evedesign's Protein() already rejects non-standard and lowercase residues at
    # construction, so length is the guard that is actually reachable here
    from evedesign.dataset import LabeledInstanceDataset
    from evedesign.system import EntityInstance, Protein, System, SystemInstance

    rep = "A" * (ProteinNPT._max_target_length + 1)
    system = System([Protein(first_index=1, rep=rep)])
    data = LabeledInstanceDataset(
        instances=[SystemInstance([EntityInstance(rep=rep)])], labels={"y": [0.0]}
    )

    ok, why = ProteinNPT.can_model(system, data)
    assert ok is False and "exceeds the embedder window" in why


def test_can_model_accepts_a_plain_protein_system():
    from evedesign.dataset import LabeledInstanceDataset
    from evedesign.system import EntityInstance, Protein, System, SystemInstance

    rep = "MKTAYIAK"
    system = System([Protein(first_index=1, rep=rep)])
    data = LabeledInstanceDataset(
        instances=[SystemInstance([EntityInstance(rep=rep)])], labels={"y": [0.0]}
    )

    assert ProteinNPT.can_model(system, data)[0] is True


def test_can_model_requires_a_labelled_dataset():
    from evedesign.system import Protein, System

    ok, why = ProteinNPT.can_model(System([Protein(first_index=1, rep="MKTA")]), None)
    assert ok is False and "LabeledInstanceDataset" in why


def test_card_hyperparameters_are_all_constructor_arguments():
    """The runner feeds the card's hyper_parameters straight to __init__.

    A renamed or stale card key is a TypeError inside the container, which is an
    expensive place to find out.
    """
    import inspect
    import re

    import yaml

    card = Path(__file__).resolve().parent.parent / "README.md"
    front = re.match(r"^---\n(.*?)\n---\n", card.read_text(), re.S)
    assert front, "model card has no YAML front matter"
    hyper = yaml.safe_load(front.group(1))["hyper_parameters"]

    accepted = set(inspect.signature(ProteinNPT.__init__).parameters) - {"self"}
    unknown = sorted(set(hyper) - accepted)
    assert not unknown, f"card keys not in __init__: {unknown}"

    # and the card must actually construct
    ProteinNPT(**hyper)


def _eval_results(ids, preds):
    """Minimal stand-in for upstream's eval() return."""
    import pandas as pd

    return {
        "output_scores": pd.DataFrame(
            {"mutant": list(ids), f"predictions_{TARGET_NAME}": list(preds)}
        )
    }


def test_predictions_are_realigned_to_the_instance_order():
    # upstream merges output_scores on 'mutant', so row order follows eval batch
    # order, not ours. Rows are handed back shuffled here on purpose.
    results = _eval_results(["test2", "test0", "test1"], [20.0, 0.0, 10.0])

    got = ProteinNPT._extract_predictions(results, ["test0", "test1", "test2"])

    assert list(got) == [0.0, 10.0, 20.0]


def test_missing_prediction_is_an_error_not_a_silent_nan():
    results = _eval_results(["test0", "test1"], [1.0, 2.0])

    with pytest.raises(ValueError, match="no prediction for 1"):
        ProteinNPT._extract_predictions(results, ["test0", "test1", "test2"])


def test_eval_output_without_a_prediction_column_fails_loudly():
    import pandas as pd

    results = {"output_scores": pd.DataFrame({"mutant": ["test0"], "labels": [1.0]})}

    with pytest.raises(KeyError, match="predictions_fitness"):
        ProteinNPT._extract_predictions(results, ["test0"])


def test_train_and_test_ids_do_not_collide():
    # upstream asserts output_scores['mutant'] is unique, and an eval batch
    # carries training context rows alongside the scored ones
    m = model()
    train = m._dataset(["MKTA", "MKTV"], [0.5, -0.5], train=True)
    test = m._dataset(["MKTD"], [0.0], train=False)

    train_ids = {p[0] for p in train["mutant_mutated_seq_pairs"]}
    test_ids = {p[0] for p in test["mutant_mutated_seq_pairs"]}
    assert not (train_ids & test_ids)


def test_embeddings_are_read_from_a_cache_not_computed_in_forward():
    # The load-bearing switch. With sequence_embeddings_location unset, upstream
    # embeds inside forward(), which puts the 651 M embedder in the autograd
    # graph and in create_optimizer's parameter list - so the "frozen" embedder
    # is fine-tuned, costing ~9.8 GB of optimiser state and ~1.35 GB of retained
    # activations per batch element. Set, forward() uses the cached tensors and
    # never calls the embedder (upstream proteinnpt/model.py:244).
    m = model()
    assert m._args().sequence_embeddings_location is None

    m._emb_h5 = Path("/tmp/embeddings.h5")
    assert m._args().sequence_embeddings_location == ["/tmp/embeddings.h5"]


CHECKPOINT = Path("/opt/program/data/esm2_t33_650M_UR50D.pt")


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="embedder checkpoint is only present inside the image",
)
def test_only_the_npt_layers_are_trainable():
    # Guards the claim the card makes. The embedder is 651 M of the model's
    # 656 M parameters, so if it ever becomes trainable again the model silently
    # turns into an ESM-2 fine-tune: 130x the trainable surface, and it OOM'd a
    # 96 GB allocation on a 465-residue target.
    from proteinnpt.proteinnpt.model import ProteinNPTModel

    m = ProteinNPT(device="cpu")
    m._emb_h5 = Path("/tmp/unused.h5")
    npt = ProteinNPTModel(m._args(), m._alphabet())

    frozen = m._freeze_embedder(npt)
    total = sum(p.numel() for p in npt.parameters())
    trainable = sum(p.numel() for p in npt.parameters() if p.requires_grad)

    assert frozen > 600_000_000, f"embedder unexpectedly small: {frozen}"
    assert not any(p.requires_grad for p in npt.aa_embedding.parameters())
    assert trainable == total - frozen
    # the NPT layers and target head only
    assert trainable < 10_000_000, f"{trainable} trainable params is too many"


def test_cache_layout_is_what_upstream_reads(tmp_path):
    # Upstream resolves embeddings by mutant NAME, with a pandas inner merge
    # (utils/data_utils.get_indices_retrieved_embeddings). It asserts only on
    # count, not order, so a layout mismatch would pair the wrong embedding with
    # a sequence and still look fine. Pin the contract the writer must satisfy.
    import h5py
    import numpy as np
    from proteinnpt.utils.data_utils import get_indices_retrieved_embeddings

    names = ["train0", "train1", "train2"]
    h5_path = tmp_path / "embeddings.h5"
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset(
            "embeddings",
            data=np.zeros((3, 6, 4), dtype=np.float32),
            maxshape=(None, 6, 4),
            chunks=True,
        )
        h5f.create_dataset(
            "mutants",
            data=np.array(names, dtype=object),
            maxshape=(None,),
            dtype=h5py.string_dtype(),
        )

    # a batch in a different order to the cache, as eval batching produces
    batch = {"mutant_mutated_seq_pairs": [("train2", "MKTA"), ("train0", "MKTV")]}
    indices = get_indices_retrieved_embeddings(batch, str(h5_path))

    assert list(indices) == [2, 0], "cache rows must resolve by name, in batch order"
