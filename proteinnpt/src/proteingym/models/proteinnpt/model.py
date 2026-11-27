"""
Wrapper class around ProteinNPT (https://github.com/OATML-Markslab/ProteinNPT),
the non-parametric transformer of Notin et al., NeurIPS 2023
"""
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Self, Sequence

import numpy as np
import pandas as pd

from evedesign.model import Scorer, SupervisedBaseModel, assign_scores_to_instances
from evedesign.system import System, SystemInstance
from evedesign.types import DeviceType, StatusCallback
from evedesign.utils import status_done, status_start

# Upstream's single-property config (proteinnpt/utils/target_configs/fitness.json).
# Hardcoded rather than read from the installed package: it is three lines, and
# reading it would couple this wrapper to upstream's data directory layout.
TARGET_NAME = "fitness"
TARGET_VAR = "DMS_score"
TARGET_CONFIG = {
    TARGET_NAME: {
        "type": "continuous",
        "dim": 1,
        "var_name": TARGET_VAR,
        "in_NPT_loss": True,
        "main_target": True,
    }
}
FOLD_VAR = "train_test_split"


class ProteinNPT(SupervisedBaseModel, Scorer):
    """
    ProteinNPT regression on the variants of a single protein, over frozen ESM-2
    residue embeddings.
    """
    name: str = "proteinnpt"
    citations: list[str] = [
        # ProteinNPT: Improving Protein Property Prediction and Design with
        # Non-Parametric Transformers
        "10.1101/2023.12.06.570473",
        # Evolutionary-scale prediction of atomic-level protein structure with a
        # language model (ESM2)
        "10.1126/science.ade2574",
    ]

    requires_target: bool = True
    # the NPT attends across a batch of equal-length sequences
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    # upstream's Trainer is patched at image build time to honour this device
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    # embedder window (1024) less the BOS/EOS tokens it adds
    _max_target_length: int = 1022
    # Sequences embedded per forward pass when filling the cache. Independent of
    # the NPT's own batch sizes: embedding runs under no_grad, so only one
    # layer's activations are live, and a small chunk keeps that bounded.
    _embed_chunk: int = 8

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        embedder_checkpoint_path: str = "/opt/program/data/esm2_t33_650M_UR50D.pt",
        aa_embeddings: str = "ESM2_650M",
        num_total_training_steps: int = 10000,
        num_protein_npt_layers: int = 5,
        embed_dim: int = 200,
        attention_heads: int = 4,
        ffn_embed_dim: int = 400,
        target_prediction_model: str = "CNN",
        conv_kernel_size: int = 7,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        weight_decay: float = 0.005,
        max_learning_rate: float = 3e-4,
        num_warmup_steps: int = 100,
        training_num_assay_sequences_per_batch: int = 425,
        eval_num_sequences_to_score_per_batch: int = 15,
        eval_num_training_sequences_per_batch: int = 1000,
        max_positions: int = 1024,
        seed: int = 2023,
        device: DeviceType = "cpu",
    ):
        """
        Instantiate a ProteinNPT model.

        Parameters
        ----------
        embedder_checkpoint_path
            fair-esm checkpoint (.pt) for the frozen residue embedder, baked
            into the image at build time. Upstream loads it with its own
            vendored ESM, so a HuggingFace identifier will not work here
        aa_embeddings
            Upstream's name for the embedder family; must match the checkpoint
        num_total_training_steps
            Training steps per fold. Upstream's published config uses 10000
        num_protein_npt_layers, embed_dim, attention_heads, ffn_embed_dim
            Non-parametric transformer depth, width and head count
        target_prediction_model, conv_kernel_size
            Head mapping residue embeddings to the target
        dropout, attention_dropout, activation_dropout, weight_decay
            Regularisation
        max_learning_rate, num_warmup_steps
            Learning rate schedule
        training_num_assay_sequences_per_batch
            Sequences attended over jointly per training step. The NPT learns
            across a whole batch, so this is architectural, not just memory
        eval_num_sequences_to_score_per_batch
            Sequences scored per evaluation batch
        eval_num_training_sequences_per_batch
            Labelled training rows given as context at prediction time
        max_positions
            Longest sequence window the embedder sees
        seed
            Random seed
        device
            Device for both the embedder and the NPT
        """
        self.embedder_checkpoint_path = Path(embedder_checkpoint_path)
        self.aa_embeddings = aa_embeddings
        self.num_total_training_steps = num_total_training_steps
        self.max_positions = max_positions
        self.device = device
        self.seed = seed

        # Collected verbatim into upstream's args namespace. Upstream reads
        # these off a flat argparse object, so keeping them in one dict avoids
        # restating every name twice.
        self._config = {
            "aa_embeddings": aa_embeddings,
            "num_protein_npt_layers": num_protein_npt_layers,
            "embed_dim": embed_dim,
            "attention_heads": attention_heads,
            "ffn_embed_dim": ffn_embed_dim,
            "target_prediction_model": target_prediction_model,
            "conv_kernel_size": conv_kernel_size,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "activation_dropout": activation_dropout,
            "weight_decay": weight_decay,
            "max_learning_rate": max_learning_rate,
            "num_warmup_steps": num_warmup_steps,
            "training_num_assay_sequences_per_batch_per_gpu": (
                training_num_assay_sequences_per_batch
            ),
            "eval_num_sequences_to_score_per_batch_per_gpu": (
                eval_num_sequences_to_score_per_batch
            ),
            "eval_num_training_sequences_per_batch_per_gpu": (
                eval_num_training_sequences_per_batch
            ),
            "max_positions": max_positions,
            "seed": seed,
        }

        self._system: System | None = None
        self._trainer = None
        self._train_data = None
        self._target_processing = None
        self._emb_dir: Path | None = None
        self._emb_h5: Path | None = None
        self._n_train_emb: int = 0

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._trainer is not None

    @classmethod
    def can_model(cls, system: System, data) -> tuple[bool, str]:
        from evedesign.dataset import LabeledInstanceDataset

        if not isinstance(data, LabeledInstanceDataset):
            return False, "Model requires a LabeledInstanceDataset as data parameter"

        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle a single-component protein system"

        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have a defined rep sequence"

        target_rep = "".join(target.rep)
        # upstream slices longer sequences, but the NPT attends across a batch of
        # equal-length sequences, so refuse rather than silently window
        if len(target_rep) > cls._max_target_length:
            return False, (
                f"Sequence length ({len(target_rep)}) exceeds the embedder window "
                f"({cls._max_target_length})"
            )

        return True, ""

    def _args(self) -> Namespace:
        """Build the flat argparse namespace upstream reads its config from."""
        return Namespace(
            model_type="ProteinNPT",
            target_config=TARGET_CONFIG,
            # as above: upstream splits by this column when loading its own CSVs
            fold_variable_name=FOLD_VAR,
            embedding_model_location=str(self.embedder_checkpoint_path),
            # no zero-shot scores are available from the benchmark; upstream
            # states they are not strictly required
            augmentation="None",
            augmentation_short="none",
            # Embeddings are precomputed once by _precompute_embeddings and
            # read from this .h5. This is what keeps the embedder frozen: with
            # it set, forward() uses these tensors and never calls the embedder,
            # so its parameters stay out of the graph and out of the optimizer.
            sequence_embeddings_location=(
                [str(self._emb_h5)] if self._emb_h5 is not None else None
            ),
            sequence_embeddings_folder=None,
            training_fp16=False,
            # upstream imports wandb unconditionally but only logs when this
            # is set; the benchmark runs offline and forbids telemetry
            use_wandb=False,
            device=self.device,
            num_total_training_steps=self.num_total_training_steps,
            num_logging_training_steps=max(1, self.num_total_training_steps // 10),
            num_saving_training_steps=self.num_total_training_steps + 1,
            num_eval_steps=self.num_total_training_steps + 1,
            gradient_accumulation=1,
            eval_training_sequences_sampling_method="random",
            # upstream's final config value; the other heads expect extra inputs
            target_prediction_head="Target_embeddings_only",
            # only read by upstream's CSV loader, which this wrapper bypasses;
            # kept so the namespace stays complete if that path is ever reached
            test_fold_index=1,
            use_validation_set=False,
            # 0 workers on purpose: >0 forks dataloader workers, and forked pools
            # deadlock in this zoo's containers once torch has started threads
            num_data_loaders_workers=0,
            # optimiser defaults
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-8,
            grad_norm_clip=1.0,
            min_learning_rate=3e-5,
            label_smoothing=0.0,
            early_stopping_patience=None,
            # substitutions only; can_model refuses anything longer than the window
            indel_mode=False,
            long_sequences_slicing_method="center",
            max_tokens_per_msa=16384,
            num_MSA_sequences_per_training_instance=384,
            # no checkpointing: the benchmark trains per fold and discards
            save_model_checkpoint=False,
            model_location=None,
            # no zero-shot scores available from the benchmark
            zero_shot_fitness_predictions_location=None,
            zero_shot_fitness_predictions_var_name=None,
            clustalomega_path=None,
            **self._config,
        )

    def _dataset(self, sequences: Sequence[str], values, train: bool):
        """
        Shape sequences and labels the way upstream's Trainer expects.

        Upstream reaches this via get_train_val_test_data(), which reads CSVs off
        disk; the resulting object is a HuggingFace Dataset keyed by
        'mutant_mutated_seq_pairs' plus the target name. Built directly here so no
        CSV round-trip or data directory layout is needed.
        """
        from datasets import Dataset as HFDataset
        from proteinnpt.utils.data_utils import (
            preprocess_test_targets,
            preprocess_training_targets,
        )

        prefix = "train" if train else "test"
        pairs = [(f"{prefix}{i}", s) for i, s in enumerate(sequences)]
        raw = {TARGET_NAME: pd.Series([float(v) for v in values], dtype=float)}

        if train:
            raw, self._target_processing = preprocess_training_targets(
                raw, TARGET_CONFIG, verbose=False
            )
        else:
            raw = preprocess_test_targets(
                raw, TARGET_CONFIG, self._target_processing
            )

        return HFDataset.from_dict({
            "mutant_mutated_seq_pairs": pairs,
            TARGET_NAME: list(raw[TARGET_NAME]),
        })

    @staticmethod
    def _freeze_embedder(model) -> int:
        """
        Mark the residue embedder's parameters as not trainable.

        Precomputed embeddings already keep the embedder out of the autograd
        graph, so this is belt and braces - but upstream's create_optimizer
        collects `self.named_parameters()` with no requires_grad filter, so the
        intent is worth stating where it can be asserted. Returns the number of
        parameters frozen.
        """
        frozen = 0
        for param in model.aa_embedding.parameters():
            param.requires_grad = False
            frozen += param.numel()

        return frozen

    def _precompute_embeddings(self, model, pairs) -> None:
        """
        Embed each sequence once with the frozen embedder, into upstream's .h5.

        This is upstream's intended path, and the reason the embedder is frozen
        rather than merely described as such. With `sequence_embeddings_location`
        set, `process_batch` reads these tensors and `forward()` never calls the
        embedder. Leaving it unset embeds inside `forward()` on every training
        step, which re-does the work each step and, because upstream puts every
        parameter in the optimiser, fine-tunes the 651 M embedder as well.

        Tokenisation mirrors the NPT's own batch processor exactly (BOS + EOS),
        so the cached tensors line up with the tokens forward() is given.
        """
        import h5py
        import torch
        from proteinnpt.utils.embedding_utils import (
            get_embeddings_ESM,
            process_ESM_batch,
        )

        embedder = model.aa_embedding
        embedder.eval()
        names = [str(name) for name, _ in pairs]
        seqs = [str(seq) for _, seq in pairs]

        with h5py.File(self._emb_h5, "a") as h5f:
            for start in range(0, len(names), self._embed_chunk):
                sl = slice(start, start + self._embed_chunk)
                processed = process_ESM_batch(
                    batch={
                        "mutant_mutated_seq_pairs": list(zip(names[sl], seqs[sl]))
                    },
                    model=embedder,
                    alphabet=model.alphabet,
                    max_positions=self.max_positions,
                    long_sequences_slicing_method="center",
                    eval_mode=True,
                    start_idx=1,
                    # BOS + EOS, as the NPT's own batch processor passes
                    num_extra_tokens=2,
                )
                with torch.no_grad():
                    embeddings, _ = get_embeddings_ESM(
                        embedder,
                        self.aa_embeddings,
                        processed,
                        model.alphabet_size,
                        return_pseudo_ll=False,
                    )
                arr = embeddings.detach().to("cpu", torch.float32).numpy()
                chunk_names = np.array(names[sl], dtype=object)

                if "embeddings" not in h5f:
                    h5f.create_dataset(
                        "embeddings",
                        data=arr,
                        maxshape=(None, *arr.shape[1:]),
                        chunks=True,
                    )
                    h5f.create_dataset(
                        "mutants",
                        data=chunk_names,
                        maxshape=(None,),
                        dtype=h5py.string_dtype(),
                    )
                    continue

                emb, mut = h5f["embeddings"], h5f["mutants"]
                if emb.shape[1:] != arr.shape[1:]:
                    raise ValueError(
                        f"Embedding shape {arr.shape[1:]} does not match the "
                        f"cached {tuple(emb.shape[1:])}. All sequences must share "
                        "one length; requires_fixed_length is True."
                    )
                rows = emb.shape[0]
                emb.resize(rows + arr.shape[0], axis=0)
                emb[rows:] = arr
                mut.resize(rows + len(chunk_names), axis=0)
                mut[rows:] = chunk_names

    def _truncate_embeddings_to_train(self) -> None:
        """Drop cached test embeddings, keeping the training rows written by build()."""
        import h5py

        if self._emb_h5 is None or not self._emb_h5.exists():
            return
        with h5py.File(self._emb_h5, "a") as h5f:
            if "embeddings" not in h5f:
                return
            for key in ("embeddings", "mutants"):
                h5f[key].resize(self._n_train_emb, axis=0)

    def build(
        self,
        system: System,
        data,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        import torch
        from proteinnpt.proteinnpt.model import ProteinNPTModel
        from proteinnpt.utils.model_utils import Trainer

        self.can_model_or_raise(system, data)
        status_start(status_callback, "Fitting ProteinNPT model")

        self._system = system
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if not self.embedder_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Embedder checkpoint not found at {self.embedder_checkpoint_path}. "
                "It is baked into the image at build time; see the Dockerfile."
            )

        # single target series, so select(None) picks it
        instances, values, _, _ = data.select(None, drop_missing=True)
        sequences = ["".join(inst[0].rep) for inst in instances]
        self._train_data = self._dataset(sequences, values, train=True)

        # The cache path has to be in args before the model is built: upstream's
        # process_batch reads it per batch and forward() branches on it.
        self._emb_dir = Path(tempfile.mkdtemp(prefix="proteinnpt-emb-"))
        self._emb_h5 = self._emb_dir / "embeddings.h5"

        args = self._args()
        model = ProteinNPTModel(args, self._alphabet())
        self._freeze_embedder(model)
        model.to(self.device)
        model.set_device()

        self._precompute_embeddings(
            model, self._train_data["mutant_mutated_seq_pairs"]
        )
        self._n_train_emb = len(self._train_data)

        self._trainer = Trainer(
            model=model,
            args=args,
            train_data=self._train_data,
            val_data=None,
            # produced by preprocess_training_targets in _dataset(); the
            # batch processor indexes it per target name
            target_processing=self._target_processing,
        )
        self._trainer.train()

        status_done(status_callback)
        return self

    def _alphabet(self):
        """Alphabet of the frozen embedder, as upstream's model constructor wants."""
        import argparse

        import torch
        from proteinnpt.utils.esm.pretrained import load_model_and_alphabet

        # fair-esm checkpoints pickle an argparse.Namespace, which torch>=2.6's
        # weights_only=True default refuses. Allowlist that one global rather
        # than disabling the integrity check wholesale.
        torch.serialization.add_safe_globals([argparse.Namespace])
        _, alphabet = load_model_and_alphabet(str(self.embedder_checkpoint_path))
        return alphabet

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score instances with the trained NPT, conditioning on the labelled
        training rows as context (the model is non-parametric at inference).
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        if len(instances) == 0:
            return []

        status_start(status_callback, "Scoring with ProteinNPT")
        sequences = ["".join(inst[0].rep) for inst in instances]
        # eval masks every input target, so these placeholders are never read
        test_data = self._dataset(sequences, [0.0] * len(sequences), train=False)

        # eval conditions on the training rows, so the cache must hold both. Test
        # ids are positional ("test0"...), so a second score() call would reuse
        # names already cached against different sequences; upstream matches
        # embeddings to batches by name, so drop the old test rows first.
        self._truncate_embeddings_to_train()
        self._precompute_embeddings(
            self._trainer.model, test_data["mutant_mutated_seq_pairs"]
        )

        results = self._trainer.eval(
            test_data=test_data,
            output_all_predictions=True,
            train_data=self._train_data,
        )
        scores = self._extract_predictions(
            results, [f"test{i}" for i in range(len(sequences))]
        )

        # upstream standardises continuous targets when fitting and never
        # inverts, so map back to the assay's units: the benchmark compares
        # these against the raw target column, and the sklearn-based models
        # here predict in those units too
        stats = self._target_processing[TARGET_NAME]
        scores = scores * stats["std"] + stats["mean"]

        status_done(status_callback)
        return assign_scores_to_instances(instances, scores.astype(float))

    @staticmethod
    def _extract_predictions(results, mutant_ids: Sequence[str]) -> np.ndarray:
        """
        Pull the per-sequence predictions out of upstream's eval results, in the
        order the instances were given.

        eval() returns a dict whose 'output_scores' is a DataFrame assembled
        from the eval batches and then merged on 'mutant', so its row order
        follows batch order rather than ours. Reindex on the ids assigned in
        _dataset() instead of trusting that order.
        """
        if not isinstance(results, dict) or "output_scores" not in results:
            raise KeyError(
                "ProteinNPT eval returned no 'output_scores'; got "
                f"{sorted(results) if isinstance(results, dict) else type(results)}"
            )

        frame = results["output_scores"]
        column = f"predictions_{TARGET_NAME}"
        if column not in frame.columns:
            raise KeyError(
                f"No '{column}' in ProteinNPT eval output; found "
                f"{sorted(frame.columns)}"
            )

        preds = frame.set_index("mutant")[column].reindex(mutant_ids)
        if preds.isna().any():
            raise ValueError(
                f"ProteinNPT returned no prediction for {int(preds.isna().sum())} "
                f"of {len(mutant_ids)} sequences"
            )

        return preds.to_numpy(dtype=float)
