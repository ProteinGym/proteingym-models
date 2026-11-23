"""
Wrapper class around Tranception (https://github.com/OATML-Markslab/Tranception)
"""
import tempfile
from os import PathLike
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast

from evedesign.model import BaseModel, Scorer, MutationScorer, ConditionalMutationScorer, assign_scores_to_instances
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback
from evedesign.utils import status_done, status_start

from evcouplings.align.alignment import Alignment

from .tranception import config as tranception_config
from .tranception import model_pytorch
from .tranception.utils import msa_utils

_DIR = Path(__file__).parent
_TOKENIZER_FILE = _DIR / "tranception" / "utils" / "tokenizers" / "Basic_tokenizer"
_CHECKPOINT_SIZES = ("Small", "Medium", "Large")


class Tranception(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around Tranception zero-shot scoring, with optional MSA retrieval.
    """
    name: str = "Tranception"
    # Tranception: Protein Fitness Prediction with Autoregressive Transformers
    # and Inference-time Retrieval
    citations: list[str] = ["arXiv:2205.13760"]
    available = True

    requires_target: bool = True
    requires_fixed_length: bool = False
    # autoregressive whole-sequence scoring, so indels are scored directly
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = ["sequences"]

    def __init__(
        self,
        checkpoint_dir: str | PathLike = "/opt/program/data",
        checkpoint_size: str = "Large",
        inference_time_retrieval: bool = False,
        retrieval_inference_weight: float = 0.6,
        scoring_mirror: bool = True,
        batch_size_inference: int = 20,
        num_workers: int = 4,
        clustal_omega_binary: str = "clustalo",
        device: str = "cuda",
    ):
        """
        Instantiate a Tranception model.

        Parameters
        ----------
        checkpoint_dir
            Directory containing one HuggingFace-style subdirectory per
            checkpoint size (Small/Medium/Large)
        checkpoint_size
            Checkpoint to load: "Small" (~85M params), "Medium" (~300M) or
            "Large" (~700M)
        inference_time_retrieval
            Whether to blend the transformer's log-probabilities with an
            MSA-derived per-position prior. Requires the entity to carry an MSA
        retrieval_inference_weight
            Weight (alpha) of the retrieval prior when aggregating with the
            transformer log-probabilities
        scoring_mirror
            Whether to score each sequence in both directions and average
        batch_size_inference
            Batch size for scoring; forced to 1 for indels with retrieval,
            which upstream requires
        num_workers
            Number of workers for the scoring data loader
        clustal_omega_binary
            Path to / name of the Clustal Omega binary, used only to realign
            indel sequences against the MSA when retrieval is enabled
        device
            Device to run the model on ("cuda" or "cpu")
        """
        if checkpoint_size not in _CHECKPOINT_SIZES:
            raise ValueError(f"checkpoint_size must be one of {_CHECKPOINT_SIZES}, got {checkpoint_size!r}")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_size = checkpoint_size
        self.inference_time_retrieval = inference_time_retrieval
        self.retrieval_inference_weight = retrieval_inference_weight
        self.scoring_mirror = scoring_mirror
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.clustal_omega_binary = clustal_omega_binary
        self.device = device

        self._system: System | None = None
        self._model = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._msa_path: Path | None = None
        self._weights_path: Path | None = None

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._model is not None

    def ready_or_raise(self) -> None:
        if not self.ready:
            raise ValueError("Must call build() first to use model")

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"
        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle single-component protein system"
        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have defined rep sequence"
        return True, ""

    @staticmethod
    def _match_state_seqs(target) -> tuple[str, list[str]]:
        """
        Convert the MSA to uppercase match-state sequences, target first.

        Parameters
        ----------
        target
            Target entity carrying the MSA

        Returns
        -------
        tuple[str, list[str]]
            Target sequence and match-state MSA sequences
        """
        target_rep = "".join(target.rep)
        length = len(target_rep)
        raw_seqs = target.sequences.to_a3m().remove_inserts().seqs
        seqs = [s.seq.upper() for s in raw_seqs]
        bad = [i for i, s in enumerate(seqs) if len(s) != length]
        if bad:
            raise ValueError(
                f"MSA match-state length does not match target length ({length}) "
                f"for {len(bad)} sequence(s), e.g. sequence index {bad[0]}"
            )
        if seqs[0] != target_rep:
            raise ValueError(
                "First MSA sequence (match states) must equal the target/focus sequence"
            )
        return target_rep, seqs

    def _prepare_retrieval(self, target, target_rep: str) -> None:
        """
        Write a temporary MSA fasta and sequence weights (.npy) for retrieval.

        Upstream's retrieval prior reads both from files. Weights are computed
        with evcouplings (faster than upstream) on the same filtered sequence
        set upstream will reload, so rows line up.

        Parameters
        ----------
        target
            Target entity carrying the MSA
        target_rep
            Target sequence
        """
        _, seqs = self._match_state_seqs(target)
        length = len(target_rep)

        self._tmpdir = tempfile.TemporaryDirectory(prefix="tranception_msa_")
        tmp_dir = Path(self._tmpdir.name)
        self._msa_path = tmp_dir / "msa.fasta"
        with open(self._msa_path, "w") as f:
            f.write(f">focus/1-{length}\n{seqs[0]}\n")
            for i, s in enumerate(seqs[1:], start=1):
                f.write(f">seq{i}\n{s}\n")

        # Reproduce upstream's filtering to get the sequence set/order it will
        # see, skipping its slow weight computation
        filtering = msa_utils.MSA_processing(
            MSA_location=str(self._msa_path), use_weights=False, preprocess_MSA=True,
        )
        filtered_seqs = {
            name: "".join(cols) for name, cols in filtering.seq_name_to_sequence.items()
        }

        alignment = Alignment.from_dict(filtered_seqs)
        alignment.set_weights(identity_threshold=0.8, method="nogaps")

        self._weights_path = tmp_dir / "weights.npy"
        np.save(self._weights_path, alignment.weights)

    def build(self, system: System, data: Any = None, status_callback: StatusCallback | None = None) -> Self:
        self.can_model_or_raise(system, data)
        status_start(status_callback, "Loading Tranception model")

        self._system = system
        target = system[0]
        target_rep = "".join(target.rep)

        if self.inference_time_retrieval and target.sequences is None:
            raise ValueError(
                "inference_time_retrieval=True requires the system's entity to carry an MSA (sequences)"
            )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(_TOKENIZER_FILE),
            unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
            cls_token="[CLS]", mask_token="[MASK]",
        )

        checkpoint_path = self.checkpoint_dir / self.checkpoint_size
        with open(checkpoint_path / "config.json") as f:
            import json
            config_dict = json.load(f)
        config = tranception_config.TranceptionConfig(**config_dict)
        config.attention_mode = "tranception"
        config.position_embedding = "grouped_alibi"
        config.tokenizer = tokenizer
        config.scoring_window = "optimal"

        if self.inference_time_retrieval:
            self._prepare_retrieval(target, target_rep)
            config.retrieval_aggregation_mode = "aggregate_substitution"
            config.MSA_filename = str(self._msa_path)
            config.MSA_weight_file_name = str(self._weights_path)
            config.MSA_start = 0
            config.MSA_end = len(target_rep)
            config.full_protein_length = len(target_rep)
            config.retrieval_inference_weight = self.retrieval_inference_weight
            config.clustal_omega_location = self.clustal_omega_binary
        else:
            config.retrieval_aggregation_mode = None

        model = model_pytorch.TranceptionLMHeadModel.from_pretrained(
            pretrained_model_name_or_path=str(checkpoint_path), config=config,
        )
        if self.device == "cuda" and torch.cuda.is_available():
            model.cuda()
        model.eval()
        self._model = model

        status_done(status_callback, "Tranception model loaded")
        return self

    def _score_group(self, seqs: list[str], target_rep: str, indel_mode: bool) -> dict[str, float]:
        """
        Score a group of unique sequences with upstream score_mutants().

        Parameters
        ----------
        seqs
            Unique sequences, all substitutions or all indels
        target_rep
            Target sequence
        indel_mode
            Whether the group contains indels

        Returns
        -------
        dict[str, float]
            Score per sequence
        """
        if self.inference_time_retrieval:
            self._model.retrieval_aggregation_mode = "aggregate_indel" if indel_mode else "aggregate_substitution"

        # upstream forward() asserts batch size 1 for indels with retrieval
        batch_size = 1 if (indel_mode and self.inference_time_retrieval) else self.batch_size_inference

        dms_data = pd.DataFrame({"mutated_sequence": seqs})
        if not indel_mode:
            # Upstream needs a 'mutant' column ("A123G:B456C") to center the
            # scoring window; WT gets a self-substitution placeholder
            dms_data["mutant"] = [
                ":".join(
                    f"{ref}{i + 1}{alt}"
                    for i, (ref, alt) in enumerate(zip(target_rep, seq))
                    if ref != alt
                ) or f"{target_rep[0]}1{target_rep[0]}"
                for seq in seqs
            ]
        result = self._model.score_mutants(
            DMS_data=dms_data,
            target_seq=target_rep,
            scoring_mirror=self.scoring_mirror,
            batch_size_inference=batch_size,
            num_workers=self.num_workers,
            indel_mode=indel_mode,
        )
        return dict(zip(result["mutated_sequence"], result["avg_score"]))

    def score(
        self, instances: Sequence[SystemInstance], status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score sequences as the log-likelihood ratio to the target, averaged
        over both directions if scoring_mirror is set.

        Parameters
        ----------
        instances
            Instances to score
        status_callback
            Optional callback for progress updates

        Returns
        -------
        list[SystemInstance]
            Instances with scores assigned
        """
        self.ready_or_raise()
        self._validate_instances(instances)
        status_start(status_callback, "Scoring sequences")

        target_rep = "".join(self._system[0].rep)
        target_len = len(target_rep)

        seqs = ["".join(instance[0].normalized_rep()) for instance in instances]
        is_indel = [len(s) != target_len for s in seqs]

        # Deduplicate: score_mutants() joins its two directions on
        # mutated_sequence, which breaks on duplicate sequences
        score_by_seq: dict[str, float] = {}
        for mode in (False, True):
            group_seqs = list(dict.fromkeys(s for s, ind in zip(seqs, is_indel) if ind == mode))
            if group_seqs:
                score_by_seq.update(self._score_group(group_seqs, target_rep, indel_mode=mode))

        scores = [score_by_seq[s] for s in seqs]
        status_done(status_callback, "Scoring complete")
        return assign_scores_to_instances(instances, scores)
