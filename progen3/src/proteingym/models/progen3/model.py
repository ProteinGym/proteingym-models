"""
Wrapper class around ProGen3 (https://github.com/Profluent-AI/progen3)
"""
from pathlib import Path
from os import PathLike
from typing import Literal, Self, Sequence

import numpy as np
import torch

from evedesign.model import BaseModel, Scorer, assign_scores_to_instances
from evedesign.system import System, SystemInstance
from evedesign.types import BatchSize, DeviceType, StatusCallback
from evedesign.utils import model_param_context

from progen3.common.model_loading import get_model
from progen3.modeling import ProGen3ForCausalLM
from progen3.scorer import ProGen3Scorer

ModelName = Literal[
    "Profluent-Bio/progen3-3b",
    "Profluent-Bio/progen3-1b",
    "Profluent-Bio/progen3-762m",
    "Profluent-Bio/progen3-339m",
    "Profluent-Bio/progen3-219m",
    "Profluent-Bio/progen3-112m",
]


class ProGen3(BaseModel, Scorer):
    """
    Wrapper around ProGen3's bidirectional whole-sequence log-likelihood scoring.
    """
    name: str = "ProGen3"
    citations: list[str] = [
        # ProGen3 (GitHub repository)
        "https://github.com/Profluent-AI/progen3",
    ]

    requires_target: bool = True
    requires_fixed_length: bool = False
    # indels are scored on the normalized (concrete) sequence
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = True
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = None

    # 64k position embeddings minus <bos>, <eos> and two direction-marker tokens
    MAX_SEQ_LENGTH: int = 1024 * 64 - 4

    def __init__(
        self,
        model_name: ModelName = "Profluent-Bio/progen3-762m",
        model_dir_path: str | PathLike | None = None,
        batch_size: BatchSize = 64,
        max_batch_tokens: int = 65536,
        device: DeviceType = "cuda",
    ):
        """
        Instantiate a ProGen3 model.

        Parameters
        ----------
        model_name
            HuggingFace Hub checkpoint to load
        model_dir_path
            Local checkpoint directory; takes precedence over model_name
        batch_size
            Number of sequences passed to ProGen3Scorer at once
        max_batch_tokens
            Maximum number of tokens per forward pass (bounds GPU memory)
        device
            Device to run the model on
        """
        self.model_name = model_name
        self.model_dir_path = Path(model_dir_path) if model_dir_path is not None else None
        self.device = device
        self.max_batch_tokens = max_batch_tokens

        if batch_size != "auto" and batch_size < 1:
            raise ValueError("batch_size must be at least 1 or 'auto'")
        if batch_size == "auto":
            raise NotImplementedError("Automatic batch_size not yet implemented")
        self.batch_size = batch_size

        # keep the model loaded across score() calls
        self.keep_model_after_pred = True

        self._system: System | None = None
        self.model: ProGen3ForCausalLM | None = None
        self.scorer: ProGen3Scorer | None = None

    @property
    def ready(self) -> bool:
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle a single-component protein system"

        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have a defined rep sequence"

        if len(target.rep) > cls.MAX_SEQ_LENGTH:
            return False, (
                f"Sequence length ({len(target.rep)}) exceeds maximum allowed "
                f"({cls.MAX_SEQ_LENGTH})"
            )

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)

        # unconditional model: nothing to fit
        self._system = system

        return self

    def _load_model(self) -> None:
        if self.model is not None:
            return

        name_to_load = str(self.model_dir_path) if self.model_dir_path is not None else self.model_name
        self.model = get_model(
            model_name_or_path=name_to_load,
            model_class=ProGen3ForCausalLM,
            fsdp=False,
            dtype=torch.bfloat16,
        )
        self.model.to(self.device)
        self.model.eval()

        # "sum" rather than the default "mean" so length changes from indels count
        self.scorer = ProGen3Scorer(self.model, max_batch_tokens=self.max_batch_tokens, reduction="sum")

    def _delete_model(self) -> None:
        self.model = None
        self.scorer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score full sequences by their summed log-likelihood, averaged over N->C and C->N.

        Parameters
        ----------
        instances
            Instances to score
        status_callback
            Optional status callback

        Returns
        -------
        list[SystemInstance]
            Instances with scores assigned
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        # tokenizer has no gap/insert symbols, so drop gaps and uppercase inserts
        sequences = ["".join(instance[0].normalized_rep()) for instance in instances]

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            all_scores: list[float] = []
            for batch_start in range(0, len(sequences), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(sequences))
                batch_seqs = sequences[batch_start:batch_end]

                if status_callback:
                    progress = (batch_start / len(sequences)) * 100
                    status_callback("running", progress, f"Scoring batch {batch_start // self.batch_size + 1}")

                batch_results = self.scorer.score_batch(batch_seqs)
                all_scores.extend(batch_results["log_likelihood"])

        return assign_scores_to_instances(instances, np.asarray(all_scores, dtype=float))
