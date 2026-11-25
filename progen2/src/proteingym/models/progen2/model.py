"""
Wrapper class around ProGen2 (https://github.com/salesforce/progen)
"""
from os import PathLike
from pathlib import Path
from typing import Literal, Self, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from evedesign.model import BaseModel, Scorer, assign_scores_to_instances
from evedesign.system import System, SystemInstance
from evedesign.types import BatchSize, DeviceType, StatusCallback
from evedesign.utils import model_param_context

VENDOR_DIR = Path(__file__).parent / "vendor"

ModelName = Literal[
    "progen2-small", "progen2-medium", "progen2-base",
    "progen2-large", "progen2-xlarge", "progen2-BFD90",
]

# Vocabulary layout of the vendored tokenizer: 0 <|pad|>, 1 <|bos|>, 2 <|eos|>,
# 3 "1" (N->C marker), 4 "2" (C->N marker), 5..29 the amino-acid alphabet.
# Upstream likelihood.py calls 3 and 4 "bos_token, eos_token"; they are in fact
# the direction markers, which is what its stripping logic actually removes.
NC_MARKER, CN_MARKER = "1", "2"
FIRST_AA_TOKEN, LAST_AA_TOKEN = 5, 29


def canonical_log_likelihood(logits: torch.Tensor, ids: torch.Tensor) -> float:
    """
    Summed log-likelihood of one tokenized sequence, as upstream computes it.

    Reproduces progen2/likelihood.py: shift by one, drop a trailing direction
    marker, then restrict the logits to the amino-acid slice so the distribution
    is renormalized over residues only. Skipping that slice renormalizes over
    the special tokens too and silently shifts every score.

    Parameters
    ----------
    logits
        (n_tokens, vocab) model output for the sequence
    ids
        (n_tokens,) token ids the logits were produced from

    Returns
    -------
    float
        Summed log-likelihood over the sequence's residues
    """
    logits, target = logits[:-1, ...], ids[1:]

    if target[-1].item() in (FIRST_AA_TOKEN - 2, FIRST_AA_TOKEN - 1):
        logits, target = logits[:-1, ...], target[:-1]

    logits = logits[:, FIRST_AA_TOKEN:LAST_AA_TOKEN + 1]
    target = target - FIRST_AA_TOKEN

    keep = (target >= 0) & (target <= LAST_AA_TOKEN - FIRST_AA_TOKEN)
    if not bool(keep.any()):
        raise ValueError("Sequence contains no scorable amino-acid tokens")

    return float(-F.cross_entropy(logits[keep], target[keep], reduction="sum"))


class ProGen2(BaseModel, Scorer):
    """
    ProGen2 whole-sequence log-likelihood, averaged over both decoding directions.
    """
    name: str = "ProGen2"
    citations: list[str] = [
        # ProGen2: Exploring the Boundaries of Protein Language Models
        "10.1016/j.cels.2023.10.002",
    ]

    requires_target: bool = True
    requires_fixed_length: bool = False
    # indels are scored on the normalized (concrete) sequence
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = None
    optional_entity_attributes: list[str] | None = None

    # n_positions, minus the direction marker and the terminal token
    MAX_SEQ_LENGTH: int = 1024 - 2

    def __init__(
        self,
        model_name: ModelName = "progen2-small",
        model_dir_path: str | PathLike | None = None,
        batch_size: BatchSize = 8,
        bidirectional: bool = True,
        device: DeviceType = "cpu",
    ):
        """
        Instantiate a ProGen2 model.

        Parameters
        ----------
        model_name
            Official checkpoint name, used only for reporting when
            model_dir_path is set
        model_dir_path
            Directory holding the checkpoint's config.json and weights, baked
            in at build time
        batch_size
            Number of sequences scored per forward pass
        bidirectional
            Average the N->C and C->N decoding directions, as upstream does.
            False scores N->C only, which halves the cost
        device
            Device to run the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.model_dir_path = Path(model_dir_path) if model_dir_path is not None else None
        self.bidirectional = bidirectional
        self.device = device

        if batch_size != "auto" and batch_size < 1:
            raise ValueError("batch_size must be at least 1 or 'auto'")
        if batch_size == "auto":
            raise NotImplementedError("Automatic batch_size not yet implemented")
        self.batch_size = batch_size

        self.keep_model_after_pred = True

        self._system: System | None = None
        self.model = None
        self.tokenizer = None

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

        import json

        from tokenizers import Tokenizer

        from .vendor.configuration_progen import ProGenConfig
        from .vendor.modeling_progen import ProGenForCausalLM

        if self.model_dir_path is None:
            raise ValueError(
                "model_dir_path must point at a checkpoint directory; ProGen2 "
                "weights are baked into the image at build time"
            )

        config = ProGenConfig(**json.loads((self.model_dir_path / "config.json").read_text()))
        model = ProGenForCausalLM(config)

        # the official 2022 checkpoint is a plain tensor state_dict, so the
        # weights_only default needs no escape hatch here
        state = torch.load(
            self.model_dir_path / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f"Checkpoint is missing {len(missing)} expected tensors: {missing[:5]}")

        self.model = model.to(self.device).eval()
        self.tokenizer = Tokenizer.from_file(str(VENDOR_DIR / "tokenizer.json"))
        self.tokenizer.no_padding()

    def _delete_model(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _score_one(self, sequence: str, marker: str) -> float:
        seq = sequence if marker == NC_MARKER else sequence[::-1]
        ids = torch.tensor(
            self.tokenizer.encode(marker + seq + CN_MARKER).ids, device=self.device
        )
        with torch.no_grad():
            logits = self.model(ids).logits
        return canonical_log_likelihood(logits.float().cpu(), ids.cpu())

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score sequences by summed log-likelihood, averaged over both directions.

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

        # the tokenizer has no gap/insert symbols, so drop gaps and uppercase inserts
        sequences = ["".join(instance[0].normalized_rep()) for instance in instances]
        markers = (NC_MARKER, CN_MARKER) if self.bidirectional else (NC_MARKER,)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            scores: list[float] = []
            for i, sequence in enumerate(sequences):
                if status_callback and i % max(1, self.batch_size) == 0:
                    status_callback("running", (i / len(sequences)) * 100, f"Scoring {i}")
                scores.append(
                    float(np.mean([self._score_one(sequence, m) for m in markers]))
                )

        return assign_scores_to_instances(instances, np.asarray(scores, dtype=float))
