"""
Wrapper class around PoET-2 (https://github.com/OpenProteinAI/PoET-2), a
retrieval-augmented protein language model that conditions on an MSA-derived
prompt
"""
from os import PathLike
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np
import torch

from evedesign.model import BaseModel, Scorer, assign_scores_to_instances
from evedesign.system import System, SystemInstance
from evedesign.types import BatchSize, DeviceType, StatusCallback
from evedesign.utils import status_done, status_start

# poet_2_helpers annotates its tokenizer alphabets as Uniprot21 but defaults them
# to this subclass, and only it defines a gap symbol (gap_token 25, vs -1 for a
# bare Uniprot21). The MSA must be encoded with the same alphabet the model
# tokenizes with, or compute_homology_weights gets a sentinel gap token.
from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet
from poet_2.models import poet_2_helpers as helpers
from poet_2.msa.sampling import MSASampler, NeighborsSampler


class PoET2(BaseModel, Scorer):
    """
    PoET-2 scored as log P(variant) - log P(wildtype) under one MSA prompt.
    """
    available = True
    name: str = "PoET-2"
    citations: list[str] = [
        # Understanding protein function with a multimodal retrieval-augmented
        # foundation model
        "10.48550/arXiv.2508.04724",
    ]

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = False
    # reps are normalized before encoding, so indels score as the actual
    # shorter/longer sequence
    handles_deletions: bool = True
    handles_insertions: bool = True
    # unlike PoET, upstream selects float32 on CPU and gates flash attention on
    # qkv.is_cuda, so there is a working CPU path
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = ["sequences"]
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        ckpt_path: str | PathLike = "/opt/program/data/poet-2.ckpt",
        max_tokens: int = 8192,
        max_similarity: float = 1.0,
        theta: float = 0.2,
        batch_size: BatchSize = 8,
        device: DeviceType = "cpu",
        seed: int = 0,
    ):
        """
        Instantiate a PoET-2 model.

        Parameters
        ----------
        ckpt_path
            Path to the PoET-2 checkpoint, baked into the image at build time
        max_tokens
            Token budget for the MSA prompt. Prompt cost is quadratic in this,
            so it dominates runtime
        max_similarity
            Drop MSA sequences more similar than this to the target before
            sampling (1.0 keeps all)
        theta
            Identity threshold for the homology reweighting that biases prompt
            sampling towards diverse sequences
        batch_size
            Number of variants scored per forward pass against the prompt
        device
            Device to run the model on ("cpu" or "cuda")
        seed
            Seed for prompt sampling, so a run is reproducible
        """
        self.ckpt_path = Path(ckpt_path)
        self.max_tokens = max_tokens
        self.max_similarity = max_similarity
        self.theta = theta
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

        self._system: System | None = None
        self._model = None
        self._memory = None
        self._wt_logp: float | None = None
        self._alphabet = Alphabet()

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._memory is not None and self._wt_logp is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if len(system) != 1:
            return False, "PoET-2 models a single protein entity"
        if system[0].sequences is None or len(system[0].sequences.seqs) == 0:
            return False, "PoET-2 needs an MSA to build its prompt from"
        if system[0].rep is None:
            return False, "PoET-2 needs the target's rep sequence"
        return True, ""

    def _sample_prompt(self, target) -> list[helpers.NamedInput]:
        """
        Choose the prompt sequences for the target's MSA.

        Homology-reweighted sampling, then fill up to the token budget. The
        prompt is sequence-only: the dataset's structure covers the target, not
        its MSA, and NamedInput takes structure per prompt sequence.
        """
        a3m = target.sequences.to_a3m()
        raw_seqs = [s.seq for s in a3m.seqs]
        match_seqs = [s.seq for s in a3m.remove_inserts().seqs]

        msa = np.stack(
            [self._alphabet.encode(s.encode()) for s in match_seqs]
        ).astype(np.uint8)

        sampler = MSASampler(
            # the torch path of the O(depth^2) similarity calculation calls
            # .cuda() unconditionally, so it is only usable on a GPU
            method=NeighborsSampler(
                theta=self.theta, can_use_torch=(self.device == "cuda")
            ),
            max_similarity=self.max_similarity,
            # upstream's default, and it matters: forcing the target in as
            # prompt[0] lets the model copy it, so log P(wildtype | prompt)
            # collapses to about -10 for a 465-residue target instead of -900.
            # Every score is a delta against that term, so including the
            # target cost ~0.06-0.19 Spearman on HXK4.
            force_include_first=False,
        )
        sample_idxs = sampler.get_sample_idxs(
            msa=msa, gap_token=self._alphabet.gap_token, seed=self.seed
        )

        # Dropping force_include_first only stops the target being *forced* in;
        # the sampler can still draw it, which it does whenever the MSA is small
        # enough that most rows fit the budget. Exclude it explicitly.
        target_seq = "".join(target.rep).upper().replace("-", "")

        prompt, total_tokens = [], 0
        for idx in sample_idxs:
            seq = raw_seqs[idx].upper().replace("-", "")
            if not seq or seq == target_seq:
                continue
            if total_tokens + len(seq) > self.max_tokens:
                break
            prompt.append(helpers.NamedInput(sequence=seq))
            total_tokens += len(seq)
        return prompt

    @torch.inference_mode()
    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)

        status_start(status_callback, "Fitting PoET-2 model")

        self._system = system
        target = system[0]

        # load_model picks float32 on CPU (half precision is GPU-only) and
        # torch.load's weights_only=False upstream, so the Lightning ckpt's
        # hyper_parameters load; the file is the official release pinned in the
        # container recipes and baked in at build time
        self._model = helpers.load_model(
            path=self.ckpt_path, device=torch.device(self.device)
        )

        prompt = self._sample_prompt(target)
        if not prompt:
            raise ValueError(
                "PoET-2 prompt is empty; max_tokens is smaller than the "
                "shortest MSA sequence"
            )
        self._memory, _ = helpers.compute_memory(model=self._model, prompts=[prompt])

        # every variant is reported relative to the wildtype, so the wildtype's
        # own log-likelihood under this prompt is scored once here
        wt = "".join(target.rep)
        self._wt_logp = float(
            helpers.score_sequences(
                model=self._model,
                memory=self._memory,
                sequences=[wt],
                batch_size=1,
            )[0]
        )

        status_done(status_callback, "PoET-2 model finished fitting")

        return self

    @torch.inference_mode()
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score sequences by log P(variant) - log P(wildtype) under the prompt.

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

        if len(instances) == 0:
            return []

        status_start(status_callback, "Scoring sequences")

        sequences = [
            "".join(instance[0].normalized_rep()) for instance in instances
        ]
        logps = helpers.score_sequences(
            model=self._model,
            memory=self._memory,
            sequences=sequences,
            batch_size=self.batch_size,
        )
        scores = logps.cpu().float().numpy() - self._wt_logp

        status_done(status_callback, "Finished scoring sequences")

        return assign_scores_to_instances(instances, scores.tolist())
