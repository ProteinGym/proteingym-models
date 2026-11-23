"""
Wrapper class around PoET (https://github.com/OpenProteinAI/PoET)
"""
import copy
import itertools
from os import PathLike
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np
import pandas as pd

from evedesign.model import BaseModel, Scorer, MutationScorer, ConditionalMutationScorer, assign_scores_to_instances
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback
from evedesign.utils import status_done, status_start

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence

    from .poet.alphabets import Uniprot21
    from .poet.models.poet import PoET as PoETModel
    from .poet.msa.sampling import MSASampler, NeighborsSampler
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

# published ensemble from upstream scripts/score.py: 3 max_tokens x 5 max_similarity
_DEFAULT_ENSEMBLE_PARAMS = list(itertools.product(
    [6144, 12288, 24576], [1.0, 0.95, 0.90, 0.70, 0.50],
))


class PoET(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around PoET's ensembled log-likelihood scoring.
    """
    available = IMPORT_AVAILABLE
    name: str = "PoET"
    citations: list[str] = [
        # PoET: A generative model of protein families as sequences-of-sequences
        "10.48550/arXiv.2306.06156",
    ]

    requires_target: bool = True
    # whole-sequence likelihood, so indels are scored on the normalized sequence
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = True
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = ["sequences"]
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        ckpt_path: str | PathLike = "/opt/program/data/poet.ckpt",
        ensemble_params: Sequence[tuple[int, float]] | None = None,
        seed: int = 188257,
        batch_size: int = 8,
        max_similarity_can_use_torch: bool = False,
    ):
        """
        Instantiate a PoET model.

        Parameters
        ----------
        ckpt_path
            Path to the PoET checkpoint file
        ensemble_params
            (max_tokens, max_similarity) MSA-subsampling configurations to
            average over, each scored forward and reverse. Defaults to the
            published 15-member ensemble
        seed
            Random seed for MSA subsampling and shuffling
        batch_size
            Number of variants scored per forward pass against a fixed context
        max_similarity_can_use_torch
            Whether the homology-weight computation may use CUDA torch (faster,
            but uses extra GPU memory)
        """
        if not self.available:
            raise ImportError(
                "torch/poet packages could not be imported. Install the PoET "
                "package (this repo) and torch to use this model."
            )

        self.ckpt_path = Path(ckpt_path)
        self.ensemble_params = list(ensemble_params) if ensemble_params is not None else list(_DEFAULT_ENSEMBLE_PARAMS)
        self.seed = seed
        self.batch_size = batch_size
        self.max_similarity_can_use_torch = max_similarity_can_use_torch

        self._system: System | None = None
        self._model: PoETModel | None = None
        self._alphabet = Uniprot21(include_gap=True, include_startstop=True, distinct_startstop=True)
        # one (forward_memory, backward_memory) pair per ensemble_params entry
        self._memories: list[tuple[list, list]] | None = None

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self._memories is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle a single-component protein system"

        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have a defined rep sequence"

        if target.sequences is None or len(target.sequences.seqs) == 0:
            return False, "Must provide an MSA (entity.sequences) for model inference"

        if not target.sequences.aligned:
            return False, "Provided sequences must be aligned"

        return True, ""

    def _append_startstop(self, x: np.ndarray) -> np.ndarray:
        x_ = np.empty(x.shape[0] + 2, dtype=x.dtype)
        x_[0] = self._alphabet.start_token
        x_[-1] = self._alphabet.stop_token
        x_[1:-1] = x
        return x_

    def _sample_context_sequences(
        self,
        raw_seqs: list[bytes],
        sample_idxs: np.ndarray,
        max_tokens: int,
        seed: int,
    ) -> list[np.ndarray]:
        """
        Build the context sequences for one ensemble member, as in upstream sample_msa_sequences.

        Parameters
        ----------
        raw_seqs
            Raw a3m sequences (inserts kept)
        sample_idxs
            Indices of the sampled sequences
        max_tokens
            Token budget for the context
        seed
            Random seed for shuffling

        Returns
        -------
        list[np.ndarray]
            Encoded context sequences with start/stop tokens
        """
        seqs, total_tokens = [], 0
        for idx in sample_idxs:
            seq = raw_seqs[idx].upper().translate(None, delete=b"-")
            seqs.append(self._append_startstop(self._alphabet.encode(seq)))
            total_tokens += len(seqs[-1])
            if total_tokens > max_tokens:
                break

        rng = np.random.default_rng(seed)
        final_seqs, total_tokens = [], 0
        for seq in [seqs[i] for i in rng.permutation(len(seqs))]:
            final_seqs.append(seq)
            total_tokens += len(seq)
            if total_tokens >= max_tokens:
                break
        return final_seqs

    @torch.inference_mode()
    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)

        status_start(status_callback, "Fitting PoET model")

        self._system = system
        target = system[0]

        self._memories = None

        # weights_only=False: the PoET checkpoint is a Lightning ckpt carrying
        # hyper_parameters alongside the tensors, which torch's weights_only
        # unpickler (the default since torch 2.6) refuses. The file is the
        # official release pinned in the Dockerfile (Zenodo record 10061322)
        # and is baked into the image at build time, so it is a trusted source.
        ckpt = torch.load(
            str(self.ckpt_path), map_location="cpu", weights_only=False
        )
        model = PoETModel(**ckpt["hyper_parameters"]["model_spec"]["init_args"])
        model.load_state_dict({k.split(".", 1)[1]: v for k, v in ckpt["state_dict"].items()})
        del ckpt
        self._model = model.cuda().half().eval()

        # raw a3m (inserts kept) for the context; match states only for neighbor weights
        raw_seqs = [s.seq.encode() for s in target.sequences.to_a3m().seqs]
        match_seqs = [s.seq for s in target.sequences.to_a3m().remove_inserts().seqs]
        msa = np.stack([self._alphabet.encode(s.encode()) for s in match_seqs])

        # O(depth^2) weights do not vary across ensemble members; compute once
        neighbors_method = NeighborsSampler(can_use_torch=self.max_similarity_can_use_torch)
        _, cached_weights = neighbors_method.get_weights(msa=msa, gap_token=self._alphabet.gap_token)

        memories = []
        for max_tokens, max_similarity in self.ensemble_params:
            sampler = MSASampler(
                method=neighbors_method,
                max_similarity=max_similarity,
            )
            sample_idxs = sampler.get_sample_idxs(
                msa=msa, gap_token=self._alphabet.gap_token, seed=self.seed,
                weights=cached_weights,
            )
            context_seqs = self._sample_context_sequences(
                raw_seqs, sample_idxs, max_tokens=max_tokens, seed=self.seed,
            )

            forward_memory = self._embed(context_seqs)
            backward_memory = self._embed([np.ascontiguousarray(s[::-1]) for s in context_seqs])
            memories.append((forward_memory, backward_memory))

        self._memories = memories

        status_done(status_callback, "PoET model finished fitting")

        return self

    def _embed(self, context_seqs: list[np.ndarray]):
        """
        Embed one MSA context and offload the memory to CPU.

        Holding every ensemble member's memory on the GPU does not fit in VRAM.

        Parameters
        ----------
        context_seqs
            Encoded context sequences

        Returns
        -------
        list or None
            Per-layer memory (on CPU), or None for an empty context
        """
        if len(context_seqs) == 0:
            return None
        segment_sizes = torch.tensor([len(s) for s in context_seqs]).cuda()
        xs = torch.cat([torch.from_numpy(s).long() for s in context_seqs]).cuda()
        memory = self._model.embed(
            xs.unsqueeze(0), segment_sizes.unsqueeze(0), allow_cpu_offload=True,
        )
        for mem in memory:
            mem.x = mem.x.cpu()
        del xs, segment_sizes
        torch.cuda.empty_cache()
        return memory

    def _score_variants(self, memory, variants: list[np.ndarray]) -> np.ndarray:
        """
        Score variants against one context memory.

        Parameters
        ----------
        memory
            Context memory from _embed()
        variants
            Encoded variants with start/stop tokens

        Returns
        -------
        np.ndarray
            Log-likelihood of each variant
        """
        max_len = max(len(v) for v in variants)
        # logits_allocate_memory mutates memory in place; copy to keep the cache intact
        memory = [copy.copy(mem) for mem in memory] if memory is not None else memory
        memory = self._model.logits_allocate_memory(
            memory=memory, batch_size=self.batch_size, length=max_len - 1,
        )
        criteria = nn.CrossEntropyLoss(ignore_index=self._alphabet.mask_token, reduction="none")

        logps = []
        for start in range(0, len(variants), self.batch_size):
            batch = variants[start:start + self.batch_size]
            x = pad_sequence(
                [torch.from_numpy(v).long() for v in batch],
                batch_first=True, padding_value=self._alphabet.mask_token,
            )
            if x.size(1) < max_len:
                x = F.pad(x, (0, max_len - x.size(1)), value=self._alphabet.mask_token)
            x = x.cuda()
            logits = self._model.logits(x[:, :-1], memory, preallocated_memory=True)
            targets = x[:, 1:]
            score = -criteria.forward(logits.transpose(1, 2), targets).float().sum(dim=1)
            logps.append(score.cpu().numpy())
        return np.hstack(logps)

    @torch.inference_mode()
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score full sequences by ensemble-averaged forward+reverse log-likelihood.

        Reps are normalized (gaps dropped, insertions uppercased) before encoding,
        so indels are scored as the actual shorter/longer sequence.

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

        variants_fwd = [
            self._append_startstop(self._alphabet.encode("".join(instance[0].normalized_rep()).encode()))
            for instance in instances
        ]
        variants_bwd = [np.ascontiguousarray(v[::-1]) for v in variants_fwd]

        ensemble_logps = []
        for forward_memory, backward_memory in self._memories:
            forward_logps = self._score_variants(forward_memory, variants_fwd)
            backward_logps = self._score_variants(backward_memory, variants_bwd)
            ensemble_logps.append((forward_logps + backward_logps) / 2)
            # free per-call GPU copies promptly; VRAM margins are tight
            torch.cuda.empty_cache()

        scores = np.vstack(ensemble_logps).mean(axis=0)

        status_done(status_callback, "Scoring complete")

        return assign_scores_to_instances(instances, np.asarray(scores, dtype=float))
