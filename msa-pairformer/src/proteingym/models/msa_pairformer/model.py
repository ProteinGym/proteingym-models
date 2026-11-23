"""
Wrapper class around MSA Pairformer (https://github.com/yoakiyama/MSA_Pairformer)
"""
import math
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
    import torch.nn.functional as F

    from MSA_Pairformer.model import MSAPairformer
    from MSA_Pairformer.dataset import aa2tok_d, prepare_msa_masks
    from MSA_Pairformer.proteingym_utils import (
        calc_weights_fast, map_from_alphabet, map_matrix,
        get_num_cpus, GAP, ALPHABET_PROTEIN_GAP,
    )
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

# the 20 standard amino acids (aa2tok_d tokens 0-19)
_STANDARD_AA = "ARNDCQGHILKMFPSTWYV"[:0] or "ARNDCQEGHILKMFPSTWYV"
assert sorted(_STANDARD_AA) == sorted(k for k, v in aa2tok_d.items() if v < 20) if IMPORT_AVAILABLE else True


class MSAPairformerModel(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around MSA Pairformer's masked-marginal log-probability scoring.
    """
    name: str = "MSAPairformer"
    citations: list[str] = [
        # Expanding the scope of protein language modeling to protein-protein
        # interactions with MSA Pairformer
        "10.1016/j.cell.2026.06.029",
    ]
    available = IMPORT_AVAILABLE

    requires_target: bool = True
    requires_fixed_length: bool = True
    # masked-marginal scoring is per query position; indels are not scored
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = True
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = ["sequences"]
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        ckpt_dir: str | PathLike = "/opt/program/data",
        n_seqs: int = 512,
        theta: float = 0.2,
        seed: int = 42,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        """
        Instantiate an MSA Pairformer model.

        Parameters
        ----------
        ckpt_dir
            HuggingFace hub cache root holding yakiyama/MSA-Pairformer (downloaded
            there by from_pretrained if empty)
        n_seqs
            Number of sequences (including the query) sampled from the MSA as
            model context
        theta
            Sequence-reweighting threshold (1 - identity); 0.2 for most
            families, 0.01 for viruses
        seed
            Random seed for the weighted MSA subsampling
        device
            Torch device to run on ("cuda" recommended; CPU is impractical)
        batch_size
            Number of masked positions scored per forward pass
        """
        if not self.available:
            raise ImportError(
                "torch/MSA_Pairformer packages could not be imported. Install "
                "the msa-pairformer package and torch to use this model."
            )

        self.ckpt_dir = Path(ckpt_dir)
        self.n_seqs = n_seqs
        self.theta = theta
        self.seed = seed
        self.device = device
        self.batch_size = batch_size

        self._system: System | None = None
        self._model: MSAPairformer | None = None
        # (length x 26) masked-marginal log-probabilities, computed in build()
        self._log_probs: np.ndarray | None = None

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self._log_probs is not None

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

    @staticmethod
    def _match_state_seqs(target) -> tuple[str, list[str]]:
        """
        Convert the system MSA to match-state-only (insertions stripped,
        uppercased) sequences, with the query/WT sequence first.

        Parameters
        ----------
        target
            Entity carrying the MSA

        Returns
        -------
        tuple[str, list[str]]
            Target sequence and match-state sequences
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
                "First MSA sequence (match states) must equal the target/focus "
                "sequence, matching MSA Pairformer's own convention of treating "
                "the first alignment record as the query"
            )

        return target_rep, seqs

    def _neighbor_weights(self, seqs: list[str]) -> np.ndarray:
        """
        Compute EVE-style sequence weights with upstream's calc_weights_fast.

        Parameters
        ----------
        seqs
            Match-state sequences

        Returns
        -------
        np.ndarray
            Weight per sequence (1 / number of neighbours within 1 - theta identity)
        """
        arr = np.array([list(s) for s in seqs])
        alphabet_mapper = map_from_alphabet(ALPHABET_PROTEIN_GAP, default=GAP)
        mapped = map_matrix(arr, alphabet_mapper)
        return calc_weights_fast(
            mapped, identity_threshold=1 - self.theta, empty_value=0,
            num_cpus=get_num_cpus(),
        )

    def _sample_msa(self, seqs: list[str]) -> list[str]:
        """
        Sample n_seqs sequences (with replacement) proportionally to their weights,
        keeping the query first, as in upstream's ProteinGym reference code.

        Parameters
        ----------
        seqs
            Match-state sequences, query first

        Returns
        -------
        list[str]
            Sampled sequences, query first
        """
        if len(seqs) <= self.n_seqs:
            return seqs

        weights = self._neighbor_weights(seqs)
        non_wt_weights = weights[1:]
        non_wt_weights = non_wt_weights / non_wt_weights.sum()

        rng = np.random.default_rng(self.seed)
        sampled_idx = rng.choice(
            len(seqs) - 1, size=self.n_seqs - 1, replace=True, p=non_wt_weights,
        )
        return [seqs[0]] + [seqs[1 + i] for i in sampled_idx]

    @torch.inference_mode()
    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)

        status_start(status_callback, "Fitting MSA Pairformer model")

        self._system = system
        target = system[0]

        self._log_probs = None

        self._model = MSAPairformer.from_pretrained(
            device=torch.device(self.device), weights_dir=str(self.ckpt_dir),
        ).to(torch.bfloat16)
        self._model.eval()

        target_rep, seqs = self._match_state_seqs(target)
        sampled_seqs = self._sample_msa(seqs)
        length = len(target_rep)

        tokenized = np.array(
            [[aa2tok_d[c] for c in seq] for seq in sampled_seqs], dtype=np.int64,
        )
        tokenized_t = torch.from_numpy(tokenized).unsqueeze(0)  # [1, s, n]
        mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokenized_t)
        mask = mask.to(self.device)
        msa_mask = msa_mask.to(self.device)
        full_mask = full_mask.to(self.device)
        pairwise_mask = pairwise_mask.to(self.device)

        n_token_types = len(np.unique(list(aa2tok_d.values())))

        log_probs = np.empty((length, 26), dtype=np.float32)
        n_batches = math.ceil(length / self.batch_size)
        for batch_idx in range(n_batches):
            positions = range(
                batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, length),
            )
            # one MSA copy per position, each with a single query position masked
            batch_tokenized = tokenized_t.repeat(len(positions), 1, 1)
            for row, pos in enumerate(positions):
                batch_tokenized[row, 0, pos] = aa2tok_d["<mask>"]

            batch_onehot = F.one_hot(
                batch_tokenized, num_classes=n_token_types,
            ).to(torch.bfloat16).to(self.device)
            batch_mask = mask.expand(len(positions), -1)
            batch_msa_mask = msa_mask.expand(len(positions), -1)
            batch_full_mask = full_mask.expand(len(positions), -1, -1)
            batch_pairwise_mask = pairwise_mask.expand(len(positions), -1, -1)

            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda" if self.device == "cuda" else "cpu"):
                logits = self._model(
                    msa=batch_onehot,
                    mask=batch_mask,
                    msa_mask=batch_msa_mask,
                    full_mask=batch_full_mask,
                    pairwise_mask=batch_pairwise_mask,
                    query_only=True,
                    return_seq_weights=False,
                    return_cb_contacts=False,
                    return_confind_contacts=False,
                )["logits"]

            batch_log_probs = torch.log_softmax(logits.float(), dim=-1)
            for row, pos in enumerate(positions):
                log_probs[pos] = batch_log_probs[row, 0, pos].cpu().numpy()

            if status_callback is not None:
                status_callback(
                    "running", 100 * (batch_idx + 1) / n_batches,
                    f"Scored position batch {batch_idx + 1}/{n_batches}",
                )

        self._log_probs = log_probs

        status_done(status_callback, "MSA Pairformer model finished fitting")

        return self

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> list[tuple[int, int]]:
        self.ready_or_raise()
        first_index = self.system[0].first_index
        return [
            (0, pos + first_index) for pos in range(self._log_probs.shape[0])
        ]

    @torch.inference_mode()
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score full sequences as the sum of log P(mutant) - log P(WT) over
        positions that differ from the query.

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

        target_rep = "".join(self.system[0].rep)

        scores = []
        for instance in instances:
            rep = np.asarray(instance[0].rep)
            total = 0.0
            for pos, (wt_aa, mut_aa) in enumerate(zip(target_rep, rep)):
                mut_aa = str(mut_aa)
                if mut_aa == wt_aa:
                    continue
                wt_tok = aa2tok_d.get(wt_aa)
                mut_tok = aa2tok_d.get(mut_aa)
                if wt_tok is None or mut_tok is None or wt_tok >= 26 or mut_tok >= 26:
                    continue
                total += self._log_probs[pos, mut_tok] - self._log_probs[pos, wt_tok]
            scores.append(total)

        status_done(status_callback, "Scoring complete")

        return assign_scores_to_instances(instances, np.asarray(scores, dtype=float))

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int | None = None,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None,
    ) -> pd.DataFrame:
        """
        Compute all single substitutions to an instance, as log P(mutant) -
        log P(instance's residue).

        Parameters
        ----------
        instance
            Instance to mutate
        entity
            Entity index (only 0 is supported)
        positions
            Positions to scan (all if None)
        status_callback
            Optional status callback

        Returns
        -------
        pd.DataFrame
            Scores indexed by (entity, pos, ref), one column per symbol
        """
        self.ready_or_raise()
        self._validate_instances([instance])

        if entity is not None and entity != 0:
            raise ValueError("MSAPairformer only models entity 0")

        if positions is not None:
            if entity is None:
                raise ValueError(
                    "Parameter entity must be explicitly specified if using parameter positions"
                )
            self.valid_positions(positions, instance, entity, raise_invalid=True)

        status_start(status_callback, "Computing single mutation scan")

        first_index = self.system[0].first_index
        rep = instance[0].rep
        tok_order = sorted(aa2tok_d.items(), key=lambda kv: kv[1])
        symbols = [aa for aa, tok in tok_order if tok < 26]

        pos_filter = set(positions) if positions is not None else None

        rows = []
        index_tuples = []
        for pos in range(self._log_probs.shape[0]):
            evc_pos = pos + first_index
            if pos_filter is not None and evc_pos not in pos_filter:
                continue

            ref_symbol = str(rep[pos])
            ref_tok = aa2tok_d.get(ref_symbol)
            ref_log_prob = self._log_probs[pos, ref_tok] if ref_tok is not None and ref_tok < 26 else np.nan

            rows.append(self._log_probs[pos, :] - ref_log_prob)
            index_tuples.append((0, evc_pos, ref_symbol))

        df = pd.DataFrame(
            rows,
            columns=symbols,
            index=pd.MultiIndex.from_tuples(index_tuples, names=["entity", "pos", "ref"]),
        )

        merged_alphabet = self.system[0].alphabet(
            include_gap=self.handles_deletions, include_inserts=self.handles_insertions,
        )
        df = df.reindex(merged_alphabet, axis=1)

        status_done(status_callback, "Single mutation scan complete")

        return df

    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None,
    ) -> pd.DataFrame:
        """
        Compute raw masked-marginal log-probabilities for one position per instance.
        The profile is computed from the query, so rows do not depend on the instance.

        Parameters
        ----------
        instances
            Instances to score
        entities
            Entity index per instance
        positions
            Position per instance
        status_callback
            Optional status callback

        Returns
        -------
        pd.DataFrame
            Scores indexed by (instance, entity, pos), one column per symbol
        """
        self.ready_or_raise()

        if not len(instances) == len(entities) == len(positions):
            raise ValueError(
                "Sequences for instances, entities and positions must all have same length"
            )

        self._validate_instances(instances)

        for instance, entity, pos in zip(instances, entities, positions):
            self.valid_positions(
                positions=[pos], instance=instance, entities=[entity], raise_invalid=True
            )

        status_start(status_callback, "Computing conditional scores")

        first_index = self.system[0].first_index
        tok_order = sorted(aa2tok_d.items(), key=lambda kv: kv[1])
        symbols = [aa for aa, tok in tok_order if tok < 26]

        rows = []
        index_tuples = []
        for inst_idx, (instance, entity, pos) in enumerate(zip(instances, entities, positions)):
            mi = int(pos) - first_index
            rows.append(self._log_probs[mi, :])
            index_tuples.append((inst_idx, int(entity), int(pos)))

        df = pd.DataFrame(
            rows,
            columns=symbols,
            index=pd.MultiIndex.from_tuples(index_tuples, names=["instance", "entity", "pos"]),
        )

        merged_alphabet = self.system[0].alphabet(
            include_gap=self.handles_deletions, include_inserts=self.handles_insertions,
        )
        df = df.reindex(merged_alphabet, axis=1)

        status_done(status_callback, "Conditional scoring complete")

        return df
