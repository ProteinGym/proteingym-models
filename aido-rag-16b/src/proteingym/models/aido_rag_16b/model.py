"""
Wrapper class around AIDO.Protein-RAG-16B (https://huggingface.co/genbio-ai/AIDO.Protein-RAG-16B-proteingym-dms-zeroshot)
"""
import os
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np
import pandas as pd

from evedesign.model import BaseModel, Scorer, MutationScorer, ConditionalMutationScorer, assign_scores_to_instances
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback
from evedesign.utils import status_done, status_start, ensure_sequence

try:
    import torch
    from scipy.special import softmax
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .vendor import protein as aido_protein
    from .vendor.misc import AIDO_Structure_Tokenizer, get_logits_table_sliding
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

# Dedicated zero-shot checkpoint; the base GB.Protein-RAG-16B has no usable scoring head
_DEFAULT_MODEL_NAME = "genbio-ai/AIDO.Protein-RAG-16B-proteingym-dms-zeroshot"


class AidoProteinRAG16B(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around AIDO.Protein-RAG-16B's asymmetric-temperature masked-marginal scoring.
    """
    name: str = "AIDO.Protein-RAG-16B-proteingym-dms-zeroshot"
    citations: list[str] = [
        # Mixture of Experts Enable Efficient and Effective Protein Understanding and Design
        "10.1101/2024.11.29.625425",
        # Retrieval Augmented Protein Language Models for Protein Structure Prediction
        "10.1101/2024.12.02.626519",
    ]
    available = IMPORT_AVAILABLE

    requires_target: bool = True
    # structure tokens and MSA columns are tied to a fixed sequence length
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = True
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = ["sequences", "structures"]
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        hf_cache_dir: str | PathLike | None = "/opt/program/data",
        device: str = "cuda",
        sliding_window: int = 768,
        sliding_step: int = 768,
        mask_str: bool = False,
        temp_mt: float = 1.0,
        temp_wt: float = 1.5,
        structure_key: str | None = None,
    ):
        """
        Instantiate an AIDO.Protein-RAG-16B model.

        Parameters
        ----------
        model_name
            HuggingFace Hub repo id of the zero-shot checkpoint
        hf_cache_dir
            HuggingFace cache directory for the checkpoints (also set as
            HF_HOME); None leaves the default cache untouched
        device
            torch device to run on; a GPU is required in practice
        sliding_window, sliding_step
            Sliding-window size/stride (query positions) for long sequences
        mask_str
            If True, zero out the structure embedding (structure ablation)
        temp_mt, temp_wt
            Temperatures applied to the logits for the mutant and wildtype
            log-softmax terms of the score
        structure_key
            Key of entity.structures to use; defaults to the first key
        """
        if not self.available:
            raise ImportError(
                "torch/transformers/scipy packages could not be imported. "
                "Install this wrapper's dependencies to use this model."
            )

        self.model_name = model_name
        self.hf_cache_dir = Path(hf_cache_dir) if hf_cache_dir is not None else None
        self.device = device
        self.sliding_window = sliding_window
        self.sliding_step = sliding_step
        self.mask_str = mask_str
        self.temp_mt = temp_mt
        self.temp_wt = temp_wt
        self.structure_key = structure_key

        self._system: System | None = None
        self._model = None
        self._tokenizer = None
        self._str_tokenizer: AIDO_Structure_Tokenizer | None = None
        self._vocab: dict[str, int] | None = None

        # (length x vocab) window-averaged logits and their log-softmax at
        # temp_mt/temp_wt, computed once in build()
        self._logit_table: np.ndarray | None = None
        self._table_mt: np.ndarray | None = None
        self._table_wt: np.ndarray | None = None

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self._logit_table is not None

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

        if target.structures is None or len(target.structures) == 0:
            return False, "Must provide a 3D structure (entity.structures) for model inference"

        return True, ""

    def _get_structure_pdb_text(self, target) -> str:
        """
        Write the target's single-chain structure as PDB text for the vendored
        from_pdb_string() parser.

        Parameters
        ----------
        target
            Target entity with a structures mapping

        Returns
        -------
        str
            PDB-format text of the selected structure
        """
        key = self.structure_key or list(target.structures.keys())[0]
        model = ensure_sequence(target.structures[key])[0]

        chains = model.chains()
        if len(chains) != 1:
            raise ValueError(
                f"Expected a single-chain structure, got chains {chains} for key {key!r}"
            )

        buf = StringIO()
        model.to_file(buf, format="pdb")
        return buf.getvalue()

    @staticmethod
    def _get_msa_hit_seqs(target, target_rep: str) -> list[str]:
        """
        Convert the target MSA to upper-cased match-state sequences, excluding the query.

        Parameters
        ----------
        target
            Target entity with an aligned MSA
        target_rep
            Query/WT sequence, which must equal the first MSA row

        Returns
        -------
        list[str]
            Match-state hit sequences (MSA rows 1..N)
        """
        seqs = [s.seq.upper() for s in target.sequences.to_a3m().remove_inserts().seqs]

        bad = [i for i, s in enumerate(seqs) if len(s) != len(target_rep)]
        if bad:
            raise ValueError(
                f"MSA match-state length does not match target length ({len(target_rep)}) "
                f"for {len(bad)} sequence(s), e.g. sequence index {bad[0]}"
            )

        if len(seqs) == 0 or seqs[0] != target_rep:
            raise ValueError(
                "First MSA sequence (match states) must equal the target/focus "
                "sequence, matching this package family's convention of treating "
                "the first alignment record as the query"
            )

        return seqs[1:]

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)

        status_start(status_callback, "Loading AIDO.Protein-RAG-16B and building masked-marginal profile")

        self._system = system
        target = system[0]

        self._logit_table = None
        self._table_mt = None
        self._table_wt = None

        # set before anything touches the HF cache, as upstream compute_fitness.py does
        if self.hf_cache_dir is not None:
            os.environ["HF_HOME"] = str(self.hf_cache_dir)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.hf_cache_dir)
        cache_dir = str(self.hf_cache_dir) if self.hf_cache_dir is not None else None

        self._str_tokenizer = AIDO_Structure_Tokenizer(device=self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=cache_dir,
        )
        self._model = model.bfloat16().eval().to(self.device)
        self._vocab = self._tokenizer.get_vocab()

        target_rep = "".join(target.rep)
        length = len(target_rep)

        pdb_text = self._get_structure_pdb_text(target)
        prot = aido_protein.from_pdb_string(pdb_text, molecular_type="protein")

        structure_seq = prot.seq(True)
        if structure_seq != target_rep:
            raise ValueError(
                "Target structure sequence does not match target rep sequence - "
                f"structure gives {len(structure_seq)} residues, target rep has "
                f"{length}. This should never happen for a ProteinGym2 archive "
                "(structure/MSA/rep are built together from the same dataset "
                "record), so treat this as a real bug, not an edge case to work "
                "around."
            )

        hit_seqs = self._get_msa_hit_seqs(target, target_rep)
        msa = [target_rep] + hit_seqs

        all_poses = list(range(length))

        with torch.no_grad():
            covered_poses, logit_table = get_logits_table_sliding(
                target_rep, prot, msa, dms_df=None,
                model=self._model, tokenizer=self._tokenizer, str_tokenizer=self._str_tokenizer,
                start=0,
                sliding_window=self.sliding_window, sliding_step=self.sliding_step,
                mask_str=self.mask_str, disable_tqdm=True,
                all_poses_override=all_poses,
            )

        assert covered_poses == all_poses

        self._logit_table = logit_table
        # as in upstream get_scores_from_table
        self._table_mt = np.log(softmax(logit_table / self.temp_mt, axis=-1))
        self._table_wt = np.log(softmax(logit_table / self.temp_wt, axis=-1))

        status_done(status_callback, "AIDO.Protein-RAG-16B model finished fitting")

        return self

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score full sequences against the WT.

        Sums the asymmetric-temperature log-odds over positions that differ
        from the WT; the WT itself scores 0.

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
        vocab = self._vocab

        scores = []
        for instance in instances:
            rep = np.asarray(instance[0].rep)
            total = 0.0
            for pos, (wt_aa, mut_aa) in enumerate(zip(target_rep, rep)):
                mut_aa = str(mut_aa)
                if mut_aa == wt_aa:
                    continue
                wt_tok = vocab.get(wt_aa)
                mut_tok = vocab.get(mut_aa)
                if wt_tok is None or mut_tok is None:
                    continue
                total += self._table_mt[pos, mut_tok] - self._table_wt[pos, wt_tok]
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
        Compute all single substitutions to an instance via the cached tables.

        Self-substitutions are set to 0, since with temp_mt != temp_wt the
        formula would not give 0.

        Parameters
        ----------
        instance
            Instance to mutate
        entity
            Entity to scan (only 0 is supported)
        positions
            Positions to scan; all positions if None
        status_callback
            Optional status callback

        Returns
        -------
        pd.DataFrame
            Scores indexed by (entity, pos, ref) with one column per symbol
        """
        self.ready_or_raise()
        self._validate_instances([instance])

        if entity is not None and entity != 0:
            raise ValueError("AIDO.Protein-RAG-16B only models entity 0")

        if positions is not None:
            if entity is None:
                raise ValueError(
                    "Parameter entity must be explicitly specified if using parameter positions"
                )
            self.valid_positions(positions, instance, entity, raise_invalid=True)

        status_start(status_callback, "Computing single mutation scan")

        first_index = self.system[0].first_index
        rep = instance[0].rep
        vocab = self._vocab
        alphabet = self.system[0].alphabet(
            include_gap=self.handles_deletions, include_inserts=self.handles_insertions,
        )

        pos_filter = set(positions) if positions is not None else None

        rows = []
        index_tuples = []
        for pos in range(self._logit_table.shape[0]):
            evc_pos = pos + first_index
            if pos_filter is not None and evc_pos not in pos_filter:
                continue

            ref_symbol = str(rep[pos])
            ref_tok = vocab.get(ref_symbol)

            row = np.full(len(alphabet), np.nan, dtype=float)
            for j, to in enumerate(alphabet):
                if to == ref_symbol:
                    row[j] = 0.0
                    continue

                to_tok = vocab.get(to)
                if to_tok is None or ref_tok is None:
                    continue

                row[j] = self._table_mt[pos, to_tok] - self._table_wt[pos, ref_tok]

            rows.append(row)
            index_tuples.append((0, evc_pos, ref_symbol))

        df = pd.DataFrame(
            rows,
            columns=alphabet,
            index=pd.MultiIndex.from_tuples(index_tuples, names=["entity", "pos", "ref"]),
        )

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
        Return the cached temp_mt log-softmax row for one position per instance.

        The table depends only on the WT, MSA and structure, so rows do not
        depend on the rest of the instance's sequence.

        Parameters
        ----------
        instances
            Instances to score
        entities
            Entity index for each instance
        positions
            Position for each instance
        status_callback
            Optional status callback

        Returns
        -------
        pd.DataFrame
            Scores indexed by (instance, entity, pos) with one column per symbol
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
        vocab = self._vocab
        alphabet = self.system[0].alphabet(
            include_gap=self.handles_deletions, include_inserts=self.handles_insertions,
        )

        rows = []
        index_tuples = []
        for inst_idx, (instance, entity, pos) in enumerate(zip(instances, entities, positions)):
            mi = int(pos) - first_index
            row = np.array([
                self._table_mt[mi, vocab[to]] if to in vocab else np.nan
                for to in alphabet
            ])
            rows.append(row)
            index_tuples.append((inst_idx, int(entity), int(pos)))

        df = pd.DataFrame(
            rows,
            columns=alphabet,
            index=pd.MultiIndex.from_tuples(index_tuples, names=["instance", "entity", "pos"]),
        )

        status_done(status_callback, "Conditional scoring complete")

        return df
