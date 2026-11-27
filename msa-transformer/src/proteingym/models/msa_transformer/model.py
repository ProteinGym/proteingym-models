"""
Wrapper class around MSA Transformer (ESM-MSA-1b), the alignment-conditioned
masked language model from https://github.com/facebookresearch/esm
"""
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np

from evedesign.constants import VALID_AA_SORTED
from evedesign.model import BaseModel, Scorer, assign_scores_to_instances
from evedesign.system import System, SystemInstance
from evedesign.types import StatusCallback
from evedesign.utils import status_done, status_start

try:
    import torch
    from esm import pretrained

    # The checkpoint pickles an argparse.Namespace, which torch>=2.6's
    # weights_only=True default refuses to load. Allowlisting that one global
    # keeps the safety check on, rather than disabling it wholesale.
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])

    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class MSATransformer(BaseModel, Scorer):
    """
    Zero-shot variant scoring with MSA Transformer by masked marginals.

    build() runs one masked forward pass per target position over the
    subsampled MSA and caches the resulting per-position log-probabilities, so
    scoring any number of variants afterwards is a table lookup.
    """

    available = IMPORT_AVAILABLE
    name: str = "MSA Transformer"
    citations: list[str] = [
        # MSA Transformer (Rao et al., ICML 2021)
        "10.1101/2021.02.12.430858",
    ]

    requires_target: bool = True
    # Scores are per-position log-probability differences, so variants must
    # align to the target column-for-column
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    # the alignment is what conditions the model, so it is required
    required_entity_attributes: list[str] | None = ["sequences"]
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        checkpoint_path: str = (
            "/opt/program/data/esm_msa1b_t12_100M_UR50S.pt"
        ),
        n_seqs: int = 384,
        seed: int = 0,
        positions_per_batch: int = 1,
        device: str = "cpu",
    ):
        """
        Instantiate an MSA Transformer scorer.

        Parameters
        ----------
        checkpoint_path
            Path to the fair-esm esm_msa1b_t12_100M_UR50S checkpoint. The
            matching *-contact-regression.pt must sit next to it, which is
            what fair-esm's local loader expects
        n_seqs
            Rows of the alignment passed to the model, target included. The
            published zero-shot protocol subsamples the MSA; 384 is the depth
            used for MSA Transformer in the ESM-1v comparisons
        seed
            Seed for the row subsample, so a run is reproducible
        positions_per_batch
            Masked positions evaluated per forward pass. Each one carries its
            own copy of the alignment, so raising this trades memory for speed
        device
            Device to run the model on ("cpu" or "cuda")
        """
        if not self.available:
            raise ImportError(
                "fair-esm and torch could not be imported. Are they installed?"
            )

        self.checkpoint_path = Path(checkpoint_path)
        self.n_seqs = int(n_seqs)
        self.seed = int(seed)
        self.positions_per_batch = max(1, int(positions_per_batch))
        self.device = device

        self._system: System | None = None
        # (length, 256) log-probabilities indexed by residue byte value
        self._logp_by_char: np.ndarray | None = None

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self._logp_by_char is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle a single-component protein system"

        if not system[0].defined_sequence():
            return False, "Entity must have a defined rep sequence"

        sequences = system[0].sequences
        if sequences is None or len(sequences.seqs) == 0:
            return False, "Entity must carry an MSA to condition the model on"

        if not sequences.aligned:
            return False, "Entity sequences must be an alignment"

        return True, ""

    def _alignment(self, target: str, sequences: Any) -> list[tuple[str, str]]:
        """
        Build the (label, sequence) rows handed to fair-esm's batch converter.

        Insertions are dropped so every row spans the target's columns, the
        target is row 0 (the query the model conditions on), and the remaining
        rows are subsampled to n_seqs.
        """
        match = sequences.to_a3m().remove_inserts()
        rows = ["".join(str(s.seq)).upper() for s in match.seqs]

        bad = [len(r) for r in rows if len(r) != len(target)]
        if bad:
            raise ValueError(
                f"{len(bad)} alignment rows have a different length than the "
                f"target ({sorted(set(bad))} vs {len(target)}) after removing "
                "insertions; the MSA does not span the target sequence"
            )

        # the dataset's first row is usually the target already - only prepend
        # it when it is not, so the query row is never duplicated
        context = rows[1:] if rows and rows[0] == target else rows

        if len(context) > self.n_seqs - 1:
            rng = np.random.default_rng(self.seed)
            picked = rng.choice(len(context), size=self.n_seqs - 1, replace=False)
            context = [context[i] for i in sorted(picked)]

        return [("target", target)] + [
            (f"seq{i}", seq) for i, seq in enumerate(context)
        ]

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """
        Compute and cache the masked-marginal log-probability profile.

        Parameters
        ----------
        system
            Single-entity protein system with a defined target and an MSA
        data
            Unused, must be None
        status_callback
            Optional progress callback

        Returns
        -------
        Self
            The built model
        """
        self.can_model_or_raise(system, data)
        status_start(status_callback, "Loading MSA Transformer")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MSA Transformer checkpoint not found at {self.checkpoint_path}"
            )

        self._system = system
        target = "".join(str(symbol) for symbol in system[0].rep)

        model, alphabet = pretrained.load_model_and_alphabet_local(
            str(self.checkpoint_path)
        )
        model = model.to(self.device).eval()

        rows = self._alignment(target, system[0].sequences)
        _, _, tokens = alphabet.get_batch_converter()([rows])
        tokens = tokens.to(self.device)

        length = len(target)
        # MSA Transformer prepends a BOS token, so target position i is column i+1
        aa_idx = [alphabet.get_idx(aa) for aa in VALID_AA_SORTED]
        logp = np.full((length, len(VALID_AA_SORTED)), np.nan, dtype=float)

        status_start(status_callback, f"Masked marginals over {length} positions")
        with torch.no_grad():
            for start in range(0, length, self.positions_per_batch):
                positions = range(start, min(start + self.positions_per_batch, length))
                batch = tokens.repeat(len(positions), 1, 1)
                for row, pos in enumerate(positions):
                    batch[row, 0, pos + 1] = alphabet.mask_idx

                logits = model(batch)["logits"]
                for row, pos in enumerate(positions):
                    column = torch.log_softmax(logits[row, 0, pos + 1], dim=-1)
                    logp[pos] = column[aa_idx].float().cpu().numpy()

        # indexed by byte value so scoring is a vectorized gather; symbols
        # outside the 20 standard residues contribute 0
        by_char = np.zeros((length, 256), dtype=float)
        for col, aa in enumerate(VALID_AA_SORTED):
            by_char[:, ord(aa)] = logp[:, col]
        self._logp_by_char = by_char

        status_done(status_callback, "Masked marginals complete")
        return self

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score sequences as the summed log-probability ratio to the target.

        Parameters
        ----------
        instances
            Instances to score
        status_callback
            Optional progress callback

        Returns
        -------
        list[SystemInstance]
            Scored copies of the instances (the target scores 0)
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        if len(instances) == 0:
            return []

        status_start(status_callback, "Scoring sequences")

        length = self._logp_by_char.shape[0]
        pos_index = np.arange(length)
        target = "".join(str(symbol) for symbol in self.system[0].rep)
        wt = self._logp_by_char[
            pos_index, np.frombuffer(target.encode("ascii", "replace"), dtype=np.uint8)
        ].sum()

        # chunked to bound memory on large variant sets
        chunk = max(1, 4_000_000 // max(length, 1))
        scores = np.empty(len(instances), dtype=float)
        for start in range(0, len(instances), chunk):
            batch = instances[start:start + chunk]
            buf = "".join(
                "".join(str(symbol) for symbol in instance[0].rep)
                for instance in batch
            ).encode("ascii", "replace")
            codes = np.frombuffer(buf, dtype=np.uint8).reshape(len(batch), length)
            scores[start:start + len(batch)] = (
                self._logp_by_char[pos_index[None, :], codes].sum(axis=1) - wt
            )

        status_done(status_callback, "Scoring complete")
        return assign_scores_to_instances(instances, scores)
