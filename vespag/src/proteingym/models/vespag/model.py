"""
Wrapper class around VespaG (https://github.com/jschlensok/vespag), a small
feed-forward head over ESM-2 embeddings distilled from GEMME scores
"""
from pathlib import Path
from typing import Any, Literal, Self, Sequence

import numpy as np
import torch

from evedesign.model import BaseModel, Scorer, assign_scores_to_instances
from evedesign.models.esm2 import ESM2
from evedesign.system import EntityInstance, System, SystemInstance
from evedesign.types import BatchSize, DeviceType, StatusCallback
from evedesign.utils import status_done, status_start

# Column order of the head's 20 outputs, from upstream's utils.GEMME_ALPHABET.
# Getting this wrong silently permutes every score, so it is asserted in tests.
GEMME_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Upstream's published default (utils.MODEL_VERSION = "v2", DEFAULT_MODEL_PARAMETERS):
# FNN(hidden_layer_sizes=[256], input_dim=2560, output_dim=20, dropout_rate=0.2),
# which lays out as Linear/Dropout/LeakyReLU/Linear -> state dict keys net.0, net.3
_HIDDEN_DIM = 256
_EMBEDDING_DIM = 2560
_DROPOUT_RATE = 0.2


def load_head(checkpoint_path: Path | str, device: DeviceType) -> torch.nn.Module:
    """
    Rebuild upstream's v2 FNN and load its published weights.

    The architecture is four standard torch layers, so it is reconstructed here
    rather than importing the (GPL-3.0) upstream package.
    """
    net = torch.nn.Sequential(
        torch.nn.Linear(_EMBEDDING_DIM, _HIDDEN_DIM),
        torch.nn.Dropout(_DROPOUT_RATE),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(_HIDDEN_DIM, len(GEMME_ALPHABET)),
    )
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    # upstream stores the Sequential under a "net." prefix
    state = {k.removeprefix("net."): v for k, v in state.items()}

    # a v1 checkpoint has no dropout and so keys 0/2 rather than 0/3; loading it
    # here would silently mis-assign the output layer
    expected = {"0.weight", "0.bias", "3.weight", "3.bias"}
    if set(state) != expected:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has keys {sorted(state)}; expected "
            f"{sorted(expected)} (upstream's v2 FNN). A v1 checkpoint is not supported."
        )

    net.load_state_dict(state)
    return net.eval().to(device)


def landscape_by_char(landscape: np.ndarray) -> np.ndarray:
    """
    Re-index a (length, 20) landscape by byte value so scoring is one lookup.

    Symbols outside GEMME_ALPHABET stay at 0, so they contribute nothing rather
    than raising mid-scan.
    """
    by_char = np.zeros((landscape.shape[0], 256), dtype=float)
    for j, aa in enumerate(GEMME_ALPHABET):
        by_char[:, ord(aa)] = landscape[:, j]
    return by_char


class VespaG(BaseModel, Scorer):
    """
    Score variants from a GEMME-distilled head over ESM-2 embeddings.
    """
    available = True
    name: str = "VespaG"
    citations: list[str] = [
        # Marquet et al., Bioinformatics 2024 (btae621)
        "10.1093/bioinformatics/btae621",
    ]

    # core properties
    requires_target: bool = True
    # the landscape is indexed by position against the target, so a variant must
    # align to it residue for residue
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        head_checkpoint_path: str | Path = "/opt/program/data/vespag/esm2.pt",
        embedder_model_name: str = "esm2_t36_3B_UR50D",
        normalize: bool = False,
        batch_size: BatchSize = 1,
        device: DeviceType = "cpu",
    ):
        """
        Instantiate VespaG.

        Parameters
        ----------
        head_checkpoint_path
            Upstream's v2 ESM-2 head (model_weights/v2/esm2.pt), baked at build time
        embedder_model_name
            ESM-2 checkpoint supplying the embeddings. The head was trained on
            2560-dimensional embeddings, so only the 3B model fits it
        normalize
            Apply upstream's min-max rescaling of the landscape to [0, 1]. It is
            monotonic, so it cannot change rankings; left off by default because
            it makes a variant's score depend on the rest of the landscape
        batch_size
            Sequences per embedder forward pass (only the target is embedded)
        device
            Device for both the embedder and the head
        """
        self.head_checkpoint_path = Path(head_checkpoint_path)
        self.embedder_model_name = embedder_model_name
        self.normalize = normalize
        self.device = device

        self._embedder = ESM2(
            model_name=embedder_model_name, batch_size=batch_size, device=device
        )
        self._system: System | None = None
        # (length, 20) landscape, wild type anchored to 0
        self._landscape: np.ndarray | None = None
        # the same landscape indexed by byte value, for vectorized scoring
        self._landscape_by_char: np.ndarray | None = None

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self._landscape is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle a single-component protein system"

        if not system[0].defined_sequence():
            return False, "Entity must have a defined rep sequence"

        return True, ""

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        """
        Embed the target once and turn the head's output into a score landscape.

        The head emits 20 scores per position, so the whole mutational landscape
        comes from a single embedder pass and scoring is a lookup afterwards.
        """
        ok, reason = self.can_model(system, data)
        if not ok:
            raise ValueError(reason)

        target = "".join(str(symbol) for symbol in system[0].rep)

        status_start(status_callback, "Embedding target")
        self._embedder.build(system)
        [embedded] = self._embedder.transform(
            [SystemInstance([EntityInstance(rep=target)])]
        )
        embedding = np.asarray(embedded[0].embedding, dtype=np.float32)

        if embedding.shape != (len(target), _EMBEDDING_DIM):
            raise ValueError(
                f"Embedder returned {embedding.shape} for a {len(target)}-residue "
                f"target; the head needs (length, {_EMBEDDING_DIM}). Is "
                f"'{self.embedder_model_name}' the 3B ESM-2 checkpoint?"
            )

        status_start(status_callback, "Predicting landscape")
        head = load_head(self.head_checkpoint_path, self.device)
        with torch.no_grad():
            landscape = head(
                torch.from_numpy(embedding).to(self.device)
            ).cpu().numpy().astype(float)

        if self.normalize:
            lo, hi = landscape.min(), landscape.max()
            landscape = np.clip((landscape - lo) / (hi - lo), 0, 1)

        # upstream's mask_non_mutations: a non-mutation has no effect
        for pos, wt_aa in enumerate(target):
            if wt_aa in GEMME_ALPHABET:
                landscape[pos, GEMME_ALPHABET.index(wt_aa)] = 0.0

        self._landscape = landscape
        self._landscape_by_char = landscape_by_char(landscape)

        self._system = system
        status_done(status_callback, "Build complete")
        return self

    def positions(
        self,
        instance: SystemInstance | None = None,  # noqa: ARG002
    ) -> list[tuple[int, int]]:
        """
        Return the modelled positions (all positions of entity 0).
        """
        self.ready_or_raise()
        first_index = self.system[0].first_index
        return [(0, pos + first_index) for pos in range(self._landscape.shape[0])]

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score sequences as the sum of their substitutions' landscape values.

        Equivalent to upstream, which sums the per-substitution scores of a
        multiple mutant; the wild type scores 0 because its columns are zeroed.
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        if len(instances) == 0:
            return []

        status_start(status_callback, "Scoring sequences")

        length = self._landscape.shape[0]
        pos_index = np.arange(length)
        chunk = max(1, 4_000_000 // max(length, 1))
        scores = np.empty(len(instances), dtype=float)
        for start in range(0, len(instances), chunk):
            batch = instances[start:start + chunk]
            buf = "".join(
                "".join(str(symbol) for symbol in instance[0].rep)
                for instance in batch
            ).encode("ascii", "replace")
            codes = np.frombuffer(buf, dtype=np.uint8).reshape(len(batch), length)
            scores[start:start + len(batch)] = self._landscape_by_char[
                pos_index[None, :], codes
            ].sum(axis=1)

        status_done(status_callback, "Scoring complete")

        return assign_scores_to_instances(instances, scores)
