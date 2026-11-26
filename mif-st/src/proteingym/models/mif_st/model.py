"""
Wrapper class around MIF-ST (Masked Inverse Folding with Sequence Transfer),
the structure-conditioned zero-shot variant-effect model from
microsoft/protein-sequence-models
(https://github.com/microsoft/protein-sequence-models).
"""
from typing import Any, Self, Sequence

import numpy as np

from evedesign.model import (
    BaseModel,
    ConditionalMutationScorer,
    MutationScorer,
    Scorer,
    assign_scores_to_instances,
)
from evedesign.structure import Structure
from evedesign.system import System, SystemInstance
from evedesign.types import BatchSize, DeviceType, StatusCallback
from evedesign.utils import ensure_sequence, status_done, status_start

try:
    import biotite.structure as struc

    # biotite 2.0 renamed filter_backbone
    if not hasattr(struc, "filter_backbone"):
        struc.filter_backbone = struc.filter_peptide_backbone

    import torch
    import torch.nn.functional as F

    from sequence_models.pdb_utils import process_coords
    from sequence_models.pretrained import load_model_and_alphabet

    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

# Checkpoint name understood by sequence_models.pretrained. MIF-ST is really two
# checkpoints - this GNN plus the CARP-640M sequence model it transfers from -
# and upstream resolves both through torch.hub's cache, which the image
# pre-populates so nothing is fetched at run time (see Dockerfile).
CHECKPOINT_NAME = "mifst"

# Backbone atoms MIF-ST derives its structure features from, in the order
# process_coords() expects them
BACKBONE_ATOMS = ("N", "CA", "C")


def _extract_backbone_coords(
    chain: Structure,
    first_index: int,
    length: int,
    rep: str,
) -> np.ndarray:
    """
    Extract N/CA/C backbone coordinates for the target chain, reindexed to a
    full (length, 3, 3) array aligned 1:1 with the entity's rep sequence
    (position i <-> coords[i]), with np.nan rows for any position the structure
    has no coordinates for.

    Parameters
    ----------
    chain
        Single-chain Structure to extract from (see Structure.chains())
    first_index
        Entity's first_index (see evedesign.system.Entity)
    length
        Full entity rep length (the output array's first dimension)
    rep
        Full entity rep sequence string, used only to validate that the
        structure's own residue identities agree with it

    Returns
    -------
    (length, 3, 3) float32 array of N/CA/C coordinates, np.nan where the
    structure has no coordinate for that entity position. np.nan is what
    MIF-ST's own collater expects for unresolved residues: it drops them from
    the k-nearest-neighbour graph and masks their edge features.
    """
    positions = list(range(first_index, first_index + length))
    # Raises ValueError if the structure is multi-chain/has insertion codes,
    # defines a position outside [first_index, first_index+length), or
    # disagrees with rep at any position it does cover
    chain.represents(
        positions=positions, sequence=list(rep), allow_missing=True, raise_invalid=True
    )

    bb_atoms = chain.atom_array[struc.filter_backbone(chain.atom_array)]
    if len(bb_atoms) == 0:
        raise ValueError("Structure chain has no peptide backbone (N/CA/C) atoms")

    coords = np.full((length, 3, 3), np.nan, dtype=np.float32)
    offsets = bb_atoms.res_id.astype(int) - first_index
    for axis, atom_name in enumerate(BACKBONE_ATOMS):
        selected = bb_atoms.atom_name == atom_name
        at = offsets[selected]
        inside = (at >= 0) & (at < length)
        coords[at[inside], axis] = bb_atoms.coord[selected][inside]
    return coords


class MIFST(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around MIF-ST's whole-sequence, structure-conditioned
    log-likelihood scoring.
    """
    available = IMPORT_AVAILABLE
    name: str = "MIF-ST"
    citations: list[str] = [
        # Masked inverse folding with sequence transfer for protein
        # representation learning
        "10.1101/2022.05.25.493516",
    ]
    requires_target: bool = True
    # MIF-ST conditions on a fixed backbone graph, so a scored sequence must
    # align 1:1 with the structure
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = ["structures"]
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        checkpoint_name: str = CHECKPOINT_NAME,
        structure_name: str | None = None,
        batch_size: BatchSize = 8,
        device: DeviceType = "cpu",
    ):
        """
        Instantiate a MIF-ST model.

        Parameters
        ----------
        checkpoint_name
            Checkpoint to load through sequence_models.pretrained; "mifst" is
            MIF-ST, "mif" the same architecture without sequence transfer
        structure_name
            Key into the entity's `structures` mapping to select which
            structure to use, if more than one is present. If None (default)
            and there is exactly one structure, that one is used
        batch_size
            Number of sequences scored per forward pass
        device
            Device to load the model onto and run scoring on
        """
        if not self.available:
            raise ImportError(
                "torch/sequence-models dependencies could not be imported. "
                "Install them to use this model."
            )

        self.checkpoint_name = checkpoint_name
        self.structure_name = structure_name
        self.batch_size = batch_size
        self.device = device

        self._system: System | None = None
        self._model = None
        self._collater = None
        # (dist, omega, theta, phi) structure features, which depend only on
        # the backbone and so are computed once in build()
        self._features: tuple[Any, ...] | None = None

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self._features is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle a single-component protein system"

        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have a defined rep sequence"

        if not target.structures:
            return False, "Must provide a structure (entity.structures) for model inference"

        return True, ""

    def _select_structure_chain(self, target) -> Structure:
        """
        Select the single structure chain to condition on. MIF-ST builds its
        graph from one backbone, so unlike ESM-IF1 it has no multi-chain
        context path.
        """
        structures = target.structures
        if self.structure_name is not None:
            if self.structure_name not in structures:
                raise ValueError(
                    f"structure_name={self.structure_name!r} not found in entity "
                    f"structures, available keys: {list(structures.keys())}"
                )
            structure = structures[self.structure_name]
        else:
            if len(structures) != 1:
                raise ValueError(
                    "Entity has more than one structure; set structure_name to "
                    f"select one of: {list(structures.keys())}"
                )
            structure = next(iter(structures.values()))

        chains = list(ensure_sequence(structure))
        if len(chains) != 1 or len(chains[0].chains()) != 1:
            raise ValueError(
                "MIF-ST conditions on a single chain, got "
                f"{sum(len(c.chains()) for c in chains)}"
            )
        return chains[0]

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)

        status_start(status_callback, "Loading MIF-ST and extracting structure coordinates")

        self._system = system
        self._features = None
        target = system[0]

        rep = "".join(target.rep)
        coords = _extract_backbone_coords(
            self._select_structure_chain(target),
            target.first_index,
            len(rep),
            rep,
        )

        dist, omega, theta, phi = process_coords(
            {name: coords[:, axis] for axis, name in enumerate(BACKBONE_ATOMS)}
        )
        # kept on CPU: the collater allocates its own output tensors there and
        # assigns these into them, so a device mismatch would break it. Only
        # the collated result is moved to self.device.
        self._features = tuple(
            torch.tensor(f, dtype=torch.float) for f in (dist, omega, theta, phi)
        )

        self._model, self._collater = load_model_and_alphabet(self.checkpoint_name)
        self._model = self._model.to(self.device).eval()

        status_done(status_callback, "MIF-ST model finished building")

        return self

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score full sequences as the sum of MIF-ST's per-position log
        likelihoods of their own residues, conditioned on the fixed structure
        built in build().
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        if len(instances) == 0:
            return []

        status_start(status_callback, "Scoring sequences")

        sequences = ["".join(str(s) for s in i[0].rep) for i in instances]

        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch = sequences[start:start + self.batch_size]
                # every sequence is scored against the same backbone, so the
                # cached structure features are simply repeated across the batch
                src, nodes, edges, connections, edge_mask = self._collater(
                    [[seq, *self._features] for seq in batch]
                )
                src = src.to(self.device)
                logits = self._model(
                    src,
                    nodes.to(self.device),
                    edges.to(self.device),
                    connections.to(self.device),
                    edge_mask.to(self.device),
                    result="logits",
                )
                # src is the collater's own tokenization, so gathering on it
                # needs no alphabet handling here
                token_logprobs = F.log_softmax(logits, dim=-1).gather(
                    -1, src.unsqueeze(-1)
                ).squeeze(-1)
                scores.extend(token_logprobs.sum(dim=-1).cpu().tolist())

                if status_callback:
                    done = min(start + self.batch_size, len(sequences))
                    status_callback(
                        "running", 100.0 * done / len(sequences),
                        f"Scored {done}/{len(sequences)}",
                    )

        status_done(status_callback, "Scoring complete")

        return assign_scores_to_instances(instances, np.asarray(scores, dtype=float))
