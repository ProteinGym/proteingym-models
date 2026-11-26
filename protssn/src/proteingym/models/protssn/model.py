"""
Wrapper class around ProtSSN (https://github.com/tyang816/ProtSSN)
"""
import io
from argparse import Namespace
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np
import pandas as pd

from evedesign.model import (
    BaseModel,
    ConditionalMutationScorer,
    MutationScorer,
    Scorer,
    assign_scores_to_instances,
)
from evedesign.structure import Structure
from evedesign.system import Entity, System, SystemInstance
from evedesign.types import StatusCallback
from evedesign.utils import ensure_sequence, status_done, status_start

try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as biotite_pdb
    import torch

    from .upstream.dataset_utils import NormalizeProtein
    from .upstream.models import GNN_model, PLM_model
    from .upstream.mutant_dataset import MutantDataset, one_letter

    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

# Index order of the 20 output logits. Verified identical in all three places
# upstream defines it: MutantDataset.allowable_features["possible_amino_acids"]
# (first 20 of 37), PLM_model.possible_amino_acids, and compute_fitness.py's
# amino_acids_type. A mismatch here would silently mis-score every variant.
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

# Residues the featurizer can encode: one_hot_res() returns False for any index
# >= 20, and safe_index() maps an unknown residue name to the "misc" index, so
# a single non-standard residue makes the whole graph build fail.
_STANDARD_RESIDUES = frozenset(one_letter) if IMPORT_AVAILABLE else frozenset()

# Backbone atoms get_receptor_inference() requires on every residue it keeps
_BACKBONE_ATOMS = ("N", "CA", "C")

# upstream src/config/egnn.yaml, with hidden_channels set from the checkpoint
_EGNN_CONFIG: dict[str, Any] = {
    "edge_attr_dim": 93,
    "output_dim": 20,
    "dropout": 0,
    "n_layers": 6,
    "residual": False,
    "embedding": False,
    "embedding_dim": 64,
    "mlp_num": 2,
}


class _Featurizer(MutantDataset):
    """
    MutantDataset's residue-graph featurization without its Dataset machinery.

    MutantDataset.__init__ scans a directory tree and triggers torch_geometric's
    process()/download() hooks; only the featurizer methods are wanted here, so
    this sets the attributes those methods read and nothing else.
    """

    def __init__(self, c_alpha_max_neighbors: int):
        # upstream MutantDataset defaults, except c_alpha_max_neighbors
        self.num_residue_type = 20
        self.micro_radius = 20
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.cutoff = 30
        self.seq_dist_cut = 64
        self.use_micro = False
        self.use_angle = False
        self.use_omega = False

        from Bio.PDB import PDBParser, ShrakeRupley

        self.sr = ShrakeRupley(probe_radius=1.4, n_points=100)
        self.biopython_parser = PDBParser()
        # deliberately no super().__init__(): see class docstring


def _encodable_residues(atom_array) -> "struc.AtomArray":
    """
    Keep non-hetero standard residues with a complete N/CA/C backbone.

    Parameters
    ----------
    atom_array
        Biotite AtomArray of one chain

    Returns
    -------
    struc.AtomArray
        Residues the upstream featurizer can encode
    """
    atom_array = atom_array[~atom_array.hetero]
    keep = np.zeros(len(atom_array), dtype=bool)
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:]):
        if atom_array.res_name[start] not in _STANDARD_RESIDUES:
            continue
        names = set(atom_array.atom_name[start:stop].tolist())
        if all(a in names for a in _BACKBONE_ATOMS):
            keep[start:stop] = True
    return atom_array[keep]


def _scatter_to_positions(
    node_logits: np.ndarray,
    res_ids: np.ndarray,
    length: int,
    first_index: int,
) -> np.ndarray:
    """
    Map graph nodes onto target sequence positions by residue number.

    Nodes are placed by their residue number rather than their order, so a
    residue dropped before featurization (non-standard, or missing a backbone
    atom) leaves its own position empty without shifting any later one. Every
    position the graph does not cover stays NaN.

    Parameters
    ----------
    node_logits
        (nodes, 20) per-node log-probabilities, in graph node order
    res_ids
        Residue number of each node, in the same order
    length
        Target sequence length
    first_index
        Residue number the target sequence starts at

    Returns
    -------
    np.ndarray
        (length, 20) log-probabilities, NaN where uncovered
    """
    logits = np.full((length, node_logits.shape[1]), np.nan, dtype=float)
    for node, res_id in enumerate(res_ids):
        pos = int(res_id) - first_index
        if 0 <= pos < length:
            logits[pos] = node_logits[node]
    return logits


class ProtSSN(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around ProtSSN zero-shot substitution scoring.
    """
    available = IMPORT_AVAILABLE
    name: str = "ProtSSN"
    citations: list[str] = [
        # Semantical and geometrical protein encoding toward enhanced
        # bioactivity and thermostability (eLife 2024)
        "10.7554/eLife.98033",
    ]

    requires_target: bool = True
    # one graph node per target residue
    requires_fixed_length: bool = True
    # the 20-way output has no gap or insert symbol
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
        gnn_model_path: str = "/opt/program/data/protssn_k20_h512.pt",
        norm_dir: str = "/opt/program/data/norm",
        plm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        c_alpha_max_neighbors: int = 20,
        hidden_channels: int = 512,
        structure_name: str | None = None,
        allow_missing_structure: bool = True,
        device: str | None = None,
    ):
        """
        Instantiate a ProtSSN model.

        Parameters
        ----------
        gnn_model_path
            Path to a protssn_k<k>_h<h>.pt checkpoint. The k and h it encodes
            must match c_alpha_max_neighbors and hidden_channels.
        norm_dir
            Directory holding upstream's cath_k<k>_mean_attr.pt normalization
            statistics, one per supported k.
        plm_model_name
            ESM-2 checkpoint whose residue embeddings are the graph's node
            features, fetched from the Hub on first use.
        c_alpha_max_neighbors
            Neighbours per node (k) when building the CA residue graph; selects
            the matching normalization file.
        hidden_channels
            EGNN hidden width (h) of the checkpoint.
        structure_name
            Key into the entity's structures mapping, if it has more than one.
            If None, the entity must have exactly one.
        allow_missing_structure
            If True, target positions the structure does not cover score as 0
            contribution. If False, raise instead.
        device
            Device to run on ("cpu" or "cuda"; None = cuda if available).
        """
        if not self.available:
            raise ImportError(
                "torch/transformers/torch_geometric/biopython/biotite dependencies "
                "could not be imported. Install them to use this model."
            )

        # the checkpoint names its own k and h; a mismatch loads the wrong
        # architecture or the wrong normalization and scores silently wrongly
        stem = Path(gnn_model_path).stem
        expected = f"protssn_k{c_alpha_max_neighbors}_h{hidden_channels}"
        if stem.startswith("protssn_k") and stem != expected:
            raise ValueError(
                f"Checkpoint {stem!r} does not match c_alpha_max_neighbors="
                f"{c_alpha_max_neighbors}, hidden_channels={hidden_channels} "
                f"(expected {expected!r})"
            )

        self.gnn_model_path = gnn_model_path
        self.norm_dir = norm_dir
        self.plm_model_name = plm_model_name
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.hidden_channels = hidden_channels
        self.structure_name = structure_name
        self.allow_missing_structure = allow_missing_structure
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._system: System | None = None
        # (length, 20) log-probabilities in AA_ORDER, NaN where uncovered
        self._logits: np.ndarray | None = None

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self._logits is not None

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

    def _select_target_chain(self, target: Entity) -> Structure:
        """
        Pick the structure (see structure_name) and return its first chain.

        Parameters
        ----------
        target
            Target entity holding the structures mapping

        Returns
        -------
        Structure
            First chain of the selected structure
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

        chain = list(ensure_sequence(structure))[0]
        if len(chain.chains()) != 1:
            raise ValueError(f"Expected a single-chain structure, got chains={chain.chains()}")
        return chain

    def _build_graph(self, chain: Structure) -> tuple[Any, np.ndarray]:
        """
        Featurize one chain into upstream's normalized residue graph.

        Reproduces what MutantDataset does across generate_protein_graph_
        evaluation(), process() and get(), for a single in-memory structure.

        Parameters
        ----------
        chain
            Single-chain structure to featurize

        Returns
        -------
        tuple[Any, np.ndarray]
            The graph and the residue numbers of its nodes, in node order
        """
        atom_array = _encodable_residues(chain.atom_array)
        if len(atom_array) == 0:
            raise ValueError(
                "Structure has no standard residues with a complete N/CA/C backbone"
            )
        res_ids, _ = struc.get_residues(atom_array)

        # PDBParser.get_structure accepts a handle, so no temporary file is
        # needed; pre-filtering above makes upstream's own residue filter a
        # no-op, which keeps graph node i aligned with res_ids[i]
        pdb_file = biotite_pdb.PDBFile()
        pdb_file.set_structure(atom_array)
        buffer = io.StringIO()
        pdb_file.write(buffer)
        buffer.seek(0)

        featurizer = _Featurizer(self.c_alpha_max_neighbors)
        rec, _, c_alpha_coords, n_coords, c_coords, seq = (
            featurizer.get_receptor_inference(buffer)
        )
        graph = featurizer.get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, seq)
        if not graph:
            raise ValueError(
                "Upstream featurizer rejected this structure (a residue is "
                "non-standard or missing a backbone atom)"
            )
        if graph.x.shape[0] != len(res_ids):
            raise ValueError(
                f"Graph has {graph.x.shape[0]} nodes but {len(res_ids)} residues "
                "were passed; node-to-residue alignment would be wrong"
            )

        # process(): 3-letter residue names to a 1-letter string, then normalize
        graph.seq = "".join(one_letter[residue] for residue in graph.seq)
        norm_file = Path(self.norm_dir) / f"cath_k{self.c_alpha_max_neighbors}_mean_attr.pt"
        if not norm_file.exists():
            raise FileNotFoundError(
                f"Normalization statistics {norm_file} not found; k="
                f"{self.c_alpha_max_neighbors} needs its own cath_k<k>_mean_attr.pt"
            )
        graph = NormalizeProtein(filename=str(norm_file))(graph)

        # get(): label each node with its own residue type
        graph.y = torch.argmax(graph.x[:, :20], dim=1)
        graph.protein_name = "target"

        return graph, np.asarray(res_ids, dtype=int)

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)
        status_start(status_callback, "Featurizing structure and running ProtSSN")

        self._system = system
        target = system[0]
        rep = "".join(target.rep)
        first_index = target.first_index

        chain = self._select_target_chain(target)
        # raises on residue identity mismatch with rep / out-of-range res_ids
        chain.represents(
            positions=list(range(first_index, first_index + len(rep))),
            sequence=list(rep),
            allow_missing=True,
            raise_invalid=True,
        )

        graph, res_ids = self._build_graph(chain)

        args = Namespace(
            plm=self.plm_model_name,
            gnn="egnn",
            gnn_config={**_EGNN_CONFIG, "hidden_channels": self.hidden_channels},
            c_alpha_max_neighbors=self.c_alpha_max_neighbors,
            device=self.device,
        )
        plm_model = PLM_model(args)
        args.plm_hidden_size = plm_model.model.config.hidden_size
        gnn_model = GNN_model(args)
        gnn_model.load_state_dict(
            torch.load(self.gnn_model_path, map_location=self.device)
        )
        gnn_model.eval()

        # one pass over the wild type yields all 20 log-probabilities per
        # position, so every variant is a lookup rather than a forward pass
        with torch.no_grad():
            batch_graph = plm_model(graph)
            out, _ = gnn_model(batch_graph)
            # upstream's exact expression: the +1e-9 floors a near-zero
            # probability at about -20.7 instead of a much larger negative
            probs = torch.softmax(out[:, :20].float(), dim=1)
            node_logits = torch.log(probs + 1e-9).cpu().numpy()

        del plm_model, gnn_model
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        # by residue number, so a dropped residue shifts nothing downstream
        logits = _scatter_to_positions(node_logits, res_ids, len(rep), first_index)

        n_missing = int(np.isnan(logits[:, 0]).sum())
        if n_missing and not self.allow_missing_structure:
            raise ValueError(
                f"Structure covers only {len(rep) - n_missing}/{len(rep)} target "
                "positions; set allow_missing_structure=True to score them as 0"
            )

        self._logits = logits

        status_done(status_callback, "ProtSSN model built")

        return self

    def _position_score(self, pos: int, mutant: str, wildtype: str) -> float:
        """
        Log-ratio of mutant to wild type at one position.

        Parameters
        ----------
        pos
            Zero-based offset into the target sequence
        mutant
            Mutant residue symbol
        wildtype
            Wild-type residue symbol

        Returns
        -------
        float
            log P(mutant) - log P(wild type), or 0.0 for an uncovered position
        """
        row = self._logits[pos]
        if np.isnan(row[0]):
            return 0.0

        try:
            mut_idx = AA_ORDER.index(mutant.upper())
            wt_idx = AA_ORDER.index(wildtype.upper())
        except ValueError:
            raise ValueError(
                f"Symbol {mutant!r} or {wildtype!r} at position {pos} is not one of "
                f"ProtSSN's 20 output symbols ({AA_ORDER})"
            ) from None

        return float(row[mut_idx] - row[wt_idx])

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score sequences as the summed log P(mutant) - log P(wild type) over
        positions differing from the target.

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

        if len(instances) == 0:
            return []

        status_start(status_callback, "Scoring sequences")

        target_rep = np.asarray(self.system[0].rep)
        scores = []
        for instance in instances:
            rep = np.asarray(instance[0].rep)
            scores.append(sum(
                self._position_score(pos, str(rep[pos]), str(target_rep[pos]))
                for pos in np.flatnonzero(rep != target_rep)
            ))

        status_done(status_callback, "Scoring complete")

        return assign_scores_to_instances(instances, np.asarray(scores, dtype=float))

    def _scan_frame(
        self,
        rows: list[np.ndarray],
        index_tuples: list[tuple],
        names: list[str],
        entities: Sequence[int],
    ) -> pd.DataFrame:
        """
        Assemble a score frame reindexed onto the system's alphabet.

        Parameters
        ----------
        rows
            One row of per-symbol values per index entry
        index_tuples
            Index entries matching rows
        names
            Index level names
        entities
            Entities whose alphabets the columns are reindexed onto

        Returns
        -------
        pd.DataFrame
            Scores with one column per alphabet symbol
        """
        df = pd.DataFrame(
            rows,
            columns=list(AA_ORDER),
            index=pd.MultiIndex.from_tuples(index_tuples, names=names),
        )
        merged_alphabet = Entity.merge_alphabet_symbols([
            self.system[entity_idx].alphabet(
                include_gap=self.handles_deletions,
                include_inserts=self.handles_insertions,
            ) for entity_idx in set(entities)
        ])
        return df.reindex(merged_alphabet, axis=1)

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int | None = None,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None,
    ) -> pd.DataFrame:
        """
        Compute all single substitutions to an instance from the cached
        log-probability matrix, relative to the instance's own residue.

        Parameters
        ----------
        instance
            Instance to scan
        entity
            Entity to scan (only 0 is modelled)
        positions
            Positions to scan; all if None
        status_callback
            Optional callback for progress updates

        Returns
        -------
        pd.DataFrame
            Scores indexed by (entity, pos, ref) with one column per symbol
        """
        self.ready_or_raise()
        self._validate_instances([instance])

        if entity is not None and entity != 0:
            raise ValueError("ProtSSN only models entity 0")

        if positions is not None:
            if entity is None:
                raise ValueError(
                    "Parameter entity must be explicitly specified if using parameter positions"
                )
            self.valid_positions(positions, instance, entity, raise_invalid=True)

        status_start(status_callback, "Computing single mutation scan")

        first_index = self.system[0].first_index
        rep = instance[0].rep
        pos_filter = set(positions) if positions is not None else None

        rows, index_tuples = [], []
        for i in range(self._logits.shape[0]):
            pos = i + first_index
            if pos_filter is not None and pos not in pos_filter:
                continue

            ref_symbol = str(rep[i])
            ref_idx = AA_ORDER.find(ref_symbol.upper())
            ref_score = self._logits[i, ref_idx] if ref_idx >= 0 else np.nan

            rows.append(self._logits[i] - ref_score)
            index_tuples.append((0, pos, ref_symbol))

        df = self._scan_frame(rows, index_tuples, ["entity", "pos", "ref"], [0])

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
        Return raw log-probabilities for one position per instance.

        Rows come from the target-sequence matrix computed in build(), so they
        do not depend on the rest of the instance.

        Parameters
        ----------
        instances
            Instances to score
        entities
            Entity index per instance
        positions
            Position per instance
        status_callback
            Optional callback for progress updates

        Returns
        -------
        pd.DataFrame
            Log-probabilities indexed by (instance, entity, pos)
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

        rows, index_tuples = [], []
        for inst_idx, (instance, entity, pos) in enumerate(zip(instances, entities, positions)):
            rows.append(self._logits[int(pos) - first_index])
            index_tuples.append((inst_idx, int(entity), int(pos)))

        df = self._scan_frame(
            rows, index_tuples, ["instance", "entity", "pos"], entities
        )

        status_done(status_callback, "Conditional scoring complete")

        return df
