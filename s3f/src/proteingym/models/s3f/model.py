"""
Wrapper class around S3F / S2F (https://github.com/DeepGraphLearning/S3F)
"""
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np

from evedesign.model import (
    BaseModel,
    Scorer,
    MutationScorer,
    ConditionalMutationScorer,
    assign_scores_to_instances,
)
from evedesign.structure import Structure
from evedesign.system import Entity, System, SystemInstance
from evedesign.types import StatusCallback
from evedesign.utils import ensure_sequence, status_done, status_start

# KeOps needs a writable, host-named kernel cache: copy the kernels baked into
# the read-only image to a temp folder under this host's name.
_BAKED_KEOPS_CACHE = Path("/opt/program/keops_cache")
if "KEOPS_CACHE_FOLDER" not in os.environ and _BAKED_KEOPS_CACHE.is_dir():
    import platform
    import sys

    _keops_cache = Path(tempfile.gettempdir()) / f"s3f_keops_cache_{os.getuid()}"
    _host_folder = _keops_cache / ("_".join(platform.uname()[:3]) + f"_p{sys.version.split(' ')[0]}")
    if not _host_folder.exists():
        for _baked in _BAKED_KEOPS_CACHE.iterdir():
            if _baked.is_dir():
                shutil.copytree(_baked, _host_folder, dirs_exist_ok=True)
    os.environ["KEOPS_CACHE_FOLDER"] = str(_keops_cache)

try:
    import biotite.structure.io.pdb as biotite_pdb
    import torch
    from torch.nn import functional as F

    # pre-load to bypass torchdrug's torch_geometric import hack, as upstream does
    import torch_geometric.data  # noqa: F401
    from torchdrug import core, data

    # torchdrug torch.load()s the ESM-2 file, whose pickle holds an
    # argparse.Namespace; allowlist it for torch>=2.6's weights_only default
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])

    from .s3f import dataset as s3f_dataset
    from .s3f import gvp, model as s3f_model, protein as s3f_protein, surface, task  # noqa: F401

    # Run KeOps on CPU only (its GPU backend needs CUDA headers); the network's
    # kNN gets CUDA tensors on GPU, so run it on CPU copies.
    import pykeops

    pykeops.config.gpu_available = False
    _keops_knn_atoms = surface.knn_atoms

    def _knn_atoms_on_cpu(x, y, k, batch_x=None, batch_y=None):
        device = x.device
        idx, dists = _keops_knn_atoms(
            x.cpu(), y.cpu(), k,
            batch_x=None if batch_x is None else batch_x.cpu(),
            batch_y=None if batch_y is None else batch_y.cpu(),
        )
        return idx.to(device), dists.to(device)

    surface.knn_atoms = _knn_atoms_on_cpu

    IMPORT_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as e:
    IMPORT_AVAILABLE = False
    _IMPORT_ERROR = e

DATA_DIR = Path(os.environ.get("S3F_DATA_DIR", "/opt/program/data"))

# ESM-2 context minus BOS/EOS (torchdrug EvolutionaryScaleModeling.max_input_length)
ESM_WINDOW = 1022

# task configs from upstream config/evaluate/{s3f,s2f}.yaml
_STRUCTURE_MODELS = {
    "s3f": {
        "class": "SurfGVP",
        "node_in_dim": [1280, 0],
        "node_h_dim": [256, 16],
        "edge_in_dim": [16, 1],
        "edge_h_dim": [64, 1],
        "surf_in_dim": [42, 0],
        "surf_edge_in_dim": [16, 1],
        "num_surf_res_neighbor": 3,
        "num_surf_graph_neighbor": 16,
        "num_layers": 5,
        "vector_gate": True,
        "readout": "mean",
        "drop_rate": 0.1,
    },
    "s2f": {
        "class": "GVPGNN",
        "node_in_dim": [1280, 0],
        "node_h_dim": [256, 16],
        "edge_in_dim": [16, 1],
        "edge_h_dim": [64, 1],
        "num_layers": 5,
        "vector_gate": True,
        "readout": "mean",
        "drop_rate": 0.1,
    },
}
# upstream evaluate configs: batch_size 2 for S3F, 4 for S2F
_DEFAULT_BATCH_SIZE = {"s3f": 2, "s2f": 4}


def _task_config(variant: str, esm_dir: str) -> dict:
    """
    Build the torchdrug task config for a variant.

    Parameters
    ----------
    variant
        "s3f" or "s2f"
    esm_dir
        Directory holding the ESM-2 weights

    Returns
    -------
    dict
        Config dict for core.Configurable.load_config_dict
    """
    return {
        "class": "ResidueTypePrediction",
        "mask_rate": 0.15,
        "dropout": 0.5,
        # disabled: ProteinGym2 structures carry no per-residue pLDDT, so the
        # ESM-2 fallback would apply everywhere
        "plddt_threshold": None,
        "model": {
            "class": "FusionNetwork",
            "sequence_model": {"class": "MyESM", "path": esm_dir, "model": "ESM-2-650M"},
            "structure_model": dict(_STRUCTURE_MODELS[variant]),
        },
        "graph_construction_model": {
            "class": "GraphConstruction",
            "node_layers": [{"class": "AlphaCarbonNode"}],
            "edge_layers": [{"class": "SpatialEdge", "radius": 10.0, "min_distance": 0}],
            "edge_feature": None,
        },
    }


def get_optimal_window(mutation_position_relative, seq_len_wo_special, model_window):
    """
    Pick a model window around a mutation (verbatim from upstream evaluate.py).

    Parameters
    ----------
    mutation_position_relative
        0-based mutated position
    seq_len_wo_special
        Sequence length
    model_window
        Maximum window length

    Returns
    -------
    list[int]
        [start, end) of the window
    """
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0, seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0, model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [max(0, mutation_position_relative - half_model_window),
                min(seq_len_wo_special, mutation_position_relative + half_model_window)]


def compute_surface(data_dict: dict, device="cpu", seed: int = 0) -> dict:
    """
    Compute the protein surface as upstream script/process_surface.py, in memory.

    Parameters
    ----------
    data_dict
        Protein dict from protein.from_pdb_string(...).to_dict()
    device
        Device for surface generation
    seed
        Seed for the random surface point sampling

    Returns
    -------
    dict
        Surface points, normals, HKS, curvatures and residue-to-surface map
    """
    ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}  # noqa: F841 (upstream)

    with torch.random.fork_rng(devices=[] if str(device) == "cpu" else [device]):
        torch.manual_seed(seed)

        num_residue = len(data_dict["aatype"])
        atom_position = torch.as_tensor(data_dict["atom_positions"][:, :3]).float()  # take backbone atoms
        atom_type = torch.tensor([3, 0, 0], dtype=torch.long)[None, :].repeat(num_residue, 1)  # (N, CA, C)
        atom_position = atom_position.flatten(0, 1).to(device)
        atom_type = F.one_hot(atom_type.flatten(0, 1), num_classes=6).to(device)
        num_atom = len(atom_position)
        batch = torch.zeros((num_atom,), dtype=torch.long, device=device)

        surf_points, surf_normals, _ = surface.atoms_to_points_normals(atom_position, batch, atomtypes=atom_type)
        num_surf_points = len(surf_points)

        # Surface -> residue graph correspondence (num_residue, 3, 21)
        res2surf, _ = surface.knn_atoms(atom_position, surf_points, k=20)
        res2surf = res2surf.view(num_residue, 3, -1)

        batch_surf = torch.zeros((num_surf_points,), dtype=torch.long, device=device)
        surf_curvatures = surface.compute_curvatures(
            surf_points, surf_normals, batch=batch_surf, curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0]
        )

        surf_points = surf_points.cpu().detach().numpy()
        eigs_ratio = 0.01 if num_surf_points > 20000 else 0.06
        surf_eig_vals, surf_eig_vecs, surf_eig_vecs_inv = surface.compute_eigens(
            num_surf_points, surf_points, min_n_eigs=50, eigs_ratio=eigs_ratio
        )
        surf_hks = surface.compute_HKS(surf_eig_vecs, surf_eig_vals, num_t=32, t_min=0.1, t_max=1000, scale=1000)

    return {
        "surf_points": surf_points.astype(np.float32),
        "surf_normals": surf_normals.cpu().detach().numpy().astype(np.float32),
        "surf_hks": surf_hks.astype(np.float32),
        "surf_curvatures": surf_curvatures.cpu().detach().numpy().astype(np.float32),
        "res2surf": res2surf.cpu().detach().numpy(),
    }


class S3F(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around S3F/S2F zero-shot masked-marginal substitution scoring.
    """
    available = IMPORT_AVAILABLE
    name: str = "S3F"
    citations: list[str] = [
        # Multi-Scale Representation Learning for Protein Fitness Prediction
        "10.48550/arXiv.2412.01108",
    ]

    requires_target: bool = True
    # masked marginals over the target sequence on a fixed structure graph
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
        variant: str = "s3f",
        checkpoint_path: str | os.PathLike | None = None,
        esm_dir: str | os.PathLike | None = None,
        structure_name: str | None = None,
        batch_size: int | None = None,
        surface_seed: int = 0,
        surface_device: str = "cpu",
        device: str | None = None,
    ):
        """
        Instantiate an S3F or S2F model.

        Parameters
        ----------
        variant
            "s3f" (sequence + structure + surface) or "s2f" (sequence + structure)
        checkpoint_path
            Head-only (<variant>_head.pth) or full Zenodo (<variant>.pth)
            checkpoint; defaults to $S3F_DATA_DIR/<variant>_head.pth
        esm_dir
            Directory holding esm2_t33_650M_UR50D.pt; defaults to
            $S3F_DATA_DIR/esm-model-weights
        structure_name
            Key into entity.structures if the entity has more than one
        batch_size
            Masked sequences per forward pass; defaults to 2 (S3F) or 4 (S2F)
        surface_seed
            Seed for the random surface point sampling (S3F only)
        surface_device
            Device for surface generation
        device
            Device for the network; defaults to cuda if available, else cpu
        """
        if not self.available:
            raise ImportError(
                "torch/torchdrug/torch_cluster/pykeops dependencies could not be "
                f"imported ({_IMPORT_ERROR!r}). Install them to use this model."
            )
        if variant not in _STRUCTURE_MODELS:
            raise ValueError(f"variant must be one of {list(_STRUCTURE_MODELS)}, got {variant!r}")

        self.variant = variant
        self.checkpoint_path = str(checkpoint_path or DATA_DIR / f"{variant}_head.pth")
        self.esm_dir = str(esm_dir or DATA_DIR / "esm-model-weights")
        self.structure_name = structure_name
        self.batch_size = batch_size or _DEFAULT_BATCH_SIZE[variant]
        self.surface_seed = surface_seed
        self.surface_device = surface_device
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._system: System | None = None
        self._task = None
        self._wild_type = None
        self._surf_graph = None
        # [start, end) of target positions (0-based) covered by the structure
        self._structure_range: tuple[int, int] | None = None
        # {sorted site tuple: (offset, (window_len, 20) log-probs)}
        self._cache: dict[tuple[int, ...], tuple[int, np.ndarray]] = {}

    def __getstate__(self):
        # the network is reloaded from disk on demand; don't pickle 650M params
        state = self.__dict__.copy()
        state["_task"] = None
        return state

    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self._wild_type is not None

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
        Select the target chain from the entity's structures.

        Parameters
        ----------
        target
            Target entity

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

        # first chain is the target; S3F/S2F are single-chain models
        chain = list(ensure_sequence(structure))[0]
        if len(chain.chains()) != 1:
            raise ValueError(f"Expected a single-chain structure, got chains={chain.chains()}")
        return chain

    def _load_task(self):
        """
        Load the checkpoint into a torchdrug task.

        Returns
        -------
        ResidueTypePrediction
            Task in eval mode on self.device
        """
        model_task = core.Configurable.load_config_dict(_task_config(self.variant, self.esm_dir))
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)["model"]
        missing, unexpected = model_task.load_state_dict(state, strict=False)
        # a head-only checkpoint leaves exactly the frozen ESM-2 backbone,
        # already loaded from the public weights file, unset
        missing = [k for k in missing if not k.startswith("model.sequence_model.model.")]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint {self.checkpoint_path} does not match the {self.variant} "
                f"architecture: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        return model_task.to(self.device).eval()

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)
        status_start(status_callback, f"Building {self.variant.upper()} model")

        self._system = system
        self._cache = {}
        target = system[0]
        rep = "".join(target.rep)
        length = len(rep)
        first_index = target.first_index

        chain = self._select_target_chain(target)
        chain.represents(
            positions=list(range(first_index, first_index + length)),
            sequence=list(rep),
            allow_missing=True,
            raise_invalid=True,
        )

        atom_array = chain.atom_array[~chain.atom_array.hetero]
        res_ids = np.unique(atom_array.res_id)
        start, end = int(res_ids.min()) - first_index, int(res_ids.max()) - first_index + 1
        if len(res_ids) != end - start:
            raise ValueError(
                "Structure has missing residues inside its range; S3F needs the "
                "structure to cover one contiguous range of target positions"
            )
        self._structure_range = (start, end)

        pdb_file = biotite_pdb.PDBFile()
        pdb_file.set_structure(atom_array)
        buffer = io.StringIO()
        pdb_file.write(buffer)
        pdb_str = buffer.getvalue()

        # wild-type residue graph, as in upstream evaluate.py
        wild_type = s3f_dataset.bio_load_pdb(io.StringIO(pdb_str))[0]
        ca_index = wild_type.atom_name == wild_type.atom_name2id["CA"]
        wild_type = wild_type.subgraph(ca_index)
        wild_type.view = "residue"
        if wild_type.num_residue != end - start:
            raise ValueError(
                f"Structure has {wild_type.num_residue} residues with a CA atom but "
                f"spans {end - start} positions; every residue needs a CA"
            )
        with wild_type.graph():
            wild_type.start = torch.as_tensor(start)
            wild_type.end = torch.as_tensor(end)

        if self.variant == "s3f":
            status_start(status_callback, "Computing protein surface")
            data_dict = s3f_protein.from_pdb_string(pdb_str).to_dict()
            if len(data_dict["aatype"]) != wild_type.num_residue:
                raise ValueError("Surface residue count does not match the structure graph")
            surf_dict = compute_surface(data_dict, device=self.surface_device, seed=self.surface_seed)
            self._surf_graph = s3f_dataset.load_surface(surf_dict)
            with wild_type.residue():
                wild_type.res2surf = torch.as_tensor(surf_dict["res2surf"])
        else:
            self._surf_graph = None

        self._wild_type = wild_type
        self._task = self._load_task()

        status_done(status_callback, f"{self.variant.upper()} model built")
        return self

    def _masked_sequence(self, sites: tuple[int, ...], protein, mask_id: int):
        """
        Window the target and mask one site set, as upstream evaluate.py load_dataset().

        Windows are also clipped into the structure-covered range.

        Parameters
        ----------
        sites
            Sorted 0-based mutated positions
        protein
            Target sequence as a torchdrug Protein
        mask_id
            Token id of <mask>

        Returns
        -------
        tuple
            Masked windowed protein and window start offset
        """
        length = protein.num_residue
        s_start, s_end = self._structure_range
        if sites and (min(sites) < s_start or max(sites) >= s_end):
            raise ValueError(
                f"Mutated positions {[p + self.system[0].first_index for p in sites]} lie "
                "outside the structure-covered range"
            )
        if length > ESM_WINDOW:
            start, end = get_optimal_window(
                mutation_position_relative=sites[0], seq_len_wo_special=length, model_window=ESM_WINDOW
            )
        else:
            start, end = 0, length
        # clip the window into the structure range
        if start < s_start or end > s_end:
            width = min(end - start, s_end - s_start)
            start = min(max(start, s_start), s_end - width)
            end = start + width
            if sites and (min(sites) < start or max(sites) >= end):
                raise ValueError("Mutated positions do not fit in one model window")

        masked_seq = protein.clone()
        residue_mask = torch.zeros((masked_seq.num_residue,), dtype=torch.bool)
        residue_mask[start:end] = 1
        masked_seq = masked_seq.subresidue(residue_mask)
        with masked_seq.graph():
            masked_seq.start = torch.as_tensor(start)
            masked_seq.end = torch.as_tensor(end)

        node_index = torch.tensor(sites, dtype=torch.long) - start
        with masked_seq.residue():
            masked_seq.residue_feature[node_index] = 0
            masked_seq.residue_type[node_index] = mask_id
        return masked_seq, start

    def _compute_log_probs(self, site_sets: list[tuple[int, ...]], status_callback=None) -> None:
        """
        Fill the cache with masked log-probabilities for uncached site sets.

        Parameters
        ----------
        site_sets
            Site sets to compute
        status_callback
            Optional status callback
        """
        todo = [s for s in site_sets if s not in self._cache]
        if not todo:
            return
        if self._task is None:
            self._task = self._load_task()

        rep = "".join(self.system[0].rep)
        protein = data.Protein.from_sequence(rep, atom_feature=None, bond_feature=None)
        protein.view = "residue"
        mask_id = self._task.model.sequence_model.alphabet.get_idx("<mask>")

        for chunk_start in range(0, len(todo), 256):
            chunk = todo[chunk_start:chunk_start + 256]
            masked, offsets = zip(*(self._masked_sequence(s, protein, mask_id) for s in chunk))
            mutant_dataset = s3f_dataset.MutantDataset(list(masked), self._wild_type, surf_graph=self._surf_graph)
            loader = data.DataLoader(mutant_dataset, self.batch_size, shuffle=False, num_workers=0)
            seq_prob = []
            with torch.no_grad():
                for batch in loader:
                    batch = _to_device(batch, self.device)
                    prob, sizes = self._task.inference(batch)
                    cum_sizes = sizes.cumsum(dim=0)
                    for i in range(len(sizes)):
                        seq_prob.append(prob[cum_sizes[i] - sizes[i]:cum_sizes[i]])
            for sites, offset, prob in zip(chunk, offsets, seq_prob):
                self._cache[sites] = (offset, torch.log_softmax(prob, dim=-1).float().cpu().numpy())

            if status_callback:
                done = min(chunk_start + len(chunk), len(todo))
                status_callback("running", 100.0 * done / len(todo), f"Scored {done}/{len(todo)} site sets")

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Score sequences by masked marginals relative to the target.

        Sums log p(instance residue) - log p(target residue) over mutated
        positions, masked jointly (upstream evaluate.py get_prob).

        Parameters
        ----------
        instances
            Instances to score
        status_callback
            Optional status callback

        Returns
        -------
        list[SystemInstance]
            Instances with scores assigned; the target scores 0
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        if len(instances) == 0:
            return []

        status_start(status_callback, "Scoring sequences")

        target_rep = np.asarray(self.system[0].rep)
        symbol2id = data.Protein.residue_symbol2id
        per_instance = []
        for instance in instances:
            rep = np.asarray(instance[0].rep)
            sites = tuple(int(p) for p in np.flatnonzero(rep != target_rep))
            mt = [symbol2id.get(str(rep[p]).upper(), -1) for p in sites]
            wt = [symbol2id.get(str(target_rep[p]).upper(), -1) for p in sites]
            if any(i < 0 or i >= 20 for i in mt + wt):
                raise ValueError("Instance contains a residue outside the 20 standard amino acids")
            per_instance.append((sites, mt, wt))

        self._compute_log_probs(sorted({s for s, _, _ in per_instance if s}), status_callback)

        scores = []
        for sites, mt, wt in per_instance:
            if not sites:
                scores.append(0.0)
                continue
            offset, log_prob = self._cache[sites]
            idx = np.asarray(sites) - offset
            scores.append(float((log_prob[idx, mt] - log_prob[idx, wt]).sum()))

        status_done(status_callback, "Scoring complete")

        return assign_scores_to_instances(instances, np.asarray(scores, dtype=float))


def _to_device(obj, device):
    """
    Move a (nested) torchdrug batch to a device.

    Parameters
    ----------
    obj
        Batch to move
    device
        Target device

    Returns
    -------
    Any
        Batch on the device
    """
    if str(device) == "cpu":
        return obj
    from torchdrug import utils as td_utils
    return td_utils.cuda(obj, device=torch.device(device))
