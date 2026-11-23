"""
Wrapper class around ProSST (https://github.com/ai4protein/ProSST)
"""
import io
import warnings
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np
import pandas as pd

from evedesign.constants import VALID_AA_SORTED
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

try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as biotite_pdb

    # biotite >=1.x renamed filter_backbone, which vendored data_utils imports
    if not hasattr(struc, "filter_backbone"):
        struc.filter_backbone = struc.filter_peptide_backbone

    import joblib
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch
    from torch_scatter import scatter_mean
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from .structure.build_graph import generate_graph
    from .structure.build_subgraph import generate_pos_subgraph
    from .structure.encoder.gvp import AutoGraphEncoder
    from .structure.utils.data_utils import convert_graph

    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

STATIC_DIR = Path(__file__).parent / "structure" / "static"
STRUCTURE_VOCAB_SIZES = (20, 64, 128, 512, 1024, 2048, 4096)

# ss_input_ids layout used by upstream's tokenize_structure_sequence():
# 0 = pad, 1 = BOS, 2 = EOS, quantizer cluster k -> k + 3
SS_PAD, SS_BOS, SS_EOS, SS_OFFSET = 0, 1, 2, 3

# backbone atoms build_graph.generate_graph() indexes on every residue (it
# raises KeyError if any is missing)
_BACKBONE_ATOMS = ("N", "CA", "C", "O")


def _load_structure_encoder(device) -> "AutoGraphEncoder":
    # architecture hyperparameters as hardcoded in upstream SSTPredictor
    encoder = AutoGraphEncoder(
        node_in_dim=(20, 3),
        node_h_dim=(256, 32),
        edge_in_dim=(32, 1),
        edge_h_dim=(64, 2),
        num_layers=6,
    )
    encoder.load_state_dict(torch.load(STATIC_DIR / "AE.pt", map_location="cpu"))
    return encoder.to(device).eval()


def _load_codebook(structure_vocab_size: int) -> np.ndarray:
    # Pickled with sklearn 1.2.2, whose KMeans.predict rejects float32 input
    # today; only cluster_centers_ is read, so the unpickling warning is moot
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        kmeans = joblib.load(STATIC_DIR / f"{structure_vocab_size}.joblib")
    return np.asarray(kmeans.cluster_centers_, dtype=np.float64)


def _complete_backbone_residues(atom_array) -> "struc.AtomArray":
    """
    Keep only non-hetero residues with a complete N/CA/C/O backbone.

    Parameters
    ----------
    atom_array
        Biotite AtomArray of one chain

    Returns
    -------
    struc.AtomArray
        Residues that build_graph.generate_graph() can featurize
    """
    atom_array = atom_array[~atom_array.hetero]
    keep = np.zeros(len(atom_array), dtype=bool)
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:]):
        names = set(atom_array.atom_name[start:stop].tolist())
        if all(a in names for a in _BACKBONE_ATOMS):
            keep[start:stop] = True
    return atom_array[keep]


def quantize_structure(
    chain: "Structure",
    encoder: "AutoGraphEncoder",
    codebook: np.ndarray,
    max_distance: float = 10,
    max_batch_nodes: int = 10000,
    device="cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run ProSST's structure quantizer on one chain, in memory.

    Equivalent to upstream SSTPredictor.predict_from_pdb(), except residues
    with an incomplete backbone are dropped instead of failing.

    Parameters
    ----------
    chain
        Single-chain structure to quantize
    encoder
        GVP structure encoder (static/AE.pt)
    codebook
        (vocab, dim) k-means cluster centers
    max_distance
        CA-CA distance cutoff (Angstrom) for the residue graph
    max_batch_nodes
        Node budget per encoder batch
    device
        Device to run the encoder on

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Residue numbers of the quantized residues and their structure tokens
        (cluster ids in [0, len(codebook)))
    """
    atom_array = _complete_backbone_residues(chain.atom_array)
    if len(atom_array) == 0:
        raise ValueError("Structure has no residues with a complete N/CA/C/O backbone")

    res_ids, _ = struc.get_residues(atom_array)

    # generate_graph accepts a file handle, so no temporary file is needed
    pdb_file = biotite_pdb.PDBFile()
    pdb_file.set_structure(atom_array)
    buffer = io.StringIO()
    pdb_file.write(buffer)
    buffer.seek(0)
    graph = generate_graph(buffer, max_distance)

    n_nodes = len(graph.node_s)
    if n_nodes != len(res_ids):
        raise ValueError(
            f"Graph has {n_nodes} residues but structure has {len(res_ids)}"
        )

    # All anchors in one call: sorts the distance matrix once, same subgraphs
    subgraphs = generate_pos_subgraph(
        graph, None, max_distance, list(range(n_nodes)), pure_subgraph=True
    )
    subgraphs = [convert_graph(subgraphs[i]) for i in range(n_nodes)]

    # Subgraphs are independent, so batching by node budget doesn't change results
    batches, current, current_nodes = [], [], 0
    for sg in subgraphs:
        n = sg.node_s.shape[0]
        if current and current_nodes + n > max_batch_nodes:
            batches.append(current)
            current, current_nodes = [], 0
        current.append(sg)
        current_nodes += n
    if current:
        batches.append(current)

    embeddings = []
    with torch.no_grad():
        for batch_graphs in batches:
            batch = Batch.from_data_list(batch_graphs)
            batch.node_s = torch.zeros_like(batch.node_s)
            batch = batch.to(device)
            node_embeddings = encoder.get_embedding(
                (batch.node_s, batch.node_v), batch.edge_index, (batch.edge_s, batch.edge_v)
            )
            graph_embeddings = scatter_mean(node_embeddings, batch.batch, dim=0).cpu()
            embeddings.append(F.normalize(graph_embeddings, p=2, dim=1))

    x = torch.cat(embeddings, dim=0).numpy().astype(np.float64)
    distances = (
        (x ** 2).sum(axis=1, keepdims=True)
        - 2.0 * x @ codebook.T
        + (codebook ** 2).sum(axis=1)[None, :]
    )
    tokens = np.argmin(distances, axis=1)

    return np.asarray(res_ids, dtype=int), tokens


class ProSST(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Wrapper around ProSST zero-shot substitution scoring.
    """
    available = IMPORT_AVAILABLE
    name: str = "ProSST"
    citations: list[str] = [
        # ProSST: Protein Language Modeling with Quantized Structure and
        # Disentangled Attention (NeurIPS 2024)
        "10.1101/2024.04.15.589672",
    ]

    requires_target: bool = True
    # one structure token per target residue
    requires_fixed_length: bool = True
    # vocabulary has no gap/insert tokens
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
        model_name: str = "AI4Protein/ProSST-2048",
        structure_vocab_size: int | None = None,
        structure_name: str | None = None,
        max_distance: float = 10,
        max_batch_nodes: int = 10000,
        allow_missing_structure: bool = True,
        device: str | None = None,
    ):
        """
        Instantiate a ProSST model.

        Parameters
        ----------
        model_name
            HuggingFace hub identifier or local directory of a ProSST
            checkpoint (loaded with trust_remote_code=True).
        structure_vocab_size
            Structure token vocabulary size; must match the checkpoint. If
            None, inferred from the trailing "-<size>" of model_name.
        structure_name
            Key into the entity's structures mapping, if it has more than
            one structure. If None, the entity must have exactly one.
        max_distance
            CA-CA distance cutoff (Angstrom) for the quantizer's residue
            graph (upstream default 10).
        max_batch_nodes
            Node budget per quantizer encoder batch (upstream default 10000).
        allow_missing_structure
            If True, target positions without a complete backbone in the
            structure get the structure pad token. If False, raise instead.
        device
            Device for the quantizer encoder and the PLM; defaults to cuda
            if available, else cpu.
        """
        if not self.available:
            raise ImportError(
                "torch/transformers/torch_geometric/biotite dependencies could not "
                "be imported. Install them to use this model."
            )

        if structure_vocab_size is None:
            suffix = str(model_name).rstrip("/").rsplit("-", 1)[-1]
            if not suffix.isdigit():
                raise ValueError(
                    f"Cannot infer structure_vocab_size from model_name={model_name!r}; "
                    "set it explicitly"
                )
            structure_vocab_size = int(suffix)

        if structure_vocab_size not in STRUCTURE_VOCAB_SIZES:
            raise ValueError(
                f"structure_vocab_size must be one of {STRUCTURE_VOCAB_SIZES}, "
                f"got {structure_vocab_size}"
            )

        self.model_name = model_name
        self.structure_vocab_size = structure_vocab_size
        self.structure_name = structure_name
        self.max_distance = max_distance
        self.max_batch_nodes = max_batch_nodes
        self.allow_missing_structure = allow_missing_structure
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._system: System | None = None
        # (length, 25) log-probabilities over the ProSST tokenizer vocab
        self._logits: np.ndarray | None = None
        self._vocab: dict[str, int] | None = None
        # (length,) structure tokens fed to the model, kept for inspection
        self.structure_tokens: np.ndarray | None = None

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
            First chain of the selected structure, taken as the target
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

    def _structure_tokens(self, target: Entity, rep: str) -> np.ndarray:
        """
        Quantize the target structure and align tokens to rep by residue number.

        Parameters
        ----------
        target
            Target entity holding the structures mapping
        rep
            Target sequence

        Returns
        -------
        np.ndarray
            ss_input_ids for the target residues without BOS/EOS (cluster
            id + 3, or SS_PAD where the structure doesn't cover a position)
        """
        length = len(rep)
        first_index = target.first_index
        chain = self._select_target_chain(target)

        # raises on residue identity mismatch with rep / out-of-range res_ids
        chain.represents(
            positions=list(range(first_index, first_index + length)),
            sequence=list(rep),
            allow_missing=True,
            raise_invalid=True,
        )

        encoder = _load_structure_encoder(self.device)
        codebook = _load_codebook(self.structure_vocab_size)
        res_ids, tokens = quantize_structure(
            chain, encoder, codebook,
            max_distance=self.max_distance,
            max_batch_nodes=self.max_batch_nodes,
            device=self.device,
        )
        del encoder

        ss_ids = np.full(length, SS_PAD, dtype=np.int64)
        for res_id, token in zip(res_ids, tokens):
            pos = int(res_id) - first_index
            if 0 <= pos < length:
                ss_ids[pos] = int(token) + SS_OFFSET

        n_missing = int((ss_ids == SS_PAD).sum())
        if n_missing and not self.allow_missing_structure:
            raise ValueError(
                f"Structure lacks a complete backbone for {n_missing}/{length} target "
                "positions; set allow_missing_structure=True to pad them"
            )

        return ss_ids

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)
        status_start(status_callback, "Quantizing structure and running ProSST")

        self._system = system
        target = system[0]
        rep = "".join(target.rep)

        ss_ids = self._structure_tokens(target, rep)
        self.structure_tokens = ss_ids

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
        # structure embedding is sized vocab + {pad, BOS, EOS}; catches a wrong codebook
        expected_ss_vocab = self.structure_vocab_size + SS_OFFSET
        if model.config.ss_vocab_size != expected_ss_vocab:
            raise ValueError(
                f"Checkpoint {self.model_name!r} has ss_vocab_size="
                f"{model.config.ss_vocab_size}, but structure_vocab_size="
                f"{self.structure_vocab_size} implies {expected_ss_vocab}"
            )
        # transformers 5 leaves the tied MLM decoder randomly initialized
        decoder = model.get_output_embeddings().weight
        embeddings = model.get_input_embeddings().weight
        if decoder.data_ptr() != embeddings.data_ptr() and not torch.equal(decoder, embeddings):
            raise RuntimeError(
                "ProSST MLM decoder is not tied to the word embeddings (randomly "
                "initialized decoder) - check the installed transformers version (<5)"
            )
        model = model.to(self.device).eval()

        tokenized = tokenizer([rep], return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        ss_input_ids = torch.tensor(
            [[SS_BOS, *ss_ids.tolist(), SS_EOS]], dtype=torch.long, device=self.device
        )
        if input_ids.shape != ss_input_ids.shape:
            raise ValueError(
                f"Tokenized sequence shape {tuple(input_ids.shape)} does not match "
                f"structure token shape {tuple(ss_input_ids.shape)}"
            )

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ss_input_ids=ss_input_ids,
            )
            logits = torch.log_softmax(outputs.logits[0, 1:-1, :].float(), dim=-1)

        self._logits = logits.cpu().numpy()
        self._vocab = tokenizer.get_vocab()

        # not retained on self, to keep a dill-pickled model small
        del model
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        status_done(status_callback, "ProSST model built")

        return self

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
        vocab = self._vocab
        scores = []
        for instance in instances:
            rep = np.asarray(instance[0].rep)
            total = 0.0
            for pos in np.flatnonzero(rep != target_rep):
                mut_idx = vocab.get(str(rep[pos]).upper())
                wt_idx = vocab.get(str(target_rep[pos]).upper())
                if mut_idx is None or wt_idx is None:
                    raise ValueError(
                        f"Symbol {rep[pos]!r} or {target_rep[pos]!r} at position "
                        f"{pos + self.system[0].first_index} not in ProSST vocabulary"
                    )
                total += float(self._logits[pos, mut_idx] - self._logits[pos, wt_idx])
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
            raise ValueError("ProSST only models entity 0")

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
        alphabet_idx = [self._vocab[aa] for aa in VALID_AA_SORTED]

        rows = []
        index_tuples = []
        for i in range(self._logits.shape[0]):
            pos = i + first_index
            if pos_filter is not None and pos not in pos_filter:
                continue

            ref_symbol = str(rep[i])
            ref_idx = self._vocab.get(ref_symbol.upper())
            ref_score = self._logits[i, ref_idx] if ref_idx is not None else np.nan

            rows.append(self._logits[i, alphabet_idx] - ref_score)
            index_tuples.append((0, pos, ref_symbol))

        df = pd.DataFrame(
            rows,
            columns=VALID_AA_SORTED,
            index=pd.MultiIndex.from_tuples(index_tuples, names=["entity", "pos", "ref"]),
        )

        merged_alphabet = Entity.merge_alphabet_symbols([
            self.system[0].alphabet(
                include_gap=self.handles_deletions,
                include_inserts=self.handles_insertions,
            )
        ])
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
        alphabet_idx = [self._vocab[aa] for aa in VALID_AA_SORTED]

        rows = []
        index_tuples = []
        for inst_idx, (instance, entity, pos) in enumerate(zip(instances, entities, positions)):
            rows.append(self._logits[int(pos) - first_index, alphabet_idx])
            index_tuples.append((inst_idx, int(entity), int(pos)))

        df = pd.DataFrame(
            rows,
            columns=VALID_AA_SORTED,
            index=pd.MultiIndex.from_tuples(index_tuples, names=["instance", "entity", "pos"]),
        )

        merged_alphabet = Entity.merge_alphabet_symbols([
            self.system[entity_idx].alphabet(
                include_gap=self.handles_deletions,
                include_inserts=self.handles_insertions,
            ) for entity_idx in set(entities)
        ])
        df = df.reindex(merged_alphabet, axis=1)

        status_done(status_callback, "Conditional scoring complete")

        return df
