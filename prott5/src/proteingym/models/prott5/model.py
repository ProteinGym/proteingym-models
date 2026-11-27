"""
ProtT5 (ProtTrans, https://doi.org/10.1109/TPAMI.2021.3095381) as a frozen
embedder feeding a supervised head.

ProtT5 is an encoder-decoder trained with span corruption, so it has no
masked-marginal scoring convention like ESM's. Its published use, and the one
the literature cites it for, is as a sequence encoder; this wrapper therefore
loads the encoder only and fits a scikit-learn head on the pooled embeddings,
the same shape as the ESM-2 embedding baseline.
"""
from typing import Any, Literal, Self, Sequence

from evedesign.dataset import LabeledInstanceDataset
from evedesign.model import BaseModel, Transformer
from evedesign.models.supervised import SklearnPredictorOnEmbeddingsScores
from evedesign.system import Protein, System, SystemInstance
from evedesign.types import BatchSize, DeviceType, StatusCallback
from evedesign.utils import model_param_context

try:
    import re

    import torch
    from transformers import T5EncoderModel, T5Tokenizer  # noqa
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

# U/Z/O/B do have vocabulary entries, but ProtT5 was trained with ProtTrans'
# preprocessing, which folds them to X - so those entries are effectively
# untrained. Mapping keeps inference consistent with training.
_RARE_RESIDUES = "UZOB"


class ProtT5Embedder(BaseModel, Transformer):
    """
    Per-residue ProtT5 encoder embeddings.
    """
    available = IMPORT_AVAILABLE
    name: str = "ProtT5 embedder"
    citations: list[str] = ["doi:10.1109/TPAMI.2021.3095381"]

    requires_target: bool = True
    # T5 uses relative position embeddings, so unlike ESM there is no hard
    # context limit and length may vary between instances
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False
    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = []

    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        batch_size: BatchSize = 8,
        device: DeviceType = "cpu",
        keep_model_after_pred: bool = True,
    ):
        if not self.available:
            raise ImportError(
                "transformers/sentencepiece could not be imported. Installed?"
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.keep_model_after_pred = keep_model_after_pred
        self.model = None
        self.tokenizer = None
        self._system = None

    def ready(self):
        return self._system is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"
        if len(system) != 1 or not isinstance(system[0], Protein):
            return False, "Can only handle single-component protein system"
        if not system[0].defined_sequence():
            return False, "Entity must have defined rep sequence"
        return True, ""

    def _load_model(self):
        if self.model is not None:
            return
        # the decoder is never used, so load the encoder alone - this is
        # upstream's own recommendation for embedding and halves the weights
        self.model = T5EncoderModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name, do_lower_case=False
        )
        self.model.eval()

    def _delete_model(self):
        self.model = None
        self.tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,  # noqa
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system
        return self

    @staticmethod
    def _to_tokenizer_input(rep: Sequence[str]) -> str:
        """ProtT5 expects residues space-separated, with UZOB folded to X."""
        return " ".join(re.sub(f"[{_RARE_RESIDUES}]", "X", "".join(rep).upper()))

    def _residue_positions(self, token_ids: Sequence[int]) -> list[int]:
        """Indices of the residue tokens, i.e. everything that is not special.

        Derived rather than hardcoded: T5 appends </s> and has no BOS, so the
        ESM convention of slicing [1:L+1] would shift every embedding by one.
        Deriving it also drops padding in batched calls.
        """
        special = set(self.tokenizer.all_special_ids)
        return [i for i, t in enumerate(token_ids) if t not in special]

    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None,  # noqa
    ) -> list[SystemInstance]:
        self.ready_or_raise()
        for instance in instances:
            self.system.valid_instance(
                instance,
                validate_reps=True,
                require_reps=True,
                fixed_length=False,
                allow_deletions=True,
                raise_invalid=True,
            )

        entity = 0 if entity is None else entity
        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        with model_param_context(
            self._load_model, self._delete_model, self.keep_model_after_pred
        ):
            transformed = []
            for start in range(0, len(instances), self.batch_size):
                batch = instances[start:start + self.batch_size]
                sequences = [self._to_tokenizer_input(i[0].rep) for i in batch]
                inputs = self.tokenizer(
                    sequences, return_tensors="pt", padding=True
                ).to(self.device)

                with torch.no_grad():
                    hidden = self.model(**inputs).last_hidden_state

                for i, instance in enumerate(batch):
                    keep = self._residue_positions(inputs.input_ids[i].tolist())
                    if len(keep) != len(batch[i][0].rep):
                        raise ValueError(
                            f"Tokenized {len(keep)} residues for a sequence of "
                            f"{len(batch[i][0].rep)}; ProtT5 tokenization is "
                            "misaligned with the rep sequence"
                        )
                    new_entity = instance[0].copy()
                    new_instance = instance.copy()
                    new_instance.entity_instances = [new_entity]
                    new_entity.embedding = hidden[i, keep].cpu().numpy()
                    new_instance.data = [new_entity]
                    # encoder-only: there are no logits, so no score is set
                    new_instance.score = None
                    new_instance.confidence = None
                    transformed.append(new_instance)

        return transformed


class ProtT5(SklearnPredictorOnEmbeddingsScores):
    """
    Any scikit-learn regressor fitted on pooled ProtT5 encoder embeddings.
    """
    name: str = "prott5"
    citations: list[str] = [
        # ProtTrans: towards cracking the language of life's code
        "10.1109/TPAMI.2021.3095381",
        # Scikit-learn: Machine Learning in Python
        "10.48550/arXiv.1201.0490",
    ]

    def __init__(
        self,
        predictor: str = "Ridge",
        predictor_kwargs: dict[str, Any] | None = None,
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
        pooling: Literal["mean", "max"] = "mean",
        batch_size: BatchSize = 8,
        device: DeviceType = "cpu",
    ):
        """
        Instantiate a supervised head on ProtT5 embeddings.

        Parameters
        ----------
        predictor
            Name of any scikit-learn regressor, as listed by
            sklearn.utils.all_estimators(type_filter=["regressor"])
        predictor_kwargs
            Keyword arguments forwarded to that regressor's constructor
        model_name
            ProtT5 checkpoint on the Hub. The encoder alone is loaded, so the
            encoder-only "-enc" variants work and are smaller
        pooling
            How per-residue embeddings are reduced to one vector per sequence
        batch_size
            Number of sequences embedded per forward pass
        device
            Device the embedder runs on ("cpu", "cuda", or "mps")
        """
        super().__init__(
            predictor=predictor,
            predictor_kwargs=predictor_kwargs,
            embedder=ProtT5Embedder(
                model_name=model_name,
                batch_size=batch_size,
                device=device,
            ),
            scorer=None,
            use_embeddings=True,
            use_scores=False,
            target_name=None,
            pooling=pooling,
            batch_size=batch_size,
        )

    def build(
        self,
        system: System,
        data: LabeledInstanceDataset,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        # the embedder holds no learned state, so it is built here on the same
        # system rather than outside, as evedesign's convention expects
        self.embedder.build(system)

        return super().build(system, data=data, status_callback=status_callback)
