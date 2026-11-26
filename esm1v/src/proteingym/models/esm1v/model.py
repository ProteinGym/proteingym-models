"""
Wrapper class around ESM-1v (https://doi.org/10.1101/2021.07.09.450648)
"""
from typing import Sequence

from evedesign.model import assign_scores_to_instances
from evedesign.models.esm2 import ESM2
from evedesign.system import SystemInstance
from evedesign.types import BatchSize, DeviceType, StatusCallback

# the five checkpoints released as ESM-1v; the published model is their mean
ENSEMBLE = tuple(f"esm1v_t33_650M_UR90S_{i}" for i in range(1, 6))


class ESM1v(ESM2):
    """
    ESM-1v scored as the mean over its five released checkpoints.

    ESM-1v shares ESM-2's architecture and tokenizer, so every member loads
    through the inherited transformers path; only the averaging is new.
    """
    name: str = "ESM-1v"
    citations: list[str] = ["doi:10.1101/2021.07.09.450648"]

    def __init__(
        self,
        model_names: Sequence[str] = ENSEMBLE,
        batch_size: BatchSize = 64,
        keep_model_after_build: bool = False,
        device: DeviceType = "cpu",
    ):
        """
        Instantiate an ESM-1v ensemble.

        Parameters
        ----------
        model_names
            Checkpoints to average, each loaded from facebook/<name>. Defaults
            to all five released members. A shorter list scores proportionally
            faster and is useful for smoke tests
        batch_size
            Number of sequences scored per forward pass
        keep_model_after_build
            Forwarded to ESM2
        device
            Device to run the model on ("cpu", "cuda" or "mps")
        """
        if not model_names:
            raise ValueError("model_names must name at least one checkpoint")

        self.model_names = list(model_names)

        # model_dir_path is deliberately not exposed: ESM2 ignores model_name
        # when it is set, which would silently score one local checkpoint five
        # times instead of the ensemble
        super().__init__(
            model_name=self.model_names[0],
            batch_size=batch_size,
            keep_model_after_build=keep_model_after_build,
            device=device,
        )

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        per_member = []
        for model_name in self.model_names:
            self.model_name = model_name
            # ESM2 sets keep_model_after_pred, so the previous member stays
            # resident and _load_model would return it instead of this one
            self._delete_model()
            scored = super().score(instances, status_callback)
            per_member.append([float(i.score) for i in scored])

        self._delete_model()
        # leave the first member selected so inherited single-checkpoint
        # methods (transform, mutation scoring) do not depend on call order
        self.model_name = self.model_names[0]

        means = [sum(member) / len(member) for member in zip(*per_member)]
        return assign_scores_to_instances(instances, means)
