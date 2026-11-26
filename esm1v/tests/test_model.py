"""Checks the ESM-1v ensembling, without downloading any checkpoint."""

from evedesign.model import assign_scores_to_instances
from evedesign.models.esm2 import ESM2
from evedesign.system import EntityInstance, SystemInstance

from proteingym.models.esm1v.model import ENSEMBLE, ESM1v


def instances():
    return [
        SystemInstance([EntityInstance(rep="AC")]),
        SystemInstance([EntityInstance(rep="AD")]),
    ]


def test_score_is_the_mean_over_members(monkeypatch):
    model = ESM1v(model_names=["a", "b", "c"])

    used = []
    per_member = iter([[1.0, 3.0], [2.0, 6.0], [3.0, 9.0]])

    def fake_score(self, inst, status_callback=None):
        used.append(self.model_name)
        return assign_scores_to_instances(inst, next(per_member))

    monkeypatch.setattr(ESM2, "score", fake_score)
    monkeypatch.setattr(ESM2, "_delete_model", lambda self: None)

    scored = model.score(instances())

    assert [float(i.score) for i in scored] == [2.0, 6.0]
    assert used == ["a", "b", "c"], "every member must contribute"
    # inherited single-checkpoint methods must not depend on call order
    assert model.model_name == "a"


def test_default_is_the_five_released_checkpoints():
    assert len(ENSEMBLE) == 5
    assert ESM1v().model_names == list(ENSEMBLE)


def test_an_empty_ensemble_is_rejected():
    try:
        ESM1v(model_names=[])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
