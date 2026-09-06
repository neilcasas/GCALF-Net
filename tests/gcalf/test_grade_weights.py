import pytest
import torch

from nndet.io.load import save_pickle
from nndet.ptmodule.retinaunet.base import RetinaUNetModule


def _case(tmp_path, name, grades, supervised):
    properties_path = tmp_path / f"{name}.pkl"
    save_pickle({"grades": grades, "grade_supervised": supervised}, properties_path)
    return {"properties_file": str(properties_path)}


def test_grade_weights_use_only_supervised_lesions_and_are_mean_normalized(tmp_path):
    dataset = {
        "a": _case(tmp_path, "a", {"1": 2, "2": 3}, {"1": True, "2": True}),
        "b": _case(tmp_path, "b", {"1": 2, "2": 4, "3": 5}, {"1": True, "2": True, "3": True}),
        "c": _case(tmp_path, "c", {}, {"1": False}),
    }

    weights = RetinaUNetModule.compute_grade_class_weights(dataset)

    expected = torch.tensor([0.5, 1.0, 1.0, 1.0])
    expected = expected / expected.mean()
    assert torch.allclose(weights, expected)


def test_grade_weights_reject_a_training_fold_missing_a_grade(tmp_path):
    dataset = {"a": _case(tmp_path, "a", {"1": 2}, {"1": True})}

    with pytest.raises(ValueError, match="no grade-supervised lesions"):
        RetinaUNetModule.compute_grade_class_weights(dataset)
