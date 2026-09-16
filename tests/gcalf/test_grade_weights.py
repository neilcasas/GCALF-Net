import numpy as np
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


def test_grade_weights_can_use_the_measured_sampled_anchor_prior():
    weights = RetinaUNetModule.compute_grade_class_weights(anchor_class_counts=[100, 50, 25, 25])

    assert torch.allclose(weights, torch.tensor([0.5, 1.0, 2.0, 2.0]) / 1.375)


def test_anchor_grade_weights_reject_missing_or_malformed_counts():
    with pytest.raises(ValueError, match="Expected 4"):
        RetinaUNetModule.compute_grade_class_weights(anchor_class_counts=[1, 2, 3])
    with pytest.raises(ValueError, match="finite and positive"):
        RetinaUNetModule.compute_grade_class_weights(anchor_class_counts=[1, 2, 0, 4])


def test_bprime_anchor_counts_restore_near_uniform_weights():
    weights = RetinaUNetModule.compute_grade_class_weights(
        anchor_class_counts=[30118, 26390, 23403, 24923])

    assert torch.allclose(weights, torch.tensor([0.863, 0.985, 1.110, 1.043]), atol=0.001)


def test_weighted_grade_mean_ignores_unsupervised_batches():
    assert RetinaUNetModule.weighted_grade_mean([1.6, 0.0], [3.2, 0.0]) == pytest.approx(1.6)
    assert np.mean([1.6, 0.0]) == pytest.approx(0.8)
    assert RetinaUNetModule.weighted_grade_mean([0.0], [0.0]) == float("inf")
