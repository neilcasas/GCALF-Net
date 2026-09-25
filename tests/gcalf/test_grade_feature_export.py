import pytest
import torch
import torch.nn as nn

from nndet.arch.encoder.gcalf.grade_head import GradeHead
from nndet.arch.heads.grade_classifier import GradeClassifierHead
from scripts.export_grade_features import _validate_export


class IdentityFeatureExtractor(nn.Module):
    def forward(self, features, level):
        return features


def test_grade_classifier_returns_pre_logit_features_only_when_requested():
    head = GradeClassifierHead(IdentityFeatureExtractor(), GradeHead(in_channels=3))
    feature_map = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    logits = head([feature_map])
    diagnostic_logits, features = head([feature_map], return_features=True)

    assert torch.equal(logits, diagnostic_logits)
    assert torch.equal(features, feature_map.reshape(-1, 3))
    assert logits.shape == (8, 4)


def test_export_validation_requires_aligned_instance_ids_and_anchor_ious():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    grades = torch.tensor([2, 5]).numpy()
    supervised = torch.tensor([True, True]).numpy()
    case_ids = torch.tensor([0, 0]).numpy().astype(str)
    instance_ids = torch.tensor([7, 9]).numpy()
    anchor_ious = torch.tensor([0.5, 1.0]).numpy()

    _validate_export(features, grades, supervised, case_ids, instance_ids, anchor_ious)
    with pytest.raises(ValueError, match="anchor IoUs"):
        _validate_export(features, grades, supervised, case_ids, instance_ids, anchor_ious[:1])
