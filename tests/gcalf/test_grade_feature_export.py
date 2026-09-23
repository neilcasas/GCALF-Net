import torch
import torch.nn as nn

from nndet.arch.encoder.gcalf.grade_head import GradeHead
from nndet.arch.heads.grade_classifier import GradeClassifierHead


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
