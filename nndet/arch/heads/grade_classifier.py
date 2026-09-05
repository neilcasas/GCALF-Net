"""Per-anchor feature extraction and layout for the masked GGG2-5 grade head
(ARCHITECTURE.md Sec 8).

BaseClassifier's conv tower and forward already produce exactly the per-anchor
layout the grade head needs: image-major, then per-level concatenation, then
per-anchor-slot -- the same layout DetectionHead.forward's box_logits uses, so
grade features align position-for-position with box_logits/sampled_pos_inds.
GradeAnchorFeatureExtractor reuses that tower via subclassing rather than
duplicating it, and only reinterprets the final projection: BaseClassifier's
num_classes-per-anchor-slot output becomes a plain internal_channels-per-
anchor-slot feature vector, which GradeHead
(nndet/arch/encoder/gcalf/grade_head.py) projects to 4 grade logits.
GradeClassifierHead composes the two and concatenates across levels exactly
like DetectionHead.forward does for box_logits.
"""
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from nndet.arch.encoder.gcalf.grade_head import GradeHead
from nndet.arch.heads.classifier import BaseClassifier


class GradeAnchorFeatureExtractor(BaseClassifier):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 **kwargs):
        self.prior_prob = None  # unused: this head has no class-prior init
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            num_classes=internal_channels,  # reinterpreted: per-anchor feature width, not a class count
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            **kwargs,
        )

    def compute_loss(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            "GradeAnchorFeatureExtractor only extracts features; "
            "nndet.arch.encoder.gcalf.grade_head.grade_loss computes the masked loss")

    def box_logits_to_probs(self, box_logits: Tensor) -> Tensor:
        raise NotImplementedError("GradeAnchorFeatureExtractor emits features, not logits")


class GradeClassifierHead(nn.Module):
    def __init__(self, feature_extractor: GradeAnchorFeatureExtractor, grade_head: GradeHead):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.grade_head = grade_head

    def forward(self, fmaps: List[Tensor]) -> Tensor:
        features = [self.feature_extractor(p, level=level) for level, p in enumerate(fmaps)]
        features = torch.cat(features, dim=1).flatten(0, -2)
        return self.grade_head(features)
