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
(nndet/arch/encoder/gcalf/grade_head.py) projects to CE or CORAL grade logits.
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
                 per_anchor_features: bool = False,
                 **kwargs):
        self.per_anchor_features = per_anchor_features
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

    def build_conv_out(self, conv):
        """Emit feature channels per voxel or per anchor slot depending on configuration.

        Default (`per_anchor_features=False`, 1.33M params): emits `internal_channels`
        per voxel and expands across anchors in forward via `repeat_interleave`, sharing
        weights across the 27 anchors at each voxel.

        Legacy/Ablation (`per_anchor_features=True`, 12.83M params): emits
        `internal_channels * anchors_per_pos` channels, giving independent convolutional
        projections to each of the 27 anchor shapes.
        """
        out_channels = (self.internal_channels * self.anchors_per_pos
                        if self.per_anchor_features
                        else self.internal_channels)
        return conv(
            self.internal_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            add_norm=False,
            add_act=False,
            bias=True,
        )

    def forward(self, x: Tensor, level: int, **kwargs) -> Tensor:
        """Produce per-anchor features matching the anchor generator row contract:
        voxel-major, anchor fastest (``row = voxel * anchors_per_pos + anchor``).
        """
        features = self.conv_out(self.conv_internal(x))
        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        features = features.permute(*axes).contiguous()
        if self.per_anchor_features:
            return features.view(x.size()[0], -1, self.internal_channels)
        else:
            features = features.view(x.size()[0], -1, self.internal_channels)
            return features.repeat_interleave(self.anchors_per_pos, dim=1)


class GradeClassifierHead(nn.Module):
    def __init__(self, feature_extractor: GradeAnchorFeatureExtractor, grade_head: GradeHead):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.grade_head = grade_head

    @property
    def loss_type(self):
        return self.grade_head.loss_type

    def logits_to_probs(self, logits: Tensor) -> Tensor:
        """Convert grade logits through the configured grade head."""
        return self.grade_head.logits_to_probs(logits)

    def forward(self, fmaps: List[Tensor]) -> Tensor:
        features = [self.feature_extractor(p, level=level) for level, p in enumerate(fmaps)]
        features = torch.cat(features, dim=1).flatten(0, -2)
        return self.grade_head(features)
