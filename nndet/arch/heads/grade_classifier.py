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

    def build_conv_out(self, conv):
        """Emit one internal_channels-wide feature per voxel, not one per anchor slot.

        BaseClassifier.build_conv_out emits ``num_classes * anchors_per_pos`` channels
        so each anchor at a voxel gets its own logits -- correct for detection, where
        anchors sharing a voxel need different box scores. The grade decision has no
        such per-anchor signal: GradeHead projects a single per-voxel feature to grade
        logits with one small Linear layer, so 27 independently-parameterized copies of
        that projection (inherited from `num_classes=internal_channels` at 27
        anchors/position) bought ~11.9M of this head's ~12.8M parameters with no
        matching increase in real capacity. ``forward`` below expands the single
        per-voxel feature across anchors explicitly instead of relying on distinct
        learned channels per anchor.
        """
        return conv(
            self.internal_channels,
            self.internal_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            add_norm=False,
            add_act=False,
            bias=True,
        )

    def forward(self, x: Tensor, level: int, **kwargs) -> Tensor:
        """Produce per-anchor features by repeating each voxel's feature across its
        anchors_per_pos slots, rather than learning anchors_per_pos independent
        projections of it.

        Row order must match the anchor generator / box_logits contract: voxel-major,
        anchor fastest, i.e. ``row = voxel * anchors_per_pos + anchor``
        (`nndet/core/boxes/anchors.py`, `BaseClassifier.forward`). `repeat_interleave`
        on the voxel axis produces exactly that; `repeat` would instead tile the whole
        voxel sequence `anchors_per_pos` times (``row = anchor * num_voxels +
        voxel``), which is shape-identical but silently attaches each detection's
        grade to the wrong voxel.
        """
        features = self.conv_out(self.conv_internal(x))
        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        features = features.permute(*axes).contiguous()
        features = features.view(x.size()[0], -1, self.internal_channels)
        return features.repeat_interleave(self.anchors_per_pos, dim=1)


class GradeClassifierHead(nn.Module):
    def __init__(self, feature_extractor: GradeAnchorFeatureExtractor, grade_head: GradeHead):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.grade_head = grade_head

    def forward(self, fmaps: List[Tensor]) -> Tensor:
        features = [self.feature_extractor(p, level=level) for level, p in enumerate(fmaps)]
        features = torch.cat(features, dim=1).flatten(0, -2)
        return self.grade_head(features)
