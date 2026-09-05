"""Masked 4-logit GGG2-5 grade head (ARCHITECTURE.md Sec 8).

A 4-logit head reading each matched positive detection's feature, parallel to
(not replacing) nnDetection's existing single-class csPCa anchor classifier.
Loss is masked to grade_supervised instances only: unsupervised positives and
all negatives contribute zero grade loss and zero grade-head gradient.
"""
import torch.nn as nn
import torch.nn.functional as F


class GradeHead(nn.Module):
    def __init__(self, in_channels, num_grades=4):
        super().__init__()
        self.classifier = nn.Linear(in_channels, num_grades)

    def forward(self, features):
        return self.classifier(features)


def grade_loss(logits, grade_targets, grade_supervised_mask, class_weights):
    if not grade_supervised_mask.any():
        return logits.new_zeros(())          # legitimate: no supervised lesion in this batch
    supervised_logits = logits[grade_supervised_mask]
    supervised_targets = grade_targets[grade_supervised_mask]
    return F.cross_entropy(supervised_logits, supervised_targets, weight=class_weights)
