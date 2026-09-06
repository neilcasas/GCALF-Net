"""Masked 4-logit GGG2-5 grade head (ARCHITECTURE.md Sec 8).

A 4-logit head reading each matched positive detection's feature, parallel to
(not replacing) nnDetection's existing single-class csPCa anchor classifier.
Loss is masked to grade_supervised instances only: unsupervised positives and
all negatives contribute zero grade loss and zero grade-head gradient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


GRADE_MIN = 2
GRADE_MAX = 5
NUM_GRADES = GRADE_MAX - GRADE_MIN + 1


def grade_to_index(grades):
    """Map stored GGG2-5 labels to the GradeHead's zero-based logits."""
    return grades - GRADE_MIN


def index_to_grade(indices):
    """Map zero-based GradeHead predictions back to GGG2-5 labels."""
    return indices + GRADE_MIN


class GradeHead(nn.Module):
    def __init__(self, in_channels, num_grades=NUM_GRADES):
        super().__init__()
        self.classifier = nn.Linear(in_channels, num_grades)

    def forward(self, features):
        return self.classifier(features)


def grade_loss(logits, grade_targets, grade_supervised_mask, class_weights):
    if not grade_supervised_mask.any():
        return logits.new_zeros(())          # legitimate: no supervised lesion in this batch
    supervised_logits = logits[grade_supervised_mask]
    supervised_targets = grade_targets[grade_supervised_mask]
    if torch.any(supervised_targets < GRADE_MIN) or torch.any(supervised_targets > GRADE_MAX):
        raise ValueError(f"Grade targets must be GGG{GRADE_MIN}-{GRADE_MAX}, got {supervised_targets.tolist()}")
    return F.cross_entropy(supervised_logits, grade_to_index(supervised_targets), weight=class_weights)
