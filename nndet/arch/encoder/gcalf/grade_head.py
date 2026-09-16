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


def grade_loss_weight(grade_targets, grade_supervised_mask, class_weights):
    """Return the denominator used by the masked weighted cross-entropy."""
    if not grade_supervised_mask.any():
        return grade_targets.new_zeros((), dtype=torch.float)
    supervised_targets = grade_targets[grade_supervised_mask]
    if class_weights is None:
        return torch.tensor(supervised_targets.numel(), device=grade_targets.device, dtype=torch.float)
    return class_weights[grade_to_index(supervised_targets)].sum()


def grade_anchor_class_counts(grade_targets, grade_supervised_mask):
    """Count CE-eligible sampled positive anchors by GGG2--5.

    ``grade_targets`` has already been indexed by the detection head's
    positive-anchor sampler.  Counting here therefore measures the class prior
    actually presented to the grade cross-entropy, rather than lesion metadata
    or pre-sampling ATSS candidates.
    """
    if grade_targets.shape != grade_supervised_mask.shape:
        raise ValueError("Grade targets and supervision mask must have the same shape")
    counts = torch.zeros(NUM_GRADES, dtype=torch.long, device=grade_targets.device)
    if not grade_supervised_mask.any():
        return counts
    supervised_targets = grade_targets[grade_supervised_mask]
    if torch.any(supervised_targets < GRADE_MIN) or torch.any(supervised_targets > GRADE_MAX):
        raise ValueError(f"Grade targets must be GGG{GRADE_MIN}-{GRADE_MAX}, got {supervised_targets.tolist()}")
    return torch.bincount(grade_to_index(supervised_targets), minlength=NUM_GRADES)
