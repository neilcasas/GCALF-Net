"""Masked GGG2-5 grade head with CE and CORAL output modes (ARCHITECTURE.md Sec 8).

A grade head reading each matched positive detection's feature, parallel to (not
replacing) nnDetection's existing single-class csPCa anchor classifier.
Loss is masked to grade_supervised instances only: unsupervised positives and
all negatives contribute zero grade loss and zero grade-head gradient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


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
    def __init__(self, in_channels, num_grades=NUM_GRADES, loss_type="ce"):
        super().__init__()
        if loss_type not in ("ce", "coral"):
            raise ValueError(f"Unknown grade loss type: {loss_type!r}")
        if num_grades < 2:
            raise ValueError("GradeHead requires at least two grades")
        self.num_grades = num_grades
        self.loss_type = loss_type
        if loss_type == "coral":
            self.classifier = nn.Linear(in_channels, 1, bias=False)
            self.thresholds = nn.Parameter(torch.zeros(num_grades - 1))
        else:
            self.classifier = nn.Linear(in_channels, num_grades)

    def forward(self, features):
        logits = self.classifier(features)
        if self.loss_type == "coral":
            logits = logits + self.thresholds
        return logits

    def logits_to_probs(self, logits):
        if self.loss_type == "ce":
            return torch.softmax(logits, dim=1)
        q = torch.sigmoid(logits)
        probabilities = torch.cat((1.0 - q[:, :1], q[:, :-1] - q[:, 1:], q[:, -1:]), dim=1)
        invalid = probabilities < 0
        invalid_count = int(invalid.sum().item())
        if invalid_count:
            logger.warning(
                "CORAL grade probabilities violated threshold ordering; "
                f"clamped {invalid_count} entries"
            )
        probabilities = probabilities.clamp_min(0)
        return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(torch.finfo(probabilities.dtype).eps)


def grade_loss(logits, grade_targets, grade_supervised_mask, class_weights, loss_type="ce"):
    if loss_type not in ("ce", "coral"):
        raise ValueError(f"Unknown grade loss type: {loss_type!r}")
    if not grade_supervised_mask.any():
        return logits.new_zeros(())          # legitimate: no supervised lesion in this batch
    supervised_logits = logits[grade_supervised_mask]
    supervised_targets = grade_targets[grade_supervised_mask]
    if torch.any(supervised_targets < GRADE_MIN) or torch.any(supervised_targets > GRADE_MAX):
        raise ValueError(f"Grade targets must be GGG{GRADE_MIN}-{GRADE_MAX}, got {supervised_targets.tolist()}")
    target_indices = grade_to_index(supervised_targets)
    if loss_type == "ce":
        return F.cross_entropy(supervised_logits, target_indices, weight=class_weights)

    ordinal_targets = (target_indices[:, None] > torch.arange(
        logits.shape[1], device=target_indices.device)).to(dtype=supervised_logits.dtype)
    per_sample = F.binary_cross_entropy_with_logits(
        supervised_logits, ordinal_targets, reduction="none").sum(dim=1)
    if class_weights is None:
        sample_weights = torch.ones_like(per_sample)
    else:
        sample_weights = class_weights[target_indices]
    return (per_sample * sample_weights).sum() / sample_weights.sum()


def grade_loss_weight(grade_targets, grade_supervised_mask, class_weights):
    """Return the denominator used by the masked weighted grade loss."""
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
