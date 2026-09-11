"""Controlled, detector-frozen refitting of an existing GCALF grade branch."""
import torch
import torch.nn as nn

from nndet.arch.encoder.gcalf.grade_head import grade_loss
from nndet.arch.layers.norm import GroupNorm as NNDetGroupNorm


NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d,
              nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm, nn.GroupNorm,
              nn.SyncBatchNorm, nn.LocalResponseNorm, NNDetGroupNorm)


def reset_grade_branch(model):
    """Reset every resettable module in the attached grade branch in-place."""
    if getattr(model, "grade_head", None) is None:
        raise ValueError("The source model has no grade branch to refit")
    for module in model.grade_head.modules():
        reset = getattr(module, "reset_parameters", None)
        if reset is not None:
            reset()


def freeze_detector(model):
    """Make all non-grade modules immutable, including their running buffers."""
    if getattr(model, "grade_head", None) is None:
        raise ValueError("The source model has no grade branch to refit")
    grade_parameters = {id(parameter) for parameter in model.grade_head.parameters()}
    for module in model.modules():
        is_grade_module = module is model.grade_head or any(
            id(parameter) in grade_parameters for parameter in module.parameters(recurse=False)
        )
        if not is_grade_module:
            module.eval()
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in grade_parameters
    model.grade_head.train()


def detector_state(model):
    """Clone state outside the grade branch for an exact post-run audit."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("grade_head.")
    }


def assert_detector_unchanged(model, before):
    after = detector_state(model)
    if before.keys() != after.keys():
        raise AssertionError("Detector state keys changed during grade refitting")
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if changed:
        raise AssertionError("Frozen detector parameters or buffers changed: " + ", ".join(changed[:5]))


def grade_optimizer(grade_branch, weight_decay):
    """SGD groups: decay only non-bias, non-normalization grade parameters."""
    decay, no_decay = [], []
    for module in grade_branch.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            (no_decay if name == "bias" or isinstance(module, NORM_TYPES) else decay).append(parameter)
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=0.001, momentum=0.9, nesterov=True,
    )


class GradeRefitLoop:
    """Small explicit loop used by the pilot instead of the full multitask trainer."""
    def __init__(self, module, class_weights, weight_decay):
        self.module = module
        self.class_weights = class_weights
        reset_grade_branch(module.model)
        freeze_detector(module.model)
        self.before_detector = detector_state(module.model)
        self.optimizer = grade_optimizer(module.model.grade_head, weight_decay)
        self.supervised_updates = 0

    def _loss(self, batch):
        with torch.no_grad():
            batch = self.module.pre_trafo(**batch)
        batch = self._to_device(batch)
        model = self.module.model
        prediction, anchors, _ = model(batch["data"])
        labels, matched_boxes = model.assign_targets_to_anchors(anchors, batch["boxes"], batch["classes"])
        _, pos_idx, _ = model.head.compute_loss(prediction, labels, matched_boxes, anchors)
        matched_grades, supervised = model.assign_grades_to_anchors(
            anchors, batch["boxes"], batch["grades"], batch["grade_supervised"])
        grades = torch.cat(matched_grades, dim=0)[pos_idx]
        supervised = torch.cat(supervised, dim=0)[pos_idx]
        loss = grade_loss(prediction["grade_logits"][pos_idx], grades, supervised, self.class_weights)
        denominator = self.class_weights[grades[supervised] - 2].sum() if supervised.any() else None
        return loss, denominator

    def _to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(next(self.module.model.parameters()).device)
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._to_device(item) for item in value)
        return value

    def train_batch(self, batch):
        self.module.model.grade_head.train()
        loss, _ = self._loss(batch)
        if not loss.requires_grad:
            return None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.supervised_updates += 1
        return float(loss.detach())

    @torch.no_grad()
    def validation_loss(self, batches):
        self.module.model.eval()
        numerator = denominator = 0.0
        for batch in batches:
            loss, weight = self._loss(batch)
            if weight is not None:
                numerator += float(loss) * float(weight)
                denominator += float(weight)
        return numerator / denominator if denominator else None

    def verify(self):
        assert_detector_unchanged(self.module.model, self.before_detector)
