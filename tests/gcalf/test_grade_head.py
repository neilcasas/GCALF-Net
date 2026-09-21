import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nndet.arch.encoder.gcalf.grade_head import (
    GradeHead,
    grade_anchor_class_counts,
    grade_loss,
    grade_loss_weight,
    grade_to_index,
    index_to_grade,
)
from nndet.training.grade_refit import grade_optimizer, reset_grade_branch


def test_batch_with_zero_supervised_lesions_returns_a_loss_disconnected_from_the_graph():
    """A batch with zero grade-supervised lesions is expected, not a bug: the
    returned loss must be a fresh zero, structurally unable to backprop into
    GradeHead (there is nothing to compare gradients against -- it never
    touches the computation graph at all)."""
    head = GradeHead(in_channels=8)
    features = torch.randn(4, 8, requires_grad=True)
    logits = head(features)
    grade_targets = torch.zeros(4, dtype=torch.long)
    grade_supervised_mask = torch.zeros(4, dtype=torch.bool)

    loss = grade_loss(logits, grade_targets, grade_supervised_mask, class_weights=None)

    assert loss.item() == 0.0
    assert loss.requires_grad is False


def test_coral_batch_with_zero_supervised_lesions_keeps_the_same_empty_contract():
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.zeros(4, dtype=torch.long)
    mask = torch.zeros(4, dtype=torch.bool)

    loss = grade_loss(logits, targets, mask, class_weights=None, loss_type="coral")

    assert loss.item() == 0.0
    assert loss.requires_grad is False


def test_mixed_batch_updates_grade_head_only_from_the_supervised_subset():
    torch.manual_seed(0)
    head = GradeHead(in_channels=8)
    features = torch.randn(4, 8)
    grade_targets = torch.tensor([2, 3, 4, 5])
    grade_supervised_mask = torch.tensor([True, False, True, False])

    logits = head(features)
    loss = grade_loss(logits, grade_targets, grade_supervised_mask, class_weights=None)
    loss.backward()
    grad_from_masked_loss = head.classifier.weight.grad.clone()

    head.zero_grad()
    supervised_idx = grade_supervised_mask.nonzero(as_tuple=True)[0]
    reference_logits = head(features[supervised_idx])
    reference_loss = F.cross_entropy(reference_logits, grade_targets[supervised_idx] - 2)
    reference_loss.backward()
    grad_from_supervised_subset_only = head.classifier.weight.grad.clone()

    assert torch.allclose(grad_from_masked_loss, grad_from_supervised_subset_only, atol=1e-6)


def test_grade_loss_applies_the_given_class_weights():
    torch.manual_seed(0)
    head = GradeHead(in_channels=8)
    features = torch.randn(4, 8)
    grade_targets = torch.tensor([2, 3, 4, 5])
    mask = torch.ones(4, dtype=torch.bool)
    logits = head(features)

    uniform_weights = torch.ones(4)
    skewed_weights = torch.tensor([2.0, 0.5, 1.0, 1.0])

    loss_uniform = grade_loss(logits, grade_targets, mask, uniform_weights)
    loss_skewed = grade_loss(logits, grade_targets, mask, skewed_weights)

    assert not torch.allclose(loss_uniform, loss_skewed)


def test_ce_loss_type_is_exactly_the_existing_weighted_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(4, 4)
    grades = torch.tensor([2, 3, 4, 5])
    mask = torch.tensor([True, False, True, True])
    weights = torch.tensor([0.256, 0.622, 1.747, 1.376])

    actual = grade_loss(logits, grades, mask, weights, loss_type="ce")
    expected = F.cross_entropy(logits[mask], grades[mask] - 2, weight=weights)

    assert torch.equal(actual, expected)


def test_coral_head_emits_three_threshold_logits_and_four_class_probabilities():
    head = GradeHead(in_channels=8, loss_type="coral")
    logits = head(torch.randn(5, 8))
    probabilities = head.logits_to_probs(logits)

    assert logits.shape == (5, 3)
    assert probabilities.shape == (5, 4)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5))
    assert torch.all(probabilities >= 0)
    assert head.classifier.bias is None
    assert head.thresholds.shape == (3,)


def test_coral_penalizes_a_two_grade_error_more_than_a_one_grade_error():
    true_g5 = torch.tensor([5])
    supervised = torch.tensor([True])
    predicted_g2 = torch.tensor([[-10.0, -10.0, -10.0]])
    predicted_g4 = torch.tensor([[10.0, 10.0, -10.0]])

    loss_g2 = grade_loss(predicted_g2, true_g5, supervised, None, loss_type="coral")
    loss_g4 = grade_loss(predicted_g4, true_g5, supervised, None, loss_type="coral")

    assert loss_g2 > loss_g4


def test_coral_loss_is_near_zero_for_perfectly_ordered_logits():
    logits = torch.tensor([
        [-20.0, -20.0, -20.0],
        [20.0, -20.0, -20.0],
        [20.0, 20.0, -20.0],
        [20.0, 20.0, 20.0],
    ])
    grades = torch.tensor([2, 3, 4, 5])

    loss = grade_loss(logits, grades, torch.ones(4, dtype=torch.bool), None, loss_type="coral")

    assert loss.item() < 1e-6


def test_output_shape_is_always_num_detections_by_num_grades():
    head = GradeHead(in_channels=8, num_grades=4)
    for num_detections in (0, 1, 5):
        logits = head(torch.randn(num_detections, 8))
        assert logits.shape == (num_detections, 4)


def test_grade_indices_round_trip_between_stored_ggg_and_head_logits():
    grades = torch.tensor([2, 3, 4, 5])
    assert grade_to_index(grades).tolist() == [0, 1, 2, 3]
    assert torch.equal(index_to_grade(grade_to_index(grades)), grades)


def test_grade_loss_rejects_supervised_labels_outside_ggg2_to_5():
    logits = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="GGG2-5"):
        grade_loss(logits, torch.tensor([1]), torch.tensor([True]), class_weights=None)


def test_grade_loss_weight_reconstructs_pooled_weighted_cross_entropy():
    logits_one = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    grades_one = torch.tensor([2, 3])
    mask_one = torch.tensor([True, True])
    logits_two = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    grades_two = torch.tensor([4])
    mask_two = torch.tensor([False])
    weights = torch.tensor([1.0, 2.0, 4.0, 8.0])

    loss_one = grade_loss(logits_one, grades_one, mask_one, weights)
    loss_two = grade_loss(logits_two, grades_two, mask_two, weights)
    weight_one = grade_loss_weight(grades_one, mask_one, weights)
    weight_two = grade_loss_weight(grades_two, mask_two, weights)
    pooled = grade_loss(torch.cat([logits_one, logits_two]), torch.cat([grades_one, grades_two]),
                        torch.cat([mask_one, mask_two]), weights)

    assert torch.allclose((loss_one * weight_one + loss_two * weight_two) / (weight_one + weight_two), pooled,
                          atol=1e-6)
    assert not torch.allclose((loss_one + loss_two) / 2, pooled)
    assert weight_two.item() == 0.0
    assert grade_loss_weight(grades_one, mask_one, None).item() == 2.0


def test_grade_loss_weight_reconstructs_pooled_weighted_coral_loss():
    logits_one = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.5]])
    grades_one = torch.tensor([2, 3])
    mask_one = torch.tensor([True, True])
    logits_two = torch.tensor([[0.0, 0.0, 1.0]])
    grades_two = torch.tensor([4])
    mask_two = torch.tensor([False])
    weights = torch.tensor([1.0, 2.0, 4.0, 8.0])

    loss_one = grade_loss(logits_one, grades_one, mask_one, weights, loss_type="coral")
    loss_two = grade_loss(logits_two, grades_two, mask_two, weights, loss_type="coral")
    weight_one = grade_loss_weight(grades_one, mask_one, weights)
    weight_two = grade_loss_weight(grades_two, mask_two, weights)
    pooled = grade_loss(
        torch.cat([logits_one, logits_two]), torch.cat([grades_one, grades_two]),
        torch.cat([mask_one, mask_two]), weights, loss_type="coral")

    assert torch.allclose((loss_one * weight_one + loss_two * weight_two) / (weight_one + weight_two), pooled)
    assert weight_two.item() == 0.0


def test_coral_refit_resets_thresholds_and_optimizes_the_bare_parameter():
    model = nn.Module()
    model.grade_head = GradeHead(in_channels=4, loss_type="coral")
    with torch.no_grad():
        model.grade_head.thresholds.fill_(3.0)
        model.grade_head.classifier.weight.fill_(2.0)

    reset_grade_branch(model)
    optimizer = grade_optimizer(model.grade_head, weight_decay=1e-3)
    optimized_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert torch.equal(model.grade_head.thresholds, torch.zeros(3))
    assert id(model.grade_head.thresholds) in optimized_parameters
    assert id(model.grade_head.classifier.weight) in optimized_parameters


def test_grade_anchor_counts_measure_only_sampled_supervised_positive_anchors():
    grades = torch.tensor([0, 2, 2, 3, 4, 5, 5])
    supervised = torch.tensor([False, True, True, True, False, True, True])

    counts = grade_anchor_class_counts(grades, supervised)

    assert counts.tolist() == [2, 1, 0, 2]
    assert counts.sum().item() == supervised.sum().item()


def test_grade_anchor_counts_reject_bad_supervised_label():
    with pytest.raises(ValueError, match="GGG2-5"):
        grade_anchor_class_counts(torch.tensor([1]), torch.tensor([True]))
