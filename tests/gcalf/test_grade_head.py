import pytest
import torch
import torch.nn.functional as F

from nndet.arch.encoder.gcalf.grade_head import GradeHead, grade_loss, grade_to_index, index_to_grade


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
