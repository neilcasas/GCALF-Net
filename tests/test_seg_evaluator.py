import numpy as np
import pytest

from nndet.evaluator.seg import PerCaseSegmentationEvaluator, SegmentationEvaluator


def test_segmentation_evaluator_reports_perfect_dice_for_a_perfect_prediction():
    evaluator = SegmentationEvaluator(per_class=True)
    target = np.asarray([[0, 1, 1, 0]])
    # channel 0 (background) vs channel 1 (foreground) logits that argmax
    # to exactly `target`.
    seg_probs = np.asarray([[[10, -10, -10, 10], [-10, 10, 10, -10]]], dtype=np.float32)

    evaluator.run_online_evaluation(seg_probs, target)
    results, _ = evaluator.finish_online_evaluation()

    assert results["seg_dice"] == pytest.approx(1.0, abs=1e-5)
    assert results["0_seg_dice"] == pytest.approx(1.0, abs=1e-5)


def test_segmentation_evaluator_reports_known_dice_for_a_partial_prediction():
    evaluator = SegmentationEvaluator(per_class=False)
    target = np.asarray([[0, 1, 1, 1]])
    # Predicts foreground only at voxel 1: tp=1, fp=0, fn=2 -> dice = 2/4 = 0.5.
    seg_probs = np.asarray([[[10, -10, 10, 10], [-10, 10, -10, -10]]], dtype=np.float32)

    evaluator.run_online_evaluation(seg_probs, target)
    results, _ = evaluator.finish_online_evaluation()

    assert results["seg_dice"] == pytest.approx(0.5, abs=1e-5)


def test_per_case_segmentation_evaluator_finish_online_evaluation_does_not_raise():
    """Regression test for the `dice_full.mean(axies=0)` typo (nndet/evaluator/seg.py),
    which raised TypeError on first real use since nothing previously exercised this class."""
    evaluator = PerCaseSegmentationEvaluator(classes=["bg", "fg"])
    seg = np.asarray([[0, 1, 1, 0]])
    target = np.asarray([[0, 1, 1, 0]])

    evaluator.run_online_evaluation(seg, target)
    results, _ = evaluator.finish_online_evaluation()

    assert results["dice"] == pytest.approx(1.0, abs=1e-5)
    assert results["dice_cls_0"] == pytest.approx(1.0, abs=1e-5)
