import numpy as np

from gcalf_eval.harmonized_froc import (
    froc_curve,
    harmonized_froc_events,
    sensitivity_at_fp,
)


def test_froc_uses_iou_matching_and_grade_probability_weighted_candidates():
    gt_boxes = np.asarray([
        [0, 0, 2, 2, 0, 2],
        [5, 5, 7, 7, 5, 7],
    ], dtype=np.float32)
    grades = np.asarray([2, 5])
    supervised = np.asarray([True, True])
    prediction = {
        "pred_boxes": np.asarray([
            [0, 0, 2, 2, 0, 2],
            [5, 5, 7, 7, 5, 7],
            [9, 9, 10, 10, 9, 10],
        ], dtype=np.float32),
        "pred_scores": np.asarray([0.9, 0.95, 0.7], dtype=np.float32),
        "pred_grade_probs": np.asarray([
            [0.8, 0.1, 0.05, 0.05],
            [0.05, 0.05, 0.1, 0.8],
            [0.1, 0.1, 0.1, 0.7],
        ], dtype=np.float32),
    }

    generic, grade_events = harmonized_froc_events(
        prediction["pred_boxes"],
        prediction["pred_scores"],
        prediction["pred_grade_probs"],
        gt_boxes,
        grades,
        supervised,
    )

    assert sum(event["fp"] for event in generic) == 1
    assert sum(event["tp"] for event in grade_events[2]) == 1
    assert sum(event["tp"] for event in grade_events[5]) == 1
    support = {2: 1, 3: 0, 4: 0, 5: 1}
    fp_rates, sensitivities = froc_curve(generic, 1, support)
    assert sensitivity_at_fp(fp_rates, sensitivities[2], 0.5) == 1.0
    assert sensitivity_at_fp(fp_rates, sensitivities[5], 1.0) == 1.0
