"""Patch-level grade-head monitor using the reporting metric implementation."""
from collections import defaultdict

import numpy as np

from gcalf_eval.grade_metrics import match_grade_predictions, summarize_grade_matches
from nndet.evaluator import AbstractEvaluator


__all__ = ["GradeEvaluator"]


class GradeEvaluator(AbstractEvaluator):
    """Accumulate score-ordered lesion matches for one validation epoch."""
    def __init__(self):
        self.results_list = []

    @classmethod
    def create(cls):
        return cls()

    def reset(self):
        self.results_list = []

    def run_online_evaluation(self, pred_boxes, pred_scores, pred_grade_probs,
                              gt_boxes, gt_grades, gt_grade_supervised):
        for values in zip(pred_boxes, pred_scores, pred_grade_probs,
                          gt_boxes, gt_grades, gt_grade_supervised):
            truth, predicted, misses, false_positives, matched_ungraded = match_grade_predictions(
                *values, return_details=True, return_probabilities=False)
            self.results_list.append({
                "truth": truth,
                "predicted": predicted,
                "misses": misses,
                "false_positives": false_positives,
                "matched_ungraded": matched_ungraded,
            })
        return {}

    def finish_online_evaluation(self):
        results = defaultdict(list)
        for item in self.results_list:
            for key, value in item.items():
                results[key].extend(value) if isinstance(value, list) else results[key].append(value)
        summary = summarize_grade_matches(
            results["truth"], results["predicted"], sum(results["misses"]),
            sum(results["false_positives"]), sum(results["matched_ungraded"]))
        scores = {
            "grade_weighted_f1": float(summary["grade_weighted_f1"]),
            "grade_balanced_accuracy": float(summary["grade_balanced_accuracy"]),
            "grade_quadratic_weighted_kappa": float(summary["grade_quadratic_weighted_kappa"] or 0.0),
            "grade_mae": float(summary["grade_mae"]),
            "grade_adjacent_accuracy": float(summary["grade_adjacent_accuracy"]),
            "grade_matched_lesions": float(summary["grade_matched_detections"]),
        }
        curves = {
            "grade_confusion_matrix": np.asarray(summary["grade_confusion_matrix"]),
            "grade_per_class_sensitivity": summary["grade_per_class_sensitivity"],
        }
        return scores, curves
