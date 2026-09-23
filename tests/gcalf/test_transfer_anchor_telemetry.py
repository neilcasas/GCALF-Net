import pytest

from scripts.measure_transfer_anchor_telemetry import summarize_counts


def test_summarize_counts_reports_the_exact_four_measured_counts_in_order():
    summary = summarize_counts([570, 240, 80, 110], 125)

    assert summary["batches"] == 125
    assert summary["sampled_supervised_anchors"] == 1000
    assert summary["class_counts"] == {"GGG2": 570, "GGG3": 240, "GGG4": 80, "GGG5": 110}
    assert summary["grade_anchor_class_counts"] == [570, 240, 80, 110]


def test_summarize_counts_handles_an_all_zero_batch():
    summary = summarize_counts([0, 0, 0, 0], 1)

    assert summary["sampled_supervised_anchors"] == 0
    assert summary["grade_anchor_class_counts"] == [0, 0, 0, 0]


def test_summarize_counts_rejects_the_wrong_number_of_classes():
    with pytest.raises(ValueError, match="four non-negative"):
        summarize_counts([1, 2, 3], 1)


def test_summarize_counts_rejects_negative_counts():
    with pytest.raises(ValueError, match="four non-negative"):
        summarize_counts([1, -1, 3, 4], 1)
