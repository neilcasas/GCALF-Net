import pytest
import torch

from scripts.measure_transfer_anchor_telemetry import _move_tensors_to_device, summarize_counts


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


def test_move_tensors_to_device_moves_tensors_and_passes_through_metadata():
    # Regression test: the raw augmenter batch mixes tensor/array payloads
    # with plain-string metadata (case IDs, properties dicts full of paths
    # and anatomy-frame fields). A `str` is itself a `Sequence`, so a naive
    # recursive device-mover that doesn't special-case strings recurses
    # forever on any string leaf -- this previously crashed with a real
    # RecursionError partway through a live Vast run.
    batch = {
        "data": torch.zeros(2),
        "keys": ["10005_1000005", "10008_1000008"],
        "properties": [{"case_id": "10005_1000005", "crop_bbox": ((0, 1), (0, 2))}],
        "nested": {"source": "Bosma22b", "count": torch.ones(1)},
    }

    moved = _move_tensors_to_device(batch, "cpu")

    assert moved["data"].device.type == "cpu"
    assert moved["keys"] == ["10005_1000005", "10008_1000008"]
    assert moved["properties"] == [{"case_id": "10005_1000005", "crop_bbox": ((0, 1), (0, 2))}]
    assert moved["nested"]["source"] == "Bosma22b"
    assert moved["nested"]["count"].device.type == "cpu"


def test_move_tensors_to_device_does_not_recurse_on_a_bare_string():
    assert _move_tensors_to_device("10005_1000005", "cpu") == "10005_1000005"
