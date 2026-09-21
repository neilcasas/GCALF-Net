"""Regression tests for SegmentationEnsembler state round-tripping.

`save_state` persisted `model_results` but not `overlap`, while
`get_case_result` divides by `overlap`. A reloaded ensembler therefore
divided finite predictions by an all-zero overlap map, producing inf in
every channel; `argmax` over equal infs then returned background for every
voxel. This silently turned a working segmentation head into an all-zero
prediction in the two-stage sweep -> export path, and scored lesion-matched
Dice as 0.0 on real data.
"""
import torch

import pytest

from nndet.inference.ensembler.segmentation import SegmentationEnsembler


def _properties(shape=(4, 6, 6)):
    return {
        "shape": shape,
        "transpose_backward": [0, 1, 2],
        "original_spacing": (1.0, 1.0, 1.0),
        "spacing_after_resampling": (1.0, 1.0, 1.0),
        "crop_bbox": [[0, shape[0]], [0, shape[1]], [0, shape[2]]],
        "size_after_cropping": shape,
        "original_size_before_cropping": shape,
        "itk_origin": (0.0, 0.0, 0.0),
        "itk_spacing": (1.0, 1.0, 1.0),
        "itk_direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }


def _ensembler_with_foreground(shape=(4, 6, 6)):
    """An ensembler holding a foreground-predicting result and a live overlap."""
    ensembler = SegmentationEnsembler(properties=_properties(shape),
                                      parameters={"use_gaussian": True, "argmax": True})
    ensembler.add_model()
    # channel 0 background, channel 1 foreground; foreground wins in a sub-block.
    results = torch.zeros((2, *shape))
    results[0] = 1.0
    results[1, :, :3, :3] = 4.0
    ensembler.model_results = results
    ensembler.overlap = torch.full(shape, 2.0)
    return ensembler


def test_save_state_persists_the_overlap_map(tmp_path):
    ensembler = _ensembler_with_foreground()
    ensembler.save_state(tmp_path, "case0")

    state = torch.load(str(tmp_path / "case0_seg.pt"), weights_only=False)
    assert "overlap" in state, "overlap must round-trip or get_case_result divides by zero"
    assert torch.equal(torch.as_tensor(state["overlap"]), ensembler.overlap)


def test_reloaded_ensembler_reproduces_the_original_segmentation(tmp_path):
    """The actual bug: identical input must give identical output after a round trip."""
    ensembler = _ensembler_with_foreground()
    expected = ensembler.get_case_result(restore=False)["pred_seg"]
    assert int((expected > 0).sum()) > 0, "fixture must predict some foreground"

    ensembler.save_state(tmp_path, "case0")
    reloaded = SegmentationEnsembler.from_checkpoint(tmp_path, "case0")
    actual = reloaded.get_case_result(restore=False)["pred_seg"]

    assert torch.equal(actual, expected)
    assert int((actual > 0).sum()) == int((expected > 0).sum())


def test_empty_overlap_raises_instead_of_silently_returning_background():
    """Guards states written before `overlap` was persisted: they must fail
    loudly rather than score as an all-background prediction."""
    ensembler = _ensembler_with_foreground()
    ensembler.overlap = torch.zeros(ensembler.properties["shape"])

    with pytest.raises(RuntimeError, match="overlap map is empty"):
        ensembler.get_case_result(restore=False)
