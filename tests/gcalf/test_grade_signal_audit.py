import numpy as np
import pytest

from scripts.audit_grade_signal import _feature_vector, _source_fov_masks, _unit_class


def test_adc_unit_classifier_distinguishes_expected_and_alternate_scales():
    assert _unit_class(1000.0) == "micro_mm2_per_s"
    assert _unit_class(1e-3) == "mm2_per_s"
    assert _unit_class(42.0) == "unrecognized"


def test_raw_fov_comes_from_source_geometry_even_when_in_field_values_are_zero(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    case_id = "12345_0000001"
    source_dir = tmp_path / "patient"
    source_dir.mkdir()
    source = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.float32))
    for suffix in ("t2w", "adc", "hbv"):
        sitk.WriteImage(source, str(source_dir / f"{case_id}_{suffix}.mha"))
    reference = sitk.GetImageFromArray(np.zeros((3, 3, 3), dtype=np.float32))

    fovs = _source_fov_masks({"12345": source_dir}, case_id, reference)

    assert len(fovs) == 3
    assert all(mask.sum() == 8 for mask in fovs)
    assert all(mask[0, 0, 0] for mask in fovs)
    assert all(not mask[-1, -1, -1] for mask in fovs)


def test_oracle_roi_features_exclude_values_outside_each_modality_fov():
    data = np.asarray([
        [[[1.0, 3.0, 7.0, 9.0]]],
        [[[100.0, 200.0, 300.0, 400.0]]],
        [[[10.0, 20.0, 30.0, 40.0]]],
    ])
    lesion = np.asarray([[[True, True, False, False]]])
    gland = np.asarray([[[False, False, True, True]]])
    fovs = [
        np.ones((1, 1, 4), dtype=bool),
        np.asarray([[[True, False, True, True]]]),
        np.asarray([[[True, False, True, True]]]),
    ]

    features = _feature_vector(data, lesion, gland, fovs, voxel_volume=3.0)

    np.testing.assert_allclose(features[:2], [100.0, 100.0])
    assert features[2] == pytest.approx(10.0 / 35.0)
    assert features[3] == pytest.approx(0.25)
    assert features[4] == 6.0
