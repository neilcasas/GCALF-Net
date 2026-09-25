import numpy as np
import pytest
import SimpleITK as sitk

from gcalf_data import preprocessing


def _image(array, spacing=(1.0, 1.0, 1.0)):
    """array is (z, y, x); spacing is (x, y, z), matching sitk convention."""
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    return image


def _mask_image(array):
    return _image(array.astype(np.uint8))


def test_whole_gland_centroid_index_matches_known_blob_center():
    array = np.zeros((8, 10, 12), dtype=np.uint8)
    array[3:5, 4:6, 5:7] = 1  # centroid at z=3.5, y=4.5, x=5.5
    centroid = preprocessing.whole_gland_centroid_index(_image(array))
    assert centroid == pytest.approx((3.5, 4.5, 5.5))


def test_whole_gland_centroid_index_rejects_empty_mask():
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        preprocessing.whole_gland_centroid_index(_image(array))


def test_resolve_crop_center_uses_gland_centroid():
    gland = np.zeros((4, 40, 40), dtype=np.uint8)
    gland[:, 15:25, 15:25] = 1  # centroid (y, x) = (19.5, 19.5)

    (center_y, center_x), strategy = preprocessing.resolve_crop_center(_mask_image(gland))
    assert strategy == "gland"
    assert (center_y, center_x) == pytest.approx((19.5, 19.5))


def test_resolve_crop_center_ignores_lesion_and_falls_back_to_t2w_centre_when_gland_empty():
    """Mirrors ADR 0002 D6 item 4: the lesion mask must never select or adjust
    the crop, even when a gland-centered window would clip it -- an empty
    gland mask instead falls back to the T2W geometric centre."""
    empty_gland = np.zeros((4, 40, 60), dtype=np.uint8)

    (center_y, center_x), strategy = preprocessing.resolve_crop_center(_mask_image(empty_gland))
    assert strategy == "t2w_geometric_centre"
    assert (center_y, center_x) == pytest.approx((19.5, 29.5))


def test_fov_mm_to_inplane_size_converts_physical_fov_to_voxels_per_axis():
    size_y, size_x = preprocessing.fov_mm_to_inplane_size(spacing_yx=(0.5, 0.25), fov_mm=10.0)
    assert (size_y, size_x) == (20, 40)


def test_center_crop_or_pad_inplane_crops_around_center_without_resampling():
    array = np.arange(4 * 20 * 30).reshape(4, 20, 30).astype(np.float32)
    image = _image(array, spacing=(0.5, 0.5, 3.0))
    result = preprocessing.center_crop_or_pad_inplane(image, center_yx=(10.0, 15.0), size_yx=(8, 6))

    assert result.GetSize() == (6, 8, 4)
    assert result.GetSpacing() == (0.5, 0.5, 3.0)
    result_array = sitk.GetArrayFromImage(result)
    np.testing.assert_array_equal(result_array, array[:, 6:14, 12:18])


def test_center_crop_or_pad_inplane_pads_when_window_exceeds_bounds():
    array = np.ones((2, 4, 4), dtype=np.float32)
    image = _image(array)
    result = preprocessing.center_crop_or_pad_inplane(image, center_yx=(2.0, 2.0), size_yx=(8, 8))

    assert result.GetSize() == (8, 8, 2)
    result_array = sitk.GetArrayFromImage(result)
    # the whole 4x4 array is kept and centered in the padded 8x8 frame
    np.testing.assert_array_equal(result_array[:, 2:6, 2:6], array)
    assert np.all(result_array[:, :2, :] == 0)
    assert np.all(result_array[:, :, :2] == 0)
    assert np.all(result_array[:, 6:, :] == 0)
    assert np.all(result_array[:, :, 6:] == 0)


def test_pad_or_crop_depth_pads_symmetrically():
    array = np.ones((4, 3, 3), dtype=np.float32)
    image = _image(array)
    result = preprocessing.pad_or_crop_depth(image, target_slices=8)
    result_array = sitk.GetArrayFromImage(result)
    assert result_array.shape == (8, 3, 3)
    np.testing.assert_array_equal(result_array[2:6], array)
    assert np.all(result_array[:2] == 0)
    assert np.all(result_array[6:] == 0)


def test_pad_or_crop_depth_crops_symmetrically():
    array = np.arange(8 * 3 * 3).reshape(8, 3, 3).astype(np.float32)
    image = _image(array)
    result = preprocessing.pad_or_crop_depth(image, target_slices=4)
    result_array = sitk.GetArrayFromImage(result)
    np.testing.assert_array_equal(result_array, array[2:6])


def test_build_target_reference_preserves_inplane_geometry_and_resamples_slice_axis():
    t2w = _image(np.zeros((12, 20, 24), dtype=np.float32), spacing=(0.5, 0.6, 1.0))
    reference = preprocessing.build_target_reference(t2w, target_slice_spacing=3.0)

    assert reference.GetSpacing() == pytest.approx((0.5, 0.6, 3.0))
    assert reference.GetSize()[:2] == (24, 20)  # (x, y) unchanged
    assert reference.GetSize()[2] == 4  # 12 slices at 1mm -> 4 slices at 3mm
    assert reference.GetOrigin() == t2w.GetOrigin()
    assert reference.GetDirection() == t2w.GetDirection()


def test_resample_to_reference_matches_reference_geometry():
    reference = _image(np.zeros((4, 8, 8), dtype=np.float32), spacing=(0.5, 0.5, 3.0))
    moving = _image(np.ones((6, 4, 4), dtype=np.float32), spacing=(1.0, 1.0, 2.0))
    result = preprocessing.resample_to_reference(moving, reference, is_label=False)
    assert result.GetSize() == reference.GetSize()
    assert result.GetSpacing() == reference.GetSpacing()
    assert result.GetOrigin() == reference.GetOrigin()


def test_resample_to_reference_does_not_overshoot_like_bspline_would():
    """Regression test for the B-spline interpolator bug: cubic/B-spline
    overshoots at sharp edges and can emit out-of-range values on the
    quantitative ADC/HBV maps (ARCHITECTURE.md Sec 3 item 2)."""
    array = np.zeros((4, 20, 20), dtype=np.float32)
    array[:, 10:, :] = 100.0  # a sharp step edge
    moving = _image(array, spacing=(1.0, 1.0, 1.0))
    reference = _image(np.zeros((4, 60, 60), dtype=np.float32), spacing=(1.0 / 3, 1.0 / 3, 1.0))  # upsample 3x

    result = preprocessing.resample_to_reference(moving, reference, is_label=False)
    result_array = sitk.GetArrayFromImage(result)
    assert result_array.min() >= -1e-3
    assert result_array.max() <= 100.0 + 1e-3


def test_resample_to_reference_nearest_neighbour_preserves_label_domain():
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    array[:2] = 2
    array[2:] = 3
    moving = _mask_image(array)
    reference = _image(np.zeros((8, 8, 8), dtype=np.float32), spacing=(0.5, 0.5, 0.5))
    result = preprocessing.resample_to_reference(moving, reference, is_label=True)
    result_array = sitk.GetArrayFromImage(result)
    assert set(np.unique(result_array).tolist()) <= {0, 2, 3}


def test_n4_bias_correct_preserves_geometry_and_produces_finite_output():
    array = np.random.RandomState(0).normal(100, 10, (8, 32, 32)).astype(np.float32)
    image = _image(array, spacing=(0.5, 0.5, 3.0))
    result = preprocessing.n4_bias_correct(image, shrink_factor=2)
    assert result.GetSize() == image.GetSize()
    assert result.GetSpacing() == image.GetSpacing()
    result_array = sitk.GetArrayFromImage(result)
    assert np.all(np.isfinite(result_array))


def test_preprocess_case_produces_matching_geometry_and_rejects_bad_mask_values():
    t2w = _image(np.random.RandomState(0).normal(100, 10, (16, 20, 20)).astype(np.float32), spacing=(1.0, 1.0, 3.0))
    adc = _image(np.random.RandomState(1).normal(50, 5, (8, 10, 10)).astype(np.float32), spacing=(2.0, 2.0, 4.0))
    hbv = _image(np.random.RandomState(2).normal(30, 3, (8, 10, 10)).astype(np.float32), spacing=(2.0, 2.0, 4.0))
    whole_gland = np.zeros((16, 20, 20), dtype=np.uint8)
    whole_gland[6:10, 8:12, 8:12] = 1
    whole_gland_image = _image(whole_gland, spacing=(1.0, 1.0, 3.0))
    lesion_mask = np.zeros((16, 20, 20), dtype=np.uint8)
    lesion_mask[7:9, 9:11, 9:11] = 3
    lesion_mask_image = _image(lesion_mask, spacing=(1.0, 1.0, 3.0))

    out_t2w, out_adc, out_hbv, out_mask, crop_strategy = preprocessing.preprocess_case(
        t2w, adc, hbv, lesion_mask_image, whole_gland_image,
        target_fov_mm=8.0, target_slice_spacing=6.0, target_num_slices=4,
    )

    for image in (out_t2w, out_adc, out_hbv, out_mask):
        assert image.GetSize() == (8, 8, 4)
    assert out_t2w.GetSpacing() == out_adc.GetSpacing() == out_hbv.GetSpacing() == out_mask.GetSpacing()
    mask_values = set(np.unique(sitk.GetArrayFromImage(out_mask)).tolist())
    assert mask_values <= {0, 1, 2, 3, 4, 5}
    assert crop_strategy == "gland"


def test_task2202_fixed_adc_mapping_and_fov_exclusion_from_gland_statistics():
    image = lambda array: _image(np.asarray(array, dtype=np.float32))
    t2w = image([[[10, 20, 1000], [30, 40, 0]]])
    adc = image([[[1000, 1500, 2000], [3000, 4000, 5000]]])
    hbv = image([[[10, 20, 10000], [30, 40, 9000]]])
    gland = np.array([[[1, 1, 1], [0, 0, 0]]], dtype=bool)
    fovs = [
        _mask_image(np.array([[[1, 1, 0], [1, 1, 0]]], dtype=np.uint8)),
        _mask_image(np.array([[[1, 1, 0], [1, 1, 0]]], dtype=np.uint8)),
        _mask_image(np.array([[[1, 1, 0], [1, 1, 0]]], dtype=np.uint8)),
    ]

    (out_t2w, out_adc, out_hbv), coverage = preprocessing._normalize_task2202(
        t2w, adc, hbv, gland, fovs, preprocessing.load_task2202_config())

    t2w_values = sitk.GetArrayFromImage(out_t2w)
    np.testing.assert_allclose(t2w_values[0, 0, :2], [-1.0, 1.0])
    assert t2w_values[0, 0, 2] == 0.0
    adc_values = sitk.GetArrayFromImage(out_adc)
    np.testing.assert_allclose(adc_values[0, 0, :2], [0.0, 1.0])
    assert adc_values[0, 0, 2] == 0.0
    hbv_values = sitk.GetArrayFromImage(out_hbv)
    assert hbv_values[0, 0, 2] == 0.0
    np.testing.assert_allclose(hbv_values[0, 0, :2], [-1.0 / 6.0, 1.0 / 6.0], atol=1e-7)
    assert coverage == {"t2w": pytest.approx(4.0 / 6.0), "adc": pytest.approx(4.0 / 6.0),
                        "hbv": pytest.approx(4.0 / 6.0)}


def test_task2202_adc_mapping_is_fixed_and_clips_high_values():
    images = [_image(np.array([[[1, 2, 3]]], dtype=np.float32)) for _ in range(3)]
    adc = _image(np.array([[[0, 1000, 3000]]], dtype=np.float32))
    changed_case_adc = _image(np.array([[[500, 1000, 4000]]], dtype=np.float32))
    gland = np.ones((1, 1, 3), dtype=bool)
    fov = _mask_image(np.ones((1, 1, 3), dtype=np.uint8))
    config = preprocessing.load_task2202_config()

    first = preprocessing._normalize_task2202(images[0], adc, images[2], gland,
                                               [fov, fov, fov], config)[0][1]
    second = preprocessing._normalize_task2202(images[0], changed_case_adc, images[2], gland,
                                                [fov, fov, fov], config)[0][1]

    first_values = sitk.GetArrayFromImage(first)
    second_values = sitk.GetArrayFromImage(second)
    np.testing.assert_allclose(first_values.ravel(), [-2.0, 0.0, 4.0])
    np.testing.assert_allclose(second_values.ravel(), [-1.0, 0.0, 4.0])
    assert first_values[0, 0, 1] == second_values[0, 0, 1]


def test_fov_mask_uses_image_geometry_even_when_voxels_are_zero():
    source = _image(np.zeros((2, 3, 4), dtype=np.float32))
    reference = preprocessing.build_target_reference(source, target_slice_spacing=1.0)

    coverage = sitk.GetArrayFromImage(preprocessing.resample_fov_to_reference(source, reference))

    assert coverage.shape == (2, 3, 4)
    assert np.all(coverage == 1)


def test_preprocess_case_rejects_unsupported_mask_values():
    t2w = _image(np.random.RandomState(0).normal(100, 10, (8, 10, 10)).astype(np.float32), spacing=(0.5, 0.5, 3.0))
    adc = _image(np.random.RandomState(1).normal(50, 5, (8, 10, 10)).astype(np.float32), spacing=(0.5, 0.5, 3.0))
    hbv = _image(np.random.RandomState(2).normal(30, 3, (8, 10, 10)).astype(np.float32), spacing=(0.5, 0.5, 3.0))
    whole_gland = np.zeros((8, 10, 10), dtype=np.uint8)
    whole_gland[3:5, 4:6, 4:6] = 1
    whole_gland_image = _image(whole_gland, spacing=(0.5, 0.5, 3.0))
    bad_mask = np.zeros((8, 10, 10), dtype=np.uint8)
    bad_mask[3:5, 4:6, 4:6] = 9  # outside the supported {0,1,2,3,4,5} domain
    bad_mask_image = _image(bad_mask, spacing=(0.5, 0.5, 3.0))

    with pytest.raises(ValueError, match="unsupported values"):
        preprocessing.preprocess_case(
            t2w, adc, hbv, bad_mask_image, whole_gland_image,
            target_fov_mm=4.0, target_slice_spacing=6.0, target_num_slices=4,
        )


def test_preprocess_case_ignores_lesion_mask_when_choosing_crop_center():
    """Mirrors ADR 0002 D6 item 4 and the rejected "Lesion-guided
    union/lesion-only crop" alternative: a lesion far from the gland must
    never re-center or widen the crop -- it is retained or clipped exactly
    like any other post-crop consequence, never used to choose the window."""
    t2w = _image(np.random.RandomState(0).normal(100, 10, (8, 40, 40)).astype(np.float32), spacing=(1.0, 1.0, 3.0))
    adc = _image(np.random.RandomState(1).normal(50, 5, (8, 40, 40)).astype(np.float32), spacing=(1.0, 1.0, 3.0))
    hbv = _image(np.random.RandomState(2).normal(30, 3, (8, 40, 40)).astype(np.float32), spacing=(1.0, 1.0, 3.0))
    whole_gland = np.zeros((8, 40, 40), dtype=np.uint8)
    whole_gland[:, 18:22, 18:22] = 1  # centroid (y, x) = (19.5, 19.5)
    whole_gland_image = _image(whole_gland, spacing=(1.0, 1.0, 3.0))
    lesion_mask = np.zeros((8, 40, 40), dtype=np.uint8)
    lesion_mask[:, 35:38, 35:38] = 2  # far outside an 8x8 gland-centered window
    lesion_mask_image = _image(lesion_mask, spacing=(1.0, 1.0, 3.0))

    _, _, _, out_mask, crop_strategy = preprocessing.preprocess_case(
        t2w, adc, hbv, lesion_mask_image, whole_gland_image,
        target_fov_mm=8.0, target_slice_spacing=6.0, target_num_slices=4,
    )

    assert crop_strategy == "gland"
    # the crop was centered on the gland alone, so the distant lesion is clipped away entirely
    assert not np.any(sitk.GetArrayFromImage(out_mask) == 2)


def test_preprocess_case_with_retention_records_complete_target_independent_crop_loss():
    t2w = _image(np.ones((8, 40, 40), dtype=np.float32), spacing=(1.0, 1.0, 3.0))
    adc = _image(np.ones((8, 40, 40), dtype=np.float32), spacing=(1.0, 1.0, 3.0))
    hbv = _image(np.ones((8, 40, 40), dtype=np.float32), spacing=(1.0, 1.0, 3.0))
    gland = np.zeros((8, 40, 40), dtype=np.uint8)
    gland[:, 18:22, 18:22] = 1
    lesion = np.zeros((8, 40, 40), dtype=np.uint8)
    lesion[:, 35:38, 35:38] = 2

    *_, crop_strategy, retention = preprocessing.preprocess_case_with_retention(
        t2w,
        adc,
        hbv,
        _mask_image(lesion),
        _mask_image(gland),
        target_fov_mm=8.0,
        target_slice_spacing=6.0,
        target_num_slices=4,
    )

    assert crop_strategy == "gland"
    assert retention["source"]["voxels"] > 0
    assert retention["resampled"]["voxels"] > 0
    assert retention["inplane"]["voxels"] == 0
    assert retention["final"]["voxels"] == 0
