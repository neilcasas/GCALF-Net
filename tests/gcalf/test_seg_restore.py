import numpy as np

from nndet.inference.restore import restore_fmap


def test_restore_fmap_inserts_a_cropped_segmentation_map_into_original_space():
    fmap = np.ones((1, 3, 4, 5), dtype=np.float32)

    restored = restore_fmap(
        fmap=fmap,
        transpose_backward=[0, 1, 2],
        original_spacing=[1.0, 1.0, 1.0],
        spacing_after_resampling=[1.0, 1.0, 1.0],
        original_size_before_cropping=[5, 6, 7],
        size_after_cropping=[3, 4, 5],
        crop_bbox=[[1, 4], [1, 5], [1, 6]],
        interpolation_order=1,
        interpolation_order_z=0,
        do_separate_z=None,
    )

    assert restored.shape == (1, 5, 6, 7)
    assert np.array_equal(restored[:, 1:4, 1:5, 1:6], fmap)
    assert restored.sum() == fmap.sum()
