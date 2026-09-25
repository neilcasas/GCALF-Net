import numpy as np
import pytest

from nndet.io.augmentation.bg_aug import ChannelSelectiveIntensityTransform


class _AddIntensity:
    def __call__(self, **data_dict):
        data_dict["data"] += np.float32(1.25)
        return data_dict


def test_channel_selective_intensity_transform_preserves_adc_bit_identically():
    data = np.random.RandomState(15).normal(size=(2, 3, 4, 5, 6)).astype(np.float32)
    before = data.copy()
    result = ChannelSelectiveIntensityTransform(_AddIntensity(), channels=(0, 2))(data=data)

    assert np.array_equal(result["data"][:, 1], before[:, 1])
    assert np.array_equal(result["data"][:, 0], before[:, 0] + np.float32(1.25))
    assert np.array_equal(result["data"][:, 2], before[:, 2] + np.float32(1.25))


def test_channel_selective_intensity_transform_rejects_unknown_channel():
    transform = ChannelSelectiveIntensityTransform(_AddIntensity(), channels=(3,))
    with pytest.raises(ValueError, match="exceed input channel count"):
        transform(data=np.zeros((1, 3, 2, 2, 2), dtype=np.float32))
