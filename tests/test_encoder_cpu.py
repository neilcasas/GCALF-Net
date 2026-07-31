import torch

from nndet.arch.blocks.basic import StackedConvBlock2
from nndet.arch.conv import ConvInstanceRelu, Generator
from nndet.arch.encoder.modular import Encoder


def test_encoder_cpu_forward_pass_returns_expected_feature_pyramid():
    torch.manual_seed(0)
    model = Encoder(
        conv=Generator(ConvInstanceRelu, 3),
        conv_kernels=[(3, 3, 3)] * 6,
        strides=[(2, 2, 2)] * 5,
        block_cls=StackedConvBlock2,
        in_channels=3,
        start_channels=4,
        max_channels=128,
    ).cpu().eval()

    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 16, 64, 64))

    expected_shapes = [
        (1, 4, 16, 64, 64),
        (1, 8, 8, 32, 32),
        (1, 16, 4, 16, 16),
        (1, 32, 2, 8, 8),
        (1, 64, 1, 4, 4),
        (1, 128, 1, 2, 2),
    ]
    assert [tuple(output.shape) for output in outputs] == expected_shapes
    assert model.get_channels() == [shape[1] for shape in expected_shapes]
    assert all(output.device.type == "cpu" and torch.isfinite(output).all() for output in outputs)
