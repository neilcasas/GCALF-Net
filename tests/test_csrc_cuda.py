import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the nndet extension smoke test")
def test_cuda_nms_extension_runs_on_3d_boxes():
    from nndet import _C

    boxes = torch.tensor(
        [[0, 0, 4, 4, 0, 4], [1, 1, 5, 5, 1, 5], [10, 10, 14, 14, 10, 14]],
        dtype=torch.float32,
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device="cuda")

    keep = _C.nms(boxes, scores, 0.5)
    torch.cuda.synchronize()

    torch.testing.assert_close(keep.cpu(), torch.tensor([0, 2]))
