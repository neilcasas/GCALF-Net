import torch

from nndet.arch.heads.comb import DetectionHead


class _Coder:
    def decode(self, box_deltas, anchors):
        return box_deltas + 1


class _Classifier:
    def box_logits_to_probs(self, box_logits):
        return box_logits.sigmoid()


class _HeadHarness:
    coder = _Coder()
    classifier = _Classifier()


def test_detection_postprocessing_preserves_anchor_grade_logits():
    prediction = {
        "box_deltas": torch.tensor([[1.0, 2.0]]),
        "box_logits": torch.tensor([[0.0]]),
        "grade_logits": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    }

    result = DetectionHead.postprocess_for_inference(
        _HeadHarness(), prediction, anchors=[torch.zeros((1, 2))]
    )

    assert torch.equal(result["grade_logits"], prediction["grade_logits"])
    assert result["pred_probs"].shape == (1, 1)
