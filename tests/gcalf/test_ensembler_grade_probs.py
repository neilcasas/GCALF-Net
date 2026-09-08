import torch

from nndet.inference.ensembler.detection import BoxEnsembler, BoxEnsemblerSelective

def _populate(ensembler, grade_probs=None):
    # Two well-separated, positive-volume boxes (nnDetection's x1,y1,x2,y2,z1,z2
    # order) inside the (4, 4, 4) test image, scored above the default 0.0
    # threshold and non-overlapping (IoU 0 < the default 0.1 NMS threshold),
    # so BoxEnsemblerSelective's postprocess_image step (NMS + score-thresh +
    # remove_small_boxes) passes both through unchanged. Built fresh each
    # call: postprocessing mutates boxes/scores in place, so reusing one
    # tensor across the multiple process_model() calls below would degrade
    # later calls.
    entry = {
        "boxes": [torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 1.0], [2.0, 2.0, 3.0, 3.0, 2.0, 3.0]])],
        "scores": [torch.tensor([0.9, 0.8])],
        "labels": [torch.zeros(2, dtype=torch.long)],
        "weights": [torch.ones(2)],
    }
    if grade_probs is not None:
        entry["grade_probs"] = grade_probs
    ensembler.model_results["model_0"] = entry


def test_process_model_handles_every_grade_probs_state_without_crashing():
    """`save_state()`'s top-k truncation can replace the stored grade_probs
    list with a single multi-row Tensor; `process_model` must handle that
    form exactly like the pre-truncation List[Tensor] form, and must still
    skip cleanly when nothing was ever accumulated."""
    for ensembler_cls in (BoxEnsembler, BoxEnsemblerSelective):
        ensembler = ensembler_cls(
            properties={"shape": (4, 4, 4)}, parameters=ensembler_cls.get_default_parameters()
        )

        # No grade-supervised model ever contributed: key absent entirely.
        _populate(ensembler)
        *_, grade_probs = ensembler.process_model("model_0")
        assert grade_probs is None

        # Key present but every contributing patch had zero rows.
        _populate(ensembler, grade_probs=[])
        *_, grade_probs = ensembler.process_model("model_0")
        assert grade_probs is None

        # Normal pre-truncation form: a list of per-instance tensors.
        _populate(ensembler, grade_probs=[torch.rand(1, 4), torch.rand(1, 4)])
        *_, grade_probs = ensembler.process_model("model_0")
        assert grade_probs.shape == (2, 4)

        # Post-truncation form: save_state() already replaced the list with
        # a single multi-row Tensor. This is the exact shape that raised
        # `RuntimeError: Boolean value of Tensor with more than one value
        # is ambiguous` before the fix.
        _populate(ensembler, grade_probs=torch.rand(2, 4))
        *_, grade_probs = ensembler.process_model("model_0")
        assert grade_probs.shape == (2, 4)
