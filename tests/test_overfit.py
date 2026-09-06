"""GPU/data integration gate for Milestone 2 baseline verification.

Set ``RUN_GCALF_M2=1`` on a CUDA host with the validated M1 task mounted at
``$det_data`` to run this test.  It intentionally skips during the normal unit
test suite because it uses real PI-CAI preprocessed cases.
"""

import copy
import math
import os
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
from hydra import initialize_config_module
from omegaconf import OmegaConf

from nndet.arch.encoder.gcalf.fdsf import FrequencyDomainSeparationAndShunting3D
from nndet.arch.encoder.gcalf.waf import WindowAttentionFusion3D
from nndet.io.datamodule.bg_loader import DataLoader3DOffset
from nndet.io.load import load_pickle
from nndet.io.utils import load_dataset_id
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
from nndet.utils.config import compose


RUN_M2 = os.getenv("RUN_GCALF_M2") == "1"
pytestmark = pytest.mark.skipif(not RUN_M2, reason="set RUN_GCALF_M2=1 to run the M2 GPU/data gate")

TASK = os.getenv("GCALF_M2_TASK", "Task2201_PICAI_csPCa")
PLAN_ID = os.getenv("GCALF_M2_PLAN", "D3V001_3d")
OVERFIT_STEPS = 200
INITIAL_WINDOW = 10
FINAL_LOSS_MAX = 0.1
LOSS_REDUCTION_MAX = 0.1
POSITIVE_PROBABILITY_MIN = 0.9
NEGATIVE_PROBABILITY_MAX = 0.1


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail("M2 requires a CUDA GPU; run this test on an NVIDIA-enabled Docker host")
    return torch.device("cuda")


def _load_runtime() -> Tuple[dict, dict, dict, Path, torch.device]:
    data_root = os.getenv("det_data")
    if not data_root:
        pytest.fail("det_data must point to the M1 task root")

    task_dir = Path(data_root) / TASK
    plan_path = task_dir / "preprocessed" / f"{PLAN_ID}.pkl"
    if not plan_path.is_file():
        pytest.fail(f"M1 plan is missing: {plan_path}")
    if not (task_dir / "dataset.json").is_file():
        pytest.fail(f"M1 dataset metadata is missing: {task_dir / 'dataset.json'}")

    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose(TASK, "config.yaml", overrides=["train=gcalf_baseline"])
    plan = load_pickle(plan_path)
    return (
        plan,
        OmegaConf.to_container(cfg["model_cfg"], resolve=True),
        OmegaConf.to_container(cfg["trainer_cfg"], resolve=True),
        task_dir,
        _require_cuda(),
    )


def _assert_baseline_contract(plan: dict, model: torch.nn.Module) -> None:
    architecture = plan["architecture"]
    assert architecture["in_channels"] == 3
    assert architecture["classifier_classes"] == 1

    encoder = model.encoder
    assert encoder.num_stages == 5
    assert model.grade_head is not None
    assert isinstance(encoder.frequency_module, FrequencyDomainSeparationAndShunting3D)
    assert set(encoder.fusion_modules.keys()) == {str(level) for level in encoder.fusion_levels}
    for fusion in encoder.fusion_modules.values():
        assert isinstance(fusion, WindowAttentionFusion3D)


def _expected_encoder_spatial_shapes(plan: dict) -> list:
    shape = list(plan["patch_size"])
    expected = [tuple(shape)]
    for stride in plan["architecture"]["strides"]:
        if isinstance(stride, int):
            stride = (stride,) * len(shape)
        shape = [int(math.ceil(size / factor)) for size, factor in zip(shape, stride)]
        expected.append(tuple(shape))
    return expected


def _assert_forward_shapes(plan: dict, model: torch.nn.Module, device: torch.device) -> None:
    captured = {}

    def save_encoder(_module, _inputs, output):
        captured["encoder"] = output

    def save_decoder(_module, _inputs, output):
        captured["decoder"] = output

    handles = [model.encoder.register_forward_hook(save_encoder), model.decoder.register_forward_hook(save_decoder)]
    try:
        inputs = torch.randn(1, 3, *plan["patch_size"], device=device)
        model.eval()
        with torch.no_grad():
            pred_detection, anchors, pred_seg = model(inputs)
    finally:
        for handle in handles:
            handle.remove()

    encoder_outputs = captured["encoder"]
    decoder_outputs = captured["decoder"]
    assert len(encoder_outputs) == len(plan["architecture"]["conv_kernels"])
    assert len(decoder_outputs) == len(encoder_outputs)
    assert [feature.shape[1] for feature in encoder_outputs] == model.encoder.get_channels()
    assert [tuple(feature.shape[2:]) for feature in encoder_outputs] == _expected_encoder_spatial_shapes(plan)
    assert all(torch.isfinite(feature).all() for feature in encoder_outputs + decoder_outputs)

    expected_anchors = sum(anchor.shape[0] for anchor in anchors)
    assert pred_detection["box_logits"].shape == (expected_anchors, 1)
    assert pred_detection["grade_logits"].shape == (expected_anchors, 4)
    assert pred_detection["box_deltas"].shape == (expected_anchors, 6)
    assert pred_seg["seg_logits"].shape[:2] == (1, 2)
    assert tuple(pred_seg["seg_logits"].shape[2:]) == tuple(decoder_outputs[0].shape[2:])
    assert all(torch.isfinite(value).all() for value in pred_detection.values())
    assert torch.isfinite(pred_seg["seg_logits"]).all()


def _select_cases(data_dir: Path, task_dir: Path) -> Tuple[Dict, str, str, int]:
    dataset = load_dataset_id(data_dir)
    splits_path = task_dir / "preprocessed" / "splits_final.pkl"
    if not splits_path.is_file():
        pytest.fail(f"M1 official splits are missing: {splits_path}")
    train_cases = sorted(load_pickle(splits_path)[0]["train"])

    positive = None
    negative = None
    positive_instance = None
    for case_id in train_cases:
        if case_id not in dataset:
            pytest.fail(f"Official fold-0 case is absent from preprocessed data: {case_id}")
        instances = load_pickle(dataset[case_id]["boxes_file"])["instances"]
        if not instances and negative is None:
            negative = case_id
        if len(instances) == 1 and positive is None:
            instance_id = int(instances[0])
            properties = load_pickle(dataset[case_id]["properties_file"])
            if properties.get("grade_supervised", {}).get(str(instance_id), False):
                positive = case_id
                positive_instance = instance_id
        if positive is not None and negative is not None:
            break

    if positive is None:
        pytest.fail("Fold 0 has no single-lesion grade-supervised case for the deterministic M2 overfit batch")
    if negative is None:
        pytest.fail("Fold 0 has no zero-instance negative case for the deterministic M2 overfit batch")
    return dataset, positive, negative, positive_instance


def _make_fixed_microbatches(plan: dict, task_dir: Path, device: torch.device) -> Tuple[dict, dict]:
    data_dir = task_dir / "preprocessed" / plan["data_identifier"] / "imagesTr"
    if not data_dir.is_dir():
        pytest.fail(f"M1 preprocessed images are missing: {data_dir}")
    dataset, positive_case, negative_case, positive_instance = _select_cases(data_dir, task_dir)
    subset = OrderedDict((case_id, dataset[case_id]) for case_id in (negative_case, positive_case))
    patch_size = tuple(int(value) for value in plan["patch_size"])
    loader = DataLoader3DOffset(
        data=subset,
        batch_size=2,
        patch_size_generator=patch_size,
        patch_size_final=patch_size,
        oversample_foreground_percent=0.5,
        memmap_mode="r",
        num_batches_per_epoch=1,
    )
    loader.select = lambda: ([negative_case, positive_case], [-1, positive_instance])
    np.random.seed(0)
    batch = loader.generate_train_batch()

    module_batch = {
        "data": torch.as_tensor(batch["data"], dtype=torch.float32, device=device),
        "target": torch.as_tensor(batch["seg"], dtype=torch.float32, device=device),
        "instance_mapping": batch["instance_mapping"],
        "properties": batch["properties"],
    }
    return module_batch, {"positive_case": positive_case, "negative_case": negative_case}


def _prepare_microbatch(module, module_batch: dict, index: int) -> dict:
    batch = {
        "data": module_batch["data"][index:index + 1].clone(),
        "target": module_batch["target"][index:index + 1].clone(),
        "instance_mapping": [module_batch["instance_mapping"][index]],
        "properties": [module_batch["properties"][index]],
    }
    with torch.no_grad():
        return module.pre_trafo(**batch)


def _loss_target(batch: dict) -> dict:
    return {
        "target_boxes": batch["boxes"],
        "target_classes": batch["classes"],
        "target_seg": batch["target"][:, 0],
        "target_grades": batch["grades"],
        "target_grade_supervised": batch["grade_supervised"],
        "grade_class_weights": torch.ones(4, device=batch["data"].device),
    }


def _gradient_norm(module: torch.nn.Module) -> float:
    norms = [parameter.grad.detach().norm().item() for parameter in module.parameters() if parameter.grad is not None]
    return float(sum(norms))


def _assert_memorized(model: torch.nn.Module, positive: dict, negative: dict) -> None:
    model.eval()
    with torch.no_grad():
        positive_prediction, positive_anchors, _ = model(positive["data"])
        labels, _ = model.assign_targets_to_anchors(
            positive_anchors,
            positive["boxes"],
            positive["classes"],
        )
        positive_indices = torch.where(torch.cat(labels) > 0)[0]
        assert positive_indices.numel() > 0
        target_class = int(positive["classes"][0][0].item())
        probabilities = torch.sigmoid(positive_prediction["box_logits"][positive_indices])
        best_index = probabilities[:, target_class].argmax()
        best_probabilities = probabilities[best_index]
        assert int(best_probabilities.argmax().item()) == target_class
        assert float(best_probabilities[target_class].item()) >= POSITIVE_PROBABILITY_MIN

        grade_probabilities = torch.softmax(positive_prediction["grade_logits"][positive_indices], dim=1)
        target_grade = int(positive["grades"][0][0].item()) - 2
        assert float(grade_probabilities[:, target_grade].max().item()) >= POSITIVE_PROBABILITY_MIN

        negative_prediction, _, _ = model(negative["data"])
        maximum_negative_probability = torch.sigmoid(negative_prediction["box_logits"]).max().item()
        assert maximum_negative_probability <= NEGATIVE_PROBABILITY_MAX


def test_m2_baseline_forward_and_overfit_two_cases():
    torch.manual_seed(0)
    np.random.seed(0)
    plan, model_cfg, trainer_cfg, task_dir, device = _load_runtime()

    forward_model = RetinaUNetV001.from_config_plan(
        model_cfg=copy.deepcopy(model_cfg),
        plan_arch=copy.deepcopy(plan["architecture"]),
        plan_anchors=copy.deepcopy(plan["anchors"]),
    ).to(device)
    _assert_baseline_contract(plan, forward_model)
    _assert_forward_shapes(plan, forward_model, device)
    del forward_model
    torch.cuda.empty_cache()

    module = RetinaUNetV001(
        model_cfg=copy.deepcopy(model_cfg),
        trainer_cfg=copy.deepcopy(trainer_cfg),
        plan=copy.deepcopy(plan),
    ).to(device)
    model = module.model
    _assert_baseline_contract(plan, model)
    module_batch, _case_selection = _make_fixed_microbatches(plan, task_dir, device)
    negative = _prepare_microbatch(module, module_batch, 0)
    positive = _prepare_microbatch(module, module_batch, 1)
    assert positive["boxes"][0].shape == (1, 6)
    assert negative["boxes"][0].numel() == 0
    assert positive["grade_supervised"][0].tolist() == [True]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    benign_losses, _ = model.train_step(
        images=negative["data"], targets=_loss_target(negative), evaluation=False, batch_num=0)
    sum(benign_losses.values()).backward()
    assert _gradient_norm(model.grade_head) == 0.0

    scaler = torch.cuda.amp.GradScaler()
    loss_history = []
    loss_components = {}
    started = perf_counter()
    model.train()
    for step in range(OVERFIT_STEPS):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for batch in (positive, negative):
            with torch.cuda.amp.autocast():
                losses, _ = model.train_step(
                    images=batch["data"],
                    targets=_loss_target(batch),
                    evaluation=False,
                    batch_num=step,
                )
                loss = sum(losses.values()) / 2
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            total_loss = total_loss + loss.detach()
            for name, value in losses.items():
                assert torch.isfinite(value)
                loss_components[name] = float(value.detach().item())

        if step == 0:
            scaler.unscale_(optimizer)
            for component in (model.encoder, model.head.classifier, model.head.regressor, model.segmenter, model.grade_head):
                assert _gradient_norm(component) > 0
        scaler.step(optimizer)
        scaler.update()
        loss_history.append(float(total_loss.item()))

    duration = perf_counter() - started
    initial_loss = float(np.mean(loss_history[:INITIAL_WINDOW]))
    final_loss = float(np.mean(loss_history[-INITIAL_WINDOW:]))
    print(
        "M2 overfit summary: "
        f"initial_loss={initial_loss:.6f} final_loss={final_loss:.6f} "
        f"duration_seconds={duration:.1f} peak_memory_bytes={torch.cuda.max_memory_allocated(device)} "
        f"loss_components={loss_components}"
    )
    assert final_loss <= FINAL_LOSS_MAX
    assert final_loss <= initial_loss * LOSS_REDUCTION_MAX
    _assert_memorized(model, positive, negative)
