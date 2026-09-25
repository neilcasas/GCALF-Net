#!/usr/bin/env python3
"""Measure real post-paste grade-anchor counts for the ADR-0005 Phase 1 gate.

ADR 0005 requires a fresh, no-more-than-250-batch Phase 1 measurement of the
final (transfer-enabled) training configuration after the transfer-path
fixes: persist the paste-rejection counters and a ``grade_anchor_counts.json``
record, then replace the null ``trainer_cfg.grade_anchor_class_counts`` in
the selected Task2201 or Task2202 final config with the four measured counts.
This script performs exactly that measurement and nothing else -- it defines
no pass/fail gate of its own, because ADR 0005 does not specify one for this
step.

Each batch passes through the production training dataloader and positive-
anchor assignment/sampling path under ``torch.no_grad()``. No checkpoint,
optimizer, backward pass, or Lightning Trainer is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

MAX_BATCHES = 250
GRADE_KEYS = tuple(f"GGG{grade}" for grade in range(2, 6))
CASE_ID_PATTERN = re.compile(r"\b\d{5}_\d{7}\b")
TASK2202 = "Task2202_PICAI_csPCa"


def resolve_train_config(task: str, requested: Optional[str] = None) -> str:
    required = "gcalf_task2202" if task == TASK2202 else "gcalf_final"
    if requested is not None and requested != required:
        raise ValueError(f"{task} telemetry requires train={required}, got train={requested}")
    return required


def summarize_counts(counts: Iterable[int], batches: int) -> dict:
    """Match the schema ``RetinaUNetModule.on_train_epoch_end`` already writes
    to ``grade_anchor_counts.json`` during real training, so this measurement
    produces the same artifact shape the rest of the codebase expects."""
    values = [int(value) for value in counts]
    if len(values) != 4 or any(value < 0 for value in values):
        raise ValueError("Expected four non-negative GGG2-5 anchor counts")
    total = sum(values)
    return {
        "batches": int(batches),
        "sampled_supervised_anchors": total,
        "class_counts": dict(zip(GRADE_KEYS, values)),
        "grade_anchor_class_counts": values,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_patient_disjoint(datamodule) -> None:
    train_patients = {case.split("_", 1)[0] for case in datamodule.dataset_tr}
    val_patients = {case.split("_", 1)[0] for case in datamodule.dataset_val}
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(f"Fold split has {len(overlap)} patient(s) in both train and validation")


def _loader_core(augmenter):
    current = augmenter
    for _ in range(4):
        if hasattr(current, "paste_counters"):
            return current
        current = getattr(current, "data_loader", None)
        if current is None:
            break
    raise RuntimeError("Could not inspect the production dataloader behind the augmenter")


def _move_tensors_to_device(inp, device):
    """Move only tensor leaves to `device`, passing everything else through
    unchanged. The raw augmenter batch carries per-case metadata alongside
    its tensor payload (``properties``, ``keys``: plain strings, paths, and
    nested dicts including the anatomy-frame fields), which
    ``nndet.utils.tensor.to_device``'s generic recursive walk cannot handle
    -- a Python `str` is itself a `Sequence`, so recursing into one never
    terminates. Real Lightning training never hits this: its own
    `transfer_batch_to_device` already skips non-tensor leaves before
    `training_step` is called, and this measurement bypasses the Trainer
    (deliberately, to avoid an optimizer/backward pass) but must still
    replicate that same safe behavior rather than a generic tensor mover."""
    import torch

    if isinstance(inp, torch.Tensor):
        return inp.to(device=device)
    if isinstance(inp, (str, bytes)):
        return inp
    if isinstance(inp, Mapping):
        return type(inp)({key: _move_tensors_to_device(item, device) for key, item in inp.items()})
    if isinstance(inp, Sequence):
        return type(inp)(_move_tensors_to_device(item, device) for item in inp)
    return inp


def _run_measurement(module, cfg, plan: dict, data_dir: Path, batch_limit: int, seed: int,
                     bank_dir: Path, device) -> dict:
    import torch
    from omegaconf import OmegaConf

    from nndet.io.datamodule.bg_module import Datamodule

    _set_seed(seed)
    # OmegaConf.to_container builds a fresh plain dict/list tree (not a view
    # into cfg), so it is already safe to mutate directly.
    augment_cfg = OmegaConf.to_container(cfg.augment_cfg, resolve=True)
    augment_cfg["seed"] = seed
    augment_cfg["multiprocessing"] = False
    augment_cfg["num_threads"] = 1
    augment_cfg["num_train_batches_per_epoch"] = batch_limit
    dataloader_kwargs = augment_cfg.setdefault("dataloader_kwargs", {})
    dataloader_kwargs["grade_balanced_sampling"] = False
    transfer_cfg = dataloader_kwargs.setdefault("lesion_transfer_cfg", {})
    transfer_cfg["enabled"] = True
    transfer_cfg["bank_dir"] = str(bank_dir)

    datamodule = Datamodule(
        augment_cfg=augment_cfg,
        plan=plan,
        data_dir=data_dir,
        fold=int(cfg.exp.fold),
    )
    _assert_patient_disjoint(datamodule)
    datamodule.setup(stage="fit")
    augmenter = datamodule.train_dataloader()
    core_loader = _loader_core(augmenter)
    totals = [0, 0, 0, 0]
    positive = supervised = unsupervised = 0
    observed_batches = 0
    try:
        iterator = iter(augmenter)
        with torch.no_grad():
            for batch_idx in range(batch_limit):
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("Training dataloader ended before the requested batch cap") from exc
                output = module.training_step(_move_tensors_to_device(batch, device), batch_idx=batch_idx)
                counts = [int(output[f"grade_anchor_count_GGG{grade}"]) for grade in range(2, 6)]
                totals = [left + right for left, right in zip(totals, counts)]
                positive += int(output["grade_positive_anchors"])
                supervised += int(output["grade_supervised_anchors"])
                unsupervised += int(output["grade_unsupervised_anchors"])
                observed_batches += 1
                module.training_step_outputs.clear()
                if observed_batches % 25 == 0:
                    print(f"batches: {observed_batches}/{batch_limit}", flush=True)
    finally:
        finish = getattr(augmenter, "_finish", None)
        if callable(finish):
            finish()
        module.training_step_outputs.clear()

    if supervised != sum(totals) or positive != supervised + unsupervised:
        raise RuntimeError("Sampled-anchor telemetry did not reconcile")
    result = summarize_counts(totals, observed_batches)
    result.update({
        "sampled_positive_anchors": positive,
        "sampled_unsupervised_anchors": unsupervised,
        "sampled_supervised_fraction": supervised / positive if positive else 0.0,
        "transfer_counters": {key: int(value) for key, value in core_loader.paste_counters.items()},
    })
    bank = core_loader.lesion_bank
    result["bank_filter"] = {
        "kept_donors": int(bank.kept_count),
        "dropped_donors": int(bank.dropped_count),
        "training_cases": len(datamodule.dataset_tr),
    }
    if result["transfer_counters"]["attempted"] == 0 or result["transfer_counters"]["succeeded"] == 0:
        raise RuntimeError("The measurement run did not complete any lesion paste; this is not post-paste telemetry")
    return result


def _validate_config(cfg, bank_dir: Path) -> None:
    if cfg.model_cfg.head_grade_kwargs.grade_loss_type != "coral":
        raise ValueError("Final telemetry requires grade_loss_type=coral")
    if cfg.trainer_cfg.grade_class_weight_source != "anchor":
        raise ValueError("Final telemetry requires grade_class_weight_source=anchor")
    if cfg.trainer_cfg.grade_anchor_class_counts is not None:
        raise ValueError("Clear grade_anchor_class_counts before measuring the post-paste prior")
    if cfg.augment_cfg.dataloader_kwargs.grade_balanced_sampling:
        raise ValueError("Final telemetry requires grade_balanced_sampling=false")
    if cfg.augment_cfg.dataloader != "DataLoader{}DLesionTransfer":
        raise ValueError("Final telemetry requires the lesion-transfer dataloader")
    transfer_cfg = cfg.augment_cfg.dataloader_kwargs.lesion_transfer_cfg
    if not transfer_cfg.enabled or transfer_cfg.bank_dir is None:
        raise ValueError("The final config must enable transfer and define a bank directory")
    if Path(str(transfer_cfg.bank_dir)).resolve() != bank_dir.resolve():
        raise ValueError("Resolved final-config bank path does not match --bank-dir")
    if transfer_cfg.shuffle_labels:
        raise ValueError("Phase 1 telemetry must use the unshuffled-label final arm")


def _write_evidence(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=False, exist_ok=False)
    (path / "anchor-telemetry.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    # ADR 0005's own name for this artifact, in the schema RetinaUNetModule's
    # on_train_epoch_end already produces during real training -- so anyone
    # reading either one recognizes the same record shape.
    grade_anchor_counts_record = {
        "epoch": 0,
        "sampled_positive_anchors": payload["sampled_positive_anchors"],
        "sampled_supervised_anchors": payload["sampled_supervised_anchors"],
        "sampled_unsupervised_anchors": payload["sampled_unsupervised_anchors"],
        "sampled_supervised_fraction": payload["sampled_supervised_fraction"],
        "class_counts": payload["class_counts"],
    }
    (path / "grade_anchor_counts.json").write_text(
        json.dumps(grade_anchor_counts_record, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Task2201_PICAI_csPCa")
    parser.add_argument("--train-config", choices=("gcalf_final", "gcalf_task2202"),
                        help="must match the task; defaults to gcalf_task2202 for Task2202 and gcalf_final otherwise")
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--total-batches", type=int, default=MAX_BATCHES,
                        help="Batch cap for the measurement run; must be from 1 through 250")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    import torch
    from hydra import initialize_config_module
    from omegaconf import OmegaConf

    from nndet.io.load import load_pickle
    from nndet.ptmodule import MODULE_REGISTRY
    from nndet.utils.config import compose

    args = parse_args()
    train_config = resolve_train_config(args.task, args.train_config)
    if args.total_batches < 1 or args.total_batches > MAX_BATCHES:
        raise ValueError("--total-batches must be from 1 through 250")
    if not args.bank_dir.is_dir() or not (args.bank_dir / "bank_index.pkl").is_file():
        raise FileNotFoundError("--bank-dir must contain bank_index.pkl")
    if args.evidence.exists():
        raise FileExistsError("Evidence directory already exists; choose a fresh path")
    task_dir = Path(os.environ["det_data"]) / args.task
    if os.path.commonpath([str(args.evidence.resolve()), str(task_dir.resolve())]) == str(task_dir.resolve()):
        raise ValueError("Evidence directory must be outside det_data")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1 telemetry requires the requested Vast CUDA instance")

    os.environ["GCALF_LESION_BANK_DIR"] = str(args.bank_dir.resolve())
    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose(args.task, "config.yaml", overrides=[f"train={train_config}", "exp.fold=0", "exp.seed=2026"])
    _validate_config(cfg, args.bank_dir)
    plan = load_pickle(Path(str(cfg.host.plan_path)))
    data_dir = Path(cfg.host.preprocessed_output_dir) / plan["data_identifier"] / "imagesTr"

    plan_path = Path(str(cfg.host.plan_path))
    plan_hash = _sha256(plan_path)
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    config_hash = hashlib.sha256(
        json.dumps(resolved_config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    source_paths = [
        args.repo_dir / "nndet/core/retina.py",
        args.repo_dir / "nndet/io/datamodule/bg_loader.py",
        args.repo_dir / "nndet/io/datamodule/bg_module.py",
        args.repo_dir / "nndet/io/augmentation/lesion_transfer.py",
        args.repo_dir / f"nndet/conf/train/{train_config}.yaml",
    ]
    source_hashes = {str(path.relative_to(args.repo_dir)): _sha256(path) for path in source_paths}
    source_hashes["scripts/measure_transfer_anchor_telemetry.py"] = _sha256(Path(__file__).resolve())

    module = MODULE_REGISTRY[cfg.module](
        model_cfg=OmegaConf.to_container(cfg.model_cfg, resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg.trainer_cfg, resolve=True),
        plan=plan,
    )
    device = torch.device("cuda")
    module.to(device)
    module.eval()
    measurement = _run_measurement(
        module=module,
        cfg=cfg,
        plan=plan,
        data_dir=data_dir,
        batch_limit=args.total_batches,
        seed=args.seed,
        bank_dir=args.bank_dir,
        device=device,
    )
    payload = {
        "protocol": "ADR-0005 Phase 1 post-paste anchor telemetry",
        "task": args.task,
        "train_config": train_config,
        "fold": 0,
        "seed": args.seed,
        "batch_cap": args.total_batches,
        "batches_observed": measurement["batches"],
        "optimizer_steps": 0,
        "backward_passes": 0,
        "gradient_mode": "torch.no_grad",
        "checkpoint_loaded": False,
        "config_sha256": config_hash,
        "plan_sha256": plan_hash,
        "source_hashes": source_hashes,
        **measurement,
    }
    _write_evidence(args.evidence, payload)
    print(json.dumps({
        "grade_anchor_class_counts": payload["grade_anchor_class_counts"],
        "next_step": (
            "Paste grade_anchor_class_counts into trainer_cfg.grade_anchor_class_counts "
            f"in nndet/conf/train/{train_config}.yaml"
        ),
        "transfer_counters": payload["transfer_counters"],
        "bank_filter": payload["bank_filter"],
        "evidence": str(args.evidence),
    }, indent=2))
    return 0


if __name__ == "__main__":
    from loguru import logger

    logger.remove()
    try:
        raise SystemExit(main())
    except Exception as exc:
        message = CASE_ID_PATTERN.sub("[case]", str(exc))
        print(f"{type(exc).__name__}: {message}", file=sys.stderr)
        raise SystemExit(2)
