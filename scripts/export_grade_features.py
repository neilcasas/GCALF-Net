#!/usr/bin/env python3
"""Export matched-positive pre-logit features for the ADR 0005 Day-0 probe."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from nndet.io.datamodule import DATALOADER_REGISTRY
from nndet.io.datamodule.bg_module import Datamodule
from nndet.io.load import load_pickle
from nndet.ptmodule import MODULE_REGISTRY


GRADE_VALUES = (2, 3, 4, 5)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _supervised_tasks(dataset):
    tasks = []
    expected_cases = set()
    for case_id, item in dataset.items():
        properties = load_pickle(item["properties_file"])
        grades = properties.get("grades", {})
        supervised = properties.get("grade_supervised", {})
        for raw_instance_id, is_supervised in supervised.items():
            if not is_supervised:
                continue
            instance_id = int(raw_instance_id)
            grade = grades.get(str(raw_instance_id), grades.get(raw_instance_id))
            if grade is None or int(grade) not in GRADE_VALUES:
                raise ValueError(f"{item['properties_file']}: invalid supervised GGG {grade}")
            tasks.append((case_id, instance_id, int(grade)))
            expected_cases.add(case_id)
    if not tasks:
        raise ValueError("Fold-0 training data contains no grade-supervised lesions")
    return tasks, expected_cases


def _case_loader(datamodule, case_id, item):
    kwargs = dict(datamodule.dataloader_kwargs)
    kwargs["grade_balanced_sampling"] = False
    kwargs["lesion_transfer_cfg"] = None
    loader_cls = DATALOADER_REGISTRY.get(datamodule.dataloader)
    loader = loader_cls(
        data={case_id: item},
        batch_size=1,
        patch_size_generator=datamodule.patch_size_generator,
        patch_size_final=datamodule.patch_size,
        oversample_foreground_percent=1.0,
        pad_mode="constant",
        num_batches_per_epoch=1,
        **kwargs,
    )
    return loader


def _export_one(module, loader, transform, case_id, instance_id, batch_num, device):
    loader.select = lambda: ([case_id], [instance_id])
    batch = transform(**loader.generate_train_batch())
    batch = module.pre_trafo(**batch)
    targets = {
        "target_boxes": [value.to(device) for value in batch["boxes"]],
        "target_classes": [value.to(device) for value in batch["classes"]],
        "target_seg": batch["target"][:, 0].to(device),
        "target_grades": [value.to(device) for value in batch["grades"]],
        "target_grade_supervised": [value.to(device) for value in batch["grade_supervised"]],
        "grade_class_weights": module.grade_class_weights,
        "export_grade_features": True,
    }
    with torch.no_grad():
        _, prediction, _ = module.model.train_step(
            images=batch["data"].to(device),
            targets=targets,
            evaluation=False,
            batch_num=batch_num,
        )
    if prediction is None or "grade_feature_export" not in prediction:
        raise RuntimeError("The model did not return requested grade feature diagnostics")
    exported = prediction["grade_feature_export"]
    image_indices = exported["image_indices"].cpu().tolist()
    case_ids = [batch["keys"][int(index)] for index in image_indices]
    return (
        exported["features"].cpu().numpy(),
        exported["grades"].cpu().numpy(),
        exported["supervised_mask"].cpu().numpy(),
        np.asarray(case_ids, dtype=str),
    )


def _validate_export(features, grades, supervised, case_ids):
    rows = len(features)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError(f"Expected nonempty feature vectors, got shape {features.shape}")
    if any(np.asarray(values).ndim != 1 or len(values) != rows
           for values in (grades, supervised, case_ids)):
        raise ValueError("Exported feature rows, grades, masks, and case IDs are misaligned")
    if not np.isfinite(features).all():
        raise ValueError("Exported features contain NaN or infinite values")
    supervised = np.asarray(supervised, dtype=bool)
    if not supervised.any():
        raise ValueError("Export contains no grade-supervised matched positive anchors")
    invalid = sorted(set(np.asarray(grades)[supervised].tolist()) - set(GRADE_VALUES))
    if invalid:
        raise ValueError(f"Supervised rows contain invalid grades: {invalid}")
    missing = sorted(set(GRADE_VALUES) - set(np.asarray(grades)[supervised].tolist()))
    if missing:
        raise ValueError(f"Export has no supervised matched positive anchors for GGG {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Resolved checkpoint-run Hydra config")
    parser.add_argument("--plan", type=Path, required=True, help="Checkpoint-run plan.pkl")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-one-per-grade", action="store_true",
                        help="Export one real supervised lesion per grade to validate checkpoint loading")
    args = parser.parse_args()
    if args.seed != 2026:
        parser.error("ADR 0005 locks the diagnostic seed to 2026")
    if not args.config.is_file() or not args.plan.is_file() or not args.checkpoint.is_file():
        parser.error("--config, --plan, and --checkpoint must name existing files")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output directory: {args.output_dir}")

    _seed_everything(args.seed)
    cfg = OmegaConf.load(args.config)
    if int(cfg.exp.fold) != 0:
        parser.error(f"the Day-0 source checkpoint must be fold 0, got fold {cfg.exp.fold}")
    plan = load_pickle(args.plan)
    module = MODULE_REGISTRY[cfg.module](
        model_cfg=OmegaConf.to_container(cfg.model_cfg, resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg.trainer_cfg, resolve=True),
        plan=plan,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no Lightning state_dict")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(args.device).eval()

    augment_cfg = OmegaConf.to_container(cfg.augment_cfg, resolve=True)
    augment_cfg["seed"] = args.seed
    augment_cfg["multiprocessing"] = False
    augment_cfg["num_threads"] = 1
    augment_cfg.setdefault("dataloader_kwargs", {})["grade_balanced_sampling"] = False
    augment_cfg["dataloader_kwargs"]["lesion_transfer_cfg"] = None
    data_dir = Path(str(cfg.host.preprocessed_output_dir)) / plan["data_identifier"] / "imagesTr"
    datamodule = Datamodule(augment_cfg=augment_cfg, plan=plan, data_dir=data_dir, fold=0)
    datamodule.setup()
    transform = datamodule.augmentation.get_validation_transforms()
    source_tasks, expected_cases = _supervised_tasks(datamodule.dataset_tr)
    tasks = source_tasks
    if args.smoke_one_per_grade:
        selected = {}
        for task in sorted(source_tasks):
            selected.setdefault(task[2], task)
        missing = sorted(set(GRADE_VALUES) - set(selected))
        if missing:
            raise ValueError(f"Fold-0 data has no smoke lesion for GGG {missing}")
        tasks = [selected[grade] for grade in GRADE_VALUES]

    feature_rows, grade_rows, mask_rows, case_rows = [], [], [], []
    seen_tasks = set()
    loaders = {}
    for batch_num, (case_id, instance_id, target_grade) in enumerate(tasks):
        if case_id not in loaders:
            loaders[case_id] = _case_loader(datamodule, case_id, datamodule.dataset_tr[case_id])
        features, grades, masks, case_ids = _export_one(
            module, loaders[case_id], transform,
            case_id, instance_id, batch_num, args.device)
        if not len(features):
            raise RuntimeError(f"No sampled positive anchor matched {case_id} instance {instance_id}")
        supervised = np.asarray(masks, dtype=bool)
        if not np.any(supervised & (np.asarray(grades) == target_grade)):
            raise RuntimeError(
                f"Selected GGG{target_grade} lesion {case_id} instance {instance_id} "
                "produced no supervised matched-positive feature"
            )
        if set(case_ids.tolist()) != {case_id}:
            raise RuntimeError("Exported image indices did not map back to the selected case")
        feature_rows.append(features)
        grade_rows.append(grades)
        mask_rows.append(masks)
        case_rows.append(case_ids)
        seen_tasks.add((case_id, instance_id))

    if not args.smoke_one_per_grade:
        expected_tasks = {(case, instance) for case, instance, _ in source_tasks}
        missing_tasks = expected_tasks - seen_tasks
        if missing_tasks:
            raise RuntimeError(f"Feature export omitted {len(missing_tasks)} supervised source lesions")
    features = np.concatenate(feature_rows, axis=0)
    grades = np.concatenate(grade_rows).astype(np.int64, copy=False)
    supervised = np.concatenate(mask_rows).astype(bool, copy=False)
    case_ids = np.concatenate(case_rows).astype(str, copy=False)
    _validate_export(features, grades, supervised, case_ids)
    counts = {f"GGG{grade}": int(np.sum(supervised & (grades == grade))) for grade in GRADE_VALUES}
    exported_cases = sorted(set(case_ids[supervised].tolist()))
    missing_cases = expected_cases - set(exported_cases)
    if not args.smoke_one_per_grade and missing_cases:
        raise RuntimeError(f"Feature export omitted supervised rows for {len(missing_cases)} source cases")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = args.output_dir / "grade_features.npz"
    np.savez_compressed(
        feature_path,
        features=features,
        grades=grades,
        supervised_mask=supervised,
        case_ids=case_ids,
    )
    manifest = {
        "schema_version": 1,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "source_checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "source_config": str(args.config),
        "source_plan": str(args.plan),
        "source_fold": 0,
        "seed": args.seed,
        "smoke_one_per_grade": args.smoke_one_per_grade,
        "feature_file": str(feature_path),
        "feature_rows": int(len(features)),
        "supervised_rows_by_grade": counts,
        "supervised_case_count": len(exported_cases),
        "matched_positive_anchor_path": True,
        "lesion_transfer_enabled": False,
        "pre_logit_feature_width": int(features.shape[1]),
    }
    (args.output_dir / "grade_features_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "source_checkpoint_sha256", "feature_rows", "supervised_rows_by_grade",
        "supervised_case_count", "smoke_one_per_grade", "pre_logit_feature_width")}, indent=2))


if __name__ == "__main__":
    main()
