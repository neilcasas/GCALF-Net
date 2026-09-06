"""Build the supervised PI-CAI csPCa nnDetection task and install its splits.

All 1,500 cases train csPCa detection (a single foreground class); only
grade-resolved lesions (human_expert masks directly, plus audit-recovered
Pooch25 cases) additionally carry GGG2-5 grade metadata for the separate
grade head (PHASE_1_data_pipeline.md; ADR 0002 D2).
"""

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from picai_prep import nnunet2nndet

from gcalf_data import preprocessing
from gcalf_data.audit_crop_retention import (
    excluded_case_ids,
    make_record,
    validate_audit,
    write_audit,
)
from gcalf_data.audit_unifocal import run_audit
from gcalf_data.build_labels import inject_grade_metadata

DEFAULT_TASK = "Task2201_PICAI_csPCa"
TINY_NUM_MODALITIES = 3
_MODALITY_SUFFIXES = ("t2w", "adc", "hbv")
_WHOLE_GLAND_SOURCE = "Bosma22b"  # the PI-CAI maintainers' own AI segmentation; Guerbet23 also covers all 1,500 cases


def load_splits(path: Path) -> List[Dict[str, List[str]]]:
    with path.open() as file:
        splits = json.load(file)
    if not isinstance(splits, list) or len(splits) != 5:
        raise ValueError("Official PI-CAI splits must contain exactly five folds")
    for fold, split in enumerate(splits):
        if set(split) != {"train", "val"}:
            raise ValueError(f"Fold {fold} must contain only train and val keys")
        if not all(isinstance(case_id, str) for case_id in split["train"] + split["val"]):
            raise ValueError(f"Fold {fold} contains a non-string case identifier")
    return splits


def apply_exclusions_to_splits(
    splits: Sequence[Dict[str, List[str]]], exclusions: Set[str]
) -> List[Dict[str, List[str]]]:
    """Remove predeclared preprocessing exclusions from every official fold.

    A source-positive lesion that is wholly lost by the target-independent crop
    cannot be relabelled as benign. ADR 0002 D6 item 7 requires its removal
    from every arm, which means deriving the retained-cohort folds here.
    """
    retained_splits = []
    for fold, split in enumerate(splits):
        train = [case_id for case_id in split["train"] if case_id not in exclusions]
        val = [case_id for case_id in split["val"] if case_id not in exclusions]
        if set(train) & set(val):
            raise ValueError(f"Fold {fold} has overlapping train and validation cases after exclusions")
        retained_splits.append({"train": train, "val": val})
    return retained_splits


def dataset_json(task: str) -> Dict[str, object]:
    return {
        "task": task,
        "name": "PI-CAI csPCa lesion detection",
        "description": "Supervised PI-CAI csPCa lesion detection; GGG2-5 grade carried as instance metadata.",
        "tensorImageSize": "4D",
        "reference": "PI-CAI public training and development dataset",
        "licence": "CC BY-NC 4.0",
        "release": "1.0",
        "modality": {"0": "T2W", "1": "ADC", "2": "HBV"},
        "labels": {"0": "background", "1": "csPCa"},
    }


def patient_id(case_id: str) -> str:
    """Return the PI-CAI patient identifier embedded in a case identifier."""
    return case_id.split("_", 1)[0]


def _index_patient_dirs(images_dirs: Sequence[Path]) -> Dict[str, Path]:
    """Map patient_id -> the fold directory containing that patient's scans."""
    index: Dict[str, Path] = {}
    for images_dir in images_dirs:
        for candidate in sorted(Path(images_dir).iterdir()):
            if not candidate.is_dir():
                continue
            if candidate.name in index:
                raise ValueError(f"Patient {candidate.name} found in multiple image directories")
            index[candidate.name] = candidate
    return index


def _case_image_paths(patient_dir: Path, case_id: str) -> Dict[str, Path]:
    paths = {}
    for suffix in _MODALITY_SUFFIXES:
        path = patient_dir / f"{case_id}_{suffix}.mha"
        if not path.is_file():
            raise ValueError(f"Missing {suffix.upper()} scan for {case_id}: {path}")
        paths[suffix] = path
    return paths


def _lesion_mask_path(labels_root: Path, case_id: str) -> Path:
    delineations = labels_root / "csPCa_lesion_delineations" / "human_expert"
    resampled = delineations / "resampled" / f"{case_id}.nii.gz"
    if resampled.is_file():
        return resampled
    pooch25 = delineations / "Pooch25" / f"{case_id}.nii.gz"
    if pooch25.is_file():
        return pooch25
    raise ValueError(f"No lesion mask found for {case_id} in human_expert/resampled or Pooch25")


def _whole_gland_mask_path(labels_root: Path, case_id: str) -> Path:
    path = labels_root / "anatomical_delineations" / "whole_gland" / "AI" / _WHOLE_GLAND_SOURCE / f"{case_id}.nii.gz"
    if not path.is_file():
        raise ValueError(f"No whole-gland mask found for {case_id}: {path}")
    return path


def build_task(
    images_dirs: Sequence[Path],
    labels_root: Path,
    task_dir: Path,
    splits_json: Path,
    work_dir: Path,
    task_name: str = DEFAULT_TASK,
) -> None:
    """Preprocess every PI-CAI case (PHASE_1_data_pipeline.md Sec 1.2) and
    assemble the nnDetection csPCa task."""
    import SimpleITK as sitk

    marksheet_path = labels_root / "clinical_information" / "marksheet.csv"
    pooch25_dir = labels_root / "csPCa_lesion_delineations" / "human_expert" / "Pooch25"
    if not marksheet_path.is_file() or not pooch25_dir.is_dir():
        raise ValueError("labels_root must be the picai_labels checkout")
    if task_dir.exists() and any(task_dir.iterdir()):
        raise FileExistsError(f"Refusing to merge into an existing task directory: {task_dir}")

    splits = load_splits(splits_json)
    case_ids = sorted({case_id for split in splits for key in ("train", "val") for case_id in split[key]})

    patient_index = _index_patient_dirs(images_dirs)

    nnunet_task_dir = Path(work_dir) / "nnUNet_raw" / task_name
    images_out = nnunet_task_dir / "imagesTr"
    labels_out = nnunet_task_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    total_cases = len(case_ids)
    crop_strategy_exceptions: Dict[str, str] = {}
    crop_retention_records = []
    for index, case_id in enumerate(case_ids, start=1):
        patient_dir = patient_index.get(patient_id(case_id))
        if patient_dir is None:
            raise ValueError(f"No image directory found for patient {patient_id(case_id)} ({case_id})")
        image_paths = _case_image_paths(patient_dir, case_id)
        lesion_mask_path = _lesion_mask_path(labels_root, case_id)
        whole_gland_path = _whole_gland_mask_path(labels_root, case_id)

        t2w = sitk.ReadImage(str(image_paths["t2w"]))
        adc = sitk.ReadImage(str(image_paths["adc"]))
        hbv = sitk.ReadImage(str(image_paths["hbv"]))
        lesion_mask = sitk.ReadImage(str(lesion_mask_path))
        whole_gland = sitk.ReadImage(str(whole_gland_path))

        t2w, adc, hbv, lesion_mask, crop_strategy, retention = preprocessing.preprocess_case_with_retention(
            t2w, adc, hbv, lesion_mask, whole_gland
        )
        retention_record = make_record(case_id, crop_strategy, retention)
        crop_retention_records.append(retention_record)

        excluded = retention_record["status"] == "excluded_no_retained_voxels"
        if excluded:
            note = " [excluded: gland-centred crop retained no lesion voxels]"
        else:
            sitk.WriteImage(t2w, str(images_out / f"{case_id}_0000.nii.gz"))
            sitk.WriteImage(adc, str(images_out / f"{case_id}_0001.nii.gz"))
            sitk.WriteImage(hbv, str(images_out / f"{case_id}_0002.nii.gz"))
            sitk.WriteImage(lesion_mask, str(labels_out / f"{case_id}.nii.gz"))

        elapsed = time.time() - start_time
        eta = elapsed / index * (total_cases - index)
        if crop_strategy != "gland":
            crop_strategy_exceptions[case_id] = crop_strategy
            note += f" [used target-independent '{crop_strategy}' crop strategy]"
        print(
            f"[{index}/{total_cases}] preprocessed {case_id} "
            f"(elapsed {elapsed / 60:.1f} min, eta {eta / 60:.1f} min){note}",
            flush=True,
        )

    with (nnunet_task_dir / "dataset.json").open("w") as file:
        json.dump(dataset_json(task_name), file, indent=2)

    task_dir.parent.mkdir(parents=True, exist_ok=True)
    nnunet2nndet(nnunet_task_dir, task_dir)

    audit_results = run_audit(pooch25_dir, marksheet_path)
    inject_grade_metadata(task_dir / "raw_splitted" / "labelsTr", audit_results)

    exclusions = excluded_case_ids(crop_retention_records)
    splits = apply_exclusions_to_splits(splits, exclusions)
    with (task_dir / "splits.json").open("w") as file:
        json.dump(splits, file, indent=2)

    audit_paths = write_audit(crop_retention_records, task_dir)
    raw_cases = {
        path.name[:-12]
        for path in (task_dir / "raw_splitted" / "imagesTr").glob("*_0000.nii.gz")
    }
    split_cases = {case_id for split in splits for key in ("train", "val") for case_id in split[key]}
    validate_audit(task_dir, raw_cases, split_cases)

    with (task_dir / "crop_strategy_exceptions.json").open("w") as file:
        json.dump(crop_strategy_exceptions, file, indent=2, sort_keys=True)
    if crop_strategy_exceptions:
        print(
            f"{len(crop_strategy_exceptions)} case(s) needed a non-gland-centered crop "
            f"(recorded in crop_strategy_exceptions.json): {crop_strategy_exceptions}"
        )
    if exclusions:
        print(f"Excluded {len(exclusions)} source-positive case(s): {sorted(exclusions)}")
    print(f"Wrote crop-retention audit: {audit_paths['audit']}")


def install_splits(task_dir: Path, preprocessed_dir: Path) -> Path:
    """Install the official JSON folds as nnDetection's ``splits_final.pkl``."""
    splits = load_splits(task_dir / "splits.json")
    raw_cases = {
        path.name[:-12]
        for path in (task_dir / "raw_splitted" / "imagesTr").glob("*_0000.nii.gz")
    }
    split_cases = {case_id for split in splits for key in ("train", "val") for case_id in split[key]}
    if raw_cases != split_cases:
        raise ValueError(
            f"Raw task cases ({len(raw_cases)}) and official split cases ({len(split_cases)}) differ"
        )
    validate_audit(task_dir, raw_cases, split_cases)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    output_path = preprocessed_dir / "splits_final.pkl"
    with output_path.open("wb") as file:
        pickle.dump(splits, file)
    return output_path


def _case_ids(images_dir: Path) -> List[str]:
    return sorted(path.name[:-12] for path in images_dir.glob("*_0000.nii.gz"))


def _load_case_grades(labels_dir: Path, case_id: str) -> Set[int]:
    """Return the set of grade_supervised grades present for a case (empty if
    benign or positive-but-ungraded)."""
    metadata_path = labels_dir / f"{case_id}.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing instances metadata for {case_id}")
    with metadata_path.open() as file:
        metadata = json.load(file)
    return {int(grade) for grade in metadata.get("grades", {}).values()}


def _has_any_instance(labels_dir: Path, case_id: str) -> bool:
    metadata_path = labels_dir / f"{case_id}.json"
    with metadata_path.open() as file:
        return bool(json.load(file)["instances"])


def _validate_case_files(source_dir: Path, case_id: str) -> None:
    images_dir = source_dir / "raw_splitted" / "imagesTr"
    for modality in range(TINY_NUM_MODALITIES):
        image_path = images_dir / f"{case_id}_{modality:04d}.nii.gz"
        if not image_path.is_file():
            raise ValueError(f"Missing modality {modality} for {case_id}: {image_path}")


def _select_case(candidates: Iterable[str], used_patients: Set[str], description: str) -> str:
    for case_id in candidates:
        if patient_id(case_id) not in used_patients:
            used_patients.add(patient_id(case_id))
            return case_id
    raise ValueError(f"Could not select a distinct-patient {description} case")


def select_tiny_cases(source_task_dir: Path) -> Dict[str, List[str]]:
    """Select the fixed six-case M3 smoke cohort from an M1 task: one
    grade-supervised case per grade 2-5, one additional positive case, and one
    benign case, all from distinct patients."""
    images_dir = source_task_dir / "raw_splitted" / "imagesTr"
    labels_dir = source_task_dir / "raw_splitted" / "labelsTr"
    case_ids = _case_ids(images_dir)
    if not case_ids:
        raise ValueError(f"No training cases found in {images_dir}")

    grades_by_case = {}
    positive_by_case = {}
    for case_id in case_ids:
        _validate_case_files(source_task_dir, case_id)
        grades_by_case[case_id] = _load_case_grades(labels_dir, case_id)
        positive_by_case[case_id] = _has_any_instance(labels_dir, case_id)

    used_patients: Set[str] = set()
    representatives = []
    for grade in (2, 3, 4, 5):
        candidates = sorted(
            (case_id for case_id, grades in grades_by_case.items() if grade in grades),
            key=lambda case_id: (grades_by_case[case_id] != {grade}, case_id),
        )
        representatives.append(_select_case(candidates, used_patients, f"GGG{grade}"))

    remaining_positive = sorted(
        case_id for case_id in case_ids if positive_by_case[case_id] and case_id not in representatives
    )
    additional_positive = _select_case(remaining_positive, used_patients, "additional positive")
    benign_candidates = sorted(case_id for case_id in case_ids if not positive_by_case[case_id])
    benign = _select_case(benign_candidates, used_patients, "benign")

    return {
        "train": representatives,
        "val": [additional_positive, benign],
        "all": representatives + [additional_positive, benign],
    }


def _copy_case(source_task_dir: Path, target_task_dir: Path, case_id: str, split: str) -> None:
    source_root = source_task_dir / "raw_splitted"
    target_root = target_task_dir / "raw_splitted"
    source_images = source_root / "imagesTr"
    source_labels = source_root / "labelsTr"
    target_images = target_root / f"imagesT{split}"
    target_labels = target_root / f"labelsT{split}"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    for modality in range(TINY_NUM_MODALITIES):
        shutil.copy2(source_images / f"{case_id}_{modality:04d}.nii.gz", target_images)
    shutil.copy2(source_labels / f"{case_id}.nii.gz", target_labels)
    shutil.copy2(source_labels / f"{case_id}.json", target_labels)


def build_tiny_task(source_task_dir: Path, task_dir: Path) -> Dict[str, object]:
    """Build the deterministic M3 smoke task from an already validated M1 task."""
    source_task_dir = Path(source_task_dir)
    task_dir = Path(task_dir)
    source_metadata_path = source_task_dir / "dataset.json"
    if not source_metadata_path.is_file():
        raise ValueError(f"M1 dataset metadata is missing: {source_metadata_path}")
    if task_dir.exists() and any(task_dir.iterdir()):
        raise FileExistsError(f"Refusing to merge into an existing task directory: {task_dir}")

    selected = select_tiny_cases(source_task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    for case_id in selected["all"]:
        _copy_case(source_task_dir, task_dir, case_id, "r")
    for case_id in selected["val"]:
        _copy_case(source_task_dir, task_dir, case_id, "s")

    with source_metadata_path.open() as file:
        metadata = json.load(file)
    metadata["task"] = task_dir.name
    metadata["name"] = "PI-CAI csPCa tiny M3 smoke task"
    metadata["test_labels"] = True
    with (task_dir / "dataset.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    splits = [{"train": selected["train"], "val": selected["val"]}]
    preprocessed_dir = task_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    with (preprocessed_dir / "splits_final.pkl").open("wb") as file:
        pickle.dump(splits, file)

    manifest = {
        "source_task": source_task_dir.name,
        "task": task_dir.name,
        "train_cases": selected["train"],
        "validation_cases": selected["val"],
        "test_cases": selected["val"],
        "test_split_note": "The smoke-test imagesTs/labelsTs are copies of held-out validation cases.",
    }
    with (task_dir / "tiny_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build the raw nnDetection task")
    build.add_argument(
        "--images-dir", type=Path, required=True, action="append", dest="images_dirs",
        help="PI-CAI public MHA fold directory (repeat for each of the 5 folds)",
    )
    build.add_argument("--labels-root", type=Path, required=True, help="picai_labels checkout")
    build.add_argument("--task-dir", type=Path, required=True, help="target $det_data task directory")
    build.add_argument("--splits-json", type=Path, required=True, help="official picai_nnunet splits.json")
    build.add_argument("--work-dir", type=Path, required=True, help="intermediate nnU-Net workspace")
    build.add_argument("--task-name", default=DEFAULT_TASK)

    splits = subparsers.add_parser("install-splits", help="write nnDetection's splits_final.pkl")
    splits.add_argument("--task-dir", type=Path, required=True)
    splits.add_argument("--preprocessed-dir", type=Path, required=True)

    tiny = subparsers.add_parser("tiny", help="build the deterministic six-case M3 smoke task")
    tiny.add_argument("--source-task-dir", type=Path, required=True, help="validated M1 task directory")
    tiny.add_argument("--task-dir", type=Path, required=True, help="new $det_data tiny task directory")

    args = parser.parse_args()
    if args.command == "build":
        build_task(
            images_dirs=args.images_dirs,
            labels_root=args.labels_root,
            task_dir=args.task_dir,
            splits_json=args.splits_json,
            work_dir=args.work_dir,
            task_name=args.task_name,
        )
    elif args.command == "install-splits":
        print(install_splits(args.task_dir, args.preprocessed_dir))
    else:
        print(json.dumps(build_tiny_task(args.source_task_dir, args.task_dir), indent=2))


if __name__ == "__main__":
    main()
