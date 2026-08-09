"""Build the supervised PI-CAI GGG2--5 nnDetection task and install its splits."""

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set

from gcalf_data.build_labels import GGG_LABELS, remap_label_images


DEFAULT_TASK = "Task2201_PICAI_GGG"
TINY_NUM_MODALITIES = 3


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


def dataset_json(task: str) -> Dict[str, object]:
    return {
        "task": task,
        "name": "PI-CAI GGG2-5 lesion detection",
        "description": "Supervised PI-CAI csPCa lesion detection with granular expert labels.",
        "tensorImageSize": "4D",
        "reference": "PI-CAI public training and development dataset",
        "licence": "CC BY-NC 4.0",
        "release": "1.0",
        "modality": {"0": "T2W", "1": "ADC", "2": "HBV"},
        "labels": {"0": "background", **{str(key): value for key, value in GGG_LABELS.items()}},
    }


def build_task(
    images_dir: Path,
    labels_root: Path,
    task_dir: Path,
    splits_json: Path,
    work_dir: Path,
    task_name: str,
) -> None:
    """Convert the MHA archive, remap semantic labels, and create an nnDetection task."""
    try:
        from picai_prep import MHA2nnUNetConverter, nnunet2nndet
        from picai_prep.examples.mha2nnunet.picai_archive import generate_mha2nnunet_settings
    except ImportError as error:
        raise RuntimeError("prepare_picai requires picai_prep from the M0 environment") from error

    annotations_dir = labels_root / "csPCa_lesion_delineations" / "human_expert" / "resampled"
    marksheet = labels_root / "clinical_information" / "marksheet.csv"
    if not annotations_dir.is_dir() or not marksheet.is_file():
        raise ValueError("labels_root must be the picai_labels checkout with resampled expert masks and marksheet.csv")
    if task_dir.exists() and any(task_dir.iterdir()):
        raise FileExistsError(f"Refusing to merge into an existing task directory: {task_dir}")

    splits = load_splits(splits_json)
    settings_path = work_dir / "mha2nnunet_settings.json"
    nnunet_root = work_dir / "nnUNet_raw"
    nnunet_task_dir = nnunet_root / task_name
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    generate_mha2nnunet_settings(
        archive_dir=images_dir,
        annotations_dir=annotations_dir,
        output_path=settings_path,
        task=task_name,
    )
    with settings_path.open() as file:
        settings = json.load(file)
    settings["dataset_json"] = dataset_json(task_name)
    with settings_path.open("w") as file:
        json.dump(settings, file, indent=2)

    converter = MHA2nnUNetConverter(
        scans_dir=images_dir,
        annotations_dir=annotations_dir,
        output_dir=nnunet_root,
        mha2nnunet_settings=settings,
    )
    converter.convert()
    converter.create_dataset_json()

    remap_label_images(nnunet_task_dir / "labelsTr")
    with (nnunet_task_dir / "splits.json").open("w") as file:
        json.dump(splits, file, indent=2)

    task_dir.parent.mkdir(parents=True, exist_ok=True)
    nnunet2nndet(nnunet_task_dir, task_dir)
    shutil.copy2(nnunet_task_dir / "splits.json", task_dir / "splits.json")


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
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    output_path = preprocessed_dir / "splits_final.pkl"
    with output_path.open("wb") as file:
        pickle.dump(splits, file)
    return output_path


def patient_id(case_id: str) -> str:
    """Return the PI-CAI patient identifier embedded in a case identifier."""
    return case_id.split("_", 1)[0]


def _case_ids(images_dir: Path) -> List[str]:
    return sorted(path.name[:-12] for path in images_dir.glob("*_0000.nii.gz"))


def _load_case_classes(labels_dir: Path, case_id: str) -> Set[int]:
    label_path = labels_dir / f"{case_id}.nii.gz"
    metadata_path = labels_dir / f"{case_id}.json"
    if not label_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Missing label image or instances metadata for {case_id}")
    with metadata_path.open() as file:
        instances = json.load(file).get("instances")
    if not isinstance(instances, dict):
        raise ValueError(f"Invalid instances metadata for {case_id}")
    try:
        return {int(class_id) for class_id in instances.values()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid class identifier in {metadata_path}") from error


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
    """Select the fixed six-case M3 smoke cohort from an M1 task."""
    images_dir = source_task_dir / "raw_splitted" / "imagesTr"
    labels_dir = source_task_dir / "raw_splitted" / "labelsTr"
    case_ids = _case_ids(images_dir)
    if not case_ids:
        raise ValueError(f"No training cases found in {images_dir}")

    class_by_case = {}
    for case_id in case_ids:
        _validate_case_files(source_task_dir, case_id)
        class_by_case[case_id] = _load_case_classes(labels_dir, case_id)

    used_patients: Set[str] = set()
    representatives = []
    for class_id in range(4):
        candidates = sorted(
            (case_id for case_id, classes in class_by_case.items() if class_id in classes),
            key=lambda case_id: (class_by_case[case_id] != {class_id}, case_id),
        )
        representatives.append(_select_case(candidates, used_patients, f"GGG{class_id + 2}"))

    remaining_positive = sorted(
        case_id
        for case_id, classes in class_by_case.items()
        if classes and case_id not in representatives
    )
    additional_positive = _select_case(remaining_positive, used_patients, "additional positive")
    benign_candidates = sorted(case_id for case_id, classes in class_by_case.items() if not classes)
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
    metadata["name"] = "PI-CAI GGG2-5 tiny M3 smoke task"
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
    build.add_argument("--images-dir", type=Path, required=True, help="PI-CAI public MHA archive")
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
            images_dir=args.images_dir,
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
