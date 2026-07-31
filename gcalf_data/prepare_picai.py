"""Build the supervised PI-CAI GGG2--5 nnDetection task and install its splits."""

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List

from gcalf_data.build_labels import GGG_LABELS, remap_label_images


DEFAULT_TASK = "Task2201_PICAI_GGG"


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
    else:
        print(install_splits(args.task_dir, args.preprocessed_dir))


if __name__ == "__main__":
    main()
