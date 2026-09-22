"""Build the complete preprocessed lesion-transfer bank.

The command is deliberately dry-run by default.  It must run after anatomy
metadata backfill so each index record can carry its PZ fraction.  No fold
filtering happens here; the training dataloader applies the patient-level fold
filter when it constructs :class:`LesionBank`.
"""

import argparse
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file)


def _load_case(images_dir: Path, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    npy_path = images_dir / f"{case_id}.npy"
    if npy_path.is_file():
        data = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        seg_path = images_dir / f"{case_id}_seg.npy"
        if not seg_path.is_file():
            raise FileNotFoundError(f"Missing segmentation array: {seg_path}")
        seg = np.load(seg_path, mmap_mode="r", allow_pickle=False)
    else:
        npz_path = images_dir / f"{case_id}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing preprocessed data for {case_id}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            data = loaded["data"]
            if "seg" in loaded:
                seg = loaded["seg"]
            else:
                seg = data[-1]
                data = data[:-1]
    if seg.ndim == 4:
        seg = seg[0]
    if data.ndim != 4 or seg.ndim != 3 or data.shape[1:] != seg.shape:
        raise ValueError(f"Invalid preprocessed shapes for {case_id}: {data.shape}, {seg.shape}")
    if data.shape[0] != 3:
        raise ValueError(f"Lesion bank requires exactly three modalities for {case_id}, got {data.shape[0]}")
    return np.asarray(data), np.asarray(seg)


def _case_ids(images_dir: Path) -> Iterable[str]:
    paths = sorted(path for path in images_dir.glob("*.pkl") if not path.stem.endswith("_boxes"))
    if not paths:
        raise ValueError(f"No per-case properties found in {images_dir}")
    return [path.stem for path in paths]


def _write_npz(path: Path, data: np.ndarray, mask: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as file:
        temporary_path = Path(file.name)
        np.savez_compressed(file, data=data.astype(np.float16), mask=mask.astype(bool))
    os.replace(temporary_path, path)


def _supervised(properties: dict):
    if "anatomy_centers" not in properties or "anatomy_frame" not in properties:
        raise ValueError("Anatomy metadata is missing; run backfill_anatomy_metadata.py first")
    grades = properties.get("grades", {})
    supervised = properties.get("grade_supervised", {})
    fractions = properties.get("anatomy_instance_zone_pz_frac", {})
    for raw_instance_id, is_supervised in supervised.items():
        if not is_supervised:
            continue
        instance_id = int(raw_instance_id)
        raw_grade = grades.get(str(raw_instance_id), grades.get(instance_id))
        if raw_grade is None:
            raise ValueError(f"Supervised instance {instance_id} has no grade metadata")
        grade = int(raw_grade)
        if grade not in (2, 3, 4, 5):
            raise ValueError(f"Invalid supervised GGG{grade} for instance {instance_id}")
        fraction = fractions.get(str(raw_instance_id), fractions.get(instance_id))
        yield instance_id, grade, None if fraction is None else float(fraction)


def build_bank(task_dir: Path, bank_dir: Path, write: bool) -> Tuple[list, int]:
    task_dir = Path(task_dir)
    bank_dir = Path(bank_dir)
    images_dir = task_dir / "preprocessed" / "D3V001_3d" / "imagesTr"
    records = []
    total_bytes = 0
    pending = []
    for case_id in _case_ids(images_dir):
        properties = _load_pickle(images_dir / f"{case_id}.pkl")
        data, seg = _load_case(images_dir, case_id)
        spacing = tuple(float(value) for value in properties.get("spacing_after_resampling", ()))
        for instance_id, grade, zone_pz_frac in _supervised(properties):
            lesion = seg == instance_id
            coordinates = np.argwhere(lesion)
            if not len(coordinates):
                raise ValueError(f"Supervised instance {case_id}:{instance_id} is absent from the segmentation")
            mins = np.maximum(coordinates.min(axis=0) - 4, 0)
            maxs = np.minimum(coordinates.max(axis=0) + 5, np.asarray(seg.shape))
            slices = tuple(slice(int(start), int(stop)) for start, stop in zip(mins, maxs))
            crop_data = np.asarray(data[(slice(None), *slices)], dtype=np.float16)
            crop_mask = np.asarray(lesion[slices], dtype=bool)
            filename = f"{case_id}__{instance_id}.npz"
            records.append({
                "case_id": case_id,
                "instance_id": instance_id,
                "grade": grade,
                "voxels": int(lesion.sum()),
                "shape": tuple(int(value) for value in crop_mask.shape),
                "zone_pz_frac": zone_pz_frac,
                "spacing": spacing,
            })
            pending.append((filename, crop_data, crop_mask))
            total_bytes += crop_data.nbytes + crop_mask.nbytes

    print(f"Lesions: {len(records)}; estimated uncompressed payload: {total_bytes / 1024 / 1024:.1f} MiB")
    if write:
        bank_dir.mkdir(parents=True, exist_ok=True)
        for filename, crop_data, crop_mask in pending:
            _write_npz(bank_dir / filename, crop_data, crop_mask)
        with tempfile.NamedTemporaryFile("wb", dir=bank_dir, delete=False) as file:
            temporary_index = Path(file.name)
            pickle.dump(records, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_index, bank_dir / "bank_index.pkl")
    print("Mode:", "WRITE" if write else "DRY_RUN")
    return records, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, default=None)
    parser.add_argument("--write", action="store_true", help="write bank files after reviewing the dry run")
    args = parser.parse_args()
    bank_dir = args.bank_dir or (args.task_dir / "preprocessed" / "lesion_bank")
    build_bank(args.task_dir, bank_dir, args.write)


if __name__ == "__main__":
    main()
