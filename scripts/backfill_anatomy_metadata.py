"""Backfill gland/zonal candidate centres into preprocessed case properties.

The default is a read-only dry run.  The frame is reconstructed by putting the
raw anatomical masks on the already-produced ``raw_splitted`` image grid and
then applying the exact crop/transpose/resize metadata stored by nnDetection.
"""

import argparse
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file)


def _write_pickle(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


def _case_ids(properties_dir: Path) -> Iterable[str]:
    paths = sorted(path for path in properties_dir.glob("*.pkl") if not path.stem.endswith("_boxes"))
    if not paths:
        raise ValueError(f"No per-case properties found in {properties_dir}")
    return [path.stem for path in paths]


def _load_preprocessed_case(images_dir: Path, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
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
            raise FileNotFoundError(f"Missing preprocessed data for {case_id} in {images_dir}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            data = loaded["data"]
            seg = loaded["seg"] if "seg" in loaded else data[-1]
            if "seg" not in loaded:
                data = data[:-1]
    if seg.ndim == 4:
        seg = seg[0]
    if data.ndim != 4 or seg.ndim != 3 or data.shape[1:] != seg.shape:
        raise ValueError(f"Invalid preprocessed shapes for {case_id}: {data.shape}, {seg.shape}")
    return np.asarray(data), np.asarray(seg)


def _fit_shape(array: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Correct the rare one-voxel rounding difference after ``ndimage.zoom``."""
    if tuple(array.shape) == tuple(shape):
        return array
    result = np.zeros(shape, dtype=array.dtype)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(array.shape, shape):
        length = min(source_size, target_size)
        source_start = (source_size - length) // 2
        target_start = (target_size - length) // 2
        source_slices.append(slice(source_start, source_start + length))
        target_slices.append(slice(target_start, target_start + length))
    result[tuple(target_slices)] = array[tuple(source_slices)]
    return result


def _mask_in_preprocessed_frame(mask, properties: Mapping[str, object], target_shape: Tuple[int, ...],
                                transpose_forward=None):
    from scipy import ndimage

    crop_bbox = properties.get("crop_bbox")
    transpose_forward = properties.get("transpose_forward") if transpose_forward is None else transpose_forward
    if crop_bbox is None or transpose_forward is None:
        raise ValueError("Properties must contain crop_bbox and the preprocessing plan must contain transpose_forward")
    cropped = mask[tuple(slice(int(bounds[0]), int(bounds[1])) for bounds in crop_bbox)]
    transposed = np.transpose(cropped, axes=tuple(int(axis) for axis in transpose_forward))
    zoom = np.asarray(target_shape, dtype=np.float64) / np.asarray(transposed.shape, dtype=np.float64)
    return _fit_shape(ndimage.zoom(transposed, zoom=zoom, order=0, prefilter=False), tuple(target_shape))


def _stable_case_rng(seed: int, case_id: str) -> np.random.Generator:
    case_bytes = case_id.encode("utf-8")
    words = [int.from_bytes(case_bytes[index:index + 4].ljust(4, b"\0"), "little")
             for index in range(0, len(case_bytes), 4)]
    return np.random.default_rng(np.random.SeedSequence([int(seed), *words]))


def _centres(mask: np.ndarray, max_centres: int, rng: np.random.Generator) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if len(coordinates) > max_centres:
        coordinates = coordinates[rng.choice(len(coordinates), size=max_centres, replace=False)]
    return coordinates.astype(np.int16, copy=False)


def _case_update(task_dir: Path, labels_root: Path, case_id: str, seed: int, max_centres: int,
                 transpose_forward) -> Tuple[dict, float]:
    import SimpleITK as sitk
    from scipy import ndimage
    from gcalf_data.preprocessing import resample_to_reference
    from gcalf_data.prepare_picai import (
        _WHOLE_GLAND_SOURCE,
        _ZONAL_SOURCE,
        _whole_gland_mask_path,
        _zonal_mask_path,
    )

    images_dir = task_dir / "preprocessed" / "D3V001_3d" / "imagesTr"
    properties_path = images_dir / f"{case_id}.pkl"
    properties = _load_pickle(properties_path)
    _data, seg = _load_preprocessed_case(images_dir, case_id)
    target_shape = tuple(int(value) for value in properties["size_after_resampling"])

    reference_path = task_dir / "raw_splitted" / "imagesTr" / f"{case_id}_0000.nii.gz"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing raw reference image: {reference_path}")
    reference = sitk.ReadImage(str(reference_path))
    gland = sitk.ReadImage(str(_whole_gland_mask_path(labels_root, case_id)))
    zones = sitk.ReadImage(str(_zonal_mask_path(labels_root, case_id)))
    gland = sitk.GetArrayFromImage(resample_to_reference(gland, reference, is_label=True)) > 0
    zones = sitk.GetArrayFromImage(resample_to_reference(zones, reference, is_label=True))
    gland = _mask_in_preprocessed_frame(gland, properties, target_shape, transpose_forward).astype(bool)
    zones = _mask_in_preprocessed_frame(zones, properties, target_shape, transpose_forward).astype(np.uint8)

    real_lesions = np.asarray(seg) > 0
    containment = float(gland[real_lesions].mean()) if real_lesions.any() else 1.0
    eroded = ndimage.binary_erosion(gland, iterations=2)
    rng = _stable_case_rng(seed, case_id)
    anatomy_centers = {
        "pz": _centres(eroded & (zones == 1), max_centres, rng),
        "tz": _centres(eroded & (zones == 2), max_centres, rng),
    }
    instance_zone_fraction = {}
    for instance_id in np.unique(seg):
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        instance = np.asarray(seg) == instance_id
        instance_zone_fraction[str(instance_id)] = float((zones[instance] == 1).mean())
    properties["anatomy_centers"] = anatomy_centers
    properties["anatomy_instance_zone_pz_frac"] = instance_zone_fraction
    properties["anatomy_frame"] = {
        "space": "preprocessed_3d",
        "axis_order": ["z", "y", "x"],
        "reference": str(reference_path),
        "crop_bbox": properties["crop_bbox"],
        "transpose_forward": list(transpose_forward),
        "size_after_resampling": tuple(target_shape),
        "whole_gland_source": _WHOLE_GLAND_SOURCE,
        "zonal_source": _ZONAL_SOURCE,
    }
    return properties, containment


def backfill(task_dir: Path, labels_root: Path, write: bool, seed: int = 2026,
             max_centres: int = 2000, plan_path: Path = None) -> Dict[str, float]:
    task_dir = Path(task_dir)
    labels_root = Path(labels_root)
    images_dir = task_dir / "preprocessed" / "D3V001_3d" / "imagesTr"
    plan_path = Path(plan_path) if plan_path is not None else task_dir / "preprocessed" / "D3V001_3d.pkl"
    plan = _load_pickle(plan_path)
    transpose_forward = plan.get("transpose_forward")
    if transpose_forward is None:
        raise ValueError(f"Preprocessing plan is missing transpose_forward: {plan_path}")
    updates = {}
    containments = {}
    for case_id in _case_ids(images_dir):
        properties, containment = _case_update(
            task_dir, labels_root, case_id, seed, max_centres, transpose_forward
        )
        updates[case_id] = properties
        containments[case_id] = containment

    values = np.asarray(list(containments.values()), dtype=np.float64)
    lesion_values = values[values < 1.0] if np.any(values < 1.0) else values
    print(
        "Containment distribution:",
        {"n": int(len(lesion_values)), "min": float(lesion_values.min()),
         "median": float(np.median(lesion_values)), "p05": float(np.quantile(lesion_values, 0.05)),
         "max": float(lesion_values.max())},
    )
    if float(np.median(lesion_values)) < 0.9 or float(lesion_values.min()) < 0.5:
        raise ValueError("Anatomy frame validation failed: median containment must be >= 0.9 and every case >= 0.5")
    if write:
        for case_id, properties in updates.items():
            _write_pickle(images_dir / f"{case_id}.pkl", properties)
    print("Mode:", "WRITE" if write else "DRY_RUN")
    print("Cases:", len(updates), "updated:" if write else "would update:", len(updates))
    return containments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, default=None,
                        help="picai_labels checkout; defaults to $PICAI_LABELS_ROOT")
    parser.add_argument("--plan", type=Path, default=None,
                        help="preprocessing plan; defaults to task-dir/preprocessed/D3V001_3d.pkl")
    parser.add_argument("--write", action="store_true", help="write metadata after validation")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-centres", type=int, default=2000)
    args = parser.parse_args()
    labels_root = args.labels_root or (Path(os.environ["PICAI_LABELS_ROOT"])
                                       if os.environ.get("PICAI_LABELS_ROOT") else None)
    if labels_root is None:
        parser.error("--labels-root or PICAI_LABELS_ROOT is required")
    if args.max_centres <= 0:
        parser.error("--max-centres must be positive")
    backfill(args.task_dir, labels_root, args.write, args.seed, args.max_centres, args.plan)


if __name__ == "__main__":
    main()
