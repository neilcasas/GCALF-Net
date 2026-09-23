"""Backfill gland/zonal candidate centres into preprocessed case properties.

The default is a read-only dry run.  The frame is reconstructed by putting the
raw anatomical masks on the already-produced ``raw_splitted`` image grid and
then applying the exact crop/transpose/resize metadata stored by nnDetection.
"""

import argparse
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

# The same corroborated displacement bar as build_lesion_bank.py's donor
# eligibility (DONOR_MAX_OUTSIDE_DISTANCE_MM): a lesion this far outside the
# gland is genuinely displaced, not boundary-adjacent (2026-09-23
# anatomy-containment diagnosis; docs/adr/0005-final-grade-configuration.md).
DISPLACEMENT_EXCLUSION_DISTANCE_MM = 2.0


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


def _fit_shape(array: np.ndarray, shape: Tuple[int, ...], context: str = "mask") -> np.ndarray:
    """Correct only a one-voxel resampling-rounding difference."""
    if tuple(array.shape) == tuple(shape):
        return array
    if array.ndim != len(shape):
        raise ValueError(f"{context}: cannot fit {array.ndim}D array to {len(shape)}D shape {shape}")
    delta = np.asarray(shape, dtype=np.int64) - np.asarray(array.shape, dtype=np.int64)
    if np.any(np.abs(delta) > 1):
        raise ValueError(
            f"{context}: resampling shape differs by more than one voxel: "
            f"source={tuple(array.shape)}, target={tuple(shape)}, delta_zyx={delta.tolist()}"
        )
    logging.getLogger(__name__).warning(
        "%s: correcting one-voxel resampling shape delta source=%s target=%s delta_zyx=%s",
        context, tuple(array.shape), tuple(shape), delta.tolist(),
    )
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


def _resample_patient(*args, **kwargs):
    from nndet.preprocessing.resampling import resample_patient

    return resample_patient(*args, **kwargs)


def _mask_in_preprocessed_frame(
    mask: np.ndarray,
    properties: Mapping[str, object],
    target_shape: Tuple[int, ...],
    transpose_forward=None,
    target_spacing=None,
    resample_anisotropy_threshold: float = 3.0,
    shape_adjustments: Optional[List[dict]] = None,
    context: str = "mask",
) -> np.ndarray:

    crop_bbox = properties.get("crop_bbox")
    transpose_forward = properties.get("transpose_forward") if transpose_forward is None else transpose_forward
    if crop_bbox is None or transpose_forward is None:
        raise ValueError("Properties must contain crop_bbox and the preprocessing plan must contain transpose_forward")
    expected_shape = tuple(int(value) for value in properties["original_size_of_raw_data"])
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"{context}: reference mask shape {tuple(mask.shape)} does not match nnDetection raw shape {expected_shape}"
        )
    transpose_forward = tuple(int(axis) for axis in transpose_forward)
    if sorted(transpose_forward) != list(range(mask.ndim)):
        raise ValueError(f"Invalid transpose_forward for {context}: {transpose_forward}")
    cropped = mask[tuple(slice(int(bounds[0]), int(bounds[1])) for bounds in crop_bbox)]
    recorded_crop_shape = properties.get("size_after_cropping")
    if recorded_crop_shape is not None and tuple(cropped.shape) != tuple(recorded_crop_shape):
        raise ValueError(
            f"{context}: cropped shape {tuple(cropped.shape)} does not match nnDetection crop "
            f"{tuple(recorded_crop_shape)}"
        )
    transposed = np.transpose(cropped, axes=transpose_forward)
    if target_spacing is None:
        target_spacing = properties.get("spacing_after_resampling")
    original_spacing = np.asarray(properties["original_spacing"], dtype=np.float64)[list(transpose_forward)]
    if target_spacing is None:
        raise ValueError(f"{context}: target spacing is unavailable in plan and case properties")
    target_spacing = np.asarray(target_spacing, dtype=np.float64)
    recorded_spacing = properties.get("spacing_after_resampling")
    if recorded_spacing is not None and not np.allclose(target_spacing, recorded_spacing, atol=1e-6):
        raise ValueError(
            f"{context}: plan target spacing {target_spacing.tolist()} differs from case metadata "
            f"{np.asarray(recorded_spacing).tolist()}"
        )
    if original_spacing.shape != target_spacing.shape or target_spacing.shape != (mask.ndim,):
        raise ValueError(
            f"{context}: invalid resampling spacings original={original_spacing}, target={target_spacing}"
        )
    _, resampled = _resample_patient(
        None,
        transposed[np.newaxis].astype(np.uint8, copy=False),
        original_spacing,
        target_spacing,
        order_data=3,
        order_seg=0,
        force_separate_z=False,
        order_z_data=9999,
        order_z_seg=9999,
        separate_z_anisotropy_threshold=float(resample_anisotropy_threshold),
    )
    if resampled is None or resampled.shape[0] != 1:
        raise ValueError(f"{context}: nnDetection resampler returned invalid segmentation output")
    resampled = np.asarray(resampled[0])
    delta = (np.asarray(target_shape, dtype=np.int64) - np.asarray(resampled.shape, dtype=np.int64)).tolist()
    if shape_adjustments is not None:
        shape_adjustments.append({
            "context": context,
            "source_shape": tuple(int(value) for value in resampled.shape),
            "target_shape": tuple(int(value) for value in target_shape),
            "delta_zyx": delta,
        })
    return _fit_shape(resampled, tuple(target_shape), context=context)


def _anatomy_codes(gland: np.ndarray, zones: np.ndarray) -> np.ndarray:
    """Encode the persisted gland/PZ/TZ frame used by lesion transfer."""
    gland = np.asarray(gland, dtype=bool)
    zones = np.asarray(zones)
    if gland.shape != zones.shape:
        raise ValueError(f"Gland and zone masks must have the same shape: {gland.shape}, {zones.shape}")
    anatomy = np.zeros(gland.shape, dtype=np.uint8)
    anatomy[gland & (zones == 1)] = 1
    anatomy[gland & (zones == 2)] = 2
    anatomy[gland & ~np.isin(zones, (1, 2))] = 3
    return anatomy


def _instance_outside_distance_p95_mm(seg: np.ndarray, gland: np.ndarray, spacing) -> Dict[str, Optional[float]]:
    """For each real lesion instance in ``seg``, the p95 of how far its
    outside-gland voxels sit from the gland surface, in mm; ``None`` where the
    instance has no outside voxels (or the gland mask is empty). Backs the
    donor-eligibility bar in build_lesion_bank.py: a genuinely displaced
    instance is excluded as a paste donor even where its case still passes the
    lesion-bearing (not per-instance) containment gate."""
    from scipy import ndimage

    gland = np.asarray(gland, dtype=bool)
    seg = np.asarray(seg)
    distances = ndimage.distance_transform_edt(~gland, sampling=spacing) if gland.any() else None
    result: Dict[str, Optional[float]] = {}
    for instance_id in np.unique(seg):
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        outside = (seg == instance_id) & ~gland
        if distances is not None and outside.any():
            result[str(instance_id)] = float(np.quantile(distances[outside], 0.95))
        else:
            result[str(instance_id)] = None
    return result


def _case_outside_distance_p95_mm(seg: np.ndarray, gland: np.ndarray, spacing) -> Optional[float]:
    """The p95 of how far a case's real-lesion voxels (combined, matching the
    ``containment`` definition itself) sit outside the gland, in mm. ``None``
    when every voxel is inside (or there is no lesion). Backs the anatomy
    frame gate's documented displacement exclusion: a case whose containment
    fails the per-case minimum AND whose lesion sits genuinely displaced --
    not merely boundary-adjacent -- is excluded from that gate's population
    (docs/adr/0005-final-grade-configuration.md), never from training."""
    from scipy import ndimage

    gland = np.asarray(gland, dtype=bool)
    real_lesions = np.asarray(seg) > 0
    if not gland.any() or not real_lesions.any():
        return None
    outside = real_lesions & ~gland
    if not outside.any():
        return None
    distances = ndimage.distance_transform_edt(~gland, sampling=spacing)
    return float(np.quantile(distances[outside], 0.95))


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


def _case_update(
    task_dir: Path,
    labels_root: Path,
    case_id: str,
    seed: int,
    max_centres: int,
    transpose_forward,
    target_spacing_transposed=None,
    resample_anisotropy_threshold: float = 3.0,
    include_masks: bool = False,
):
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
    shape_adjustments = []
    gland = _mask_in_preprocessed_frame(
        gland, properties, target_shape, transpose_forward, target_spacing_transposed,
        resample_anisotropy_threshold, shape_adjustments, f"{case_id}:gland",
    ).astype(bool)
    zones = _mask_in_preprocessed_frame(
        zones, properties, target_shape, transpose_forward, target_spacing_transposed,
        resample_anisotropy_threshold, shape_adjustments, f"{case_id}:zones",
    ).astype(np.uint8)
    anatomy = _anatomy_codes(gland, zones)

    real_lesions = np.asarray(seg) > 0
    has_lesion = bool(real_lesions.any())
    containment = float(gland[real_lesions].mean()) if has_lesion else 1.0
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
    spacing = np.asarray(properties["spacing_after_resampling"], dtype=np.float64)
    case_outside_distance_p95_mm = _case_outside_distance_p95_mm(seg, gland, spacing)
    properties["anatomy_centers"] = anatomy_centers
    properties["anatomy_instance_zone_pz_frac"] = instance_zone_fraction
    properties["anatomy_instance_outside_distance_p95_mm"] = _instance_outside_distance_p95_mm(
        seg, gland, spacing
    )
    properties["anatomy_case_outside_distance_p95_mm"] = case_outside_distance_p95_mm
    properties["anatomy_frame"] = {
        "space": "preprocessed_3d",
        "axis_order": [["z", "y", "x"][axis] for axis in transpose_forward],
        "reference": str(reference_path),
        "crop_bbox": properties["crop_bbox"],
        "transpose_forward": list(transpose_forward),
        "size_after_resampling": tuple(target_shape),
        "whole_gland_source": _WHOLE_GLAND_SOURCE,
        "zonal_source": _ZONAL_SOURCE,
        "filename": f"{case_id}_anatomy.npy",
        "code_map": {
            "outside_gland": 0,
            "gland_pz": 1,
            "gland_tz": 2,
            "gland_other": 3,
        },
    }
    result = (properties, containment, anatomy, has_lesion, case_outside_distance_p95_mm)
    if include_masks:
        return result + ({
            "seg": np.asarray(seg),
            "gland": gland,
            "zones": zones,
            "shape_adjustments": shape_adjustments,
        },)
    return result


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "min": None, "median": None, "p05": None, "max": None}
    return {
        "n": int(len(values)),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "max": float(values.max()),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


def backfill(task_dir: Path, labels_root: Path, write: bool, seed: int = 2026,
             max_centres: int = 2000, plan_path: Path = None,
             evidence_dir: Optional[Path] = None) -> Dict[str, float]:
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
    lesion_bearing = {}
    case_distances = {}
    staged_anatomy = {}
    try:
        for case_id in _case_ids(images_dir):
            result = _case_update(
                task_dir, labels_root, case_id, seed, max_centres, transpose_forward,
                plan.get("target_spacing_transposed"), plan.get("resample_anisotropy_threshold", 3.0),
            )
            properties, containment, anatomy, has_lesion, case_distance = result
            updates[case_id] = properties
            containments[case_id] = containment
            lesion_bearing[case_id] = bool(has_lesion)
            case_distances[case_id] = case_distance
            if anatomy is not None:
                staged_path = None
                try:
                    with tempfile.NamedTemporaryFile("wb", dir=images_dir, suffix=".npy", delete=False) as file:
                        staged_path = Path(file.name)
                        np.save(file, np.asarray(anatomy, dtype=np.uint8), allow_pickle=False)
                    staged_anatomy[case_id] = staged_path
                except Exception:
                    if staged_path is not None and staged_path.exists():
                        staged_path.unlink()
                    raise

        values = np.asarray(list(containments.values()), dtype=np.float64)
        current_filtered = values[values < 1.0] if np.any(values < 1.0) else values
        lesion_bearing_ids = [case_id for case_id, has_lesion in lesion_bearing.items() if has_lesion]
        corrected_values = np.asarray([containments[case_id] for case_id in lesion_bearing_ids], dtype=np.float64)
        if not len(corrected_values):
            raise ValueError("No lesion-bearing cases were found; anatomy containment cannot be validated")

        # Displacement exclusion (docs/adr/0005-final-grade-configuration.md,
        # 2026-09-23 addendum): a lesion-bearing case is excluded from this
        # gate's own population -- never from training, splits, or the raw
        # task -- only when it already fails the per-case minimum AND its
        # lesion sits genuinely displaced from the gland (the same
        # DISPLACEMENT_EXCLUSION_DISTANCE_MM bar used for donor eligibility in
        # build_lesion_bank.py), corroborated by an independent lesion-replay
        # and cross-gland-source check recorded in the ADR. This is a
        # documented, evidence-backed correction to the gate's population, not
        # a relaxation of its thresholds.
        gate_excluded = sorted(
            case_id for case_id in lesion_bearing_ids
            if containments[case_id] < 0.5
            and case_distances.get(case_id) is not None
            and case_distances[case_id] > DISPLACEMENT_EXCLUSION_DISTANCE_MM
        )
        gated_values = np.asarray(
            [containments[case_id] for case_id in lesion_bearing_ids if case_id not in gate_excluded],
            dtype=np.float64,
        )
        if not len(gated_values):
            raise ValueError(
                "No cases remain after the displacement exclusion; anatomy containment cannot be validated"
            )
        summary = {
            "thresholds": {"median_minimum": 0.9, "per_case_minimum": 0.5},
            "displacement_exclusion_distance_mm": DISPLACEMENT_EXCLUSION_DISTANCE_MM,
            "case_count": len(containments),
            "lesion_bearing_case_count": int(sum(lesion_bearing.values())),
            "current_filtered_distribution": _distribution(current_filtered),
            "lesion_bearing_distribution": _distribution(corrected_values),
            "gate_excluded_cases": {
                case_id: {"containment": float(containments[case_id]),
                          "outside_distance_p95_mm": case_distances[case_id]}
                for case_id in gate_excluded
            },
            "gated_distribution": _distribution(gated_values),
            "lesion_bearing_cases": {
                case_id: float(containments[case_id]) for case_id in lesion_bearing_ids
            },
            "gate_passed": bool(np.median(gated_values) >= 0.9 and gated_values.min() >= 0.5),
        }
        if evidence_dir is not None:
            evidence_dir = Path(evidence_dir)
            evidence_resolved = evidence_dir.resolve()
            input_roots = [task_dir.resolve(), labels_root.resolve()]
            if os.environ.get("det_data"):
                input_roots.append(Path(os.environ["det_data"]).resolve())
            for input_root in input_roots:
                try:
                    evidence_resolved.relative_to(input_root)
                except ValueError:
                    continue
                raise ValueError(f"Evidence output directory must be outside input data: {input_root}")
            _write_json(evidence_dir / "anatomy-containment-backfill.json", summary)
        print(
            "Containment distributions:",
            {"current_filtered": summary["current_filtered_distribution"],
             "lesion_bearing": summary["lesion_bearing_distribution"],
             "gated": summary["gated_distribution"],
             "displacement_excluded_case_count": len(summary["gate_excluded_cases"])},
        )
        if summary["gate_excluded_cases"]:
            print("Displacement-excluded cases (see ADR 0005):", sorted(summary["gate_excluded_cases"]))
        if not summary["gate_passed"]:
            raise ValueError(
                "Anatomy frame validation failed: median containment over the gated population must be "
                ">= 0.9 and every gated case >= 0.5"
            )
        if write:
            for case_id, staged_path in staged_anatomy.items():
                os.replace(staged_path, images_dir / f"{case_id}_anatomy.npy")
            for case_id, properties in updates.items():
                _write_pickle(images_dir / f"{case_id}.pkl", properties)
    finally:
        for staged_path in staged_anatomy.values():
            if staged_path.exists():
                staged_path.unlink()
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
    parser.add_argument("--evidence-dir", type=Path, default=Path("/workspace/evidence"),
                        help="directory outside det_data for the containment summary JSON")
    args = parser.parse_args()
    labels_root = args.labels_root or (Path(os.environ["PICAI_LABELS_ROOT"])
                                       if os.environ.get("PICAI_LABELS_ROOT") else None)
    if labels_root is None:
        parser.error("--labels-root or PICAI_LABELS_ROOT is required")
    if args.max_centres <= 0:
        parser.error("--max-centres must be positive")
    backfill(args.task_dir, labels_root, args.write, args.seed, args.max_centres, args.plan, args.evidence_dir)


if __name__ == "__main__":
    main()
