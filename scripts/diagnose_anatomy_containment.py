"""Read-only anatomy-frame and lesion-containment audit for the PI-CAI task.

The script imports the backfill's reconstruction path and writes reports only
to ``--evidence-dir``. It never writes into the task, label, or source-image
directories.
"""

import argparse
import csv
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from gcalf_data.prepare_picai import (
    _WHOLE_GLAND_SOURCE,
    _ZONAL_SOURCE,
    _lesion_mask_path,
    _whole_gland_mask_path,
    _zonal_mask_path,
)
from gcalf_data.preprocessing import resample_to_reference
from scripts import backfill_anatomy_metadata as backfill


CASE_FIELDS = (
    "case_id", "lesion_voxels", "gland_voxels", "intersect_voxels", "containment", "has_lesion",
    "gland_volume_ml", "gland_volume_suspect", "lesion_centroid_zyx", "gland_centroid_zyx",
    "centroid_offset_mm", "reconstructed_shape", "size_after_resampling", "shape_delta_zyx",
    "n_instances", "per_instance_containment", "per_instance_outside_distance_mm",
    "pz_center_count", "tz_center_count",
    "replay_dice", "replay_centroid_offset_mm", "crop_strategy", "crop_status",
    "source_voxels", "resampled_voxels", "inplane_voxels", "final_voxels", "source_components",
    "resampled_components", "inplane_components", "final_components", "crop_strategy_exception",
    "lesion_source",
)


def _distribution(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"n": 0, "min": None, "median": None, "p05": None, "max": None}
    return {
        "n": int(len(array)), "min": float(array.min()), "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)), "max": float(array.max()),
    }


def _containment(gland: np.ndarray, lesion: np.ndarray) -> Tuple[int, int, float]:
    lesion = np.asarray(lesion, dtype=bool)
    gland = np.asarray(gland, dtype=bool)
    lesion_voxels = int(lesion.sum())
    intersect_voxels = int(np.logical_and(gland, lesion).sum())
    value = float(intersect_voxels / lesion_voxels) if lesion_voxels else 1.0
    return lesion_voxels, intersect_voxels, value


def _centroid(mask: np.ndarray) -> Optional[Sequence[float]]:
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if not len(coordinates):
        return None
    return [float(value) for value in coordinates.mean(axis=0)]


def _to_zyx(values: Optional[Sequence[float]], axis_order: Sequence[str]) -> Optional[Sequence[float]]:
    if values is None:
        return None
    by_axis = dict(zip(axis_order, values))
    return [float(by_axis[axis]) for axis in ("z", "y", "x")]


def _centroid_offset_mm(
    first: Optional[Sequence[float]], second: Optional[Sequence[float]], spacing
) -> Optional[float]:
    if first is None or second is None:
        return None
    delta = (np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64))
    return float(np.linalg.norm(delta * np.asarray(spacing, dtype=np.float64)))


def _instance_stats(seg: np.ndarray, gland: np.ndarray) -> Dict[str, float]:
    result = {}
    for instance_id in np.unique(seg):
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        instance = np.asarray(seg) == instance_id
        result[str(instance_id)] = float(np.asarray(gland, dtype=bool)[instance].mean())
    return result


def _instance_surface_stats(seg: np.ndarray, gland: np.ndarray, spacing,
                            properties: Mapping[str, object]) -> Dict[str, dict]:
    """Per-instance containment plus how far the instance's outside-gland voxels
    lie from the gland surface. The distance transform is computed once per case;
    one per instance would repeat identical work. Grade metadata is carried through
    so donor eligibility can be measured against the same GGG2-5 supervised set
    `build_lesion_bank._supervised` admits."""
    from scipy import ndimage

    gland = np.asarray(gland, dtype=bool)
    seg = np.asarray(seg)
    distances = ndimage.distance_transform_edt(~gland, sampling=spacing) if gland.any() else None
    grades = properties.get("grades") or {}
    supervised = properties.get("grade_supervised") or {}

    result = {}
    for instance_id in np.unique(seg):
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        key = str(instance_id)
        instance = seg == instance_id
        outside = instance & ~gland
        raw_grade = grades.get(key, grades.get(instance_id))
        record = {
            "voxels": int(instance.sum()),
            "containment": float(gland[instance].mean()),
            "outside_voxels": int(outside.sum()),
            "grade": None if raw_grade is None else int(raw_grade),
            "grade_supervised": bool(supervised.get(key, supervised.get(instance_id, False))),
            "distance_mm": None,
        }
        if distances is not None and outside.any():
            outside_distances = distances[outside]
            record["distance_mm"] = {
                "min": float(outside_distances.min()),
                "p25": float(np.quantile(outside_distances, 0.25)),
                "median": float(np.median(outside_distances)),
                "p75": float(np.quantile(outside_distances, 0.75)),
                "p95": float(np.quantile(outside_distances, 0.95)),
                "max": float(outside_distances.max()),
            }
        result[key] = record
    return result


def _donor_counts(donors: Sequence[Mapping[str, object]], predicate) -> dict:
    """Surviving donor instances under one eligibility bar, split by grade."""
    kept = [item for item in donors if predicate(item)]
    counts = {"total": len(kept)}
    counts.update({f"ggg{grade}": sum(1 for item in kept if item["grade"] == grade)
                   for grade in (2, 3, 4, 5)})
    return counts


def _read_json(path: Path, default):
    if not path.is_file():
        return default
    with path.open() as file:
        return json.load(file)


def _load_crop_audit(path: Path) -> Dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing crop-retention audit: {path}")
    with path.open(newline="") as file:
        return {row["case_id"]: row for row in csv.DictReader(file)}


def _t2w_path(images_dir: Path, case_id: str) -> Path:
    patient_id = case_id.split("_", 1)[0]
    candidates = [images_dir / patient_id / f"{case_id}_t2w.mha"]
    candidates.extend(images_dir.glob(f"*/{patient_id}/{case_id}_t2w.mha"))
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one source T2W for {case_id} under {images_dir}, found {len(found)}")
    return found[0]


def _sitk_geometry(image) -> dict:
    return {
        "size_xyz": [int(value) for value in image.GetSize()],
        "spacing_xyz": [float(value) for value in image.GetSpacing()],
        "origin_xyz": [float(value) for value in image.GetOrigin()],
        "direction_xyz": [float(value) for value in image.GetDirection()],
    }


def _geometry_delta(reference, image) -> dict:
    reference_geometry = _sitk_geometry(reference)
    image_geometry = _sitk_geometry(image)
    size_delta = np.asarray(image_geometry["size_xyz"]) - np.asarray(reference_geometry["size_xyz"])
    spacing_delta = np.asarray(image_geometry["spacing_xyz"]) - np.asarray(reference_geometry["spacing_xyz"])
    origin_delta = np.asarray(image_geometry["origin_xyz"]) - np.asarray(reference_geometry["origin_xyz"])
    direction_delta = np.asarray(image_geometry["direction_xyz"]) - np.asarray(reference_geometry["direction_xyz"])
    return {
        "size_matches": image.GetSize() == reference.GetSize(),
        "size_delta_xyz": size_delta.tolist(),
        "spacing_delta_xyz": spacing_delta.tolist(),
        "origin_delta_xyz": origin_delta.tolist(),
        "direction_delta_xyz": direction_delta.tolist(),
        "spacing_matches_atol_1e-4": bool(np.allclose(image.GetSpacing(), reference.GetSpacing(), atol=1e-4)),
        "origin_matches_atol_1e-4": bool(np.allclose(image.GetOrigin(), reference.GetOrigin(), atol=1e-4)),
        "direction_matches_atol_1e-4": bool(np.allclose(image.GetDirection(), reference.GetDirection(), atol=1e-4)),
    }


def _mask_bbox_fov_fraction(mask_image, reference_image, samples_per_axis: int = 25) -> Optional[float]:
    """Estimate the fraction of the foreground bounding box inside a reference FOV.

    A regular grid samples the continuous voxel-edge bounding box in physical
    space. The estimate is deterministic and handles differing oblique
    directions without resampling either image.
    """
    import SimpleITK as sitk

    binary = sitk.Cast(mask_image > 0, sitk.sitkUInt8)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(binary)
    if not statistics.HasLabel(1):
        return None
    box = statistics.GetBoundingBox(1)
    bounds_xyz = [
        (float(box[axis]) - 0.5, float(box[axis] + box[axis + 3] - 1) + 0.5)
        for axis in range(3)
    ]
    axes = [np.linspace(low, high, samples_per_axis) for low, high in bounds_xyz]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    mask_direction = np.asarray(mask_image.GetDirection(), dtype=np.float64).reshape(3, 3)
    mask_linear = mask_direction @ np.diag(mask_image.GetSpacing())
    world = grid @ mask_linear.T + np.asarray(mask_image.GetOrigin())
    reference_direction = np.asarray(reference_image.GetDirection(), dtype=np.float64).reshape(3, 3)
    reference_linear = reference_direction @ np.diag(reference_image.GetSpacing())
    reference_indices = (world - np.asarray(reference_image.GetOrigin())) @ np.linalg.inv(reference_linear).T
    size = np.asarray(reference_image.GetSize(), dtype=np.float64)
    inside = np.all((reference_indices >= -0.5) & (reference_indices <= (size - 0.5)), axis=1)
    return float(inside.mean())


def _load_mask_on_preprocessed_frame(path: Path, reference, properties, target_shape, transpose_forward,
                                     target_spacing, anisotropy_threshold, shape_adjustments, context):
    import SimpleITK as sitk

    source = sitk.ReadImage(str(path))
    on_reference = sitk.GetArrayFromImage(resample_to_reference(source, reference, is_label=True))
    return backfill._mask_in_preprocessed_frame(
        on_reference, properties, target_shape, transpose_forward, target_spacing,
        anisotropy_threshold, shape_adjustments, context,
    ) > 0


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    denominator = int(first.sum()) + int(second.sum())
    return float(2 * np.logical_and(first, second).sum() / denominator) if denominator else 1.0


def _surface_summary(gland: np.ndarray, lesion: np.ndarray, spacing) -> dict:
    from scipy import ndimage

    outside = np.asarray(lesion, dtype=bool) & ~np.asarray(gland, dtype=bool)
    if not outside.any() or not np.asarray(gland, dtype=bool).any():
        return {"outside_voxels": int(outside.sum()), "distance_mm": None}
    distances = ndimage.distance_transform_edt(~np.asarray(gland, dtype=bool), sampling=spacing)[outside]
    return {
        "outside_voxels": int(outside.sum()),
        "distance_mm": {
            "min": float(distances.min()), "p25": float(np.quantile(distances, 0.25)),
            "median": float(np.median(distances)), "p75": float(np.quantile(distances, 0.75)),
            "p95": float(np.quantile(distances, 0.95)), "max": float(distances.max()),
        },
    }


def _summary_distribution(values: Iterable[float]) -> dict:
    return _distribution(values)


def _write_case_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_overlay(case_id: str, labels_root: Path, task_dir: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    reference_path = task_dir / "raw_splitted" / "imagesTr" / f"{case_id}_0000.nii.gz"
    reference = sitk.ReadImage(str(reference_path))
    t2w = sitk.GetArrayFromImage(reference)
    gland_image = sitk.ReadImage(str(_whole_gland_mask_path(labels_root, case_id)))
    lesion_image = sitk.ReadImage(str(_lesion_mask_path(labels_root, case_id)))
    gland = sitk.GetArrayFromImage(resample_to_reference(gland_image, reference, is_label=True)) > 0
    lesion = sitk.GetArrayFromImage(resample_to_reference(lesion_image, reference, is_label=True)) > 0
    if not lesion.any():
        return
    z_index = int(round(float(np.argwhere(lesion).mean(axis=0)[0])))
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(t2w[z_index], cmap="gray")
    axis.contour(gland[z_index], levels=[0.5], colors=["lime"], linewidths=0.8)
    axis.contour(lesion[z_index], levels=[0.5], colors=["red"], linewidths=0.8)
    axis.set_title(f"{case_id}, axial z={z_index}: gland (green), lesion (red)")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / f"anatomy-overlay-{case_id}.png", dpi=160)
    plt.close(figure)


def _assert_output_is_evidence_only(output_dir: Path, roots: Sequence[Path]) -> None:
    output_resolved = output_dir.resolve()
    for root in roots:
        root_resolved = root.resolve()
        try:
            output_resolved.relative_to(root_resolved)
        except ValueError:
            continue
        raise ValueError(f"Evidence output directory {output_resolved} must be outside input tree {root_resolved}")


def diagnose(task_dir: Path, labels_root: Path, source_images_dir: Path, evidence_dir: Path,
             plan_path: Optional[Path] = None, seed: int = 2026, max_centres: int = 2000) -> dict:
    import SimpleITK as sitk
    from scipy import ndimage

    task_dir = Path(task_dir)
    labels_root = Path(labels_root)
    source_images_dir = Path(source_images_dir)
    evidence_dir = Path(evidence_dir)
    _assert_output_is_evidence_only(evidence_dir, (task_dir, labels_root, source_images_dir))
    images_dir = task_dir / "preprocessed" / "D3V001_3d" / "imagesTr"
    plan_path = Path(plan_path) if plan_path is not None else task_dir / "preprocessed" / "D3V001_3d.pkl"
    with plan_path.open("rb") as file:
        plan = pickle.load(file)
    transpose_forward = tuple(int(value) for value in plan["transpose_forward"])
    target_spacing = plan.get("target_spacing_transposed")
    if target_spacing is None:
        target_spacing = np.asarray(plan["target_spacing"])[list(transpose_forward)]
    target_spacing = np.asarray(target_spacing, dtype=np.float64)
    anisotropy_threshold = float(plan.get("resample_anisotropy_threshold", 3.0))
    crop_rows = _load_crop_audit(task_dir / "crop_retention.csv")
    exceptions = _read_json(task_dir / "crop_strategy_exceptions.json", {})
    rows = []
    geometry = {}
    routes = {}
    alternate_sources = {}
    shape_adjustments = []
    instance_records = []
    by_source = Counter()
    for case_id in backfill._case_ids(images_dir):
        properties, containment, _anatomy, has_lesion, masks = backfill._case_update(
            task_dir, labels_root, case_id, seed, max_centres, transpose_forward,
            target_spacing, anisotropy_threshold, include_masks=True,
        )
        seg = masks["seg"]
        gland = masks["gland"]
        lesion = seg > 0
        spacing = np.asarray(properties["spacing_after_resampling"], dtype=np.float64)
        array_axis_order = [["z", "y", "x"][axis] for axis in transpose_forward]
        lesion_voxels, intersect_voxels, containment_value = _containment(gland, lesion)
        gland_voxels = int(gland.sum())
        lesion_center = _centroid(lesion)
        gland_center = _centroid(gland)
        instance_surface = _instance_surface_stats(seg, gland, spacing, properties)
        instance_stats = {key: value["containment"] for key, value in instance_surface.items()}
        instance_records.extend(
            dict(value, case_id=case_id, instance_id=int(key))
            for key, value in sorted(instance_surface.items())
        )
        row = {
            "case_id": case_id,
            "lesion_voxels": lesion_voxels,
            "gland_voxels": gland_voxels,
            "intersect_voxels": intersect_voxels,
            "containment": containment_value,
            "has_lesion": bool(has_lesion),
            "gland_volume_ml": float(gland_voxels * np.prod(spacing) / 1000.0),
            "gland_volume_suspect": not 15 <= gland_voxels * np.prod(spacing) / 1000.0 <= 200,
            "lesion_centroid_zyx": json.dumps(_to_zyx(lesion_center, array_axis_order)),
            "gland_centroid_zyx": json.dumps(_to_zyx(gland_center, array_axis_order)),
            "centroid_offset_mm": _centroid_offset_mm(lesion_center, gland_center, spacing),
            "reconstructed_shape": json.dumps(list(gland.shape)),
            "size_after_resampling": json.dumps(list(properties["size_after_resampling"])),
            "shape_delta_zyx": json.dumps(_to_zyx(
                next((item["delta_zyx"] for item in masks["shape_adjustments"]
                      if item["context"].endswith(":gland")), [0, 0, 0]),
                array_axis_order,
            )),
            "n_instances": len(instance_stats),
            "per_instance_containment": json.dumps(instance_stats, sort_keys=True),
            "per_instance_outside_distance_mm": json.dumps(
                {key: None if value["distance_mm"] is None else value["distance_mm"]["median"]
                 for key, value in instance_surface.items()},
                sort_keys=True,
            ),
            "pz_center_count": int(len(properties["anatomy_centers"]["pz"])),
            "tz_center_count": int(len(properties["anatomy_centers"]["tz"])),
        }
        audit = crop_rows.get(case_id, {})
        row.update({
            "crop_strategy": audit.get("crop_strategy", ""),
            "crop_status": audit.get("status", ""),
            "crop_strategy_exception": exceptions.get(case_id, ""),
        })
        for stage in ("source", "resampled", "inplane", "final"):
            row[f"{stage}_voxels"] = audit.get(f"{stage}_voxels", "")
            row[f"{stage}_components"] = audit.get(f"{stage}_components", "")
        lesion_path = _lesion_mask_path(labels_root, case_id)
        source_name = lesion_path.parent.name
        by_source[source_name] += int(has_lesion)
        row["lesion_source"] = source_name

        reference_path = task_dir / "raw_splitted" / "imagesTr" / f"{case_id}_0000.nii.gz"
        reference = sitk.ReadImage(str(reference_path))
        raw_lesion_image = sitk.ReadImage(str(lesion_path))
        raw_lesion = sitk.GetArrayFromImage(resample_to_reference(raw_lesion_image, reference, is_label=True)) > 0
        replay_adjustments = []
        replay = backfill._mask_in_preprocessed_frame(
            raw_lesion, properties, tuple(properties["size_after_resampling"]), transpose_forward,
            target_spacing, anisotropy_threshold, replay_adjustments, f"{case_id}:lesion-replay",
        ) > 0
        shape_adjustments.extend(masks["shape_adjustments"] + replay_adjustments)
        row["replay_dice"] = _dice(replay, lesion)
        row["replay_centroid_offset_mm"] = _centroid_offset_mm(
            _centroid(replay), lesion_center, spacing,
        )
        routes[case_id] = {
            "dice": row["replay_dice"], "centroid_offset_mm": row["replay_centroid_offset_mm"],
            "raw_replay_voxels": int(replay.sum()), "preprocessed_seg_voxels": lesion_voxels,
        }

        source_paths = {
            "t2w": _t2w_path(source_images_dir, case_id),
            "lesion": lesion_path,
            "bosma22b": _whole_gland_mask_path(labels_root, case_id),
            "heviai23": _zonal_mask_path(labels_root, case_id),
            "raw_splitted_reference": reference_path,
        }
        t2w_image = sitk.ReadImage(str(source_paths["t2w"]))
        loaded = {
            name: sitk.ReadImage(str(path)) for name, path in source_paths.items()
            if name != "raw_splitted_reference"
        }
        pairwise = {}
        for name, image in loaded.items():
            pairwise[name] = {
                "geometry": _sitk_geometry(image),
                "against_t2w": _geometry_delta(t2w_image, image),
            }
        pairwise["raw_splitted_reference"] = {
            "geometry": _sitk_geometry(reference),
            "against_t2w": _geometry_delta(t2w_image, reference),
        }
        pairwise["mask_fov_fraction"] = {
            name: {
                "inside_source_t2w_fov": _mask_bbox_fov_fraction(image, t2w_image),
                "inside_raw_splitted_fov": _mask_bbox_fov_fraction(image, reference),
            }
            for name, image in loaded.items() if name != "t2w"
        }
        gland_binary = sitk.Cast(loaded["bosma22b"] > 0, sitk.sitkUInt8)
        gland_statistics = sitk.LabelShapeStatisticsImageFilter()
        gland_statistics.Execute(gland_binary)
        gland_source_voxels = int(gland_statistics.GetNumberOfPixels(1)) if gland_statistics.HasLabel(1) else 0
        pairwise["gland_source_voxels"] = gland_source_voxels
        pairwise["gland_source_volume_ml"] = float(
            gland_source_voxels * np.prod(loaded["bosma22b"].GetSpacing()) / 1000.0
        )
        pairwise["gland_source_volume_suspect"] = not 15 <= pairwise["gland_source_volume_ml"] <= 200
        pairwise["lesion_source"] = source_name
        geometry[case_id] = pairwise
        rows.append(row)

    positive_rows = [row for row in rows if row["has_lesion"]]
    all_containment = [float(row["containment"]) for row in rows]
    corrected = [float(row["containment"]) for row in positive_rows]
    current_filtered = [value for value in all_containment if value < 1.0]
    if not current_filtered:
        current_filtered = all_containment
    source_distributions = {
        source: _summary_distribution(
            float(row["containment"]) for row in positive_rows if row["lesion_source"] == source
        )
        for source in sorted({str(row["lesion_source"]) for row in positive_rows})
    }
    crop_positive = {case_id for case_id, item in crop_rows.items() if int(item.get("source_voxels", 0)) > 0}
    crop_retained_positive = {
        case_id for case_id in crop_positive
        if crop_rows[case_id].get("status") != "excluded_no_retained_voxels"
        and int(crop_rows[case_id].get("final_voxels", 0)) > 0
    }
    frame_cases = {row["case_id"] for row in positive_rows}
    q05 = float(np.quantile(corrected, 0.05)) if corrected else None
    tail_ids = {
        row["case_id"] for row in positive_rows
        if float(row["containment"]) < 0.5 or (q05 is not None and float(row["containment"]) <= q05)
    }
    zero_ids = {row["case_id"] for row in positive_rows if float(row["containment"]) == 0.0}
    worst_nonzero = [row["case_id"] for row in sorted(
        (row for row in positive_rows if 0 < float(row["containment"])),
        key=lambda row: float(row["containment"]),
    )[:5]]
    for case_id in sorted(tail_ids):
        row = next(item for item in rows if item["case_id"] == case_id)
        properties, _containment_value, _anatomy, _has_lesion, masks = backfill._case_update(
            task_dir, labels_root, case_id, seed, max_centres, transpose_forward,
            target_spacing, anisotropy_threshold, include_masks=True,
        )
        gland = masks["gland"]
        lesion = masks["seg"] > 0
        spacing = properties["spacing_after_resampling"]
        distance = _surface_summary(gland, lesion, spacing)
        dilation = {}
        for radius in range(4):
            dilated = gland if radius == 0 else ndimage.binary_dilation(gland, iterations=radius)
            dilation[str(radius)] = _containment(dilated, lesion)[2]
        source_results = {"Bosma22b": _containment(gland, lesion)[2]}
        reference_path = task_dir / "raw_splitted" / "imagesTr" / f"{case_id}_0000.nii.gz"
        reference = sitk.ReadImage(str(reference_path))
        guerbet_path = (
            labels_root / "anatomical_delineations" / "whole_gland" / "AI" / "Guerbet23"
            / f"{case_id}.nii.gz"
        )
        if guerbet_path.is_file():
            guerbet = _load_mask_on_preprocessed_frame(
                guerbet_path, reference, properties, tuple(properties["size_after_resampling"]),
                transpose_forward, target_spacing, anisotropy_threshold, shape_adjustments,
                f"{case_id}:guerbet23",
            )
            source_results["Guerbet23"] = _containment(guerbet, lesion)[2]
            source_results["union"] = _containment(gland | guerbet, lesion)[2]
            source_results["Bosma22b_Guerbet23_dice"] = _dice(gland, guerbet)
        else:
            source_results["Guerbet23"] = None
            source_results["union"] = None
            source_results["Bosma22b_Guerbet23_dice"] = None
        original_path = labels_root / "csPCa_lesion_delineations" / "human_expert" / "original" / f"{case_id}.nii.gz"
        original_containment = None
        if original_path.is_file():
            original_lesion = _load_mask_on_preprocessed_frame(
                original_path, reference, properties, tuple(properties["size_after_resampling"]),
                transpose_forward, target_spacing, anisotropy_threshold, shape_adjustments,
                f"{case_id}:human-expert-original",
            )
            original_containment = _containment(gland, original_lesion)[2]
        alternate_sources[case_id] = {
            "surface_distance": distance,
            "dilation_containment_voxels_0_to_3": dilation,
            "gland_source_comparison": source_results,
            "original_lesion_containment": original_containment,
        }

    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write_case_csv(evidence_dir / "anatomy-containment.csv", rows)
    geometry_path = evidence_dir / "anatomy-geometry.json"
    geometry_path.write_text(json.dumps(geometry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Donor eligibility is measured over exactly the set build_lesion_bank._supervised
    # admits: grade-supervised GGG2-5 instances.
    donor_instances = [item for item in instance_records
                       if item["grade_supervised"] and item["grade"] in (2, 3, 4, 5)]
    report = {
        "case_count": len(rows),
        "lesion_bearing_case_count": len(positive_rows),
        "lesion_source_case_counts": dict(sorted(by_source.items())),
        "containment_by_lesion_source": source_distributions,
        "containment_current_filtered": _summary_distribution(current_filtered),
        "containment_lesion_bearing": _summary_distribution(corrected),
        "containment_thresholds": {"median_minimum": 0.9, "per_case_minimum": 0.5},
        "physical_bbox_fov_fraction_method": {
            "sampling_grid_points_per_axis": 25,
            "inside_bounds": "reference continuous index in [-0.5, size-0.5]",
        },
        "crop_audit": {
            "source_positive_cases": len(crop_positive),
            "retained_positive_cases": len(crop_retained_positive),
            "preprocessed_positive_cases": len(frame_cases),
            "expected_retained_positive_cases_from_adr_0002": 424,
            "retained_positive_count_matches_adr_0002": len(crop_retained_positive) == 424,
            "crop_and_preprocessed_positive_sets_match": frame_cases == crop_retained_positive,
            "retained_positive_missing_from_preprocessed": sorted(crop_retained_positive - frame_cases),
            "preprocessed_positive_missing_from_retention_audit": sorted(frame_cases - crop_retained_positive),
            "partially_clipped_preprocessed_positive_cases": sorted(
                case_id for case_id in frame_cases if crop_rows.get(case_id, {}).get("status") == "partially_clipped"
            ),
        },
        "plan": {
            "transpose_forward": list(transpose_forward),
            "is_identity_zyx": list(transpose_forward) == [0, 1, 2],
            "persisted_axis_order": [["z", "y", "x"][axis] for axis in transpose_forward],
            "target_spacing_transposed": target_spacing.tolist(),
            "resample_anisotropy_threshold": anisotropy_threshold,
        },
        "lesion_replay": {
            "dice_distribution": _summary_distribution(item["dice"] for item in routes.values()),
            "centroid_offset_mm_distribution": _summary_distribution(
                item["centroid_offset_mm"] for item in routes.values() if item["centroid_offset_mm"] is not None
            ),
            "below_0_97": {case_id: value for case_id, value in routes.items() if value["dice"] < 0.97},
            "per_case": routes,
        },
        "shape_reconciliation": {
            "adjustments": shape_adjustments,
            "nonzero_delta_histogram": {
                json.dumps(item["delta_zyx"]): sum(
                    1 for candidate in shape_adjustments if candidate["delta_zyx"] == item["delta_zyx"]
                )
                for item in shape_adjustments if any(item["delta_zyx"])
            },
            "greater_than_one_voxel": [item for item in shape_adjustments
                                       if any(abs(delta) > 1 for delta in item["delta_zyx"])],
        },
        "gland_center_counts": {
            "pz_zero_cases": [row["case_id"] for row in rows if row["pz_center_count"] == 0],
            "tz_zero_cases": [row["case_id"] for row in rows if row["tz_center_count"] == 0],
        },
        "instance_surface_distance": {
            "instance_count": len(instance_records),
            "fully_contained_instances": sum(
                1 for item in instance_records if item["outside_voxels"] == 0
            ),
            "outside_distance_median_mm": _summary_distribution(
                item["distance_mm"]["median"] for item in instance_records if item["distance_mm"]
            ),
            "outside_distance_p95_mm": _summary_distribution(
                item["distance_mm"]["p95"] for item in instance_records if item["distance_mm"]
            ),
            "donor_yield_by_outside_distance_p95_mm": {
                str(bar): _donor_counts(
                    donor_instances,
                    lambda item, bar=bar: item["distance_mm"] is None or item["distance_mm"]["p95"] <= bar,
                )
                for bar in (1.0, 1.5, 2.0, 3.0, 5.0)
            },
            "donor_yield_by_containment": {
                str(bar): _donor_counts(donor_instances, lambda item, bar=bar: item["containment"] >= bar)
                for bar in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)
            },
            "donor_instance_count": len(donor_instances),
            "per_instance": instance_records,
        },
        "tail_cases": sorted(tail_ids),
        "zero_containment_cases": sorted(zero_ids),
        "alternate_source_and_tail_analysis": alternate_sources,
        "overlay_cases": sorted(zero_ids | set(worst_nonzero)),
        "containment_gate_passed": bool(corrected and np.median(corrected) >= 0.9 and min(corrected) >= 0.5),
    }
    (evidence_dir / "anatomy-containment.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    for case_id in report["overlay_cases"]:
        _write_overlay(case_id, labels_root, task_dir, evidence_dir)
    print(json.dumps({
        "case_count": report["case_count"],
        "lesion_bearing_case_count": report["lesion_bearing_case_count"],
        "containment_current_filtered": report["containment_current_filtered"],
        "containment_lesion_bearing": report["containment_lesion_bearing"],
        "lesion_replay_dice": report["lesion_replay"]["dice_distribution"],
        "containment_gate_passed": report["containment_gate_passed"],
        "evidence_dir": str(evidence_dir),
    }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, default=None)
    parser.add_argument("--source-images-dir", "--images-dir", dest="source_images_dir", type=Path, required=True,
                        help="directory containing PI-CAI patient T2W source images")
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--evidence-dir", type=Path, default=Path("/workspace/evidence"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-centres", type=int, default=2000)
    args = parser.parse_args()
    labels_root = args.labels_root or (
        Path(os.environ["PICAI_LABELS_ROOT"]) if os.environ.get("PICAI_LABELS_ROOT") else None
    )
    if labels_root is None:
        parser.error("--labels-root or PICAI_LABELS_ROOT is required")
    if args.max_centres <= 0:
        parser.error("--max-centres must be positive")
    diagnose(args.task_dir, labels_root, args.source_images_dir, args.evidence_dir,
             args.plan, args.seed, args.max_centres)


if __name__ == "__main__":
    main()
