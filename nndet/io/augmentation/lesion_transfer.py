"""Lesion-centred transfer augmentation for grade-supervised 3D patches.

The bank contains real, preprocessed lesions.  This module only recombines a
bank entry with a training patch; it never creates a new lesion appearance.
SciPy is imported lazily so importing the dataloader remains possible in the
lightweight repository test environment.
"""

from __future__ import annotations

import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


_SPATIAL_DEFAULTS = {
    "p_rot": 0.5,
    "rot_range_deg": 30.0,
    "p_scale": 0.5,
    "scale_range": (0.8, 1.2),
    "p_flip": 0.5,
    "flip_axes": (2,),
}
_INTENSITY_DEFAULTS = {
    "p_gamma": 0.3,
    "gamma_range": (0.8, 1.25),
    "p_blur": 0.2,
}


def _ndimage():
    from scipy import ndimage

    return ndimage


def _probability(value: Any, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _pair(value: Any, name: str, positive: bool = False) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    result = (float(value[0]), float(value[1]))
    if result[0] > result[1] or (positive and result[0] <= 0):
        raise ValueError(f"{name} must be an increasing positive range, got {result}")
    return result


def _strict_mapping(value: Optional[Mapping[str, Any]], defaults: Mapping[str, Any], name: str) -> Dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    unknown = sorted(set(value) - set(defaults))
    if unknown:
        raise TypeError(f"Unknown {name} keys: {unknown}")
    result = dict(defaults)
    result.update(value)
    return result


@dataclass(frozen=True)
class LesionTransferConfig:
    """Validated configuration for lesion transfer.

    ``spatial`` and ``intensity`` remain mappings to keep the Hydra schema
    directly readable and to avoid a second public configuration shape.
    """

    enabled: bool = False
    bank_dir: Optional[str] = None
    source_grades: Tuple[int, ...] = (4, 5)
    p_paste: float = 0.5
    max_pastes_per_patch: int = 1
    region_mode: str = "dilated"
    dilation_vox: int = 3
    feather_sigma: float = 1.0
    zone_match: str = "prefer"
    min_gland_frac: float = 0.95
    min_zone_frac: float = 0.50
    max_placement_attempts: int = 8
    instance_id_offset: int = 10000
    jitter_channels: Tuple[int, ...] = (0, 2)
    shuffle_labels: bool = False
    spatial: Dict[str, Any] = field(default_factory=lambda: dict(_SPATIAL_DEFAULTS))
    intensity: Dict[str, Any] = field(default_factory=lambda: dict(_INTENSITY_DEFAULTS))

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> "LesionTransferConfig":
        """Build a config and reject misspelled top-level or nested keys."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("lesion_transfer_cfg must be a mapping or None")
        allowed = {
            "enabled", "bank_dir", "source_grades", "p_paste", "max_pastes_per_patch",
            "region_mode", "dilation_vox", "feather_sigma", "zone_match",
            "min_gland_frac", "min_zone_frac", "max_placement_attempts",
            "instance_id_offset", "jitter_channels", "shuffle_labels", "spatial", "intensity",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise TypeError(f"Unknown lesion_transfer_cfg keys: {unknown}")

        source_grades = tuple(int(grade) for grade in value.get("source_grades", (4, 5)))
        if not source_grades or any(grade not in (2, 3, 4, 5) for grade in source_grades):
            raise ValueError(f"source_grades must contain GGG2-5 values, got {source_grades}")
        jitter_channels = tuple(int(channel) for channel in value.get("jitter_channels", (0, 2)))
        if any(channel not in (0, 1, 2) for channel in jitter_channels):
            raise ValueError(f"jitter_channels must contain channel indices 0-2, got {jitter_channels}")
        if 1 in jitter_channels:
            raise ValueError("ADC channel 1 is quantitative and cannot receive intensity jitter")

        region_mode = str(value.get("region_mode", "dilated"))
        if region_mode not in {"mask", "box", "dilated"}:
            raise ValueError(f"Unknown region_mode: {region_mode}")
        zone_match = str(value.get("zone_match", "prefer"))
        if zone_match not in {"prefer", "strict", "off"}:
            raise ValueError(f"Unknown zone_match: {zone_match}")

        spatial = _strict_mapping(value.get("spatial"), _SPATIAL_DEFAULTS, "spatial")
        spatial["p_rot"] = _probability(spatial["p_rot"], "spatial.p_rot")
        spatial["p_scale"] = _probability(spatial["p_scale"], "spatial.p_scale")
        spatial["p_flip"] = _probability(spatial["p_flip"], "spatial.p_flip")
        spatial["rot_range_deg"] = float(spatial["rot_range_deg"])
        if spatial["rot_range_deg"] < 0:
            raise ValueError("spatial.rot_range_deg must be non-negative")
        spatial["scale_range"] = _pair(spatial["scale_range"], "spatial.scale_range", positive=True)
        spatial["flip_axes"] = tuple(int(axis) for axis in spatial["flip_axes"])
        if any(axis != 2 for axis in spatial["flip_axes"]):
            raise ValueError("lesion transfer permits only the in-plane L-R flip axis 2")

        intensity = _strict_mapping(value.get("intensity"), _INTENSITY_DEFAULTS, "intensity")
        intensity["p_gamma"] = _probability(intensity["p_gamma"], "intensity.p_gamma")
        intensity["p_blur"] = _probability(intensity["p_blur"], "intensity.p_blur")
        intensity["gamma_range"] = _pair(intensity["gamma_range"], "intensity.gamma_range", positive=True)

        max_pastes = int(value.get("max_pastes_per_patch", 1))
        if max_pastes < 0:
            raise ValueError("max_pastes_per_patch must be non-negative")
        dilation_vox = int(value.get("dilation_vox", 3))
        if dilation_vox < 0:
            raise ValueError("dilation_vox must be non-negative")
        feather_sigma = float(value.get("feather_sigma", 1.0))
        if feather_sigma <= 0:
            raise ValueError("feather_sigma must be positive")
        min_gland_frac = float(value.get("min_gland_frac", 0.95))
        if not 0.0 <= min_gland_frac <= 1.0:
            raise ValueError("min_gland_frac must be in [0, 1]")
        min_zone_frac = float(value.get("min_zone_frac", 0.50))
        if not 0.0 <= min_zone_frac <= 1.0:
            raise ValueError("min_zone_frac must be in [0, 1]")
        max_placement_attempts = int(value.get("max_placement_attempts", 8))
        if max_placement_attempts <= 0:
            raise ValueError("max_placement_attempts must be positive")
        instance_id_offset = int(value.get("instance_id_offset", 10000))
        if instance_id_offset <= 0:
            raise ValueError("instance_id_offset must be positive")

        bank_dir = value.get("bank_dir")
        if bank_dir is not None:
            bank_dir = str(bank_dir)
        return cls(
            enabled=bool(value.get("enabled", False)),
            bank_dir=bank_dir,
            source_grades=source_grades,
            p_paste=_probability(value.get("p_paste", 0.5), "p_paste"),
            max_pastes_per_patch=max_pastes,
            region_mode=region_mode,
            dilation_vox=dilation_vox,
            feather_sigma=feather_sigma,
            zone_match=zone_match,
            min_gland_frac=min_gland_frac,
            min_zone_frac=min_zone_frac,
            max_placement_attempts=max_placement_attempts,
            instance_id_offset=instance_id_offset,
            jitter_channels=jitter_channels,
            shuffle_labels=bool(value.get("shuffle_labels", False)),
            spatial=spatial,
            intensity=intensity,
        )


def _patient_id(case_id: str) -> str:
    return str(case_id).split("_", 1)[0]


def _record_zone(record: Mapping[str, Any]) -> Optional[str]:
    value = record.get("zone")
    if value in {"pz", "tz"}:
        return value
    fraction = record.get("zone_pz_frac")
    if fraction is None:
        return None
    fraction = float(fraction)
    if fraction > 0.5:
        return "pz"
    if fraction < 0.5:
        return "tz"
    return None


class LesionBank:
    """Index and lazily decode bank lesions, filtered to a training cohort."""

    def __init__(self, bank_dir: os.PathLike, allowed_case_ids: Set[str], max_cached: int = 128):
        self.bank_dir = Path(bank_dir)
        index_path = self.bank_dir / "bank_index.pkl"
        if not index_path.is_file():
            raise FileNotFoundError(f"Lesion bank index is missing: {index_path}")
        if max_cached <= 0:
            raise ValueError("max_cached must be positive")
        import pickle

        with index_path.open("rb") as file:
            raw_index = pickle.load(file)
        if isinstance(raw_index, Mapping):
            records = raw_index.get("lesions", raw_index.get("records", raw_index))
            if isinstance(records, Mapping):
                records = list(records.values())
        else:
            records = raw_index
        if not isinstance(records, (list, tuple)):
            raise ValueError(f"Unsupported lesion bank index format: {type(raw_index).__name__}")

        allowed = {str(case_id) for case_id in allowed_case_ids}
        allowed_patients = {_patient_id(case_id) for case_id in allowed}
        self.records = []
        self.by_grade = defaultdict(list)
        # Keep only pickleable built-in factories: the bank is sent to spawned
        # augmentation workers, and a local lambda cannot be pickled there.
        self.by_grade_zone = defaultdict(dict)
        dropped = 0
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("Every lesion bank index record must be a mapping")
            case_id = str(record["case_id"])
            if case_id not in allowed and _patient_id(case_id) not in allowed_patients:
                dropped += 1
                continue
            normalized = dict(record)
            normalized["case_id"] = case_id
            normalized["instance_id"] = int(record["instance_id"])
            normalized["grade"] = int(record["grade"])
            normalized["zone"] = _record_zone(normalized)
            self.records.append(normalized)
            self.by_grade[normalized["grade"]].append(normalized)
            if normalized["zone"] is not None:
                grade_zones = self.by_grade_zone[normalized["grade"]]
                grade_zones.setdefault(normalized["zone"], []).append(normalized)

        self.max_cached = int(max_cached)
        self._cache: "OrderedDict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self.kept_count = len(self.records)
        self.dropped_count = dropped

    def __len__(self) -> int:
        return len(self.records)

    def load(self, record: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        key = (str(record["case_id"]), int(record["instance_id"]))
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        path = self.bank_dir / f"{key[0]}__{key[1]}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Lesion bank array is missing: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            data = np.asarray(loaded["data"], dtype=np.float32)
            mask = np.asarray(loaded["mask"], dtype=bool)
        if data.ndim != 4 or data.shape[0] != 3 or mask.ndim != 3 or data.shape[1:] != mask.shape:
            raise ValueError(f"Invalid lesion bank array shapes in {path}: {data.shape}, {mask.shape}")
        value = (data, mask)
        self._cache[key] = value
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return value

    def sample(self, grade: int, zone: Optional[str], zone_match: str, rng: np.random.Generator) -> Optional[dict]:
        grade = int(grade)
        if zone_match not in {"prefer", "strict", "off"}:
            raise ValueError(f"Unknown zone_match: {zone_match}")
        if zone_match != "off" and zone in {"pz", "tz"}:
            same_zone = self.by_grade_zone.get(grade, {}).get(zone, [])
            if same_zone:
                return same_zone[int(rng.integers(len(same_zone)))]
            if zone_match == "strict":
                return None
        candidates = self.by_grade.get(grade, [])
        if not candidates:
            return None
        return candidates[int(rng.integers(len(candidates)))]


def _slice_start(value: Any) -> int:
    if isinstance(value, slice):
        return int(value.start or 0)
    return int(value[0])


def sample_target_center(
    properties: Mapping[str, Any],
    zone: Optional[str],
    crop: Sequence[Any],
    patch_shape: Sequence[int],
    lesion_shape: Sequence[int],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Sample a full-lesion-fitting center from the stored case-frame centers."""
    centers = properties.get("anatomy_centers", {})
    if zone not in {"pz", "tz"}:
        return None
    if len(crop) != 3 or len(patch_shape) != 3 or len(lesion_shape) != 3:
        raise ValueError("sample_target_center expects three spatial dimensions")
    centers = np.asarray(centers.get(zone, []), dtype=np.int64)
    if centers.size == 0:
        return None
    centers = centers.reshape(-1, len(patch_shape))
    crop_start = np.asarray([_slice_start(item) for item in crop], dtype=np.int64)
    patch_shape = np.asarray(patch_shape, dtype=np.int64)
    lesion_shape = np.asarray(lesion_shape, dtype=np.int64)
    centers = centers - crop_start
    lower = lesion_shape // 2
    upper = lesion_shape - lower
    valid = np.all(centers - lower >= 0, axis=1) & np.all(centers + upper <= patch_shape, axis=1)
    candidates = centers[valid]
    if not len(candidates):
        return None
    return candidates[int(rng.integers(len(candidates)))].astype(np.int64)


def placement_slices(
    center: Sequence[int],
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> Optional[Tuple[Tuple[slice, ...], Tuple[slice, ...]]]:
    """Return aligned target/source slices for a centred placement.

    The source crop is centred at ``center`` in the target array.  The result
    is ``(target_slices, source_slices)`` and is ``None`` when the two arrays
    do not overlap.  Keeping this calculation in one place is important:
    anatomy validation and the eventual paste must examine the same voxels.
    """
    center = np.asarray(center, dtype=np.int64)
    source_shape = np.asarray(source_shape, dtype=np.int64)
    target_shape = np.asarray(target_shape, dtype=np.int64)
    if center.ndim != 1 or source_shape.ndim != 1 or target_shape.ndim != 1:
        raise ValueError("center and shapes must be one-dimensional")
    if not (len(center) == len(source_shape) == len(target_shape)):
        raise ValueError("center and shapes must have the same number of dimensions")
    if np.any(source_shape <= 0) or np.any(target_shape <= 0):
        raise ValueError("source and target shapes must be positive")

    start = center - source_shape // 2
    target_slices = []
    source_slices = []
    for origin, size, target_size in zip(start, source_shape, target_shape):
        target_start = max(int(origin), 0)
        target_stop = min(int(origin + size), int(target_size))
        if target_start >= target_stop:
            return None
        source_start = target_start - int(origin)
        source_stop = source_start + (target_stop - target_start)
        target_slices.append(slice(target_start, target_stop))
        source_slices.append(slice(source_start, source_stop))
    return tuple(target_slices), tuple(source_slices)


def _rotation_matrix(angles: Sequence[float]) -> np.ndarray:
    rz, ry, rx = angles
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    return z @ y @ x


def augment_lesion(
    data: np.ndarray,
    mask: np.ndarray,
    config: LesionTransferConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply shared 3D geometry and selected-channel intensity jitter."""
    data = np.asarray(data, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if data.ndim != 4 or mask.ndim != 3 or data.shape[1:] != mask.shape:
        raise ValueError(f"Expected data [C,D,H,W] and mask [D,H,W], got {data.shape}, {mask.shape}")
    spatial = config.spatial
    transformed_data = data.copy()
    transformed_mask = mask.astype(np.float32)
    center = (np.asarray(mask.shape, dtype=np.float64) - 1.0) / 2.0
    matrix = np.eye(3, dtype=np.float64)
    if rng.random() < spatial["p_rot"]:
        angle = np.deg2rad(float(spatial["rot_range_deg"]))
        matrix = _rotation_matrix(rng.uniform(-angle, angle, size=3))
    if rng.random() < spatial["p_scale"]:
        scale = float(rng.uniform(*spatial["scale_range"]))
        matrix = matrix / scale
    offset = center - matrix @ center
    ndimage = _ndimage()
    for channel in range(data.shape[0]):
        transformed_data[channel] = ndimage.affine_transform(
            data[channel], matrix, offset=offset, output_shape=mask.shape,
            order=3, mode="constant", cval=0.0, prefilter=True,
        )
    transformed_mask = ndimage.affine_transform(
        transformed_mask, matrix, offset=offset, output_shape=mask.shape,
        order=0, mode="constant", cval=0.0, prefilter=False,
    ) > 0.5

    if spatial["flip_axes"] and rng.random() < spatial["p_flip"]:
        # Axis 2 is the in-plane left-right direction in [z, y, x] arrays.
        transformed_data = np.flip(transformed_data, axis=3).copy()
        transformed_mask = np.flip(transformed_mask, axis=2).copy()

    channels = set(config.jitter_channels)
    intensity = config.intensity
    if rng.random() < intensity["p_gamma"]:
        gamma = float(rng.uniform(*intensity["gamma_range"]))
        for channel in channels:
            transformed_data[channel] = np.sign(transformed_data[channel]) * (
                np.abs(transformed_data[channel]) ** gamma
            )
    if rng.random() < intensity["p_blur"]:
        for channel in channels:
            transformed_data[channel] = ndimage.gaussian_filter(transformed_data[channel], sigma=0.5)
    return transformed_data.astype(np.float32, copy=False), transformed_mask.astype(bool, copy=False)


def resolve_region(mask: np.ndarray, region_mode: str, dilation_vox: int) -> np.ndarray:
    """Resolve the hard transfer support from a lesion mask."""
    mask = np.asarray(mask, dtype=bool)
    if region_mode == "mask":
        return mask.copy()
    if region_mode == "box":
        result = np.zeros_like(mask, dtype=bool)
        coordinates = np.argwhere(mask)
        if len(coordinates):
            mins = coordinates.min(axis=0)
            maxs = coordinates.max(axis=0) + 1
            result[tuple(slice(int(start), int(stop)) for start, stop in zip(mins, maxs))] = True
        return result
    if region_mode == "dilated":
        if dilation_vox < 0:
            raise ValueError("dilation_vox must be non-negative")
        return _ndimage().binary_dilation(mask, iterations=int(dilation_vox))
    raise ValueError(f"Unknown region_mode: {region_mode}")


def feather_alpha(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Create a soft support with a finite two-voxel halo."""
    mask = np.asarray(mask, dtype=bool)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    ndimage = _ndimage()
    alpha = ndimage.gaussian_filter(mask.astype(np.float32), sigma=float(sigma))
    peak = float(alpha[mask].max()) if np.any(mask) else 0.0
    if peak > 0.0:
        alpha /= peak
    alpha = np.clip(alpha, 0.0, 1.0)
    support = ndimage.binary_dilation(mask, iterations=2)
    alpha[~support] = 0.0
    return alpha.astype(np.float32, copy=False)


def anatomy_occupancy_fractions(
    source_mask: np.ndarray,
    anatomy_patch: np.ndarray,
    center: Sequence[int],
    target_zone: str,
    code_map: Optional[Mapping[str, int]] = None,
) -> Tuple[float, float]:
    """Return gland and requested-zone fractions for a candidate placement.

    ``anatomy_patch`` uses the persisted code map from ``anatomy_frame``. The
    optional default keeps this low-level helper convenient in isolation; the
    runtime passes the map recorded by the backfill. The denominator is every
    positive voxel in the full transformed ``source_mask``; voxels clipped by
    the patch therefore count as outside the anatomy rather than disappearing
    from the denominator.
    """
    source_mask = np.asarray(source_mask, dtype=bool)
    anatomy_patch = np.asarray(anatomy_patch)
    if source_mask.ndim != 3 or anatomy_patch.ndim != 3:
        raise ValueError("source_mask and anatomy_patch must be three-dimensional")
    if target_zone not in {"pz", "tz"}:
        raise ValueError(f"Unknown target zone: {target_zone}")
    code_map = code_map or {
        "outside_gland": 0,
        "gland_pz": 1,
        "gland_tz": 2,
        "gland_other": 3,
    }
    required_codes = {"outside_gland", "gland_pz", "gland_tz", "gland_other"}
    if not required_codes <= set(code_map):
        raise ValueError(f"Anatomy code map is missing keys: {sorted(required_codes - set(code_map))}")
    denominator = int(source_mask.sum())
    if denominator == 0:
        return 0.0, 0.0
    slices = placement_slices(center, source_mask.shape, anatomy_patch.shape)
    if slices is None:
        return 0.0, 0.0
    target_index, source_index = slices
    mask_local = source_mask[source_index]
    anatomy_local = anatomy_patch[target_index]
    gland_codes = {int(code_map["gland_pz"]), int(code_map["gland_tz"]), int(code_map["gland_other"])}
    gland = np.isin(anatomy_local, tuple(gland_codes))
    zone = anatomy_local == int(code_map[f"gland_{target_zone}"])
    return (
        float(np.count_nonzero(mask_local & gland)) / denominator,
        float(np.count_nonzero(mask_local & zone)) / denominator,
    )


def blend_footprint_collides(
    seg_patch: np.ndarray,
    source_mask: np.ndarray,
    center: Sequence[int],
    config: LesionTransferConfig,
) -> bool:
    """Return whether the feathered blend footprint overlaps a real lesion."""
    if seg_patch.ndim == 4:
        if seg_patch.shape[0] != 1:
            raise ValueError("seg_patch must have one channel")
        seg_view = seg_patch[0]
    elif seg_patch.ndim == 3:
        seg_view = seg_patch
    else:
        raise ValueError("seg_patch must be [D,H,W] or [1,D,H,W]")
    source_mask = np.asarray(source_mask, dtype=bool)
    region = resolve_region(source_mask, config.region_mode, config.dilation_vox)
    alpha = feather_alpha(region, config.feather_sigma)
    slices = placement_slices(center, source_mask.shape, seg_view.shape)
    if slices is None:
        return False
    target_index, source_index = slices
    occupied = seg_view[target_index]
    alpha_local = alpha[source_index]
    return bool(np.any((occupied > 0) & (alpha_local > 0)))


def paste_lesion(
    data_patch: np.ndarray,
    seg_patch: np.ndarray,
    properties: Dict[str, Any],
    source_data: np.ndarray,
    source_mask: np.ndarray,
    center: Sequence[int],
    new_id: int,
    grade: int,
    source_case_id: str,
    source_instance_id: int,
    config: LesionTransferConfig,
) -> bool:
    """Blend one lesion into a patch and update all per-instance metadata."""
    if int(new_id) <= 0:
        raise ValueError(f"Synthetic instance id must be positive, got {new_id}")
    existing = {int(value) for value in np.unique(seg_patch) if int(value) > 0}
    if int(new_id) in existing:
        raise ValueError(f"Synthetic instance id {new_id} already exists in the target patch")
    source_data = np.asarray(source_data, dtype=np.float32)
    source_mask = np.asarray(source_mask, dtype=bool)
    if data_patch.ndim != 4 or seg_patch.ndim not in (3, 4):
        raise ValueError("data_patch must be [C,D,H,W] and seg_patch must be [D,H,W] or [1,D,H,W]")
    if seg_patch.ndim == 4:
        if seg_patch.shape[0] != 1:
            raise ValueError("seg_patch must have one channel")
        seg_view = seg_patch[0]
    else:
        seg_view = seg_patch
    if source_data.shape[1:] != source_mask.shape or data_patch.shape[0] != source_data.shape[0]:
        raise ValueError("Source and target channel/spatial shapes are incompatible")

    region = resolve_region(source_mask, config.region_mode, config.dilation_vox)
    alpha = feather_alpha(region, config.feather_sigma)
    slices = placement_slices(center, source_mask.shape, seg_view.shape)
    if slices is None:
        return False
    target_index, source_index = slices
    alpha_local = alpha[source_index]
    occupied = seg_view[target_index]
    pad = occupied == -1
    paste_allowed = ~pad
    if np.any((occupied > 0) & (alpha_local > 0)):
        return False
    label_local = source_mask[source_index]
    hard_mask = label_local & paste_allowed
    if not np.any(hard_mask):
        return False
    target_data = data_patch[(slice(None), *target_index)]
    source_values = source_data[(slice(None), *source_index)]
    blend = alpha_local * paste_allowed
    target_data[...] = target_data * (1.0 - blend[None]) + source_values * blend[None]
    data_patch[(slice(None), *target_index)] = target_data
    target_seg = seg_view[target_index]
    target_seg[hard_mask] = int(new_id)
    seg_view[target_index] = target_seg

    instance_mapping = properties.setdefault("instances", {})
    grades = properties.setdefault("grades", {})
    supervised = properties.setdefault("grade_supervised", {})
    sources = properties.setdefault("grade_sources", {})
    instance_mapping[str(new_id)] = 0
    grades[str(new_id)] = int(grade)
    supervised[str(new_id)] = True
    sources[str(new_id)] = f"lesion_transfer:{source_case_id}:{int(source_instance_id)}"
    return True
