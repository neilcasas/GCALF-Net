"""Case-level image preprocessing (PHASE_1_data_pipeline.md Sec 1.2; ARCHITECTURE.md Sec 3).

Pipeline, in order: N4 bias correction (T2W only) -> build one reference grid
from the corrected T2W (native in-plane geometry, slice axis resampled to
3.0mm) -> resample ADC/HBV/masks onto that grid in a single pass, linear for
images and nearest-neighbour for masks -> in-plane centre-crop to a fixed
128mm physical field of view, centred on the whole-gland mask's centroid only
(never the lesion -- ADR 0002 D6 item 4) -> pad/crop the slice axis to a
fixed depth of 32 slices about its own geometric centre. No normalization
happens here: nnDetection applies its own `nonCT` per-case/per-modality
normalization once, after its planned resampling (ARCHITECTURE.md Sec 3 item 8).

Masks always use nearest-neighbour interpolation, never linear -- a
linear-interpolated {0,2,3,4,5}-valued mask would fabricate nonexistent grade
values. Images use linear, never cubic/B-spline -- B-spline overshoots at
edges and can emit out-of-range or negative values on the quantitative
ADC/HBV maps.
"""

from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

TARGET_FOV_MM = 128.0
TARGET_SLICE_SPACING = 3.0
TARGET_NUM_SLICES = 32


def n4_bias_correct(image: sitk.Image, shrink_factor: int = 4) -> sitk.Image:
    """N4 bias-field correction. Applied to T2W only -- ADC/HBV are quantitative
    maps and N4 would distort their values (ARCHITECTURE.md Sec 3 item 1).

    The pinned SimpleITK 2.0.2 predates ``GetLogBiasFieldAsImage``, so the usual
    shrink-then-reconstruct speedup is done manually instead: N4 runs on a
    `shrink_factor`-downsampled copy (`shrink_factor**3` fewer voxels, far
    faster), the low-resolution bias field is recovered as the ratio of the
    shrunk image to its N4-corrected version, then that field is upsampled
    back to full resolution and divided out of the original image."""
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    shrunk_image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    shrunk_mask = sitk.Shrink(mask, [shrink_factor] * image.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    shrunk_corrected = corrector.Execute(shrunk_image, shrunk_mask)

    epsilon = 1e-6
    shrunk_bias_field = sitk.Cast((shrunk_image + epsilon) / (shrunk_corrected + epsilon), sitk.sitkFloat32)
    bias_field = sitk.Resample(shrunk_bias_field, image, sitk.Transform(), sitk.sitkLinear, 1.0)
    corrected = image / bias_field
    return sitk.Cast(corrected, sitk.sitkFloat32)


def build_target_reference(t2w: sitk.Image, target_slice_spacing: float = TARGET_SLICE_SPACING) -> sitk.Image:
    """Build the common reference grid from the (already N4-corrected) `t2w`:
    its native in-plane size/spacing/origin/direction are preserved exactly,
    and only the slice axis is resampled to `target_slice_spacing` mm
    (ARCHITECTURE.md Sec 3 item 2). Folding the slice-spacing decision into
    this grid costs ADC/HBV one interpolation instead of two."""
    original_spacing = t2w.GetSpacing()  # (x, y, z)
    original_size = t2w.GetSize()  # (x, y, z)
    new_spacing = (original_spacing[0], original_spacing[1], target_slice_spacing)
    new_size_z = max(1, int(round(original_size[2] * original_spacing[2] / target_slice_spacing)))
    new_size = (original_size[0], original_size[1], new_size_z)

    reference = sitk.Image(new_size, t2w.GetPixelID())
    reference.SetSpacing(new_spacing)
    reference.SetOrigin(t2w.GetOrigin())
    reference.SetDirection(t2w.GetDirection())
    return reference


def resample_to_reference(image: sitk.Image, reference: sitk.Image, is_label: bool) -> sitk.Image:
    """Resample `image` onto `reference`'s grid (size/spacing/origin/direction).
    Nearest-neighbour for masks; linear -- never cubic/B-spline -- for scans,
    since B-spline overshoots at edges and can emit out-of-range or negative
    values on the quantitative ADC/HBV maps (ARCHITECTURE.md Sec 3 item 2)."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    output = resampler.Execute(image)
    return sitk.Cast(output, sitk.sitkUInt8 if is_label else sitk.sitkFloat32)


def whole_gland_centroid_index(gland_mask: sitk.Image) -> Tuple[float, float, float]:
    """Return the (z, y, x) continuous voxel-index centroid of the gland mask's
    foreground, in the mask's own array frame."""
    array = sitk.GetArrayFromImage(gland_mask)  # (z, y, x)
    if not np.any(array):
        raise ValueError("Whole-gland mask is empty; cannot center the crop")
    return ndimage.center_of_mass(array > 0)


def resolve_crop_center(whole_gland: sitk.Image) -> Tuple[Tuple[float, float], str]:
    """Return ((center_y, center_x), strategy) for the in-plane crop, derived
    only from the whole-gland mask's centroid (ARCHITECTURE.md Sec 3 items
    3-4; ADR 0002 D6 items 3-4). The lesion mask is never consulted here: it
    is unavailable at inference, and a lesion-guided crop leaks target
    location into validation/test preprocessing -- rejected outright as
    "Lesion-guided union/lesion-only crop" in ADR 0002. An empty gland mask
    falls back to the T2W geometric centre; the caller is responsible for
    recording any non-"gland" strategy."""
    try:
        _, center_y, center_x = whole_gland_centroid_index(whole_gland)
        return (center_y, center_x), "gland"
    except ValueError:
        _, height, width = sitk.GetArrayFromImage(whole_gland).shape
        return ((height - 1) / 2, (width - 1) / 2), "t2w_geometric_centre"


def fov_mm_to_inplane_size(spacing_yx: Tuple[float, float], fov_mm: float = TARGET_FOV_MM) -> Tuple[int, int]:
    """Convert a physical in-plane field of view (mm) to a voxel crop size
    (size_y, size_x), given the image's own native in-plane spacing. Voxel
    extent varies per case -- this is the point: a fixed voxel count would
    span 60-160mm of anatomy depending on scanner (ARCHITECTURE.md Sec 3,
    "why a physical FOV and not 256 voxels")."""
    spacing_y, spacing_x = spacing_yx
    return int(round(fov_mm / spacing_y)), int(round(fov_mm / spacing_x))


def _crop_pad_bounds(center: float, extent: int, target: int) -> Tuple[int, int, int, int]:
    start = int(round(center - target / 2))
    end = start + target
    pad_before = max(0, -start)
    pad_after = max(0, end - extent)
    start = max(0, start)
    end = min(extent, end)
    return start, end, pad_before, pad_after


def center_crop_or_pad_inplane(
    image: sitk.Image, center_yx: Tuple[float, float], size_yx: Tuple[int, int]
) -> sitk.Image:
    """Crop/pad the in-plane (Y, X) axes to `size_yx`, centered at `center_yx`
    (continuous voxel-index (y, x)). A pure index-space operation -- no
    resampling, so native in-plane spacing is preserved exactly."""
    array = sitk.GetArrayFromImage(image)  # (z, y, x)
    depth, height, width = array.shape
    center_y, center_x = center_yx
    size_y, size_x = size_yx

    y0, y1, y_pad_before, y_pad_after = _crop_pad_bounds(center_y, height, size_y)
    x0, x1, x_pad_before, x_pad_after = _crop_pad_bounds(center_x, width, size_x)

    cropped = array[:, y0:y1, x0:x1]
    padded = np.pad(
        cropped,
        ((0, 0), (y_pad_before, y_pad_after), (x_pad_before, x_pad_after)),
        mode="constant",
        constant_values=0,
    )
    assert padded.shape == (depth, size_y, size_x)

    result = sitk.GetImageFromArray(padded)
    result.SetSpacing(image.GetSpacing())
    result.SetDirection(image.GetDirection())

    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    origin = np.array(image.GetOrigin())
    # sitk axis order is (x, y, z); the new array's voxel 0 along (x, y) maps to
    # old index (x0 - x_pad_before, y0 - y_pad_before) in the old grid.
    offset_index = np.array([x0 - x_pad_before, y0 - y_pad_before, 0.0])
    result.SetOrigin(tuple(origin + direction @ (offset_index * spacing)))
    return result


def pad_or_crop_depth(image: sitk.Image, target_slices: int = TARGET_NUM_SLICES) -> sitk.Image:
    """Pad/crop the z axis to exactly `target_slices`, centered. A pure index
    operation -- spacing is untouched. Centring is purely geometric (never
    gland-derived); the adopted rule per ARCHITECTURE.md Sec 3 item 5."""
    array = sitk.GetArrayFromImage(image)  # (z, y, x)
    depth = array.shape[0]
    if depth == target_slices:
        result_array, z_offset = array, 0
    elif depth > target_slices:
        z_offset = (depth - target_slices) // 2
        result_array = array[z_offset : z_offset + target_slices]
    else:
        pad_before = (target_slices - depth) // 2
        pad_after = target_slices - depth - pad_before
        result_array = np.pad(array, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")
        z_offset = -pad_before

    result = sitk.GetImageFromArray(result_array)
    result.SetSpacing(image.GetSpacing())
    result.SetDirection(image.GetDirection())

    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    origin = np.array(image.GetOrigin())
    result.SetOrigin(tuple(origin + direction @ (np.array([0.0, 0.0, z_offset]) * spacing)))
    return result


def mask_retention_stats(mask: sitk.Image) -> Dict[str, int]:
    """Return foreground voxel and connected-component counts for a lesion mask."""
    array = sitk.GetArrayFromImage(mask)
    foreground = array > 0
    _, components = ndimage.label(foreground, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return {"voxels": int(foreground.sum()), "components": int(components)}


def preprocess_case_with_retention(
    t2w: sitk.Image,
    adc: sitk.Image,
    hbv: sitk.Image,
    lesion_mask: sitk.Image,
    whole_gland: sitk.Image,
    target_fov_mm: float = TARGET_FOV_MM,
    target_slice_spacing: float = TARGET_SLICE_SPACING,
    target_num_slices: int = TARGET_NUM_SLICES,
) -> Tuple[sitk.Image, sitk.Image, sitk.Image, sitk.Image, str, Dict[str, Dict[str, int]]]:
    """Run the full preprocessing pipeline for one case. `lesion_mask` may be an
    all-zero mask for csPCa-negative cases -- every case has an annotation file
    (human_expert or Pooch25), so this is never None; it is resampled and
    cropped like every other volume but never used to choose the crop.
    Returns (t2w, adc, hbv, lesion_mask, crop_strategy), all sharing identical
    geometry. `crop_strategy` is "gland" (the default) or
    "t2w_geometric_centre" (only when the whole-gland mask is empty) -- see
    `resolve_crop_center`; the caller should log any non-"gland" case. No
    normalization is performed here (ARCHITECTURE.md Sec 3 item 8). The
    returned retention dictionary captures the lesion-mask state before
    preprocessing, after common-grid resampling, after the in-plane crop, and
    after depth handling. It is post-crop QC only and never influences crop
    coordinates."""
    retention = {"source": mask_retention_stats(lesion_mask)}
    t2w = n4_bias_correct(t2w)

    reference = build_target_reference(t2w, target_slice_spacing)
    t2w = resample_to_reference(t2w, reference, is_label=False)
    adc = resample_to_reference(adc, reference, is_label=False)
    hbv = resample_to_reference(hbv, reference, is_label=False)
    whole_gland = resample_to_reference(whole_gland, reference, is_label=True)
    lesion_mask = resample_to_reference(lesion_mask, reference, is_label=True)
    retention["resampled"] = mask_retention_stats(lesion_mask)

    (center_y, center_x), crop_strategy = resolve_crop_center(whole_gland)
    spacing_x, spacing_y, _ = t2w.GetSpacing()  # in-plane spacing is untouched by the reference resample
    size_yx = fov_mm_to_inplane_size((spacing_y, spacing_x), target_fov_mm)

    t2w = center_crop_or_pad_inplane(t2w, (center_y, center_x), size_yx)
    adc = center_crop_or_pad_inplane(adc, (center_y, center_x), size_yx)
    hbv = center_crop_or_pad_inplane(hbv, (center_y, center_x), size_yx)
    lesion_mask = center_crop_or_pad_inplane(lesion_mask, (center_y, center_x), size_yx)
    retention["inplane"] = mask_retention_stats(lesion_mask)

    t2w = pad_or_crop_depth(t2w, target_num_slices)
    adc = pad_or_crop_depth(adc, target_num_slices)
    hbv = pad_or_crop_depth(hbv, target_num_slices)
    lesion_mask = pad_or_crop_depth(lesion_mask, target_num_slices)
    retention["final"] = mask_retention_stats(lesion_mask)

    # Nearest-neighbour is used for every mask resample above; a linear-interpolated
    # {0,2,3,4,5}-valued mask would fabricate nonexistent grade values, so verify the
    # domain explicitly rather than trust the interpolator choice silently.
    mask_values = set(np.unique(sitk.GetArrayFromImage(lesion_mask)).tolist())
    unsupported = mask_values - {0, 1, 2, 3, 4, 5}
    if unsupported:
        raise ValueError(f"Lesion mask contains unsupported values after preprocessing: {sorted(unsupported)}")

    return t2w, adc, hbv, lesion_mask, crop_strategy, retention


def preprocess_case(
    t2w: sitk.Image,
    adc: sitk.Image,
    hbv: sitk.Image,
    lesion_mask: sitk.Image,
    whole_gland: sitk.Image,
    target_fov_mm: float = TARGET_FOV_MM,
    target_slice_spacing: float = TARGET_SLICE_SPACING,
    target_num_slices: int = TARGET_NUM_SLICES,
) -> Tuple[sitk.Image, sitk.Image, sitk.Image, sitk.Image, str]:
    """Run preprocessing while preserving the historical five-value API.

    Builders that must audit post-crop lesion retention use
    :func:`preprocess_case_with_retention` instead.
    """
    result = preprocess_case_with_retention(
        t2w,
        adc,
        hbv,
        lesion_mask,
        whole_gland,
        target_fov_mm=target_fov_mm,
        target_slice_spacing=target_slice_spacing,
        target_num_slices=target_num_slices,
    )
    return result[:5]
