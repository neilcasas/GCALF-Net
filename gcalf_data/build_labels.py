"""Convert PI-CAI's granular csPCa labels to contiguous GGG2--5 labels."""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable


SOURCE_LABELS = {0, 2, 3, 4, 5}
GGG_LABELS = {1: "GGG2", 2: "GGG3", 3: "GGG4", 4: "GGG5"}


def remap_source_label(label: int) -> int:
    """Map PI-CAI's mask value to a contiguous nnU-Net foreground value."""
    if label == 0:
        return 0
    if label not in SOURCE_LABELS:
        raise ValueError(f"Unsupported PI-CAI label value: {label}")
    return label - 1


def parse_lesion_isup(value: str) -> Iterable[int]:
    """Parse the comma-separated lesion_ISUP field, ignoring ungraded lesions."""
    for item in value.split(","):
        item = item.strip()
        if item and item != "N/A":
            yield int(item)


def marksheet_summary(marksheet_path: Path) -> Dict[str, Counter]:
    """Return case- and lesion-level grade distributions for reporting only."""
    case_isup = Counter()
    lesion_isup = Counter()
    with marksheet_path.open(newline="") as file:
        for row in csv.DictReader(file):
            case_isup[int(row["case_ISUP"])] += 1
            lesion_isup.update(parse_lesion_isup(row["lesion_ISUP"]))
    return {"case_isup": case_isup, "lesion_isup": lesion_isup}


def remap_label_images(labels_dir: Path) -> Counter:
    """Rewrite PI-CAI masks from ``{0, 2, 3, 4, 5}`` to ``{0, 1, 2, 3, 4}``."""
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError("build_labels requires numpy and SimpleITK from the M0 environment") from error

    counts = Counter()
    label_paths = sorted(labels_dir.glob("*.nii.gz"))
    if not label_paths:
        raise ValueError(f"No label images found in {labels_dir}")

    for label_path in label_paths:
        image = sitk.ReadImage(str(label_path))
        source = sitk.GetArrayFromImage(image)
        found = set(np.unique(source).tolist())
        unsupported = found - SOURCE_LABELS
        if unsupported:
            raise ValueError(f"{label_path.name} contains unsupported labels: {sorted(unsupported)}")

        remapped = np.zeros_like(source, dtype=np.uint8)
        for source_label in sorted(SOURCE_LABELS - {0}):
            remapped[source == source_label] = remap_source_label(source_label)
            counts[source_label] += int((source == source_label).sum())

        result = sitk.GetImageFromArray(remapped)
        result.CopyInformation(image)
        sitk.WriteImage(result, str(label_path))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True, help="nnU-Net labelsTr directory")
    parser.add_argument("--marksheet", type=Path, required=True, help="PI-CAI marksheet.csv")
    args = parser.parse_args()

    voxel_counts = remap_label_images(args.labels_dir)
    summary = marksheet_summary(args.marksheet)
    print("Remapped lesion-label voxels:", dict(sorted(voxel_counts.items())))
    print("Marksheet case_ISUP distribution:", dict(sorted(summary["case_isup"].items())))
    print("Marksheet lesion_ISUP distribution:", dict(sorted(summary["lesion_isup"].items())))


if __name__ == "__main__":
    main()
