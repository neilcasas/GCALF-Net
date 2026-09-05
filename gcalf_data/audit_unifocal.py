"""PI-CAI unifocal linkage recovery audit (PHASE_1_data_pipeline.md Sec 1.1).

For every Pooch25 (binary, ungraded) case, compares the number of connected
components in its lesion mask to the number of comma-separated entries in
that case's marksheet lesion_ISUP field. A case is recoverable only when both
counts are exactly 1 -- the single component's grade is then that lesion's
lesion_ISUP value, unambiguously. Any other case (multiple components, or
multiple marksheet lesions) is left ungraded: this script never infers,
pools, or splits grades across components.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VALID_GRADES = {2, 3, 4, 5}


def marksheet_lesion_entries(marksheet_path: Path) -> Dict[str, List[str]]:
    """Return every raw comma-separated lesion_ISUP entry per case, N/A included."""
    entries = {}
    with marksheet_path.open(newline="") as file:
        for row in csv.DictReader(file):
            case_id = f"{row['patient_id']}_{row['study_id']}"
            entries[case_id] = [item.strip() for item in row["lesion_ISUP"].split(",") if item.strip()]
    return entries


def count_mask_components(mask_path: Path) -> int:
    try:
        import numpy as np
        import SimpleITK as sitk
        from scipy import ndimage
    except ImportError as error:
        raise RuntimeError("audit_unifocal requires numpy, SimpleITK, and scipy from the M0 environment") from error

    image = sitk.ReadImage(str(mask_path))
    array = sitk.GetArrayFromImage(image)
    unsupported = set(np.unique(array).tolist()) - {0, 1}
    if unsupported:
        raise ValueError(f"{mask_path.name}: Pooch25 masks must be binary, found {sorted(unsupported)}")
    _, num_components = ndimage.label(array == 1, structure=np.ones((3, 3, 3)))
    return int(num_components)


def audit_case(case_id: str, mask_path: Path, lesion_entries: List[str]) -> Dict[str, object]:
    num_components = count_mask_components(mask_path)
    num_lesions = len(lesion_entries)
    grade: Optional[int] = None
    grade_source: Optional[str] = None
    grade_supervised = num_components == 1 and num_lesions == 1
    if grade_supervised:
        grade = int(lesion_entries[0])
        if grade not in VALID_GRADES:
            raise ValueError(f"{case_id}: recovered grade {grade} is outside GGG2-5")
        grade_source = "audit_unifocal"
    return {
        "case_id": case_id,
        "num_components": num_components,
        "num_marksheet_lesions": num_lesions,
        "grade_supervised": grade_supervised,
        "grade": grade,
        "grade_source": grade_source,
    }


def run_audit(pooch25_dir: Path, marksheet_path: Path) -> Dict[str, Dict[str, object]]:
    lesion_entries = marksheet_lesion_entries(marksheet_path)
    mask_paths = sorted(p for p in pooch25_dir.glob("*.nii.gz"))
    if not mask_paths:
        raise ValueError(f"No Pooch25 masks found in {pooch25_dir}")

    results = {}
    for mask_path in mask_paths:
        case_id = mask_path.name[: -len(".nii.gz")]
        if case_id not in lesion_entries:
            raise ValueError(f"{case_id}: no marksheet row found")
        results[case_id] = audit_case(case_id, mask_path, lesion_entries[case_id])
    return results


def summarize(results: Dict[str, Dict[str, object]]) -> Tuple[Counter, int]:
    """Return (recovered-grade counts, count of cases left ungraded)."""
    recovered = Counter()
    not_recovered = 0
    for result in results.values():
        if result["grade_supervised"]:
            recovered[result["grade"]] += 1
        else:
            not_recovered += 1
    return recovered, not_recovered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooch25-dir", type=Path, required=True)
    parser.add_argument("--marksheet", type=Path, required=True)
    parser.add_argument(
        "--output-json", type=Path, required=True, help="per-case audit results, consumed by build_labels.py"
    )
    args = parser.parse_args()

    results = run_audit(args.pooch25_dir, args.marksheet)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as file:
        json.dump(results, file, indent=2, sort_keys=True)

    recovered, not_recovered = summarize(results)
    print(f"Audited {len(results)} Pooch25 cases.")
    print("Recovered grade-supervised counts:", {f"GGG{g}": c for g, c in sorted(recovered.items())})
    print(f"Recovered: {sum(recovered.values())} / {len(results)}; left ungraded: {not_recovered}")


if __name__ == "__main__":
    main()
