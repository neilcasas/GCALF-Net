"""PI-CAI Pooch25 grade-linkage recovery audit (PHASE_1_data_pipeline.md Sec 1.1).

For every Pooch25 (binary, ungraded) case, retain its raw marksheet
``lesion_ISUP`` entries for provenance and restrict them to GGG2--5. A case
is recoverable when it has at least one valid entry and every valid entry has
the same grade. That grade applies to every csPCa component without assigning
marksheet lesions to components. Cases with distinct valid grades, or no
valid grade, remain grade-unsupervised.
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


def valid_lesion_grades(lesion_entries: List[str]) -> List[int]:
    """Return the GGG2--5 entries, discarding GGG0/1 and non-integer values."""
    valid_grades = []
    for entry in lesion_entries:
        try:
            grade = int(entry)
        except ValueError:
            continue
        if grade in VALID_GRADES:
            valid_grades.append(grade)
    return valid_grades


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
    valid_grades = valid_lesion_grades(lesion_entries)
    grade: Optional[int] = None
    grade_source: Optional[str] = None
    if not valid_grades:
        audit_reason = "no_valid_grade"
    elif len(set(valid_grades)) > 1:
        audit_reason = "heterogeneous"
    else:
        audit_reason = "homogeneous"
    grade_supervised = audit_reason == "homogeneous"
    if grade_supervised:
        grade = valid_grades[0]
        grade_source = "audit_unifocal"
    return {
        "case_id": case_id,
        "num_components": num_components,
        "num_marksheet_lesions": num_lesions,
        "valid_grades": valid_grades,
        "audit_reason": audit_reason,
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


def summarize(results: Dict[str, Dict[str, object]]) -> Tuple[Counter, int, int]:
    """Return recovered lesion counts, ungraded cases, and ambiguous cases."""
    recovered = Counter()
    not_recovered = 0
    ambiguous = 0
    for result in results.values():
        if result["grade_supervised"]:
            recovered[result["grade"]] += result["num_components"]
        else:
            not_recovered += 1
            if result["audit_reason"] == "heterogeneous":
                ambiguous += 1
    return recovered, not_recovered, ambiguous


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

    recovered, not_recovered, ambiguous = summarize(results)
    print(f"Audited {len(results)} Pooch25 cases.")
    print("Recovered grade-supervised counts:", {f"GGG{g}": c for g, c in sorted(recovered.items())})
    print(f"Recovered: {sum(recovered.values())} / {len(results)}; left ungraded: {not_recovered}")
    print(f"Genuinely ambiguous cases left ungraded: {ambiguous}")


if __name__ == "__main__":
    main()
