"""Rework PI-CAI's csPCa labels into the single-class detection contract with
per-instance GGG2--5 grade metadata (PHASE_1_data_pipeline.md; ADR 0002 D2/D4).

Every positive lesion is nnDetection detection class 0 ("csPCa"), regardless of
its source annotation. Grade is carried as separate per-instance metadata, never
as the detection class, so ungraded positive lesions (Pooch25) still train the
detector without needing a class to be assigned to.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

VALID_GRADES = {2, 3, 4, 5}
GRADE_NAMES = {grade: f"GGG{grade}" for grade in VALID_GRADES}

# nnunet2nndet's convert_and_save_label assigns each instance a raw class of
# (original mask label value - 1), derived from whichever raw PI-CAI mask value
# it saw for that connected component. human_expert masks carry the grade
# directly as the mask value (2-5, i.e. raw class 1-4); Pooch25's binary mask
# (value 1, i.e. raw class 0) carries no grade information at all.
_HUMAN_EXPERT_RAW_CLASS_TO_GRADE = {grade - 1: grade for grade in VALID_GRADES}  # {1:2, 2:3, 3:4, 4:5}
_POOCH25_RAW_CLASS = 0


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


def _grade_for_instance(
    case_id: str, raw_class: int, audit_results: Dict[str, dict]
) -> Tuple[Optional[int], Optional[str]]:
    """Return (grade, grade_source) for one instance, or (None, None) if it
    stays grade-unsupervised."""
    if raw_class in _HUMAN_EXPERT_RAW_CLASS_TO_GRADE:
        return _HUMAN_EXPERT_RAW_CLASS_TO_GRADE[raw_class], "human_expert_mask"
    if raw_class == _POOCH25_RAW_CLASS:
        audit = audit_results.get(case_id)
        if audit and audit.get("grade_supervised"):
            return int(audit["grade"]), "audit_unifocal"
        return None, None
    raise ValueError(f"{case_id}: unexpected raw instance class {raw_class}")


def inject_grade_metadata(labels_dir: Path, audit_results: Dict[str, dict]) -> Tuple[Counter, int]:
    """Rewrite every raw_splitted/labelsTr/<case_id>.json in place: collapse
    every instance's detection class to 0, and attach grade/grade_source/
    grade_supervised metadata derived from the instance's original
    (pre-collapse) class, per ARCHITECTURE.md's instance schema. Returns
    (grade-supervised counts per grade, count of instances left ungraded).
    """
    json_paths = sorted(labels_dir.glob("*.json"))
    if not json_paths:
        raise ValueError(f"No instance metadata found in {labels_dir}")

    grade_counts: Counter = Counter()
    ungraded_positive_count = 0
    for json_path in json_paths:
        case_id = json_path.stem
        with json_path.open() as file:
            metadata = json.load(file)
        instances = metadata["instances"]

        new_instances: Dict[str, int] = {}
        grades: Dict[str, int] = {}
        grade_sources: Dict[str, str] = {}
        grade_supervised: Dict[str, bool] = {}
        for instance_id, raw_class in instances.items():
            grade, grade_source = _grade_for_instance(case_id, int(raw_class), audit_results)
            new_instances[instance_id] = 0
            if grade is not None:
                grades[instance_id] = grade
                grade_sources[instance_id] = grade_source
                grade_supervised[instance_id] = True
                grade_counts[grade] += 1
            else:
                grade_supervised[instance_id] = False
                ungraded_positive_count += 1

        metadata["instances"] = new_instances
        metadata["grades"] = grades
        metadata["grade_sources"] = grade_sources
        metadata["grade_supervised"] = grade_supervised
        with json_path.open("w") as file:
            json.dump(metadata, file, indent=2)

    return grade_counts, ungraded_positive_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True, help="nnDetection raw_splitted/labelsTr directory")
    parser.add_argument("--audit-json", type=Path, required=True, help="output of audit_unifocal.py")
    parser.add_argument("--marksheet", type=Path, required=True, help="PI-CAI marksheet.csv")
    args = parser.parse_args()

    with args.audit_json.open() as file:
        audit_results = json.load(file)

    grade_counts, ungraded_positive_count = inject_grade_metadata(args.labels_dir, audit_results)
    summary = marksheet_summary(args.marksheet)
    print("Grade-supervised instance counts:", {GRADE_NAMES[g]: c for g, c in sorted(grade_counts.items())})
    print("Ungraded positive instances:", ungraded_positive_count)
    print("Marksheet case_ISUP distribution:", dict(sorted(summary["case_isup"].items())))
    print("Marksheet lesion_ISUP distribution:", dict(sorted(summary["lesion_isup"].items())))


if __name__ == "__main__":
    main()
