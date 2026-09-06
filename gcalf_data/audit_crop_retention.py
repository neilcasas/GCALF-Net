"""Post-crop lesion-retention audit for the PI-CAI M1 task.

The gland-centred crop is deliberately target-independent. This module records
what happened to every source-positive lesion afterwards and makes an empty
post-crop lesion an explicit exclusion rather than a silently relabelled
benign case (ADR 0002 D6 item 7).
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set


AUDIT_FILENAME = "crop_retention.csv"
EXCLUSIONS_FILENAME = "excluded_cases.json"
STAGES = ("source", "resampled", "inplane", "final")
STATUSES = {"negative", "fully_retained", "partially_clipped", "excluded_no_retained_voxels"}
FIELDNAMES = (
    "case_id",
    "crop_strategy",
    "status",
    "exclusion_reason",
    *(f"{stage}_{metric}" for stage in STAGES for metric in ("voxels", "components")),
)


def make_record(case_id: str, crop_strategy: str, retention: Mapping[str, Mapping[str, int]]) -> Dict[str, object]:
    """Turn stage-wise mask statistics into one auditable case record."""
    missing = set(STAGES) - set(retention)
    if missing:
        raise ValueError(f"{case_id}: missing retention stages {sorted(missing)}")

    record: Dict[str, object] = {"case_id": case_id, "crop_strategy": crop_strategy}
    for stage in STAGES:
        stats = retention[stage]
        for metric in ("voxels", "components"):
            value = int(stats[metric])
            if value < 0:
                raise ValueError(f"{case_id}: {stage}_{metric} cannot be negative")
            record[f"{stage}_{metric}"] = value

    if record["source_voxels"] == 0:
        record["status"] = "negative"
        record["exclusion_reason"] = ""
    elif record["final_voxels"] == 0:
        record["status"] = "excluded_no_retained_voxels"
        record["exclusion_reason"] = "gland_centred_crop_retained_no_lesion_voxels"
    elif (
        record["inplane_voxels"] < record["resampled_voxels"]
        or record["final_voxels"] < record["inplane_voxels"]
        or record["inplane_components"] < record["resampled_components"]
        or record["final_components"] < record["inplane_components"]
    ):
        record["status"] = "partially_clipped"
        record["exclusion_reason"] = ""
    else:
        record["status"] = "fully_retained"
        record["exclusion_reason"] = ""
    return record


def excluded_case_ids(records: Iterable[Mapping[str, object]]) -> Set[str]:
    """Return source-positive cases that must not enter any experiment arm."""
    return {
        str(record["case_id"])
        for record in records
        if record["status"] == "excluded_no_retained_voxels"
    }


def write_audit(records: Sequence[Mapping[str, object]], task_dir: Path) -> Dict[str, Path]:
    """Write the generated audit and its explicit exclusion manifest."""
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    ordered_records = sorted(records, key=lambda record: str(record["case_id"]))
    case_ids = [str(record["case_id"]) for record in ordered_records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Crop-retention audit contains duplicate case IDs")

    audit_path = task_dir / AUDIT_FILENAME
    with audit_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered_records)

    excluded = sorted(excluded_case_ids(ordered_records))
    exclusions_path = task_dir / EXCLUSIONS_FILENAME
    exclusions_path.write_text(
        json.dumps(
            {
                "reason": "gland-centred 128 mm crop retained no lesion voxels",
                "case_ids": excluded,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {"audit": audit_path, "exclusions": exclusions_path}


def load_audit(task_dir: Path) -> List[Dict[str, object]]:
    """Load and type-check a generated crop-retention audit."""
    audit_path = Path(task_dir) / AUDIT_FILENAME
    if not audit_path.is_file():
        raise ValueError(f"Missing crop-retention audit: {audit_path}")
    with audit_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != FIELDNAMES:
            raise ValueError(f"{audit_path}: unexpected columns")
        records = []
        for row in reader:
            record: Dict[str, object] = {
                "case_id": row["case_id"],
                "crop_strategy": row["crop_strategy"],
                "status": row["status"],
                "exclusion_reason": row["exclusion_reason"],
            }
            for stage in STAGES:
                for metric in ("voxels", "components"):
                    key = f"{stage}_{metric}"
                    record[key] = int(row[key])
            records.append(record)
    return records


def load_exclusions(task_dir: Path) -> Set[str]:
    """Load the declared exclusions that accompany a generated audit."""
    exclusions_path = Path(task_dir) / EXCLUSIONS_FILENAME
    if not exclusions_path.is_file():
        raise ValueError(f"Missing crop-exclusion manifest: {exclusions_path}")
    with exclusions_path.open() as file:
        manifest = json.load(file)
    case_ids = manifest.get("case_ids")
    if not isinstance(case_ids, list) or not all(isinstance(case_id, str) for case_id in case_ids):
        raise ValueError(f"{exclusions_path}: case_ids must be a string list")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{exclusions_path}: duplicate case IDs")
    return set(case_ids)


def validate_audit(
    task_dir: Path,
    raw_case_ids: Set[str],
    split_case_ids: Set[str],
    expected_source_positive_count: int = None,
) -> List[Dict[str, object]]:
    """Assert that crop loss is explicit and exclusions reached raw data and folds."""
    records = load_audit(task_dir)
    record_ids = [str(record["case_id"]) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise AssertionError("Crop-retention audit contains duplicate case IDs")

    positives = []
    for record in records:
        status = record["status"]
        if status not in STATUSES:
            raise AssertionError(f"{record['case_id']}: unknown crop-retention status {status}")
        source_voxels = int(record["source_voxels"])
        final_voxels = int(record["final_voxels"])
        if source_voxels == 0:
            if status != "negative":
                raise AssertionError(f"{record['case_id']}: negative source mask has status {status}")
            continue
        positives.append(record)
        if final_voxels == 0 and status != "excluded_no_retained_voxels":
            raise AssertionError(f"{record['case_id']}: source-positive case lost all lesion voxels")
        if final_voxels > 0 and status == "excluded_no_retained_voxels":
            raise AssertionError(f"{record['case_id']}: retained lesion is incorrectly excluded")

    if expected_source_positive_count is not None and len(positives) != expected_source_positive_count:
        raise AssertionError(
            f"Expected {expected_source_positive_count} source-positive cases, found {len(positives)}"
        )

    exclusions = load_exclusions(task_dir)
    audited_exclusions = excluded_case_ids(records)
    if exclusions != audited_exclusions:
        raise AssertionError("Crop-exclusion manifest does not match the retention audit")
    if exclusions & raw_case_ids:
        raise AssertionError(f"Excluded cases remain in raw task data: {sorted(exclusions & raw_case_ids)}")
    if exclusions & split_case_ids:
        raise AssertionError(f"Excluded cases remain in splits: {sorted(exclusions & split_case_ids)}")

    retained_positive_ids = {str(record["case_id"]) for record in positives} - exclusions
    missing = retained_positive_ids - raw_case_ids
    if missing:
        raise AssertionError(f"Retained source-positive cases missing from raw task data: {sorted(missing)}")
    return records


def summarize(records: Iterable[Mapping[str, object]]) -> Counter:
    """Count audit records by their final retention disposition."""
    return Counter(str(record["status"]) for record in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--expected-source-positive-count", type=int)
    args = parser.parse_args()

    records = load_audit(args.task_dir)
    print("Crop-retention summary:", dict(sorted(summarize(records).items())))
    print("Declared exclusions:", sorted(load_exclusions(args.task_dir)))
    if args.expected_source_positive_count is not None:
        positive_count = sum(int(record["source_voxels"]) > 0 for record in records)
        if positive_count != args.expected_source_positive_count:
            raise SystemExit(
                f"Expected {args.expected_source_positive_count} source-positive cases, found {positive_count}"
            )


if __name__ == "__main__":
    main()
