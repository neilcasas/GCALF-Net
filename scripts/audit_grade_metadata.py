"""Verify canonical grade metadata and write a checksum manifest.

This is deliberately strict for the D3-rev.2 Task2201 labels: it makes a
partial or stale off-instance backup fail before it is used for preprocessing.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_COUNTS = {"GGG2": 253, "GGG3": 104, "GGG4": 37, "GGG5": 47}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(labels_dir):
    labels = sorted(labels_dir.glob("*.json"))
    if not labels:
        raise ValueError(f"No label JSON files found in {labels_dir}")
    counts = Counter()
    total_instances = 0
    files = []
    for path in labels:
        with path.open() as stream:
            record = json.load(stream)
        for key in ("instances", "grades", "grade_supervised"):
            if key not in record:
                raise ValueError(f"{path}: missing {key}")
        instance_ids = set(record["instances"])
        if set(record["grade_supervised"]) != instance_ids:
            raise ValueError(f"{path}: grade_supervised does not cover every instance")
        total_instances += len(instance_ids)
        for instance_id, supervised in record["grade_supervised"].items():
            if not supervised:
                continue
            if instance_id not in record["grades"]:
                raise ValueError(f"{path}: supervised instance {instance_id} has no grade")
            grade = int(record["grades"][instance_id])
            if grade not in range(2, 6):
                raise ValueError(f"{path}: invalid supervised grade {grade}")
            counts[grade] += 1
        files.append({"path": path.name, "sha256": sha256(path), "size": path.stat().st_size})
    observed = {f"GGG{grade}": counts[grade] for grade in range(2, 6)}
    return {
        "schema_version": 1,
        "labels_dir": str(labels_dir),
        "label_files": len(files),
        "total_lesions": total_instances,
        "grade_supervised_lesions": sum(counts.values()),
        "heterogeneous_or_unsupervised_lesions": total_instances - sum(counts.values()),
        "grade_counts": observed,
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        parser.error(f"refusing to overwrite existing manifest: {args.manifest}")
    report = audit(args.labels_dir)
    if report["grade_supervised_lesions"] != 441 or report["heterogeneous_or_unsupervised_lesions"] != 17:
        raise ValueError("Expected 441 supervised and 17 heterogeneous/unsupervised lesions, got "
                         f"{report['grade_supervised_lesions']} and {report['heterogeneous_or_unsupervised_lesions']}")
    if report["grade_counts"] != EXPECTED_COUNTS:
        raise ValueError(f"Expected {EXPECTED_COUNTS}, got {report['grade_counts']}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("label_files", "total_lesions", "grade_supervised_lesions",
                                                    "heterogeneous_or_unsupervised_lesions", "grade_counts")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
