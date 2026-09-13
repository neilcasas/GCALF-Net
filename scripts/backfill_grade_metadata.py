"""Propagate canonical raw-label grade metadata into preprocessed properties.

Run without ``--write`` first. The default dry run compares every destination
store with ``raw_splitted/labelsTr`` and reports the prospective grade totals;
only an explicit second invocation with ``--write`` changes pickle files.
"""

import argparse
import json
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple


METADATA_KEYS = ("grades", "grade_supervised", "grade_sources")
STORE_NAMES = ("raw_cropped", "preprocessed_3d", "preprocessed_3dlr1", "props_per_case")


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file)


def _write_pickle(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


def _canonical_metadata(labels_dir: Path) -> Dict[str, Dict[str, object]]:
    metadata = {}
    for path in sorted(labels_dir.glob("*.json")):
        with path.open() as file:
            label = json.load(file)
        if not all(key in label for key in ("instances", *METADATA_KEYS)):
            raise ValueError(f"{path}: missing grade metadata")
        metadata[path.stem] = {key: label[key] for key in METADATA_KEYS}
    if not metadata:
        raise ValueError(f"No label JSON files found in {labels_dir}")
    return metadata


def _per_case_paths(task_dir: Path) -> Dict[str, Dict[str, Path]]:
    stores = {
        "raw_cropped": task_dir / "raw_cropped" / "imagesTr",
        "preprocessed_3d": task_dir / "preprocessed" / "D3V001_3d" / "imagesTr",
        "preprocessed_3dlr1": task_dir / "preprocessed" / "D3V001_3dlr1" / "imagesTr",
    }
    result = {}
    for store_name, directory in stores.items():
        paths = {path.stem: path for path in directory.glob("*.pkl") if not path.stem.endswith("_boxes")}
        if not paths:
            raise ValueError(f"No per-case properties found in {directory}")
        result[store_name] = paths
    return result


def _grade_counts(metadata: Iterable[Mapping[str, object]]) -> Counter:
    counts: Counter = Counter()
    for record in metadata:
        supervised = record["grade_supervised"]
        grades = record["grades"]
        for instance_id, is_supervised in supervised.items():
            if is_supervised:
                counts[int(grades[instance_id])] += 1
    return counts


def _assert_case_sets(canonical: Mapping[str, object], stores: Mapping[str, Mapping[str, Path]], aggregate: Mapping[str, object]) -> None:
    expected = set(canonical)
    for store_name, paths in stores.items():
        if set(paths) != expected:
            raise AssertionError(f"{store_name}: case IDs differ from raw labels")
    if set(aggregate) != expected:
        raise AssertionError("props_per_case: case IDs differ from raw labels")


def backfill(task_dir: Path, write: bool) -> Tuple[Counter, Dict[str, int]]:
    """Compare or propagate metadata, returning canonical totals and diff counts."""
    task_dir = Path(task_dir)
    canonical = _canonical_metadata(task_dir / "raw_splitted" / "labelsTr")
    per_case_paths = _per_case_paths(task_dir)
    aggregate_path = task_dir / "preprocessed" / "properties" / "props_per_case.pkl"
    aggregate = _load_pickle(aggregate_path)
    _assert_case_sets(canonical, per_case_paths, aggregate)

    diff_counts = {store_name: 0 for store_name in STORE_NAMES}
    for store_name, paths in per_case_paths.items():
        for case_id, path in paths.items():
            properties = _load_pickle(path)
            if any(properties.get(key) != canonical[case_id][key] for key in METADATA_KEYS):
                diff_counts[store_name] += 1
                if write:
                    properties.update(canonical[case_id])
                    _write_pickle(path, properties)

    for case_id, properties in aggregate.items():
        if any(properties.get(key) != canonical[case_id][key] for key in METADATA_KEYS):
            diff_counts["props_per_case"] += 1
            if write:
                properties.update(canonical[case_id])
    if write and diff_counts["props_per_case"]:
        _write_pickle(aggregate_path, aggregate)
    return _grade_counts(canonical.values()), diff_counts


def verify(task_dir: Path, audit_json: Path = None) -> Counter:
    """Assert grade metadata agrees across every store and ambiguous cases stay latent."""
    task_dir = Path(task_dir)
    canonical = _canonical_metadata(task_dir / "raw_splitted" / "labelsTr")
    per_case_paths = _per_case_paths(task_dir)
    aggregate = _load_pickle(task_dir / "preprocessed" / "properties" / "props_per_case.pkl")
    _assert_case_sets(canonical, per_case_paths, aggregate)

    for case_id, expected in canonical.items():
        records = [(_load_pickle(paths[case_id]), store_name) for store_name, paths in per_case_paths.items()]
        records.append((aggregate[case_id], "props_per_case"))
        for actual, store_name in records:
            for key in METADATA_KEYS:
                if actual.get(key) != expected[key]:
                    raise AssertionError(f"{case_id}: {store_name} {key} differs from raw labels")

    if audit_json is not None:
        with Path(audit_json).open() as file:
            audit = json.load(file)
        for case_id, result in audit.items():
            if result.get("audit_reason") == "heterogeneous":
                if any(canonical.get(case_id, {}).get("grade_supervised", {}).values()):
                    raise AssertionError(f"{case_id}: ambiguous Pooch25 case gained grade supervision")
    return _grade_counts(canonical.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="propagate changes after reviewing the dry run")
    parser.add_argument("--verify", action="store_true", help="assert agreement after the propagation")
    parser.add_argument("--audit-json", type=Path, help="check heterogeneous audit cases remain unsupervised")
    args = parser.parse_args()

    if args.verify:
        totals = verify(args.task_dir, args.audit_json)
        print("Cross-store grade totals:", {f"GGG{grade}": count for grade, count in sorted(totals.items())})
        return

    totals, diff_counts = backfill(args.task_dir, args.write)
    print("Mode:", "WRITE" if args.write else "DRY_RUN")
    print("Per-store metadata differences:", diff_counts)
    print("Resulting grade-supervised totals:", {f"GGG{grade}": count for grade, count in sorted(totals.items())})


if __name__ == "__main__":
    main()
