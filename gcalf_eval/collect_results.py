"""Collect four-arm GCALF evaluation outputs into a raw and baseline-delta table."""

import argparse
import csv
import json
from pathlib import Path


CONFIGS = ("baseline", "lff_only", "caf_only", "gcalf_full")
SCALAR_METRICS = ("picai_score", "auroc", "lesion_ap", "grade_weighted_f1")


def _config_name(path: Path) -> str | None:
    """Map the conventional experiment-directory name to an ablation arm."""
    name = path.parent.name.lower()
    if "baseline" in name:
        return "baseline"
    if "lff" in name:
        return "lff_only"
    if "caf" in name and "full" not in name:
        return "caf_only"
    if "full" in name or "gcalf" in name:
        return "gcalf_full"
    return None


def collect(experiments_dir: Path) -> list[dict]:
    """Read every metrics.csv recursively and emit one row per config/fold result."""
    rows = []
    for metrics_path in sorted(experiments_dir.glob("**/metrics.csv")):
        config = _config_name(metrics_path)
        if config is None:
            continue
        with metrics_path.open(newline="") as stream:
            for metric_row in csv.DictReader(stream):
                row = {"config": config, "source": str(metrics_path)}
                row.update(metric_row)
                grade_path = metrics_path.with_name("grade_metrics.json")
                if grade_path.is_file():
                    with grade_path.open() as stream:
                        grade = json.load(stream)
                    row["grade_per_class_sensitivity"] = json.dumps(
                        grade.get("grade_per_class_sensitivity", {}), sort_keys=True
                    )
                rows.append(row)
    return rows


def write_comparison(rows: list[dict], output_path: Path) -> None:
    baseline = next((row for row in rows if row["config"] == "baseline"), None)
    if baseline is None:
        raise ValueError("A baseline metrics.csv is required for delta computation")
    fieldnames = ["config", "source", "fold", "split", "num_cases", "num_lesions", *SCALAR_METRICS]
    fieldnames += [f"delta_{metric}" for metric in SCALAR_METRICS]
    fieldnames += ["grade_per_class_sensitivity"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            result = dict(row)
            for metric in SCALAR_METRICS:
                try:
                    result[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
                except (KeyError, TypeError, ValueError):
                    result[f"delta_{metric}"] = ""
            writer.writerow(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=Path("gcalf_experiments"))
    parser.add_argument("--output", type=Path, default=Path("gcalf_experiments/collected_results.csv"))
    args = parser.parse_args()
    rows = collect(args.experiments_dir)
    if not rows:
        raise ValueError(f"No metrics.csv files found below {args.experiments_dir}")
    write_comparison(rows, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
