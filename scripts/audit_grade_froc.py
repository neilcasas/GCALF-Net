#!/usr/bin/env python3
"""Report IoU-matched generic and grade-specific FROC-style sensitivity."""

import argparse
import csv
import json
from pathlib import Path

from gcalf_eval.harmonized_froc import evaluate_prediction_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument(
        "--case-ids-file",
        type=Path,
        help="One validation case ID per line; defaults to task splits.json fold 0 val",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.iou_threshold <= 1:
        parser.error("--iou-threshold must be in (0, 1]")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output directory: {args.output_dir}")
    if args.case_ids_file:
        case_ids = [line.strip() for line in args.case_ids_file.read_text().splitlines() if line.strip()]
    else:
        with (args.task_dir / "splits.json").open() as file:
            case_ids = json.load(file)[0]["val"]
    summary, _ = evaluate_prediction_dir(
        args.prediction_dir,
        args.task_dir / "raw_splitted" / "labelsTr",
        case_ids,
        iou_threshold=args.iou_threshold,
    )
    args.output_dir.mkdir(parents=True)
    records = summary["records"]
    with (args.output_dir / "grade_froc.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "grade_froc.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
