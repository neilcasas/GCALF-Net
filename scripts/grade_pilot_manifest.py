"""Create the seed-2026 patient-disjoint grade-pilot split manifest."""
import argparse
from pathlib import Path

from gcalf_eval.grade_pilot import make_split_manifest
from nndet.io.load import load_pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-splits", type=Path, required=True, help="Source fold's splits.pkl")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Preprocessed imagesTr directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    train_cases = load_pickle(args.fold_splits)[args.fold]["train"]
    case_grades = {}
    for case in train_cases:
        properties = load_pickle(args.dataset_dir / (str(case) + ".pkl"))
        grades = properties.get("grades", {})
        supervised = properties.get("grade_supervised", {})
        case_grades[str(case)] = [int(grades[key]) for key, value in supervised.items() if value]
    make_split_manifest(case_grades, args.output, seed=2026)


if __name__ == "__main__":
    main()
