"""Unpack nnDetection ``.npz`` cases to loader-ready ``.npy`` arrays.

This deliberately does not crop, resample, normalize, or alter labels.  It is
the reversible-on-regeneration storage expansion required by nnDetection's
real-case data loader after preprocessing has produced compressed cases.
"""

import argparse
from pathlib import Path

from nndet.io.load import unpack_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_dir", type=Path, help="Preprocessed <data_identifier>/imagesTr directory")
    parser.add_argument("--processes", type=int, default=8, help="Parallel CPU workers (default: 8)")
    args = parser.parse_args()

    if args.processes < 1:
        raise ValueError("--processes must be at least 1")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)
    if not any(args.images_dir.glob("*.npz")):
        raise RuntimeError(f"No compressed preprocessed cases found in {args.images_dir}")

    # nnDetection's pinned 2021 API takes the overwrite flag positionally.
    unpack_dataset(args.images_dir, args.processes, False)
    missing = [case.with_suffix(".npy") for case in args.images_dir.glob("*.npz") if not case.with_suffix(".npy").is_file()]
    if missing:
        raise RuntimeError(f"Unpacking did not produce {len(missing)} expected arrays; first: {missing[0]}")
    print(f"Loader-ready arrays verified in {args.images_dir}")


if __name__ == "__main__":
    main()
