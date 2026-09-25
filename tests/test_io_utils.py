from pathlib import Path

from nndet.io.utils import get_np_paths_from_dir


def test_npz_cases_take_precedence_over_npy_anatomy_sidecars(tmp_path):
    (tmp_path / "case_000.npz").touch()
    (tmp_path / "case_000_anatomy.npy").touch()

    paths = get_np_paths_from_dir(tmp_path)

    assert [Path(path).name for path in paths] == ["case_000"]


def test_npy_fallback_ignores_segmentation_and_anatomy_sidecars(tmp_path):
    (tmp_path / "case_000.npy").touch()
    (tmp_path / "case_000_seg.npy").touch()
    (tmp_path / "case_000_anatomy.npy").touch()

    paths = get_np_paths_from_dir(tmp_path)

    assert [Path(path).name for path in paths] == ["case_000"]
