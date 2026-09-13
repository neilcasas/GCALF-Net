import csv
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from gcalf_data.audit_unifocal import run_audit, summarize


def _write_binary_mask(path: Path, array: np.ndarray) -> None:
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    sitk.WriteImage(image, str(path))


def _write_marksheet(path: Path, rows) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["patient_id", "study_id", "lesion_ISUP"])
        writer.writeheader()
        for patient_id, study_id, lesion_isup in rows:
            writer.writerow({"patient_id": patient_id, "study_id": study_id, "lesion_ISUP": lesion_isup})


def _single_component_volume() -> np.ndarray:
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[2:4, 2:4, 2:4] = 1
    return array


def _two_component_volume() -> np.ndarray:
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[0:2, 0:2, 0:2] = 1
    array[6:8, 6:8, 6:8] = 1
    return array


def _three_component_volume() -> np.ndarray:
    array = _two_component_volume()
    array[0:2, 6:8, 0:2] = 1
    return array


def test_run_audit_recovers_unifocal_case_with_single_marksheet_lesion(tmp_path):
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10013_1000013.nii.gz", _single_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10013", "1000013", "2")])

    results = run_audit(pooch25_dir, marksheet_path)

    assert results["10013_1000013"]["grade_supervised"] is True
    assert results["10013_1000013"]["grade"] == 2
    assert results["10013_1000013"]["grade_source"] == "audit_unifocal"
    recovered, not_recovered, ambiguous = summarize(results)
    assert recovered == {2: 1}
    assert not_recovered == 0
    assert ambiguous == 0


def test_run_audit_leaves_multi_lesion_marksheet_case_ungraded(tmp_path):
    """Matches real PI-CAI case 10008_1000008: one drawn component, but two
    marksheet grades differ, so the drawn region cannot be assigned a grade."""
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10008_1000008.nii.gz", _single_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10008", "1000008", "3,2")])

    results = run_audit(pooch25_dir, marksheet_path)

    assert results["10008_1000008"]["grade_supervised"] is False
    assert results["10008_1000008"]["grade"] is None
    recovered, not_recovered, ambiguous = summarize(results)
    assert recovered == {}
    assert not_recovered == 1
    assert ambiguous == 1


def test_run_audit_recovers_case_with_ggg1_entry_alongside_single_cspca_lesion(tmp_path):
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10029_1000029.nii.gz", _single_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10029", "1000029", "0,2")])

    results = run_audit(pooch25_dir, marksheet_path)

    assert results["10029_1000029"]["grade_supervised"] is True
    assert results["10029_1000029"]["grade"] == 2
    assert results["10029_1000029"]["valid_grades"] == [2]


def test_run_audit_recovers_multi_component_case_when_every_valid_grade_is_identical(tmp_path):
    """ADR 0002 D3 rev. 2: same-grade entries do not need component linkage."""
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10029_1000029.nii.gz", _three_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10029", "1000029", "2,2")])

    results = run_audit(pooch25_dir, marksheet_path)

    assert results["10029_1000029"]["grade_supervised"] is True
    assert results["10029_1000029"]["grade"] == 2
    recovered, not_recovered, ambiguous = summarize(results)
    assert recovered == {2: 3}
    assert not_recovered == 0
    assert ambiguous == 0


def test_run_audit_leaves_case_with_two_distinct_grades_ungraded_regardless_of_component_count(tmp_path):
    """ADR 0002 D3 rev. 2: matching counts never resolves different grades."""
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10029_1000029.nii.gz", _two_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10029", "1000029", "2,5")])

    results = run_audit(pooch25_dir, marksheet_path)

    assert results["10029_1000029"]["grade_supervised"] is False
    assert results["10029_1000029"]["audit_reason"] == "heterogeneous"


def test_run_audit_leaves_case_with_no_valid_grade_ungraded(tmp_path):
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10000_1000000.nii.gz", _single_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10000", "1000000", "0,1")])

    results = run_audit(pooch25_dir, marksheet_path)

    assert results["10000_1000000"]["grade_supervised"] is False
    assert results["10000_1000000"]["audit_reason"] == "no_valid_grade"


def test_audit_case_records_reason_for_every_unrecovered_case(tmp_path):
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10000_1000000.nii.gz", _single_component_volume())
    _write_binary_mask(pooch25_dir / "10001_1000001.nii.gz", _single_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10000", "1000000", "0,1"), ("10001", "1000001", "2,3")])

    results = run_audit(pooch25_dir, marksheet_path)

    assert results["10000_1000000"]["audit_reason"] == "no_valid_grade"
    assert results["10001_1000001"]["audit_reason"] == "heterogeneous"


def test_run_audit_rejects_non_binary_mask(tmp_path):
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    array = _single_component_volume()
    array[2, 2, 2] = 3
    _write_binary_mask(pooch25_dir / "10013_1000013.nii.gz", array)
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10013", "1000013", "2")])

    with pytest.raises(ValueError, match="must be binary"):
        run_audit(pooch25_dir, marksheet_path)
