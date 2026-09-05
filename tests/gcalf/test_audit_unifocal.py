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
    recovered, not_recovered = summarize(results)
    assert recovered == {2: 1}
    assert not_recovered == 0


def test_run_audit_leaves_multi_lesion_marksheet_case_ungraded(tmp_path):
    """Matches real PI-CAI case 10008_1000008: one drawn component, but two
    marksheet lesion entries -- ambiguous which grade the drawn region is."""
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10008_1000008.nii.gz", _single_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10008", "1000008", "3,2")])

    results = run_audit(pooch25_dir, marksheet_path)

    assert results["10008_1000008"]["grade_supervised"] is False
    assert results["10008_1000008"]["grade"] is None
    recovered, not_recovered = summarize(results)
    assert recovered == {}
    assert not_recovered == 1


def test_run_audit_leaves_multi_component_case_ungraded(tmp_path):
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10029_1000029.nii.gz", _two_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10029", "1000029", "2")])

    results = run_audit(pooch25_dir, marksheet_path)

    assert results["10029_1000029"]["grade_supervised"] is False
    assert results["10029_1000029"]["num_components"] == 2


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


def test_run_audit_rejects_out_of_domain_recovered_grade(tmp_path):
    pooch25_dir = tmp_path / "Pooch25"
    pooch25_dir.mkdir()
    _write_binary_mask(pooch25_dir / "10000_1000000.nii.gz", _single_component_volume())
    marksheet_path = tmp_path / "marksheet.csv"
    _write_marksheet(marksheet_path, [("10000", "1000000", "1")])  # ISUP1/GGG1, outside GGG2-5

    with pytest.raises(ValueError, match="outside GGG2-5"):
        run_audit(pooch25_dir, marksheet_path)
