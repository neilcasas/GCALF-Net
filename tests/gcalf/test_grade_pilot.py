import numpy as np
import pytest

from gcalf_eval.grade_pilot import fit_temperature, make_split_manifest, temperature_scale


def test_manifest_is_patient_disjoint_and_has_all_grades(tmp_path):
    grades = {f"{patient}_study": [2, 3, 4, 5] for patient in range(20)}
    manifest = make_split_manifest(grades, tmp_path / "split.json")
    partitions = [set(values) for values in manifest["patients"].values()]
    assert not (partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])
    assert all(all(support[str(grade)] for grade in range(2, 6)) for support in manifest["grade_support"].values())


def test_temperature_is_positive_and_returns_normalized_probabilities():
    probs = np.array([[.85, .05, .05, .05], [.05, .85, .05, .05]])
    temperature = fit_temperature(probs, [2, 3])
    scaled = temperature_scale(probs, temperature)
    assert temperature > 0
    assert np.allclose(scaled.sum(axis=1), 1.0)
    assert scaled.argmax(axis=1).tolist() == [0, 1]


def test_manifest_refuses_unrepresented_grade(tmp_path):
    with pytest.raises(ValueError, match="GGG5"):
        make_split_manifest({"1_a": [2, 3, 4], "2_a": [2, 3, 4]}, tmp_path / "split.json")


def test_manifest_allows_ungraded_patients_without_splitting_them(tmp_path):
    grades = {f"{patient}_study": [2, 3, 4, 5] for patient in range(20)}
    grades.update({"ungraded_a": [], "ungraded_b": []})
    manifest = make_split_manifest(grades, tmp_path / "split.json")
    assigned = set().union(*(set(values) for values in manifest["patients"].values()))
    assert "ungraded" in assigned


def test_manifest_stratifies_single_grade_patient_groups(tmp_path):
    grades = {f"p{grade}-{patient}_study": [grade] for grade in range(2, 6) for patient in range(20)}
    manifest = make_split_manifest(grades, tmp_path / "split.json")
    assert all(all(support[str(grade)] for grade in range(2, 6))
               for support in manifest["grade_support"].values())
