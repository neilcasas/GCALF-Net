import json
import pickle

import numpy as np
import pytest

from scripts import backfill_anatomy_metadata as anatomy


def test_dry_run_mutates_nothing(tmp_path, monkeypatch):
    plan_path = tmp_path / "preprocessed" / "D3V001_3d.pkl"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("wb") as file:
        pickle.dump({"transpose_forward": [0, 1, 2]}, file)
    monkeypatch.setattr(anatomy, "_case_ids", lambda _: ["case"])
    monkeypatch.setattr(anatomy, "_case_update", lambda *args: ({"updated": True}, 1.0, None, True, None))

    anatomy.backfill(tmp_path, tmp_path / "labels", write=False)

    assert not (tmp_path / "preprocessed" / "D3V001_3d" / "imagesTr" / "case.pkl").exists()


def test_write_aborts_when_containment_threshold_fails(tmp_path, monkeypatch):
    plan_path = tmp_path / "preprocessed" / "D3V001_3d.pkl"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("wb") as file:
        pickle.dump({"transpose_forward": [0, 1, 2]}, file)
    monkeypatch.setattr(anatomy, "_case_ids", lambda _: ["case"])
    monkeypatch.setattr(anatomy, "_case_update", lambda *args: ({"updated": True}, 0.49, None, True, None))

    with pytest.raises(ValueError, match="frame validation failed"):
        anatomy.backfill(tmp_path, tmp_path / "labels", write=True)
    assert not (tmp_path / "preprocessed" / "D3V001_3d" / "imagesTr" / "case.pkl").exists()


def test_fit_shape_allows_only_one_voxel_resampling_rounding():
    source = np.arange(12).reshape(2, 2, 3)
    resized = anatomy._fit_shape(source, (3, 2, 3))
    assert resized.shape == (3, 2, 3)

    with pytest.raises(ValueError, match="more than one voxel"):
        anatomy._fit_shape(source, (3, 4, 3))


def test_mask_uses_nndetection_label_resampling(monkeypatch):
    source = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    properties = {
        "crop_bbox": ((0, 2), (0, 3), (0, 4)),
        "original_size_of_raw_data": source.shape,
        "original_spacing": (3.0, 1.0, 1.0),
        "size_after_cropping": source.shape,
        "spacing_after_resampling": (3.0, 1.0, 1.0),
    }
    observed = {}

    def fake_resample(data, seg, original_spacing, target_spacing, **kwargs):
        observed.update({
            "data": data,
            "original_spacing": original_spacing.copy(),
            "target_spacing": target_spacing.copy(),
            "kwargs": kwargs,
        })
        return None, seg

    monkeypatch.setattr(anatomy, "_resample_patient", fake_resample)
    adjustments = []
    actual = anatomy._mask_in_preprocessed_frame(
        source, properties, source.shape, [0, 1, 2], [3.0, 1.0, 1.0], 3.0, adjustments, "test:gland",
    )

    np.testing.assert_array_equal(actual, source)
    assert observed["data"] is None
    np.testing.assert_array_equal(observed["original_spacing"], [3.0, 1.0, 1.0])
    np.testing.assert_array_equal(observed["target_spacing"], [3.0, 1.0, 1.0])
    assert observed["kwargs"] == {
        "order_data": 3,
        "order_seg": 0,
        "force_separate_z": False,
        "order_z_data": 9999,
        "order_z_seg": 9999,
        "separate_z_anisotropy_threshold": 3.0,
    }
    assert adjustments[0]["delta_zyx"] == [0, 0, 0]


def test_mask_rejects_pre_crop_raw_shape_mismatch(monkeypatch):
    properties = {
        "crop_bbox": ((0, 2), (0, 3), (0, 4)),
        "original_size_of_raw_data": (2, 3, 5),
        "original_spacing": (3.0, 1.0, 1.0),
        "size_after_cropping": (2, 3, 4),
        "spacing_after_resampling": (3.0, 1.0, 1.0),
    }
    monkeypatch.setattr(anatomy, "_resample_patient", lambda *args, **kwargs: pytest.fail("resampler was called"))

    with pytest.raises(ValueError, match="does not match nnDetection raw shape"):
        anatomy._mask_in_preprocessed_frame(
            np.zeros((2, 3, 4), dtype=np.uint8), properties, (2, 3, 4), [0, 1, 2],
        )


def test_instance_outside_distance_p95_mm_none_when_fully_contained():
    gland = np.ones((4, 6, 6), dtype=bool)
    seg = np.zeros((4, 6, 6), dtype=np.int16)
    seg[1:3, 2:4, 2:4] = 1

    result = anatomy._instance_outside_distance_p95_mm(seg, gland, (3.0, 0.5, 0.5))

    assert result == {"1": None}


def test_instance_outside_distance_p95_mm_measures_displacement():
    gland = np.zeros((4, 20, 20), dtype=bool)
    gland[1:3, 5:15, 5:15] = True
    seg = np.zeros((4, 20, 20), dtype=np.int16)
    seg[1:3, 7:9, 7:9] = 1  # fully inside
    seg[1:3, 17:19, 7:9] = 2  # clearly displaced, several mm beyond the gland face

    result = anatomy._instance_outside_distance_p95_mm(seg, gland, (3.0, 0.5, 0.5))

    assert result["1"] is None
    assert result["2"] is not None
    assert result["2"] > 0.5


def test_instance_outside_distance_p95_mm_none_for_every_instance_when_gland_empty():
    gland = np.zeros((2, 4, 4), dtype=bool)
    seg = np.zeros((2, 4, 4), dtype=np.int16)
    seg[0, 0, 0] = 1

    result = anatomy._instance_outside_distance_p95_mm(seg, gland, (3.0, 0.5, 0.5))

    assert result == {"1": None}


def test_case_outside_distance_p95_mm_none_when_fully_contained():
    gland = np.ones((4, 6, 6), dtype=bool)
    seg = np.zeros((4, 6, 6), dtype=np.int16)
    seg[1:3, 2:4, 2:4] = 1

    assert anatomy._case_outside_distance_p95_mm(seg, gland, (3.0, 0.5, 0.5)) is None


def test_case_outside_distance_p95_mm_measures_displacement_across_all_lesion_voxels():
    gland = np.zeros((4, 20, 20), dtype=bool)
    gland[1:3, 5:15, 5:15] = True
    seg = np.zeros((4, 20, 20), dtype=np.int16)
    seg[1:3, 7:9, 7:9] = 1  # inside, instance 1
    seg[1:3, 17:19, 7:9] = 2  # clearly displaced, instance 2

    result = anatomy._case_outside_distance_p95_mm(seg, gland, (3.0, 0.5, 0.5))

    assert result is not None
    assert result > 0.5


def test_case_outside_distance_p95_mm_none_when_gland_or_lesion_empty():
    spacing = (3.0, 0.5, 0.5)
    gland = np.zeros((2, 4, 4), dtype=bool)
    seg = np.zeros((2, 4, 4), dtype=np.int16)
    seg[0, 0, 0] = 1
    assert anatomy._case_outside_distance_p95_mm(seg, gland, spacing) is None

    gland_full = np.ones((2, 4, 4), dtype=bool)
    empty_seg = np.zeros((2, 4, 4), dtype=np.int16)
    assert anatomy._case_outside_distance_p95_mm(empty_seg, gland_full, spacing) is None


def test_anatomy_codes_preserve_gland_and_zone_semantics():
    gland = np.asarray([[True, True, True, False]])
    zones = np.asarray([[1, 2, 7, 1]])
    np.testing.assert_array_equal(anatomy._anatomy_codes(gland, zones), [[1, 2, 3, 0]])


def test_anatomy_volumes_stage_until_validation_then_commit(tmp_path, monkeypatch):
    images_dir = tmp_path / "preprocessed" / "D3V001_3d" / "imagesTr"
    images_dir.mkdir(parents=True)
    plan_path = tmp_path / "preprocessed" / "D3V001_3d.pkl"
    with plan_path.open("wb") as file:
        pickle.dump({"transpose_forward": [0, 1, 2]}, file)
    (images_dir / "case_a.pkl").write_bytes(b"old")
    (images_dir / "case_b.pkl").write_bytes(b"old")
    monkeypatch.setattr(anatomy, "_case_ids", lambda _: ["case_a", "case_b"])

    def updates(*args):
        case_id = args[2]
        has_lesion = case_id == "case_a"
        return ({"case": case_id}, 1.0,
                np.full((2, 2, 2), 1 if case_id == "case_a" else 2, dtype=np.uint8), has_lesion, None)

    monkeypatch.setattr(anatomy, "_case_update", updates)
    anatomy.backfill(tmp_path, tmp_path / "labels", write=False)
    assert not (images_dir / "case_a_anatomy.npy").exists()
    assert not (images_dir / "case_b_anatomy.npy").exists()

    monkeypatch.setattr(anatomy, "_case_update", lambda *args: (
        {"case": args[2]}, 0.49, np.ones((2, 2, 2), dtype=np.uint8), True, None
    ) if args[2] == "case_b" else updates(*args))
    with pytest.raises(ValueError, match="frame validation failed"):
        anatomy.backfill(tmp_path, tmp_path / "labels", write=True)
    assert not (images_dir / "case_a_anatomy.npy").exists()
    assert not (images_dir / "case_b_anatomy.npy").exists()

    monkeypatch.setattr(anatomy, "_case_update", updates)
    anatomy.backfill(tmp_path, tmp_path / "labels", write=True)
    np.testing.assert_array_equal(np.load(images_dir / "case_a_anatomy.npy"), 1)
    np.testing.assert_array_equal(np.load(images_dir / "case_b_anatomy.npy"), 2)


def test_gate_uses_explicit_lesion_bearing_flag_and_persists_both_distributions(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    plan_path = task_dir / "preprocessed" / "D3V001_3d.pkl"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("wb") as file:
        pickle.dump({"transpose_forward": [0, 1, 2]}, file)
    monkeypatch.setattr(anatomy, "_case_ids", lambda _: ["perfect_lesion", "negative", "partial_lesion"])
    monkeypatch.setattr(anatomy, "_case_update", lambda *args: (
        {"updated": True},
        0.8 if args[2] == "partial_lesion" else 1.0,
        None,
        args[2] != "negative",
        None,
    ))

    anatomy.backfill(task_dir, tmp_path / "labels", write=False, evidence_dir=tmp_path / "evidence")

    report = json.loads((tmp_path / "evidence" / "anatomy-containment-backfill.json").read_text())
    assert report["lesion_bearing_case_count"] == 2
    assert report["current_filtered_distribution"]["n"] == 1
    assert report["current_filtered_distribution"]["median"] == 0.8
    assert report["lesion_bearing_distribution"]["median"] == pytest.approx(0.9)
    assert report["gate_passed"] is True
    assert report["gate_excluded_cases"] == {}


def _case_update_with_distance(case_id_to_result):
    def _fake(*args):
        return case_id_to_result[args[2]]
    return _fake


def test_gate_excludes_only_corroborated_displaced_cases(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    plan_path = task_dir / "preprocessed" / "D3V001_3d.pkl"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("wb") as file:
        pickle.dump({"transpose_forward": [0, 1, 2]}, file)
    monkeypatch.setattr(
        anatomy, "_case_ids", lambda _: ["displaced", "boundary_noise", "unmeasured", "healthy"]
    )
    monkeypatch.setattr(anatomy, "_case_update", _case_update_with_distance({
        # Fails the per-case minimum AND corroborated as genuinely displaced -> excluded.
        "displaced": ({"case": "displaced"}, 0.1, None, True,
                      anatomy.DISPLACEMENT_EXCLUSION_DISTANCE_MM + 1.0),
        # Fails the per-case minimum but within the boundary-noise distance -> not excluded.
        "boundary_noise": ({"case": "boundary_noise"}, 0.1, None, True,
                           anatomy.DISPLACEMENT_EXCLUSION_DISTANCE_MM),
        # Fails the per-case minimum with no distance measurement -> not excluded.
        "unmeasured": ({"case": "unmeasured"}, 0.1, None, True, None),
        "healthy": ({"case": "healthy"}, 1.0, None, True, None),
    }))

    with pytest.raises(ValueError, match="frame validation failed"):
        anatomy.backfill(task_dir, tmp_path / "labels", write=False, evidence_dir=tmp_path / "evidence")

    report = json.loads((tmp_path / "evidence" / "anatomy-containment-backfill.json").read_text())
    assert set(report["gate_excluded_cases"]) == {"displaced"}
    assert report["gate_excluded_cases"]["displaced"]["outside_distance_p95_mm"] == pytest.approx(
        anatomy.DISPLACEMENT_EXCLUSION_DISTANCE_MM + 1.0
    )
    assert report["gate_passed"] is False
    assert report["gated_distribution"]["n"] == 3


def test_gate_passes_once_displacement_exclusion_clears_the_minimum(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    plan_path = task_dir / "preprocessed" / "D3V001_3d.pkl"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("wb") as file:
        pickle.dump({"transpose_forward": [0, 1, 2]}, file)
    monkeypatch.setattr(anatomy, "_case_ids", lambda _: ["displaced"] + [f"healthy_{i}" for i in range(9)])
    results = {"displaced": ({"case": "displaced"}, 0.1, None, True,
                             anatomy.DISPLACEMENT_EXCLUSION_DISTANCE_MM + 5.0)}
    results.update({f"healthy_{i}": ({"case": f"healthy_{i}"}, 0.95, None, True, None) for i in range(9)})
    monkeypatch.setattr(anatomy, "_case_update", _case_update_with_distance(results))

    anatomy.backfill(task_dir, tmp_path / "labels", write=False, evidence_dir=tmp_path / "evidence")

    report = json.loads((tmp_path / "evidence" / "anatomy-containment-backfill.json").read_text())
    assert set(report["gate_excluded_cases"]) == {"displaced"}
    assert report["gate_passed"] is True
    assert report["gated_distribution"]["n"] == 9
