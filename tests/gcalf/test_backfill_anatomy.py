import pytest
import pickle
import numpy as np

from scripts import backfill_anatomy_metadata as anatomy


def test_dry_run_mutates_nothing(tmp_path, monkeypatch):
    plan_path = tmp_path / "preprocessed" / "D3V001_3d.pkl"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("wb") as file:
        pickle.dump({"transpose_forward": [0, 1, 2]}, file)
    monkeypatch.setattr(anatomy, "_case_ids", lambda _: ["case"])
    monkeypatch.setattr(anatomy, "_case_update", lambda *args: ({"updated": True}, 1.0))

    anatomy.backfill(tmp_path, tmp_path / "labels", write=False)

    assert not (tmp_path / "preprocessed" / "D3V001_3d" / "imagesTr" / "case.pkl").exists()


def test_write_aborts_when_containment_threshold_fails(tmp_path, monkeypatch):
    plan_path = tmp_path / "preprocessed" / "D3V001_3d.pkl"
    plan_path.parent.mkdir(parents=True)
    with plan_path.open("wb") as file:
        pickle.dump({"transpose_forward": [0, 1, 2]}, file)
    monkeypatch.setattr(anatomy, "_case_ids", lambda _: ["case"])
    monkeypatch.setattr(anatomy, "_case_update", lambda *args: ({"updated": True}, 0.49))

    with pytest.raises(ValueError, match="frame validation failed"):
        anatomy.backfill(tmp_path, tmp_path / "labels", write=True)
    assert not (tmp_path / "preprocessed" / "D3V001_3d" / "imagesTr" / "case.pkl").exists()


def test_zoom_shape_is_exact_after_rounding():
    import numpy as np

    source = np.arange(12).reshape(2, 2, 3)
    resized = anatomy._fit_shape(source, (3, 4, 5))
    assert resized.shape == (3, 4, 5)


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
        return {"case": case_id}, 1.0, np.full((2, 2, 2), 1 if case_id == "case_a" else 2, dtype=np.uint8)

    monkeypatch.setattr(anatomy, "_case_update", updates)
    anatomy.backfill(tmp_path, tmp_path / "labels", write=False)
    assert not (images_dir / "case_a_anatomy.npy").exists()
    assert not (images_dir / "case_b_anatomy.npy").exists()

    monkeypatch.setattr(anatomy, "_case_update", lambda *args: (
        {"case": args[2]}, 0.49, np.ones((2, 2, 2), dtype=np.uint8)
    ) if args[2] == "case_b" else updates(*args))
    with pytest.raises(ValueError, match="frame validation failed"):
        anatomy.backfill(tmp_path, tmp_path / "labels", write=True)
    assert not (images_dir / "case_a_anatomy.npy").exists()
    assert not (images_dir / "case_b_anatomy.npy").exists()

    monkeypatch.setattr(anatomy, "_case_update", updates)
    anatomy.backfill(tmp_path, tmp_path / "labels", write=True)
    np.testing.assert_array_equal(np.load(images_dir / "case_a_anatomy.npy"), 1)
    np.testing.assert_array_equal(np.load(images_dir / "case_b_anatomy.npy"), 2)
