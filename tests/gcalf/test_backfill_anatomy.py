import pytest
import pickle

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
