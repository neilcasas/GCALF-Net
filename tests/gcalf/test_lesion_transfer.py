import pickle

import numpy as np
import pytest

from nndet.io.augmentation.lesion_transfer import (
    LesionBank,
    LesionTransferConfig,
    anatomy_occupancy_fractions,
    feather_alpha,
    paste_lesion,
    placement_slices,
    resolve_region,
    sample_target_center,
)
from nndet.io.datamodule.bg_loader import DataLoader3DFast, DataLoader3DLesionTransfer


def _config(**overrides):
    values = {
        "region_mode": "mask",
        "feather_sigma": 0.75,
        "spatial": {
            "p_rot": 0.0,
            "p_scale": 0.0,
            "p_flip": 0.0,
            "flip_axes": [2],
        },
        "intensity": {"p_gamma": 0.0, "p_blur": 0.0},
    }
    values.update(overrides)
    return LesionTransferConfig.from_dict(values)


def test_base_transfer_hook_is_identity():
    loader = DataLoader3DFast.__new__(DataLoader3DFast)
    data = np.zeros((3, 5, 6, 7), dtype=np.float32)
    seg = np.zeros((1, 5, 6, 7), dtype=np.float32)
    properties = {"instances": {"1": 1}}
    result = loader.maybe_transfer_lesion(data, seg, properties, "case", [slice(0, 5)] * 3, 1)
    assert result[0] is data and result[1] is seg and result[2] is properties
    assert data.tobytes() == np.zeros_like(data).tobytes()


def test_paste_writes_segmentation_and_all_grade_properties():
    pytest.importorskip("scipy")
    data = np.zeros((3, 9, 9, 9), dtype=np.float32)
    seg = np.zeros((1, 9, 9, 9), dtype=np.float32)
    properties = {"instances": {}, "grades": {}, "grade_supervised": {}, "grade_sources": {}}
    source_data = np.ones((3, 3, 3, 3), dtype=np.float32)
    source_mask = np.ones((3, 3, 3), dtype=bool)
    assert paste_lesion(
        data, seg, properties, source_data, source_mask, [4, 4, 4],
        10001, 5, "src_1", 17, _config(),
    )
    assert np.all(data[:, 3:6, 3:6, 3:6] > 0)
    assert 10001 in np.unique(seg)
    assert properties["instances"]["10001"] == 0
    assert properties["grades"]["10001"] == 5
    assert properties["grade_supervised"]["10001"] is True
    assert properties["grade_sources"]["10001"] == "lesion_transfer:src_1:17"


def test_pasted_instance_round_trips_through_detection_and_grade_transforms():
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from nndet.io.transforms.instances import Instances2Boxes, Instances2Grades

    data = np.zeros((3, 9, 9, 9), dtype=np.float32)
    seg = np.zeros((1, 9, 9, 9), dtype=np.float32)
    properties = {"instances": {}, "grades": {}, "grade_supervised": {}, "grade_sources": {}}
    paste_lesion(
        data, seg, properties, np.ones((3, 3, 3, 3), dtype=np.float32),
        np.ones((3, 3, 3), dtype=bool), [4, 4, 4], 10001, 5, "src_1", 17, _config(),
    )
    batch = {
        "target": torch.from_numpy(seg[None]),
        "instance_mapping": [properties["instances"]],
        "properties": [properties],
        "present_instances": [torch.tensor([10001])],
    }
    batch = Instances2Boxes(
        instance_key="target", map_key="instance_mapping", box_key="target_boxes",
        class_key="target_classes", present_instances="present_instances",
    )(**batch)
    batch = Instances2Grades(
        properties_key="properties", present_instances="present_instances",
        grade_key="target_grades", grade_supervised_key="target_grade_supervised",
    )(**batch)
    assert batch["target_classes"][0].tolist() == [0]
    assert batch["target_grades"][0].tolist() == [5]
    assert batch["target_grade_supervised"][0].tolist() == [True]
    box = batch["target_boxes"][0][0].tolist()
    assert box[0] <= 3 and box[1] <= 3 and box[2] >= 5 and box[3] >= 5


def test_preexisting_synthetic_id_is_rejected():
    pytest.importorskip("scipy")
    data = np.zeros((3, 7, 7, 7), dtype=np.float32)
    seg = np.zeros((1, 7, 7, 7), dtype=np.float32)
    seg[0, 0, 0, 0] = 10000
    with pytest.raises(ValueError, match="already exists"):
        paste_lesion(
            data, seg, {"instances": {"10000": 1}},
            np.ones((3, 3, 3, 3), dtype=np.float32), np.ones((3, 3, 3), dtype=bool),
            [3, 3, 3], 10000, 4, "case", 1, _config(),
        )


def test_pad_sentinel_is_never_overwritten():
    pytest.importorskip("scipy")
    data = np.zeros((3, 7, 7, 7), dtype=np.float32)
    seg = np.zeros((1, 7, 7, 7), dtype=np.float32)
    seg[:, :2] = -1
    paste_lesion(
        data, seg, {"instances": {}},
        np.ones((3, 3, 3, 3), dtype=np.float32), np.ones((3, 3, 3), dtype=bool),
        [2, 3, 3], 10001, 4, "case", 1, _config(),
    )
    assert np.all(seg[:, :2] == -1)


def test_dilated_region_blends_beyond_but_labels_only_the_transformed_lesion():
    pytest.importorskip("scipy")
    data = np.zeros((3, 11, 11, 11), dtype=np.float32)
    seg = np.zeros((1, 11, 11, 11), dtype=np.float32)
    source_data = np.ones((3, 7, 7, 7), dtype=np.float32)
    source_mask = np.zeros((7, 7, 7), dtype=bool)
    source_mask[3, 3, 3] = True

    assert paste_lesion(
        data, seg, {"instances": {}}, source_data, source_mask, [5, 5, 5],
        10001, 5, "src_1", 17, _config(region_mode="dilated", dilation_vox=3),
    )
    assert int(np.count_nonzero(seg == 10001)) == 1
    assert int(np.count_nonzero(data > 0)) > 1


def test_positive_target_collision_is_rejected_without_mutation_or_metadata():
    pytest.importorskip("scipy")
    data = np.zeros((3, 9, 9, 9), dtype=np.float32)
    seg = np.zeros((1, 9, 9, 9), dtype=np.float32)
    seg[0, 4, 4, 4] = 7
    properties = {"instances": {"7": 0}}
    before_data = data.copy()
    before_seg = seg.copy()

    assert not paste_lesion(
        data, seg, properties, np.ones((3, 3, 3, 3), dtype=np.float32),
        np.ones((3, 3, 3), dtype=bool), [4, 4, 4], 10001, 5, "src", 1,
        _config(region_mode="dilated", dilation_vox=3),
    )
    assert data.tobytes() == before_data.tobytes()
    assert seg.tobytes() == before_seg.tobytes()
    assert "10001" not in properties


def test_empty_hard_mask_is_rejected_without_metadata():
    pytest.importorskip("scipy")
    data = np.zeros((3, 7, 7, 7), dtype=np.float32)
    seg = np.full((1, 7, 7, 7), -1, dtype=np.float32)
    properties = {"instances": {}}
    assert not paste_lesion(
        data, seg, properties, np.ones((3, 3, 3, 3), dtype=np.float32),
        np.ones((3, 3, 3), dtype=bool), [3, 3, 3], 10001, 5, "src", 1, _config(),
    )
    assert properties == {"instances": {}}


def test_anatomy_occupancy_rejects_boundary_and_wrong_zone():
    mask = np.ones((3, 3, 3), dtype=bool)
    anatomy = np.ones((7, 7, 7), dtype=np.uint8)
    anatomy[:3] = 0
    gland_frac, zone_frac = anatomy_occupancy_fractions(mask, anatomy, [3, 3, 3], "pz")
    assert gland_frac == pytest.approx(2 / 3)
    assert zone_frac == pytest.approx(2 / 3)

    anatomy[...] = 2
    gland_frac, zone_frac = anatomy_occupancy_fractions(mask, anatomy, [3, 3, 3], "pz")
    assert gland_frac == 1.0
    assert zone_frac == 0.0


def test_placement_slices_and_rng_are_reproducible():
    target, source = placement_slices([2, 2, 2], (3, 3, 3), (5, 5, 5))
    assert target[0] == slice(1, 4)
    assert source[0] == slice(0, 3)

    np.random.seed(2026)
    drawn_seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    expected = np.random.default_rng(np.random.SeedSequence([drawn_seed]))
    np.random.seed(2026)
    loader = DataLoader3DLesionTransfer.__new__(DataLoader3DLesionTransfer)
    loader._lesion_transfer_rng = None
    np.testing.assert_array_equal(loader._rng().integers(100, size=8), expected.integers(100, size=8))


def test_maybe_transfer_retries_a_clean_center(monkeypatch):
    import nndet.io.datamodule.bg_loader as bg_loader

    cfg = _config(
        enabled=True, p_paste=1.0, max_pastes_per_patch=1,
        max_placement_attempts=3, min_gland_frac=0.95, min_zone_frac=0.5,
    )
    loader = DataLoader3DLesionTransfer.__new__(DataLoader3DLesionTransfer)
    loader.lesion_transfer_cfg = cfg
    loader.lesion_bank = type("Bank", (), {
        "sample": lambda self, *args: {"case_id": "src", "instance_id": 1},
        "load": lambda self, record: (np.ones((3, 3, 3, 3), dtype=np.float32),
                                       np.ones((3, 3, 3), dtype=bool)),
    })()
    loader._lesion_transfer_rng = np.random.default_rng(2026)
    loader._anatomy_cache = {}
    loader.paste_counters = {"attempted": 0, "succeeded": 0,
                             "rejected_collision": 0, "rejected_anatomy": 0}
    calls = []
    monkeypatch.setattr(bg_loader, "augment_lesion", lambda data, mask, config, rng: (data, mask))
    monkeypatch.setattr(bg_loader, "anatomy_occupancy_fractions", lambda *args: (1.0, 1.0))
    monkeypatch.setattr(bg_loader, "blend_footprint_collides", lambda *args: True)
    monkeypatch.setattr(bg_loader, "sample_target_center", lambda *args: [3, 3, 3])
    monkeypatch.setattr(loader, "_load_anatomy", lambda case_id: np.ones((7, 7, 7), dtype=np.uint8))

    def paste(*args, **kwargs):
        calls.append(kwargs["center"])
        return len(calls) == 2

    monkeypatch.setattr(bg_loader, "paste_lesion", paste)
    loader.maybe_transfer_lesion(
        np.zeros((3, 7, 7, 7), dtype=np.float32), np.zeros((1, 7, 7, 7), dtype=np.float32),
        {"anatomy_centers": {"pz": np.asarray([[3, 3, 3]])}, "instances": {},
         "anatomy_frame": {"code_map": {
             "outside_gland": 0, "gland_pz": 1, "gland_tz": 2, "gland_other": 3,
         }}},
        "case", [slice(0, 7)] * 3, -1,
    )
    assert len(calls) == 2
    assert loader.paste_counters == {
        "attempted": 2, "succeeded": 1, "rejected_collision": 1, "rejected_anatomy": 0,
    }


def test_region_modes_and_feather_support():
    pytest.importorskip("scipy")
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[4, 4, 4] = True
    assert resolve_region(mask, "mask", 3).sum() == 1
    assert resolve_region(mask, "box", 3).sum() == 1
    assert resolve_region(mask, "dilated", 3).sum() > 1
    alpha = feather_alpha(np.pad(np.ones((3, 3, 3), dtype=bool), 2), 1.0)
    assert alpha[3, 3, 3] > 0.99
    assert alpha[0, 0, 0] == 0
    assert 0 < alpha[2, 3, 3] < 1


def test_sample_target_center_maps_case_frame_and_rejects_no_fit():
    properties = {"anatomy_centers": {"pz": np.asarray([[10, 11, 12]], dtype=np.int16)}}
    center = sample_target_center(
        properties, "pz", [slice(4, 14), slice(5, 15), slice(6, 16)],
        (8, 8, 8), (3, 3, 3), np.random.default_rng(2026),
    )
    np.testing.assert_array_equal(center, [6, 6, 6])
    assert sample_target_center(
        properties, "pz", [slice(0, 10)] * 3, (4, 4, 4), (9, 9, 9), np.random.default_rng(2026)
    ) is None


def test_bank_filters_case_ids_at_construction_and_prefers_same_zone(tmp_path):
    records = [
        {"case_id": "123_1", "instance_id": 1, "grade": 4, "zone_pz_frac": 1.0},
        {"case_id": "999_1", "instance_id": 1, "grade": 4, "zone_pz_frac": 1.0},
        {"case_id": "123_2", "instance_id": 2, "grade": 4, "zone_pz_frac": 0.0},
    ]
    with (tmp_path / "bank_index.pkl").open("wb") as file:
        pickle.dump(records, file)
    for record in records:
        np.savez(tmp_path / f"{record['case_id']}__{record['instance_id']}.npz",
                 data=np.zeros((3, 2, 2, 2), dtype=np.float16), mask=np.ones((2, 2, 2), dtype=bool))
    bank = LesionBank(tmp_path, {"123_1"})
    assert len(bank) == 2
    assert bank.dropped_count == 1
    selected = bank.sample(4, "pz", "prefer", np.random.default_rng(2026))
    assert selected["case_id"] == "123_1"


def test_unknown_config_key_is_rejected():
    with pytest.raises(TypeError, match="Unknown lesion_transfer_cfg keys"):
        LesionTransferConfig.from_dict({"enabled": False, "typo": True})
    with pytest.raises(ValueError, match="ADC channel 1"):
        LesionTransferConfig.from_dict({"jitter_channels": [0, 1, 2]})
