import pytest

from gcalf_data.audit_crop_retention import (
    excluded_case_ids,
    load_audit,
    make_record,
    validate_audit,
    write_audit,
)
from gcalf_data.prepare_picai import apply_exclusions_to_splits


def _retention(source_voxels, resampled_voxels, inplane_voxels, final_voxels):
    return {
        "source": {"voxels": source_voxels, "components": int(source_voxels > 0)},
        "resampled": {"voxels": resampled_voxels, "components": int(resampled_voxels > 0)},
        "inplane": {"voxels": inplane_voxels, "components": int(inplane_voxels > 0)},
        "final": {"voxels": final_voxels, "components": int(final_voxels > 0)},
    }


def test_complete_crop_loss_creates_an_explicit_exclusion(tmp_path):
    lost = make_record("11050_1001070", "gland", _retention(30, 30, 0, 0))
    retained = make_record("10000_1000000", "gland", _retention(20, 20, 20, 20))

    assert lost["status"] == "excluded_no_retained_voxels"
    assert excluded_case_ids([lost, retained]) == {"11050_1001070"}
    write_audit([lost, retained], tmp_path)

    records = validate_audit(
        tmp_path,
        raw_case_ids={"10000_1000000"},
        split_case_ids={"10000_1000000"},
        expected_source_positive_count=2,
    )
    assert [record["case_id"] for record in records] == ["10000_1000000", "11050_1001070"]
    assert load_audit(tmp_path) == records


def test_validate_audit_rejects_an_excluded_case_left_in_a_split(tmp_path):
    lost = make_record("11050_1001070", "gland", _retention(30, 30, 0, 0))
    write_audit([lost], tmp_path)

    with pytest.raises(AssertionError, match="remain in splits"):
        validate_audit(tmp_path, raw_case_ids=set(), split_case_ids={"11050_1001070"})


def test_apply_exclusions_removes_case_from_every_fold_without_relabelling_it():
    splits = [
        {"train": ["a", "lost"], "val": ["b"]},
        {"train": ["b"], "val": ["a", "lost"]},
    ]

    retained = apply_exclusions_to_splits(splits, {"lost"})

    assert retained == [
        {"train": ["a"], "val": ["b"]},
        {"train": ["b"], "val": ["a"]},
    ]
