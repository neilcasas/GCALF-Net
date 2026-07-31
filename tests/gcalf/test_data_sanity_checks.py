import pytest

from gcalf_data.sanity_checks import validate_splits


def test_validate_splits_rejects_patient_leakage():
    splits = [
        {"train": ["10000_1000000"], "val": ["10000_1000001"]},
        {"train": ["10000_1000001"], "val": ["10000_1000000"]},
    ]
    with pytest.raises(AssertionError, match="leaks a patient"):
        validate_splits(splits, {"10000_1000000", "10000_1000001"})
