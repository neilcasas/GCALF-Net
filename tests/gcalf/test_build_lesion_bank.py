import pytest

from scripts import build_lesion_bank as bank


def _properties(**overrides):
    base = {
        "anatomy_centers": {"pz": [], "tz": []},
        "anatomy_frame": {"space": "preprocessed_3d"},
        "grades": {"1": 4, "2": 2},
        "grade_supervised": {"1": True, "2": True},
        "anatomy_instance_zone_pz_frac": {"1": 0.6, "2": 0.1},
        "anatomy_instance_outside_distance_p95_mm": {"1": None, "2": None},
    }
    base.update(overrides)
    return base


def test_supervised_admits_instances_within_the_distance_bar():
    properties = _properties(
        anatomy_instance_outside_distance_p95_mm={"1": 1.5, "2": None},
    )

    donors = list(bank._supervised(properties))

    assert donors == [(1, 4, 0.6), (2, 2, 0.1)]


def test_supervised_excludes_instances_beyond_the_distance_bar():
    properties = _properties(
        anatomy_instance_outside_distance_p95_mm={
            "1": bank.DONOR_MAX_OUTSIDE_DISTANCE_MM + 0.01,
            "2": None,
        },
    )

    donors = list(bank._supervised(properties))

    assert donors == [(2, 2, 0.1)]


def test_supervised_admits_instance_exactly_at_the_distance_bar():
    properties = _properties(
        anatomy_instance_outside_distance_p95_mm={"1": bank.DONOR_MAX_OUTSIDE_DISTANCE_MM, "2": None},
    )

    donors = list(bank._supervised(properties))

    assert (1, 4, 0.6) in donors


def test_supervised_requires_outside_distance_metadata():
    properties = _properties()
    del properties["anatomy_instance_outside_distance_p95_mm"]

    with pytest.raises(ValueError, match="donor outside-distance measurements"):
        list(bank._supervised(properties))


def test_supervised_still_requires_anatomy_frame_metadata():
    properties = _properties()
    del properties["anatomy_frame"]

    with pytest.raises(ValueError, match="Anatomy metadata is missing"):
        list(bank._supervised(properties))
