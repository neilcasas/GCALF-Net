"""Planner-level guards for GCALF-Net's fixed encoder depth."""

import pytest

from nndet.planning.architecture.boxes.c002 import BoxC002


def _planner(model_cfg):
    planner = BoxC002.__new__(BoxC002)
    planner.architecture_kwargs = {}
    planner.dim = 3
    planner.model_cfg = model_cfg
    return planner


def test_gcalf_planner_caps_pooling_to_one_less_than_encoder_levels():
    planner = _planner({"encoder_kwargs": {"gcalf_cfg": {"num_levels": 5}}})
    planner.create_default_settings()
    assert planner.max_num_pool == 4


def test_plain_planner_keeps_legacy_pooling_cap():
    planner = _planner({"encoder_kwargs": {}})
    planner.create_default_settings()
    assert planner.max_num_pool == 999


def test_gcalf_planner_rejects_an_impossible_level_count():
    planner = _planner({"encoder_kwargs": {"gcalf_cfg": {"num_levels": 1}}})
    with pytest.raises(ValueError, match="at least two"):
        planner.create_default_settings()
