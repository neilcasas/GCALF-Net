import json
from pathlib import Path

import pytest

from gcalf_data.build_labels import inject_grade_metadata, parse_lesion_isup


def test_marksheet_parser_ignores_ungraded_lesions():
    assert list(parse_lesion_isup("2,N/A,4")) == [2, 4]


def _write_case_json(labels_dir: Path, case_id: str, instances):
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{case_id}.json").write_text(json.dumps({"instances": instances}))


def test_inject_grade_metadata_derives_grade_from_human_expert_raw_class(tmp_path):
    labels_dir = tmp_path / "labelsTr"
    # raw classes 1-4 came from human_expert mask values 2-5 respectively.
    _write_case_json(labels_dir, "10000_1000000", {"1": 2, "2": 4})

    grade_counts, ungraded = inject_grade_metadata(labels_dir, audit_results={})

    metadata = json.loads((labels_dir / "10000_1000000.json").read_text())
    assert metadata["instances"] == {"1": 0, "2": 0}
    assert metadata["grades"] == {"1": 3, "2": 5}
    assert metadata["grade_sources"] == {"1": "human_expert_mask", "2": "human_expert_mask"}
    assert metadata["grade_supervised"] == {"1": True, "2": True}
    assert grade_counts == {3: 1, 5: 1}
    assert ungraded == 0


def test_inject_grade_metadata_uses_audit_result_for_pooch25_instance(tmp_path):
    labels_dir = tmp_path / "labelsTr"
    _write_case_json(labels_dir, "10013_1000013", {"1": 0})  # raw class 0 == Pooch25 mask value 1
    audit_results = {
        "10013_1000013": {"grade_supervised": True, "grade": 2, "grade_source": "audit_unifocal"},
    }

    grade_counts, ungraded = inject_grade_metadata(labels_dir, audit_results)

    metadata = json.loads((labels_dir / "10013_1000013.json").read_text())
    assert metadata["instances"] == {"1": 0}
    assert metadata["grades"] == {"1": 2}
    assert metadata["grade_sources"] == {"1": "audit_unifocal"}
    assert metadata["grade_supervised"] == {"1": True}
    assert grade_counts == {2: 1}
    assert ungraded == 0


def test_inject_grade_metadata_leaves_unrecovered_pooch25_instance_ungraded(tmp_path):
    labels_dir = tmp_path / "labelsTr"
    _write_case_json(labels_dir, "10008_1000008", {"1": 0})
    audit_results = {
        "10008_1000008": {"grade_supervised": False, "grade": None, "grade_source": None},
    }

    grade_counts, ungraded = inject_grade_metadata(labels_dir, audit_results)

    metadata = json.loads((labels_dir / "10008_1000008.json").read_text())
    assert metadata["instances"] == {"1": 0}
    assert metadata["grades"] == {}
    assert metadata["grade_supervised"] == {"1": False}
    assert grade_counts == {}
    assert ungraded == 1


def test_inject_grade_metadata_rejects_unexpected_raw_class(tmp_path):
    labels_dir = tmp_path / "labelsTr"
    _write_case_json(labels_dir, "10000_1000000", {"1": 5})
    with pytest.raises(ValueError, match="unexpected raw instance class"):
        inject_grade_metadata(labels_dir, audit_results={})
