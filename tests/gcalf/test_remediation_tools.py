import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "scripts" / "audit_grade_metadata.py"
M5 = ROOT / "scripts" / "record_m5_budget.py"


def _label_record():
    grades = [2] * 253 + [3] * 104 + [4] * 37 + [5] * 47
    instance_ids = [str(index) for index in range(1, 459)]
    return {
        "instances": {instance_id: {} for instance_id in instance_ids},
        "grades": {str(index): grade for index, grade in enumerate(grades, start=1)},
        "grade_supervised": {str(index): index <= len(grades) for index in range(1, 459)},
    }


def test_grade_metadata_audit_writes_a_verified_manifest(tmp_path):
    labels = tmp_path / "labelsTr"
    labels.mkdir()
    (labels / "case.json").write_text(json.dumps(_label_record()))
    manifest = tmp_path / "grade_metadata_manifest.json"

    result = subprocess.run([sys.executable, str(AUDIT), "--labels-dir", str(labels), "--manifest", str(manifest)],
                            text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    report = json.loads(manifest.read_text())
    assert report["grade_supervised_lesions"] == 441
    assert report["heterogeneous_or_unsupervised_lesions"] == 17
    assert report["grade_counts"] == {"GGG2": 253, "GGG3": 104, "GGG4": 37, "GGG5": 47}


def test_m5_record_uses_instance_hours_not_serialized_gpu_hours(tmp_path):
    output = tmp_path / "m5.json"
    result = subprocess.run(
        [sys.executable, str(M5), "--arm", "baseline=0.5", "--arm", "lff=0.6", "--arm", "caf=0.7",
         "--arm", "full=0.8", "--contention-factor", "1.1", "--hourly-instance-cost", "1.0",
         "--rung", "full", "--output", str(output)], text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())
    assert record["projection"]["gpu_hours"] == 541.6666666666666
    assert record["projection"]["instance_hours"] == 135.41666666666666
