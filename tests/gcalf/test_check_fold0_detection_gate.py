import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_fold0_detection_gate.py"
spec = importlib.util.spec_from_file_location("check_fold0_detection_gate", SCRIPT_PATH)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


TASK = "Task2201_PICAI_csPCa"


def _write_metrics(root, arm, lesion_ap, picai_score):
    path = root / TASK / f"RetinaUNetV001_D3V001_3d_{arm}" / "fold0" / "val_results" / "picai"
    path.mkdir(parents=True)
    with (path / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["lesion_ap", "picai_score"])
        writer.writeheader()
        writer.writerow({"lesion_ap": lesion_ap, "picai_score": picai_score})


def test_gate_passes_and_records_both_detection_values(tmp_path):
    _write_metrics(tmp_path, "baseline", 0.50, 0.60)
    _write_metrics(tmp_path, "full", 0.49, 0.58)

    assert gate.run_gate(TASK, tmp_path)
    record = gate._train_dir(tmp_path, TASK, "full", 0) / "fold0_detection_gate.json"
    assert record.is_file()
    assert '"passed": true' in record.read_text()
    gate.validate_gate_record(record, TASK)


def test_gate_record_validation_rejects_relaxed_thresholds(tmp_path):
    _write_metrics(tmp_path, "baseline", 0.50, 0.60)
    _write_metrics(tmp_path, "full", 0.49, 0.58)
    assert gate.run_gate(TASK, tmp_path)
    record = gate._train_dir(tmp_path, TASK, "full", 0) / "fold0_detection_gate.json"

    data = json.loads(record.read_text())
    data["thresholds"]["lesion_ap"] = 1.0
    record.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="locked 0.02/0.03"):
        gate.validate_gate_record(record, TASK)


def test_gate_fails_when_lesion_ap_drop_exceeds_threshold(tmp_path):
    _write_metrics(tmp_path, "baseline", 0.50, 0.60)
    _write_metrics(tmp_path, "full", 0.47, 0.60)
    assert not gate.run_gate(TASK, tmp_path)


def test_gate_fails_when_picai_score_drop_exceeds_threshold(tmp_path):
    _write_metrics(tmp_path, "baseline", 0.50, 0.60)
    _write_metrics(tmp_path, "full", 0.50, 0.56)
    assert not gate.run_gate(TASK, tmp_path)


def test_gate_surfaces_missing_metrics_when_eval_cannot_run(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("run_eval unavailable")

    monkeypatch.setattr(gate.subprocess, "run", unavailable)
    with pytest.raises(FileNotFoundError, match="run_eval unavailable"):
        gate.run_gate(TASK, tmp_path)
