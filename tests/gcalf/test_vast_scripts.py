"""Focused command-line tests for the single-instance Vast.ai workflow helpers."""

import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HOST = ROOT / "cloud" / "vast" / "host.sh"
DOWNLOAD = ROOT / "cloud" / "vast" / "download_picai.sh"
EXPORT = ROOT / "cloud" / "vast" / "export_results.sh"
PAUSE_AND_BACKUP = ROOT / "cloud" / "vast" / "pause_and_backup.sh"


def run_script(script: Path, *args: str, cwd=None):
    return subprocess.run(["bash", str(script), *args], cwd=cwd or ROOT, text=True, capture_output=True)


def test_host_requires_explicit_ids_and_confirmations():
    result = run_script(HOST, "create", "--disk-gb", "500")
    assert result.returncode == 2
    assert "--offer-id" in result.stderr

    result = run_script(HOST, "destroy", "--instance-id", "42")
    assert result.returncode == 2
    assert "--confirm" in result.stderr


def test_host_create_dry_run_does_not_require_or_execute_vastai():
    result = run_script(HOST, "create", "--offer-id", "123", "--disk-gb", "500", "--dry-run")
    assert result.returncode == 0
    assert "vastai create instance 123" in result.stdout
    assert "pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel" in result.stdout
    assert "--label gcalf-m1-m6" in result.stdout


def test_download_uses_the_pinned_kaggle_source_in_dry_run(tmp_path):
    result = run_script(DOWNLOAD, "--source-dir", str(tmp_path / "source"), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "varshithpsingh/prostate-cancer-pi-cai-dataset/3" in result.stdout
    assert "kaggle datasets download" in result.stdout


def test_download_rejects_an_unpinned_kaggle_source(tmp_path):
    result = run_script(
        DOWNLOAD,
        "--source-dir",
        str(tmp_path / "source"),
        "--dataset-ref",
        "varshithpsingh/prostate-cancer-pi-cai-dataset",
        "--dry-run",
    )
    assert result.returncode == 2
    assert "pinned" in result.stderr


def test_export_refuses_existing_output_and_archives_required_contents(tmp_path):
    task = tmp_path / "Task2201_PICAI_GGG"
    evidence = tmp_path / "evidence"
    model = tmp_path / "models" / "Task900_PICAI_TINY" / "RetinaUNetV001_D3V001_3d"
    (task / "raw_splitted").mkdir(parents=True)
    (task / "raw_splitted" / "case.txt").write_text("task")
    (evidence / "m2").mkdir(parents=True)
    (evidence / "m2" / "overfit.log").write_text("loss")
    (model / "test_results" / "picai").mkdir(parents=True)
    (model / "model_best.ckpt").write_text("best")
    (model / "model_last.ckpt").write_text("last")
    (model / "plan_inference.pkl").write_text("plan")
    (model / "test_predictions").mkdir()
    (model / "test_predictions" / "case_boxes.pkl").write_text("prediction")
    (model / "test_results" / "picai" / "metrics.csv").write_text("metrics")

    export = tmp_path / "export"
    result = run_script(
        EXPORT,
        "--task-dir",
        str(task),
        "--evidence-dir",
        str(evidence),
        "--model-dir",
        str(model),
        "--export-dir",
        str(export),
    )
    assert result.returncode == 0, result.stderr
    assert (export / "SHA256SUMS").is_file()
    with tarfile.open(export / "Task2201_PICAI_GGG.tar") as archive:
        assert "Task2201_PICAI_GGG/raw_splitted/case.txt" in archive.getnames()
    with tarfile.open(export / "Task900_PICAI_TINY_m3_model.tar") as archive:
        assert "Task900_PICAI_TINY/RetinaUNetV001_D3V001_3d/model_best.ckpt" in archive.getnames()

    result = run_script(
        EXPORT,
        "--task-dir",
        str(task),
        "--evidence-dir",
        str(evidence),
        "--model-dir",
        str(model),
        "--export-dir",
        str(export),
    )
    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr


def test_pause_and_backup_creates_a_checksumming_snapshot_without_a_process(tmp_path):
    source = tmp_path / "fold1"
    source.mkdir()
    (source / "model_last.ckpt").write_text("checkpoint")
    backup = tmp_path / "backup"

    result = run_script(PAUSE_AND_BACKUP, "--source-dir", str(source), "--backup-dir", str(backup))

    assert result.returncode == 0, result.stderr
    assert (backup / "SHA256SUMS").is_file()
    with tarfile.open(backup / "final_1_fold1.tar") as archive:
        assert "fold1/model_last.ckpt" in archive.getnames()
