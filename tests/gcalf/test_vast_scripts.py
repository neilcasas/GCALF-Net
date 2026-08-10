"""Focused command-line tests for the single-instance Vast.ai workflow helpers."""

import hashlib
import os
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HOST = ROOT / "cloud" / "vast" / "host.sh"
DOWNLOAD = ROOT / "cloud" / "vast" / "download_picai.sh"
EXPORT = ROOT / "cloud" / "vast" / "export_results.sh"


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


def test_download_manifest_verification_detects_checksum_failure(tmp_path):
    archive_dir = tmp_path / "source" / "archives"
    archive_dir.mkdir(parents=True)
    archive = archive_dir / "picai_public_images_fold0.zip"
    archive.write_bytes(b"not the expected archive")
    manifest = tmp_path / "manifest.tsv"
    expected = hashlib.md5(b"different bytes").hexdigest()
    manifest.write_text(
        "picai_public_images_fold0.zip\t%s\thttps://zenodo.org/records/6517398/files/fold0.zip\n" % expected
        + "\n".join(
            "picai_public_images_fold%d.zip\t%s\thttps://zenodo.org/records/6517398/files/fold%d.zip" % (i, "0" * 32, i)
            for i in range(1, 5)
        )
        + "\n"
    )
    result = run_script(
        DOWNLOAD,
        "--source-dir",
        str(tmp_path / "source"),
        "--manifest",
        str(manifest),
        "--verify-only",
    )
    assert result.returncode == 2
    assert "checksum mismatch" in result.stderr


def test_download_manifest_rejects_non_picai_rows(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("wrong.zip\t%s\thttps://zenodo.org/records/6517398/files/wrong.zip\n" % ("0" * 32))
    result = run_script(DOWNLOAD, "--source-dir", str(tmp_path / "source"), "--manifest", str(manifest), "--dry-run")
    assert result.returncode == 2
    assert "unexpected archive" in result.stderr


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
