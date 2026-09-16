from types import SimpleNamespace

import numpy as np
import pytest

from nndet.io.datamodule import bg_module
from nndet.io.datamodule.bg_loader import DataLoader3DFast
from nndet.io.load import save_pickle


def _loader_with_grade_instances(tmp_path, grades):
    data = {}
    for index, grade in enumerate(grades, start=1):
        boxes = tmp_path / f"case{index}_boxes.pkl"
        properties = tmp_path / f"case{index}.pkl"
        save_pickle({"instances": [1]}, boxes)
        save_pickle({"grades": {"1": grade}, "grade_supervised": {"1": True}}, properties)
        data[f"case{index}"] = {"boxes_file": str(boxes), "properties_file": str(properties)}
    loader = DataLoader3DFast.__new__(DataLoader3DFast)
    loader._data = data
    loader.grade_balanced_sampling = True
    loader.batch_size = 400
    loader.oversample_foreground_percent = 1.0
    loader.cache = loader.build_cache()
    return loader


def test_grade_balanced_sampling_reads_properties_and_selects_all_grade_classes(tmp_path):
    loader = _loader_with_grade_instances(tmp_path, [2, 3, 4, 5])
    np.random.seed(2026)

    cases, instances = loader.select()

    assert set(instances) == {1}
    selected_grades = {int(case[len("case"):]) + 1 for case in cases}
    assert selected_grades == {2, 3, 4, 5}
    assert {grade: len(loader.cache["grade_instances"][grade]) for grade in range(2, 6)} == {
        2: 1, 3: 1, 4: 1, 5: 1,
    }


def test_grade_balanced_sampling_rejects_a_training_fold_without_every_grade(tmp_path):
    with pytest.raises(ValueError, match=r"missing \[5\]"):
        _loader_with_grade_instances(tmp_path, [2, 3, 4])


def test_validation_loader_disables_grade_balanced_sampling(monkeypatch):
    captured = {}

    class Loader:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = SimpleNamespace(
        dataloader="test", dataloader_kwargs={"grade_balanced_sampling": True, "other": "kept"},
        dataset_val={}, batch_size=1, patch_size=(1, 1, 1),
        augment_cfg={"oversample_foreground_percent": 0.0, "num_val_batches_per_epoch": 1,
                     "num_threads": 1, "multiprocessing": False},
        augmentation=type("Augmentation", (), {"get_validation_transforms": lambda self: None})(),
        _augmenter_seeds=lambda stream: [],
    )
    monkeypatch.setattr(bg_module.DATALOADER_REGISTRY, "get", lambda _: Loader)
    monkeypatch.setattr(bg_module, "get_augmenter", lambda dataloader, **_: dataloader)

    bg_module.Datamodule.val_dataloader(module)

    assert captured["grade_balanced_sampling"] is False
    assert captured["other"] == "kept"
