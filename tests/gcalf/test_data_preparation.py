import json

import pytest

from gcalf_data.build_labels import GGG_LABELS, parse_lesion_isup, remap_source_label
from gcalf_data.prepare_picai import dataset_json, load_splits


def test_source_labels_are_remapped_to_contiguous_foreground_classes():
    assert [remap_source_label(label) for label in (0, 2, 3, 4, 5)] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="Unsupported"):
        remap_source_label(1)


def test_dataset_metadata_describes_four_ggg_foreground_classes():
    metadata = dataset_json("Task2201_PICAI_GGG")
    assert metadata["modality"] == {"0": "T2W", "1": "ADC", "2": "HBV"}
    assert metadata["labels"] == {"0": "background", "1": "GGG2", "2": "GGG3", "3": "GGG4", "4": "GGG5"}
    assert GGG_LABELS == {1: "GGG2", 2: "GGG3", 3: "GGG4", 4: "GGG5"}


def test_marksheet_parser_ignores_ungraded_lesions():
    assert list(parse_lesion_isup("2,N/A,4")) == [2, 4]


def test_load_splits_requires_five_train_validation_folds(tmp_path):
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps([{"train": [], "val": []}]))
    with pytest.raises(ValueError, match="five folds"):
        load_splits(split_path)
