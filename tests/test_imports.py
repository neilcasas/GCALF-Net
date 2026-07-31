from pathlib import Path

import nndet

from nndet.utils.info import SuppressPrint


def test_nevergrad_import():
    import nevergrad as ng


def test_batchgenerators_import():
    import batchgenerators


def test_pytorch_lightning_import():
    import pytorch_lightning as pl


def test_nnunet_import():
    with SuppressPrint():
        import nnunet.preprocessing.preprocessing as nn_preprocessing


def test_nndet_import_resolves_to_this_repository():
    repository_root = Path(__file__).resolve().parents[1]
    assert Path(nndet.__file__).resolve().parent == repository_root / "nndet"


def test_nndet_cuda_extension_import():
    import nndet._C


def test_picai_and_medcam_imports():
    import medcam
    import picai_eval
    import picai_prep
