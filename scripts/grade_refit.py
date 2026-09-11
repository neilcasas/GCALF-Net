"""Run one opt-in, detector-frozen GGG2-5 refitting condition."""
import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nndet.io.datamodule.bg_module import Datamodule
from nndet.io.load import load_pickle
from nndet.ptmodule import MODULE_REGISTRY
from nndet.ptmodule.retinaunet.base import RetinaUNetModule
from nndet.training.grade_refit import GradeRefitLoop


def _cases_for_patients(dataset, patients):
    return OrderedDict((case, item) for case, item in dataset.items()
                       if case.split("_", 1)[0] in set(patients))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Resolved source-run Hydra config.")
    parser.add_argument("--plan", type=Path, required=True, help="Source-run plan.pkl.")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--condition", choices=("control", "regularized"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    plan = load_pickle(args.plan)
    manifest = json.loads(args.split_manifest.read_text())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    if args.condition == "control":
        weight_decay = 3e-5
    else:
        weight_decay = 1e-3
    module = MODULE_REGISTRY[cfg.module](
        model_cfg=OmegaConf.to_container(cfg.model_cfg, resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg.trainer_cfg, resolve=True), plan=plan)
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(args.device)
    data_dir = Path(cfg.host.preprocessed_output_dir) / plan["data_identifier"] / "imagesTr"
    datamodule = Datamodule(augment_cfg=OmegaConf.to_container(cfg.augment_cfg, resolve=True),
                            plan=plan, data_dir=data_dir, fold=int(cfg.exp.fold))
    fit = _cases_for_patients(datamodule.dataset, manifest["patients"]["fit"])
    selection = _cases_for_patients(datamodule.dataset, manifest["patients"]["selection"])
    if not fit or not selection:
        raise ValueError("The manifest does not identify fitting and selection cases in this source dataset")
    class_weights = RetinaUNetModule.compute_grade_class_weights(fit).to(args.device)
    datamodule.dataset_tr, datamodule.dataset_val = fit, selection
    datamodule.setup()
    loop = GradeRefitLoop(module, class_weights, weight_decay)
    history = []
    best = (float("inf"), None)
    for epoch in range(1, 13):
        train_loader, val_loader = iter(datamodule.train_dataloader()), iter(datamodule.val_dataloader())
        updates_before = loop.supervised_updates
        [loop.train_batch(next(train_loader)) for _ in range(250)]
        val_loss = loop.validation_loss(next(val_loader) for _ in range(100))
        loop.verify()
        record = {"epoch": epoch, "validation_grade_loss": val_loss,
                  "sampled_train_batches": 250,
                  "supervised_optimizer_updates": loop.supervised_updates - updates_before}
        history.append(record)
        torch.save({"state_dict": module.state_dict(), "pilot": record, "source_checkpoint": str(args.source_checkpoint)},
                   output / ("epoch_%02d.ckpt" % epoch))
        if val_loss is not None and val_loss < best[0]:
            best = (val_loss, epoch)
            torch.save({"state_dict": module.state_dict(), "pilot": record,
                        "source_checkpoint": str(args.source_checkpoint)}, output / "model_best_grade.ckpt")
    (output / "learning_curve.json").write_text(json.dumps(history, indent=2) + "\n")
    source_hash = _sha256(args.source_checkpoint)
    (output / "manifest.json").write_text(json.dumps({"condition": args.condition, "weight_decay": weight_decay,
        "source_checkpoint": str(args.source_checkpoint), "split_manifest": str(args.split_manifest),
        "source_checkpoint_sha256": source_hash, "best_epoch": best[1],
        "selection_rule": "earliest minimum validation grade loss"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
