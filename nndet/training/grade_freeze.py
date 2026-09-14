"""Freeze the data-limited grade branch without terminating multitask training."""
import math

from pytorch_lightning.callbacks import Callback

from nndet.training.grade_refit import freeze_grade_head


class GradeHeadFreezeCallback(Callback):
    def __init__(self, monitor, patience, mode="max"):
        if mode not in ("min", "max"):
            raise ValueError(f"grade freeze mode must be min or max, got {mode}")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.bad_epochs = 0
        self.frozen = False

    def on_validation_end(self, trainer, pl_module):
        if self.frozen:
            return
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return
        score = float(score.detach().cpu()) if hasattr(score, "detach") else float(score)
        if not math.isfinite(score):
            return
        improved = self.best_score is None or (
            score > self.best_score if self.mode == "max" else score < self.best_score)
        if improved:
            self.best_score = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                freeze_grade_head(pl_module.model)
                self.frozen = True

    def on_train_epoch_start(self, trainer, pl_module):
        if self.frozen:
            freeze_grade_head(pl_module.model)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return {"best_score": self.best_score, "bad_epochs": self.bad_epochs, "frozen": self.frozen}

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.best_score = callback_state["best_score"]
        self.bad_epochs = callback_state["bad_epochs"]
        self.frozen = callback_state["frozen"]
        if self.frozen:
            freeze_grade_head(pl_module.model)
