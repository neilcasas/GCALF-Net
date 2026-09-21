"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import abstractmethod
from typing import Optional, Union, Callable

from loguru import logger

import torch
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.utilities.types import LRSchedulerConfig
from pytorch_lightning.utilities import rank_zero_warn

from nndet.training.learning_rate import CycleLinear


_AVG_FN = Callable[[torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor]


class BaseSWA(StochasticWeightAveraging):
    def __init__(
        self,
        swa_epoch_start: int,
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = torch.device("cpu"),
        update_statistics: Optional[bool] = False,
    ):
        """
        New Base Class for Stochastic Weighted Averaging

        Args:
            swa_epoch_start: Epoch to start SWA weight saving.
            avg_fn: Function to average saved weights. Defaults to None.
            device: Device to save averaged model. Defaults to 
                torch.device("cpu").
            update_statistics: Perform a final update of the normalization
                layers. Defaults to None.
                
        Notes: Does not support updating of norm weights after training
        """
        super().__init__(
            swa_epoch_start=swa_epoch_start,
            # The custom scheduler below replaces Lightning's SWALR, but 2.x
            # still validates this constructor argument.
            swa_lrs=1e-4,
            annealing_epochs=10,
            annealing_strategy="cos",
            avg_fn=avg_fn,
            device=device,
        )
        self.update_statistics = update_statistics
        logger.info(f"Initialize SWA with swa epoch start {self.swa_start}")

    def pl_module_contains_batch_norm(self, pl_module: 'pl.LightningModule'):
        return self.update_statistics and super().pl_module_contains_batch_norm(pl_module)

    def on_train_epoch_start(self,
                             trainer: 'pl.Trainer',
                             pl_module: 'pl.LightningModule',
                             ):
        """
        Repalce current lr scheduler with SWA scheduler
        """
        if (not self._initialized) and (self.swa_start <= trainer.current_epoch <= self.swa_end):
            self._initialized = True
            optimizer = trainer.optimizers[0]
            
            # move average model to request device.
            self._average_model = self._average_model.to(self._device or pl_module.device)

            _scheduler = self.get_swa_scheduler(optimizer)
            if isinstance(_scheduler, dict):
                scheduler = _scheduler["scheduler"]
                scheduler_kwargs = {key: value for key, value in _scheduler.items() if key != "scheduler"}
            else:
                scheduler = _scheduler
                scheduler_kwargs = {}
            # Lightning stores the scheduler object on the callback for
            # checkpoint state, and the wrapper config on the trainer.
            self._swa_scheduler = scheduler
            swa_scheduler_config = LRSchedulerConfig(scheduler=scheduler, **scheduler_kwargs)

            if trainer.lr_scheduler_configs:
                lr_scheduler = trainer.lr_scheduler_configs[0].scheduler
                rank_zero_warn(f"Swapping lr_scheduler {lr_scheduler} for {self._swa_scheduler}")
                trainer.lr_scheduler_configs[0] = swa_scheduler_config
            else:
                trainer.lr_scheduler_configs.append(swa_scheduler_config)

            if self._scheduler_state is not None:
                self._swa_scheduler.load_state_dict(self._scheduler_state)
            elif trainer.current_epoch != self.swa_start:
                rank_zero_warn(
                    "SWA is initializing after swa_start without checkpoint scheduler state."
                )
            if self.n_averaged is None:
                self.n_averaged = torch.tensor(
                    self._init_n_averaged, dtype=torch.long, device=pl_module.device)

        if (self.swa_start <= trainer.current_epoch <= self.swa_end
                and trainer.current_epoch > self._latest_update_epoch):
            if self.n_averaged is None:
                raise RuntimeError("SWA average count was not initialized")
            self.update_parameters(self._average_model, pl_module, self.n_averaged, self._avg_fn)
            self._latest_update_epoch = trainer.current_epoch

        if trainer.current_epoch == self.swa_end + 1:
            self.transfer_weights(self._average_model, pl_module)
            self.reset_batch_norm_and_save_state(pl_module)
            trainer.fit_loop.max_batches += 1
            trainer.fit_loop._skip_backward = True
            self._accumulate_grad_batches = trainer.accumulate_grad_batches
            trainer.accumulate_grad_batches = trainer.fit_loop.max_batches

    @abstractmethod
    def get_swa_scheduler(self, optimizer) -> Union[_LRScheduler, dict]:
        """
        Generate LR scheduler for SWA

        Args:
            optimizer: optimizer to wrap

        Returns:
            Union[_LRScheduler, dict]: If a lr scheduler is returned it will
                be stepped once per epoch. Can also return a whole config of
                the scheduler to customize steps.
        """
        raise NotImplementedError


class SWACycleLinear(BaseSWA):
    def __init__(self,
                 swa_epoch_start: int,
                 cycle_initial_lr: float,
                 cycle_final_lr: float,
                 num_iterations_per_epoch: int,
                 avg_fn: Optional[_AVG_FN] = None,
                 device: Optional[Union[torch.device, str]] = torch.device("cpu"),
                 update_statistics: Optional[bool] = None,
                 ):
        """
        SWA based on :class:`CycleLinear`

        Args:
            swa_epoch_start: Epoch to start SWA weight saving.
            cycle_initial_lr: initial learning rate of cycle
            cycle_final_lr: final learning rate of cycle
            num_iterations_per_epoch: number of train iterations per epoch
            avg_fn: Function to average saved weights. Defaults to None.
            device: Device to save averaged model. Defaults to 
                torch.device("cpu").
            update_statistics: Perform a final update of the normalization
                layers. Defaults to None.
        """
        super().__init__(
            swa_epoch_start=swa_epoch_start,
            avg_fn=avg_fn,
            device=device,
            update_statistics=update_statistics,
            )
        self.cycle_initial_lr = cycle_initial_lr
        self.cycle_final_lr = cycle_final_lr
        self.num_iterations_per_epoch = num_iterations_per_epoch

    def get_swa_scheduler(self, optimizer) -> Union[_LRScheduler, dict]:
        return {
            "scheduler": CycleLinear(
                optimizer=optimizer,
                cycle_num_iterations=self.num_iterations_per_epoch,
                cycle_initial_lr=self.cycle_initial_lr,
                cycle_final_lr=self.cycle_final_lr,
                ),
            "interval": "step",
        }
