import torch.nn as nn

from nndet.training.swa import SWACycleLinear


def test_swa_batch_norm_update_is_an_explicit_opt_in():
    callback = SWACycleLinear(
        swa_epoch_start=2, cycle_initial_lr=1e-3, cycle_final_lr=1e-4,
        num_iterations_per_epoch=1, update_statistics=True)
    assert callback.pl_module_contains_batch_norm(nn.BatchNorm3d(1))

    disabled = SWACycleLinear(
        swa_epoch_start=2, cycle_initial_lr=1e-3, cycle_final_lr=1e-4,
        num_iterations_per_epoch=1, update_statistics=False)
    assert not disabled.pl_module_contains_batch_norm(nn.BatchNorm3d(1))
