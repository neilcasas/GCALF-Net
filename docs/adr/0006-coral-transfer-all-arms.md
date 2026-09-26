# ADR 0006: CORAL + lesion transfer locked across all four matrix arms

**Status:** Accepted for the Task2202 20-run matrix

**Date:** 2026-09-26

**Supersedes:** ADR 0005's per-arm split, for the Task2202 matrix launch only. ADR 0005's
Day-0 probe, fold-0 detection safety gate, and re-measurement requirements are unchanged and
still apply.

## Decision

For the Task2202 study matrix, `model_cfg.head_grade_kwargs.grade_loss_type: coral` and the
grade-stratified lesion-centred transfer augmentation (`lesion_transfer_cfg.enabled: true`,
same locked values as `gcalf_final.yaml`) are the grade configuration for **all four arms**
(`baseline`, `lff`, `caf`, `full`), not only `full`. This is an operator decision made when
retargeting the matrix from Task2201 to Task2202, not a finding from any fold result.

Each arm now resolves from its Task2202 `_final` config
(`gcalf_task2202_baseline_final.yaml`, `gcalf_task2202_lff_final.yaml`,
`gcalf_task2202_caf_final.yaml`, `gcalf_task2202.yaml` for `full`); `cloud/vast/run_matrix.sh`
maps every arm to its `_final` variant. `scripts/check_matrix_fold.py` applies
`LOCKED_TRAINER_CFG` / `LOCKED_MODEL_CFG` / `LOCKED_DATALOADER_KWARGS` to every arm
unconditionally (the former `CONTROL_*` profile, previously required for `baseline`/`lff`/`caf`,
no longer exists).

## Rationale

ADR 0005 deliberately kept `baseline`/`lff`/`caf` as CE-loss, transfer-off detection controls so
that `full` was the single declared CORAL+transfer intervention, isolated from the architecture
comparison. This ADR removes that isolation for the Task2202 launch: the architecture arms are
now compared to each other with the grade configuration held constant at the final locked
values, rather than compared against a no-grade-intervention control.

## Consequences

- There is no longer an in-matrix CE/no-transfer arm. `scripts/check_fold0_detection_gate.py`'s
  `baseline` vs. `full` comparison still runs (it only reads `lesion_ap`/`picai_score`), but its
  failure fallback no longer relaunches a CE/transfer-off baseline, because none is configured;
  a failure stops the matrix and grade is reported as the pre-registered null.
- ADR 0005's required fold-0 control (`full` arm, `shuffle_labels: true`) is unaffected; it
  remains the only labels-shuffled run and is unchanged by this ADR.
- The thesis must report this as a change from the pre-registered ADR 0005 design: the Task2202
  matrix does not identify an architecture effect independent of the CORAL+transfer grade
  intervention, on top of ADR 0005's existing caveat that CORAL and lesion transfer are not
  separately identified from each other.
