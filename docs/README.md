# GCALF-Net Documentation

This is the working repository: a copy of the released **PDHD-Net** taken at tag `pdhd-upstream`, with the upstream git remote removed. `git diff pdhd-upstream` is the exact set of thesis changes. The sibling `PDHD-Net/` checkout stays pristine as the released reference — build here, never there, and install only one of the two (both provide the `nndet` package).

Start here:

| Document | Purpose |
|---|---|
| [`THESIS_PLAN.md`](THESIS_PLAN.md) | Master scientific and technical implementation plan. |
| [`adr/0001-gcalf-architecture-and-cloud.md`](adr/0001-gcalf-architecture-and-cloud.md) | Accepted architecture, experiment, and cloud decisions from the grilling session. |
| [`CLOUD_DEPLOYMENT_PLAN.md`](CLOUD_DEPLOYMENT_PLAN.md) | Provider-neutral GPU VM, S3-compatible storage, Docker, recovery, security, and 20-run execution plan. |
| [`GLOSSARY.md`](GLOSSARY.md) | Canonical meanings of LFF, CAF, baseline, dataset version, and run artifacts. |

Execution order:

1. [`phases/PHASE_0_environment.md`](phases/PHASE_0_environment.md)
2. [`phases/PHASE_1_data_pipeline.md`](phases/PHASE_1_data_pipeline.md)
3. [`phases/PHASE_2_baseline.md`](phases/PHASE_2_baseline.md)
4. [`phases/PHASE_3_lff.md`](phases/PHASE_3_lff.md)
5. [`phases/PHASE_4_caf.md`](phases/PHASE_4_caf.md)
6. [`phases/PHASE_5_integration_ablation.md`](phases/PHASE_5_integration_ablation.md)
7. [`phases/PHASE_6_evaluation.md`](phases/PHASE_6_evaluation.md)
8. [`phases/PHASE_7_gradcam.md`](phases/PHASE_7_gradcam.md)

The ADR is authoritative when older thesis prose or upstream repository comments use conflicting terminology. In particular, CAF means true bidirectional windowed Q/K/V cross-attention; TransFuse BiFusion is not CAF.

Where the shipped code and the papers/READMEs disagree, **the code wins** — see `THESIS_PLAN.md §0` and the ADR's rev. 2 amendments. The three that bite hardest: `classifier_classes` is 5 (foreground-only, 0-indexed, no background class), the training schedule is 50 epochs × 2500 batches (not 1000 epochs, and there is no `EarlyStopping`), and `nndet/conf/train/smoke.yaml` already exists for tiny runs.
