# GCALF-Net Documentation

This is the working repository: a copy of the released **PDHD-Net** taken at tag `pdhd-upstream`
(commit `e2330cf`), with the upstream git remote removed. `git diff pdhd-upstream` is the exact
set of thesis changes. The sibling `PDHD-Net/` checkout stays pristine as the released reference —
build here, never there, and install only one of the two (both provide the `nndet` package).

Start here:

| Document | Purpose |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Master technical spec — the what and why. |
| [`ROADMAP.md`](ROADMAP.md) | The architecture split into milestones, each with L/V pass/fail gates. |
| [`adr/0001-gcalf-architecture-and-cloud.md`](adr/0001-gcalf-architecture-and-cloud.md) | Original architecture/cloud decisions. Amended by 0002 where noted (classifier class count, in particular). |
| [`adr/0002-ggg2-5-masked-grade-head-and-built-fdr-waf.md`](adr/0002-ggg2-5-masked-grade-head-and-built-fdr-waf.md) | **Current** decisions: GGG2–5 lesion-level protocol, two-head model (detection + masked grade head), FDR/WAF to be built (not the released code), preprocessing contract, budget, local/cloud split. Read this first for *why*. |
| [`CLOUD_DEPLOYMENT_PLAN.md`](CLOUD_DEPLOYMENT_PLAN.md) | Provider-neutral GPU VM, S3-compatible storage, Docker, recovery, security, 20-run execution plan. |
| [`GLOSSARY.md`](GLOSSARY.md) | Canonical meanings of FDR, WAF, LFF, CAF, grade head, and run artifacts. |

Execution order:

1. [`phases/PHASE_0_environment.md`](phases/PHASE_0_environment.md)
2. [`phases/PHASE_1_data_pipeline.md`](phases/PHASE_1_data_pipeline.md)
3. [`phases/PHASE_2_baseline.md`](phases/PHASE_2_baseline.md) — builds FDR, wires WAF, builds the masked grade head
4. [`phases/PHASE_3_lff.md`](phases/PHASE_3_lff.md)
5. [`phases/PHASE_4_caf.md`](phases/PHASE_4_caf.md)
6. [`phases/PHASE_5_integration_ablation.md`](phases/PHASE_5_integration_ablation.md)
7. [`phases/PHASE_6_evaluation.md`](phases/PHASE_6_evaluation.md)
8. [`phases/PHASE_7_gradcam.md`](phases/PHASE_7_gradcam.md)

**ADR 0002 is authoritative** wherever it conflicts with ADR 0001, older thesis prose, or upstream
repository comments — in particular: the protocol is **GGG2–5**, not GGG1–5 (PI-CAI has no
spatial GGG1 mask); the model is **two heads** (one detection class + a separate masked
grade head), not a single `classifier_classes`-way instance classifier; and **FDR and WAF do not
exist in the released code and must be built/wired** (`ARCHITECTURE.md §0`) — they are not two
`ModuleList` slots waiting to be swapped.

Where the shipped code and any document disagree, **the code wins** — verify against
`ARCHITECTURE.md §0` before trusting a claim about what the encoder currently does. Three things
that bite hardest: `dataset.json["labels"]` is `{"0": "csPCa"}` (one detection class — grade is
separate instance metadata, not a detection class count); the training schedule is 50 epochs ×
2500 batches + 10 SWA (not 1000 epochs, and there is no `EarlyStopping`); and
`nndet/conf/train/smoke.yaml` already exists for tiny runs.

**Known stale document:** `DATASCI17-THESIS-MASTERFILE.md` predates this session's decisions and
a newer version is pending. Do not treat it as a source of truth for anything this documentation
set decides; re-check it against `ARCHITECTURE.md`/the ADRs once the new version arrives.
