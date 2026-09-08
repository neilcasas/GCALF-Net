# Phase 5 — Full GCALF-Net Integration & Ablation Experiments

**Milestone:** M8 · **Depends on:** Phases 3 (LFF) + 4 (CAF) · **Blocks:** final evaluation
**Goal:** turn both modifications on together (full GCALF-Net), then run the four-way ablation
matrix at the rung committed in Phase 2's M5 (`ARCHITECTURE.md §9`), with identical folds/seeds
so results are directly comparable. The registry (Phase 2–4) already makes this pure config
work — no new model code.

**Definition of done:**
- [ ] **V:** `gcalf_full.yaml` (LFF + CAF) trains end-to-end.
- [ ] **V:** all four configs run on the **same folds and seeds**; outputs in per-config dirs.
- [ ] **V:** `gcalf_eval/collect_results.py` emits one comparison table (baseline as reference column).
- [ ] **L:** re-running `baseline.yaml` after every registry change still matches `baseline-v1` exactly.
- [ ] **L:** all arms share the same five-level plan, preprocessing/crop-QC manifest, case set,
      **frozen `fusion_levels`**, folds, seeds, schedule, augmentation, decoder, and heads.
- [ ] **L:** the only differences between arms are `{FDSF, LFF}` and `{WAF, CAF}`; both frequency
      modules return `(x_low, x_high)`, so no arm differs in pipeline shape.

---

## 5.1 The four configurations (config-only, one codebase)

| Config file | `frequency_filter_type` | `fusion_type` | Answers |
|---|---|---|---|
| `baseline.yaml` | `fdsf` | `waf` | SOP 1 (built FDSF/WAF baseline) |
| `lff_only.yaml` | `lff` | `waf` | SOP 2a (LFF contribution) |
| `caf_only.yaml` | `fdsf` | `caf` | SOP 2b (CAF contribution) |
| `gcalf_full.yaml` | `lff` | `caf` | SOP 2c + 3 (combined + significance) |

```yaml
# gcalf_configs/gcalf_full.yaml
model_cfg:
  encoder_kwargs:
    gcalf_cfg:
      frequency_filter_type: lff
      fusion_type: caf
      num_levels: 5
      fusion_levels: [0, 1, 2, 3, 4]   # starting point; frozen by M2 profiling
      lff: {grid_size: [4, 8, 8], groups: 1}   # in_channels is 3 at the input; 3 % 8 != 0
      caf: {window_size: [2, 7, 7], num_heads: 4, dropout: 0.0}
```

No code duplication: every config drives the same `Encoder` via
`registry.build_frequency_module` / `build_fusion_module`. A fifth variant later is one YAML,
maybe one registry branch.

## 5.2 Experiment protocol (fairness is the whole point)

1. **Fix one primary seed** across all four. Additional seeds are optional sensitivity runs for
   baseline and full only, never a substitute for folds.
2. **Matrix size follows Phase 2's M5 decision** — full 20-run matrix (planned rung, ADR 0002
   D10), or the halved-schedule / 14-run reduced rung if the fold-0 pilot forced a step down.
   Do not re-decide here; the rung was committed before any post-baseline fold ran.
3. **Same schedule:** identical epochs / LR / batch / augmentation across all four configs — only
   the two module flags differ. No `EarlyStopping`. If the schedule was shortened for budget
   (rung 2), it was shortened identically for all 20 runs.
4. **One directory per run:**
   ```
   gcalf_experiments/<config>_seed<NN>/
     ├── config_snapshot.yaml      # frozen copy of the exact config
     ├── git_commit.txt            # git rev-parse HEAD
     ├── env.lock.txt              # pip freeze
     ├── checkpoints/
     ├── logs/{metrics.csv, tb/}
     └── predictions/              # out-of-fold predictions preserved for Phase 6's bootstrap CIs
   ```
5. **Launch matrix** via `cloud/run_matrix.sh` (`docs/CLOUD_DEPLOYMENT_PLAN.md`); loops configs ×
   folds × the one primary seed, each writing its own directory. Parallelize independent jobs only
   when budget allows.
6. **Immutable inputs:** every run records dataset-manifest SHA-256, crop-QC-report SHA-256,
   split-file SHA-256, container image digest, config SHA-256, git commit, command line, and the
   resolved five-level feature/decoder plan. Lesion masks never select validation crop coordinates.
7. **No test-fold tuning:** window size, grid size, learning rate, and stopping rules are frozen
   from Phase 3/4's unit/tiny/validation evidence before this matrix starts.

## 5.3 Results collection

`gcalf_eval/collect_results.py` scans `gcalf_experiments/*/logs/metrics.csv` into one table:

| config | weighted F1 | macro-F1 | grade sens (per grade) | quad-κ | FROC | case AUROC |
|---|---|---|---|---|---|---|
| baseline | (reference) | | | | | |
| lff_only | Δ vs baseline | | | | | |
| caf_only | Δ | | | | | |
| gcalf_full | Δ | | | | | |

Emit raw and Δ-vs-baseline. Report the detection denominator (1,499 retained cases) and the
grade-matched denominator (grade-supervised lesions only) beside every grade metric — never
collapse them (`ARCHITECTURE.md §10`). Full metric computation and statistics live in
`PHASE_6_evaluation.md`.

## 5.4 Sanity: baseline invariance — **L gate, run before trusting any ablation number**

Re-run `baseline.yaml` and confirm it matches `baseline-v1` byte-for-byte on a fixed-seed
synthetic input (the regression test built in Phase 3 §3.3). If the registry refactor drifted the
baseline at any point across Phases 3–5, the whole ablation comparison is invalid regardless of
how plausible any individual config's numbers look.

## 5.5 Risks & fallbacks

| Risk | Fallback |
|---|---|
| Compute cannot afford the Phase 2 M5-committed rung after all | Step down to the next pre-committed rung (halved schedule → 14-run reduced), record which, and state the deviation from the M5 record in the thesis. Never drop folds for only some configs. |
| Full GCALF-Net OOMs when baseline fits | Reduce the shared CAF/WAF window or checkpoint both modules; if levels must be reduced, freeze one shared valid subset and rerun every affected arm. BiFusion remains separately named. |
| A fusion level is unaffordable | Freeze one reduced valid subset before the matrix and rerun both WAF and CAF arms with that subset; never use mismatched fusion sites. |
| Registry drift breaks baseline invariance | Revert to `baseline-v1`'s module-construction path; add/strengthen the regression test until it catches the drift. |
| One config diverges during training | Isolate: it's config-only, so re-check that config's flags first; the other three are architecturally unaffected. |

## 5.6 Deliverables & commit

- Branch `feat/gcalf-full` → commit "gcalf: full LFF+CAF config + 4-way ablation runner + results collector".
- Files: `gcalf_configs/gcalf_full.yaml`, `cloud/run_matrix.sh`, `gcalf_eval/collect_results.py`.
- Artifacts: four (or 4×folds×seeds) experiment directories + the comparison table CSV.

**Next:** `PHASE_6_evaluation.md`.
