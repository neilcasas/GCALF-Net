# Phase 5 — Full GCALF-Net Integration & Ablation Experiments

**Milestone:** M7 · **Depends on:** Phases 3 (LFF) + 4 (CAF) · **Blocks:** final evaluation
**Goal:** turn both modifications on together (full GCALF-Net), then run the **four-way ablation matrix** with identical folds/seeds so results are directly comparable. The registry (built in Phase 3) already makes this pure config work — no new model code.

**Definition of done:**
- [ ] `gcalf_full.yaml` (LFF + CAF) trains end-to-end.
- [ ] All four configs run on the **same folds and seeds**; outputs in per-config dirs.
- [ ] `gcalf_eval/collect_results.py` emits one comparison table (baseline as reference column).

---

## 5.1 The four configurations (all config-only, one codebase)

| Config file | `frequency_filter_type` | `fusion_type` | Answers |
|---|---|---|---|
| `baseline.yaml` | `wavelet` | `channel_light` | SOP 1 (adapted PDHD-Net baseline) |
| `lff_only.yaml` | `lff` | `channel_light` | SOP 2a (LFF contribution) |
| `caf_only.yaml` | `wavelet` | `windowed_cross_attention` | SOP 2b (CAF contribution) |
| `gcalf_full.yaml` | `lff` | `windowed_cross_attention` | SOP 2c + 3 (combined + significance) |

```yaml
# gcalf_configs/gcalf_full.yaml
model_cfg:
  encoder_kwargs:
    gcalf_cfg:
      frequency_filter_type: lff
      fusion_type: windowed_cross_attention
      freq_stages: [1, 3, 4]
      fusion_stages: [2, 5]
      lff:
        grid_size: [4, 8, 8]
        groups: 8
      caf:
        window_size: [2, 7, 7]
        num_heads: 4
        dropout: 0.0
```

**No code duplication:** every config drives the same `Encoder` via `registry.build_frequency_module` / `build_fusion_module`. Adding a 5th variant later = one YAML + (maybe) one registry dict entry.

## 5.2 Experiment protocol (fairness is the whole point)

1. **Fix one primary seed** across all four. Additional seeds are optional sensitivity runs for the baseline and full model, not a substitute for folds.
2. **Required matrix:** all four configs on all five official PI-CAI folds, for 20 required training runs. At the shipped schedule (50 epochs × 2500 batches + 10 SWA ≈ 150k steps) that is ~11–21 h per run, ~10–20 GPU-days total. The fold-0 pilots decide which of the three pre-committed budget options in `SPEC.md §12` applies — **choose before launching, not after seeing results.**
3. **Same schedule:** identical epochs / LR / batch / augmentation. Only the two module flags differ. No `EarlyStopping` (nnDetection has none) — a per-config stopping point would invalidate the comparison. If the schedule is shortened for budget, shorten it identically across all 20 runs.
4. **One directory per run:**
   ```
   gcalf_experiments/<config>_seed<NN>/
     ├── config_snapshot.yaml      # frozen copy of the exact config
     ├── git_commit.txt            # git rev-parse HEAD
     ├── env.lock.txt              # pip freeze
     ├── checkpoints/
     ├── logs/{metrics.csv, tb/}
     └── predictions/
   ```
5. **Launch matrix** — one script, `cloud/run_matrix.sh` (defined in `CLOUD_DEPLOYMENT_PLAN.md §14`; it runs locally too). It loops four configs × five folds × one primary seed, each writing to its own directory. Run independent jobs in parallel only when budget allows. Do not create a second copy under `gcalf_experiments/`.
6. **Immutable inputs:** every run records dataset-manifest SHA-256, split-file SHA-256, container image digest, config SHA-256, git commit, and command line.
7. **No test-fold tuning:** window size, grid size, learning rate, and stopping rules are frozen from unit/tiny/validation evidence before the required matrix starts.

## 5.3 Results collection

`gcalf_eval/collect_results.py` scans `gcalf_experiments/*/logs/metrics.csv` → one dataframe:

| config | macro-F1 | balanced-acc | GGG2 sens | GGG3 sens | quad-κ | FROC | AUROC |
|---|---|---|---|---|---|---|---|
| baseline | … (reference) | | | | | | |
| lff_only | Δ vs baseline | | | | | | |
| caf_only | Δ | | | | | | |
| gcalf_full | Δ | | | | | | |

Emit both raw and Δ-vs-baseline. This table *is* the thesis results section for SOPs 1–3. (Metric computation + stats live in `PHASE_6_evaluation.md`.)

## 5.4 Sanity: baseline invariance

Before trusting any ablation, re-run `baseline.yaml` and confirm it matches `baseline-v1` (Phase 2). If the registry refactor drifted the baseline, the whole comparison is invalid. This is a gate.

## 5.5 Risks & fallbacks

| Risk | Fallback |
|---|---|
| Compute cannot afford 20 required runs | Apply the pre-committed ladder in `SPEC.md §12`: full matrix → halve `num_train_batches_per_epoch` for *all* runs → 14-run reduced matrix (4 configs on fold 0 + baseline/full on all 5 folds). Never drop folds for some configs only — that breaks the paired Wilcoxon test. Record which option was used. |
| Full GCALF OOM when baseline fits | Reduce CAF windows, checkpoint CAF activations, then restrict true CAF to stage 5. BiFusion is only a separately named fallback. |
| Registry drift breaks baseline invariance | Revert to `baseline-v1` module construction path; add a regression test comparing outputs. |
| One config diverges | Isolate: it's config-only, so re-check that config's flags; the other three are unaffected. |

## 5.6 Deliverables & commit

- Branch `feat/gcalf-full` → commit "gcalf: full LFF+CAF config + 4-way ablation runner + results collector".
- Files: `gcalf_configs/gcalf_full.yaml`, `cloud/run_matrix.sh`, `gcalf_eval/collect_results.py`.
- Artifacts: four (or 4×folds×seeds) experiment dirs + the comparison table CSV.

**Next:** `PHASE_6_evaluation.md`.
