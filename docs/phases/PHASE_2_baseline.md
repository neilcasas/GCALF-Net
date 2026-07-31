# Phase 2 — Baseline Adapted PDHD-Net (3-channel bpMRI)

**Milestones:** M2 (forward) → M3 (tiny train) → M4 (full train/eval) ⭐ · **Depends on:** Phase 1 · **Blocks:** LFF/CAF (they must beat this)
**Goal:** the adapted 3-channel PDHD-Net (`frequency_filter_type=wavelet`, `fusion_type=channel_light` — i.e. the code as-is) trains and evaluates end-to-end on PI-CAI. **This is the thesis's first real result and it must exist before any LFF/CAF work.**

**Definition of done:**
- [ ] M2: forward pass on `(1,3,D,H,W)` runs; overfits 1–2 samples to ~0 loss.
- [ ] M3: 2-epoch run on `Task9xx_PICAI_TINY` → predict → picai_eval produces numbers.
- [ ] M4: full training on PI-CAI folds; baseline metrics reported (confusion matrix, macro-F1, per-class sensitivity, FROC); **tagged commit `baseline-v1`**.

---

## 2.1 The only code change: 3 input channels

nnDetection reads `in_channels` from the **plan**, not from hardcoded values — so preparing the data with 3 modalities (Phase 1) already makes the fingerprint set `in_channels=3`. Two things to verify/set:

1. `plan["architecture"]["in_channels"] == 3` — consumed at `nndet/ptmodule/retinaunet/base.py::_build_encoder` (~line 531). Auto-derived from the 3 modalities; assert it.
2. `plan["architecture"]["classifier_classes"] == 4` (GGG2–5, **foreground only**) — consumed at `_build_head_classifier` (~line 596) and `_build_head` (~line 497), derived at `nndet/planning/architecture/boxes/base.py:89` as `len(dataset.json["labels"])`. Background is implicit in the sigmoid focal loss and is never a channel. GGG1 and benign tissue are zero-instance background in the M1 task; if the plan differs from four, fix the data rather than the plan.
3. The Swin branch reads the same `in_channels`: `SwinTransformer3D(in_chans=in_channels, …)` in `modular.py` line 97 — no separate change.

That's it. **No architecture edits for the baseline.** Resist the urge to touch `modular.py` yet — the registry refactor is Phase 3.

## 2.2 M2 — forward pass & overfit

1. **Instantiate**: `RetinaUNetV001.from_config_plan(...)` (`nndet/ptmodule/retinaunet/v001.py`) with the Phase-1 plan.
2. **Forward**: feed `torch.randn(1,3,20,320,320)` (or the plan's `patch_size`); assert the encoder returns feature maps, BiFPN runs, heads produce per-anchor outputs. No shape errors.
3. **Overfit 2 samples** (`RUN_GCALF_M2=1 python -m pytest -q -s tests/test_overfit.py`): train on one positive and one benign real case for 200 steps; assert the final 10-step mean total loss is ≤0.1 and ≤10% of the initial mean, the positive matched anchor predicts its GGG2–5 class at ≥0.9 probability, and the benign patch stays at ≤0.1 foreground probability. **This is the single most valuable early test** — it proves data→label→loss→backprop is correctly wired end to end.

### Expected tensor shapes (confirm exact numbers from your plan)

| Point | Shape `(B,C,D,H,W)` | Source |
|---|---|---|
| Input patch | `(2, 3, 20, 320, 320)` | plan `patch_size` |
| Swin after patch-partition `(2,4,4)` | `(2, embed, 10, 80, 80)` | `SwinTransformer3D`, `embed=start_channels` |
| CNN stem (stage 0) | `(2, 32, 20, 320, 320)` | `start_channels=32` |
| CNN stage i | ch `32·2^i` (≤ `max_channels`), spatial ÷ strides | encoder stages |
| Swin stage i | `(2, 32·2^i, d_i,h_i,w_i)` | `transformer_out_channels[i]` |
| Wavelet freq module (stages 1,3,4) | in==out, shape-preserving | `WaveletSpatialFusion.forward(x)` |
| Fusion (stages 2,5) | `(2, C_i, d,h,w)` | `MemoryEfficientFusion(cnn, interp(swin))` |
| BiFPN outputs | list of `(2, fpn_channels, d,h,w)` | `ClassBiFPN` |
| Classifier head | per-anchor → `(N_anchors, 4)` | detection is per-anchor; four foreground GGG2–5 classes |

## 2.3 M3 — tiny end-to-end (plumbing, not accuracy)

Don't hand-roll epoch overrides — `nndet/conf/train/smoke.yaml` already exists and is exactly this: `max_num_epochs: 2`, `num_train_batches_per_epoch: 20`, `num_val_batches_per_epoch: 10`, `swa_epochs: 2`, `debug.num_cases_val: 2`. Select it as a Hydra config group:

```bash
python GCALF-Net/scripts/train.py Task9xx_PICAI_TINY -o train=smoke
python GCALF-Net/scripts/predict.py Task9xx_PICAI_TINY RetinaUNetV001_D3V001_3d -f 0
python gcalf_eval/run_eval.py --task Task9xx_PICAI_TINY --fold 0
```

Two CLI facts that bite: the model argument is the full identifier `<module>_<plan>` (`RetinaUNetV001_D3V001_3d`, as in the README), not the bare module name; and `-o` takes Hydra overrides, so the epoch key — if you ever do need it directly — is `trainer_cfg.max_num_epochs`, not `exp.num_epochs`.

Exit when the pipeline runs start→finish and writes `metrics.csv`. Numbers will be garbage on 6 cases — that's fine.

## 2.4 M4 — full baseline train/eval ⭐

- Train the shipped schedule per fold, unchanged: `nndet/conf/train/v001.yaml` → `max_num_epochs: 50`, `num_train_batches_per_epoch: 2500`, `swa_epochs: 10`, SGD `initial_lr: 0.01` with poly decay, `precision: 16`. That is ≈150k steps ≈ 11–21 h per fold. There is **no `EarlyStopping`** in nnDetection and you must not add one — a stopping rule that fires at different points per config would invalidate the four-way comparison. Checkpoint selection is already handled by `monitor_key: mAP_IoU_0.14_0.90_0.05_MaxDet_100`.
- **Record measured seconds/step and total wall time for fold 0.** This is the pilot number that decides which matrix option §12 of `SPEC.md` selects; the whole 20-run budget hangs off it. If the schedule must be shortened, shorten it identically for all 20 runs.
- Class weighting in the focal classification loss derived from the generated M1 GGG2–5 instance counts (`nndet/losses/{classification,modern_classification}.py`).
- Enable AMP (`autocast`) — no FFT in the baseline so no autocast exclusions yet.
- Predict + evaluate (details in `PHASE_6_evaluation.md`): FROC/AUROC via picai_eval, plus 4-class GGG2–5 confusion matrix / macro-F1 / per-class sensitivity (esp. GGG2 vs GGG3 — the headline).
- **Tag `baseline-v1`.** Every later config is compared against this exact commit + these numbers.

## 2.5 Batch / memory / checkpoint

- Batch auto-planned by nnDetection (expect 2–4 on 11–16 GB). If forced to 1, use grad-accum 2–4.
- `ModelCheckpoint` on val detection AP; frequent checkpoints (cloud disconnects — see Phase 6/cloud notes).
- Resume via Lightning `ckpt_path`.

## 2.6 Risks & fallbacks (this phase)

| Risk | Fallback |
|---|---|
| Full multi-task detection won't converge / too slow | **Classifier-fallback model** (`SPEC.md §14`): `Encoder` + global-pool + linear GGG head, `case_ISUP` labels, focal loss. Still the baseline for the 4-way ablation; drops FROC/Dice. Build it under `nndet/arch/encoder/gcalf/classifier_model.py`. Note this model *does* have a benign class (6-way over `case_ISUP ∈ {0..5}`) because it is a plain classifier — that is a property of the fallback, not of the detection model. |
| OOM at plan's patch size | Reduce `patch_size` in the plan; batch 1 + grad-accum; AMP. |
| Classifier head shape/anchor confusion | Verify `classifier_classes=4`; inspect one prediction's per-anchor logits before scaling up. |
| Baseline numbers implausibly low | Re-run overfit test; check label mapping (Phase 1 sanity); check channel order. |

## 2.7 Deliverables & commit

- Branch `feat/baseline-3ch` → tag `baseline-v1`.
- Files: `tests/test_overfit.py`, `gcalf_eval/run_eval.py`, baseline `metrics.csv` + confusion matrix under `gcalf_experiments/baseline_seedNN/`.
- Commit "baseline: 3-channel adapted PDHD-Net trains + evals on PI-CAI (M4)".

**Next:** `PHASE_3_lff.md`.
