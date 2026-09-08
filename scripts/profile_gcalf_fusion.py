"""Profile fixed FDSF fusion subsets at a resolved GCALF plan size.

The selected subset is the first suffix of [0, 1, 2, 3, 4] that completes a
forward/backward step below the configured VRAM limit.  Removing shallow levels
first is deterministic and removes the most spatially expensive attention maps.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from hydra import initialize_config_module
from omegaconf import OmegaConf

from nndet.io.load import load_pickle
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
from nndet.utils.config import compose


LEVELS = [0, 1, 2, 3, 4]


def profile_subset(task: str, plan: dict, levels: list, train_config: str) -> dict:
    override = "model_cfg.encoder_kwargs.gcalf_cfg.fusion_levels=[{}]".format(
        ",".join(str(level) for level in levels)
    )
    cfg = compose(task, "config.yaml", overrides=[f"train={train_config}", override])
    model_cfg = OmegaConf.to_container(cfg["model_cfg"], resolve=True)
    model = RetinaUNetV001.from_config_plan(
        model_cfg=copy.deepcopy(model_cfg),
        plan_arch=copy.deepcopy(plan["architecture"]),
        plan_anchors=copy.deepcopy(plan["anchors"]),
    ).cuda().train()
    inputs = torch.randn(1, 3, *plan["patch_size"], device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with torch.cuda.amp.autocast():
            detections, _, segmentation = model(inputs)
            loss = sum(value.float().mean() for value in detections.values())
            loss = loss + segmentation["seg_logits"].float().mean()
        loss.backward()
        torch.cuda.synchronize()
        return {
            "fusion_levels": levels,
            "status": "ok",
            "step_seconds": time.perf_counter() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        torch.cuda.empty_cache()
        return {"fusion_levels": levels, "status": "oom", "error": str(error)}
    finally:
        del model, inputs
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-config", choices=("gcalf_baseline", "gcalf_caf"), default="gcalf_baseline")
    parser.add_argument("--headroom-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 <= args.headroom_fraction < 1:
        raise ValueError("--headroom-fraction must be in [0, 1)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fusion profiling")

    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    plan = load_pickle(args.plan_path)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    limit = int(total_memory * (1 - args.headroom_fraction))
    records = [profile_subset(args.task, plan, [level], args.train_config) for level in LEVELS]
    selected = None
    for start in range(len(LEVELS)):
        record = profile_subset(args.task, plan, LEVELS[start:], args.train_config)
        records.append(record)
        if record["status"] == "ok" and record["peak_memory_bytes"] <= limit:
            selected = record["fusion_levels"]
            break

    result = {
        "task": args.task,
        "train_config": args.train_config,
        "plan_path": str(args.plan_path),
        "headroom_fraction": args.headroom_fraction,
        "total_memory_bytes": total_memory,
        "accepted_peak_memory_bytes": limit,
        "records": records,
        "selected_fusion_levels": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if selected is None:
        raise SystemExit("No fusion subset met the configured VRAM headroom")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
