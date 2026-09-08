#!/usr/bin/env python3
"""Run the documented four-GPU GCALF baseline admission gate.

The gate has two deliberately separate checks:

* the same deterministic, synthetic FP32 batch is evaluated on one GPU and
  through DDP on every rank; this verifies that the exact GCALF baseline path
  is numerically equivalent without writing a patient array to evidence; and
* the actual train step (forward, backward, gradient synchronization and an
  SGD step) is timed after 50 warm-up steps and for 200 synchronized steps.

The synthetic target follows the task's one-class csPCa / masked GGG2--5
contract.  This is a hardware/runtime admission gate, not a performance
experiment and not a substitute for the M1/M3 real-data gates.

Run ``--mode single`` first, then launch ``--mode ddp`` with torchrun.  Both
commands write only hashes, scalar results and telemetry under ``--evidence``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from hydra import initialize_config_module
from omegaconf import OmegaConf

from nndet.io.load import load_pickle
from nndet.ptmodule import MODULE_REGISTRY
from nndet.utils.config import compose


WARMUP_STEPS = 50
TIMED_STEPS = 200
NUMERICAL_TOLERANCE = 1e-5
MIN_THROUGHPUT_MULTIPLIER = 2.5
MIN_FREE_VRAM_FRACTION = 0.10
FIXED_BATCH_SEED = 20260908


class TrainingStep(torch.nn.Module):
    """Make nnDetection's training_step callable through DDP."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.module.training_step(batch, batch_idx=0)["loss"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Registered M1 task, e.g. Task2201_PICAI_csPCa")
    parser.add_argument("--mode", choices=("single", "ddp"), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_primary() -> bool:
    return rank() == 0


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_module(task: str, use_ddp_config: bool) -> tuple[torch.nn.Module, dict]:
    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    overrides = ["train=gcalf_ddp" if use_ddp_config else "train=gcalf_baseline"]
    cfg = compose(task, "config.yaml", overrides=overrides)
    plan = load_pickle(Path(str(cfg.host.plan_path)))
    module = MODULE_REGISTRY[cfg.module](
        model_cfg=OmegaConf.to_container(cfg.model_cfg, resolve=True),
        trainer_cfg=OmegaConf.to_container(cfg.trainer_cfg, resolve=True),
        plan=plan,
    )
    return module, plan


def make_fixed_batch(plan: dict, batch_size: int) -> Dict[str, Any]:
    """Create a reproducible non-patient batch with one GGG2--5 lesion/item."""
    channels = int(plan["architecture"]["in_channels"])
    depth, height, width = (int(v) for v in plan["patch_size"])
    generator = torch.Generator(device="cpu").manual_seed(FIXED_BATCH_SEED)
    data = torch.randn(
        batch_size, channels, depth, height, width, generator=generator, dtype=torch.float32
    )
    target = torch.zeros(batch_size, 1, depth, height, width, dtype=torch.int16)
    # Keep each cuboid comfortably within the minimum 10x112x96 patch.
    z0, z1 = max(1, depth // 4), max(2, (3 * depth) // 4)
    y0, y1 = max(2, height // 4), max(4, (3 * height) // 4)
    x0, x1 = max(2, width // 4), max(4, (3 * width) // 4)
    target[:, 0, z0:z1, y0:y1, x0:x1] = 1
    properties = [
        {
            "grades": {"1": 2 + (item % 4)},
            "grade_supervised": {"1": True},
        }
        for item in range(batch_size)
    ]
    instance_mapping = [{"1": 0} for _ in range(batch_size)]
    return {
        "data": data,
        "target": target,
        "instance_mapping": instance_mapping,
        "properties": properties,
    }


def batch_hash(batch: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(batch["data"].numpy().tobytes())
    digest.update(batch["target"].numpy().tobytes())
    digest.update(json.dumps(batch["instance_mapping"], sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(batch["properties"], sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        "data": batch["data"].to(device=device, non_blocking=True),
        "target": batch["target"].to(device=device, non_blocking=True),
        "instance_mapping": batch["instance_mapping"],
        "properties": batch["properties"],
    }


def slice_batch(batch: Dict[str, Any], item: int) -> Dict[str, Any]:
    return {
        "data": batch["data"][item:item + 1].contiguous(),
        "target": batch["target"][item:item + 1].contiguous(),
        "instance_mapping": [batch["instance_mapping"][item]],
        "properties": [batch["properties"][item]],
    }


def free_vram() -> Dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "free_fraction": float(free / total),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def timed_steps(
    step: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any],
    warmup_steps: int,
    timed_steps_count: int,
) -> Dict[str, float]:
    def one_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = step(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite gate loss: {loss.detach().item()}")
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    for _ in range(warmup_steps):
        one_step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = 0.0
    for _ in range(timed_steps_count):
        final_loss = one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "elapsed_seconds": elapsed,
        "throughput_global_steps_per_second": timed_steps_count / elapsed,
        "final_loss": final_loss,
    }


def load_single_result(evidence: Path) -> Dict[str, Any]:
    path = evidence / "single.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run --mode single first; missing {path}")
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    if args.mode == "ddp" and world_size() != 4:
        raise ValueError(f"DDP gate requires exactly four ranks, got {world_size()}")
    if args.mode == "single" and world_size() != 1:
        raise ValueError("Single gate must run without torchrun")

    torch.manual_seed(args.seed)
    torch.cuda.set_device(rank())
    device = torch.device("cuda", rank())
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)
    args.evidence.mkdir(parents=True, exist_ok=True)

    if args.mode == "ddp":
        dist.init_process_group(backend="nccl")

    module, plan = build_module(args.task, use_ddp_config=args.mode == "ddp")
    module = module.to(device=device, dtype=torch.float32).train()
    full_batch_cpu = make_fixed_batch(plan, batch_size=int(plan["batch_size"]))
    fixed_hash = batch_hash(full_batch_cpu)
    full_batch = move_batch(full_batch_cpu, device)
    step: torch.nn.Module = TrainingStep(module)
    if args.mode == "ddp":
        step = torch.nn.parallel.DistributedDataParallel(
            step, device_ids=[rank()], output_device=rank(), find_unused_parameters=True
        )

    # This is intentionally before any backwards pass: each rank evaluates
    # precisely the same global batch, so it is a direct FP32 numerical check.
    with torch.no_grad():
        fixed_loss = float(step(full_batch).detach().item())
    torch.cuda.synchronize()

    # Benchmark the actual distributed global batch: one sample per rank, or
    # all four samples on a single GPU.  lr=0 preserves the fixed workload yet
    # executes the production SGD code path and DDP gradient all-reduce.
    benchmark_cpu = (
        slice_batch(full_batch_cpu, rank()) if args.mode == "ddp" else full_batch_cpu
    )
    benchmark_batch = move_batch(benchmark_cpu, device)
    optimizer = torch.optim.SGD(step.parameters(), lr=0.0, momentum=0.9, nesterov=True)
    torch.cuda.reset_peak_memory_stats(device)
    timing = timed_steps(step, optimizer, benchmark_batch, WARMUP_STEPS, TIMED_STEPS)
    local_result = {
        "rank": rank(),
        "fixed_loss": fixed_loss,
        "timing": timing,
        "vram": free_vram(),
    }

    if args.mode == "single":
        result = {
            "mode": "single",
            "task": args.task,
            "seed": args.seed,
            "fixed_batch_seed": FIXED_BATCH_SEED,
            "fixed_batch_sha256": fixed_hash,
            "precision": "fp32",
            "warmup_steps": WARMUP_STEPS,
            "timed_steps": TIMED_STEPS,
            **local_result,
        }
        write_json(args.evidence / "single.json", result)
        print(json.dumps(result, sort_keys=True))
        return

    gathered: List[Dict[str, Any]] = [None] * world_size()
    dist.all_gather_object(gathered, local_result)
    if is_primary():
        single = load_single_result(args.evidence)
        fixed_diffs = [abs(item["fixed_loss"] - single["fixed_loss"]) for item in gathered]
        ddp_elapsed = max(item["timing"]["elapsed_seconds"] for item in gathered)
        ddp_throughput = TIMED_STEPS / ddp_elapsed
        throughput_multiplier = ddp_throughput / single["timing"]["throughput_global_steps_per_second"]
        gate = {
            "numerical_pass": max(fixed_diffs) <= NUMERICAL_TOLERANCE,
            "throughput_pass": throughput_multiplier >= MIN_THROUGHPUT_MULTIPLIER,
            "vram_pass": all(item["vram"]["free_fraction"] >= MIN_FREE_VRAM_FRACTION for item in gathered),
        }
        gate["pass"] = all(gate.values())
        result = {
            "mode": "ddp",
            "task": args.task,
            "seed": args.seed,
            "fixed_batch_seed": FIXED_BATCH_SEED,
            "fixed_batch_sha256": fixed_hash,
            "precision": "fp32",
            "warmup_steps": WARMUP_STEPS,
            "timed_steps": TIMED_STEPS,
            "single_fixed_loss": single["fixed_loss"],
            "fixed_loss_max_abs_difference": max(fixed_diffs),
            "numerical_tolerance": NUMERICAL_TOLERANCE,
            "single_throughput_global_steps_per_second": single["timing"]["throughput_global_steps_per_second"],
            "ddp_throughput_global_steps_per_second": ddp_throughput,
            "throughput_multiplier": throughput_multiplier,
            "minimum_throughput_multiplier": MIN_THROUGHPUT_MULTIPLIER,
            "minimum_free_vram_fraction": MIN_FREE_VRAM_FRACTION,
            "ranks": gathered,
            "gate": gate,
        }
        write_json(args.evidence / "ddp.json", result)
        write_json(args.evidence / "gate.json", result)
        print(json.dumps(result, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
