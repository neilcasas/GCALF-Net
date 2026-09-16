"""Write the required pre-launch M5 cost-rung record from measured throughput."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_arm(value):
    try:
        name, seconds = value.split("=", 1)
        seconds = float(seconds)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--arm must be NAME=SECONDS_PER_STEP") from error
    if not name or seconds <= 0:
        raise argparse.ArgumentTypeError("--arm needs a name and positive seconds per step")
    return name, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True,
                        help="Repeat for each measured arm: NAME=SECONDS_PER_STEP")
    parser.add_argument("--contention-factor", type=float, required=True,
                        help="Measured four-concurrent / solo wall-time ratio")
    parser.add_argument("--hourly-instance-cost", type=float, required=True)
    parser.add_argument("--rung", choices=("full", "halved_batches", "reduced_14_run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps-per-run", type=int, default=150000)
    parser.add_argument("--concurrent-gpus", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing M5 record: {args.output}")
    if args.contention_factor <= 0 or args.hourly_instance_cost <= 0:
        parser.error("contention factor and hourly instance cost must be positive")
    if min(args.folds, args.steps_per_run, args.concurrent_gpus) <= 0:
        parser.error("folds, steps per run, and concurrent GPUs must be positive")
    arms = dict(args.arm)
    if len(arms) != len(args.arm):
        parser.error("each --arm name must be unique")
    projected_gpu_hours = sum(seconds * args.steps_per_run * args.folds / 3600 for seconds in arms.values())
    record = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "rung": args.rung,
        "measurement": {
            "seconds_per_step_by_arm": arms,
            "slowest_arm": max(arms, key=arms.get),
            "contention_factor_four_concurrent_vs_solo": args.contention_factor,
            "steps_per_run": args.steps_per_run,
            "folds": args.folds,
            "concurrent_gpus": args.concurrent_gpus,
        },
        "projection": {
            "gpu_hours": projected_gpu_hours,
            "instance_hours": projected_gpu_hours / args.concurrent_gpus,
            "instance_cost": projected_gpu_hours / args.concurrent_gpus * args.hourly_instance_cost,
            "serialized_gpu_hour_cost": projected_gpu_hours * args.hourly_instance_cost,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["projection"], sort_keys=True))


if __name__ == "__main__":
    main()
