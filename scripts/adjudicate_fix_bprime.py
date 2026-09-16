"""Adjudicate Fix B' against the rule pre-declared in ADR 0003 before its result.

Loads `grade_matched_lesions.json` (persisted by `gcalf_eval.run_eval` --
`grades`, `predictions`, `probabilities`, `patients`), computes the
point-estimate macro one-vs-rest AUROC, and a patient-level bootstrap 95% CI
via `gcalf_eval.grade_pilot.bootstrap_metrics` (2000 resamples, seed 2026 --
its own defaults). Applies the rule exactly as committed in ADR 0003's
"Fix B' adjudication rule" section:

  - CI lower bound <= 0.5: no ranking signal. Null; launch locked config.
  - Macro AUROC >= 0.70 AND CI lower bound > 0.5: real signal. Needs a
    separate ADR amendment before adopting into the matrix.
  - Otherwise: inconclusive at n=1 fold. Launch locked config, report both.

This script only computes and prints the verdict. It does not edit any ADR,
launch anything, or change any config -- the rule already decided what
happens in each branch; this just tells you which branch you're in.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gcalf_eval.grade_metrics import summarize_grade_matches
from gcalf_eval.grade_pilot import bootstrap_metrics

AUROC_FLOOR = 0.70
CI_PERCENTILES = (2.5, 97.5)


def compute_verdict(grades, predictions, probabilities, patients, samples=2000, seed=2026):
    point_summary = summarize_grade_matches(
        grades, predictions, 0, 0, matched_probabilities=probabilities,
    )
    point_estimate = point_summary["grade_macro_ovr_auroc"]

    resamples = bootstrap_metrics(grades, predictions, probabilities, patients, samples=samples, seed=seed)
    values = [row["grade_macro_ovr_auroc"] for row in resamples]
    valid = [value for value in values if value is not None]
    dropped = len(values) - len(valid)

    if not valid:
        raise ValueError("Every bootstrap resample had an undefined macro AUROC (a class missing "
                          "positives or negatives in every resample) -- cannot form a CI")

    lower, upper = (float(x) for x in np.percentile(valid, CI_PERCENTILES))

    if point_estimate is None:
        raise ValueError("Point-estimate macro AUROC is undefined on the full matched set")

    if lower <= 0.5:
        verdict = "NULL"
        action = "No ranking signal recovered. Launch the matrix on the locked configuration."
    elif point_estimate >= AUROC_FLOOR and lower > 0.5:
        verdict = "SIGNAL"
        action = ("Real signal recovered. Do NOT launch on the locked configuration without a "
                  "separate, explicit ADR amendment weighing whether to adopt Fix B''s "
                  "configuration for the matrix.")
    else:
        verdict = "INCONCLUSIVE"
        action = ("Inconclusive at n=1 fold. Launch the matrix on the locked configuration; "
                  "report both the point estimate and the CI, explicitly labeled inconclusive, "
                  "alongside the null.")

    return {
        "point_estimate_macro_ovr_auroc": point_estimate,
        "bootstrap_samples": samples,
        "bootstrap_samples_used": len(valid),
        "bootstrap_samples_dropped_undefined": dropped,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "auroc_floor": AUROC_FLOOR,
        "verdict": verdict,
        "action": action,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matched_lesions", type=Path,
                        help="Path to grade_matched_lesions.json (from gcalf_eval.run_eval)")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", action="store_true", help="Print the verdict as JSON only")
    args = parser.parse_args()

    data = json.loads(args.matched_lesions.read_text())
    for key in ("grades", "predictions", "probabilities", "patients"):
        if key not in data:
            raise ValueError(f"{args.matched_lesions} is missing required key: {key}")

    result = compute_verdict(
        data["grades"], data["predictions"], data["probabilities"], data["patients"],
        samples=args.samples, seed=args.seed,
    )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"Point estimate (macro OvR AUROC):  {result['point_estimate_macro_ovr_auroc']:.4f}")
    print(f"Bootstrap 95% CI ({result['bootstrap_samples_used']}/{result['bootstrap_samples']} "
          f"valid resamples): [{result['ci_95_lower']:.4f}, {result['ci_95_upper']:.4f}]")
    if result["bootstrap_samples_dropped_undefined"]:
        print(f"  ({result['bootstrap_samples_dropped_undefined']} resamples dropped: "
              f"macro AUROC undefined for that resample)")
    print(f"AUROC floor (ADR 0003): {result['auroc_floor']}")
    print()
    print(f"VERDICT: {result['verdict']}")
    print(result["action"])


if __name__ == "__main__":
    main()
