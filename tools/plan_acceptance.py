"""Evaluate the frozen v076/v081/candidate protocol without automatic promotion."""
from __future__ import annotations

import math
import statistics


def evaluate(manifest, results):
    versions = list(manifest["versions"])
    if set(versions) != {"v076", "v081", "candidate"}:
        raise ValueError("All three immutable comparison arms are required")
    seeds = manifest.get("seeds", [manifest.get("seed", 42)])
    tasks = manifest.get("tasks", list(manifest["splits"]))
    expected = {(version, task, seed) for version in versions for task in tasks for seed in seeds}
    observed = {}
    types = set()
    for row in results:
        key = row["version"], row["task"], row["seed"]
        if key not in expected or key in observed:
            raise ValueError("Unexpected or duplicate version/task/seed result")
        split = manifest["splits"][row["task"]]
        required = {"protocol_hash": manifest["protocol_hash"], "resources": manifest["resources"],
            "data_hash": split["data_hash"], "evaluation_keys_hash": split["evaluation_keys_hash"],
            "output_schema": split["task"]["output"]}
        if any(row.get(name) != value for name, value in required.items()):
            raise ValueError("Incomparable data, output schema, evaluation rows or resource limits")
        types.add(row["evidence_type"])
        observed[key] = row
    if len(types) > 1:
        raise ValueError("Mock, offline recipe and full Qwen evidence cannot be mixed")
    usable = lambda row: row.get("valid_csv") is True and type(row.get("raw_metric")) in (int, float) and math.isfinite(row["raw_metric"])
    complete = set(observed) == expected
    table, conditions = {}, []
    stress_conditions = []
    useful = False
    for task in tasks:
        regression = manifest["splits"][task]["task"]["metric"]["name"] == "rmse"
        summary = {}
        for reference in ("v076", "v081"):
            pairs = []
            for seed in seeds:
                a, b = observed.get((reference, task, seed)), observed.get(("candidate", task, seed))
                if a and b and usable(a) and usable(b):
                    change = a["raw_metric"] - b["raw_metric"] if regression else b["raw_metric"] - a["raw_metric"]
                    pairs.append({"seed": seed, "reference": a["raw_metric"], "candidate": b["raw_metric"], "signed_improvement": change,
                        "reference_status": a.get("status"), "candidate_status": b.get("status"),
                        "reference_fallback": a.get("fallback"), "candidate_fallback": b.get("fallback")})
            mean = statistics.mean(row["signed_improvement"] for row in pairs) if pairs else None
            summary[reference] = {"pairs": pairs, "mean_signed_improvement": mean,
                "uncertainty": "single_screening_trajectory" if len(pairs) < 3 else "descriptive_repeats_no_significance_claim"}
            if reference == "v076":
                no_worse = mean is not None and mean >= 0
                pair_guard = len(pairs) == len(seeds) and all(row["signed_improvement"] >= (-.005 * row["reference"] if regression else -.002) for row in pairs)
                majority = sum(row["signed_improvement"] > 0 for row in pairs) > len(seeds) / 2
                practical = mean is not None and (mean / max(1e-12, statistics.mean(row["reference"] for row in pairs)) >= .01 if regression else mean >= .005)
                useful |= practical
                conditions.append(no_worse and pair_guard and majority)
                stress = []
                for seed in seeds:
                    a, b = observed.get((reference, task, seed)), observed.get(("candidate", task, seed))
                    if a and b and all(type(row.get("stress_metric")) in (int, float) and math.isfinite(row["stress_metric"]) for row in (a, b)):
                        stress.append(a["stress_metric"] - b["stress_metric"] if regression else b["stress_metric"] - a["stress_metric"])
                # Unexplained opposing stress evidence cannot pass the gate.
                stress_consistent = len(stress) == len(seeds) and statistics.mean(stress) >= 0
                stress_conditions.append(stress_consistent)
                summary[reference]["stress_signed_improvements"] = stress
                summary[reference]["stress_direction_consistent"] = stress_consistent
        table[task] = summary
    valid_rate = sum(usable(row) for row in observed.values()) / max(1, len(expected))
    live = bool(observed) and types == {"full_qwen_agent"} and all(row.get("real_qwen") is True for row in observed.values())
    eligible = complete and valid_rate == 1 and live and len(seeds) >= 3 and useful and all(conditions) and all(stress_conditions)
    return {"status": "eligible_for_quality_review" if eligible else "quality_advantage_unestablished",
        "complete": complete, "valid_submission_rate": valid_rate, "comparisons": table,
        "evidence_types": sorted(types), "full_agent_evidence": live, "practical_gain_observed": useful,
        "all_failures_and_fallbacks_counted": True, "automatic_promotion": False,
        "platform_status": "not_submitted", "technical_status": "requires_separate_checks"}
