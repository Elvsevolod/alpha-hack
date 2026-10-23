from __future__ import annotations

import numpy as np
import pandas as pd

from .util import digest


def freeze_partitions(task, fit, validation, ordered_time=None, min_rows=30):
    """Freeze diagnostics from fit and unlabelled validation features, never errors."""
    if len(validation) < 2 * min_rows:
        return []
    rules = []
    if ordered_time is not None and task.kind != "ranking":
        periods = sorted(pd.unique(ordered_time))
        blocks = min(8, len(periods), len(validation) // (2 * min_rows))
        if blocks >= 2:
            dates = pd.api.types.is_datetime64_any_dtype(ordered_time)
            groups = [[v.isoformat() if dates else float(v) for v in chunk]
                for chunk in np.array_split(np.asarray(periods, dtype=object), blocks)]
            rules.append({"kind": "periods", "column": task.time_column,
                "encoding": "datetime" if dates else "number", "groups": groups})
    if task.group_column:
        counts = validation[task.group_column].astype(str).value_counts()
        blocks = min(4, len(counts) // 2, len(validation) // (2 * min_rows))
        if blocks >= 2:
            totals, group_counts, assignment = [0] * blocks, [0] * blocks, {}
            for group, count in sorted(counts.items(), key=lambda item: (-item[1], digest(item[0]))):
                eligible = [index for index in range(blocks) if group_counts[index] < 2] or list(range(blocks))
                block = min(eligible, key=lambda index: totals[index])
                assignment[digest(group)] = block
                totals[block] += int(count)
                group_counts[block] += 1
            rules.append({"kind": "group_blocks", "column": task.group_column,
                "blocks": blocks, "group_hash_to_block": assignment})
    bins = min(4, len(validation) // (2 * min_rows))
    if bins >= 2 and task.kind != "ranking":
        excluded = set(task.categorical_columns + task.text_columns + task.targets + task.id_columns
            + task.series_columns + [task.time_column, task.group_column])
        for column in sorted(set(task.feature_columns) - excluded):
            if not pd.api.types.is_numeric_dtype(fit[column]):
                continue
            values = fit[column].to_numpy(dtype=float, na_value=np.nan)
            values = values[np.isfinite(values)]
            if len(np.unique(values)) < 2:
                continue
            cuts = np.unique(np.quantile(values, np.arange(1, bins) / bins)).tolist()
            rules.append({"kind": "feature_ranges", "column": column,
                "cuts": cuts, "cut_scope": "fit", "closed_side": "left"})
            if sum(rule["kind"] == "feature_ranges" for rule in rules) == 3:
                break
    return [{"kind": "feedback_partition", "source": "fit_and_unlabelled_validation_features",
        "report_id": f"{rule['kind']}:{rule['column']}:{digest(rule)}", "definition": rule} for rule in rules]


def registered_partitions(plan):
    result = {}
    for observation in plan.observations:
        if observation.get("kind") != "feedback_partition":
            continue
        rule, identifier = observation["definition"], observation["report_id"]
        expected = f"{rule['kind']}:{rule['column']}:{digest(rule)}"
        if identifier != expected or identifier not in plan.feedback_policy.report_ids or identifier in result:
            raise ValueError("Frozen diagnostic partition does not match its registered policy")
        result[identifier] = rule
    return result


def partition_masks(rule, X):
    values = X[rule["column"]]
    kind = rule["kind"]
    if kind == "feature_ranges":
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(numeric)
        buckets = np.searchsorted(np.asarray(rule["cuts"]), numeric, side="right")
        masks = {f"range_{i}": valid & (buckets == i) for i in range(len(rule["cuts"]) + 1)}
        masks["missing"] = ~valid
    elif kind == "periods":
        numeric = rule["encoding"] == "number"
        converted = pd.to_numeric(values, errors="coerce") if numeric else pd.to_datetime(values, errors="coerce", utc=True)
        masks = {f"period_block_{i}": converted.isin(group if numeric else pd.to_datetime(group, utc=True)).to_numpy()
            for i, group in enumerate(rule["groups"])}
        masks["outside_registered_periods"] = ~np.logical_or.reduce(list(masks.values()))
    elif kind == "group_blocks":
        buckets = values.astype(str).map(lambda value: rule["group_hash_to_block"].get(digest(value), -1)).to_numpy()
        masks = {f"group_block_{i}": buckets == i for i in range(rule["blocks"])}
        masks["outside_registered_groups"] = buckets == -1
    else:
        raise ValueError("Unsupported frozen diagnostic partition")
    # Empty feature-only cells cannot disclose a target; nonempty small cells suppress the family.
    return {label: mask for label, mask in masks.items() if mask.any()}


def partition_summary(rule):
    result = {"kind": rule["kind"], "column": rule["column"], "definition_hash": digest(rule)}
    if rule["kind"] == "feature_ranges":
        result.update(cuts=rule["cuts"], cut_scope="fit", intervals="[lower, upper)")
    elif rule["kind"] == "periods":
        result["period_blocks"] = [{"first": group[0], "last": group[-1]} for group in rule["groups"]]
    else:
        result["blocks"] = rule["blocks"]
    return result


def whole_ranking_groups(X, column, mask):
    membership = pd.DataFrame({"group": X[column].to_numpy(), "selected": np.asarray(mask)})
    counts = membership.groupby("group", dropna=False)["selected"].agg(["sum", "count"])
    return bool(((counts["sum"] == 0) | (counts["sum"] == counts["count"])).all()
        and int((counts["sum"] > 0).sum()) >= 2)
