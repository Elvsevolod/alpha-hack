#!/usr/bin/env python3
"""Read organizer API limits without logging credentials or raw responses."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from dotenv import dotenv_values


ENDPOINT = "https://changellenge.up.railway.app/key/info"
NUMBER_FIELDS = (
    "spend", "total_spend", "max_budget", "rpm_limit", "tpm_limit", "tpd_limit",
    "max_parallel_requests",
)
TEXT_FIELDS = ("budget_duration", "budget_reset_at", "expires")


def load_key(env_file):
    if "OPENAI_API_KEY" in os.environ:
        key = os.environ["OPENAI_API_KEY"]
    else:
        key = dotenv_values(env_file).get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("OPENAI_API_KEY не задан в окружении или локальном .env.")
    if "\r" in key or "\n" in key:
        raise ValueError("OPENAI_API_KEY содержит недопустимый перевод строки.")
    return key.strip()


def fetch_info(key):
    # Ignore ~/.curlrc, keep TLS verification, and never follow redirects.
    result = subprocess.run(
        ["curl", "--disable", "--silent", "--show-error", "--fail",
         "--connect-timeout", "10", "--max-time", "30", "--header", "@-",
         "--write-out", "\n%{http_code}", ENDPOINT],
        input=f"Authorization: Bearer {key}\nAccept: application/json\n",
        text=True, capture_output=True,
    )
    body, _, status = result.stdout.rpartition("\n")
    if result.returncode or status != "200":
        status = status if len(status) == 3 and status.isdigit() else "unknown"
        raise ValueError(
            f"Проверка /key/info не удалась: curl={result.returncode}, HTTP={status}."
        )
    try:
        payload = json.loads(body)
    except ValueError:
        raise ValueError("API /key/info вернул некорректный JSON.") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        raise ValueError("В ответе API /key/info отсутствует объект info.")
    return payload["info"]


def number(value):
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return None


def snapshot(info, key, phase):
    limits = {name: number(info.get(name)) for name in NUMBER_FIELDS}
    for name in TEXT_FIELDS:
        value = info.get(name)
        limits[name] = value.replace(key, "[REDACTED]") if isinstance(value, str) else None
    limits["blocked"] = info.get("blocked") if type(info.get("blocked")) is bool else None
    maximum, spend = limits["max_budget"], limits["spend"]
    remaining = maximum - spend if maximum is not None and spend is not None else None
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "endpoint": ENDPOINT,
        "key_fingerprint": hashlib.sha256(key.encode()).hexdigest()[:16],
        "limits": limits,
        "remaining_budget": remaining,
        "remaining_percent": remaining / maximum * 100 if remaining is not None and maximum > 0 else None,
        "budget_exhausted": remaining <= 0 if remaining is not None else None,
    }


def compare(before, after):
    if not before.get("key_fingerprint") or before["key_fingerprint"] != after["key_fingerprint"]:
        return {"increase": None, "reason": "key_changed_or_unknown"}
    old, new = before.get("limits", {}), after["limits"]
    counter = "total_spend"
    first, last = number(old.get(counter)), number(new.get(counter))
    if first is None or last is None:
        if not old.get("budget_reset_at") or old["budget_reset_at"] != new["budget_reset_at"]:
            return {"increase": None, "reason": "budget_reset_or_unknown_period"}
        counter = "spend"
        first, last = number(old.get(counter)), number(new.get(counter))
    if first is None or last is None:
        return {"increase": None, "reason": "missing_spend_counter"}
    if last < first:
        return {"increase": None, "reason": "spend_counter_decreased"}
    return {"counter": counter, "increase": last - first}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("status", "before", "after"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--before", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.before and args.phase != "after":
        parser.error("--before используется только с after")
    try:
        baseline, baseline_reason = None, "no_before_snapshot"
        if args.before:
            try:
                baseline = json.loads(args.before.read_text())
            except (OSError, ValueError):
                baseline_reason = "before_snapshot_unreadable"
            if not isinstance(baseline, dict) or not isinstance(baseline.get("limits"), dict):
                baseline, baseline_reason = None, "before_snapshot_unavailable"
        if args.output and args.output.exists():
            raise ValueError("Файл снимка уже существует; выбери новый путь.")
        key = load_key(args.env_file)
        result = snapshot(fetch_info(key), key, args.phase)
        if args.phase == "after":
            result["comparison"] = compare(baseline, result) if baseline else {
                "increase": None, "reason": baseline_reason,
            }
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        print(rendered)
        return 3 if result["budget_exhausted"] or result["limits"]["blocked"] else 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
