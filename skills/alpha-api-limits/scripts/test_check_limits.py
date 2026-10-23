"""Offline checks; no requests to the organizer API."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import check_limits as limits


KEY = "fake-secret-never-log"


class LimitsTests(unittest.TestCase):
    def test_remaining_uses_current_period_not_lifetime_spend(self):
        result = limits.snapshot({"spend": 26, "total_spend": 153, "max_budget": 600}, KEY, "status")
        self.assertEqual(result["remaining_budget"], 574)
        self.assertFalse(result["budget_exhausted"])

    def test_unknown_and_zero_budget_are_distinct(self):
        for maximum, expected in [(None, None), (0, True), (1, True)]:
            with self.subTest(maximum=maximum):
                result = limits.snapshot({"spend": 1, "max_budget": maximum}, KEY, "before")
                self.assertIs(result["budget_exhausted"], expected)
        self.assertIsNone(limits.number(True))
        self.assertIsNone(limits.number(float("nan")))

    def test_response_credentials_are_omitted_from_snapshot(self):
        info = {"key": KEY, "metadata": {"token": KEY}, "key_alias": KEY, "expires": KEY}
        rendered = json.dumps(limits.snapshot(info, KEY, "before"))
        self.assertNotIn(KEY, rendered)
        self.assertNotIn("metadata", rendered)
        self.assertNotIn("key_alias", rendered)

    def test_key_uses_environment_before_env_file_including_empty_value(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text("OPENAI_API_KEY='file-secret'\n")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(limits.load_key(env), "file-secret")
            with patch.dict(os.environ, {"OPENAI_API_KEY": KEY}, clear=True):
                self.assertEqual(limits.load_key(env), KEY)
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
                with self.assertRaises(ValueError):
                    limits.load_key(env)

    def test_curl_receives_key_via_stdin_only_and_keeps_safe_defaults(self):
        response = subprocess.CompletedProcess([], 0, json.dumps({"key": KEY, "info": {"spend": 2}}) + "\n200", "")
        with patch.object(limits.subprocess, "run", return_value=response) as run:
            self.assertEqual(limits.fetch_info(KEY), {"spend": 2})
        command = run.call_args.args[0]
        self.assertNotIn(KEY, repr(command))
        self.assertIn(KEY, run.call_args.kwargs["input"])
        self.assertEqual(command[:2], ["curl", "--disable"])
        self.assertNotIn("--location", command)
        self.assertNotIn("--insecure", command)
        self.assertEqual(command[-1], limits.ENDPOINT)

    def test_request_failures_do_not_expose_response_or_stderr(self):
        for response in [
            subprocess.CompletedProcess([], 22, KEY + "\n401", KEY),
            subprocess.CompletedProcess([], 0, KEY + "\n200", ""),
            subprocess.CompletedProcess([], 0, json.dumps({"key": KEY}) + "\n200", ""),
        ]:
            with self.subTest(response=response.returncode):
                with patch.object(limits.subprocess, "run", return_value=response):
                    with self.assertRaises(ValueError) as raised:
                        limits.fetch_info(KEY)
                self.assertNotIn(KEY, str(raised.exception))

    def test_lifetime_delta_survives_period_reset(self):
        before = limits.snapshot({"spend": 90, "total_spend": 200, "budget_reset_at": "old"}, KEY, "before")
        after = limits.snapshot({"spend": 2, "total_spend": 205, "budget_reset_at": "new"}, KEY, "after")
        self.assertEqual(limits.compare(before, after), {"counter": "total_spend", "increase": 5})

    def test_period_delta_requires_same_known_reset(self):
        before = limits.snapshot({"spend": 2, "budget_reset_at": "same"}, KEY, "before")
        for reset, expected in [("same", 3), ("new", None), (None, None)]:
            with self.subTest(reset=reset):
                after = limits.snapshot({"spend": 5, "budget_reset_at": reset}, KEY, "after")
                self.assertEqual(limits.compare(before, after)["increase"], expected)

    def test_changed_key_or_decreased_counter_has_no_delta(self):
        before = limits.snapshot({"total_spend": 10}, KEY, "before")
        for key, spend in [("another-secret", 12), (KEY, 1)]:
            with self.subTest(spend=spend):
                after = limits.snapshot({"total_spend": spend}, key, "after")
                self.assertIsNone(limits.compare(before, after)["increase"])

    def test_before_after_persist_safe_snapshots_and_compare(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "before.json", Path(directory) / "after.json"
            responses = [{"spend": 2, "total_spend": 20, "max_budget": 100},
                         {"spend": 5, "total_spend": 23, "max_budget": 100}]
            output = io.StringIO()
            with patch.dict(os.environ, {"OPENAI_API_KEY": KEY}), redirect_stdout(output):
                with patch.object(limits, "fetch_info", side_effect=responses):
                    self.assertEqual(limits.main(["before", "--output", str(before)]), 0)
                    self.assertEqual(limits.main(["after", "--before", str(before), "--output", str(after)]), 0)
            result = json.loads(after.read_text())
            self.assertEqual(result["remaining_budget"], 95)
            self.assertEqual(result["comparison"]["increase"], 3)
            self.assertNotIn(KEY, output.getvalue() + before.read_text() + after.read_text())

    def test_missing_baseline_still_checks_after_failure_or_cancellation(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": KEY}), redirect_stdout(io.StringIO()):
            with patch.object(limits, "fetch_info", return_value={"spend": 1, "max_budget": 10}) as fetch:
                self.assertEqual(limits.main(["after", "--before", "/nonexistent/before.json"]), 0)
                fetch.assert_called_once()

    def test_blocked_or_exhausted_response_is_saved_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "snapshot.json"
            with patch.dict(os.environ, {"OPENAI_API_KEY": KEY}), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with patch.object(limits, "fetch_info", return_value={"blocked": True}):
                    self.assertEqual(limits.main(["before", "--output", str(output)]), 3)
                saved = output.read_text()
                with patch.object(limits, "fetch_info") as fetch:
                    self.assertEqual(limits.main(["before", "--output", str(output)]), 2)
                    fetch.assert_not_called()
                self.assertEqual(output.read_text(), saved)


if __name__ == "__main__":
    unittest.main()
