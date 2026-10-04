import argparse
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from moe_kernel_warmup_pairs import compare_pair, run, warmup_coverage
from test_moe_startup_pairs import record


def prepared_record():
    row = record()
    row["kernel_warmup_requested"] = True
    row["startup"] = {"kernel_warmup_unique_keys": 3, "kernel_warmup_ms": 5}
    row["compiler_after_request"].update(pipeline_creations=3, function_specializations=3)
    row["compiler_at_open"] = copy.deepcopy(row["compiler_after_request"])
    return row


class WarmupEvidenceGuards(unittest.TestCase):
    def test_lazy_compilation_invalidates_coverage_even_on_repeat(self):
        for key in ["pipeline_creations", "function_specializations", "library_compiles", "library_loads"]:
            repeated = prepared_record()
            repeated["request_index"] = 1
            repeated["compiler_after_request"][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                warmup_coverage([prepared_record(), repeated], True)
        warmup_coverage([prepared_record()], True)

    def test_missing_preparation_cannot_count_as_the_on_arm(self):
        for field, value in [("kernel_warmup_requested", False), ("startup", {"kernel_warmup_unique_keys": 0})]:
            row = {**prepared_record(), field: value}
            with self.assertRaises(ValueError):
                warmup_coverage([row], True)

    def test_pair_charges_preparation_and_requires_equal_outputs_and_repeats(self):
        off, on = prepared_record(), prepared_record()
        off["load_to_first_token_ms"] = 20
        on["load_to_first_token_ms"] = 25
        on["first_token_ms"] = 1
        self.assertEqual(compare_pair([off], [on])["total_startup_delta_ms"], 5)
        for field, value in [("token_sha256", "other"), ("request_index", 1), ("load_to_first_token_ms", None), ("load_to_first_token_ms", 0)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                compare_pair([off], [{**on, field: value}])
        with self.assertRaises(ValueError):
            compare_pair([off], [on, on])

    def test_monitor_failure_stops_process_and_retains_failure_metadata(self):
        import json
        import os
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            binary = out / "slow-probe"
            binary.write_text("#!/bin/sh\nprintf 'partial evidence\\n'\nsleep 30\n")
            binary.chmod(0o700)
            args = argparse.Namespace(cache="source", binary=binary, model=out, case="short",
                                      context=4096, max_new=64, shaping="greedy", timeout_seconds=10)
            snap = {"returncode": 0, "stdout": "", "stderr": ""}
            pids = []
            def failing_activity(owned_pid=None):
                if owned_pid is not None:
                    pids.append(owned_pid)
                    raise RuntimeError("activity monitor failed")
                return []
            with patch("moe_kernel_warmup_pairs.snapshot", return_value=snap), patch(
                    "moe_kernel_warmup_pairs.activity", side_effect=failing_activity):
                with self.assertRaisesRegex(RuntimeError, "activity monitor failed"):
                    run(args, out, out / "cache", True, "monitor-failed")
            self.assertEqual(len(pids),1)
            with self.assertRaises(ProcessLookupError):
                os.kill(pids[0],0)
            metadata = json.loads((out / "monitor-failed.metadata.json").read_text())
            self.assertIn("activity monitor failed",metadata["failure"])
            self.assertIsNotNone(metadata["exit_code"])

    def test_timeout_stops_owned_process_and_retains_partial_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            binary = out / "slow-probe"
            binary.write_text("#!/bin/sh\nprintf 'partial evidence\\n'\nsleep 30\n")
            binary.chmod(0o700)
            args = argparse.Namespace(cache="source", binary=binary, model=out, case="short",
                                      context=4096, max_new=64, shaping="greedy", timeout_seconds=0.1)
            snap = {"returncode": 0, "stdout": "", "stderr": ""}
            def started_activity(owned_pid=None):
                if owned_pid is not None:
                    # Establish partial output before testing timeout retention;
                    # scheduler delay is not the behavior under test.
                    deadline = time.monotonic() + 5
                    while not (out / "timed-out.jsonl").read_text() and time.monotonic() < deadline:
                        time.sleep(0.01)
                return []

            with patch("moe_kernel_warmup_pairs.snapshot", return_value=snap), patch("moe_kernel_warmup_pairs.activity", side_effect=started_activity):
                with self.assertRaisesRegex(RuntimeError, "partial evidence retained"):
                    run(args, out, out / "cache", True, "timed-out")
            import json
            metadata = json.loads((out / "timed-out.metadata.json").read_text())
            self.assertTrue(metadata["timeout"])
            self.assertIsNotNone(metadata["exit_code"])
            self.assertEqual((out / "timed-out.jsonl").read_text(), "partial evidence\n")


if __name__ == "__main__":
    unittest.main()
