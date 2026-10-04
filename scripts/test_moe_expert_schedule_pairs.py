import argparse
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from moe_expert_schedule_pairs import (
    EXPERIMENTS, HostMonitor, ProbeFailure, arm_environment, compare_pair,
    cpu_seconds, digest, host_eligibility, main, parse_activity, run_probe, validate_schedule,
)


def arguments(experiment="overlap", **values):
    defaults = dict(experiment=experiment, shaping="greedy", repeats=2, context=4096, max_new=64,
                    case="short-explanation", page_cache_label="unknown", quiet_cpu_percent=5,
                    sustained_seconds=2, quiet_seconds=0, monitor_interval=0.01, timeout_seconds=0.1)
    defaults.update(values)
    return argparse.Namespace(**defaults)


def records(experiment="overlap", arm="off", manifest="model"):
    residency, overlap, prep = EXPERIMENTS[experiment][arm]
    row = dict(protocol="moe-startup-v1", discarded_warmups=0, prefix_reuse=False,
               open_index=0, request_index=0, precompiled_requested=False, pipeline_cache_requested=False,
               kernel_warmup_requested=False, shared_read_overlap_requested=overlap,
               shared_read_overlap_effective=overlap, mapped_demand_prep_requested=prep,
               mapped_demand_prep_effective=prep, mapped_demand_budget_bytes=16777216,
               compiler_after_request={}, manifest_sha256=manifest, case="short-explanation", context=4096,
               max_new=64, expert_cache_slots=16, residency=residency, shaping="greedy", seed=42,
               token_sha256="identical tokens", new_tokens=8, prompt_tokens=61, stop="maxTokens",
               load_ms=20, first_token_ms=12, load_to_first_token_ms=32, request_ms=60,
               prefill_ms=10, decode_ms=40, startup={"total_open_ms": 17},
               expert_schedule={"shared_submissions_cumulative": int(overlap),
                                "mapped_prepare_calls_cumulative": int(prep != "off"),
                                "mapped_advice_calls_cumulative": int(prep == "advice"),
                                "mapped_advice_failures_cumulative": 0,
                                "mapped_page_touches_cumulative": int(prep == "touch")})
    return [row, {**copy.deepcopy(row), "request_index": 1, "load_to_first_token_ms": None}]


def host_sample(at=0, cpu=0, power="Now drawing from 'AC Power'", thermal="", **extra):
    result = dict(monotonic_s=at, background_cpu_percent=cpu,
                  max_background_cpu_percent=cpu, background_cpu_by_pid={"44": cpu} if cpu else {},
                  power=dict(returncode=0, stdout=power), thermal=dict(returncode=0, stdout=thermal),
                  activity=[], other_model_processes=[])
    result.update(extra)
    return result


class SchedulingEvidence(unittest.TestCase):
    def test_arms_scrub_inherited_diagnostics_and_isolate_each_control(self):
        for experiment, arms in EXPERIMENTS.items():
            for arm, (expected_residency, overlap, prep) in arms.items():
                residency, env = arm_environment(experiment, arm, Path("cache"),
                                                {"PATH": "safe", "TURBOSPARK_PHASES": "1", "TURBOSPARK_EXPERT_NOCACHE": "1"})
                with self.subTest(experiment=experiment, arm=arm):
                    self.assertEqual(residency, expected_residency)
                    self.assertEqual(env["TURBOSPARK_QWEN_SHARED_READ_OVERLAP"], str(int(overlap)))
                    self.assertEqual(env["TURBOSPARK_MAPPED_DEMAND_PREP"], prep)
                    self.assertEqual(env["PATH"], "safe")
                    self.assertNotIn("TURBOSPARK_PHASES", env)
                    self.assertNotIn("TURBOSPARK_EXPERT_NOCACHE", env)
                    for key in ("TURBOSPARK_METAL_KERNEL_WARMUP", "TURBOSPARK_METAL_PRECOMPILED", "TURBOSPARK_METAL_PIPELINE_CACHE"):
                        self.assertEqual(env[key], "0")

    def test_requested_modes_must_actually_engage(self):
        for experiment, arm in (("overlap", "on"), ("mapped", "advice"), ("mapped", "touch")):
            args = arguments(experiment)
            validate_schedule(records(experiment, arm), args, arm)
            mutations = [("shared_read_overlap_effective", False)] if experiment == "overlap" else [
                ("mapped_demand_prep_effective", "off"), ("mapped_demand_budget_bytes", 0)]
            for field, value in mutations:
                rows = records(experiment, arm)
                rows[1][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_schedule(rows, args, arm)
        for counter in ("shared_submissions_cumulative", "mapped_advice_calls_cumulative", "mapped_page_touches_cumulative"):
            experiment, arm = (("overlap", "on") if counter.startswith("shared") else
                               ("mapped", "advice" if "advice" in counter else "touch"))
            rows = records(experiment, arm)
            rows[0]["expert_schedule"][counter] = 0
            with self.subTest(counter=counter), self.assertRaises(ValueError):
                validate_schedule(rows, arguments(experiment), arm)
        rows = records("mapped", "advice")
        rows[0]["expert_schedule"]["mapped_advice_failures_cumulative"] = 1
        with self.assertRaisesRegex(ValueError, "advice"):
            validate_schedule(rows, arguments("mapped"), "advice")

    def test_parity_covers_outputs_shapes_and_complete_repeats(self):
        off, on = records(), records(arm="on")
        on[0]["first_token_ms"] = 8
        on[0]["load_to_first_token_ms"] = 28
        on[0]["decode_ms"] = 20
        delta = compare_pair({"off": off, "on": on})["on"][0]
        self.assertEqual(delta["load_delta_ms"], 0)
        self.assertEqual(delta["first_token_delta_ms"], -4)
        self.assertEqual(delta["load_to_first_token_delta_ms"], -4)
        self.assertEqual(delta["decode_delta_tokens_per_second"], 200)
        self.assertIsNone(compare_pair({"off": off, "on": on})["on"][1]["load_to_first_token_delta_ms"])
        for field, value in (("token_sha256", "changed"), ("new_tokens", 7), ("stop", "endOfTurn"),
                             ("prompt_tokens", 62), ("seed", 1), ("shaping", "sampled")):
            changed = records(arm="on")
            changed[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_pair({"off": off, "on": changed})
        for changed in (off[:1], list(reversed(off)), [{**off[0], "request_index": 2}, off[1]]):
            with self.assertRaises(ValueError):
                validate_schedule(changed, arguments(), "off")

    def test_invalid_timings_and_hidden_warmups_are_rejected(self):
        for field, value in (("decode_ms", 0), ("first_token_ms", None), ("load_ms", float("nan")),
                             ("kernel_warmup_requested", True), ("discarded_warmups", 1), ("context", 1024),
                             ("case", "wrong prompt"), ("expert_cache_slots", 32), ("prefix_reuse", True),
                             ("load_to_first_token_ms", None)):
            rows = records()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_schedule(rows, arguments(), "off")
        rows = records()
        rows[1]["load_to_first_token_ms"] = 32
        with self.assertRaisesRegex(ValueError, "reset request"):
            validate_schedule(rows, arguments(), "off")

    def test_request_counter_increments_do_not_double_count_cumulative_work(self):
        off, on = records(), records(arm="on")
        on[0]["expert_schedule"]["shared_submissions_cumulative"] = 3
        on[1]["expert_schedule"]["shared_submissions_cumulative"] = 7
        delta = compare_pair({"off": off, "on": on})["on"]
        self.assertEqual([row["expert_schedule_request"]["shared_submissions_cumulative"] for row in delta], [3, 4])
        self.assertEqual(sum(row["expert_schedule_request"]["shared_submissions_cumulative"] for row in delta), 7)


class QuietHostEvidence(unittest.TestCase):
    def test_cpu_units_are_one_core_and_measured_process_tree_is_excluded(self):
        own = os.getpid()
        ps = f"{own} 1 1 99 0:01.00 monitor\n11 {own} 1 99 0:01.00 ps\n22 1 22 99 0:01.00 probe\n23 22 22 99 0:01.00 child\n44 1 44 3 0:00.20 unrelated\n"
        self.assertEqual([row["pid"] for row in parse_activity(ps, 22)], [44])
        self.assertEqual(cpu_seconds("1-02:03:04.50"), 93784.5)
        self.assertEqual(cpu_seconds("3:04.50"), 184.5)
        # Record aggregate and maximum separately; percent is not divided by core count.
        snaps = [dict(returncode=0, stdout="44 1 44 3 0:00.20 a\n55 1 55 3 0:00.20 b\n", stderr="")]
        def snap(command):
            return snaps[0] if command[0] == "ps" else dict(returncode=0, stdout="'AC Power'", stderr="")
        monitor = HostMonitor()
        with patch("moe_expert_schedule_pairs.snapshot", side_effect=snap), patch("moe_expert_schedule_pairs.time.monotonic", side_effect=[0, 1]):
            first = monitor.sample()
            self.assertEqual(first["background_cpu_percent"], 6)
            self.assertEqual(first["max_background_cpu_percent"], 3)
            snaps[0]["stdout"] = "44 1 44 3 0:00.23 a\n55 1 55 3 0:00.23 b\n"
            second = monitor.sample()
            self.assertAlmostEqual(second["background_cpu_percent"], 6)
            self.assertAlmostEqual(second["max_background_cpu_percent"], 3)
            sample_series = [{**second, "monotonic_s": at} for at in range(4)]
            self.assertTrue(host_eligibility(sample_series)["eligible"])

    def test_sustained_low_cpu_load_rejected_but_single_spike_retained(self):
        self.assertFalse(host_eligibility([host_sample(t, 6) for t in range(4)])["eligible"])
        transient = [host_sample(0), host_sample(1, 75), host_sample(2), host_sample(3)]
        self.assertTrue(host_eligibility(transient)["eligible"])
        self.assertEqual(transient[1]["background_cpu_percent"], 75)
        self.assertTrue(host_eligibility([host_sample(t, 5) for t in range(4)])["eligible"])
        changing_pids = [host_sample(t, 6, background_cpu_by_pid={str(50 + t): 6}) for t in range(4)]
        self.assertTrue(host_eligibility(changing_pids)["eligible"])
        intervals = [host_sample(0), host_sample(1, 6, interval_s=1), host_sample(2, 6, interval_s=1)]
        self.assertFalse(host_eligibility(intervals)["eligible"])

    def test_midrun_power_thermal_monitor_failure_and_other_models_reject(self):
        for contaminated in (host_sample(1, power="'Battery Power'"), host_sample(1, thermal="CPU_Speed_Limit = 90"),
                             host_sample(1, cpu=None, monitor_error="missing ps"),
                             host_sample(1, other_model_processes=[{"command": "ollama"}])):
            with self.subTest(sample=contaminated):
                self.assertFalse(host_eligibility([host_sample(), contaminated, host_sample(2)])["eligible"])
        with patch("moe_expert_schedule_pairs.snapshot") as snap:
            snap.side_effect = [dict(returncode=0, stdout="44 1 44 0 0:00.20 qwen36_memory_oracle-abc\n", stderr=""),
                                dict(returncode=0, stdout="'AC Power'"), dict(returncode=0, stdout="")]
            self.assertFalse(host_eligibility([HostMonitor().sample()])["eligible"])


class FailedProcessEvidence(unittest.TestCase):
    def test_contended_bypass_runs_but_never_qualifies_even_on_quiet_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            binary = out / "fake-probe"
            payload = "\n".join(json.dumps(row) for row in records())
            binary.write_text(f"#!{sys.executable}\nprint({payload!r})\n")
            binary.chmod(0o700)
            args = arguments(binary=binary, model=out, timeout_seconds=3, allow_contended=True)
            for cpu in (0, 6):
                with self.subTest(cpu=cpu), patch("moe_expert_schedule_pairs.HostMonitor") as monitor:
                    monitor.return_value.sample.return_value = host_sample(2, cpu, interval_s=2)
                    rows, launch = run_probe(args, "off", out, f"diagnostic-{cpu}")
                self.assertEqual(len(rows), 2)
                self.assertFalse(launch["eligible"])
                self.assertTrue(launch["diagnostic_allow_contended"])
                self.assertIn("--allow-contended", " ".join(launch["rejection_reasons"]))
                self.assertEqual(launch["preflight"][0]["background_cpu_percent"], cpu)
                self.assertTrue(launch["periodic_host"])
            # Without the explicit flag, the same contended preflight refuses launch.
            with patch("moe_expert_schedule_pairs.HostMonitor") as monitor:
                monitor.return_value.sample.return_value = host_sample(2, 6, interval_s=2)
                with self.assertRaisesRegex(ProbeFailure, "quiet preflight rejected"):
                    run_probe(arguments(binary=binary, model=out), "off", out, "refused")
            self.assertFalse((out / "refused.stdout.jsonl").exists())

    def test_timeout_kills_owned_process_and_preserves_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            binary = out / "fake-probe"
            binary.write_text("#!/bin/sh\nprintf 'partial evidence\\n'\nsleep 30\n")
            binary.chmod(0o700)
            args = arguments(binary=binary, model=out)
            def sample(owned_pid=None):
                if owned_pid is not None:
                    deadline = time.monotonic() + 30
                    while not (out / "timeout.stdout.jsonl").read_text() and time.monotonic() < deadline:
                        time.sleep(0.01)
                return host_sample(time.monotonic())
            with patch("moe_expert_schedule_pairs.HostMonitor") as monitor:
                monitor.return_value.sample.side_effect = sample
                with self.assertRaisesRegex(ProbeFailure, "partial evidence retained"):
                    run_probe(args, "off", out, "timeout")
            launch = json.loads((out / "timeout.launch.json").read_text())
            self.assertTrue(launch["timed_out"])
            self.assertFalse(launch["eligible"])
            self.assertIsNotNone(launch["exit_code"])
            self.assertEqual((out / "timeout.stdout.jsonl").read_text(), "partial evidence\n")

    def test_nonzero_exit_retains_stderr_and_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            binary = out / "failed-probe"
            binary.write_text("#!/bin/sh\nprintf 'intentional failure\\n' >&2\nexit 4\n")
            binary.chmod(0o700)
            with patch("moe_expert_schedule_pairs.HostMonitor") as monitor:
                monitor.return_value.sample.side_effect = lambda owned_pid=None: host_sample(time.monotonic())
                with self.assertRaisesRegex(ProbeFailure, "partial evidence retained"):
                    # Exercise the exit status independently of interpreter/host
                    # startup scheduling; timeout refusal has its own test.
                    run_probe(arguments(binary=binary, model=out, timeout_seconds=30), "off", out, "failed")
            launch = json.loads((out / "failed.launch.json").read_text())
            self.assertEqual(launch["exit_code"], 4)
            self.assertFalse(launch["timed_out"])
            self.assertFalse(launch["eligible"])
            self.assertEqual((out / "failed.stderr.txt").read_text(), "intentional failure\n")

    def test_failed_arm_does_not_prevent_other_mapped_modes_and_attempts_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary, model = root / "binary", root / "model"
            binary.write_text("fake binary")
            model.mkdir()
            (model / "manifest.json").write_text("{}")
            called = []
            def failed(args, arm, out, tag):
                called.append(arm)
                raise ProbeFailure("retained timeout")
            argv = ["--binary", str(binary), "--model", str(model), "--out", str(root / "out"),
                    "--experiment", "mapped", "--pairs", "1", "--max-attempts", "2"]
            with patch("moe_expert_schedule_pairs.run_probe", side_effect=failed), \
                    patch("moe_expert_schedule_pairs.snapshot", return_value={"returncode": 0}), \
                    patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(main(argv), 1)
            self.assertEqual(called, ["off", "advice", "touch", "touch", "advice", "off"])
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(len(summary["attempts"]), 2)
            self.assertEqual(summary["eligible_pairs"], 0)
            self.assertFalse(summary["default_promotion_supported"])
            self.assertFalse(summary["requested_pair_count_met"])

    def test_contaminated_pair_retained_but_only_quiet_pair_enters_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary, model = root / "binary", root / "model"
            binary.write_text("fake binary")
            model.mkdir()
            manifest = model / "manifest.json"
            manifest.write_text("{}")
            called = []
            def run(args, arm, out, tag):
                called.append(arm)
                rows = records(arm=arm, manifest=digest(manifest))
                first_attempt = "attempt-0-" in tag
                if arm == "on":
                    rows[0]["load_to_first_token_ms"] += 100 if first_attempt else 12
                return rows, dict(eligible=not first_attempt, rejection_reasons=["contended"] if first_attempt else [])
            argv = ["--binary", str(binary), "--model", str(model), "--out", str(root / "out"),
                    "--experiment", "overlap", "--pairs", "1", "--max-attempts", "2"]
            with patch("moe_expert_schedule_pairs.run_probe", side_effect=run), \
                    patch("moe_expert_schedule_pairs.snapshot", return_value={"returncode": 0}), \
                    patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(main(argv), 0)
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(called, ["off", "on", "on", "off"])
            self.assertEqual(summary["eligible_pairs"], 1)
            self.assertFalse(summary["attempts"][0]["eligible"])
            self.assertEqual(summary["eligible_first_request_median_deltas"]["on"]["load_to_first_token_delta_ms"], 12)

    def test_parity_failure_is_fatal_with_partial_attempt_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary, model = root / "binary", root / "model"
            binary.write_text("fake binary")
            model.mkdir()
            manifest = model / "manifest.json"
            manifest.write_text("{}")
            def changed(args, arm, out, tag):
                rows = records(arm=arm, manifest=digest(manifest))
                if arm == "on":
                    rows[1]["token_sha256"] = "wrong output"
                return rows, dict(eligible=True)
            argv = ["--binary", str(binary), "--model", str(model), "--out", str(root / "out"),
                    "--experiment", "overlap", "--pairs", "1", "--max-attempts", "2"]
            with patch("moe_expert_schedule_pairs.run_probe", side_effect=changed), \
                    patch("moe_expert_schedule_pairs.snapshot", return_value={"returncode": 0}), \
                    patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(main(argv), 1)
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertIn("digest/count/stop", summary["fatal_error"])
            self.assertEqual(len(summary["attempts"]), 1)
            self.assertEqual(len(summary["attempts"][0]["captures"]), 2)
            self.assertEqual(summary["eligible_pairs"], 0)

    def test_one_diagnostic_mapped_triplet_cannot_satisfy_requested_quiet_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary, model = root / "binary", root / "model"
            binary.write_text("fake binary")
            model.mkdir()
            manifest = model / "manifest.json"
            manifest.write_text("{}")
            called = []
            def run(args, arm, out, tag):
                called.append(arm)
                return records("mapped", arm, manifest=digest(manifest)), dict(eligible=True)
            argv = ["--binary", str(binary), "--model", str(model), "--out", str(root / "out"),
                    "--experiment", "mapped", "--pairs", "3", "--max-attempts", "1", "--allow-contended"]
            with patch("moe_expert_schedule_pairs.run_probe", side_effect=run), \
                    patch("moe_expert_schedule_pairs.snapshot", return_value={"returncode": 0}), \
                    patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(main(argv), 1)
            self.assertEqual(called, ["off", "advice", "touch"])
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertTrue(summary["diagnostic_allow_contended"])
            self.assertEqual(summary["eligible_pairs"], 0)
            self.assertEqual(len(summary["attempts"]), 1)
            self.assertFalse(summary["attempts"][0]["eligible"])
            self.assertTrue(all(not capture["eligible"] for capture in summary["attempts"][0]["captures"].values()))
            self.assertFalse(summary["requested_pair_count_met"])
            self.assertFalse(summary["default_promotion_supported"])


if __name__ == "__main__":
    unittest.main()
