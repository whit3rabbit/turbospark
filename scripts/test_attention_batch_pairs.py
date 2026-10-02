"""Fail-closed tests for the ignored Metal attention pair runner."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import attention_batch_pairs as pairs


def record(arm="batched", **changes):
    value = {
        "arm": arm,
        "M": 2,
        "row_positions": [240, 241],
        "head_shape": {"head_dim": 128, "num_q_heads": 32, "num_kv_heads": 8},
        "source_revision": "abc123",
        "benchmark_source_dirty": False,
        "release_mode": "release",
        "metal_device": "Test Metal Device",
        "repetitions": 2,
        "sample_count": 2,
        "elapsed_seconds_total": 0.003,
        "elapsed_seconds_per_repetition_median": 0.0015,
        "elapsed_seconds_per_repetition_range": [0.001, 0.002],
        "elapsed_seconds_per_repetition_samples": [0.001, 0.002],
    }
    value.update(changes)
    return value


def cargo_artifact(path):
    return json.dumps({
        "reason": "compiler-artifact",
        "target": {"name": "attention_chunk_bench", "kind": ["test"]},
        "executable": str(path),
    })


def completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class AttentionBatchPairsTests(unittest.TestCase):
    def test_arm_order_alternates_for_three_or_more_pairs(self):
        self.assertEqual(
            [pairs.arm_order_for_pair(index) for index in range(5)],
            [
                ("batched", "serial"),
                ("serial", "batched"),
                ("batched", "serial"),
                ("serial", "batched"),
                ("batched", "serial"),
            ],
        )
        with self.assertRaises(ValueError):
            pairs.validate_pair_count(2)

    def test_build_artifact_parser_requires_one_matching_test_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "attention-bench"
            artifact.touch()
            self.assertEqual(
                pairs.parse_test_binary(cargo_artifact(artifact) + "\n", Path(directory)),
                artifact.resolve(),
            )
            with self.assertRaisesRegex(RuntimeError, "no attention_chunk_bench test artifact"):
                pairs.parse_test_binary("{}\n", Path(directory))
            with self.assertRaisesRegex(
                RuntimeError, "multiple attention_chunk_bench test artifacts"
            ):
                pairs.parse_test_binary(
                    cargo_artifact(artifact) + "\n" + cargo_artifact(artifact) + "\n",
                    Path(directory),
                )

    def test_record_parser_rejects_missing_malformed_and_multiple_json_records(self):
        valid = json.dumps(record())
        self.assertEqual(pairs.parse_record_line(f"running 1 test\n{valid}\n"), valid)
        for output in ("running 1 test\n", '{"arm":"batched"\n', f"{valid}\n{valid}\n"):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                pairs.parse_record_line(output)

    def test_record_validation_rejects_incompatible_or_malformed_metadata(self):
        good = record()
        pairs.validate_record(good, "batched", rows=2, repetitions=2)
        cases = [
            ("release_mode", "debug"),
            ("M", 3),
            ("row_positions", [241, 242]),
            ("head_shape", {"head_dim": 64, "num_q_heads": 32, "num_kv_heads": 8}),
            ("arm", "serial"),
            ("benchmark_source_dirty", True),
            ("elapsed_seconds_per_repetition_samples", [0.001, float("nan")]),
        ]
        for field, value in cases:
            changed = copy.deepcopy(good)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                pairs.validate_record(changed, "batched", rows=2, repetitions=2)

        identity = pairs._record_identity(good)
        for field, value in (("source_revision", "different revision"),
                             ("metal_device", "Other Metal Device")):
            changed = copy.deepcopy(good)
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"{field} mismatch"):
                pairs._check_compatible_record(changed, identity)

    def test_run_builds_once_then_launches_each_arm_in_fresh_alternating_processes(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "attention-bench"
            artifact.touch()

            def fake_run(command, **kwargs):
                env = kwargs.get("env", {})
                if command[0] == "git":
                    if "rev-parse" in command:
                        return completed(0, "abc123\n")
                    if "status" in command:
                        return completed(0, "")
                    raise AssertionError(f"unexpected git command: {command}")
                if command[0] == "cargo":
                    events.append(("build", command, env))
                    self.assertIn("--locked", command)
                    return completed(0, cargo_artifact(artifact) + "\n")
                arm = env[pairs.ARM_ENV]
                events.append(("child", arm, command, env))
                value = record(arm)
                value["M"] = int(env[pairs.ROWS_ENV])
                value["row_positions"] = list(range(240, 240 + value["M"]))
                value["repetitions"] = int(env[pairs.REPETITIONS_ENV])
                value["sample_count"] = value["repetitions"]
                value["elapsed_seconds_per_repetition_samples"] = [0.001] * value["repetitions"]
                value["elapsed_seconds_total"] = 0.001 * value["repetitions"]
                value["elapsed_seconds_per_repetition_median"] = 0.001
                value["elapsed_seconds_per_repetition_range"] = [0.001, 0.001]
                return completed(0, f"{json.dumps(value)}\n")

            report = pairs.run_pairs(
                pair_count=3,
                rows=2,
                repetitions=3,
                repo_root=root,
                command_runner=fake_run,
            )

        self.assertEqual([event[0] for event in events], ["build"] + ["child"] * 6)
        self.assertEqual(
            [event[1] for event in events[1:]],
            ["batched", "serial", "serial", "batched", "batched", "serial"],
        )
        for event in events[1:]:
            self.assertEqual(event[3][pairs.ROWS_ENV], "2")
            self.assertEqual(event[3][pairs.REPETITIONS_ENV], "3")
            self.assertEqual(event[3][pairs.ARM_ENV], event[1])
            self.assertIn("one_arm_attention_benchmark", event[2])
            self.assertNotEqual(event[2][0], "cargo")
        self.assertEqual(report["pair_count"], 3)
        self.assertEqual(len(report["raw_records"]), 6)
        self.assertIn(
            "\"elapsed_seconds_per_repetition_samples\"",
            report["raw_records"][0]["raw_json"],
        )
        self.assertEqual(report["summary"]["batched"]["sample_count"], 9)
        self.assertEqual(report["summary"]["batched"]["range"], {"min": 0.001, "max": 0.001})
        self.assertNotIn("performance_pass", report)
        self.assertEqual(report["source_identity"]["revision"], "abc123")
        self.assertEqual(len(report["source_identity"]["build_inputs_sha256"]), 64)
        self.assertEqual(report["source_identity"]["dirty_paths"], [])
        self.assertEqual(len(report["build"]["sha256"]), 64)

    def test_build_or_child_failure_aborts_without_a_partial_report(self):
        def failed_build(command, **_kwargs):
            if command[0] == "git":
                if "rev-parse" in command:
                    return completed(0, "abc123\n")
                return completed(0, "")
            return completed(1, "", "compiler failed")

        with self.assertRaisesRegex(RuntimeError, "release benchmark build failed"):
            pairs.run_pairs(command_runner=failed_build)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "attention-bench"
            artifact.touch()

            def failed_child(command, **_kwargs):
                if command[0] == "git":
                    if "rev-parse" in command:
                        return completed(0, "abc123\n")
                    return completed(0, "")
                if command[0] == "cargo":
                    return completed(0, cargo_artifact(artifact) + "\n")
                return completed(17, "", "test failed")

            with self.assertRaisesRegex(RuntimeError, "benchmark child failed"):
                pairs.run_pairs(repo_root=root, command_runner=failed_child)

    def test_pair_capture_rejects_mismatch_in_each_required_identity_field(self):
        mismatches = [
            (
                "source_revision",
                "different revision",
                "benchmark source_revision does not match",
            ),
            ("release_mode", "debug", "release_mode must be"),
            ("metal_device", "Other Metal Device", "metal_device mismatch"),
            ("M", 3, "M must be 2"),
            ("row_positions", [241, 242], "row_positions must equal"),
            (
                "head_shape",
                {"head_dim": 64, "num_q_heads": 32, "num_kv_heads": 8},
                "head_shape must equal",
            ),
        ]
        for field, changed_value, error in mismatches:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifact = root / "attention-bench"
                artifact.touch()
                child_count = 0

                def drifting_child(command, **kwargs):
                    nonlocal child_count
                    if command[0] == "git":
                        if "rev-parse" in command:
                            return completed(0, "abc123\n")
                        return completed(0, "")
                    if command[0] == "cargo":
                        return completed(0, cargo_artifact(artifact) + "\n")
                    arm = kwargs["env"][pairs.ARM_ENV]
                    value = record(arm)
                    if child_count == 1:
                        value[field] = changed_value
                    child_count += 1
                    return completed(0, json.dumps(value) + "\n")

                with self.assertRaisesRegex(ValueError, error):
                    pairs.run_pairs(
                        rows=2,
                        repetitions=2,
                        repo_root=root,
                        command_runner=drifting_child,
                    )

    def test_run_rejects_dirty_or_changed_build_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dirty = True

            def dirty_build(command, **_kwargs):
                if command[0] == "git":
                    if "rev-parse" in command:
                        return completed(0, "abc123\n")
                    return completed(0, " M crates/gpu/src/attention.rs\n" if dirty else "")
                raise AssertionError("the release build must not start from dirty inputs")

            with self.assertRaisesRegex(RuntimeError, "build inputs must be clean"):
                pairs.run_pairs(repo_root=root, command_runner=dirty_build)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "attention-bench"
            artifact.touch()

            def changed_during_build(command, **kwargs):
                if command[0] == "git":
                    if "rev-parse" in command:
                        return completed(0, "abc123\n")
                    source = root / "crates" / "gpu" / "src" / "attention.rs"
                    if source.exists():
                        return completed(0, " M crates/gpu/src/attention.rs\n")
                    return completed(0, "")
                if command[0] == "cargo":
                    source = root / "crates" / "gpu" / "src" / "attention.rs"
                    source.parent.mkdir(parents=True)
                    source.write_text("changed after build began")
                    return completed(0, cargo_artifact(artifact) + "\n")
                raise AssertionError(f"unexpected benchmark child: {command}")

            with self.assertRaisesRegex(RuntimeError, "changed during the release build"):
                pairs.run_pairs(repo_root=root, command_runner=changed_during_build)


if __name__ == "__main__":
    unittest.main()
