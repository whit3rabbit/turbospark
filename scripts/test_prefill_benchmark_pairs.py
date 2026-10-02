"""Fail-closed tests for the matched dense-Llama prefill runner."""

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import prefill_benchmark_pairs as pairs


BASELINE_REVISION = "324b3f03ce94c85a5356c3ce9c6c5658fde81edc"
CANDIDATE_REVISION = "80252a6f951c4f7e13acd3608855b67231cd9f36"
PROMPT = b"frozen long-synthesis prompt\n"


def completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def output_for(model_path, prefill="10.00", *, chunk=128, family="llama", kv_bits="off"):
    stdout = (
        f"turbospark-bench: real install {model_path} on Apple M4 Pro, "
        "frozen protocol real-generation-v1\n"
        f"  family={family} context=8192 max_new=1024 expert_cache_slots=16 kv_bits={kv_bits}\n"
        "  speculative=off shaping=protocol-sampled\n"
        f"  prefill=chunked chunk_tokens={chunk} routed_batch=unset batched_gemv=unset\n"
        "  power_profile=performance max_tok_s=- thermal_stepping=false\n"
        "case             prompt_tok  prefill_s  new_tok  decode_s    tok_s  peak_mib\n"
        f"long-synthesis         3444      {prefill}     1024      4.00  256.000       0.0\n"
    )
    stderr = (
        f"[stop=maxTokens prefill=3444tok/{prefill}s new=1024tok "
        "decode=4.00s tok/s=256.000]\n"
    )
    return stdout, stderr


class PrefillBenchmarkPairsTests(unittest.TestCase):
    def test_pair_order_and_chunk_validation(self):
        self.assertEqual(
            [pairs.arm_order_for_pair(index) for index in range(4)],
            [
                ("baseline", "candidate"),
                ("candidate", "baseline"),
                ("baseline", "candidate"),
                ("candidate", "baseline"),
            ],
        )
        with self.assertRaises(ValueError):
            pairs.validate_inputs(2, 128)
        with self.assertRaises(ValueError):
            pairs.validate_inputs(3, 17)

    def test_output_parser_requires_matching_resolved_inputs_and_one_footer(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout, stderr = output_for(directory)
            parsed = pairs.parse_run_output(
                stdout, stderr, expected_chunk=128, expected_model_path=directory
            )
            self.assertEqual(parsed["family"], "llama")
            self.assertEqual(parsed["kv_bits"], "off")
            self.assertEqual(parsed["prefill_seconds_raw"], "10.00")
            self.assertEqual(parsed["prompt_tokens"], 3444)
            self.assertEqual(parsed["chunk_tokens"], 128)

            for changed_stdout, changed_stderr, expected_error in (
                (stdout.replace("family=llama", "family=qwen"), stderr, "dense-Llama"),
                (stdout.replace("kv_bits=off", "kv_bits=4"), stderr, "unquantized KV"),
                (stdout.replace("chunk_tokens=128", "chunk_tokens=256"), stderr, "chunk size"),
                (
                    stdout.replace("long-synthesis", "short-explanation"),
                    stderr,
                    "long-synthesis measurement row",
                ),
                (stdout, stderr + stderr, "exactly one measured footer"),
            ):
                with self.subTest(expected_error=expected_error), self.assertRaisesRegex(
                    ValueError, expected_error
                ):
                    pairs.parse_run_output(
                        changed_stdout,
                        changed_stderr,
                        expected_chunk=128,
                        expected_model_path=directory,
                    )

    def test_model_install_must_be_dense_llama(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _binaries = self._make_inputs(Path(directory))
            snapshot = pairs.artifact_snapshot(model)
            self.assertEqual(snapshot["family"], "llama")
            self.assertEqual(snapshot["num_experts"], 0)
            self.assertEqual(snapshot["model_id"], "test/dense-llama")

            manifest_path = model / "manifest.json"
            manifest_path.write_text(
                '{"magic":"GTURBO","modelID":"test/mixtral",'
                '"arch":{"family":"llama","numExperts":8},"expertsPerLayer":8}\n'
            )
            with self.assertRaisesRegex(ValueError, "dense Llama"):
                pairs.artifact_snapshot(model)

    def _make_inputs(self, root):
        model = root / "install.gturbo"
        model.mkdir()
        (model / "model_weights.bin").write_bytes(b"pinned model weights")
        (model / "manifest.json").write_text(
            '{"magic":"GTURBO","modelID":"test/dense-llama",'
            '"arch":{"family":"llama","numExperts":0},"expertsPerLayer":0}\n'
        )
        binaries = {}
        for arm in ("baseline", "candidate"):
            binary = root / arm / "turbospark-bench"
            binary.parent.mkdir()
            binary.write_bytes(f"{arm} executable".encode())
            binary.chmod(0o755)
            binaries[arm] = binary
        return model, binaries

    def _run_report(
        self,
        root,
        *,
        child_mutation=None,
        prompt_candidate=None,
        candidate_hash_override=None,
        candidate_revision_override=None,
    ):
        model, binaries = self._make_inputs(root)
        events = []
        invocation = 0
        run_counts = {"baseline": 0, "candidate": 0}

        def fake_run(command, **kwargs):
            nonlocal invocation
            if command[0] == "git" and "rev-parse" in command:
                requested = command[-1].split("^")[0]
                return completed(0, (requested + "\n").encode())
            if command[0] == "git" and "show" in command:
                requested, path = command[-1].split(":", 1)
                if path == "Cargo.toml":
                    return completed(
                        0,
                        b'[workspace.package]\nversion = "0.2.0"\n\n'
                        b'[workspace.dependencies]\n',
                    )
                prompt = (
                    prompt_candidate
                    if requested == CANDIDATE_REVISION and prompt_candidate is not None
                    else PROMPT
                )
                return completed(0, prompt)
            if command[0] == "system_profiler":
                return completed(0, '{"SPDisplaysDataType": [{"sppci_model": "Mock Metal"}]}')
            arm = "baseline" if str(binaries["baseline"].resolve()) == command[0] else "candidate"
            events.append((arm, command, kwargs))
            prefill = {
                "baseline": ("10.01", "9.99", "10.00"),
                "candidate": ("8.01", "8.00", "7.99"),
            }[arm][run_counts[arm]]
            run_counts[arm] += 1
            stdout, stderr = output_for(model.resolve(), prefill)
            invocation += 1
            if child_mutation is not None:
                child_mutation(invocation, binaries, model)
            return completed(0, stdout, stderr)

        binary_hashes = {
            arm: pairs.file_sha256(binary) for arm, binary in binaries.items()
        }
        test_pins = {
            arm: {
                "source_revision": BASELINE_REVISION if arm == "baseline" else CANDIDATE_REVISION,
                "binary_sha256": binary_hashes[arm],
            }
            for arm in ("baseline", "candidate")
        }
        with patch.dict(pairs.PINNED_BUILD_IDENTITIES, test_pins, clear=True):
            report = pairs.run_pairs(
                baseline_binary=binaries["baseline"],
                candidate_binary=binaries["candidate"],
                baseline_binary_sha256=binary_hashes["baseline"],
                candidate_binary_sha256=(
                    candidate_hash_override
                    if candidate_hash_override is not None
                    else binary_hashes["candidate"]
                ),
                baseline_revision=BASELINE_REVISION,
                candidate_revision=(candidate_revision_override or CANDIDATE_REVISION),
                model_install=model,
                chunk_size=128,
                pair_count=3,
                repo_root=root,
                command_runner=fake_run,
                host_reader=lambda _runner, _cwd: {
                    "hostname": "test-host",
                    "architecture": "arm64",
                    "metal_device": "Mock Metal",
                },
            )
        return report, events, model, binaries, pairs.artifact_snapshot(model)["sha256"]

    def test_run_uses_fresh_alternating_processes_and_keeps_raw_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                {
                    "TURBOSPARK_DISPATCH_PROFILE": "1",
                    "TURBOSPARK_DEBUG_PROMPT_IDS": "1",
                    "TURBOSPARK_METAL_PRECISE_MATH": "1",
                    "TURBOSPARK_RESID_CAPTURE": "/tmp/residual-capture",
                },
            ):
                report, events, model, binaries, model_hash = self._run_report(root)

        self.assertEqual(
            [event[0] for event in events],
            ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"],
        )
        self.assertEqual(len({id(event[2]["env"]) for event in events}), 6)
        for arm, command, _kwargs in events:
            self.assertEqual(command[0], str(binaries[arm].resolve()))
            self.assertEqual(
                command[1:],
                [
                    "--model", str(model.resolve()),
                    "--case", "long-synthesis",
                    "--prefill-chunk", "128",
                    "--kv-bits", "off",
                    "--shaping", "protocol",
                    "--speculative", "off",
                    "--power-profile", "performance",
                    "--expert-cache-slots", "16",
                ],
            )
            self.assertFalse(
                set(pairs.CLEARED_ENV_VARS).intersection(_kwargs["env"])
            )
        self.assertEqual(
            report["environment"]["cleared_turbospark_controls"],
            {
                "TURBOSPARK_DISPATCH_PROFILE": "1",
                "TURBOSPARK_DEBUG_PROMPT_IDS": "1",
                "TURBOSPARK_METAL_PRECISE_MATH": "1",
                "TURBOSPARK_RESID_CAPTURE": "/tmp/residual-capture",
            },
        )
        self.assertEqual(report["model_install"]["sha256"], model_hash)
        self.assertEqual(report["builds"]["baseline"]["package_version"], "0.2.0")
        self.assertEqual(report["builds"]["candidate"]["package_version"], "0.2.0")
        self.assertEqual(report["protocol"]["prompt_sha256"], hashlib.sha256(PROMPT).hexdigest())
        self.assertEqual(report["protocol"]["warmup_discarded_by_binary"], True)
        self.assertEqual(report["summary"]["baseline"]["sample_count"], 3)
        self.assertEqual(report["summary"]["candidate"]["sample_count"], 3)
        self.assertEqual(
            report["summary"]["baseline"]["range_prefill_seconds"],
            {"min": 9.99, "max": 10.01},
        )
        self.assertEqual(
            report["summary"]["candidate"]["range_prefill_seconds"],
            {"min": 7.99, "max": 8.01},
        )
        self.assertEqual(len(report["builds"]["baseline"]["binary_sha256"]), 64)

    def test_runner_refuses_prompt_drift_between_source_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, binaries = self._make_inputs(root)

            def fake_run(command, **_kwargs):
                if command[0] == "git" and "rev-parse" in command:
                    return completed(0, (command[-1].split("^")[0] + "\n").encode())
                if command[0] == "git" and "show" in command:
                    revision = command[-1].split(":")[0]
                    content = (
                        b"different prompt" if revision == CANDIDATE_REVISION else PROMPT
                    )
                    return completed(0, content)
                raise AssertionError(f"unexpected command before prompt validation: {command}")

            binary_hashes = {
                arm: pairs.file_sha256(binary) for arm, binary in binaries.items()
            }
            test_pins = {
                arm: {
                    "source_revision": BASELINE_REVISION if arm == "baseline" else CANDIDATE_REVISION,
                    "binary_sha256": binary_hashes[arm],
                }
                for arm in ("baseline", "candidate")
            }
            with patch.dict(pairs.PINNED_BUILD_IDENTITIES, test_pins, clear=True):
                with self.assertRaisesRegex(ValueError, "different long-synthesis prompts"):
                    pairs.run_pairs(
                        baseline_binary=binaries["baseline"],
                        candidate_binary=binaries["candidate"],
                        baseline_binary_sha256=binary_hashes["baseline"],
                        candidate_binary_sha256=binary_hashes["candidate"],
                        baseline_revision=BASELINE_REVISION,
                        candidate_revision=CANDIDATE_REVISION,
                        model_install=model,
                        chunk_size=128,
                        repo_root=root,
                        command_runner=fake_run,
                        host_reader=lambda *_args: {},
                    )

    def test_runner_refuses_binaries_that_do_not_match_recorded_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "does not match the recorded artifact"):
                self._run_report(Path(directory), candidate_hash_override="0" * 64)

    def test_runner_binds_reported_revisions_to_pinned_artifact_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "source revision does not match"):
                self._run_report(
                    Path(directory), candidate_revision_override="3" * 40
                )

    def test_runner_refuses_model_or_binary_mutation_during_measurement(self):
        for target in ("model", "binary"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)

                def mutate_on_first_child(invocation, binaries, model):
                    if invocation != 1:
                        return
                    if target == "model":
                        (model / "model_weights.bin").write_bytes(b"changed model")
                    else:
                        binaries["baseline"].write_bytes(b"changed binary")

                with self.assertRaisesRegex(RuntimeError, "changed during measurement"):
                    self._run_report(root, child_mutation=mutate_on_first_child)


if __name__ == "__main__":
    unittest.main()
