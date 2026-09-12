"""Hostile evidence tests, independent of model downloads and GPU access."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from z_image_evidence import sha256, validate_capture, preflight
from z_image_probe import COMPONENTS, REFERENCES, REVISION, validate_header, validate_inventory


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        file = self.root / "update.npy"
        np.save(file, np.array([[1.0, -1.0]], dtype=np.float32))
        self.capture = {"complete": True, "stage": "contracts", "model_revision": REVISION,
                        "references": {k: v[1] for k, v in REFERENCES.items()}, "inputs": {},
                        "arrays": {"scheduler_updates": {"file": file.name, "sha256": sha256(file),
                                    "shape": [1, 2], "dtype": "float32", "bytes": 8}}}

    def check_capture(self, x):
        path = self.root / "contracts.json"
        path.write_text(json.dumps(x))
        return validate_capture(path)

    def test_valid_capture(self):
        self.assertEqual(self.check_capture(self.capture)["stage"], "contracts")

    def test_corrupted_fixture(self):
        (self.root / "update.npy").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.check_capture(self.capture)

    def test_altered_shape(self):
        self.capture["arrays"]["scheduler_updates"]["shape"] = [2, 1]
        with self.assertRaisesRegex(ValueError, "shape/dtype/bytes"):
            self.check_capture(self.capture)

    def test_nonfinite(self):
        file = self.root / "update.npy"
        np.save(file, np.array([[np.nan, 1]], dtype=np.float32))
        self.capture["arrays"]["scheduler_updates"]["sha256"] = sha256(file)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.check_capture(self.capture)

    def test_missing_fixture(self):
        self.capture["arrays"] = {}
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.check_capture(self.capture)

    def test_revision(self):
        self.capture["model_revision"] = "main"
        with self.assertRaisesRegex(ValueError, "wrong revision"):
            self.check_capture(self.capture)

    def test_scheduler_count(self):
        # A valid encode predecessor ensures the count guard is reached.
        enc = copy.deepcopy(self.capture)
        enc["stage"] = "encode"
        row = enc["arrays"].pop("scheduler_updates")
        enc["arrays"] = {}
        for name, array in {"token_ids": np.zeros((1, 512), dtype=np.int64),
                            "attention_mask": np.ones((1, 512), dtype=np.int64),
                            "conditioning": np.zeros((512, 2560), dtype=np.float32)}.items():
            f = self.root / (name + ".npy")
            np.save(f, array)
            enc["arrays"][name] = {"file": f.name, "shape": list(array.shape), "dtype": str(array.dtype),
                                    "sha256": sha256(f), "bytes": array.nbytes}
        file = self.root / "encode.json"
        file.write_text(json.dumps(enc))
        self.capture.update(stage="denoise", inputs={"encode.json": sha256(file)},
                            settings={"steps": 9, "height": 16, "width": 16}, actual_forwards=8, scheduler_updates=9)
        arrays = {k: np.zeros((1, 16, 2, 2), dtype=np.float32)
                  for k in ["initial_noise", "final_latents"] + [f"latent_{i:02}" for i in range(8)]}
        arrays.update(timesteps=np.zeros(8, dtype=np.float32), sigmas=np.zeros(9, dtype=np.float32))
        self.capture["arrays"] = {}
        for name, array in arrays.items():
            f = self.root / (name + ".npy")
            np.save(f, array)
            self.capture["arrays"][name] = {"file": f.name, "shape": list(array.shape), "dtype": str(array.dtype),
                                           "sha256": sha256(f), "bytes": array.nbytes}
        with self.assertRaisesRegex(ValueError, "incorrect scheduler"):
            self.check_capture(self.capture)

    def test_header_byte_shape(self):
        with self.assertRaisesRegex(ValueError, "incorrect tensor offsets"):
            validate_header({"w": {"shape": [2, 4], "dtype": "F32", "data_offsets": [0, 16]}})

    def test_missing_component(self):
        files = {k: {} for k in ("model_index.json", "scheduler/scheduler_config.json",
                                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")}
        files.update({c + "/config.json": {} for c in COMPONENTS})
        report = {"model": {"revision": REVISION}, "files": files, "weights": {},
                  "references": {k: {"revision": v[1]} for k, v in REFERENCES.items()}}
        with self.assertRaisesRegex(ValueError, "missing component"):
            validate_inventory(report)

    def busy_preflight(self, require_quiet):
        snapshots = [(float(i * 2), {(123, 0): ("busy", float(i * 2))}) for i in range(4)]
        with patch("z_image_evidence.command", return_value="AC Power"), \
             patch("z_image_evidence.background_snapshot", side_effect=snapshots), \
             patch("z_image_evidence.time.sleep"):
            return preflight(require_quiet=require_quiet)

    def test_busy_correctness_capture(self):
        samples = self.busy_preflight(False)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(not row["quiet"] for row in samples))

    def test_busy_benchmark_refused(self):
        with self.assertRaisesRegex(RuntimeError, "quiet background load"):
            self.busy_preflight(True)

    def test_battery_correctness_refused(self):
        with patch("z_image_evidence.command", return_value="Battery Power"), \
             patch("z_image_evidence.background_snapshot", return_value=(0, {})), \
             patch("z_image_evidence.background_delta", return_value={"quiet": True}), \
             patch("z_image_evidence.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "AC power"):
                preflight(require_quiet=False)


def mutation_check(output):
    cases = [
        ("z_image_evidence.py", ' or sha256(file) != row["sha256"]', "", "test_corrupted_fixture"),
        ("z_image_evidence.py", 'if list(a.shape) != row["shape"] or str(a.dtype) != row["dtype"] or a.nbytes != row["bytes"]:', "if False:", "test_altered_shape"),
        ("z_image_evidence.py", 'if not np.isfinite(a).all():\n            raise ValueError("non-finite fixture")', 'if False:\n            raise ValueError("non-finite fixture")', "test_nonfinite"),
        ("z_image_evidence.py", 'if not required[x["stage"]].issubset(x["arrays"]):', "if False:", "test_missing_fixture"),
        ("z_image_evidence.py", ' or x["model_revision"] != REVISION', "", "test_revision"),
        ("z_image_evidence.py", 'if count != x["settings"]["steps"] or count != x["scheduler_updates"]:', "if False:", "test_scheduler_count"),
        ("z_image_probe.py", 'if offsets != [end, end + size]:', "if False:", "test_header_byte_shape"),
        ("z_image_probe.py", 'raise ValueError(f"missing component: {component}")', "pass", "test_missing_component"),
        ("z_image_evidence.py", 'if require_quiet and not all(row["quiet"] for row in samples):', 'if not all(row["quiet"] for row in samples):', "test_busy_correctness_capture"),
        ("z_image_evidence.py", 'if require_quiet and not all(row["quiet"] for row in samples):', "if False:", "test_busy_benchmark_refused"),
        ("z_image_evidence.py", 'if "AC Power" not in power:', "if False:", "test_battery_correctness_refused"),
    ]
    results = []
    for file, old, new, test in cases:
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("z_image_evidence.py", "z_image_probe.py", "test_z_image_evidence.py"):
                text = Path(__file__).with_name(name).read_text()
                if name == file:
                    assert text.count(old) == 1, (file, old)
                    text = text.replace(old, new)
                Path(tmp, name).write_text(text)
            run = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_z_image_evidence"],
                                 cwd=tmp, capture_output=True, text=True)
            failed = [line for line in run.stderr.splitlines() if line.endswith((" ... FAIL", " ... ERROR"))]
            assert run.returncode and len(failed) == 1 and test in failed[0], run.stderr
            results.append({"file": file, "mutation": old, "expected_failed_case": test,
                            "applied_unique": True, "isolated_failure": True})
    Path(output).write_text(json.dumps(results, indent=2) + "\n")
    print(f"{len(cases)} unique mutations, each failed only its intended case")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--mutation-report":
        mutation_check(sys.argv[2])
    else:
        unittest.main()
