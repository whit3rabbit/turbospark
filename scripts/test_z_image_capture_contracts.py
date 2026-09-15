"""Metadata-only contract checks for IG1 Z-Image capture manifests."""

import os
import hashlib
import json
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("Z_IMAGE_TEST_ROOT", Path(__file__).resolve().parent.parent))
CASES_ROOT = ROOT / "docs/verification/z-image-ig0-captures"
CONTRACTS = ROOT / "docs/verification/z-image-ig0-contracts.json"


class CaptureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACTS.read_text())
        cls.cases = sorted(path for path in CASES_ROOT.iterdir() if path.is_dir())

    def _load(self, path):
        return json.loads(path.read_text())

    @staticmethod
    def _sha256(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def test_contract_manifest_schema(self):
        contracts = self.contract
        self.assertEqual(contracts["stage"], "contracts")
        self.assertTrue(contracts["complete"])
        self.assertIn("arrays", contracts)
        self.assertIn("schedules", contracts)
        self.assertIn("token_cases", contracts)
        expected_cases = {
            "empty", "composition", "typography", "detail", "lighting", "unicode", "overlong"
        }
        self.assertEqual(set(contracts["token_cases"]), expected_cases)
        self.assertTrue(any(row["actual_forwards"] == 1 for row in contracts["schedules"]))
        self.assertTrue(any(row["actual_forwards"] == 8 for row in contracts["schedules"]))
        self.assertTrue(any(row["actual_forwards"] == 9 for row in contracts["schedules"]))
        for case, row in contracts["token_cases"].items():
            self.assertIn(f"{case}_ids", contracts["arrays"])
            self.assertIn(f"{case}_mask", contracts["arrays"])
            arrays = contracts["arrays"]
            ids = arrays[f"{case}_ids"]
            mask = arrays[f"{case}_mask"]
            self.assertEqual(ids["shape"], [1, 512])
            self.assertEqual(mask["shape"], [1, 512])
            self.assertEqual(ids["dtype"], "int64")
            self.assertEqual(mask["dtype"], "int64")

    def test_encode_manifests(self):
        for case_dir in self.cases:
            path = case_dir / "encode.json"
            if not path.exists():
                continue
            manifest = self._load(path)
            self.assertEqual(manifest["stage"], "encode")
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["inputs"], {})
            arrays = manifest["arrays"]
            required = {"token_ids", "attention_mask", "conditioning"}
            self.assertTrue(required.issubset(arrays))

            token_ids = arrays["token_ids"]
            attention_mask = arrays["attention_mask"]
            conditioning = arrays["conditioning"]

            self.assertEqual(token_ids["shape"], [1, 512])
            self.assertEqual(attention_mask["shape"], [1, 512])
            self.assertEqual(token_ids["dtype"], "int64")
            self.assertEqual(attention_mask["dtype"], "int64")
            retained = self.contract["token_cases"][manifest["settings"]["case"]]["retained_tokens"]
            self.assertEqual(conditioning["shape"][1], 2560)
            self.assertEqual(conditioning["shape"][0], retained)
            self.assertEqual(conditioning["dtype"], "float32")

    def test_denoise_manifests(self):
        for case_dir in self.cases:
            path = case_dir / "denoise.json"
            if not path.exists():
                continue
            manifest = self._load(path)
            self.assertEqual(manifest["stage"], "denoise")
            self.assertTrue(manifest["complete"])

            inputs = manifest.get("inputs", {})
            encode_file = case_dir / "encode.json"
            self.assertEqual(set(inputs), {"encode.json"})
            self.assertEqual(inputs.get("encode.json"), self._sha256(encode_file))

            settings = manifest["settings"]
            steps = settings["steps"]
            actual_forwards = manifest["actual_forwards"]
            self.assertEqual(actual_forwards, steps)
            self.assertEqual(manifest["scheduler_updates"], actual_forwards)

            arrays = manifest["arrays"]
            self.assertIn("initial_noise", arrays)
            self.assertIn("final_latents", arrays)
            self.assertIn("timesteps", arrays)
            self.assertIn("sigmas", arrays)
            for index in range(actual_forwards):
                self.assertIn(f"latent_{index:02}", arrays)

            latent_shape = [1, 16, settings["height"] // 8, settings["width"] // 8]
            for key in ("initial_noise", "final_latents"):
                self.assertEqual(arrays[key]["shape"], latent_shape)
                self.assertEqual(arrays[key]["dtype"], "float32")
            for index in range(actual_forwards):
                self.assertEqual(arrays[f"latent_{index:02}"]["shape"], latent_shape)
                self.assertEqual(arrays[f"latent_{index:02}"]["dtype"], "float32")
            self.assertEqual(arrays["timesteps"]["shape"], [actual_forwards])
            self.assertEqual(arrays["sigmas"]["shape"], [actual_forwards + 1])

            trace = manifest.get("first_step_trace")
            if trace is not None:
                self.assertEqual(
                    trace,
                    {
                        "conditioning": "conditioning",
                        "patchification": "patches",
                        "noise_refiner": "noise_refiner_output",
                        "main_transformer": "main_transformer_output",
                        "velocity": "velocity",
                        "scheduler_latent": "latent_00",
                    },
                )
                patch_count = (settings["height"] // 16) * (settings["width"] // 16)
                encode = self._load(encode_file)
                cap_len = encode["arrays"]["conditioning"]["shape"][0]
                cap_padded_len = (cap_len + 31) // 32 * 32
                self.assertEqual(arrays["patches"]["shape"], [patch_count, 64])
                self.assertEqual(
                    arrays["noise_refiner_output"]["shape"], [1, patch_count, 3840]
                )
                self.assertEqual(
                    arrays["main_transformer_output"]["shape"],
                    [1, patch_count + cap_padded_len, 3840],
                )
                self.assertIn(
                    arrays["velocity"]["shape"],
                    [latent_shape, [16, 1, settings["height"] // 8, settings["width"] // 8]],
                )
                for name in ("patches", "noise_refiner_output", "main_transformer_output", "velocity"):
                    self.assertEqual(arrays[name]["dtype"], "float32")

    def test_decode_manifests(self):
        for case_dir in self.cases:
            path = case_dir / "decode.json"
            if not path.exists():
                continue
            manifest = self._load(path)
            self.assertEqual(manifest["stage"], "decode")
            self.assertTrue(manifest["complete"])

            inputs = manifest.get("inputs", {})
            denoise_file = case_dir / "denoise.json"
            self.assertEqual(set(inputs), {"denoise.json"})
            self.assertEqual(inputs.get("denoise.json"), self._sha256(denoise_file))

            settings = manifest["settings"]
            arrays = manifest["arrays"]
            self.assertIn("decoded_pixels", arrays)
            self.assertEqual(arrays["decoded_pixels"]["shape"], [1, 3, settings["height"], settings["width"]])
            self.assertEqual(arrays["decoded_pixels"]["dtype"], "float32")
            self.assertTrue("png_sha256" in manifest)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--mutation-report":
        output = Path(sys.argv[2])
        cases = [
            ("self.assertEqual(set(contracts[\"token_cases\"]), expected_cases)",
             "self.assertEqual(set(contracts[\"token_cases\"]), set())",
             "test_contract_manifest_schema"),
            ("self.assertTrue(any(row[\"actual_forwards\"] == 8 for row in contracts[\"schedules\"]))",
             "self.assertTrue(any(row[\"actual_forwards\"] == 7 for row in contracts[\"schedules\"]))",
             "test_contract_manifest_schema"),
            ("self.assertEqual(token_ids[\"shape\"], [1, 512])",
             "self.assertEqual(token_ids[\"shape\"], [1, 1024])",
             "test_encode_manifests"),
            ("self.assertEqual(manifest[\"stage\"], \"denoise\")",
             "self.assertEqual(manifest[\"stage\"], \"encode\")",
             "test_denoise_manifests"),
            ("self.assertEqual(manifest[\"stage\"], \"decode\")",
             "self.assertEqual(manifest[\"stage\"], \"denoise\")",
             "test_decode_manifests"),
            ("self.assertEqual(arrays[\"decoded_pixels\"][\"shape\"], [1, 3, settings[\"height\"], settings[\"width\"]])",
             "self.assertEqual(arrays[\"decoded_pixels\"][\"shape\"], [1, 1, settings[\"height\"], settings[\"width\"]])",
             "test_decode_manifests"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            records = []
            for old, new, test in cases:
                text = Path(__file__).read_text()
                assert text.count(old) == 1, old
                mutated = text.replace(old, new)
                module_path = tmp_path / "test_z_image_capture_contracts.py"
                module_path.write_text(mutated)
                env = dict(os.environ)
                env["PYTHONPATH"] = str(tmp_path)
                env["Z_IMAGE_TEST_ROOT"] = str(ROOT)
                run = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_z_image_capture_contracts"],
                                     cwd=tmp_path, capture_output=True, text=True, env=env)
                failed = [line for line in (run.stdout + run.stderr).splitlines()
                          if line.endswith(" ... FAIL") or line.endswith(" ... ERROR")]
                assert run.returncode != 0, run.stdout + run.stderr
                assert len(failed) == 1 and test in failed[0], run.stdout + run.stderr
                records.append({"mutation": old, "expected_failed_case": test, "isolated_failure": True})
            output.write_text(json.dumps(records, indent=2) + "\n")
        print("capture-contract mutations generated: " + str(output))
        raise SystemExit(0)
    unittest.main()
