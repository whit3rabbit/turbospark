"""CPU numeric oracle for generated expert programs, requires pinned coremltools.

uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
  python -I crates/coreml/tools/test_ane_expert_programs.py
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import coremltools as ct
import numpy as np


class ExpertPrograms(unittest.TestCase):
    def test_projection_orientation_and_fused_silu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "programs"
            subprocess.run([sys.executable, "-I", str(Path(__file__).with_name(
                "gen_ane_expert_programs.py")), str(root), "--hidden", "64",
                "--intermediate", "128", "--tokens", "17"], check=True)
            rng = np.random.default_rng(20261021)

            def values(shape):
                return rng.uniform(-0.25, 0.25, shape).astype(np.float16)

            a = values((1, 1, 17, 64))
            wg, wu = values((1, 1, 128, 64)), values((1, 1, 128, 64))
            wd = values((1, 1, 64, 128))
            def matmul(w, x):
                # FP64 scalar contraction avoids sharing the device's FP16 reduction.
                return np.einsum("...ik,...jk->...ij", w.astype(np.float64),
                                 x.astype(np.float64), optimize=False)
            gate, up = matmul(wg, a), matmul(wu, a)
            active = gate / (1 + np.exp(-gate)) * up
            down_input = values((1, 1, 17, 128))
            cases = {
                "gateup": (dict(a=a, w0=wg, w1=wu), [gate, up]),
                "down": (dict(a=down_input, w0=wd), [matmul(wd, down_input)]),
                "fused": (dict(a=a, wg=wg, wu=wu, wd=wd),
                          [np.einsum("...hi,...in->...hn", wd.astype(np.float64),
                                     active, optimize=False)]),
            }
            manifest = json.loads((root / "experiment.json").read_text())
            self.assertEqual(len(manifest["programs"]), 3)
            for row in manifest["programs"]:
                with self.subTest(kind=row["kind"]):
                    model = ct.models.MLModel(str(root / row["name"]),
                                              compute_units=ct.ComputeUnit.CPU_ONLY)
                    inputs, expected = cases[row["kind"]]
                    got = model.predict(inputs)
                    self.assertEqual(set(got), {f"o{i}" for i in range(len(expected))})
                    for i, want in enumerate(expected):
                        actual = got[f"o{i}"]
                        self.assertEqual(actual.shape, want.shape)
                        self.assertTrue(np.isfinite(want).all())
                        self.assertTrue(np.isfinite(actual).all())
                        scale = np.maximum(np.max(np.abs(want), axis=2, keepdims=True), 1e-6)
                        self.assertLess(float(np.max(np.abs(actual - want) / scale)), 0.005)


if __name__ == "__main__":
    unittest.main()
