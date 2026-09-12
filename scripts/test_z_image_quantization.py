"""Quantization-policy guards using small safetensors and the real code path."""

import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from z_image_quantization import apply_quantization


class QuantizationTests(unittest.TestCase):
    def test_canonical_weights_and_precision_exceptions(self):
        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            root = Path(tmp)
            component = root / "transformer"
            component.mkdir()
            model = torch.nn.Module()
            block = torch.nn.Module()
            block.attention = torch.nn.Module()
            block.attention.to_q = torch.nn.Linear(64, 2, bias=False, dtype=torch.bfloat16)
            block.adaLN_modulation = torch.nn.Sequential(torch.nn.Linear(64, 2, dtype=torch.bfloat16))
            model.layers = torch.nn.ModuleList([block])
            model.output = torch.nn.Linear(64, 2, dtype=torch.bfloat16)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            canonical = torch.zeros(2, 64, dtype=torch.float32)
            weights = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in before.items()
                       if key.endswith(".weight")}
            save_file(weights, component / "weights.safetensors")
            (component / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {key: "weights.safetensors" for key in weights}}))
            result = apply_quantization(model, root, "transformer")
            self.assertEqual([row["tensor"] for row in result["tensors"]], ["layers.0.attention.to_q.weight"])
            self.assertTrue(torch.equal(block.attention.to_q.weight, canonical.to(torch.bfloat16)))
            for key, value in model.state_dict().items():
                if key != "layers.0.attention.to_q.weight":
                    self.assertTrue(torch.equal(value, before[key]), key)


if __name__ == "__main__":
    unittest.main()
