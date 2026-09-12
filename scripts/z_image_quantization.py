"""Quality-only affine INT4 emulation from canonical weights.

The dequantized BF16 tensors exercise quantization error in the reference
pipeline. Their memory and latency do not represent a packed INT4 runtime.
"""

import json

import numpy as np
from safetensors import safe_open
import torch

from z_image_compare import quantize


def apply_quantization(model, root, component):
    index_name = "model" if component == "text_encoder" else "diffusion_pytorch_model"
    mapping = json.loads((root / component / (index_name + ".safetensors.index.json")).read_text())["weight_map"]
    selected = []
    for name, layer in model.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        if component == "text_encoder":
            eligible = ".self_attn." in name or ".mlp." in name
            key = "model." + name + ".weight"
        else:
            eligible = ".attention.to_" in name or ".feed_forward.w" in name
            key = name + ".weight"
        if eligible:
            if layer.weight.shape[1] % 64:
                raise ValueError(f"unsupported group shape: {name}")
            selected.append((key, layer))
    rows = []
    # Read original FP32/BF16 values once, rather than quantizing an already
    # narrowed or quantized model. Only a small row chunk needs FP32 scratch.
    for shard in sorted({mapping[key] for key, _ in selected}):
        with safe_open(root / component / shard, framework="pt", device="cpu") as reader:
            for key, layer in selected:
                if mapping[key] != shard:
                    continue
                tensor = reader.get_slice(key)
                square_error = square_signal = 0.0
                for start in range(0, layer.weight.shape[0], 64):
                    original = tensor[start:start + 64].float().numpy()
                    dq, _, _, _ = quantize(original, 64)
                    square_error += float(np.square(original - dq, dtype=np.float64).sum())
                    square_signal += float(np.square(original, dtype=np.float64).sum())
                    layer.weight[start:start + 64].copy_(torch.from_numpy(dq))
                rows.append({"tensor": key, "shape": list(layer.weight.shape),
                             "relative_l2": (square_error / max(square_signal, 1e-30)) ** 0.5})
    if not rows:
        raise ValueError("no eligible linear weights were quantized")
    return {"layout": "affine-int4-group64-bf16-scale-bias", "emulation": "dequantized BF16 reference",
            "components": component, "tensors": rows,
            "exceptions": "embeddings, norms, biases, time/modulation, input/output projections, VAE"}
