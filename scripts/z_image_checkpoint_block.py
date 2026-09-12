#!/usr/bin/env python3
"""Cross-check one full-width checkpoint block on bounded captured activations."""

import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from safetensors import safe_open

from z_image_compare import errors, quantize
from z_image_evidence import check_environment, command, sha256, validate_capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("target/ig0/model"))
    p.add_argument("--sources", type=Path, default=Path("target/ig0/inputs"))
    args = p.parse_args()
    if "AC Power" not in command(["pmset", "-g", "batt"]):
        raise RuntimeError("AC power required")
    reference = validate_capture(args.run / "denoise.json")
    check_environment()
    import mlx.core as mx
    for name in ("mflux", "mflux.models", "mflux.models.z_image", "mflux.models.z_image.model",
                 "mflux.models.z_image.model.z_image_transformer"):
        package = types.ModuleType(name)
        package.__path__ = [str(args.sources / "mflux/src" / name.replace(".", "/"))]
        sys.modules[name] = package
    inv = json.loads(Path("docs/verification/z-image-ig0-inputs.json").read_text())
    for path, row in inv["references"]["mflux"]["files"].items():
        if sha256(args.sources / "mflux" / path) != row["sha256"]:
            raise ValueError("reference source drift")
    from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock as MBlock
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock as TBlock
    mapping = json.loads((args.model / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    prefix = "layers.0."
    state = {}
    for shard in sorted({s for k, s in mapping.items() if k.startswith(prefix)}):
        # Verify reused payloads as well as the provenance record.
        if reference["verified_weights"]["transformer/" + shard] != inv["weights"]["transformer/" + shard]["sha256"]:
            raise ValueError("checkpoint provenance mismatch")
        if sha256(args.model / "transformer" / shard) != inv["weights"]["transformer/" + shard]["sha256"]:
            raise ValueError("checkpoint payload drift")
        with safe_open(args.model / "transformer" / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = f.get_tensor(key)
    with torch.device("meta"):
        tb = TBlock(0, 3840, 30, 30, 1e-5, True).eval()
    tb.load_state_dict(state, assign=True)
    mb = MBlock(3840, 30)
    mb.load_weights([(key, mx.array(value.numpy())) for key, value in state.items()], strict=True)
    x = torch.from_numpy(np.load(args.run / "block_00_input.npy")[:, :64].copy())
    mask = torch.from_numpy(np.load(args.run / "block_00_mask.npy")[:, :64].copy())
    freq = torch.from_numpy(np.load(args.run / "block_00_freqs.npy")[:, :64].copy())
    modulation = torch.from_numpy(np.load(args.run / "block_00_modulation.npy"))
    freq_m = mx.array(torch.view_as_real(freq[0]).numpy())
    tb.to("mps")
    with torch.inference_mode():
        expected = tb(x.to("mps"), mask.to("mps"), freq.to("mps"), modulation.to("mps")).cpu().numpy()
    actual = mb(mx.array(x.numpy()), mx.array(mask.numpy()), freq_m, mx.array(modulation.numpy()))
    mx.eval(actual)
    metrics = errors(expected, np.array(actual))
    tolerances = {"max_abs": 3e-5, "relative_l2": 1e-6}
    if any(metrics[key] > limit for key, limit in tolerances.items()):
        raise ValueError(f"checkpoint block comparison failed: {metrics}")
    quantized = []
    with torch.inference_mode():
        for name, layer in tb.named_modules():
            if isinstance(layer, torch.nn.Linear) and (name.startswith("attention.") or name.startswith("feed_forward.")):
                weight = layer.weight.cpu().numpy()
                dq = np.empty_like(weight)
                for start in range(0, len(weight), 64):
                    dq[start:start+64] = quantize(weight[start:start+64], 64)[0]
                layer.weight.copy_(torch.from_numpy(dq).to("mps"))
                quantized.append(name)
        qout = tb(x.to("mps"), mask.to("mps"), freq.to("mps"), modulation.to("mps")).cpu().numpy()
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "checkpoint_block.npz", input=x.numpy(), mask=mask.numpy(), freqs=freq.numpy(),
             modulation=modulation.numpy(), torch_fp32=expected, mlx_fp32=np.array(actual), affine64=qout)
    report = {"source_capture_sha256": sha256(args.run / "denoise.json"), "references": reference["references"],
              "script_sha256": sha256(__file__), "geometry": {"width": 3840, "heads": 30, "tokens": 64},
              "cross_reference_fp32": metrics, "tolerances": tolerances, "quantized_block": errors(expected, qout),
              "quantized_linears": quantized, "performance_measurement": False,
              "scope": "block 0, first 64 captured tokens with attention restricted to those tokens",
              "fixture_sha256": sha256(args.out / "checkpoint_block.npz")}
    (args.out / "checkpoint_block.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
