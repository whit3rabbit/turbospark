#!/usr/bin/env python3
"""Compare pinned MFLUX VAE math with a captured FP32 Diffusers decode."""

import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np
from safetensors import safe_open

from z_image_compare import errors
from z_image_evidence import check_environment, command, sha256, validate_capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sources", type=Path, default=Path("target/ig0/inputs"))
    p.add_argument("--model", type=Path, default=Path("target/ig0/model"))
    args = p.parse_args()
    if "AC Power" not in command(["pmset", "-g", "batt"]):
        raise RuntimeError("AC power required")
    packages = check_environment()
    reference = validate_capture(args.run / "decode.json")
    inv = json.loads(Path("docs/verification/z-image-ig0-inputs.json").read_text())
    for path, row in inv["references"]["mflux"]["files"].items():
        if sha256(args.sources / "mflux" / path) != row["sha256"]:
            raise ValueError("MFLUX source drift")
    weights = args.model / "vae/diffusion_pytorch_model.safetensors"
    if sha256(weights) != reference["verified_weights"]["vae/diffusion_pytorch_model.safetensors"]:
        raise ValueError("VAE payload drift")
    import mlx.core as mx
    names = ["mflux", "mflux.models", "mflux.models.common", "mflux.models.z_image",
             "mflux.models.z_image.model", "mflux.models.z_image.model.z_image_vae",
             "mflux.models.z_image.model.z_image_vae.common", "mflux.models.z_image.model.z_image_vae.decoder"]
    for name in names:
        package = types.ModuleType(name)
        package.__path__ = [str(args.sources / "mflux/src" / name.replace(".", "/"))]
        sys.modules[name] = package
    # VAE modules only read ModelConfig.precision. Supply the explicit FP32
    # comparison setting without importing the unrelated model registry.
    config = types.ModuleType("mflux.models.common.config")
    config.ModelConfig = types.SimpleNamespace(precision=mx.float32)
    sys.modules[config.__name__] = config
    from mflux.models.z_image.model.z_image_vae.decoder.decoder import Decoder
    decoder = Decoder()
    mapped = []
    with safe_open(weights, framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith("decoder."):
                continue
            value = f.get_tensor(key).float().numpy()
            name = key.removeprefix("decoder.")
            for wrapper in ("conv_in", "conv_out"):
                if name.startswith(wrapper + "."):
                    name = name.replace(wrapper + ".", wrapper + ".conv.", 1)
            if name.startswith("conv_norm_out."):
                name = name.replace("conv_norm_out.", "conv_norm_out.norm.", 1)
            if value.ndim == 4:
                value = value.transpose(0, 2, 3, 1)
            mapped.append((name, mx.array(value)))
    decoder.load_weights(mapped, strict=True)
    latents = mx.array(np.load(args.run / "final_latents.npy"))
    actual = decoder(latents / 0.3611 + 0.1159)
    mx.eval(actual)
    expected = np.load(args.run / "decoded_pixels.npy")
    actual = np.array(actual)
    result = errors(expected, actual)
    tolerances = {"max_abs": 6e-5, "relative_l2": 3e-6}
    if any(result[key] > limit for key, limit in tolerances.items()):
        raise ValueError(f"VAE comparison failed: {result}")
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "mlx_decoded_pixels.npy", actual)
    report = {"packages": packages, "references": reference["references"], "source_capture_sha256": sha256(args.run / "decode.json"),
              "script_sha256": sha256(__file__), "precision_override": "ModelConfig.precision = float32",
              "parameter_layout_adapter": "OIHW to OHWI for convolution; three wrapper-name insertions",
              "decoder_tensors": len(mapped), "shape": list(actual.shape), "errors": result, "tolerances": tolerances,
              "fixture_sha256": sha256(args.out / "mlx_decoded_pixels.npy"), "performance_measurement": False}
    (args.out / "vae_compare.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
