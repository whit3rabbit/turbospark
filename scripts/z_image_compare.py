#!/usr/bin/env python3
"""Bounded CPU cross-reference block and real-weight quantization probes.

Synthetic blocks establish operator agreement only, not checkpoint/image
quality. Weight rows are sampled directly from pinned safetensors payloads.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
import subprocess

import numpy as np
import torch

from z_image_evidence import check_environment, sha256
from z_image_probe import MODEL, REVISION, REFERENCES, fetch


def errors(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid comparison")
    diff = a - b
    return {"max_abs": float(np.max(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(a), 1e-30))}


def compare_blocks(source, out):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    sys.path.insert(0, str(source / "mflux/src"))
    # Import only the pinned block modules. Package initializers eagerly load
    # unrelated pipelines and optional dependencies; no math is substituted.
    for name in ("mflux", "mflux.models", "mflux.models.z_image", "mflux.models.z_image.model",
                 "mflux.models.z_image.model.z_image_transformer"):
        package = types.ModuleType(name)
        package.__path__ = [str(source / "mflux/src" / name.replace(".", "/"))]
        sys.modules[name] = package
    from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock as MBlock
    from mflux.models.z_image.model.z_image_transformer.rope_embedder import RopeEmbedder as MRope
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock as TBlock, RopeEmbedder as TRope
    torch.manual_seed(42)
    tb = TBlock(0, 384, 3, 3, 1e-5, True).eval()
    mb = MBlock(384, 3)
    weights = [(name, mx.array(w.detach().numpy())) for name, w in tb.state_dict().items()]
    mb.load_weights(weights, strict=True)
    rows, fixtures = [], {}
    for tokens in (1, 16, 35):
        x = torch.randn(1, tokens, 384)
        modulation = torch.randn(1, 256)
        mask = torch.ones(1, tokens, dtype=torch.bool)
        if tokens > 1:
            mask[:, -1] = False
        ids = torch.stack((torch.arange(tokens), torch.arange(tokens) % 8, torch.arange(tokens) % 7), dim=1)
        tr = TRope(theta=256, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512])(ids)
        mr = MRope(theta=256, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512])(mx.array(ids.numpy()))
        with torch.inference_mode():
            expected = tb(x, mask, tr.unsqueeze(0), modulation).numpy()
        actual = mb(mx.array(x.numpy()), mx.array(mask.numpy()), mr, mx.array(modulation.numpy()))
        mx.eval(actual)
        metrics = errors(expected, np.array(actual))
        rope_metrics = errors(torch.view_as_real(tr).numpy(), np.array(mr))
        if metrics["max_abs"] > 1e-5 or metrics["relative_l2"] > 1e-6 or rope_metrics["max_abs"] > 3e-6:
            raise ValueError("bounded FP32 reference comparison exceeded frozen tolerances")
        rows.append({"tokens": tokens, "block": metrics, "rope": rope_metrics})
        fixtures.update({f"x_{tokens}": x.numpy(), f"modulation_{tokens}": modulation.numpy(),
                         f"mask_{tokens}": mask.numpy(), f"ids_{tokens}": ids.numpy(),
                         f"torch_{tokens}": expected, f"mlx_{tokens}": np.array(actual)})
    for name, value in tb.state_dict().items():
        fixtures["weight_" + name] = value.numpy()
    np.savez(out / "synthetic_blocks.npz", **fixtures)
    return {"geometry": {"dim": 384, "heads": 3, "head_dim": 128}, "cases": rows,
            "tolerances": {"block_max_abs": 1e-5, "block_relative_l2": 1e-6, "rope_max_abs": 3e-6},
            "fixture_sha256": sha256(out / "synthetic_blocks.npz"), "checkpoint_weights": False}


def bf16(x):
    return torch.from_numpy(x.copy()).to(torch.bfloat16).float().numpy()


def quantize(w, group):
    blocks = w.reshape(w.shape[0], -1, group)
    low, high = blocks.min(-1, keepdims=True), blocks.max(-1, keepdims=True)
    scales = bf16(np.where(high == low, 1, (high - low) / 15))
    biases = bf16(low)
    # Rust f32::round rounds positive half-integers away from zero.
    inv_scale = np.divide(np.float32(1), scales, out=np.zeros_like(scales), where=scales != 0)
    values = ((blocks - biases) * inv_scale).astype(np.float64)
    q = np.clip(np.floor(values + 0.5), 0, 15).astype(np.uint8)
    dequant = (q * scales + biases).reshape(w.shape)
    packed = q.reshape(w.shape)[:, ::2] | (q.reshape(w.shape)[:, 1::2] << 4)
    return dequant, packed, scales, biases


def compare_quant(inventory, out):
    rng = np.random.default_rng(42)
    report = json.loads(inventory.read_text())
    requests = [("text_encoder", "model.layers.0.self_attn.q_proj.weight"),
                ("transformer", "layers.0.attention.to_q.weight"),
                ("transformer", "layers.0.feed_forward.w1.weight"),
                ("transformer", "layers.0.adaLN_modulation.0.weight")]
    rows = []
    binary = out / "quant_ref"
    rust_source = Path(__file__).with_name("z_image_quant_ref.rs")
    subprocess.run(["rustc", "--edition=2021", str(rust_source), "-o", str(binary)], check=True)
    for component, name in requests:
        matches = [(p, s, s["tensors"][name]) for p, s in report["weights"].items()
                   if p.startswith(component + "/") and name in s["tensors"]]
        if len(matches) != 1:
            raise ValueError(f"missing/ambiguous weight: {name}")
        path, shard, tensor = matches[0]
        n, k = min(64, tensor["shape"][0]), tensor["shape"][1]
        width = {"BF16": 2, "F32": 4}[tensor["dtype"]]
        offset = 8 + shard["header_bytes"] + tensor["data_offsets"][0]
        raw = fetch(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{path}?sample={offset}", offset, n*k*width)
        if width == 2:
            w = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(n, k)
        else:
            w = np.frombuffer(raw, dtype="<f4").reshape(n, k)
        x = rng.standard_normal((16, k)).astype(np.float32)
        label = component + "_" + name.replace(".", "_")
        np.save(out / (label + ".npy"), w)
        for group in (32, 64, 128):
            dq, packed, scales, biases = quantize(w, group)
            rows.append({"component": component, "tensor": name, "rows": n, "columns": k,
                         "group_size": group, "bits": 4, "source_range_sha256": hashlib.sha256(raw).hexdigest(),
                         "weight_error": errors(w, dq), "synthetic_projection_error": errors(x @ w.T, x @ dq.T),
                         "production_kernel_compatible": group == 64})
            if group == 64:
                raw_path = out / (label + ".f32")
                w.astype("<f4").tofile(raw_path)
                prefix = out / label
                subprocess.run([str(binary.resolve()), str(raw_path), str(prefix)], check=True)
                rust_packed = np.fromfile(str(prefix) + ".packed", dtype=np.uint8).reshape(packed.shape)
                rust_scale = (np.fromfile(str(prefix) + ".scales", dtype="<u2").astype(np.uint32) << 16).view(np.float32)
                rust_bias = (np.fromfile(str(prefix) + ".biases", dtype="<u2").astype(np.uint32) << 16).view(np.float32)
                if not (np.array_equal(packed, rust_packed) and np.array_equal(scales.ravel(), rust_scale)
                        and np.array_equal(biases.ravel(), rust_bias)):
                    raise ValueError("Python/Rust affine packing disagreement")
                rows[-1]["rust_packing_exact"] = True
                np.savez(out / (label + "_int4.npz"), packed=packed, scales=scales, biases=biases, x=x, y=x@dq.T)
    return {"cases": rows, "quality_gate_passed": False,
            "limitation": "first 64 rows per tensor, synthetic activations; not full-block or image quality"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["blocks", "quant"])
    p.add_argument("--sources", type=Path, default=Path("target/ig0/inputs"))
    p.add_argument("--inventory", type=Path, default=Path("docs/verification/z-image-ig0-inputs.json"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Reject source drift before executing cached reference Python modules.
    inv = json.loads(args.inventory.read_text())
    for label, reference in inv["references"].items():
        for path, row in reference["files"].items():
            if sha256(args.sources / label / path) != row["sha256"]:
                raise ValueError(f"reference source drift: {label}/{path}")
    data = {"model_revision": REVISION, "references": {k: v[1] for k,v in REFERENCES.items()},
            "packages": check_environment(), "device": "cpu", "performance_measurement": False}
    data["script_sha256"] = sha256(__file__)
    if args.mode == "quant":
        data["rust_quantizer_sha256"] = sha256(Path(__file__).resolve().parent.parent / "crates/compute/src/quant.rs")
    manifest = args.out / (args.mode + ".json")
    # A failed refresh must not leave an older success beside new partial fixtures.
    manifest.write_text(json.dumps({"complete": False}) + "\n")
    data[args.mode] = compare_blocks(args.sources, args.out) if args.mode == "blocks" else compare_quant(args.inventory, args.out)
    data["fixtures"] = {path.name: sha256(path) for path in args.out.iterdir() if path.suffix in (".npy", ".npz")}
    data["complete"] = True
    manifest.write_text(json.dumps(data, indent=2) + "\n")
