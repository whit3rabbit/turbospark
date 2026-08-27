#!/usr/bin/env python3
"""Vision Phase 0 probes: activation magnitude through the qwen3_5 vision
tower, and INT4-transcode quality, against the REAL reference implementation
(never a reimplementation) vendored at `../mlx-v/mlx-vlm`. See
`docs/VISION_PHASE0.md` for the results this produced and what they decided
(FP16, not INT4, for the tower's matmul weights -- reverses the original
plan's default).

Stubs the top-level `mlx_vlm` package in `sys.modules` before importing, so
loading `mlx_vlm.models.qwen3_vl.vision` does not execute the real
`mlx_vlm/__init__.py` (which pulls in the full generate/convert/lora/server
dependency chain this needs none of).

Needs only the vision_tower.* tensors (`fetch_vision_tower.py`), never the
text trunk.

Usage:
    uv run --python 3.12 --with mlx --with numpy --with pillow --with transformers -- \
      python scripts/vision_tower_probe.py \
        --vendor-root ../mlx-v/mlx-vlm \
        --config path/to/config.json \
        --tower-dir /tmp/vision-probe \
        --image path/to/test_page.png \
        --mode activation

    ... --mode int4
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


def load_vision_model(vendor_root: str):
    pkg = types.ModuleType("mlx_vlm")
    pkg.__path__ = [str(Path(vendor_root) / "mlx_vlm")]
    sys.modules["mlx_vlm"] = pkg
    from mlx_vlm.models.qwen3_vl.vision import VisionModel
    from mlx_vlm.models.qwen3_vl.config import VisionConfig
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    return VisionModel, VisionConfig, Qwen3VLImageProcessor


def load_vision_config(VisionConfig, config_path: str):
    vc = json.load(open(config_path))["vision_config"]
    return VisionConfig(
        model_type=vc["model_type"], depth=vc["depth"], hidden_size=vc["hidden_size"],
        intermediate_size=vc["intermediate_size"], out_hidden_size=vc["out_hidden_size"],
        num_heads=vc["num_heads"], patch_size=vc["patch_size"], in_channels=vc["in_channels"],
        spatial_merge_size=vc["spatial_merge_size"], temporal_patch_size=vc["temporal_patch_size"],
        num_position_embeddings=vc["num_position_embeddings"],
        deepstack_visual_indexes=vc["deepstack_visual_indexes"],
    )


def load_tower_weights(mx, tower_dir: str) -> dict:
    hdr = json.load(open(Path(tower_dir) / "vision_tower_header.json"))
    blob = open(Path(tower_dir) / "vision_tower.bin", "rb").read()
    weights = {}
    for name, meta in hdr.items():
        s, e = meta["data_offsets"]
        if meta["dtype"] != "F16":
            raise ValueError(f"unexpected dtype {meta['dtype']} for {name}")
        arr = np.frombuffer(blob[s:e], dtype=np.float16).reshape(meta["shape"])
        weights[name[len("vision_tower."):]] = mx.array(arr)
    return weights


def build_model(mx, VisionModel, vision_config, weights: dict):
    model = VisionModel(vision_config)
    model.load_weights(list(model.sanitize(dict(weights)).items()))
    mx.eval(model.parameters())
    return model


def preprocess(mx, Qwen3VLImageProcessor, vision_config, image_path: str):
    proc = Qwen3VLImageProcessor(
        patch_size=vision_config.patch_size,
        temporal_patch_size=vision_config.temporal_patch_size,
        merge_size=vision_config.spatial_merge_size,
        min_pixels=65536, max_pixels=16777216,
    )
    out = proc([image_path])
    pixel_values = mx.array(np.array(out["pixel_values"]).astype(np.float32).astype(np.float16))
    grid_thw = mx.array(np.array(out["image_grid_thw"]).astype(np.int32))
    return pixel_values, grid_thw


def run_forward(mx, model, pixel_values, grid_thw, dtype=None):
    if dtype is not None:
        pixel_values = pixel_values.astype(dtype)
    h = model.patch_embed(pixel_values)
    pos = model.fast_pos_embed_interpolate(grid_thw)
    h = h + pos
    seq_len = h.shape[0]
    h = h.reshape(seq_len, -1)
    rope = model.rot_pos_emb(grid_thw).reshape(seq_len, -1)
    if dtype is not None:
        rope = rope.astype(dtype)
    cu = mx.array([0, seq_len], dtype=mx.int32)
    for blk in model.blocks:
        h = blk(h, cu_seqlens=cu, rotary_pos_emb=rope)
    return model.merger(h)


def mode_activation(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor):
    vision_config = load_vision_config(VisionConfig, args.config)
    weights = load_tower_weights(mx, args.tower_dir)
    model = build_model(mx, VisionModel, vision_config, weights)
    pixel_values, grid_thw = preprocess(mx, Qwen3VLImageProcessor, vision_config, args.image)
    print("grid_thw:", grid_thw.tolist(), "patches:", pixel_values.shape[0])

    def stats(x, label):
        xf = x.astype(mx.float32)
        mx.eval(xf)
        absmax = float(mx.max(mx.abs(xf)).item())
        rms = float(mx.sqrt(mx.mean(xf * xf)).item())
        flag = " <-- EXCEEDS FP16 RANGE (65504)" if absmax > 65504.0 else ""
        print(f"  {label:24s} absmax={absmax:12.3f} rms={rms:10.4f}{flag}")

    print("\n-- fp16 forward pass --")
    h = model.patch_embed(pixel_values)
    pos = model.fast_pos_embed_interpolate(grid_thw)
    h = h + pos
    seq_len = h.shape[0]
    h = h.reshape(seq_len, -1)
    rope = model.rot_pos_emb(grid_thw).reshape(seq_len, -1)
    cu = mx.array([0, seq_len], dtype=mx.int32)
    for i, blk in enumerate(model.blocks):
        h = blk(h, cu_seqlens=cu, rotary_pos_emb=rope)
        stats(h, f"block {i} out")
    merged16 = model.merger(h)
    stats(merged16, "merger out (fp16)")

    print("\n-- fp32 forward pass --")
    merged32 = run_forward(mx, model, pixel_values, grid_thw, dtype=mx.float32).astype(mx.float32)
    merged16f = merged16.astype(mx.float32)
    mx.eval(merged32, merged16f)
    cos = float((mx.sum(merged16f * merged32) /
                 (mx.sqrt(mx.sum(merged16f ** 2)) * mx.sqrt(mx.sum(merged32 ** 2)))).item())
    print(f"\nfp16-vs-fp32 merger cosine similarity: {cos:.6f}")


def mode_int4(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor):
    vision_config = load_vision_config(VisionConfig, args.config)
    weights = load_tower_weights(mx, args.tower_dir)
    ref_model = build_model(mx, VisionModel, vision_config, weights)

    group_size, bits = args.group_size, args.bits
    quant_weights, n_q, n_skip = {}, 0, 0
    for name, w in weights.items():
        is_matmul = name.endswith(".weight") and w.ndim == 2 and "norm" not in name
        if is_matmul and w.shape[-1] % group_size == 0:
            wq, scales, biases = mx.quantize(w.astype(mx.float32), group_size=group_size, bits=bits)
            quant_weights[name] = mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits).astype(mx.float16)
            n_q += 1
        else:
            quant_weights[name] = w
            n_skip += 1
    print(f"quantized {n_q} matmul weights to INT{bits} g{group_size}, left {n_skip} untouched")

    quant_model = build_model(mx, VisionModel, vision_config, quant_weights)
    pixel_values, grid_thw = preprocess(mx, Qwen3VLImageProcessor, vision_config, args.image)

    ref_out = run_forward(mx, ref_model, pixel_values, grid_thw).astype(mx.float32)
    quant_out = run_forward(mx, quant_model, pixel_values, grid_thw).astype(mx.float32)
    mx.eval(ref_out, quant_out)

    diff = mx.abs(ref_out - quant_out)
    cos_per_token = mx.sum(ref_out * quant_out, axis=-1) / (
        mx.sqrt(mx.sum(ref_out ** 2, axis=-1)) * mx.sqrt(mx.sum(quant_out ** 2, axis=-1)) + 1e-8)
    cos_overall = float((mx.sum(ref_out * quant_out) /
                          (mx.sqrt(mx.sum(ref_out ** 2)) * mx.sqrt(mx.sum(quant_out ** 2)))).item())
    print(f"\noverall cosine: {cos_overall:.6f}")
    print(f"per-token cosine: min={float(mx.min(cos_per_token).item()):.6f} "
          f"mean={float(mx.mean(cos_per_token).item()):.6f}")
    print(f"max abs diff: {float(mx.max(diff).item()):.4f}  mean abs diff: {float(mx.mean(diff).item()):.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vendor-root", required=True, help="path to the mlx-v/mlx-vlm checkout")
    p.add_argument("--config", required=True, help="checkpoint's config.json")
    p.add_argument("--tower-dir", required=True, help="dir written by fetch_vision_tower.py")
    p.add_argument("--image", required=True)
    p.add_argument("--mode", choices=["activation", "int4"], required=True)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--bits", type=int, default=4)
    args = p.parse_args()

    VisionModel, VisionConfig, Qwen3VLImageProcessor = load_vision_model(args.vendor_root)
    import mlx.core as mx  # import after the module stub is installed

    if args.mode == "activation":
        mode_activation(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor)
    else:
        mode_int4(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor)


if __name__ == "__main__":
    main()
