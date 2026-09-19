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
    """Both published dtypes, both landing in FP16.

    `prism-ml/Bonsai-27B-mlx-1bit` ships the tower F16 and
    `mlx-community/Qwen3.8-27B-4bit` ships the SAME 333 tensors at the SAME
    shapes in BF16 (`crates/repack` Gotcha 13). The install's repack converts
    the second to FP16, so a parity comparison has to run the reference on the
    converted values -- otherwise the two sides differ by a storage width as
    well as by whatever is being measured.

    That conversion is EXACT in FP16's normal range: the mantissa widens from
    7 bits to 10, so it cannot round, and only the exponent can fail. Values
    above 65504 would become `inf` and the walk refuses those by name; this
    side asserts none appear rather than clamping, because an `inf` weight
    becomes NaN in the activations and NaN reads as a perfect score on every
    instrument downstream (AGENTS.md Gotcha 59).

    `numpy` cannot decode BF16 (`TypeError: data type 'bfloat16' not
    understood`), so it is read as uint16 and shifted into the top half of an
    f32 -- which is what BF16 IS.
    """
    hdr = json.load(open(Path(tower_dir) / "vision_tower_header.json"))
    blob = open(Path(tower_dir) / "vision_tower.bin", "rb").read()
    weights = {}
    for name, meta in hdr.items():
        s, e = meta["data_offsets"]
        if meta["dtype"] == "F16":
            arr = np.frombuffer(blob[s:e], dtype=np.float16).reshape(meta["shape"])
        elif meta["dtype"] == "BF16":
            bits = np.frombuffer(blob[s:e], dtype=np.uint16).astype(np.uint32) << 16
            wide = bits.view(np.float32)
            if not np.isfinite(wide).all() or np.abs(wide).max() > 65504.0:
                raise ValueError(
                    f"{name} carries a value FP16 cannot hold; the install's walk refuses "
                    f"these by name and so does this"
                )
            arr = wide.astype(np.float16).reshape(meta["shape"])
        else:
            raise ValueError(f"unexpected dtype {meta['dtype']} for {name}")
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


def mode_dump(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor):
    """Write the reference's per-stage tensors for `vision_tower_parity.rs`.

    FOUR STAGES, not just the merger: patch embed, block 0, the last block,
    and the merger output. A convention that shows as 0.99 at the merger
    localizes immediately when block 0 is exact and the last block is not,
    which merger-only cannot do.

    THE PATCH ROWS ARE DUMPED TOO, and that is the load-bearing part. The Rust
    side replays THESE rows rather than preprocessing the image itself, so a
    preprocessing difference cannot read as a kernel gap -- the same
    discipline `logit_dump` plus `kld_*.py` already use for the text path,
    where the ids are replayed and never the prose.

    Everything is written float32 in a plain `{header.json, <name>.bin}` pair
    rather than `.npy`, so the Rust reader needs no format parser.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vision_config = load_vision_config(VisionConfig, args.config)
    weights = load_tower_weights(mx, args.tower_dir)
    model = build_model(mx, VisionModel, vision_config, weights)
    pixel_values, grid_thw = preprocess(mx, Qwen3VLImageProcessor, vision_config, args.image)

    grid = [int(v) for v in np.array(grid_thw).reshape(-1)]
    print("grid_thw:", grid, "patches:", pixel_values.shape[0])

    stages = {}

    def keep(name, x):
        a = np.array(x.astype(mx.float32), copy=True).astype(np.float32)
        stages[name] = a
        print(f"  {name:16s} shape={a.shape} absmax={np.abs(a).max():.4f}")

    # THE REFERENCE'S PIXEL VALUES CARRY (C, T, P_h, P_w) INSIDE A ROW where
    # this port emits (T, P_h, P_w, C) -- deliberately, so the patch-embed
    # weight copies verbatim (`crates/vision-io` Gotcha 1). The permutation is
    # done HERE rather than in the Rust reader, because doing it on the
    # reference side keeps the comparison's own bridge next to the reference
    # it bridges from.
    p_sz, t_sz, c = vision_config.patch_size, vision_config.temporal_patch_size, vision_config.in_channels
    pv = np.array(pixel_values.astype(mx.float32)).astype(np.float32)
    rows = pv.reshape(-1, c, t_sz, p_sz, p_sz).transpose(0, 2, 3, 4, 1).reshape(pv.shape[0], -1)
    keep("patch_rows", mx.array(rows))

    h = model.patch_embed(pixel_values)
    pos = model.fast_pos_embed_interpolate(grid_thw)
    h = h + pos
    keep("patch_embed", h)

    seq_len = h.shape[0]
    h = h.reshape(seq_len, -1)
    rope = model.rot_pos_emb(grid_thw).reshape(seq_len, -1)
    cu = mx.array([0, seq_len], dtype=mx.int32)
    last = len(model.blocks) - 1
    for i, blk in enumerate(model.blocks):
        h = blk(h, cu_seqlens=cu, rotary_pos_emb=rope)
        if i == 0:
            keep("block_0", h)
        # The DEEPSTACK mergers (`qwen3_vl`'s second injection seam): one
        # per declared index, run on the block output the way the tower's
        # own forward does, so the Rust side can compare each merger's
        # `[merged, out_hidden]` rows at the same bar as the main merger.
        if i in model.deepstack_visual_indexes:
            k = model.deepstack_visual_indexes.index(i)
            keep(f"deepstack_merger_{k}", model.deepstack_merger_list[k](h))
        if i == last:
            keep(f"block_{last}", h)
    keep("merger", model.merger(h))

    header = {"grid_thw": grid, "stages": {}}
    for name, a in stages.items():
        (out_dir / f"{name}.bin").write_bytes(a.tobytes())
        header["stages"][name] = {"shape": list(a.shape)}
    (out_dir / "header.json").write_text(json.dumps(header, indent=2))
    print(f"\nwrote {len(stages)} stages to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vendor-root", required=True, help="path to the mlx-v/mlx-vlm checkout")
    p.add_argument("--config", required=True, help="checkpoint's config.json")
    p.add_argument("--tower-dir", required=True, help="dir written by fetch_vision_tower.py")
    p.add_argument("--image", required=True)
    p.add_argument("--mode", choices=["activation", "int4", "dump"], required=True)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--out-dir", help="where --mode dump writes its stages")
    args = p.parse_args()
    if args.mode == "dump" and not args.out_dir:
        p.error("--mode dump needs --out-dir")

    VisionModel, VisionConfig, Qwen3VLImageProcessor = load_vision_model(args.vendor_root)
    import mlx.core as mx  # import after the module stub is installed

    if args.mode == "activation":
        mode_activation(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor)
    elif args.mode == "dump":
        mode_dump(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor)
    else:
        mode_int4(mx, args, VisionModel, VisionConfig, Qwen3VLImageProcessor)


if __name__ == "__main__":
    main()
