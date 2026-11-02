#!/usr/bin/env python3
"""A numpy transcription of the MiniCPM-V 4.7 vision tower (16x mode), for the
parity test `crates/runtime/src/vision/minicpm.rs`
(`the_real_tower_matches_the_numpy_reference`).

**What this is and is not.** It is written from mlx-vlm's
`models/minicpmv4_6/{vision,minicpmv4_6}.py` and `models/minicpmv4_7/` (read,
never executed), so a convention both it and the Rust tower get wrong would
pass. It is NOT mlx-vlm run on the same bytes. It shares no code with the
Rust tower and is written in the reference's own raster formulation (the
`reshape(h, w, d)` windowing, the `transpose(0, 2, 1, 3, 4)` merges, the
bucketed positions), where the Rust tower permutes tokens on the host: so the
window grouping, the merge order and the position buckets are each
transcribed rather than derived.

Pure numpy and the standard library. Patch rows come from the Rust side, in
raster order with (P_h, P_w, C) inside a row (the reference's packed layout).

Usage:
    python3 -I scripts/minicpm_v_tower_reference.py \
      --model-dir /Volumes/T7/hf-models/MiniCPM-V-4.7-35B-A3B-mxfp4 \
      --rows rows.bin --grid 32 32 --out-dir dump/
Writes {header.json, <stage>.bin} float32, in the reference's raster order:
patch_embed [N, 1152], block_0 [N, 1152], vit_merger [N/4, 1152],
last_block [N/4, 1152], output [N/16, 2048].
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

EPS = 1e-6
INSERT_LAYER = 6


def read_safetensors(path, prefixes):
    """BF16/F16 tensors whose names start with a prefix, as float32 rounded
    through FP16 (the precision the install stores them at)."""
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__" or not name.startswith(prefixes):
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if meta["dtype"] == "BF16":
                bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = bits.view(np.float32)
            elif meta["dtype"] == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            else:
                raise ValueError(f"{name}: {meta['dtype']}")
            out[name] = arr.astype(np.float16).astype(np.float32).reshape(meta["shape"])
    return out


def layer_norm(x, w, b):
    x = x.astype(np.float32)
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def attention(q, k, v, heads):
    """Bidirectional attention. q, k, v: [..., n, heads * d]."""
    *lead, n, dim = q.shape
    d = dim // heads
    q = q.reshape(*lead, n, heads, d)
    k = k.reshape(*lead, n, heads, d)
    v = v.reshape(*lead, n, heads, d)
    scores = np.einsum("...qhd,...khd->...hqk", q, k) * (d**-0.5)
    scores -= scores.max(-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(-1, keepdims=True)
    out = np.einsum("...hqk,...khd->...qhd", p, v)
    return out.reshape(*lead, n, dim)


def position_buckets(n, side):
    """`_build_position_buckets` with the reference's float32 frac and its
    float64-built boundaries cast to float32."""
    boundaries = np.arange(1 / side, 1.0, 1 / side).astype(np.float32)
    n = max(int(n), 1)
    frac = np.clip(
        np.arange(n, dtype=np.float32) / np.float32(n), 0.0, np.float32(1.0 - 1e-6)
    )
    return (frac[:, None] >= boundaries[None, :]).sum(axis=1)


def siglip_block(x, wts, p, heads):
    h = layer_norm(x, wts[p + "layer_norm1.weight"], wts[p + "layer_norm1.bias"])
    q = h @ wts[p + "self_attn.q_proj.weight"].T + wts[p + "self_attn.q_proj.bias"]
    k = h @ wts[p + "self_attn.k_proj.weight"].T + wts[p + "self_attn.k_proj.bias"]
    v = h @ wts[p + "self_attn.v_proj.weight"].T + wts[p + "self_attn.v_proj.bias"]
    a = attention(q, k, v, heads)
    x = x + a @ wts[p + "self_attn.out_proj.weight"].T + wts[p + "self_attn.out_proj.bias"]
    h = layer_norm(x, wts[p + "layer_norm2.weight"], wts[p + "layer_norm2.bias"])
    h = gelu_tanh(h @ wts[p + "mlp.fc1.weight"].T + wts[p + "mlp.fc1.bias"])
    return x + h @ wts[p + "mlp.fc2.weight"].T + wts[p + "mlp.fc2.bias"]


def vit_merger(x, grid_h, grid_w, wts, heads):
    """`VitMerger.__call__`: x is [grid_h * grid_w, d] in raster order."""
    d = x.shape[-1]
    mh, mw = grid_h // 2, grid_w // 2
    windows = x.reshape(grid_h, grid_w, d)
    windows = windows.reshape(mh, 2, mw, 2, d).transpose(0, 2, 1, 3, 4)
    windows = windows.reshape(mh * mw, 4, d)
    p = "vit_merger."
    normed = layer_norm(windows, wts[p + "layer_norm1.weight"], wts[p + "layer_norm1.bias"])
    q = normed @ wts[p + "self_attn.q_proj.weight"].T + wts[p + "self_attn.q_proj.bias"]
    k = normed @ wts[p + "self_attn.k_proj.weight"].T + wts[p + "self_attn.k_proj.bias"]
    v = normed @ wts[p + "self_attn.v_proj.weight"].T + wts[p + "self_attn.v_proj.bias"]
    a = attention(q, k, v, heads)
    a = a @ wts[p + "self_attn.out_proj.weight"].T + wts[p + "self_attn.out_proj.bias"]
    windows = windows + a
    residual = windows.mean(axis=1)
    merged = windows.reshape(mh * mw, 4 * d)
    merged = layer_norm(merged, wts[p + "pre_norm.weight"], wts[p + "pre_norm.bias"])
    merged = gelu_tanh(merged @ wts[p + "linear_1.weight"].T + wts[p + "linear_1.bias"])
    merged = merged @ wts[p + "linear_2.weight"].T + wts[p + "linear_2.bias"]
    return merged + residual, mh, mw


def final_merger(x, grid_h, grid_w, wts):
    """`Merger.__call__` with merger_times == 1."""
    d = x.shape[-1]
    mh, mw = grid_h // 2, grid_w // 2
    h = x.reshape(grid_h, grid_w, d).reshape(mh, 2, mw, 2, d).transpose(0, 2, 1, 3, 4)
    h = h.reshape(mh * mw, 4 * d)
    p = "merger.mlp.0."
    h = layer_norm(h, wts[p + "pre_norm.weight"], wts[p + "pre_norm.bias"])
    h = gelu_tanh(h @ wts[p + "linear_1.weight"].T + wts[p + "linear_1.bias"])
    return h @ wts[p + "linear_2.weight"].T + wts[p + "linear_2.bias"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rows", required=True, help="f32 [N, 588], raster, (P_h, P_w, C) rows")
    ap.add_argument("--grid", type=int, nargs=2, required=True, help="patch rows, patch cols")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.load(open(model_dir / "config.json"))["vision_config"]
    hidden, heads, depth = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    side = cfg["image_size"] // cfg["patch_size"]
    gh, gw = args.grid
    n = gh * gw

    index = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
    wts = {}
    for shard in sorted({f for k, f in index.items() if k.startswith(("vision_tower.", "vit_merger.", "merger."))}):
        wts.update(read_safetensors(model_dir / shard, ("vision_tower.", "vit_merger.", "merger.")))

    rows = np.fromfile(args.rows, dtype=np.float32).reshape(n, -1)
    pw = wts["vision_tower.embeddings.patch_embedding.weight"].reshape(hidden, -1)
    assert rows.shape[1] == pw.shape[1], (rows.shape, pw.shape)
    x = rows.astype(np.float16).astype(np.float32) @ pw.T
    x = x + wts["vision_tower.embeddings.patch_embedding.bias"]
    bh, bw = position_buckets(gh, side), position_buckets(gw, side)
    ids = (bh[:, None] * side + bw[None, :]).reshape(-1)
    x = x + wts["vision_tower.embeddings.position_embedding.weight"][ids]

    stages = {"patch_embed": x.copy()}
    cur_h, cur_w = gh, gw
    for i in range(depth):
        x = siglip_block(x, wts, f"vision_tower.encoder.layers.{i}.", heads)
        if i == 0:
            stages["block_0"] = x.copy()
        if i == INSERT_LAYER:
            x, cur_h, cur_w = vit_merger(x, cur_h, cur_w, wts, heads)
            stages["vit_merger"] = x.copy()
        print(f"block {i} absmax {np.abs(x).max():.2f}", flush=True)
    stages["last_block"] = x.copy()
    x = layer_norm(x, wts["vision_tower.post_layernorm.weight"], wts["vision_tower.post_layernorm.bias"])
    stages["output"] = final_merger(x, cur_h, cur_w, wts)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = {"grid": [gh, gw], "stages": {}}
    for name, a in stages.items():
        a = np.ascontiguousarray(a.astype(np.float32))
        (out / f"{name}.bin").write_bytes(a.tobytes())
        header["stages"][name] = {"shape": list(a.shape)}
        print(f"{name:12s} shape={a.shape} absmax={np.abs(a).max():.4f}")
    (out / "header.json").write_text(json.dumps(header, indent=2))


if __name__ == "__main__":
    main()
