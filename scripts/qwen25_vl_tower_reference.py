#!/usr/bin/env python3
"""A numpy transcription of the Qwen2.5-VL vision tower, for the parity test
`crates/runtime/src/vision/qwen25.rs` (`the_real_tower_matches_the_numpy_reference`).

**What this is and is not.** It is written from mlx-vlm's
`models/qwen2_5_vl/vision.py` (read, never executed), so a convention both it
and the Rust tower get wrong would pass. It is NOT mlx-vlm run on the same
bytes. The two things that make it worth running anyway: it shares no code
with the Rust tower, and it is written in the reference's own structure
(unrolled `get_window_index`, `rot_pos_emb`, `apply_rotary_pos_emb_vision`,
the `cu_seqlens` segments), so the window permutation, the rope row layout and
the merger order are each transcribed rather than derived.

Pure numpy and the standard library: no safetensors package, no PIL, no mlx.
Patch rows arrive from the Rust side (so preprocessing is not in question),
in this port's (T, P_h, P_w, C) inner order, merge-window row order.

Usage:
    python3 -I scripts/qwen25_vl_tower_reference.py \
      --model-dir /Volumes/T7/hf-models/Qwen2.5-VL-3B-Instruct-4bit \
      --rows rows.bin --grid 1 32 32 --out-dir dump/
Writes {header.json, <stage>.bin} float32 stages: patch_embed (window order),
block_0, block_<last> (window order), merger (the page's own order).
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

EPS = 1e-6
THETA = 10000.0


def read_safetensors(path, wanted):
    """Header + raw bytes, F16 only (the tower ships unquantized F16)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__" or not name.startswith(wanted):
                continue
            if meta["dtype"] != "F16":
                raise ValueError(f"{name} is {meta['dtype']}; F16 only")
            s, e = meta["data_offsets"]
            f.seek(base + s)
            out[name] = np.frombuffer(f.read(e - s), dtype=np.float16).reshape(meta["shape"])
    return out


def rms_norm(x, w):
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w.astype(np.float32)


def erf_gelu(x):
    erf = np.frompyfunc(math.erf, 1, 1)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)).astype(np.float32))


def silu(x):
    return x / (1.0 + np.exp(-x))


def rot_pos_emb(grid, head_dim, merge):
    """Reference `rot_pos_emb`: [seq, head_dim/2] angles, h row then w row."""
    t, h, w = grid
    hp = np.repeat(np.arange(h)[:, None], w, axis=1)
    hp = hp.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
    wp = np.repeat(np.arange(w)[None, :], h, axis=0)
    wp = wp.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
    pos = np.tile(np.stack([hp, wp], axis=-1), (t, 1))  # [seq, 2]
    dim = head_dim // 2
    inv_freq = 1.0 / (THETA ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    full = np.outer(np.arange(max(h, w), dtype=np.float32), inv_freq)  # [max, dim/2]
    return full[pos].reshape(pos.shape[0], -1)  # [seq, dim]


def window_index(grid, merge, window, patch):
    """Reference `get_window_index` for one image."""
    t, h, w = grid
    unit = merge * merge
    size = window // merge // patch
    lh, lw = h // merge, w // merge
    index = np.arange(t * lh * lw).reshape(t, lh, lw)
    pad_h = size - lh % size
    pad_w = size - lw % size
    nwh, nww = (lh + pad_h) // size, (lw + pad_w) // size
    padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
    padded = padded.reshape(t, nwh, size, nww, size).transpose(0, 1, 3, 2, 4)
    padded = padded.reshape(t, nwh * nww, size, size)
    seqlens = (padded != -100).sum(axis=(2, 3)).reshape(-1)
    flat = padded.reshape(-1)
    order = flat[flat != -100]
    cu = np.concatenate([[0], np.cumsum(seqlens) * unit])
    keep = [0] + [i for i in range(1, len(cu)) if cu[i] != cu[i - 1]]
    return order, cu[keep]


def apply_rope(x, cos, sin):
    """`apply_rotary_pos_emb_vision`: tile cos/sin x2 over the head, rotate_half."""
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


def attention(q, k, v, cu):
    """Bidirectional attention inside each [cu[i], cu[i+1]) segment."""
    scale = q.shape[-1] ** -0.5
    out = np.empty_like(q)
    for a, b in zip(cu[:-1], cu[1:]):
        qs, ks, vs = q[a:b], k[a:b], v[a:b]  # [n, heads, d]
        scores = np.einsum("qhd,khd->hqk", qs, ks) * scale
        scores -= scores.max(-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(-1, keepdims=True)
        out[a:b] = np.einsum("hqk,khd->qhd", p, vs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rows", required=True, help="f32 [seq, patch_dim], (T,P,P,C) rows")
    ap.add_argument("--grid", type=int, nargs=3, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.load(open(model_dir / "config.json"))["vision_config"]
    index = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
    shard = sorted({f for n, f in index.items() if n.startswith("vision_tower.")})
    assert len(shard) == 1, shard
    wts = read_safetensors(model_dir / shard[0], "vision_tower.")

    hidden, heads = cfg["hidden_size"], cfg["num_heads"]
    head_dim = hidden // heads
    merge, patch = cfg["spatial_merge_size"], cfg["patch_size"]
    window, depth = cfg["window_size"], cfg["depth"]
    full_blocks = set(cfg["fullatt_block_indexes"])
    grid = tuple(args.grid)
    seq = grid[0] * grid[1] * grid[2]
    unit = merge * merge

    rows = np.fromfile(args.rows, dtype=np.float32).reshape(seq, -1)
    w_pe = wts["vision_tower.patch_embed.proj.weight"].astype(np.float32).reshape(hidden, -1)
    assert rows.shape[1] == w_pe.shape[1], (rows.shape, w_pe.shape)
    # The reference computes in the weights' dtype (F16); round the rows the
    # way an FP16 pipeline sees them, then accumulate in F32.
    x = rows.astype(np.float16).astype(np.float32) @ w_pe.T

    order, cu_window = window_index(grid, merge, window, patch)
    freqs = rot_pos_emb(grid, head_dim, merge)
    x = x.reshape(seq // unit, unit, -1)[order].reshape(seq, -1)
    freqs = freqs.reshape(seq // unit, unit, -1)[order].reshape(seq, -1)
    cos = np.tile(np.cos(freqs), (1, 2))[:, None, :]
    sin = np.tile(np.sin(freqs), (1, 2))[:, None, :]
    cu_full = np.array([0, seq])

    stages = {"patch_embed": x.copy()}

    def f32(name):
        return wts[f"vision_tower.{name}"].astype(np.float32)

    for i in range(depth):
        p = f"blocks.{i}."
        h = rms_norm(x, f32(p + "norm1.weight"))
        qkv = h @ f32(p + "attn.qkv.weight").T + f32(p + "attn.qkv.bias")
        qkv = qkv.reshape(seq, 3, heads, head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = attention(q, k, v, cu_full if i in full_blocks else cu_window)
        x = x + a.reshape(seq, -1) @ f32(p + "attn.proj.weight").T + f32(p + "attn.proj.bias")
        h = rms_norm(x, f32(p + "norm2.weight"))
        gate = silu(h @ f32(p + "mlp.gate_proj.weight").T + f32(p + "mlp.gate_proj.bias"))
        up = h @ f32(p + "mlp.up_proj.weight").T + f32(p + "mlp.up_proj.bias")
        x = x + (gate * up) @ f32(p + "mlp.down_proj.weight").T + f32(p + "mlp.down_proj.bias")
        if i == 0:
            stages["block_0"] = x.copy()
        if i == depth - 1:
            stages[f"block_{depth - 1}"] = x.copy()
        print(f"block {i} absmax {np.abs(x).max():.2f}", flush=True)

    h = rms_norm(x, f32("merger.ln_q.weight")).reshape(-1, hidden * unit)
    h = erf_gelu(h @ f32("merger.mlp.0.weight").T + f32("merger.mlp.0.bias"))
    h = h @ f32("merger.mlp.2.weight").T + f32("merger.mlp.2.bias")
    stages["merger"] = h[np.argsort(order)]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = {"grid_thw": list(grid), "stages": {}}
    for name, a in stages.items():
        a = np.ascontiguousarray(a.astype(np.float32))
        (out / f"{name}.bin").write_bytes(a.tobytes())
        header["stages"][name] = {"shape": list(a.shape)}
        print(f"{name:12s} shape={a.shape} absmax={np.abs(a).max():.4f}")
    (out / "header.json").write_text(json.dumps(header, indent=2))


if __name__ == "__main__":
    main()
