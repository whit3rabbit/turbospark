#!/usr/bin/env python3
"""A numpy transcription of the Muse-Glimmer vision tower, adapter and
projection, for the parity test in `crates/runtime/src/vision/muse.rs`
(`the_real_tower_matches_the_numpy_reference`).

**What this is and is not.** It is written from mlx-vlm's
`models/muse_glimmer/{vision,muse_glimmer}.py` (read, never executed), so a
convention both it and the Rust tower get wrong would pass. It is NOT mlx-vlm
run on the same bytes. What it does not share with the Rust code is any code,
and it deliberately keeps the reference's structure where the port changed it:
q, k and v stay THREE projections here (the port fuses them at repack), the
window index is the reference's padded `_window_index`, the position table is
the reference's `_position_embeddings`, and the merge is the reference's
`reshape / transpose(0, 1, 3, 5, 2, 4)`.

Pure numpy and the standard library. Patch rows arrive from the Rust side
(raster order, `(T, C, P, P)` inside a row), so preprocessing is not in
question here.

Usage:
    python3 -I scripts/muse_glimmer_tower_reference.py \
      --model-dir /Volumes/T7/hf-models/Muse-Glimmer-30B-4bit \
      --rows rows.bin --grid 1 40 50 --out-dir dump/
Writes {header.json, <stage>.bin} float32: patch_embed (after ln_pre, window
order), block_0, block_49 (window order), merger (the final rows, merged
raster order).
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

LN_EPS = 1e-5
RMS_EPS = 1e-5
THETA = 10000.0
WINDOW = 32
GROUP = 64
BITS = 4


class Shards:
    """Tensors out of the safetensors shards that hold the vision side."""

    def __init__(self, model_dir):
        self.index = {}
        index = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
        files = sorted({f for n, f in index.items() if n.startswith("vision")})
        for f in files:
            path = model_dir / f
            with open(path, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
            for name, meta in header.items():
                if name.startswith("vision"):
                    self.index[name] = (path, 8 + n, meta)

    def raw(self, name):
        path, base, meta = self.index[name]
        s, e = meta["data_offsets"]
        with open(path, "rb") as fh:
            fh.seek(base + s)
            return fh.read(e - s), meta

    def f32(self, name):
        data, meta = self.raw(name)
        if meta["dtype"] == "BF16":
            bits = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
            arr = bits.view(np.float32)
        elif meta["dtype"] == "F16":
            arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
        else:
            raise ValueError(f"{name}: {meta['dtype']}")
        # The install narrows BF16 to FP16 (exact in the normal range).
        return arr.astype(np.float16).astype(np.float32).reshape(meta["shape"])

    def dequant(self, base):
        """MLX affine 4-bit, group 64: w = q * scale + bias per group."""
        data, meta = self.raw(base + ".weight")
        rows, words = meta["shape"]
        packed = np.frombuffer(data, dtype=np.uint32).reshape(rows, words)
        shifts = (BITS * np.arange(32 // BITS)).astype(np.uint32)
        q = ((packed[:, :, None] >> shifts) & ((1 << BITS) - 1)).reshape(rows, -1)
        cols = q.shape[1]
        groups = cols // GROUP

        def bf16(name):
            d, m = self.raw(name)
            bits = np.frombuffer(d, dtype=np.uint16).astype(np.uint32) << 16
            return bits.view(np.float32).reshape(m["shape"])

        scales, biases = bf16(base + ".scales"), bf16(base + ".biases")
        w = q.reshape(rows, groups, GROUP).astype(np.float32) * scales[:, :, None] + biases[:, :, None]
        return w.reshape(rows, cols).astype(np.float16).astype(np.float32)


_ERF_X = np.linspace(-6.0, 6.0, 1_200_001)
_ERF_Y = np.array([math.erf(v) for v in _ERF_X])


def erf(x):
    """erf to ~1e-11 by interpolating a fine table (the standard library has
    no vectorized erf and the tower evaluates it on ~10^8 values)."""
    return np.interp(x, _ERF_X, _ERF_Y).astype(np.float32)


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.float32(math.sqrt(2.0))))


def layer_norm(x, w, b):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + LN_EPS) * w + b


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def window_index(grid, ws):
    """Reference `_window_index`: (patch order, non-empty segment starts)."""
    indices, cumulative, offset = [], [0], 0
    for t, h, w in grid:
        index = np.arange(t * h * w).reshape(t, h, w)
        pad_h, pad_w = (-h) % ws, (-w) % ws
        n_h, n_w = (h + pad_h) // ws, (w + pad_w) // ws
        padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        padded = padded.reshape(t, n_h, ws, n_w, ws).transpose(0, 1, 3, 2, 4)
        lengths = (padded != -100).sum(axis=(3, 4)).reshape(-1)
        flat = padded.reshape(-1)
        indices.append(flat[flat != -100] + offset)
        for length in lengths:
            cumulative.append(cumulative[-1] + int(length))
        offset += t * h * w
    cumulative = np.asarray(cumulative, dtype=np.int64)
    keep = np.concatenate(([True], cumulative[1:] != cumulative[:-1]))
    return np.concatenate(indices), cumulative[keep].tolist()


def position_embeddings(table, grid, side):
    """Reference `_position_embeddings` (half-pixel centres, validity masks)."""
    all_idx = [[] for _ in range(4)]
    all_w = [[] for _ in range(4)]
    for t, h, w in grid:
        h_grid = (np.arange(h, dtype=np.float32) + 0.5) * (side / h) - 0.5
        w_grid = (np.arange(w, dtype=np.float32) + 0.5) * (side / w) - 0.5
        h0, w0 = np.floor(h_grid).astype(np.int32), np.floor(w_grid).astype(np.int32)
        h1, w1 = h0 + 1, w0 + 1
        dh, dw = h_grid - h0, w_grid - w0
        vh0, vh1 = (h0 >= 0) & (h0 < side), (h1 >= 0) & (h1 < side)
        vw0, vw1 = (w0 >= 0) & (w0 < side), (w1 >= 0) & (w1 < side)
        h0, h1 = np.clip(h0, 0, side - 1), np.clip(h1, 0, side - 1)
        w0, w1 = np.clip(w0, 0, side - 1), np.clip(w1, 0, side - 1)
        idx = [
            (h0[:, None] * side + w0[None, :]).reshape(-1),
            (h0[:, None] * side + w1[None, :]).reshape(-1),
            (h1[:, None] * side + w0[None, :]).reshape(-1),
            (h1[:, None] * side + w1[None, :]).reshape(-1),
        ]
        wts = [
            ((1 - dh)[:, None] * (1 - dw)[None, :] * (vh0[:, None] & vw0[None, :])).reshape(-1),
            ((1 - dh)[:, None] * dw[None, :] * (vh0[:, None] & vw1[None, :])).reshape(-1),
            (dh[:, None] * (1 - dw)[None, :] * (vh1[:, None] & vw0[None, :])).reshape(-1),
            (dh[:, None] * dw[None, :] * (vh1[:, None] & vw1[None, :])).reshape(-1),
        ]
        for i in range(4):
            all_idx[i].append(np.tile(idx[i], t))
            all_w[i].append(np.tile(wts[i], t))
    indices = np.stack([np.concatenate(p) for p in all_idx])
    weights = np.stack([np.concatenate(p) for p in all_w]).astype(np.float32)
    return (table[indices] * weights[..., None]).sum(axis=0)


def position_ids(grid):
    out = []
    for t, h, w in grid:
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = np.stack((cols.reshape(-1), rows.reshape(-1)), axis=-1) + 1
        out.append(np.tile(coords, (t, 1)))
    return np.concatenate(out)


def rotary(pos, head_dim):
    spatial = head_dim // 2
    inv = 1.0 / (THETA ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial))
    fw = pos[:, 0].astype(np.float32)[:, None] * inv[None, :]
    fh = pos[:, 1].astype(np.float32)[:, None] * inv[None, :]
    freqs = np.concatenate((fw, fh, fw, fh), axis=-1)
    return np.cos(freqs)[:, None, :], np.sin(freqs)[:, None, :]


def attention(q, k, v, split_points):
    """Bidirectional attention inside each segment, one head at a time."""
    seq, heads, dim = q.shape
    scale = dim ** -0.5
    out = np.empty_like(q)
    bounds = [0] + list(split_points) + [seq]
    for a, b in zip(bounds[:-1], bounds[1:]):
        for h in range(heads):
            s = (q[a:b, h] @ k[a:b, h].T) * scale
            s -= s.max(-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(-1, keepdims=True)
            out[a:b, h] = p @ v[a:b, h]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rows", required=True, help="f32 [seq, 1176], raster, (T,C,P,P) rows")
    ap.add_argument("--grid", type=int, nargs=3, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.load(open(model_dir / "config.json"))["vision_config"]
    depth, hidden, heads = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"]
    head_dim = hidden // heads
    merge, side = cfg["merge_size"], cfg["pos_emb_height"]
    full = lambda i: (i + 1) % 4 == 0 or i == depth - 1  # noqa: E731
    grid = [tuple(args.grid)]
    t, gh, gw = grid[0]
    seq = t * gh * gw

    sh = Shards(model_dir)
    p = "vision_tower."
    rows = np.fromfile(args.rows, dtype=np.float32).reshape(seq, -1)
    w_pe = sh.f32(p + "patch_embedder.patch_embedding.weight")
    assert rows.shape[1] == w_pe.shape[1], (rows.shape, w_pe.shape)

    x = rows.astype(np.float16).astype(np.float32) @ w_pe.T
    x = x + position_embeddings(
        sh.f32(p + "patch_embedder.position_embedding_table.weight"), grid, side
    ).astype(np.float16).astype(np.float32)
    x = layer_norm(x, sh.f32(p + "ln_pre.weight"), sh.f32(p + "ln_pre.bias"))

    order, cu = window_index(grid, WINDOW)
    identity = np.array_equal(order, np.arange(order.size))
    pos = position_ids(grid)
    if not identity:
        x, pos = x[order], pos[order]
    cos, sin = rotary(pos, head_dim)
    full_split = []
    window_split = cu[1:-1]
    print(f"seq {seq}, windows {len(cu) - 1}, identity {identity}", flush=True)

    stages = {"patch_embed": x.copy()}
    for i in range(depth):
        b = f"{p}layers.{i}."
        h = layer_norm(x, sh.f32(b + "norm1.weight"), sh.f32(b + "norm1.bias"))

        def lin(name):
            return h @ sh.f32(b + f"attn.{name}_proj.weight").T + sh.f32(b + f"attn.{name}_proj.bias")

        q = lin("q").reshape(seq, heads, head_dim)
        k = lin("k").reshape(seq, heads, head_dim)
        v = lin("v").reshape(seq, heads, head_dim)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        a = attention(q, k, v, full_split if full(i) else window_split)
        x = x + a.reshape(seq, -1) @ sh.f32(b + "attn.proj.weight").T + sh.f32(b + "attn.proj.bias")
        h = layer_norm(x, sh.f32(b + "norm2.weight"), sh.f32(b + "norm2.bias"))
        h = gelu(h @ sh.f32(b + "mlp.fc1.weight").T + sh.f32(b + "mlp.fc1.bias"))
        x = x + h @ sh.f32(b + "mlp.fc2.weight").T + sh.f32(b + "mlp.fc2.bias")
        if i == 0:
            stages["block_0"] = x.copy()
        if i == depth - 1:
            stages[f"block_{depth - 1}"] = x.copy()
        print(f"block {i} absmax {np.abs(x).max():.2f}", flush=True)

    if not identity:
        x = x[np.argsort(order)]
    x = layer_norm(x, sh.f32(p + "ln_post.weight"), sh.f32(p + "ln_post.bias"))

    # `_pixel_shuffle`: (t, h/m, m, w/m, m, dim) transposed (0, 1, 3, 5, 2, 4).
    y = x.reshape(t, gh // merge, merge, gw // merge, merge, hidden)
    y = y.transpose(0, 1, 3, 5, 2, 4).reshape(-1, hidden * merge * merge)

    y = gelu(gelu(y @ sh.dequant("vision_adapter.fc1").T) @ sh.dequant("vision_adapter.fc2").T)
    y = y @ sh.dequant("vision_projection").T
    y = y / np.sqrt((y * y).mean(-1, keepdims=True) + RMS_EPS)
    stages["merger"] = y

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = {"grid_thw": list(args.grid), "stages": {}}
    for name, a in stages.items():
        a = np.ascontiguousarray(a.astype(np.float32))
        (out / f"{name}.bin").write_bytes(a.tobytes())
        header["stages"][name] = {"shape": list(a.shape)}
        print(f"{name:12s} shape={a.shape} absmax={np.abs(a).max():.4f}")
    (out / "header.json").write_text(json.dumps(header, indent=2))


if __name__ == "__main__":
    main()
