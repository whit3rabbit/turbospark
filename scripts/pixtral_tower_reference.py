#!/usr/bin/env python3
"""A numpy transcription of the Pixtral vision tower and projector, for the
parity test `crates/runtime/src/vision/pixtral.rs`
(`the_real_tower_matches_the_numpy_reference`).

**What this is and is not.** It is written from mlx-vlm's
`models/pixtral/vision.py` and `pixtral.py` (read, never executed), so a
convention both it and the Rust tower get wrong would pass. It is NOT mlx-vlm
run on the same bytes. What it does not share with the Rust tower is any code,
and it keeps the reference's own structure rather than this port's: the rope
table is built exactly as `PixtralRotaryEmbedding` builds it (a
`[max_patches^2, dim/2]` table indexed by `position_ids_in_meshgrid`'s
`row * max_width + col`), the patch convolution is a convolution rather than a
GEMM over flattened rows, and attention applies the block-diagonal mask.

Pure numpy and the standard library: no safetensors package, no PIL, no mlx.
Patch rows arrive from the Rust side (so preprocessing is not in question),
row-major over the grid, each row `(C, P_h, P_w)`.

Usage:
    python3 -I scripts/pixtral_tower_reference.py \
      --model-dir /Volumes/T7/hf-models/pixtral-12b-bf16 \
      --rows rows.bin --grid 28 28 --out-dir dump/
Writes {header.json, <stage>.bin} float32 stages: ln_pre (the reference's
hidden_states[0]), block_0, block_<last> (hidden_states[-1], what
vision_feature_layer -1 selects) and projector (the rows the trunk receives).
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

EPS = 1e-5


def read_safetensors(path, wanted):
    """Header + raw bytes; F16 and BF16 (read as the top half of an f32)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__" or not name.startswith(wanted):
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if meta["dtype"] == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif meta["dtype"] == "BF16":
                bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = bits.view(np.float32)
            else:
                raise ValueError(f"{name} is {meta['dtype']}; F16 and BF16 only")
            out[name] = arr.reshape(meta["shape"])
    return out


def to_fp16(x):
    """The engine stores weights FP16; round the reference's the same way so
    the comparison measures the compute and not the storage."""
    return x.astype(np.float16).astype(np.float32)


def rms_norm(x, w):
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def erf_gelu(x):
    erf = np.frompyfunc(math.erf, 1, 1)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)).astype(np.float32))


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def rotary_table(head_dim, theta, max_patches_per_side):
    """`PixtralRotaryEmbedding.__init__`, literally."""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
    h = np.arange(max_patches_per_side)
    w = np.arange(max_patches_per_side)
    freqs_h = np.outer(h, freqs[::2]).astype(np.float32)
    freqs_w = np.outer(w, freqs[1::2]).astype(np.float32)
    inv_freq = np.concatenate(
        [
            np.tile(freqs_h[:, None, :], (1, max_patches_per_side, 1)),
            np.tile(freqs_w[None, :, :], (max_patches_per_side, 1, 1)),
        ],
        axis=-1,
    ).reshape(-1, head_dim // 2)
    return np.concatenate((inv_freq, inv_freq), axis=-1)


def position_ids_in_meshgrid(height, width, max_width):
    h_grid, v_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return (h_grid.reshape(-1) * max_width + v_grid.reshape(-1)).astype(np.int64)


def block_mask(lengths, seq):
    """`generate_block_attention_mask`: 0 inside an image's block, -1e9 outside."""
    mask = np.full((seq, seq), -1e9, dtype=np.float32)
    start = 0
    for n in lengths:
        mask[start : start + n, start : start + n] = 0
        start += n
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rows", required=True, help="f32 [seq, 768], (C,P,P) rows")
    ap.add_argument("--grid", type=int, nargs=2, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.load(open(model_dir / "config.json"))
    vc = cfg["vision_config"]
    hidden = vc.get("hidden_size", 1024)
    depth = vc.get("num_hidden_layers", 24)
    heads = vc.get("num_attention_heads", 16)
    head_dim = vc.get("head_dim", hidden // heads)
    patch = vc.get("patch_size", 16)
    theta = vc.get("rope_theta", 10000.0)
    max_width = vc.get("image_size", 1024) // patch
    assert hidden == heads * head_dim

    index = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
    shards = sorted(
        {
            f
            for n, f in index.items()
            if n.startswith(("vision_tower.", "multi_modal_projector."))
        }
    )
    wts = {}
    for shard in shards:
        wts.update(read_safetensors(model_dir / shard, ("vision_tower.", "multi_modal_projector.")))

    def w(name):
        return to_fp16(wts[name])

    tower = "vision_tower.vision_model."
    if f"{tower}patch_conv.weight" not in wts:
        tower = "vision_tower."
    gh, gw = args.grid
    seq = gh * gw
    rows = np.fromfile(args.rows, dtype=np.float32).reshape(seq, -1)
    assert rows.shape[1] == 3 * patch * patch, rows.shape

    # The patch convolution, as a convolution: reshape each (C, P, P) row to an
    # NHWC patch and contract with the [out, kh, kw, C] kernel.
    conv = w(f"{tower}patch_conv.weight")  # MLX layout [out, kh, kw, C]
    if conv.shape[1] == 3:  # HF layout [out, C, kh, kw]
        conv = conv.transpose(0, 2, 3, 1)
    px = rows.astype(np.float16).astype(np.float32).reshape(seq, 3, patch, patch)
    px = px.transpose(0, 2, 3, 1)  # [seq, kh, kw, C]
    x = np.einsum("nhwc,ohwc->no", px, conv)

    x = rms_norm(x, w(f"{tower}ln_pre.weight"))
    stages = {"ln_pre": x.copy()}

    table = rotary_table(head_dim, theta, max_width)
    ids = position_ids_in_meshgrid(gh, gw, max_width)
    angles = table[ids]  # [seq, head_dim]
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    mask = block_mask([seq], seq)
    scale = head_dim**-0.5

    for i in range(depth):
        p = f"{tower}transformer.layers.{i}."
        h = rms_norm(x, w(p + "attention_norm.weight"))
        q = (h @ w(p + "attention.q_proj.weight").T).reshape(seq, heads, head_dim)
        k = (h @ w(p + "attention.k_proj.weight").T).reshape(seq, heads, head_dim)
        v = (h @ w(p + "attention.v_proj.weight").T).reshape(seq, heads, head_dim)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        scores = np.einsum("qhd,khd->hqk", q, k) * scale + mask[None]
        scores -= scores.max(-1, keepdims=True)
        pr = np.exp(scores)
        pr /= pr.sum(-1, keepdims=True)
        a = np.einsum("hqk,khd->qhd", pr, v).reshape(seq, -1)
        x = x + a @ w(p + "attention.o_proj.weight").T
        h = rms_norm(x, w(p + "ffn_norm.weight"))
        gate = silu(h @ w(p + "feed_forward.gate_proj.weight").T)
        up = h @ w(p + "feed_forward.up_proj.weight").T
        x = x + (gate * up) @ w(p + "feed_forward.down_proj.weight").T
        if i == 0:
            stages["block_0"] = x.copy()
        if i == depth - 1:
            stages[f"block_{depth - 1}"] = x.copy()
        print(f"block {i} absmax {np.abs(x).max():.2f}", flush=True)

    m = "multi_modal_projector."
    y = x @ w(m + "linear_1.weight").T + w(m + "linear_1.bias")
    y = erf_gelu(y)
    y = y @ w(m + "linear_2.weight").T + w(m + "linear_2.bias")
    stages["projector"] = y

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
