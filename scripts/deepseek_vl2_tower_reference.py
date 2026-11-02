#!/usr/bin/env python3
"""A numpy transcription of DeepSeek-VL2's SigLIP tower and projector, for the
tower gate in `docs/DEEPSEEK_VL2_VISION_PHASE0.md`.

**What this is and is not.** It is written from mlx-vlm's
`models/deepseek_vl_v2/{vision,deepseek_vl_v2}.py` (read, never executed):
`VisionEmbeddings` (a 14x14 stride-14 conv over NHWC tiles, then the learned
`pos_embed`), `EncoderLayer` (LayerNorm eps 1e-6, fused-qkv attention with no
mask, a tanh-GELU MLP), the tower's final `nn.LayerNorm(width)` (MLX default
eps 1e-5), and `MlpProjector`'s `downsample_mlp_gelu` branch (zero-pad 27 to
28, 2x2 unfold channel-major, Linear, erf GELU, Linear). It shares no code with
the Rust engine, but a misreading of the reference would be shared by both. The
checkpoint's 4-bit planes are dequantized by the MLX rule (`w = q * scale +
bias` per 64-wide group, element `i` of a row at bit `4 * (i % 8)` of word
`i / 8`), so it reads the same bytes the engine's sidecar was built from. It is
NOT mlx-vlm run on those bytes.

The input is the engine's own patch rows (`<prefix>.pixels.f32`, written by
`crates/runtime/tests/deepseek_vl2_tower_real.rs`): the tower is compared on
identical pixels, and tiling and padding are pinned separately by the
`turbospark-vision-io` unit tests.

Pure numpy and the standard library.

Usage:
    python3 -I scripts/deepseek_vl2_tower_reference.py \
      --model-dir /Volumes/T7/hf-models/deepseek-vl2-tiny-4bit \
      --engine-prefix /Volumes/T7/vision-dumps/dsvl2-tower --tiles 2
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

GROUP = 64
BITS = 4
BLOCK_EPS = 1e-6
FINAL_EPS = 1e-5
HEADS = 16
PATCH_DIM = 14 * 14 * 3
PER_TILE = 27 * 27


class Checkpoint:
    def __init__(self, model_dir):
        path = Path(model_dir) / "model.safetensors"
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name):
        meta = self.header[name]
        s, e = meta["data_offsets"]
        dt = {"U32": "<u4", "F16": "<f2", "F32": "<f4"}[meta["dtype"]]
        arr = np.frombuffer(self.mm[self.base + s : self.base + e], dtype=dt)
        return arr.reshape(meta["shape"])

    def dense(self, name):
        return self.raw(name).astype(np.float32)

    def linear(self, base):
        """`(weight [out, in] f32, bias [out] f32)`, dequantizing a packed plane."""
        if base + ".scales" in self.header:
            w = self.raw(base + ".weight")
            s = self.raw(base + ".scales")
            b = self.raw(base + ".biases")
            shifts = np.arange(0, 32, BITS, dtype=np.uint32)
            q = ((w[..., None] >> shifts) & ((1 << BITS) - 1)).astype(np.float32)
            q = q.reshape(*w.shape[:-1], w.shape[-1] * (32 // BITS))
            weight = q * np.repeat(s.astype(np.float32), GROUP, axis=-1) + np.repeat(
                b.astype(np.float32), GROUP, axis=-1
            )
        else:
            weight = self.dense(base + ".weight")
        return weight, self.dense(base + ".bias")


def layer_norm(x, w, b, eps):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu_erf(x):
    # numpy has no erf; math.erf, vectorized, is exact where an
    # Abramowitz-Stegun fit would be too coarse for a parity bound.
    import math

    erf = np.vectorize(math.erf, otypes=[np.float64])(x / np.sqrt(2.0))
    return 0.5 * x * (1.0 + erf.astype(np.float32))


def attention(x, qkv_w, qkv_b, proj_w, proj_b):
    """One tile: x [seq, hidden]. No mask, no rope, scale head_dim^-0.5."""
    seq, hidden = x.shape
    hd = hidden // HEADS
    qkv = x @ qkv_w.T + qkv_b
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(seq, HEADS, hd).transpose(1, 0, 2)
    k = k.reshape(seq, HEADS, hd).transpose(1, 0, 2)
    v = v.reshape(seq, HEADS, hd).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) * (hd**-0.5)
    scores -= scores.max(-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(-1, keepdims=True)
    out = (p @ v).transpose(1, 0, 2).reshape(seq, hidden)
    return out @ proj_w.T + proj_b


def unfold(tile_rows):
    """[729, hidden] -> [196, hidden * 4], channel-major, after zero padding to 28."""
    hidden = tile_rows.shape[-1]
    x = tile_rows.reshape(27, 27, hidden)
    x = np.pad(x, [(0, 1), (0, 1), (0, 0)])
    x = x.transpose(2, 0, 1)  # C, H, W
    out = []
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            out.append(x[:, i : i + 2, j : j + 2].reshape(-1))
    return np.stack(out)


def compare(label, engine_bits, ref):
    eng = np.frombuffer(engine_bits, dtype="<f2").astype(np.float32).reshape(ref.shape)
    num = float((eng * ref).sum())
    den = float(np.linalg.norm(eng) * np.linalg.norm(ref)) + 1e-30
    cos = num / den
    diff = np.abs(eng - ref)
    rel = float(np.linalg.norm(eng - ref) / (np.linalg.norm(ref) + 1e-30))
    print(
        f"{label:12s} cos {cos:.6f}  rel_l2 {rel:.5f}  max|diff| {diff.max():.4f}  "
        f"max|ref| {np.abs(ref).max():.3f}"
    )
    return cos, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--engine-prefix", required=True)
    ap.add_argument("--tiles", type=int, default=2, help="leading tiles to compare")
    args = ap.parse_args()

    ck = Checkpoint(args.model_dir)
    prefix = Path(args.engine_prefix)
    pixels = np.fromfile(str(prefix) + ".pixels.f32", dtype="<f4")
    total_tiles = pixels.size // (PER_TILE * PATCH_DIM)
    n = min(args.tiles, total_tiles)
    pixels = pixels[: n * PER_TILE * PATCH_DIM].reshape(n, PER_TILE, PATCH_DIM)
    # The engine feeds the GEMM FP16 pixels; match that rounding.
    pixels = pixels.astype(np.float16).astype(np.float32)

    T ="vision.vision_tower."
    conv = ck.dense(T + "patch_embed.proj.weight").reshape(1152, PATCH_DIM)
    conv_b = ck.dense(T + "patch_embed.proj.bias")
    pos = ck.dense(T + "pos_embed").reshape(PER_TILE, 1152)
    x = pixels @ conv.T + conv_b + pos  # [n, 729, 1152]

    stages = {}
    stages["embed"] = x.copy()
    for blk in range(27):
        p = f"{T}blocks.{blk}."
        n1w, n1b = ck.dense(p + "norm1.weight"), ck.dense(p + "norm1.bias")
        n2w, n2b = ck.dense(p + "norm2.weight"), ck.dense(p + "norm2.bias")
        qkv_w, qkv_b = ck.linear(p + "attn.qkv")
        proj_w, proj_b = ck.linear(p + "attn.proj")
        fc1_w, fc1_b = ck.linear(p + "mlp.fc1")
        fc2_w, fc2_b = ck.linear(p + "mlp.fc2")
        for t in range(n):
            y = layer_norm(x[t], n1w, n1b, BLOCK_EPS)
            x[t] = x[t] + attention(y, qkv_w, qkv_b, proj_w, proj_b)
            y = layer_norm(x[t], n2w, n2b, BLOCK_EPS)
            x[t] = x[t] + gelu_tanh(y @ fc1_w.T + fc1_b) @ fc2_w.T + fc2_b
        if blk == 0:
            stages["block_first"] = x.copy()
        print(f"block {blk} done", flush=True) if blk % 9 == 8 else None
    stages["block_last"] = x.copy()
    nw, nb = ck.dense(T + "norm.weight"), ck.dense(T + "norm.bias")
    normed = np.stack([layer_norm(x[t], nw, nb, FINAL_EPS) for t in range(n)])
    stages["normed"] = normed

    w1, b1 = ck.linear("projector.layers.0")
    w2, b2 = ck.linear("projector.layers.2")
    proj = []
    for t in range(n):
        u = unfold(normed[t])
        proj.append(gelu_erf(u @ w1.T + b1) @ w2.T + b2)
    stages["projector"] = np.stack(proj)

    results = []
    for name in ["embed", "block_first", "block_last", "normed", "projector"]:
        data = Path(str(prefix) + "." + name + ".f16").read_bytes()
        ref = stages[name]
        per = 196 if name == "projector" else PER_TILE
        width = ref.shape[-1]
        keep = n * per * width * 2
        results.append((name, *compare(name, data[:keep], ref)))
    print(f"worst stage cosine {min(r[1] for r in results):.6f}")


if __name__ == "__main__":
    main()
