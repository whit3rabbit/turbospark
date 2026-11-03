#!/usr/bin/env python3
"""A numpy transcription of the LLaVA-OneVision vision path, for the parity test
`crates/runtime/tests/onevision_vision_parity.rs`.

**What this is and is not.** It is written from mlx-vlm's
`models/llava_onevision/{llava_onevision,vision}.py` and Hugging Face's
`LlavaOnevisionForConditionalGeneration.pack_image_features` (read, never
executed), so a convention both it and the Rust code get wrong would pass. It
is NOT mlx-vlm or transformers run on the same bytes. It shares no code with
the Rust port and is written in the reference's own structure (the conv patch
embed as a GEMM, per-tile attention, `unpad_image`, `interpolate`, the newline
column), so the anyres row order and the newline placement are each
transcribed rather than derived.

It does NOT check preprocessing: the patch rows arrive from the Rust side (no
PIL in this environment). The resize/pad/tile pipeline is pinned by the
independent python token-count model in docs/LLAVA_ONEVISION_VISION_PHASE0.md
and by the unit tests in `crates/vision-io/src/onevision.rs`.

The weights are read from the pinned `llava-hf` revision by ranged HTTP reads
(only the 426 tower tensors, ~830 MB) and cached as an uncompressed .npz.

Usage:
    python3 -I scripts/llava_onevision_tower_reference.py \
      --cache-dir /Volumes/T7/hf-models/llavaov-tower --dump dump/
reads dump/{pixels.f32, meta.json}, writes dump/{embed,block_0,block_last,
projector,packed}.f32 and dump/ref_header.json.
"""

import argparse
import json
import math
import struct
import urllib.request
from pathlib import Path

import numpy as np

REPO = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
REV = "0d50680527681998e456c7b78950205bedd8a068"
SHARD = "model-00001-of-00004.safetensors"
EPS = 1e-6
HIDDEN, HEADS, DEPTH, SIDE = 1152, 16, 26, 27


def ranged(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            if attempt == 3:
                raise
    raise RuntimeError("unreachable")


def fetch_weights(cache_dir):
    """The tower tensors as float32 arrays, from the cache or the pinned shard."""
    cache = Path(cache_dir) / "tower.npz"
    if cache.exists():
        return dict(np.load(cache))
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{SHARD}"
    (n,) = struct.unpack("<Q", ranged(url, 0, 7))
    header = json.loads(ranged(url, 8, 8 + n - 1))
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__" or name.startswith("language_model."):
            continue
        if meta["dtype"] != "F16":
            raise ValueError(f"{name} is {meta['dtype']}; F16 only")
        s, e = meta["data_offsets"]
        raw = ranged(url, base + s, base + e - 1)
        out[name] = np.frombuffer(raw, dtype=np.float16).reshape(meta["shape"])
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def layer_norm(x, w, b):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def erf(x):
    """Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7, below the F16 noise floor)."""
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592) * t
    return sign * (1.0 - poly * np.exp(-ax * ax))


def gelu_erf(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def attention(q, k, v):
    """Bidirectional attention inside one tile. [n, heads, d]."""
    scale = q.shape[-1] ** -0.5
    scores = np.einsum("qhd,khd->hqk", q, k) * scale
    scores -= scores.max(-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", p, v)


def unpad_image(feat, orig_h, orig_w):
    """`unpad_image` on a (C, H, W) array."""
    _, cur_h, cur_w = feat.shape
    if orig_w / orig_h > cur_w / cur_h:
        new_h = int(round(orig_h * (cur_w / orig_w), 7))
        pad = (cur_h - new_h) // 2
        return feat[:, pad:cur_h - pad, :]
    new_w = int(round(orig_w * (cur_h / orig_h), 7))
    pad = (cur_w - new_w) // 2
    return feat[:, :, pad:cur_w - pad]


def bilinear(feat, out_h, out_w):
    """`F.interpolate(mode="bilinear")` of (C, H, W): align_corners=False."""
    c, h, w = feat.shape
    out = np.empty((c, out_h, out_w), dtype=np.float32)

    def src(i, scale, limit):
        s = max((i + 0.5) * scale - 0.5, 0.0)
        lo = min(int(math.floor(s)), limit - 1)
        return lo, min(lo + 1, limit - 1), s - lo

    for y in range(out_h):
        y0, y1, ly = src(y, h / out_h, h)
        for x in range(out_w):
            x0, x1, lx = src(x, w / out_w, w)
            out[:, y, x] = (1 - ly) * ((1 - lx) * feat[:, y0, x0] + lx * feat[:, y0, x1]) + ly * (
                (1 - lx) * feat[:, y1, x0] + lx * feat[:, y1, x1]
            )
    return out


def pack(feats, newline, orig_h, orig_w, nph, npw):
    """`pack_image_features` for one image: feats [(1 + nph*npw) * 729, C]."""
    c = feats.shape[-1]
    per = SIDE * SIDE
    base = feats[:per]
    grid = feats[per:].reshape(nph, npw, SIDE, SIDE, c)
    grid = grid.transpose(4, 0, 2, 1, 3).reshape(c, nph * SIDE, npw * SIDE)
    grid = unpad_image(grid, orig_h, orig_w)
    _, cur_h, cur_w = grid.shape
    ratio = math.sqrt(cur_h * cur_w / (9 * SIDE * SIDE))
    if ratio > 1.1:
        grid = bilinear(grid, int(cur_h // ratio), int(cur_w // ratio))
    nl = np.broadcast_to(newline[:, None, None], (c, grid.shape[1], 1))
    grid = np.concatenate([grid, nl], axis=-1)
    grid = grid.reshape(c, -1).T
    return np.concatenate([base, grid], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()
    dump = Path(args.dump)
    meta = json.load(open(dump / "meta.json"))
    nph, npw = meta["tiles_h"], meta["tiles_w"]
    tiles = 1 + nph * npw
    seq = tiles * SIDE * SIDE
    rows = np.fromfile(dump / "pixels.f32", dtype=np.float32).reshape(seq, -1)

    wts = fetch_weights(args.cache_dir)

    def f32(name):
        return wts[name].astype(np.float32)

    v = "vision_tower.vision_model."
    conv = f32(v + "embeddings.patch_embedding.weight").reshape(HIDDEN, -1)
    assert rows.shape[1] == conv.shape[1], (rows.shape, conv.shape)
    # The weights are F16; the rows reach an F16 pipeline rounded to F16.
    x = rows.astype(np.float16).astype(np.float32) @ conv.T + f32(v + "embeddings.patch_embedding.bias")
    x = x.reshape(tiles, SIDE * SIDE, HIDDEN) + f32(v + "embeddings.position_embedding.weight")[None]
    stages = {"embed": x.reshape(seq, HIDDEN).copy()}

    head_dim = HIDDEN // HEADS
    for i in range(DEPTH):
        p = f"{v}encoder.layers.{i}."
        h = layer_norm(x, f32(p + "layer_norm1.weight"), f32(p + "layer_norm1.bias"))
        q = (h @ f32(p + "self_attn.q_proj.weight").T + f32(p + "self_attn.q_proj.bias"))
        k = (h @ f32(p + "self_attn.k_proj.weight").T + f32(p + "self_attn.k_proj.bias"))
        vv = (h @ f32(p + "self_attn.v_proj.weight").T + f32(p + "self_attn.v_proj.bias"))
        a = np.stack(
            [
                attention(
                    q[t].reshape(-1, HEADS, head_dim),
                    k[t].reshape(-1, HEADS, head_dim),
                    vv[t].reshape(-1, HEADS, head_dim),
                ).reshape(-1, HIDDEN)
                for t in range(tiles)
            ]
        )
        x = x + a @ f32(p + "self_attn.out_proj.weight").T + f32(p + "self_attn.out_proj.bias")
        h = layer_norm(x, f32(p + "layer_norm2.weight"), f32(p + "layer_norm2.bias"))
        h = gelu_tanh(h @ f32(p + "mlp.fc1.weight").T + f32(p + "mlp.fc1.bias"))
        x = x + h @ f32(p + "mlp.fc2.weight").T + f32(p + "mlp.fc2.bias")
        if i == 0:
            stages["block_0"] = x.reshape(seq, HIDDEN).copy()
        print(f"block {i} absmax {np.abs(x).max():.2f}", flush=True)
    stages["block_last"] = x.reshape(seq, HIDDEN).copy()

    # hidden_states[-1] feeds the projector directly: no post_layernorm, no head.
    m = "multi_modal_projector."
    feats = x.reshape(seq, HIDDEN)
    feats = gelu_erf(feats @ f32(m + "linear_1.weight").T + f32(m + "linear_1.bias"))
    feats = feats @ f32(m + "linear_2.weight").T + f32(m + "linear_2.bias")
    stages["projector"] = feats
    stages["packed"] = pack(
        feats.astype(np.float32), f32("image_newline"), meta["orig_h"], meta["orig_w"], nph, npw
    )

    header = {"stages": {}}
    for name, a in stages.items():
        a = np.ascontiguousarray(a.astype(np.float32))
        (dump / f"ref_{name}.f32").write_bytes(a.tobytes())
        header["stages"][name] = {"shape": list(a.shape)}
        print(f"{name:12s} shape={a.shape} absmax={np.abs(a).max():.4f}")
    (dump / "ref_header.json").write_text(json.dumps(header, indent=2))


if __name__ == "__main__":
    main()
