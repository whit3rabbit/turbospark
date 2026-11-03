#!/usr/bin/env python3
"""numpy transcription of the DeepSeek-OCR vision tower (SAM-ViT-B + CLIP-L +
linear projector + layout rows), the oracle for the real-tower parity test
`crates/runtime/tests/deepseek_ocr_tower_real.rs`.

This is a TRANSCRIPTION from reading `sam.py`, `vision.py` and `deepseekocr.py`
of mlx-vlm and the checkpoint's own `deepencoder.py`. It imports nothing from
mlx-vlm and runs no code from it (docs/DEEPSEEK_OCR_VISION_PHASE0.md).
Everything is float32 on the CPU, from the BF16 checkpoint bytes.

Usage:
    python3 -I scripts/deepseek_ocr_tower_reference.py MODEL.safetensors OUT_DIR [INPUT.npy]

INPUT.npy is the NORMALIZED image, float32, shape (1024, 1024, 3), HWC, RGB,
i.e. (pixel / 255 - 0.5) / 0.5. Without it a deterministic structured image is
generated and written to OUT_DIR/input.npy. Writes OUT_DIR/stages.npz and one
raw little-endian float32 file per stage (`sam_b0.f32`, ..., `rows.f32`) plus
`input.f32`, which `crates/runtime/tests/deepseek_ocr_tower_real.rs` reads.

The conventions this test exists to pin (each was mutation-checked on the
engine side, see the Phase 0 doc):
  - SAM window blocks zero-pad the 64x64 grid to 70x70 AFTER norm1, and the
    padded tokens are real keys (they carry the qkv bias), not masked.
  - The decomposed relative-position bias is built from the UNSCALED query and
    added AFTER the 1/sqrt(d) scaling of q.k.
  - CLIP-L takes the SAM output as its patch embeddings (its own 14x14 patch
    conv is unused), prepends the class row and adds position row 0..256.
  - CLIP LayerNorms use eps 1e-5, SAM LayerNorms and LayerNorm2d use 1e-6.
  - Concatenation order is [CLIP[1:], SAM], the projector is one linear.
"""
import json
import struct
import sys

import numpy as np


class Store:
    """Header-driven safetensors reader (BF16, U32, F16, F32)."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + n

    def get(self, name):
        meta = self.hdr[name]
        s, e = meta["data_offsets"]
        self.f.seek(self.base + s)
        raw = self.f.read(e - s)
        shape = tuple(meta["shape"])
        dt = meta["dtype"]
        if dt == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if dt == "F16":
            return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
        if dt == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        if dt == "U32":
            return np.frombuffer(raw, dtype="<u4").reshape(shape).copy()
        raise ValueError(dt)


def erf(x):
    # Numerical Recipes erfc Chebyshev fit, fractional error < 1.2e-7: well
    # under the float32 noise this reference is compared at.
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277))))))))
    r = 1.0 - t * np.exp(poly)
    return np.where(x >= 0, r, -r).astype(np.float32)


def gelu(x):
    return (0.5 * x * (1.0 + erf(x / np.float32(np.sqrt(2.0))))).astype(np.float32)


def quick_gelu(x):
    return (x / (1.0 + np.exp(-1.702 * x))).astype(np.float32)


def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps) * w + b).astype(np.float32)


def linear(x, w, b=None):
    y = x @ w.T
    return (y + b).astype(np.float32) if b is not None else y.astype(np.float32)


def conv2d(x, w, stride=1, pad=0):
    """NHWC conv, weight (O, kh, kw, I), no bias. x is (H, W, C)."""
    o, kh, kw, i = w.shape
    h, wd, _ = x.shape
    if pad:
        x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    oh = (x.shape[0] - kh) // stride + 1
    ow = (x.shape[1] - kw) // stride + 1
    cols = np.empty((oh, ow, kh * kw * i), dtype=np.float32)
    for a in range(kh):
        for b in range(kw):
            patch = x[a:a + stride * oh:stride, b:b + stride * ow:stride, :]
            cols[:, :, (a * kw + b) * i:(a * kw + b + 1) * i] = patch
    return (cols.reshape(oh * ow, -1) @ w.reshape(o, -1).T).reshape(oh, ow, o)


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def get_rel_pos(q_size, k_size, rel_pos):
    # Table length 2*size-1 always matches here (27 for a 14 window, 127 for
    # the 64x64 global blocks), so the reference's interpolation arm is dead.
    assert rel_pos.shape[0] == 2 * max(q_size, k_size) - 1
    q = np.arange(q_size)[:, None]
    k = np.arange(k_size)[None, :]
    return rel_pos[(q - k) + (k_size - 1)]


def window_partition(x, ws):
    h, w, c = x.shape
    ph, pw = (-h) % ws, (-w) % ws
    x = np.pad(x, ((0, ph), (0, pw), (0, 0)))
    hp, wp = h + ph, w + pw
    x = x.reshape(hp // ws, ws, wp // ws, ws, c).transpose(0, 2, 1, 3, 4)
    return x.reshape(-1, ws, ws, c), (hp, wp)


def window_unpartition(wins, ws, pad_hw, hw):
    hp, wp = pad_hw
    c = wins.shape[-1]
    x = wins.reshape(hp // ws, wp // ws, ws, ws, c).transpose(0, 2, 1, 3, 4)
    return x.reshape(hp, wp, c)[: hw[0], : hw[1]]


def sam_attention(x, st, pfx, heads=12):
    """x: (B, H, W, C) windows (or a 1-batch global grid)."""
    b, h, w, c = x.shape
    d = c // heads
    qkv = linear(x.reshape(b, h * w, c), st.get(pfx + "qkv.weight"), st.get(pfx + "qkv.bias"))
    qkv = qkv.reshape(b, h * w, 3, heads, d).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, HW, d)
    rh = get_rel_pos(h, h, st.get(pfx + "rel_pos_h"))  # (h, kh, d)
    rw = get_rel_pos(w, w, st.get(pfx + "rel_pos_w"))
    scale = np.float32(d ** -0.5)
    out = np.empty((b, heads, h * w, d), dtype=np.float32)
    for bi in range(b):
        for hi in range(heads):
            qq = q[bi, hi]
            attn = (qq * scale) @ k[bi, hi].T  # (HW, HW)
            rq = qq.reshape(h, w, d)  # UNSCALED query
            rel_h = np.einsum("hwc,hkc->hwk", rq, rh)  # (h, w, kh)
            rel_w = np.einsum("hwc,wkc->hwk", rq, rw)  # (h, w, kw)
            attn = attn.reshape(h, w, h, w) + rel_h[:, :, :, None] + rel_w[:, :, None, :]
            p = softmax(attn.reshape(h * w, h * w))
            out[bi, hi] = p @ v[bi, hi]
    out = out.transpose(0, 2, 1, 3).reshape(b, h, w, c)
    return linear(out, st.get(pfx + "proj.weight"), st.get(pfx + "proj.bias"))


def sam(st, img, stages):
    x = conv2d(img, st.get("sam_model.patch_embed.proj.weight"), stride=16)
    x = x + st.get("sam_model.patch_embed.proj.bias")
    x = x + st.get("sam_model.pos_embed")[0]
    stages["sam_patch"] = x.copy()
    for i in range(12):
        p = f"sam_model.blocks.{i}."
        sc = x
        y = layer_norm(x, st.get(p + "norm1.weight"), st.get(p + "norm1.bias"), 1e-6)
        if i in (2, 5, 8, 11):
            y = sam_attention(y[None], st, p + "attn.")[0]
        else:
            wins, pad_hw = window_partition(y, 14)
            wins = sam_attention(wins, st, p + "attn.")
            y = window_unpartition(wins, 14, pad_hw, y.shape[:2])
        x = sc + y
        m = layer_norm(x, st.get(p + "norm2.weight"), st.get(p + "norm2.bias"), 1e-6)
        m = linear(gelu(linear(m, st.get(p + "mlp.lin1.weight"), st.get(p + "mlp.lin1.bias"))),
                   st.get(p + "mlp.lin2.weight"), st.get(p + "mlp.lin2.bias"))
        x = x + m
        if i in (0, 2, 5, 11):
            stages[f"sam_b{i}"] = x.copy()
    x = conv2d(x, st.get("sam_model.neck.0.weight"))
    x = layer_norm(x, st.get("sam_model.neck.1.weight"), st.get("sam_model.neck.1.bias"), 1e-6)
    x = conv2d(x, st.get("sam_model.neck.2.weight"), pad=1)
    x = layer_norm(x, st.get("sam_model.neck.3.weight"), st.get("sam_model.neck.3.bias"), 1e-6)
    stages["sam_neck"] = x.copy()
    x = conv2d(x, st.get("sam_model.net_2.weight"), stride=2, pad=1)
    stages["sam_n2"] = x.copy()
    x = conv2d(x, st.get("sam_model.net_3.weight"), stride=2, pad=1)
    stages["sam_out"] = x.copy()
    return x  # (16, 16, 1024)


def clip(st, sam_out, stages):
    patches = sam_out.reshape(-1, 1024)  # the SAM grid IS the patch embedding
    cls = st.get("vision_model.embeddings.class_embedding")[None]
    x = np.concatenate([cls, patches], 0) + st.get("vision_model.embeddings.position_embedding.weight")
    x = layer_norm(x, st.get("vision_model.pre_layrnorm.weight"), st.get("vision_model.pre_layrnorm.bias"), 1e-5)
    stages["clip_pre"] = x.copy()
    for i in range(24):
        p = f"vision_model.transformer.layers.{i}."
        y = layer_norm(x, st.get(p + "layer_norm1.weight"), st.get(p + "layer_norm1.bias"), 1e-5)
        qkv = linear(y, st.get(p + "self_attn.qkv_proj.weight"), st.get(p + "self_attn.qkv_proj.bias"))
        q, k, v = np.split(qkv, 3, axis=-1)
        heads, d = 16, 64
        q = q.reshape(-1, heads, d).transpose(1, 0, 2)
        k = k.reshape(-1, heads, d).transpose(1, 0, 2)
        v = v.reshape(-1, heads, d).transpose(1, 0, 2)
        a = softmax((q @ k.transpose(0, 2, 1)) * np.float32(d ** -0.5)) @ v
        a = a.transpose(1, 0, 2).reshape(-1, heads * d)
        x = x + linear(a, st.get(p + "self_attn.out_proj.weight"), st.get(p + "self_attn.out_proj.bias"))
        y = layer_norm(x, st.get(p + "layer_norm2.weight"), st.get(p + "layer_norm2.bias"), 1e-5)
        y = linear(quick_gelu(linear(y, st.get(p + "mlp.fc1.weight"), st.get(p + "mlp.fc1.bias"))),
                   st.get(p + "mlp.fc2.weight"), st.get(p + "mlp.fc2.bias"))
        x = x + y
        if i in (0, 11):
            stages[f"clip_l{i}"] = x.copy()
    stages["clip_out"] = x.copy()
    return x  # (257, 1024), no final LayerNorm


def dequant4(st, base, group=64):
    """MLX affine 4-bit: value i of a U32 sits in bits 4i..4i+3, w = q*scale + bias per group."""
    qw = st.get(base + ".weight")
    scales = st.get(base + ".scales")
    biases = st.get(base + ".biases")
    shifts = (4 * np.arange(8, dtype=np.uint32))[None, None, :]
    q = ((qw[:, :, None] >> shifts) & 0xF).reshape(qw.shape[0], -1).astype(np.float32)
    out_f, in_f = q.shape
    q = q.reshape(out_f, in_f // group, group)
    return (q * scales[:, :, None] + biases[:, :, None]).reshape(out_f, in_f)


def tower(st, img, stages):
    s = sam(st, img, stages)
    c = clip(st, s, stages)
    feat = np.concatenate([c[1:], s.reshape(-1, 1024)], axis=-1)  # [CLIP[1:], SAM]
    stages["feat"] = feat.copy()
    w = dequant4(st, "projector.layers")
    proj = linear(feat, w, st.get("projector.layers.bias"))
    stages["proj"] = proj.copy()
    grid = proj.reshape(16, 16, 1280)
    nl = st.get("image_newline")
    grid = np.concatenate([grid, np.broadcast_to(nl, (16, 1, 1280))], axis=1)
    rows = np.concatenate([grid.reshape(-1, 1280), st.get("view_separator")[None]], axis=0)
    stages["rows"] = rows
    return rows  # (273, 1280)


def synthetic_input():
    """A deterministic, structured stand-in for a normalized page: low-frequency
    colour fields plus text-like high-frequency strokes, in [-1, 1]."""
    rng = np.random.default_rng(20261102)
    yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float32) / 1024.0
    img = np.stack([
        0.6 * np.sin(6.3 * xx + 1.0 * yy),
        0.6 * np.cos(5.1 * yy - 2.0 * xx),
        0.4 * np.sin(9.0 * (xx + yy)),
    ], -1)
    strokes = (rng.random((1024, 1024, 1)) > 0.93).astype(np.float32)
    strokes = np.maximum(strokes, np.roll(strokes, 1, 1))
    img = np.clip(img + 0.8 * strokes - 0.1, -1.0, 1.0)
    return img.astype(np.float32)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    st = Store(sys.argv[1])
    out = sys.argv[2]
    import os
    os.makedirs(out, exist_ok=True)
    if len(sys.argv) > 3:
        img = np.load(sys.argv[3]).astype(np.float32)
    else:
        img = synthetic_input()
        np.save(os.path.join(out, "input.npy"), img)
    assert img.shape == (1024, 1024, 3), img.shape
    stages = {}
    rows = tower(st, img, stages)
    np.savez(os.path.join(out, "stages.npz"), **stages)
    # Raw little-endian float32 beside the npz, so the Rust parity test reads
    # them with no zip or npy parsing. `input.f32` is the exact tower input.
    np.ascontiguousarray(img, dtype="<f4").tofile(os.path.join(out, "input.f32"))
    for name, arr in stages.items():
        np.ascontiguousarray(arr, dtype="<f4").tofile(os.path.join(out, name + ".f32"))
    print("rows", rows.shape, "finite", bool(np.isfinite(rows).all()),
          "mean %.5f std %.5f" % (rows.mean(), rows.std()))


if __name__ == "__main__":
    main()
