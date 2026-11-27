#!/usr/bin/env python3
"""A numpy transcription of the Inkling 2 Mini vision tower and merger, for
the parity gates in `docs/INKLING2_VISION_PHASE0.md`.

**What this is and is not.** It is written from transformers
`models/inkling2/{modeling,image_processing}_inkling2.py` and `vision_utils.py`
at `b62646bb` (read, never executed), so a convention both it and the Rust
tower get wrong would pass. It is NOT transformers run on the same bytes. It
shares no code with the Rust side, and it keeps the reference's structure
where the port changes it: q, k and v stay three projections (the port fuses
them), the layer scale stays a separate multiply (the port folds it), the rope
stays interleaved-pair (the port permutes q and k to half-split pairs), and
the position table goes through the reference's per-axis tap and weight form.

Pure numpy and the standard library, so `python3 -I` works.

Inputs:
  --weights   a safetensors file holding the vision tensors in any of the three
              spellings (HF `model.visual.*`, transformers `model.vision_tower.*`,
              mlx-vlm `vision_tower.*`); `scripts/inkling2_vision_fetch.py`
              writes one from the HF shard.
  --image     raw 8-bit pixels, row-major `H x W x C` (C = 3 or 4), with
              --size H W --channels C. Preprocessing runs here.
  --tiles     instead of --image: f32 `[R*C, 40, 40, 3]` normalized tiles,
              with --grid R C.

Writes `header.json` and f32 `<stage>.bin` files into --out-dir:
  tiles     `[R*C, 40, 40, 3]` (only with --image)
  embed     `[1 + 4RC, 1024]` after `ln_pre`, CLS first, merge-block order
  block_0   `[1 + 4RC, 1024]` after the first block
  block_22  `[1 + 4RC, 1024]` after the last block
  merger    `[R*C, 2304]` the rows the trunk receives, tiles in raster order

`--preprocess-only` stops after `tiles` (no weights needed), for the
vision-io golden vectors.

By default every weight is narrowed BF16 -> FP16 -> F32, the precision the
install stores; `--keep-bf16` skips the narrowing.
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

TILE = 40
PATCH = 20
MERGE = 2
HIDDEN = 1024
HEADS = 16
HEAD_DIM = HIDDEN // HEADS
DEPTH = 23
GRID_SIDE = 32
LN_EPS = 1e-5
THETA = 10000.0
OUT_HIDDEN = 2304


# ---- weights -------------------------------------------------------------

# mlx-vlm and transformers renames, mapped back to the HF original names.
_RENAMES = [
    ("pre_layernorm.", "ln_pre."),
    (".layernorm_before.", ".ln_1."),
    (".layernorm_after.", ".ln_2."),
    (".lambda_1", ".ls_1.gamma"),
    (".lambda_2", ".ls_2.gamma"),
    (".mlp.fc1.", ".mlp.c_fc."),
    (".mlp.fc2.", ".mlp.c_proj."),
    ("merger.linear_1.", "proj_in."),
    ("merger.linear_2.", "proj_out."),
    ("embeddings.position_embedding.weight", "embeddings.position_embedding"),
]


def canonical(name):
    """The HF-original name relative to `model.visual.backbone.` (or
    `model.visual.` for the projector), or None for a non-vision tensor."""
    for prefix in ("model.visual.backbone.", "model.visual."):
        if name.startswith(prefix):
            return name[len(prefix):]
    for prefix in ("model.vision_tower.", "vision_tower."):
        if name.startswith(prefix):
            sub = name[len(prefix):]
            for old, new in _RENAMES:
                sub = sub.replace(old, new)
            return sub
    return None


class Weights:
    def __init__(self, path, narrow):
        self.path = Path(path)
        self.narrow = narrow
        with open(self.path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        self.base = 8 + n
        self.index = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            key = canonical(name)
            if key is None:
                continue
            if key in self.index:
                raise ValueError(f"{name}: two spellings of {key}")
            self.index[key] = meta
        if not self.index:
            raise ValueError(f"{path}: no vision tensors")

    def get(self, key, shape=None):
        meta = self.index[key]
        s, e = meta["data_offsets"]
        with open(self.path, "rb") as fh:
            fh.seek(self.base + s)
            data = fh.read(e - s)
        if meta["dtype"] == "BF16":
            bits = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
            arr = bits.view(np.float32)
        elif meta["dtype"] == "F16":
            arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
        elif meta["dtype"] == "F32":
            arr = np.frombuffer(data, dtype=np.float32).copy()
        else:
            raise ValueError(f"{key}: {meta['dtype']}")
        if self.narrow:
            arr = arr.astype(np.float16).astype(np.float32)
        arr = arr.reshape(meta["shape"])
        if shape is not None and list(arr.shape) != list(shape):
            raise ValueError(f"{key}: shape {list(arr.shape)}, expected {list(shape)}")
        return arr


# ---- preprocessing (image_processing_inkling2.py) -------------------------


def preprocess(pixels):
    """`pixels` u8 `[H, W, C]`. Returns f32 tiles `[R*C, 40, 40, 3]`, R, C."""
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    if pixels.shape[-1] == 1:
        pixels = np.repeat(pixels, 3, axis=-1)
    h, w, ch = pixels.shape
    if ch == 4:
        rows = np.arange(h)[:, None] // 8
        cols = np.arange(w)[None, :] // 8
        bg = np.where((rows + cols) % 2 == 0, 204, 153)[..., None]
        rgba = pixels.astype(np.int32)
        alpha = rgba[..., 3:4]
        pixels = ((rgba[..., :3] * alpha + bg * (255 - alpha) + 127) // 255).astype(np.uint8)
    elif ch != 3:
        raise ValueError(f"{ch} channels")
    r, c = math.ceil(h / TILE), math.ceil(w / TILE)
    # Pad with -1 on the 0-255 scale, then the fused rescale + normalize:
    # mean and std 0.5 scaled by 255.
    padded = np.full((r * TILE, c * TILE, 3), -1.0, np.float32)
    padded[:h, :w] = pixels.astype(np.float32)
    padded = (padded - np.float32(127.5)) / np.float32(127.5)
    tiles = padded.reshape(r, TILE, c, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(r * c, TILE, TILE, 3)
    return np.ascontiguousarray(tiles), r, c


# ---- tower (modeling_inkling2.py) ------------------------------------------


def erf(x):
    # Abramowitz-Stegun 7.1.26 is too coarse for a parity reference; use the
    # standard library per element (exact to f64).
    return np.vectorize(math.erf, otypes=[np.float64])(x)


def gelu(x):
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + erf(x64 / math.sqrt(2.0)))).astype(np.float32)


def layer_norm(x, w, b):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + LN_EPS) * w + b


def merge_block_coords(r, c):
    """`(row, col)` of every patch of an `R x C` tile image, merge-block
    order (vision_utils `get_vision_position_ids`)."""
    h, w = r * MERGE, c * MERGE
    hpos, wpos = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    shape = (h // MERGE, MERGE, w // MERGE, MERGE)
    return (
        hpos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1),
        wpos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1),
    )


def axis_taps(index, size, side):
    """vision_utils `_interpolation_axis_taps_weights`, bilinear,
    align_corners=False, border padding."""
    src = (index.astype(np.float32) + 0.5) * side / size - 0.5
    floor = np.floor(src)
    offsets = np.arange(0, 2)
    raw = floor.astype(np.int64)[:, None] + offsets
    taps = np.clip(raw, 0, side - 1)
    dist = np.abs(src[:, None] - floor[:, None] - offsets)
    weights = np.clip(1.0 - dist, 0.0, None)
    return taps, weights.astype(np.float32)


def position_rows(table, r, c):
    rows, cols = merge_block_coords(r, c)
    h_t, h_w = axis_taps(rows, r * MERGE, GRID_SIDE)
    w_t, w_w = axis_taps(cols, c * MERGE, GRID_SIDE)
    idx = (h_t[:, :, None] * GRID_SIDE + w_t[:, None, :]).reshape(-1, 4)
    wts = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return (table[idx] * wts[:, :, None]).sum(1)


def rope_cos_sin(r, c):
    rows, cols = merge_block_coords(r, c)
    pos = np.zeros((1 + rows.size, 2), np.float32)
    pos[1:, 0] = rows + 1
    pos[1:, 1] = cols + 1
    spatial = HEAD_DIM // 2
    inv = 1.0 / (THETA ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial))
    freqs = pos[..., None] * inv  # [seq, 2, 16]
    ang = np.concatenate([freqs[:, 1], freqs[:, 0]], axis=-1)  # [w | h]
    ang = np.repeat(ang, 2, axis=-1)  # interleaved pairs
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rotate_pairwise(x):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return np.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)


def attention(x, w, i, cos, sin):
    seq = x.shape[0]

    def lin(name):
        return x @ w.get(f"layers.{i}.self_attn.{name}.weight").T + w.get(
            f"layers.{i}.self_attn.{name}.bias"
        )

    q = lin("q_proj").reshape(seq, HEADS, HEAD_DIM)
    k = lin("k_proj").reshape(seq, HEADS, HEAD_DIM)
    v = lin("v_proj").reshape(seq, HEADS, HEAD_DIM)
    q = q * cos[:, None] + rotate_pairwise(q) * sin[:, None]
    k = k * cos[:, None] + rotate_pairwise(k) * sin[:, None]
    out = np.empty_like(q)
    scale = HEAD_DIM ** -0.5
    for h in range(HEADS):
        s = (q[:, h] @ k[:, h].T) * scale
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[:, h] = p @ v[:, h]
    out = out.reshape(seq, HIDDEN)
    return out @ w.get(f"layers.{i}.self_attn.out_proj.weight").T + w.get(
        f"layers.{i}.self_attn.out_proj.bias"
    )


def block(x, w, i, cos, sin):
    g = lambda n: w.get(f"layers.{i}.{n}")
    a = attention(layer_norm(x, g("ln_1.weight"), g("ln_1.bias")), w, i, cos, sin)
    x = x + g("ls_1.gamma") * a
    m = layer_norm(x, g("ln_2.weight"), g("ln_2.bias"))
    m = gelu(m @ g("mlp.c_fc.weight").T + g("mlp.c_fc.bias"))
    m = m @ g("mlp.c_proj.weight").T + g("mlp.c_proj.bias")
    return x + g("ls_2.gamma") * m


def tower(tiles, r, c, w, dump):
    n = r * c
    patches = tiles.reshape(n, MERGE, PATCH, MERGE, PATCH, 3).transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(n * MERGE * MERGE, PATCH * PATCH * 3)
    pe = patches @ w.get("embeddings.patch_embedding.weight", (HIDDEN, PATCH * PATCH * 3)).T
    table = w.get("embeddings.position_embedding", (GRID_SIDE * GRID_SIDE, HIDDEN))
    pe = pe + position_rows(table, r, c)
    cls = w.get("embeddings.class_embedding") + w.get("embeddings.cls_position_embedding")
    x = np.concatenate([cls[None], pe], axis=0).astype(np.float32)
    x = layer_norm(x, w.get("ln_pre.weight"), w.get("ln_pre.bias"))
    dump("embed", x)
    cos, sin = rope_cos_sin(r, c)
    for i in range(DEPTH):
        x = block(x, w, i, cos, sin).astype(np.float32)
        if i in (0, DEPTH - 1):
            dump(f"block_{i}", x)
    merged = x[1:].reshape(n, MERGE * MERGE * HIDDEN)
    h = gelu(merged @ w.get("proj_in.weight", (4096, 4096)).T)
    rows = h @ w.get("proj_out.weight", (OUT_HIDDEN, 4096)).T
    dump("merger", rows.astype(np.float32))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--image")
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"))
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--tiles")
    ap.add_argument("--grid", type=int, nargs=2, metavar=("R", "C"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preprocess-only", action="store_true")
    ap.add_argument("--keep-bf16", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = {"stages": {}}

    def dump(stage, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        (out / f"{stage}.bin").write_bytes(arr.tobytes())
        header["stages"][stage] = list(arr.shape)

    if args.image:
        if not args.size:
            ap.error("--image needs --size H W")
        h, w = args.size
        raw = np.fromfile(args.image, dtype=np.uint8)
        if raw.size != h * w * args.channels:
            ap.error(f"{args.image}: {raw.size} bytes, expected {h * w * args.channels}")
        tiles, r, c = preprocess(raw.reshape(h, w, args.channels))
        dump("tiles", tiles)
        header["image"] = {"height": h, "width": w, "channels": args.channels}
    elif args.tiles:
        if not args.grid:
            ap.error("--tiles needs --grid R C")
        r, c = args.grid
        tiles = np.fromfile(args.tiles, dtype=np.float32).reshape(r * c, TILE, TILE, 3)
    else:
        ap.error("give --image or --tiles")
    header["grid"] = [1, r, c]

    if not args.preprocess_only:
        if not args.weights:
            ap.error("--weights is required unless --preprocess-only")
        weights = Weights(args.weights, narrow=not args.keep_bf16)
        tower(tiles, r, c, weights, dump)
        header["weights"] = str(args.weights)
        header["narrowed_fp16"] = not args.keep_bf16
    (out / "header.json").write_text(json.dumps(header, indent=2))


if __name__ == "__main__":
    main()
