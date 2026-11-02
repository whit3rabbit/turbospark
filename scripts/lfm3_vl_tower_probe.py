"""Independent check of the LFM3-VL vision tower (docs/LFM3_VL_VISION_PHASE0.md).

A self-contained numpy implementation of the tower and projector, written from
the published reference formulas (`mlx_vlm/models/lfm3_vl/vision.py`,
`lfm3_vl.py::MultiModalProjector`) and reading the checkpoint's ORIGINAL
tensors: no column permutation, no expanded KV heads, no FP16 narrowing. It
executes nothing from the reference checkout, only numpy.

It compares against the stages the Rust test dumped
(`crates/runtime/tests/lfm3_vl_vision_real.rs`). Two arms:

  f32    every activation in float32 (weights are the checkpoint's BF16
         values, exact in float32)
  bf16   activations rounded to bfloat16 after every op, an emulation of the
         reference's own storage type

`floor` is cos(bf16, f32): the storage-type difference a BF16 reference has
against itself. A port stage is "at the floor" when cos(rust, f32) is within
a hair of it. The patch rows are the port's own, permuted back to the
processor's (C, T, P_h, P_w) order, so preprocessing is held out.

    uv run --python 3.12 --with numpy -- python scripts/lfm3_vl_tower_probe.py \
        --model-dir /Volumes/T7/hf-models/LFM3-VL-2B-MLX-4bit \
        --dump-dir /Volumes/T7/lfm3-vl-dump
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_safetensors(path, wanted):
    """name -> float32 array, for the BF16/F16/F32 tensors whose name passes `wanted`."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name == "__metadata__" or not wanted(name):
                continue
            start, end = info["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            if info["dtype"] == "BF16":
                u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                arr = u.view(np.float32)
            elif info["dtype"] == "F16":
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            elif info["dtype"] == "F32":
                arr = np.frombuffer(raw, dtype="<f4")
            else:
                raise ValueError(f"{name}: {info['dtype']}")
            out[name] = arr.reshape(info["shape"]).copy()
    return out


def bf16_round(x):
    """Round float32 to the nearest bfloat16 (ties to even), back to float32."""
    u = x.astype(np.float32).view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def fp16_round(x):
    """Round float32 to the nearest float16, back to float32."""
    return x.astype(np.float16).astype(np.float32)


def rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu_erf(x):
    from math import sqrt

    # erf via numpy: use the identity with np.vectorize-free tanh-free form.
    return 0.5 * x * (1.0 + _erf(x / sqrt(2.0)))


def _erf(x):
    # Abramowitz-Stegun 7.1.26 is only 1.5e-7 accurate, plenty for a 1e-3 bound.
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-a * a)
    return sign * y


def silu(x):
    return x / (1.0 + np.exp(-x))


def tower_and_projector(weights, vc, cfg, ref_rows, h, w, q):
    """The reference forward. `q` rounds activations (identity for the f32 arm)."""
    hid, heads, kv = vc["hidden_size"], vc["num_attention_heads"], vc["num_key_value_heads"]
    hd = hid // heads
    eps = vc["norm_eps"]
    merge = vc["spatial_merge_size"]
    seq = h * w
    W = lambda k: weights["vision_tower." + k]

    # grid_positions: (row, col) per token in merge-window order.
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), -1)
    coords = (
        coords.reshape(h // merge, merge, w // merge, merge, 2)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 2)
    )
    spatial_dim = hd // 2
    inv_freq = vc["rope_parameters"]["rope_theta"] ** (
        -np.arange(0, spatial_dim, 2, dtype=np.float32) / spatial_dim
    )
    angles = (coords[..., None].astype(np.float32) * inv_freq).reshape(seq, -1)
    angles = np.concatenate([angles, angles], -1)
    cos, sin = np.cos(angles), np.sin(angles)

    def rotate(t):  # t: [seq, heads, hd]
        a, b = np.split(t, 2, axis=-1)
        return t * cos[:, None] + np.concatenate([-b, a], -1) * sin[:, None]

    x = q(rms_norm(q(ref_rows @ W("patchifier.weight").T), W("embedding_norm.weight"), eps))
    stages = {"patch_embed": x.copy()}
    for i in range(vc["num_hidden_layers"]):
        pre = f"layers.{i}."
        hn = q(rms_norm(x, W(pre + "operator_norm.weight"), eps))
        if vc["layer_types"][i] == "full_attention":
            qh = q(hn @ W(pre + "self_attn.q_proj.weight").T).reshape(seq, heads, hd)
            kh = q(hn @ W(pre + "self_attn.k_proj.weight").T).reshape(seq, kv, hd)
            vh = q(hn @ W(pre + "self_attn.v_proj.weight").T).reshape(seq, kv, hd)
            qh = q(rms_norm(qh, W(pre + "self_attn.q_layernorm.weight"), eps))
            kh = q(rms_norm(kh, W(pre + "self_attn.k_layernorm.weight"), eps))
            qh, kh = q(rotate(qh)), q(rotate(kh))
            rep = heads // kv
            kh, vh = np.repeat(kh, rep, axis=1), np.repeat(vh, rep, axis=1)
            scores = np.einsum("qhd,khd->hqk", qh, kh) * hd**-0.5
            scores -= scores.max(-1, keepdims=True)
            p = np.exp(scores)
            p /= p.sum(-1, keepdims=True)
            o = q(np.einsum("hqk,khd->qhd", p, vh).reshape(seq, hid))
            mix = q(o @ W(pre + "self_attn.out_proj.weight").T)
        else:
            proj = q(hn @ W(pre + "conv.in_proj.weight").T)
            b, c, v = np.split(proj, 3, axis=-1)
            bv = q(b * v)
            grid = (
                bv.reshape(h // merge, w // merge, merge, merge, hid)
                .transpose(0, 2, 1, 3, 4)
                .reshape(h, w, hid)
            )
            k = W(pre + "conv.spatial_conv.weight")[..., 0]  # [C, 3, 3]
            pad = np.pad(grid, ((1, 1), (1, 1), (0, 0)))
            conv = np.zeros_like(grid)
            for dr in range(3):
                for dc in range(3):
                    conv += pad[dr : dr + h, dc : dc + w, :] * k[:, dr, dc]
            conv = q(conv)
            back = (
                conv.reshape(h // merge, merge, w // merge, merge, hid)
                .transpose(0, 2, 1, 3, 4)
                .reshape(seq, hid)
            )
            mix = q(q(c * back) @ W(pre + "conv.out_proj.weight").T)
        x = q(x + mix)
        m = q(rms_norm(x, W(pre + "ffn_norm.weight"), eps))
        g = q(silu(q(m @ W(pre + "feed_forward.w1.weight").T)))
        u = q(m @ W(pre + "feed_forward.w3.weight").T)
        x = q(x + q(q(g * u) @ W(pre + "feed_forward.w2.weight").T))
        stages[f"block_{i}"] = x.copy()

    pw = lambda k: weights["multi_modal_projector." + k]
    y = q(layer_norm(x, pw("layer_norm.weight"), pw("layer_norm.bias"), cfg["projector_layer_norm_eps"]))
    y = y.reshape(-1, hid * cfg["downsample_factor"] ** 2)
    y = q(gelu_erf(q(y @ pw("linear_1.weight").T + pw("linear_1.bias"))))
    stages["rows"] = q(y @ pw("linear_2.weight").T + pw("linear_2.bias"))
    return stages


def read_f16(path, shape):
    return np.fromfile(path, dtype="<f2").astype(np.float32).reshape(shape)


def cosine(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def row_cosine(a, b):
    num = (a * b).sum(-1).astype(np.float64)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30
    c = num / den
    return float(c.mean()), float(c.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dump-dir", required=True)
    args = ap.parse_args()
    model_dir, dump = Path(args.model_dir), Path(args.dump_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    vc = cfg["vision_config"]
    grid = json.loads((dump / "grid.json").read_text())
    t, h, w, patch_dim = grid["t"], grid["h"], grid["w"], grid["patch_dim"]
    assert t == 1
    seq = h * w

    # The port's rows are (T, P_h, P_w, C); the checkpoint's patchifier eats
    # (C, T, P_h, P_w).
    rows = np.fromfile(dump / "patch_rows.f32", dtype="<f4").reshape(seq, patch_dim)
    p, tp, ch = vc["patch_size"], vc["temporal_patch_size"], vc["in_channels"]
    ref_rows = (
        rows.reshape(seq, tp, p, p, ch).transpose(0, 4, 1, 2, 3).reshape(seq, patch_dim)
    )

    weights = read_safetensors(
        model_dir / "model.safetensors",
        lambda n: n.startswith(("vision_tower.", "multi_modal_projector.")),
    )
    f32 = tower_and_projector(weights, vc, cfg, ref_rows, h, w, lambda x: x)
    bf16 = tower_and_projector(weights, vc, cfg, bf16_round(ref_rows), h, w, bf16_round)
    # The port's storage type: FP16 activations with float32 accumulation. A
    # port that tracks this twin at ~1.0 but sits below the f32 arm is paying
    # FP16 storage, not carrying a bug.
    f16 = tower_and_projector(
        weights, vc, cfg, fp16_round(ref_rows), h, w, fp16_round
    )

    rust = {"patch_embed": read_f16(dump / "patch_embed.f16", (seq, vc["hidden_size"]))}
    for i in range(vc["num_hidden_layers"]):
        rust[f"block_{i}"] = read_f16(dump / f"block_{i}.f16", (seq, vc["hidden_size"]))
    merged = seq // (vc["spatial_merge_size"] ** 2)
    rust["rows"] = read_f16(dump / "rows.f16", (merged, cfg["text_config"]["hidden_size"]))

    print(
        f"{'stage':12} {'bf16floor':>10} {'f16floor':>10} {'rust~f32':>10} {'rust~f16':>10} "
        f"{'rowmean':>9} {'rowmin':>9} {'|x|max':>9}"
    )
    worst, worst_twin = 1.0, 1.0
    for name in rust:
        floor = cosine(bf16[name], f32[name])
        floor16 = cosine(f16[name], f32[name])
        c32 = cosine(rust[name], f32[name])
        ctwin = cosine(rust[name], f16[name])
        rmean, rmin = row_cosine(rust[name], f32[name])
        worst, worst_twin = min(worst, c32), min(worst_twin, ctwin)
        print(
            f"{name:12} {floor:10.8f} {floor16:10.8f} {c32:10.8f} {ctwin:10.8f} "
            f"{rmean:9.6f} {rmin:9.6f} {np.abs(f32[name]).max():9.1f}"
        )
    print(f"\nworst rust~f32 cosine over all stages: {worst:.8f}")
    print(f"worst rust~f16-twin cosine over all stages: {worst_twin:.8f}")


if __name__ == "__main__":
    main()
