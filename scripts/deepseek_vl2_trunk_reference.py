#!/usr/bin/env python3
"""A numpy transcription of the DeepSeek-VL2-tiny language model, for the
trunk gate in `docs/DEEPSEEK_VL2_VISION_PHASE0.md`.

**What this is and is not.** It is written from mlx-vlm's
`models/deepseek_vl_v2/language.py` (read, never executed): `LlamaAttention`
(plain multi-head attention, NeoX rope at theta 10000, scale head_dim^-0.5),
`DeepseekV2MLP` (SwiGLU), `MoEGate` (softmax over ALL experts, greedy top-k,
the raw softmax values as weights, `routed_scaling_factor` 1.0) and
`DeepseekV2MoE` (`switch_mlp` experts plus the fused `shared_experts`). It
shares no code with the Rust engine, but a misreading of the reference would
be shared by both, and the checkpoint's MLX 4-bit planes are dequantized here
by the MLX rule (`w = q * scale + bias` per 64-wide group, element `i` of a
row at bit `4 * (i % 8)` of word `i / 8`), so it reads the same bytes the
engine's install was built from. It is NOT mlx-vlm run on those bytes.

Pure numpy and the standard library.

Usage:
    python3 -I scripts/deepseek_vl2_trunk_reference.py \
      --model-dir /Volumes/T7/hf-models/deepseek-vl2-tiny-4bit \
      --ids ids.txt --engine-prefix /Volumes/T7/vision-dumps/dsvl2-engine

`ids.txt` is one id per line (or whitespace separated). With
`--engine-prefix P` it compares against the engine's `P.argmax.txt` and
`P.logits.bin` (`crates/runtime/tests/deepseek_vl2_trunk_real.rs`).
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

EPS = 1e-6
THETA = 10000.0
GROUP = 64
BITS = 4


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
        dt = {"U32": "<u4", "F16": "<f2", "BF16": None, "F32": "<f4"}[meta["dtype"]]
        arr = np.frombuffer(self.mm[self.base + s : self.base + e], dtype=dt)
        return arr.reshape(meta["shape"])

    def has(self, name):
        return name in self.header

    def dense(self, name):
        """A raw (unquantized) tensor as float32."""
        return self.raw(name).astype(np.float32)

    def quant(self, base, rows=None):
        """Dequantize `<base>.weight/.scales/.biases` (optionally only `rows`)."""
        w = self.raw(base + ".weight")
        s = self.raw(base + ".scales")
        b = self.raw(base + ".biases")
        if rows is not None:
            w, s, b = w[rows], s[rows], b[rows]
        shifts = np.arange(0, 32, BITS, dtype=np.uint32)
        q = ((w[..., None] >> shifts) & ((1 << BITS) - 1)).astype(np.float32)
        q = q.reshape(*w.shape[:-1], w.shape[-1] * (32 // BITS))
        sc = np.repeat(s.astype(np.float32), GROUP, axis=-1)
        bi = np.repeat(b.astype(np.float32), GROUP, axis=-1)
        return q * sc + bi


def rms_norm(x, w):
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(x, positions, head_dim):
    """NeoX half-split rope: `nn.RoPE(head_dim, traditional=False, base=theta)`."""
    half = head_dim // 2
    inv = THETA ** (-2.0 * np.arange(half, dtype=np.float64) / head_dim)
    ang = positions[:, None].astype(np.float64) * inv[None, :]  # [seq, half]
    cos = np.cos(ang).astype(np.float32)[:, None, :]
    sin = np.sin(ang).astype(np.float32)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--engine-prefix")
    ap.add_argument("--out-prefix")
    args = ap.parse_args()

    ck = Checkpoint(args.model_dir)
    cfg = json.load(open(Path(args.model_dir) / "config.json"))["language_config"]
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    head_dim = hidden // heads
    layers = cfg["num_hidden_layers"]
    lead = cfg["first_k_dense_replace"]
    n_exp = cfg["n_routed_experts"]
    top_k = cfg["num_experts_per_tok"]
    ids = [int(t) for t in Path(args.ids).read_text().split()]
    seq = len(ids)
    pos = np.arange(seq)
    P = "language_model.model."

    # Embedding rows only (the table is 129280 x 1280).
    x = ck.quant(P + "embed_tokens", rows=np.array(ids))
    mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), k=1)
    expert_cache = {}

    def expert(layer, e, proj):
        key = (layer, e, proj)
        if key not in expert_cache:
            expert_cache[key] = ck.quant(
                f"{P}layers.{layer}.mlp.switch_mlp.{proj}", rows=e
            )
        return expert_cache[key]

    for layer in range(layers):
        L = f"{P}layers.{layer}."
        h = rms_norm(x, ck.dense(L + "input_layernorm.weight"))
        q = (h @ ck.quant(L + "self_attn.q_proj").T).reshape(seq, heads, head_dim)
        k = (h @ ck.quant(L + "self_attn.k_proj").T).reshape(seq, heads, head_dim)
        v = (h @ ck.quant(L + "self_attn.v_proj").T).reshape(seq, heads, head_dim)
        q, k = rope(q, pos, head_dim), rope(k, pos, head_dim)
        scores = np.einsum("qhd,khd->hqk", q, k) * head_dim**-0.5 + mask[None]
        scores -= scores.max(-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(-1, keepdims=True)
        a = np.einsum("hqk,khd->qhd", p, v).reshape(seq, hidden)
        x = x + a @ ck.quant(L + "self_attn.o_proj").T

        h = rms_norm(x, ck.dense(L + "post_attention_layernorm.weight"))
        if layer < lead:
            g = h @ ck.quant(L + "mlp.gate_proj").T
            u = h @ ck.quant(L + "mlp.up_proj").T
            x = x + (silu(g) * u) @ ck.quant(L + "mlp.down_proj").T
        else:
            logits = h @ ck.dense(L + "mlp.gate.weight").T  # [seq, n_exp]
            logits = logits - logits.max(-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(-1, keepdims=True)  # softmax over ALL experts
            y = np.zeros_like(x)
            for t in range(seq):
                top = np.argsort(-probs[t], kind="stable")[:top_k]
                for e in top:  # raw softmax value, no renormalization
                    g = h[t] @ expert(layer, e, "gate_proj").T
                    u = h[t] @ expert(layer, e, "up_proj").T
                    y[t] += probs[t, e] * ((silu(g) * u) @ expert(layer, e, "down_proj").T)
            sg = h @ ck.quant(L + "mlp.shared_experts.gate_proj").T
            su = h @ ck.quant(L + "mlp.shared_experts.up_proj").T
            y = y + (silu(sg) * su) @ ck.quant(L + "mlp.shared_experts.down_proj").T
            x = x + y
        print(f"layer {layer:2d} absmax {np.abs(x).max():.2f}", flush=True)

    x = rms_norm(x, ck.dense(P + "norm.weight"))
    lm = ck.quant("language_model.lm_head")  # [vocab, hidden]
    logits = x @ lm.T  # [seq, vocab]
    argmax = logits.argmax(-1)
    print("argmax per position:", argmax.tolist())

    if args.out_prefix:
        Path(args.out_prefix + ".argmax.txt").write_text("".join(f"{a}\n" for a in argmax))
        logits[-1].astype("<f4").tofile(args.out_prefix + ".logits.bin")

    if args.engine_prefix:
        eng_arg = [int(t) for t in Path(args.engine_prefix + ".argmax.txt").read_text().split()]
        eng = np.fromfile(args.engine_prefix + ".logits.bin", dtype="<f4")
        ref = logits[-1]
        agree = sum(int(a == b) for a, b in zip(eng_arg, argmax.tolist()))
        print(f"teacher-forced argmax agreement: {agree}/{seq}")
        cos = float(eng @ ref / (np.linalg.norm(eng) * np.linalg.norm(ref)))
        pe = np.exp(eng - eng.max())
        pe /= pe.sum()
        pr = np.exp(ref - ref.max())
        pr /= pr.sum()
        kl = float(np.sum(pr * (np.log(pr + 1e-30) - np.log(pe + 1e-30))))
        top5_e = set(np.argsort(-eng)[:5].tolist())
        top5_r = set(np.argsort(-ref)[:5].tolist())
        print(f"last-position logits: cosine {cos:.6f}  KL(ref||engine) {kl:.5f} nats  "
              f"top-5 overlap {len(top5_e & top5_r)}/5  max|diff| {np.abs(eng - ref).max():.3f}")
        for i, (a, b) in enumerate(zip(eng_arg, argmax.tolist())):
            if a != b:
                print(f"  position {i}: engine {a}, reference {b}")


if __name__ == "__main__":
    main()
