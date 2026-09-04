#!/usr/bin/env python3
"""Bisect this port's MTP draft step against the published drafter weights.

`docs/MTP_SPECULATIVE.md` step 3: the head runs, its installed weights match
the checkpoint to 0.992-0.996, its shapes match a trunk full-attention layer
exactly, and its output is nonetheless ANTI-aligned with the trunk. Reading
mlx-vlm's reference confirmed five of the port's design choices and found one
real defect (the hidden input is the trunk's POST-final-norm state), and
fixing that did not move the symptom. So the remaining question is not "which
convention" but "which STAGE", and this answers it numerically.

WHAT IT COMPARES, AND WHY THOSE FOUR

The port dumps what survives its single command buffer (`TURBOSPARK_MTP_DUMP`):

    concat      the exact input to `fc`, both halves, already normed
    moe_x       post_attention_layernorm(fc_out + attn_out)
    block_out   the block's output after the FFN residual
    post_norm   after the head's own `mtp.norm`

This script recomputes each from `concat` using the PUBLISHED drafter weights
(`mlx-community/Qwen3.8-27B-MTP-4bit`) and reports the correlation. Reading
them in order localizes the fault to one stage:

    moe_x      disagrees -> `fc` or the attention block
    block_out  disagrees (moe_x fine) -> the FFN
    post_norm  disagrees (block fine) -> the head's final norm

`fc`'s raw output is deliberately NOT dumped: one command buffer runs the
whole step, so the attention residual overwrites it before the host can read
it. It is recomputed here instead, which costs nothing and keeps the step's
single-commit shape.

WHY THE PUBLISHED DRAFTER AND NOT THE INSTALL'S OWN WEIGHTS. Comparing the
port against weights it dequantized itself is the self-consistency trap
AGENTS.md Gotcha 48 names: it passes whenever the writer and the reader share
a mistake. These are an INDEPENDENT INT4 conversion of the same head, done by
mlx_vlm.convert at the same scheme and group size (affine, group 64), so an
agreement here is evidence and not a tautology.

    hf download mlx-community/Qwen3.8-27B-MTP-4bit --local-dir ~/models/qwen38-mtp-ref

    TURBOSPARK_MTP_DUMP=/tmp/mtp-dump TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \\
      cargo test -p turbospark-bench --test mtp_head_probe --release -- --ignored --nocapture what_the_mtp

    uv run --python 3.12 --with numpy \\
      scripts/mtp_bisect.py /tmp/mtp-dump ~/models/qwen38-mtp-ref
"""

import json
import sys
from pathlib import Path

import numpy as np

GROUP = 64
BITS = 4


def load_safetensors(path: Path) -> dict:
    """Minimal reader, because `safetensors.numpy` cannot decode BF16.

    Hand-rolled rather than routed through mlx on purpose: this script's job
    is to be an INDEPENDENT decoder of the same bytes, and borrowing the
    reference's own loader would weaken that (AGENTS.md Gotcha 48). The format
    is an 8-byte little-endian header length, that many bytes of JSON, then
    the data region, with every `data_offsets` pair relative to its start.
    """
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    head = json.loads(raw[8 : 8 + n])
    base = 8 + n
    out = {}
    for name, meta in head.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        buf = raw[base + start : base + end]
        dt = meta["dtype"]
        if dt == "BF16":
            # BF16 is the top 16 bits of an FP32 word, so widening is a shift.
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(buf, dtype=np.float32)
        elif dt in ("U32", "I32"):
            arr = np.frombuffer(buf, dtype=np.uint32)
        else:
            raise ValueError(f"{name}: unhandled dtype {dt}")
        out[name] = arr.reshape(meta["shape"])
    return out


def f16(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float16).astype(np.float32)


def dequant(w: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """MLX `affine` dequant: uint32 words, `BITS` per element, LSB first.

    The same layout `dequant_int4_gemv_simd` reads, stated here rather than
    imported so this script is a genuinely independent decoder.
    """
    rows, packed_cols = w.shape
    per_word = 32 // BITS
    q = np.empty((rows, packed_cols * per_word), dtype=np.float32)
    words = w.view(np.uint32) if w.dtype != np.uint32 else w
    for i in range(per_word):
        q[:, i::per_word] = (words >> (BITS * i)) & ((1 << BITS) - 1)
    cols = q.shape[1]
    s = scales.astype(np.float32).repeat(GROUP, axis=1)[:, :cols]
    b = biases.astype(np.float32).repeat(GROUP, axis=1)[:, :cols]
    return q * s + b


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d == 0 else float(a @ b / d)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    dump, ref = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads((dump / "meta.json").read_text())
    hidden = meta["hidden"]
    print(f"\ndump: token={meta['token']} position={meta['position']} hidden={hidden}")

    st = ref / "model.safetensors"
    if not st.exists():
        print(f"missing {st}\nhf download mlx-community/Qwen3.8-27B-MTP-4bit --local-dir {ref}")
        return 2
    W = load_safetensors(st)
    cfg = json.loads((ref / "config.json").read_text())
    eps = float(cfg.get("text_config", cfg).get("rms_norm_eps", 1e-6))

    def mat(name: str) -> np.ndarray:
        if f"{name}.scales" in W:
            return dequant(W[f"{name}.weight"], W[f"{name}.scales"], W[f"{name}.biases"])
        return W[f"{name}.weight"].astype(np.float32)

    concat = f16(dump / "concat.f16")
    assert concat.size == 2 * hidden, f"concat is {concat.size}, expected {2 * hidden}"

    # -- fc, then the block. Attention at position 0 against an EMPTY cache
    #    attends to itself alone, so softmax is 1.0 and the whole block is a
    #    function of `concat` with no history to replay. That is why the
    #    probe takes its dump at the first draft.
    x = concat @ mat("fc").T
    h = rms_norm(x, W["layers.0.input_layernorm.weight"].astype(np.float32), eps)

    n_head = cfg.get("text_config", cfg)["num_attention_heads"]
    head_dim = cfg.get("text_config", cfg)["head_dim"]
    qkv = mat("layers.0.self_attn.q_proj")
    q_all = h @ qkv.T
    # Qwen packs [query; gate] into q_proj; the gate is a sigmoid on the
    # attention OUTPUT, not on q.
    q, gate = q_all[: n_head * head_dim], q_all[n_head * head_dim :]
    v = h @ mat("layers.0.self_attn.v_proj").T
    # Position 0, one key: softmax over a single logit is exactly 1, so the
    # attention output is V. No RoPE dependence and no q/k norm dependence,
    # which is what makes this stage a clean test of fc + o_proj alone.
    n_kv = cfg.get("text_config", cfg)["num_key_value_heads"]
    # GQA maps query head i to kv head i // (n_head / n_kv), so the kv rows
    # REPEAT in blocks. `np.tile` would interleave them instead and quietly
    # pair most heads with the wrong kv row.
    ctx = np.repeat(v.reshape(n_kv, head_dim), n_head // n_kv, axis=0).reshape(-1)
    ctx = ctx * (1.0 / (1.0 + np.exp(-gate)))
    attn_out = ctx @ mat("layers.0.self_attn.o_proj").T
    x = x + attn_out
    moe_x = rms_norm(x, W["layers.0.post_attention_layernorm.weight"].astype(np.float32), eps)

    g = moe_x @ mat("layers.0.mlp.gate_proj").T
    u = moe_x @ mat("layers.0.mlp.up_proj").T
    ffn = ((g / (1.0 + np.exp(-g))) * u) @ mat("layers.0.mlp.down_proj").T
    block_out = x + ffn
    post_norm = rms_norm(block_out, W["norm.weight"].astype(np.float32), eps)

    print(f"\n{'stage':<12} {'pearson':>10}  {'max|rel|':>10}   reading")
    stages = [
        ("moe_x", moe_x, "fc + attention block"),
        ("block_out", block_out, "the FFN"),
        ("post_norm", post_norm, "the head's final norm"),
    ]
    worst = None
    for name, want, blame in stages:
        got = f16(dump / f"{name}.f16")
        r = corr(want, got)
        denom = np.maximum(np.abs(want), 1e-3)
        rel = float(np.max(np.abs(want - got) / denom))
        flag = "ok" if r > 0.99 else f"DIVERGES -> suspect {blame}"
        print(f"{name:<12} {r:>10.5f}  {rel:>10.4f}   {flag}")
        if worst is None and r <= 0.99:
            worst = name

    print(
        "\nRead these IN ORDER: the first stage that diverges is the one to look at,\n"
        "because every later stage takes its input from it. All three agreeing means\n"
        "the block is faithful and the fault is in `concat` (the two pre-fc norms, or\n"
        "which hidden state feeds them) or in the lm_head that follows.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
