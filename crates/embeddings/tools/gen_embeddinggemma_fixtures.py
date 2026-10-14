#!/usr/bin/env python3
"""Regenerate the EmbeddingGemma Rust parity fixtures.

Normative reference: Hugging Face transformers 4.57.1 `Gemma3TextModel`
(sdpa attention, f32, CPU) plus sentence-transformers 5.1.2 for the
tokenize -> backbone -> Pooling -> Dense -> Dense -> Normalize pipeline.

Two independent parts:

--tiny (no download, deterministic, seed 0)
    A random-init 3-layer Gemma3 text model (sliding, full, sliding;
    hidden 64, 4 query / 2 kv heads, head_dim 16, sliding_window 4,
    query_pre_attn_scalar 12) with a sentence-transformers layout
    (modules.json, 1_Pooling, 2_Dense, 3_Dense) and a second copy in the
    MLX layout (`model.` prefix, `dense.{0,1}` head, every eligible
    tensor MLX-affine 4-bit quantized with group 64, the rest F16). The
    norm weights are random, not zero, so `(1 + w)` is exercised. Golden
    per-layer hidden states and final embeddings come from the HF model
    loaded with the f32 weights (or the dequantized MLX weights).

--real MODEL_DIR (the pinned google/embeddinggemma-300m install)
    SentenceTransformer(MODEL_DIR).encode for a fixed text set covering
    the query and document prompts, a padded-whitespace text (strip), a
    non-ASCII text, a > 512-token text (sliding band), and a > 2048-token
    text (max_seq_length truncation). Stores the token ids
    sentence-transformers fed the model, the normalized embeddings, a
    Matryoshka 256-dim embedding (normalize_embeddings=True), and the
    embedding / layer-0 / last hidden state of one short text.

--mlx TAG MLX_DIR (with --real or --model-dir, repeatable)
    Golden embeddings for an mlx-community conversion: the same
    sentence-transformers pipeline with every weight replaced by the
    conversion's tensor (MLX affine codes dequantized in numpy, F16
    widened to f32), run on the real case texts.

--qat TAG QAT_DIR (repeatable, TAG in q8/q4)
    sentence-transformers goldens for Google's QAT checkpoints, the
    sources of the mlx-community 8-bit (q8) and 4-bit (q4) conversions.
    With --mlx and --model-dir/--real also present, writes
    mlx_provenance.json: per tensor, how many quantization steps each
    stored MLX code sits from round((w - bias) / scale) of the QAT and of
    the upstream weights.

Run with the pinned venv, isolated:

    emb-venv/bin/python -I crates/embeddings/tools/gen_embeddinggemma_fixtures.py --tiny
    emb-venv/bin/python -I crates/embeddings/tools/gen_embeddinggemma_fixtures.py \
        --real ~/models/embeddings/embeddinggemma-300m
    emb-venv/bin/python -I crates/embeddings/tools/gen_embeddinggemma_fixtures.py \
        --model-dir ~/models/embeddings/embeddinggemma-300m \
        --mlx bf16 ~/models/embeddings/mlx-embeddinggemma-300m-bf16 \
        --mlx 8bit ~/models/embeddings/mlx-embeddinggemma-300m-8bit \
        --mlx 4bit ~/models/embeddings/mlx-embeddinggemma-300m-4bit \
        --qat q8 ~/models/embeddings/embeddinggemma-300m-qat-q8_0-unquantized \
        --qat q4 ~/models/embeddings/embeddinggemma-300m-qat-q4_0-unquantized

Outputs under crates/embeddings/testdata/embeddinggemma/ plus
manifest.json (generator, package versions, pins, sha256 per file). Each
part rewrites only its own files and then rebuilds the manifest over the
whole directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

OUT = Path(__file__).resolve().parent.parent / "testdata" / "embeddinggemma"

UPSTREAM_REPO = "google/embeddinggemma-300m"
UPSTREAM_REV = "57c266a740f537b4dc058e1b0cda161fd15afa75"
MLX_PINS = {
    "bf16": ("mlx-community/embeddinggemma-300m-bf16", "2f420efb007317a1ef2aa1608f87ef24226e6807"),
    "8bit": ("mlx-community/embeddinggemma-300m-8bit", "c8316c9e35cf8830541b04d315398c67faa6e497"),
    "4bit": ("mlx-community/embeddinggemma-300m-4bit", "5d9ef074df3957afc5c77127f208fddbc3c54187"),
}
# Google's QAT checkpoints (unquantized f32 weights trained for q8_0 / q4_0).
# Per the mlx-community model cards, the 8/6/5-bit conversions come from
# q8 and the 4-bit conversion from q4.
QAT_PINS = {
    "q8": ("google/embeddinggemma-300m-qat-q8_0-unquantized", "7b5b24595322ab0ea4d08827066860a6df8cb0aa"),
    "q4": ("google/embeddinggemma-300m-qat-q4_0-unquantized", "7740c61c3ad98a9deb8423a20388ab6c3c5c45f2"),
}
MLX_SOURCE = {"8bit": "q8", "4bit": "q4"}

PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
}


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def save_npy(name: str, arr: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / name, np.ascontiguousarray(arr))


# ---------------------------------------------------------------- tiny


TINY = dict(
    vocab_size=48,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    query_pre_attn_scalar=12,
    sliding_window=4,
    layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    rope_theta=1_000_000.0,
    rope_local_base_freq=10_000.0,
    max_position_embeddings=64,
    rms_norm_eps=1e-6,
    hidden_activation="gelu_pytorch_tanh",
    use_bidirectional_attention=True,
    attn_logit_softcapping=None,
    final_logit_softcapping=None,
    rope_scaling=None,
    attention_bias=False,
    pad_token_id=0,
    bos_token_id=2,
    eos_token_id=1,
)
TINY_DENSE = [(64, 256), (256, 64)]
# Lengths below, at, and well above the window of 4 so the band is hit.
TINY_CASES = [
    [2, 7, 1],
    [2, 11, 5, 9, 1],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1],
    [2, 47, 30, 30, 18, 9, 40, 22, 13, 5, 41, 36, 27, 8, 19, 1],
]
GROUP = 64
BITS = 4


def mlx_quantize(w: np.ndarray):
    """MLX affine layout: per group of 64 along the input axis,
    w ~ q * scale + bias with q in [0, 2^bits). Scales and biases are
    stored F16 (the dtype MLX writes for an f16 model); q is packed
    little-endian, 8 values per uint32 at 4 bits. Returns the packed
    triple and the exact f32 dequantization the Rust loader must
    reproduce."""
    out_dim, in_dim = w.shape
    g = w.reshape(out_dim, in_dim // GROUP, GROUP).astype(np.float32)
    lo = g.min(axis=-1)
    hi = g.max(axis=-1)
    levels = (1 << BITS) - 1
    scale = np.maximum((hi - lo) / levels, 1e-8).astype(np.float16)
    bias = lo.astype(np.float16)
    s32 = scale.astype(np.float32)[..., None]
    b32 = bias.astype(np.float32)[..., None]
    q = np.clip(np.rint((g - b32) / s32), 0, levels).astype(np.uint32)
    deq = (q.astype(np.float32) * s32 + b32).astype(np.float32)
    q = q.reshape(out_dim, in_dim)
    per_word = 32 // BITS
    q = q.reshape(out_dim, in_dim // per_word, per_word)
    packed = np.zeros((out_dim, in_dim // per_word), dtype=np.uint32)
    for k in range(per_word):
        packed |= q[..., k] << np.uint32(BITS * k)
    return packed, scale, bias, deq.reshape(out_dim, in_dim)


def build_tiny_model(state: dict):
    from transformers import Gemma3TextConfig, Gemma3TextModel

    cfg = Gemma3TextConfig(**TINY)
    cfg._attn_implementation = "sdpa"
    model = Gemma3TextModel(cfg).eval()
    missing, unexpected = model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in state.items()}, strict=True
    )
    assert not missing and not unexpected
    return model


def tiny_forward(model, dense: list[np.ndarray], ids: list[int]):
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long)
        out = model(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states[0] is the scaled embedding, [i] the output of
        # layer i-1, and the last entry is after the final norm (it
        # replaces, not follows, the last layer's raw output).
        hs = torch.stack([h[0] for h in out.hidden_states]).numpy()
        h = out.last_hidden_state[0]
        pooled = h.mean(dim=0, keepdim=True)
        for w in dense:
            pooled = torch.nn.functional.linear(pooled, torch.from_numpy(w))
        emb = torch.nn.functional.normalize(pooled, p=2, dim=1)[0].numpy()
    return hs.astype(np.float32), emb.astype(np.float32)


def write_st_layout(root: Path, config: dict, dense_dims) -> None:
    dump_json(root / "config.json", config)
    dump_json(
        root / "modules.json",
        [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
            {"idx": 3, "name": "3", "path": "3_Dense", "type": "sentence_transformers.models.Dense"},
            {"idx": 4, "name": "4", "path": "4_Normalize", "type": "sentence_transformers.models.Normalize"},
        ],
    )
    dump_json(root / "sentence_bert_config.json", {"max_seq_length": 32, "do_lower_case": False})
    dump_json(
        root / "config_sentence_transformers.json",
        {"prompts": PROMPTS, "default_prompt_name": None, "similarity_fn_name": "cosine"},
    )
    hidden = config["hidden_size"]
    dump_json(
        root / "1_Pooling" / "config.json",
        {
            "word_embedding_dimension": hidden,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": False,
            "include_prompt": True,
        },
    )
    for i, (din, dout) in enumerate(dense_dims):
        dump_json(
            root / f"{i + 2}_Dense" / "config.json",
            {
                "in_features": din,
                "out_features": dout,
                "bias": False,
                "activation_function": "torch.nn.modules.linear.Identity",
            },
        )


def gen_tiny() -> None:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    from transformers import Gemma3TextConfig, Gemma3TextModel

    cfg = Gemma3TextConfig(**TINY)
    shapes = {k: tuple(v.shape) for k, v in Gemma3TextModel(cfg).state_dict().items()}
    state = {}
    for name, shape in sorted(shapes.items()):
        if name.endswith("norm.weight"):
            # Centered on 0 like trained Gemma norms, wide enough that a
            # missing `1 +` changes every output.
            state[name] = rng.normal(0.0, 0.5, shape).astype(np.float32)
        elif name == "embed_tokens.weight":
            state[name] = rng.normal(0.0, 0.3, shape).astype(np.float32)
        else:
            fan_in = shape[-1]
            state[name] = rng.normal(0.0, 1.0 / math.sqrt(fan_in), shape).astype(np.float32)
    dense = [
        rng.normal(0.0, 1.0 / math.sqrt(din), (dout, din)).astype(np.float32)
        for din, dout in TINY_DENSE
    ]

    config = {"architectures": ["Gemma3TextModel"], "model_type": "gemma3_text", **TINY}

    # Sentence-transformers layout, f32.
    hf = OUT / "tiny_hf"
    write_st_layout(hf, config, TINY_DENSE)
    save_file(state, str(hf / "model.safetensors"), metadata={"format": "pt"})
    for i, w in enumerate(dense):
        save_file({"linear.weight": w}, str(hf / f"{i + 2}_Dense" / "model.safetensors"))

    # MLX layout: `model.` prefix, Dense head inside the main file as
    # dense.{i}, every linear and the embedding table quantized, norms F16.
    mlx = OUT / "tiny_mlx_q4"
    mlx_config = dict(config)
    mlx_config["quantization"] = {"group_size": GROUP, "bits": BITS}
    write_st_layout(mlx, mlx_config, TINY_DENSE)
    # The real MLX conversions ship modules.json but no module dirs.
    for sub in ["1_Pooling", "2_Dense", "3_Dense"]:
        for f in (mlx / sub).iterdir():
            f.unlink()
        (mlx / sub).rmdir()
    mlx_tensors = {}
    deq_state = {}
    for name, w in state.items():
        key = "model." + name
        if w.ndim == 2 and w.shape[1] % GROUP == 0:
            packed, s, b, deq = mlx_quantize(w)
            base = key[: -len(".weight")]
            mlx_tensors[key] = packed
            mlx_tensors[base + ".scales"] = s
            mlx_tensors[base + ".biases"] = b
            deq_state[name] = deq
        else:
            mlx_tensors[key] = w.astype(np.float16)
            deq_state[name] = w.astype(np.float16).astype(np.float32)
    deq_dense = []
    for i, w in enumerate(dense):
        packed, s, b, deq = mlx_quantize(w)
        mlx_tensors[f"dense.{i}.weight"] = packed
        mlx_tensors[f"dense.{i}.scales"] = s
        mlx_tensors[f"dense.{i}.biases"] = b
        deq_dense.append(deq)
    save_file(mlx_tensors, str(mlx / "model.safetensors"), metadata={"format": "mlx"})

    cases = []
    for variant, st, dn in [("hf", state, dense), ("q4", deq_state, deq_dense)]:
        model = build_tiny_model(st)
        for i, ids in enumerate(TINY_CASES):
            hs, emb = tiny_forward(model, dn, ids)
            save_npy(f"tiny_{variant}_hidden_{i}.npy", hs)
            save_npy(f"tiny_{variant}_embed_{i}.npy", emb)
    for i, ids in enumerate(TINY_CASES):
        cases.append({"index": i, "ids": ids})
    dump_json(OUT / "tiny_cases.json", {"cases": cases, "config": TINY, "dense": TINY_DENSE})
    print(f"tiny fixtures written: {len(TINY_CASES)} cases x 2 variants")


# ---------------------------------------------------------------- real


def long_text(sentences: int) -> str:
    """Deterministic varied English prose: no repeated sentence, so the
    sliding window sees distinct content at every offset."""
    subjects = ["The river", "A small robot", "Every library", "The old bridge", "Our neighbor",
                "This algorithm", "The orchestra", "A distant star", "The committee", "My grandmother"]
    verbs = ["quietly measured", "carefully described", "suddenly remembered", "slowly rebuilt",
             "happily painted", "patiently explained", "rarely visited", "proudly announced"]
    objects = ["the northern valley", "seventeen blue lanterns", "an unfinished symphony",
               "the price of copper", "a forgotten recipe", "the winter schedule",
               "three hundred paper cranes", "the theory of tides", "a broken compass"]
    tails = ["before sunrise.", "after the long storm.", "during the festival.",
             "without telling anyone.", "for the second time.", "in complete silence.",
             "with remarkable precision."]
    out = []
    for i in range(sentences):
        out.append(
            f"{subjects[i % 10]} {verbs[(i * 3) % 8]} {objects[(i * 7) % 9]} "
            f"{tails[(i * 5) % 7]} (note {i})"
        )
    return " ".join(out)


def real_cases():
    return [
        {"prompt_name": "query", "text": "Which planet is known as the Red Planet?"},
        {"prompt_name": "document",
         "text": "Mars, known for its reddish appearance, is often referred to as the Red Planet."},
        {"prompt_name": None, "text": "  \tThe quick brown fox jumps over the lazy dog.\n  "},
        {"prompt_name": "document",
         "text": "Der schnelle braune Fuchs springt \u00fcber den faulen Hund. "
                 "\u6771\u4eac\u306f\u65e5\u672c\u306e\u9996\u90fd\u3067\u3059\u3002 "
                 "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! caf\u00e9 na\u00efve"},
        {"prompt_name": "STS", "text": "A man is playing a guitar on stage."},
        {"prompt_name": "query", "text": long_text(36)},
        {"prompt_name": None, "text": long_text(150)},
    ]


def gen_real(model_dir: Path) -> None:
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    st = SentenceTransformer(str(model_dir), device="cpu")
    st.eval()
    assert st[0].auto_model.config._attn_implementation == "sdpa"
    cases = real_cases()
    embeds = []
    for case in cases:
        prompt = PROMPTS.get(case["prompt_name"], "") if case["prompt_name"] else ""
        # Same tokenize call encode makes (it strips prompt + text).
        feats = st.tokenize([prompt + case["text"]])
        case["ids"] = feats["input_ids"][0].tolist()
        kwargs = {"prompt_name": case["prompt_name"]} if case["prompt_name"] else {}
        emb = st.encode([case["text"]], batch_size=1, convert_to_numpy=True, **kwargs)[0]
        embeds.append(emb.astype(np.float32))
        print(f"case {len(embeds) - 1}: {len(case['ids'])} tokens, prompt {case['prompt_name']}")
    lens = [len(c["ids"]) for c in cases]
    assert any(512 < n < 2048 for n in lens), lens
    assert max(lens) == st.max_seq_length == 2048, lens
    assert cases[-1]["ids"][-1] == 1, "EOS should survive truncation"

    # Matryoshka: truncate then renormalize (normalize_embeddings=True).
    mrl = st.encode([cases[0]["text"]], prompt_name="query", truncate_dim=256,
                    normalize_embeddings=True, convert_to_numpy=True)[0]

    # Hidden-state trace for case 0, and a cache on/off cross-check on
    # the > 512 case (sentence-transformers runs with the config's
    # use_cache=True; the sliding cache must not truncate the prefill).
    model = st[0].auto_model
    with torch.no_grad():
        x = torch.tensor([cases[0]["ids"]])
        out = model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True)
        trace = torch.stack(
            [out.hidden_states[0][0], out.hidden_states[1][0], out.last_hidden_state[0]]
        ).numpy().astype(np.float32)
        long_ids = torch.tensor([cases[5]["ids"]])
        a = model(input_ids=long_ids, attention_mask=torch.ones_like(long_ids), use_cache=True)
        b = model(input_ids=long_ids, attention_mask=torch.ones_like(long_ids), use_cache=False)
        cache_diff = float((a.last_hidden_state - b.last_hidden_state).abs().max())
    print(f"use_cache on/off max abs diff on {long_ids.shape[1]} tokens: {cache_diff}")
    assert cache_diff == 0.0

    save_npy("real_embed.npy", np.stack(embeds))
    save_npy("real_embed_mrl256.npy", mrl.astype(np.float32))
    save_npy("real_trace_case0.npy", trace)
    dump_json(
        OUT / "real_cases.json",
        {
            "repo": UPSTREAM_REPO,
            "revision": UPSTREAM_REV,
            "max_seq_length": st.max_seq_length,
            "cases": cases,
            "mrl": {"case": 0, "dim": 256},
            "trace": {"case": 0, "rows": ["hidden_states[0]", "hidden_states[1]", "last_hidden_state"]},
            "use_cache_max_abs_diff": cache_diff,
        },
    )


# ---------------------------------------------------------------- mlx


def np_dequantize(tensors: dict, base: str, bits: int, group: int) -> np.ndarray:
    """MLX affine dequantization written independently of the Rust
    loader: unpack little-endian codes from uint32 words (8- and 4-bit
    only, so no code straddles a word), then q * scale + bias in f32."""
    w = tensors[base + ".weight"]
    scales = tensors[base + ".scales"].astype(np.float32)
    biases = tensors[base + ".biases"].astype(np.float32)
    assert 32 % bits == 0, bits
    per = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    codes = np.stack([(w >> np.uint32(bits * k)) & mask for k in range(per)], axis=-1)
    codes = codes.reshape(w.shape[0], -1).astype(np.float32)
    rows, cols = codes.shape
    assert cols == scales.shape[1] * group, (base, codes.shape, scales.shape)
    g = codes.reshape(rows, cols // group, group)
    return (g * scales[..., None] + biases[..., None]).reshape(rows, cols).astype(np.float32)


def gen_mlx(model_dir: Path, mlx_dir: Path, tag: str) -> None:
    """Golden embeddings for an MLX conversion: the sentence-transformers
    pipeline of the upstream install with every weight replaced by the
    MLX tensor (dequantized in numpy, or the stored F16/BF16 value
    widened to f32). This checks the Rust MLX loader independently of
    how far the conversion's weights are from upstream."""
    from safetensors.numpy import load_file
    from sentence_transformers import SentenceTransformer

    cfg = json.loads((mlx_dir / "config.json").read_text())
    quant = cfg.get("quantization")
    tensors = load_file(str(mlx_dir / "model.safetensors"))
    if tensors and any(v.dtype == np.uint16 for v in tensors.values()):
        raise SystemExit("bf16 safetensors need torch loading; not expected for these pins")

    def weight(base: str) -> np.ndarray:
        if base + ".scales" in tensors:
            return np_dequantize(tensors, base, quant["bits"], quant["group_size"])
        return tensors[base + ".weight"].astype(np.float32)

    st = SentenceTransformer(str(model_dir), device="cpu")
    st.eval()
    backbone = st[0].auto_model
    state = {}
    for name in backbone.state_dict():
        key = "model." + name
        if name.endswith(".weight") and key[: -len(".weight")] + ".scales" in tensors:
            state[name] = torch.from_numpy(weight(key[: -len(".weight")]))
        else:
            state[name] = torch.from_numpy(tensors[key].astype(np.float32))
    backbone.load_state_dict(state, strict=True)
    with torch.no_grad():
        st[2].linear.weight.copy_(torch.from_numpy(weight("dense.0")))
        st[3].linear.weight.copy_(torch.from_numpy(weight("dense.1")))
    cases = json.loads((OUT / "real_cases.json").read_text())["cases"]
    embeds = []
    for case in cases:
        kwargs = {"prompt_name": case["prompt_name"]} if case["prompt_name"] else {}
        emb = st.encode([case["text"]], batch_size=1, convert_to_numpy=True, **kwargs)[0]
        assert st.tokenize([PROMPTS.get(case["prompt_name"], "") + case["text"]])["input_ids"][0].tolist() == case["ids"]
        embeds.append(emb.astype(np.float32))
    save_npy(f"real_mlx_{tag}_embed.npy", np.stack(embeds))
    print(f"mlx {tag}: {len(embeds)} golden embeddings from dequantized weights")


# ---------------------------------------------------------------- qat


def gen_qat(qat_dir: Path, tag: str) -> None:
    """sentence-transformers goldens for a QAT checkpoint on the real
    cases. Same tokenizer and config as upstream, so the ids must match
    the ones already recorded."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(qat_dir), device="cpu")
    st.eval()
    cases = json.loads((OUT / "real_cases.json").read_text())["cases"]
    embeds = []
    for case in cases:
        prompt = PROMPTS.get(case["prompt_name"], "") if case["prompt_name"] else ""
        assert st.tokenize([prompt + case["text"]])["input_ids"][0].tolist() == case["ids"]
        kwargs = {"prompt_name": case["prompt_name"]} if case["prompt_name"] else {}
        emb = st.encode([case["text"]], batch_size=1, convert_to_numpy=True, **kwargs)[0]
        embeds.append(emb.astype(np.float32))
    save_npy(f"real_qat_{tag}_embed.npy", np.stack(embeds))
    print(f"qat {tag}: {len(embeds)} golden embeddings")


PROVENANCE_TENSORS = [
    "embed_tokens", "layers.0.self_attn.q_proj", "layers.0.mlp.down_proj",
    "layers.11.self_attn.o_proj", "layers.23.mlp.gate_proj", "dense.0",
]


def code_offsets(mlx_dir: Path, src_dir: Path, bits: int) -> dict:
    """Distance, in quantization steps, between each stored MLX code and
    round((w - bias) / scale) of a candidate source weight. Codes made
    from that source sit near 0.25 mean (uniform rounding error)."""
    from safetensors import safe_open
    from safetensors.numpy import load_file

    q = load_file(str(mlx_dir / "model.safetensors"))
    main = safe_open(str(src_dir / "model.safetensors"), "np")
    dense = safe_open(str(src_dir / "2_Dense" / "model.safetensors"), "np")
    out = {}
    for name in PROVENANCE_TENSORS:
        if name == "dense.0":
            ref, base = dense.get_tensor("linear.weight"), "dense.0"
        else:
            ref, base = main.get_tensor(name + ".weight"), "model." + name
        w = q[base + ".weight"]
        per, mask = 32 // bits, np.uint32((1 << bits) - 1)
        codes = np.stack([(w >> np.uint32(bits * k)) & mask for k in range(per)], axis=-1)
        codes = codes.reshape(w.shape[0], -1).astype(np.float32)
        scales = q[base + ".scales"].astype(np.float32)[..., None]
        biases = q[base + ".biases"].astype(np.float32)[..., None]
        ideal = ((ref.reshape(ref.shape[0], scales.shape[1], -1) - biases) / scales).reshape(ref.shape)
        off = np.abs(ideal - codes)
        out[name] = {
            "mean": round(float(off.mean()), 4),
            "p99": round(float(np.percentile(off, 99)), 4),
            "max": round(float(off.max()), 4),
            "frac_over_half_step": float(f"{float(np.mean(off > 0.501)):.3e}"),
        }
    return out


def gen_provenance(base_dir: Path, mlx_dirs: dict, qat_dirs: dict) -> None:
    report = {}
    for tag, bits in [("8bit", 8), ("4bit", 4)]:
        if tag not in mlx_dirs or MLX_SOURCE[tag] not in qat_dirs:
            continue
        report[tag] = {
            f"qat_{MLX_SOURCE[tag]}": code_offsets(mlx_dirs[tag], qat_dirs[MLX_SOURCE[tag]], bits),
            "upstream": code_offsets(mlx_dirs[tag], base_dir, bits),
        }
    if report:
        dump_json(OUT / "mlx_provenance.json", report)
        print("mlx provenance written")


# ---------------------------------------------------------------- manifest


def write_manifest() -> None:
    files = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(OUT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "numpy", "safetensors", "transformers", "sentence-transformers", "tokenizers"]
    }
    dump_json(
        OUT / "manifest.json",
        {
            "generator": "crates/embeddings/tools/gen_embeddinggemma_fixtures.py",
            "packages": versions,
            "reference": "transformers Gemma3TextModel (sdpa, f32, CPU) + sentence-transformers",
            "pins": {
                "real": {"repo": UPSTREAM_REPO, "revision": UPSTREAM_REV},
                "mlx": {tag: {"repo": r, "revision": v} for tag, (r, v) in MLX_PINS.items()},
                "qat": {tag: {"repo": r, "revision": v} for tag, (r, v) in QAT_PINS.items()},
                "tiny": "seeded random init (numpy default_rng(0)), no download",
            },
            "files": files,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--real", type=Path)
    ap.add_argument("--mlx", nargs=2, action="append", metavar=("TAG", "MLX_DIR"), default=[],
                    help="TAG in bf16/8bit/4bit; needs --real MODEL_DIR (or a prior --real run)")
    ap.add_argument("--qat", nargs=2, action="append", metavar=("TAG", "QAT_DIR"), default=[],
                    help="TAG in q8/q4; needs a prior or concurrent --real run")
    ap.add_argument("--model-dir", type=Path, help="upstream install for --mlx without --real")
    args = ap.parse_args()
    if not args.tiny and not args.real and not args.mlx and not args.qat:
        ap.error("pass --tiny, --real MODEL_DIR, --mlx TAG MLX_DIR, and/or --qat TAG QAT_DIR")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if args.tiny:
        gen_tiny()
    if args.real:
        gen_real(args.real.expanduser())
    for tag, mlx_dir in args.mlx:
        assert tag in MLX_PINS, tag
        upstream = args.real or args.model_dir
        if upstream is None:
            ap.error("--mlx needs --real or --model-dir")
        gen_mlx(upstream.expanduser(), Path(mlx_dir).expanduser(), tag)
    for tag, qat_dir in args.qat:
        assert tag in QAT_PINS, tag
        gen_qat(Path(qat_dir).expanduser(), tag)
    upstream = args.real or args.model_dir
    if upstream is not None and args.mlx and args.qat:
        gen_provenance(
            upstream.expanduser(),
            {t: Path(d).expanduser() for t, d in args.mlx},
            {t: Path(d).expanduser() for t, d in args.qat},
        )
    write_manifest()


if __name__ == "__main__":
    main()
