#!/usr/bin/env python3
"""Regenerate the BERT / XLM-RoBERTa (Snowflake arctic-embed) Rust parity
fixtures.

Normative reference: Hugging Face transformers 4.57.1 `BertModel` and
`XLMRobertaModel` (sdpa attention, f32, CPU) plus sentence-transformers
5.1.2 for the tokenize -> Transformer -> Pooling (CLS) -> Normalize
pipeline. No trust_remote_code anywhere.

Independent parts:

--tiny (no download, deterministic, seed 0)
    Random-init 2-layer BertModel (pad 0, 24 positions, 2 token types)
    and XLMRobertaModel (pad 1, 26 positions, 1 token type), each with a
    sentence-transformers layout (modules.json, 1_Pooling CLS, Normalize,
    prompts). The XLM-R model is also written in the MLX layout (no
    1_Pooling dir, like the mlx-community repos) as affine 4-bit and
    6-bit (group 64; every 2-D tensor with a width divisible by 64,
    including the word, position, and token-type tables, the rest F16)
    and as BF16. Goldens: every hidden state and the CLS + L2 embedding
    for id sequences with and without padding (pad ids inside the
    sequence with mask 1 exercise the XLM-R position rule; right padding
    with mask 0 exercises the key mask). Quantized goldens come from HF
    on weights dequantized in numpy (continuous little-endian bit stream,
    written independently of the Rust loader).

--real TAG MODEL_DIR (repeatable; TAG in l_v2, m_v15, l, m, s, xs)
    SentenceTransformer(MODEL_DIR).encode (batch_size 1) for six texts:
    query and document prompts, padded whitespace, a multilingual
    non-ASCII query, a paragraph, and a long text that BERT truncates at
    512. Stores the ids sentence-transformers fed the model,
    the normalized embeddings, Matryoshka 256 (m_v15 and l_v2), and the
    embedding / layer-0 / last hidden state of case 0 (l_v2 and xs). For
    l_v2 also the ids of a > 8192-token text (tokenize only).

--mlx TAG MLX_DIR (repeatable; TAG in bf16, 8bit, 6bit, 4bit; needs
    --model-dir or --real l_v2 for the upstream install)
    The l_v2 sentence-transformers pipeline with every backbone tensor
    replaced by the conversion's (affine codes dequantized in numpy,
    F16/BF16 widened to f32), run on the real case texts.

    emb-venv/bin/python -I crates/embeddings/tools/gen_bert_fixtures.py --tiny
    M=~/models/embeddings
    emb-venv/bin/python -I crates/embeddings/tools/gen_bert_fixtures.py \\
        --real l_v2 $M/snowflake-arctic-embed-l-v2.0 \\
        --real m_v15 $M/snowflake-arctic-embed-m-v1.5 \\
        --real l $M/snowflake-arctic-embed-l --real m $M/snowflake-arctic-embed-m \\
        --real s $M/snowflake-arctic-embed-s --real xs $M/snowflake-arctic-embed-xs \\
        --mlx bf16 $M/mlx-snowflake-arctic-embed-l-v2.0-bf16 \\
        --mlx 8bit $M/mlx-snowflake-arctic-embed-l-v2.0-8bit \\
        --mlx 6bit $M/mlx-snowflake-arctic-embed-l-v2.0-6bit \\
        --mlx 4bit $M/mlx-snowflake-arctic-embed-l-v2.0-4bit

Outputs under crates/embeddings/testdata/bert/ plus manifest.json
(generator, package versions, pins, sha256 per file). Each part rewrites
only its own files and then rebuilds the manifest over the directory.
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

OUT = Path(__file__).resolve().parent.parent / "testdata" / "bert"

UPSTREAM_PINS = {
    "l_v2": ("Snowflake/snowflake-arctic-embed-l-v2.0", "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"),
    "m_v15": ("Snowflake/snowflake-arctic-embed-m-v1.5", "e58a8f756156a1293d763f17e3aae643474e9b8a"),
    "l": ("Snowflake/snowflake-arctic-embed-l", "d8fb21ca8d905d2832ee8b96c894d3298964346b"),
    "m": ("Snowflake/snowflake-arctic-embed-m", "fc74610d18462d218e312aa986ec5c8a75a98152"),
    "s": ("Snowflake/snowflake-arctic-embed-s", "e596f507467533e48a2e17c007f0e1dacc837b33"),
    "xs": ("Snowflake/snowflake-arctic-embed-xs", "d8c86521100d3556476a063fc2342036d45c106f"),
}
MLX_PINS = {
    "bf16": ("mlx-community/snowflake-arctic-embed-l-v2.0-bf16", "3da10e38e7875d6501adc14665c95b6708646121"),
    "8bit": ("mlx-community/snowflake-arctic-embed-l-v2.0-8bit", "98033f618b7503c73aa857bc0656f763a1587e73"),
    "6bit": ("mlx-community/snowflake-arctic-embed-l-v2.0-6bit", "c8125c01b2142f0065f2494c123a7df46f9ef08e"),
    "4bit": ("mlx-community/snowflake-arctic-embed-l-v2.0-4bit", "430b52bb3dcb9ad2404b1ac2b60f2fdc00f26033"),
}
MRL_TAGS = {"m_v15", "l_v2"}
TRACE_TAGS = {"l_v2", "xs"}
GROUP = 64


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=True) + "\n")


def save_npy(name: str, arr: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / name, np.ascontiguousarray(arr.astype(np.float32)))


# ---------------------------------------------------------------- mlx codec


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """[rows, cols] codes < 2^bits -> [rows, cols * bits / 32] uint32 as one
    continuous little-endian bit stream per row (value i occupies bits
    i*bits .. i*bits+bits-1, bit 0 = LSB of byte 0). 6-bit values cross
    word boundaries."""
    rows, cols = codes.shape
    assert (cols * bits) % 32 == 0
    shifts = np.arange(bits, dtype=np.uint32)
    stream = ((codes[..., None].astype(np.uint32) >> shifts) & 1).astype(np.uint8)
    stream = stream.reshape(rows, cols * bits)
    packed = np.packbits(stream, axis=1, bitorder="little")
    return packed.view("<u4").reshape(rows, cols * bits // 32)


def unpack_codes(words: np.ndarray, bits: int, cols: int) -> np.ndarray:
    """Inverse of pack_codes, written against the same bit-stream
    definition (not shared code with the Rust loader)."""
    rows = words.shape[0]
    raw = np.ascontiguousarray(words.astype("<u4")).view(np.uint8).reshape(rows, -1)
    stream = np.unpackbits(raw, axis=1, bitorder="little")[:, : cols * bits]
    stream = stream.reshape(rows, cols, bits).astype(np.uint32)
    return (stream << np.arange(bits, dtype=np.uint32)).sum(axis=-1).astype(np.uint32)


def mlx_quantize(w: np.ndarray, bits: int):
    """MLX affine: per group of 64 along the last axis, w ~ q * scale +
    bias, q in [0, 2^bits). Scales and biases stored F16 (as in the
    mlx-community arctic repos). Returns (packed, scales, biases, exact
    f32 dequantization)."""
    rows, cols = w.shape
    g = w.reshape(rows, cols // GROUP, GROUP).astype(np.float32)
    lo, hi = g.min(axis=-1), g.max(axis=-1)
    levels = (1 << bits) - 1
    scale = np.maximum((hi - lo) / levels, 1e-6).astype(np.float16)
    bias = lo.astype(np.float16)
    s32 = scale.astype(np.float32)[..., None]
    b32 = bias.astype(np.float32)[..., None]
    q = np.clip(np.rint((g - b32) / s32), 0, levels).astype(np.uint32).reshape(rows, cols)
    packed = pack_codes(q, bits)
    assert np.array_equal(unpack_codes(packed, bits, cols), q)
    deq = (q.reshape(rows, cols // GROUP, GROUP).astype(np.float32) * s32 + b32).reshape(rows, cols)
    return packed, scale, bias, deq.astype(np.float32)


def np_dequantize(tensors: dict, base: str, bits: int) -> np.ndarray:
    w = tensors[base + ".weight"]
    scales = tensors[base + ".scales"].astype(np.float32)
    biases = tensors[base + ".biases"].astype(np.float32)
    rows, groups = scales.shape
    cols = groups * GROUP
    assert w.shape == (rows, cols * bits // 32), (base, w.shape, scales.shape)
    codes = unpack_codes(w, bits, cols).astype(np.float32)
    g = codes.reshape(rows, groups, GROUP)
    return (g * scales[..., None] + biases[..., None]).reshape(rows, cols).astype(np.float32)


# ---------------------------------------------------------------- tiny


TINY_BERT = dict(
    vocab_size=96, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    intermediate_size=128, hidden_act="gelu", max_position_embeddings=24,
    type_vocab_size=2, layer_norm_eps=1e-12, pad_token_id=0,
    position_embedding_type="absolute",
)
TINY_XLMR = dict(
    vocab_size=96, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    intermediate_size=128, hidden_act="gelu", max_position_embeddings=26,
    type_vocab_size=1, layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0,
    eos_token_id=2, position_embedding_type="absolute",
)
# (ids, mask). BERT pad is 0; XLM-R pad is 1. Pad ids with mask 1 test
# the position rule; trailing pads with mask 0 test the key mask. The
# 24-token XLM-R case reaches the last position row (1 + 24 = 25).
TINY_CASES = {
    "bert": [
        ([2, 7, 3], None),
        ([2, 11, 5, 9, 40, 3], None),
        ([2] + [(5 * i + 3) % 90 + 4 for i in range(22)] + [3], None),
        ([2, 17, 23, 3, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]),
        ([2, 0, 15, 0, 3], None),
    ],
    "xlmr": [
        ([0, 7, 2], None),
        ([0, 11, 5, 9, 40, 2], None),
        ([0] + [(7 * i + 5) % 90 + 4 for i in range(22)] + [2], None),
        ([0, 17, 23, 2, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]),
        ([0, 1, 15, 1, 1, 30, 2], None),
    ],
}
TINY_PROMPTS = {"query": "query: "}


def tiny_state(model_cls, cfg_cls, cfg: dict, rng) -> dict:
    shapes = {k: tuple(v.shape) for k, v in model_cls(cfg_cls(**cfg), add_pooling_layer=False).state_dict().items()}
    state = {}
    for name, shape in sorted(shapes.items()):
        if "LayerNorm.weight" in name:
            state[name] = (1.0 + rng.normal(0.0, 0.2, shape)).astype(np.float32)
        elif name.endswith(".bias"):
            state[name] = rng.normal(0.0, 0.1, shape).astype(np.float32)
        elif "embeddings" in name:
            state[name] = rng.normal(0.0, 0.5, shape).astype(np.float32)
        else:
            state[name] = rng.normal(0.0, 1.0 / math.sqrt(shape[-1]), shape).astype(np.float32)
    return state


def build(model_cls, cfg_cls, cfg: dict, state: dict):
    c = cfg_cls(**cfg)
    c._attn_implementation = "sdpa"
    model = model_cls(c, add_pooling_layer=False).eval()
    missing, unexpected = model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
    assert not missing and not unexpected
    return model


def tiny_forward(model, ids, mask):
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long)
        m = torch.tensor([mask if mask else [1] * len(ids)], dtype=torch.long)
        out = model(input_ids=x, attention_mask=m, output_hidden_states=True)
        hs = torch.stack([h[0] for h in out.hidden_states]).numpy()
        cls = out.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(cls, p=2, dim=1)[0].numpy()
    return hs, emb


def write_st_layout(root: Path, config: dict, max_seq_length, tokenizer_max) -> None:
    dump_json(root / "config.json", config)
    dump_json(
        root / "modules.json",
        [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ],
    )
    if max_seq_length is not None:
        dump_json(root / "sentence_bert_config.json",
                  {"max_seq_length": max_seq_length, "do_lower_case": False,
                   "model_args": {"add_pooling_layer": False}})
    if tokenizer_max is not None:
        dump_json(root / "tokenizer_config.json", {"model_max_length": tokenizer_max})
    dump_json(root / "config_sentence_transformers.json",
              {"prompts": TINY_PROMPTS, "default_prompt_name": None})
    dump_json(
        root / "1_Pooling" / "config.json",
        {
            "word_embedding_dimension": config["hidden_size"],
            "pooling_mode_cls_token": True,
            "pooling_mode_mean_tokens": False,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": False,
            "include_prompt": True,
        },
    )


def gen_tiny() -> None:
    from safetensors.numpy import save_file
    from safetensors.torch import save_file as save_torch
    from transformers import BertConfig, BertModel, XLMRobertaConfig, XLMRobertaModel

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    specs = {
        "bert": (BertModel, BertConfig, TINY_BERT, "bert", "BertModel", 16, None),
        # No sentence_bert_config: max_seq_length falls back to
        # min(max_position_embeddings, tokenizer model_max_length).
        "xlmr": (XLMRobertaModel, XLMRobertaConfig, TINY_XLMR, "xlm-roberta", "XLMRobertaModel", None, 20),
    }
    states = {}
    variants = {}
    for tag, (mcls, ccls, cfg, mtype, arch, msl, tmax) in specs.items():
        state = tiny_state(mcls, ccls, cfg, rng)
        states[tag] = state
        config = {"architectures": [arch], "model_type": mtype, **cfg}
        root = OUT / f"tiny_{tag}"
        write_st_layout(root, config, msl, tmax)
        save_file(state, str(root / "model.safetensors"), metadata={"format": "pt"})
        variants[tag] = (mcls, ccls, cfg, state, tag)

    # MLX layouts of the XLM-R model.
    xstate = states["xlmr"]
    xconfig = {"architectures": ["XLMRobertaModel"], "model_type": "xlm-roberta", **TINY_XLMR}
    for bits in (4, 6):
        root = OUT / f"tiny_xlmr_mlx_q{bits}"
        cfg = dict(xconfig, quantization={"group_size": GROUP, "bits": bits})
        write_st_layout(root, cfg, None, 20)
        for f in (root / "1_Pooling").iterdir():
            f.unlink()
        (root / "1_Pooling").rmdir()
        tensors, deq = {}, {}
        for name, w in xstate.items():
            if w.ndim == 2 and w.shape[1] % GROUP == 0:
                base = name[: -len(".weight")]
                packed, s, b, d = mlx_quantize(w, bits)
                tensors[name] = packed
                tensors[base + ".scales"] = s
                tensors[base + ".biases"] = b
                deq[name] = d
            else:
                tensors[name] = w.astype(np.float16)
                deq[name] = w.astype(np.float16).astype(np.float32)
        # Independent read-back through np_dequantize, as the real goldens do.
        for name in deq:
            if name.endswith(".weight") and name[: -len(".weight")] + ".scales" in tensors:
                assert np.array_equal(np_dequantize(tensors, name[: -len(".weight")], bits), deq[name])
        save_file(tensors, str(root / "model.safetensors"), metadata={"format": "mlx"})
        variants[f"q{bits}"] = (XLMRobertaModel, XLMRobertaConfig, TINY_XLMR, deq, "xlmr")
    root = OUT / "tiny_xlmr_mlx_bf16"
    write_st_layout(root, xconfig, None, 20)
    for f in (root / "1_Pooling").iterdir():
        f.unlink()
    (root / "1_Pooling").rmdir()
    bf = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in xstate.items()}
    save_torch(bf, str(root / "model.safetensors"), metadata={"format": "mlx"})
    variants["bf16"] = (XLMRobertaModel, XLMRobertaConfig, TINY_XLMR,
                        {k: v.float().numpy() for k, v in bf.items()}, "xlmr")

    for variant, (mcls, ccls, cfg, state, cases) in variants.items():
        model = build(mcls, ccls, cfg, state)
        for i, (ids, mask) in enumerate(TINY_CASES[cases]):
            hs, emb = tiny_forward(model, ids, mask)
            save_npy(f"tiny_{variant}_hidden_{i}.npy", hs)
            save_npy(f"tiny_{variant}_embed_{i}.npy", emb)
    dump_json(
        OUT / "tiny_cases.json",
        {
            "cases": {k: [{"ids": ids, "mask": mask} for ids, mask in v] for k, v in TINY_CASES.items()},
            "variants": {k: v[4] for k, v in variants.items()},
            "config": {"bert": TINY_BERT, "xlmr": TINY_XLMR},
        },
    )
    print(f"tiny fixtures written: {len(variants)} variants")


# ---------------------------------------------------------------- real


def long_text(sentences: int) -> str:
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
    return " ".join(
        f"{subjects[i % 10]} {verbs[(i * 3) % 8]} {objects[(i * 7) % 9]} {tails[(i * 5) % 7]} (note {i})"
        for i in range(sentences)
    )


def real_cases():
    return [
        {"prompt_name": "query", "text": "What is a cloud data warehouse?"},
        {"prompt_name": None,
         "text": "Snowflake is a cloud data platform that separates storage from compute."},
        {"prompt_name": None, "text": "  \tThe quick brown fox jumps over the lazy dog.\n  "},
        {"prompt_name": "query",
         "text": "Wie hei\u00dft die Hauptstadt von Japan? "
                 "\u6771\u4eac\u306f\u65e5\u672c\u306e\u9996\u90fd\u3067\u3059\u3002 "
                 "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! caf\u00e9 na\u00efve "
                 "\u4f60\u597d \u0645\u0631\u062d\u0628\u0627"},
        {"prompt_name": "document",
         "text": "Arctic-embed models are trained in two stages: large-scale contrastive "
                 "pretraining on query-document pairs, followed by fine-tuning on a smaller "
                 "set of curated examples with mined hard negatives."},
        {"prompt_name": None, "text": long_text(48)},
    ]


def encode_ids(st, case) -> list[int]:
    prompt = st.prompts.get(case["prompt_name"], "") if case["prompt_name"] else ""
    # Same tokenize call encode makes (Transformer.tokenize strips).
    return st.tokenize([prompt + case["text"]])["input_ids"][0].tolist()


def encode(st, case) -> np.ndarray:
    kwargs = {"prompt_name": case["prompt_name"]} if case["prompt_name"] else {}
    return st.encode([case["text"]], batch_size=1, convert_to_numpy=True, **kwargs)[0].astype(np.float32)


def load_cases_file() -> dict:
    path = OUT / "real_cases.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"cases": real_cases(), "checkpoints": {}}


def gen_real(tag: str, model_dir: Path, doc: dict) -> None:
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    st = SentenceTransformer(str(model_dir), device="cpu")
    st.eval()
    model = st[0].auto_model
    assert model.config._attn_implementation == "sdpa"
    assert [type(m).__name__ for m in st] == ["Transformer", "Pooling", "Normalize"]
    assert st[1].pooling_mode_cls_token and st[1].get_pooling_mode_str() == "cls"
    cases = doc["cases"]
    ids, embeds = [], []
    for case in cases:
        ids.append(encode_ids(st, case))
        embeds.append(encode(st, case))
    lens = [len(x) for x in ids]
    print(f"{tag}: lengths {lens}, max_seq_length {st.max_seq_length}")
    if st.max_seq_length == 512:
        assert lens[-1] == 512, lens
        assert ids[-1][-1] == st.tokenizer.sep_token_id
    entry = {
        "repo": UPSTREAM_PINS[tag][0],
        "revision": UPSTREAM_PINS[tag][1],
        "max_seq_length": st.max_seq_length,
        "prompts": st.prompts,
        "ids": ids,
    }
    save_npy(f"real_{tag}_embed.npy", np.stack(embeds))
    if tag in MRL_TAGS:
        mrl = st.encode([cases[0]["text"]], prompt_name="query", truncate_dim=256,
                        normalize_embeddings=True, convert_to_numpy=True)[0]
        save_npy(f"real_{tag}_mrl256.npy", mrl)
    if tag in TRACE_TAGS:
        with torch.no_grad():
            x = torch.tensor([ids[0]])
            out = model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True)
            trace = torch.stack([out.hidden_states[0][0], out.hidden_states[1][0], out.last_hidden_state[0]])
        save_npy(f"real_{tag}_trace_case0.npy", trace.numpy())
    if tag == "l_v2":
        # Tokenize-only check of the 8192 cut (running 8192 tokens through
        # the Rust CPU port in a test would take minutes).
        text = long_text(700)
        long_ids = st.tokenize([text])["input_ids"][0].tolist()
        full = st.tokenizer(text)["input_ids"]
        assert len(long_ids) == st.max_seq_length == 8192 and len(full) > 8192, (len(long_ids), len(full))
        assert long_ids[-1] == 2 and long_ids[0] == 0
        entry["long_tokenize"] = {"sentences": 700, "untruncated_len": len(full), "ids": long_ids}
    doc["checkpoints"][tag] = entry


def gen_mlx(model_dir: Path, mlx_dir: Path, tag: str, doc: dict) -> None:
    from safetensors import safe_open
    from sentence_transformers import SentenceTransformer

    cfg = json.loads((mlx_dir / "config.json").read_text())
    quant = cfg.get("quantization")
    # numpy has no bfloat16: read BF16 through torch and widen (exact),
    # everything else (U32 codes, F16) through numpy.
    np_t, dtypes = {}, set()
    with safe_open(str(mlx_dir / "model.safetensors"), "pt") as fp, \
            safe_open(str(mlx_dir / "model.safetensors"), "np") as fn:
        for k in fp.keys():
            dt = fp.get_slice(k).get_dtype()
            dtypes.add(dt)
            np_t[k] = fp.get_tensor(k).float().numpy() if dt == "BF16" else fn.get_tensor(k)
    st = SentenceTransformer(str(model_dir), device="cpu")
    st.eval()
    backbone = st[0].auto_model
    state = {}
    for name in backbone.state_dict():
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        if base is not None and base + ".scales" in np_t:
            state[name] = torch.from_numpy(np_dequantize(np_t, base, quant["bits"]))
        else:
            state[name] = torch.from_numpy(np_t[name].astype(np.float32))
    # Provenance and unpack sanity: distance in quantization steps between
    # each code and round((w_upstream - bias) / scale). A wrong bit-stream
    # decode or a different source puts the mean far from 0.25.
    offsets = {}
    if quant:
        up = backbone.state_dict()
        for base in ["embeddings.word_embeddings", "embeddings.position_embeddings",
                     "encoder.layer.0.attention.self.query", "encoder.layer.23.output.dense"]:
            w = up[base + ".weight"].numpy()
            scales = np_t[base + ".scales"].astype(np.float32)[..., None]
            biases = np_t[base + ".biases"].astype(np.float32)[..., None]
            codes = unpack_codes(np_t[base + ".weight"], quant["bits"], w.shape[1]).astype(np.float32)
            ideal = (w.reshape(w.shape[0], scales.shape[1], -1) - biases) / scales
            off = np.abs(ideal.reshape(w.shape) - codes)
            offsets[base] = {"mean": round(float(off.mean()), 4), "p99": round(float(np.percentile(off, 99)), 4),
                             "max": round(float(off.max()), 4)}
            assert off.mean() < 0.3, (base, offsets[base])
    backbone.load_state_dict(state, strict=True)
    ref_ids = doc["checkpoints"]["l_v2"]["ids"]
    embeds = []
    for i, case in enumerate(doc["cases"]):
        assert encode_ids(st, case) == ref_ids[i]
        embeds.append(encode(st, case))
    save_npy(f"real_mlx_{tag}_embed.npy", np.stack(embeds))
    dtypes = sorted(dtypes)
    doc.setdefault("mlx", {})[tag] = {
        "repo": MLX_PINS[tag][0], "revision": MLX_PINS[tag][1],
        "quantization": quant, "dtypes": dtypes, "code_offsets_vs_upstream": offsets,
    }
    print(f"mlx {tag}: {len(embeds)} golden embeddings, dtypes {dtypes}")


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
            "generator": "crates/embeddings/tools/gen_bert_fixtures.py",
            "packages": versions,
            "reference": "transformers BertModel / XLMRobertaModel (sdpa, f32, CPU) + sentence-transformers",
            "pins": {
                "real": {t: {"repo": r, "revision": v} for t, (r, v) in UPSTREAM_PINS.items()},
                "mlx": {t: {"repo": r, "revision": v, "converted_from": UPSTREAM_PINS["l_v2"][0]}
                        for t, (r, v) in MLX_PINS.items()},
                "tiny": "seeded random init (numpy default_rng(0)), no download",
            },
            "files": files,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--real", nargs=2, action="append", metavar=("TAG", "MODEL_DIR"), default=[])
    ap.add_argument("--mlx", nargs=2, action="append", metavar=("TAG", "MLX_DIR"), default=[])
    ap.add_argument("--model-dir", type=Path, help="l_v2 upstream install for --mlx without --real l_v2")
    args = ap.parse_args()
    if not args.tiny and not args.real and not args.mlx:
        ap.error("pass --tiny, --real TAG DIR, and/or --mlx TAG DIR")
    if args.tiny:
        gen_tiny()
    if args.real or args.mlx:
        doc = load_cases_file()
        if args.real and "checkpoints" in doc and doc["cases"] != real_cases():
            doc = {"cases": real_cases(), "checkpoints": {}}
        for tag, d in args.real:
            assert tag in UPSTREAM_PINS, tag
            gen_real(tag, Path(d).expanduser(), doc)
        upstream = dict(args.real).get("l_v2") or args.model_dir
        for tag, d in args.mlx:
            assert tag in MLX_PINS, tag
            if upstream is None:
                ap.error("--mlx needs --real l_v2 DIR or --model-dir")
            gen_mlx(Path(upstream).expanduser(), Path(d).expanduser(), tag, doc)
        dump_json(OUT / "real_cases.json", doc)
    write_manifest()


if __name__ == "__main__":
    main()
