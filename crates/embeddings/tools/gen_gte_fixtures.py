#!/usr/bin/env python3
"""Regenerate the GTE (snowflake-arctic-embed-m-v2.0) Rust parity fixtures.

Normative reference: the remote modeling code pinned with the checkpoint
(`modeling_hf_alibaba_nlp_gte.py`, `configuration_hf_alibaba_nlp_gte.py`)
run under transformers 4.57.1, f32, CPU, plus sentence-transformers 5.1.2
for tokenize -> GteModel -> Pooling (CLS) -> Normalize.

The remote code is imported only from the local pinned directory passed
as MODEL_DIR (trust_remote_code). HF_HUB_OFFLINE is forced so nothing is
fetched. `pytorch_model.bin` is never read (the pin ships safetensors).

The pinned config sets `use_memory_efficient_attention` and
`unpad_inputs` to the string "true", and GteAttention asserts that
xformers is installed when the first is truthy. xformers does not build
on macOS, so every load here passes both as False (the workaround the
model card gives for CPU). They select a kernel and a packed layout, not
different math; the --tiny part measures the unpadded path against the
padded one to show it.

--tiny MODEL_DIR (deterministic, numpy default_rng(0))
    A random-init 3-layer GteModel built from the remote classes (hidden
    64, 4 heads of 16, gated MLP 96, rope_theta 160000, type_vocab 1).
    Goldens: the embedding-LayerNorm output and every layer output, plus
    the CLS + L2 embedding, per case. Cross-checks recorded in
    tiny_cases.json: eager vs sdpa attention, and unpad_inputs=True.

--real MODEL_DIR (the pinned install)
    SentenceTransformer(MODEL_DIR, trust_remote_code=True).encode on a
    fixed text set (query prompt, multilingual, whitespace strip, ~512
    and ~2000 tokens), the token ids it fed the model, a Matryoshka 256
    embedding, a hidden-state trace of case 0, the rotary tables, and a
    > 8192-token text whose ids only are stored (max_seq_length cut).

    emb-venv/bin/python -I crates/embeddings/tools/gen_gte_fixtures.py \
        --tiny ~/models/embeddings/snowflake-arctic-embed-m-v2.0 \
        --real ~/models/embeddings/snowflake-arctic-embed-m-v2.0

Outputs under crates/embeddings/testdata/gte/ plus manifest.json
(package versions, pin, sha256 of the remote code used, sha256 per
file). Each part rewrites only its own files.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "testdata" / "gte"

REPO = "Snowflake/snowflake-arctic-embed-m-v2.0"
REV = "95c2741480856aa9666782eb4afe11959938017f"
REMOTE_FILES = ["modeling_hf_alibaba_nlp_gte.py", "configuration_hf_alibaba_nlp_gte.py"]
CONFIG_KWARGS = {"use_memory_efficient_attention": False, "unpad_inputs": False}
PROMPTS = {"query": "query: "}


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def save_npy(name: str, arr: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / name, np.ascontiguousarray(arr))


def check_remote(model_dir: Path) -> None:
    """Refuse an auto_map that points outside the local directory
    (`repo--module.Class`), which would make transformers fetch code."""
    cfg = json.loads((model_dir / "config.json").read_text())
    for k, v in cfg.get("auto_map", {}).items():
        assert "--" not in v, f"auto_map {k} = {v} names another repository"
        assert (model_dir / (v.split(".")[0] + ".py")).is_file(), v


def remote(model_dir: Path, name: str):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    check_remote(model_dir)
    return get_class_from_dynamic_module(name, str(model_dir))


def remote_hashes(model_dir: Path) -> dict:
    return {f: hashlib.sha256((model_dir / f).read_bytes()).hexdigest() for f in REMOTE_FILES}


# ---------------------------------------------------------------- tiny

TINY = dict(
    vocab_size=96,
    hidden_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    intermediate_size=96,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.0,
    max_position_embeddings=64,
    type_vocab_size=1,
    layer_norm_type="layer_norm",
    layer_norm_eps=1e-12,
    position_embedding_type="rope",
    rope_theta=160000,
    rope_scaling=None,
    pack_qkv=True,
    unpad_inputs=False,
    use_memory_efficient_attention=False,
    logn_attention_scale=False,
    logn_attention_clip1=False,
    pad_token_id=1,
    matryoshka_dimensions=[16],
)
# <s> = 0 and </s> = 2 like the XLM-R vocabulary; pad id 1 appears as an
# ordinary token in one case (no mask: one unpadded sequence).
TINY_CASES = [
    [0, 2],
    [0, 17, 2],
    [0, 5, 9, 33, 71, 1, 2],
    [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2],
    [0] + [int(v) for v in np.random.default_rng(7).integers(3, 96, 38)] + [2],
]


def tiny_state(model_cls, cfg_cls, rng) -> dict:
    shapes = {k: tuple(v.shape) for k, v in model_cls(cfg_cls(**TINY)).state_dict().items()}
    state = {}
    for name, shape in sorted(shapes.items()):
        if "LayerNorm" in name or "_ln." in name:
            if name.endswith("weight"):
                state[name] = rng.normal(1.0, 0.3, shape).astype(np.float32)
            else:
                state[name] = rng.normal(0.0, 0.2, shape).astype(np.float32)
        elif name.endswith("bias"):
            state[name] = rng.normal(0.0, 0.2, shape).astype(np.float32)
        elif "embeddings" in name:
            state[name] = rng.normal(0.0, 0.5, shape).astype(np.float32)
        else:
            state[name] = rng.normal(0.0, 1.0 / math.sqrt(shape[-1]), shape).astype(np.float32)
    return state


def build(model_cls, cfg_cls, state: dict, **over):
    cfg = cfg_cls(**{**TINY, **{k: v for k, v in over.items() if k != "attn"}})
    if "attn" in over:
        cfg._attn_implementation = over["attn"]
    model = model_cls(cfg).eval()
    missing, unexpected = model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in state.items()}, strict=True
    )
    assert not missing and not unexpected
    return model


def traced_forward(model, ids: list[int]):
    """(states [layers + 1, seq, hidden], CLS + L2 embedding). State 0 is
    the embedding LayerNorm output, state i the output of layer i - 1."""
    caps = []
    hooks = [model.embeddings.register_forward_hook(lambda m, i, o: caps.append(o[0][0].clone()))]
    for layer in model.encoder.layer:
        hooks.append(layer.register_forward_hook(lambda m, i, o: caps.append(o[0][0].clone())))
    try:
        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long)
            out = model(input_ids=x, attention_mask=torch.ones_like(x))
            last = out.last_hidden_state[0]
            emb = torch.nn.functional.normalize(last[:1], p=2, dim=1)[0]
    finally:
        for h in hooks:
            h.remove()
    states = torch.stack(caps)
    assert torch.equal(states[-1], last)
    return states.numpy().astype(np.float32), emb.numpy().astype(np.float32)


def write_st_layout(root: Path, config: dict, max_seq_length: int | None) -> None:
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
        dump_json(root / "sentence_bert_config.json", {"max_seq_length": max_seq_length, "do_lower_case": False})
    dump_json(
        root / "config_sentence_transformers.json",
        {"prompts": PROMPTS, "default_prompt_name": None},
    )
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


def gen_tiny(model_dir: Path) -> None:
    torch.manual_seed(0)
    cfg_cls = remote(model_dir, "configuration_hf_alibaba_nlp_gte.GteConfig")
    model_cls = remote(model_dir, "modeling_hf_alibaba_nlp_gte.GteModel")
    state = tiny_state(model_cls, cfg_cls, np.random.default_rng(0))

    root = OUT / "tiny"
    config = {"architectures": ["GteModel"], "model_type": "gte", **TINY}
    # No sentence_bert_config.json, like the pinned checkpoint: the
    # loader must fall back to max_position_embeddings.
    write_st_layout(root, config, None)
    save_file(state, str(root / "model.safetensors"), metadata={"format": "pt"})

    model = build(model_cls, cfg_cls, state)
    attn = model.config._attn_implementation
    eager = build(model_cls, cfg_cls, state, attn="eager")
    assert eager.config._attn_implementation == "eager"
    unpad = build(model_cls, cfg_cls, state, unpad_inputs=True)
    cross = {"eager_vs_" + attn: 0.0, "unpad_vs_padded": 0.0}
    cases = []
    for i, ids in enumerate(TINY_CASES):
        assert len(ids) <= TINY["max_position_embeddings"]
        states, emb = traced_forward(model, ids)
        save_npy(f"tiny_hidden_{i}.npy", states)
        save_npy(f"tiny_embed_{i}.npy", emb)
        e_states, _ = traced_forward(eager, ids)
        cross["eager_vs_" + attn] = max(cross["eager_vs_" + attn], float(np.abs(e_states - states).max()))
        with torch.no_grad():
            x = torch.tensor([ids])
            u = unpad(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0].numpy()
        cross["unpad_vs_padded"] = max(cross["unpad_vs_padded"], float(np.abs(u - states[-1]).max()))
        cases.append({"index": i, "ids": ids})
    print(f"tiny cross-checks (max abs, last hidden): {cross}")
    assert all(v < 1e-5 for v in cross.values()), cross

    # Rotary reference tables for the tiny head size.
    rot = model.embeddings.rotary_emb
    dump_json(
        OUT / "tiny_cases.json",
        {"cases": cases, "config": TINY, "attn_implementation": attn, "cross_checks": cross},
    )
    save_npy("tiny_rotary_inv_freq.npy", rot.inv_freq.numpy().astype(np.float32))
    print(f"tiny fixtures written: {len(TINY_CASES)} cases")


# ---------------------------------------------------------------- real


def long_text(sentences: int, offset: int = 0) -> str:
    """Deterministic varied English prose with no repeated sentence."""
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
    for j in range(sentences):
        i = j + offset
        out.append(
            f"{subjects[i % 10]} {verbs[(i * 3) % 8]} {objects[(i * 7) % 9]} "
            f"{tails[(i * 5) % 7]} (note {i})"
        )
    return " ".join(out)


def text_with_tokens(st, target: int, offset: int) -> str:
    """Smallest long_text whose tokenization reaches `target` ids."""
    n = 1
    while len(st.tokenize([long_text(n, offset)])["input_ids"][0]) < target:
        n += 1
    return long_text(n, offset)


def real_cases(st):
    return [
        {"prompt_name": "query", "text": "Which planet is known as the Red Planet?"},
        {"prompt_name": None,
         "text": "Mars, known for its reddish appearance, is often referred to as the Red Planet."},
        {"prompt_name": None, "text": "  \tThe quick brown fox jumps over the lazy dog.\n  "},
        {"prompt_name": None,
         "text": "Der schnelle braune Fuchs springt \u00fcber den faulen Hund. "
                 "\u6771\u4eac\u306f\u65e5\u672c\u306e\u9996\u90fd\u3067\u3059\u3002 "
                 "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! caf\u00e9 na\u00efve"},
        {"prompt_name": "query", "text": "\u00bfCu\u00e1l es la capital de Francia?"},
        {"prompt_name": None,
         "text": "\u5317\u4eac\u662f\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd\u7684\u9996\u90fd\uff0c"
                 "\u4e5f\u662f\u5168\u56fd\u7684\u653f\u6cbb\u548c\u6587\u5316\u4e2d\u5fc3\u3002 "
                 "\u0627\u0644\u0642\u0627\u0647\u0631\u0629 \u0647\u064a \u0639\u0627\u0635\u0645\u0629 "
                 "\u0645\u0635\u0631."},
        {"prompt_name": None, "text": text_with_tokens(st, 512, 0)},
        {"prompt_name": None, "text": text_with_tokens(st, 2000, 100)},
        # ids only: past max_seq_length, so the stored ids show the cut.
        {"prompt_name": None, "text": long_text(700, 300), "tokenize_only": True},
    ]


def load_st(model_dir: Path):
    from sentence_transformers import SentenceTransformer

    check_remote(model_dir)
    st = SentenceTransformer(
        str(model_dir), device="cpu", trust_remote_code=True, config_kwargs=dict(CONFIG_KWARGS)
    )
    st.eval()
    # Every parameter must be the pinned tensor, bit for bit.
    from safetensors.torch import load_file

    file = load_file(str(model_dir / "model.safetensors"))
    sd = st[0].auto_model.state_dict()
    assert set(sd) == set(file), (set(sd) ^ set(file))
    for k, v in file.items():
        assert torch.equal(sd[k], v), k
    return st


def traced_real(model, ids):
    caps = []
    hooks = [model.embeddings.register_forward_hook(lambda m, i, o: caps.append(o[0][0].clone()))]
    hooks.append(model.encoder.layer[0].register_forward_hook(lambda m, i, o: caps.append(o[0][0].clone())))
    try:
        with torch.no_grad():
            x = torch.tensor([ids])
            last = model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0]
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([caps[0], caps[1], last]).numpy().astype(np.float32)


def gen_real(model_dir: Path) -> None:
    torch.manual_seed(0)
    st = load_st(model_dir)
    model = st[0].auto_model
    attn = model.config._attn_implementation
    cases = real_cases(st)
    embeds = []
    for i, case in enumerate(cases):
        prompt = PROMPTS[case["prompt_name"]] if case["prompt_name"] else ""
        # The call encode makes: prompt + text, stripped, then tokenized.
        case["ids"] = st.tokenize([prompt + case["text"]])["input_ids"][0].tolist()
        if case.get("tokenize_only"):
            print(f"case {i}: {len(case['ids'])} tokens (ids only)")
            continue
        kwargs = {"prompt_name": case["prompt_name"]} if case["prompt_name"] else {}
        emb = st.encode([case["text"]], batch_size=1, convert_to_numpy=True, **kwargs)[0]
        embeds.append(emb.astype(np.float32))
        print(f"case {i}: {len(case['ids'])} tokens, prompt {case['prompt_name']}")
    assert st.max_seq_length == 8192, st.max_seq_length
    assert len(cases[-1]["ids"]) == 8192 and cases[-1]["ids"][-1] == 2, "</s> must survive the cut"

    mrl = st.encode([cases[0]["text"]], prompt_name="query", truncate_dim=256,
                    normalize_embeddings=True, convert_to_numpy=True)[0]
    trace = traced_real(model, cases[0]["ids"])

    rot = model.embeddings.rotary_emb
    rows = [0, 1, 7, 511, 2047, 8191]
    save_npy("real_rotary_inv_freq.npy", rot.inv_freq.numpy().astype(np.float32))
    half = rot.inv_freq.numel()
    save_npy("real_rotary_cos_rows.npy", rot.cos_cached[rows, :half].numpy().astype(np.float32))
    save_npy("real_rotary_sin_rows.npy", rot.sin_cached[rows, :half].numpy().astype(np.float32))

    save_npy("real_embed.npy", np.stack(embeds))
    save_npy("real_embed_mrl256.npy", mrl.astype(np.float32))
    save_npy("real_trace_case0.npy", trace)
    dump_json(
        OUT / "real_cases.json",
        {
            "repo": REPO,
            "revision": REV,
            "max_seq_length": st.max_seq_length,
            "attn_implementation": attn,
            "config_kwargs": CONFIG_KWARGS,
            "cases": cases,
            "mrl": {"case": 0, "dim": 256},
            "trace": {"case": 0, "rows": ["embeddings (LayerNorm out)", "layer 0 out", "last_hidden_state"]},
            "rotary_rows": rows,
        },
    )


# ---------------------------------------------------------------- manifest


def write_manifest(model_dir: Path) -> None:
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
            "generator": "crates/embeddings/tools/gen_gte_fixtures.py",
            "packages": versions,
            "reference": "remote GteModel (pinned .py, transformers 4.57.1, f32, CPU) + sentence-transformers",
            "pins": {
                "real": {"repo": REPO, "revision": REV},
                "tiny": "seeded random init (numpy default_rng(0)) of the remote classes, no download",
            },
            "remote_code_sha256": remote_hashes(model_dir),
            "config_kwargs": CONFIG_KWARGS,
            "files": files,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", type=Path, metavar="MODEL_DIR", help="pinned install (remote code source)")
    ap.add_argument("--real", type=Path, metavar="MODEL_DIR")
    args = ap.parse_args()
    if not args.tiny and not args.real:
        ap.error("pass --tiny MODEL_DIR and/or --real MODEL_DIR")
    if args.tiny:
        gen_tiny(args.tiny.expanduser())
    if args.real:
        gen_real(args.real.expanduser())
    write_manifest((args.real or args.tiny).expanduser())


if __name__ == "__main__":
    main()
