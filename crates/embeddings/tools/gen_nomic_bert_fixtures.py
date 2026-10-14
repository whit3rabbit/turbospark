#!/usr/bin/env python3
"""Regenerate the Nomic BERT (snowflake-arctic-embed-m-long) Rust parity
fixtures.

Normative reference: the remote modeling code pinned with the checkpoint
(`modeling_hf_nomic_bert.py`, `configuration_hf_nomic_bert.py`) run under
transformers 4.57.1, f32, CPU, plus sentence-transformers 5.1.2 for
tokenize -> NomicBertModel -> Pooling (CLS) -> Normalize.

The remote code is imported only from the local pinned directory passed
as MODEL_DIR (trust_remote_code). HF_HUB_OFFLINE is forced so nothing is
fetched. `NomicBertPreTrainedModel.from_pretrained` would read
`pytorch_model.bin` if one existed in the directory; the generator
refuses to run when it does.

The rotary embedding is stateful in the reference: with
rotary_scaling_factor set, a forward longer than max_trained_positions
replaces `inv_freq` with an NTK-scaled one and caches cos/sin for that
length, and a later shorter forward reuses that cache instead of the
unscaled table (modeling_hf_nomic_bert.py:635-691). The goldens here are
the stateless semantics the Rust port implements: every sequence runs on
a model whose rotary state is fresh, which is what one call on a newly
loaded model computes. The drift a reused model shows is measured and
recorded (tiny_cases.json, real_cases.json).

--tiny MODEL_DIR (deterministic, numpy default_rng(0))
    A random-init 3-layer NomicBertModel built from the remote classes
    (n_embd 64, 4 heads of 16, n_inner 200 which the gated MLP rounds up
    to 256, vocab 60 padded to 64, max_trained_positions 16, rotary
    scaling factor 2). Cases run below, at, and above 16 tokens so the
    dynamic NTK path is pinned. Goldens: emb_ln output, every layer
    output, CLS + L2 embedding. Also the rotary inv_freq per case length.

--real MODEL_DIR (the pinned install)
    SentenceTransformer(MODEL_DIR, trust_remote_code=True).encode on a
    fixed text set (query prompt, accents, whitespace strip, ~512 tokens,
    ~3000 tokens for the NTK path), each text on a deep copy of the
    freshly loaded model, plus the token ids, a trace of case 0, the
    rotary tables at the boundary, a > 8192-token text whose ids only
    are stored (max_seq_length cut), and a copy of the pinned
    tokenizer.json.

    emb-venv/bin/python -I crates/embeddings/tools/gen_nomic_bert_fixtures.py \
        --tiny ~/models/embeddings/snowflake-arctic-embed-m-long \
        --real ~/models/embeddings/snowflake-arctic-embed-m-long

Outputs under crates/embeddings/testdata/nomic_bert/ plus manifest.json.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "testdata" / "nomic_bert"

REPO = "Snowflake/snowflake-arctic-embed-m-long"
REV = "92d97331f1f4b6a366c1f161354b9f3390cc219f"
REMOTE_FILES = ["modeling_hf_nomic_bert.py", "configuration_hf_nomic_bert.py"]
PROMPTS = {"query": "Represent this sentence for searching relevant passages: "}


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def save_npy(name: str, arr: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / name, np.ascontiguousarray(arr))


def check_remote(model_dir: Path) -> None:
    cfg = json.loads((model_dir / "config.json").read_text())
    for k, v in cfg.get("auto_map", {}).items():
        assert "--" not in v, f"auto_map {k} = {v} names another repository"
        assert (model_dir / (v.split(".")[0] + ".py")).is_file(), v
    # The remote from_pretrained prefers this pickle when present.
    assert not (model_dir / "pytorch_model.bin").exists(), "refusing: pytorch_model.bin present"


def remote(model_dir: Path, name: str):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    check_remote(model_dir)
    return get_class_from_dynamic_module(name, str(model_dir))


def remote_hashes(model_dir: Path) -> dict:
    return {f: hashlib.sha256((model_dir / f).read_bytes()).hexdigest() for f in REMOTE_FILES}


# ---------------------------------------------------------------- tiny

TINY = dict(
    vocab_size=60,
    n_embd=64,
    n_head=4,
    n_inner=200,
    n_layer=3,
    n_positions=64,
    activation_function="swiglu",
    rotary_emb_fraction=1.0,
    rotary_emb_base=1000,
    rotary_emb_interleaved=False,
    rotary_emb_scale_base=None,
    rotary_scaling_factor=2,
    max_trained_positions=16,
    type_vocab_size=2,
    pad_vocab_size_multiple=64,
    qkv_proj_bias=False,
    mlp_fc1_bias=False,
    mlp_fc2_bias=False,
    prenorm=False,
    parallel_block=False,
    parallel_block_tied_norm=False,
    causal=False,
    use_rms_norm=False,
    layer_norm_epsilon=1e-12,
    fused_bias_fc=True,
    fused_dropout_add_ln=True,
    use_flash_attn=True,
    use_xentropy=True,
    dense_seq_output=True,
    attn_pdrop=0.0,
    embd_pdrop=0.1,
    resid_pdrop=0.1,
    pad_token_id=0,
)
# [CLS] = 1, [SEP] = 2 here. Lengths straddle max_trained_positions 16:
# 16 is not scaled (the reference scales only when seqlen > 16), 17 is.
_rng = np.random.default_rng(11)
TINY_CASES = [
    [1, 2],
    [1, 7, 33, 2],
    [1] + [int(v) for v in _rng.integers(3, 60, 14)] + [2],
    [1] + [int(v) for v in _rng.integers(3, 60, 15)] + [2],
    [1] + [int(v) for v in _rng.integers(3, 60, 38)] + [2],
    [1] + [int(v) for v in _rng.integers(3, 60, 62)] + [2],
]


def tiny_state(model_cls, cfg_cls, rng) -> dict:
    model = model_cls(cfg_cls(**TINY), add_pooling_layer=False)
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    state = {}
    for name, shape in sorted(shapes.items()):
        if "norm" in name or "emb_ln" in name:
            if name.endswith("weight"):
                state[name] = rng.normal(1.0, 0.3, shape).astype(np.float32)
            else:
                state[name] = rng.normal(0.0, 0.2, shape).astype(np.float32)
        elif "embeddings" in name:
            state[name] = rng.normal(0.0, 0.5, shape).astype(np.float32)
        else:
            state[name] = rng.normal(0.0, 1.0 / math.sqrt(shape[-1]), shape).astype(np.float32)
    return state


def build(model_cls, cfg_cls, state: dict):
    model = model_cls(cfg_cls(**TINY), add_pooling_layer=False).eval()
    missing, unexpected = model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in state.items()}, strict=True
    )
    assert not missing and not unexpected
    return model


def traced_forward(model, ids: list[int]):
    """(states [layers + 1, seq, hidden], CLS + L2 embedding). State 0 is
    the emb_ln output, state i the output of layer i - 1."""
    caps = []
    hooks = [model.emb_ln.register_forward_hook(lambda m, i, o: caps.append(o[0].clone()))]
    for layer in model.encoder.layers:
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


def write_st_layout(root: Path, config: dict, max_seq_length: int) -> None:
    dump_json(root / "config.json", config)
    dump_json(
        root / "modules.json",
        [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ],
    )
    dump_json(root / "sentence_bert_config.json", {"max_seq_length": max_seq_length, "do_lower_case": False})
    dump_json(root / "config_sentence_transformers.json", {"prompts": PROMPTS, "default_prompt_name": None})
    dump_json(
        root / "1_Pooling" / "config.json",
        {
            "word_embedding_dimension": config["n_embd"],
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
    cfg_cls = remote(model_dir, "configuration_hf_nomic_bert.NomicBertConfig")
    model_cls = remote(model_dir, "modeling_hf_nomic_bert.NomicBertModel")
    state = tiny_state(model_cls, cfg_cls, np.random.default_rng(0))
    assert state["embeddings.word_embeddings.weight"].shape == (64, 64), "vocab must pad to 64"
    assert state["encoder.layers.0.mlp.fc11.weight"].shape == (256, 64), "n_inner must round to 256"

    root = OUT / "tiny"
    config = {"architectures": ["NomicBertModel"], "model_type": "nomic_bert", **TINY}
    write_st_layout(root, config, TINY["n_positions"])
    save_file(state, str(root / "model.safetensors"), metadata={"format": "pt"})

    cases = []
    for i, ids in enumerate(TINY_CASES):
        model = build(model_cls, cfg_cls, state)  # fresh rotary state per case
        states, emb = traced_forward(model, ids)
        save_npy(f"tiny_hidden_{i}.npy", states)
        save_npy(f"tiny_embed_{i}.npy", emb)
        rot = model.encoder.layers[0].attn.rotary_emb
        save_npy(f"tiny_rotary_inv_freq_{i}.npy", rot.inv_freq.numpy().astype(np.float32))
        scaled = len(ids) > TINY["max_trained_positions"]
        cases.append({"index": i, "ids": ids, "ntk_scaled": scaled})
    assert any(c["ntk_scaled"] for c in cases) and any(len(c["ids"]) == 16 for c in cases)

    # Reference statefulness: the 40-token case, then the 16-token case,
    # on one model. The second reuses the scaled cache.
    model = build(model_cls, cfg_cls, state)
    traced_forward(model, TINY_CASES[4])
    reused, _ = traced_forward(model, TINY_CASES[2])
    fresh, _ = traced_forward(build(model_cls, cfg_cls, state), TINY_CASES[2])
    drift = float(np.abs(reused - fresh).max())
    print(f"tiny stateful drift (case 2 after case 4 on one model): {drift}")
    assert drift > 1e-3, "expected the reused scaled cache to change the output"

    dump_json(
        OUT / "tiny_cases.json",
        {"cases": cases, "config": TINY, "stateful_reference_drift": {"after": 4, "case": 2, "max_abs": drift}},
    )
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
         "text": "Caf\u00e9 na\u00efve r\u00e9sum\u00e9: \u00dcBER die Br\u00fccke! \u00c0\u00c9\u00ce\u00d5 "
                 "\u6771\u4eac\u306f\u65e5\u672c\u306e\u9996\u90fd\u3067\u3059\u3002"},
        {"prompt_name": None, "text": text_with_tokens(st, 512, 0)},
        {"prompt_name": None, "text": text_with_tokens(st, 3000, 100)},
        {"prompt_name": None, "text": long_text(700, 300), "tokenize_only": True},
    ]


def load_st(model_dir: Path):
    from sentence_transformers import SentenceTransformer

    check_remote(model_dir)
    st = SentenceTransformer(str(model_dir), device="cpu", trust_remote_code=True)
    st.eval()
    # The remote from_pretrained loads with strict=False; check every
    # tensor against the pinned file instead of trusting its log line.
    from safetensors.torch import load_file

    file = load_file(str(model_dir / "model.safetensors"))
    sd = st[0].auto_model.state_dict()
    assert set(sd) == set(file), (set(sd) ^ set(file))
    for k, v in file.items():
        assert torch.equal(sd[k], v), k
    return st


def traced_real(model, ids):
    caps = []
    hooks = [model.emb_ln.register_forward_hook(lambda m, i, o: caps.append(o[0].clone()))]
    hooks.append(model.encoder.layers[0].register_forward_hook(lambda m, i, o: caps.append(o[0][0].clone())))
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
    pristine = load_st(model_dir)
    cases = real_cases(pristine)
    embeds = []
    for i, case in enumerate(cases):
        prompt = PROMPTS[case["prompt_name"]] if case["prompt_name"] else ""
        case["ids"] = pristine.tokenize([prompt + case["text"]])["input_ids"][0].tolist()
        if case.get("tokenize_only"):
            print(f"case {i}: {len(case['ids'])} tokens (ids only)")
            continue
        st = copy.deepcopy(pristine)  # fresh rotary state
        kwargs = {"prompt_name": case["prompt_name"]} if case["prompt_name"] else {}
        emb = st.encode([case["text"]], batch_size=1, convert_to_numpy=True, **kwargs)[0]
        embeds.append(emb.astype(np.float32))
        print(f"case {i}: {len(case['ids'])} tokens, prompt {case['prompt_name']}")
    assert pristine.max_seq_length == 8192, pristine.max_seq_length
    assert any(len(c["ids"]) > 2048 and not c.get("tokenize_only") for c in cases), "need an NTK case"
    assert len(cases[-1]["ids"]) == 8192 and cases[-1]["ids"][-1] == 102, "[SEP] must survive the cut"

    # Reference statefulness on the real model: case 0 after the NTK case.
    st = copy.deepcopy(pristine)
    st.encode([cases[5]["text"]], batch_size=1)
    reused = st.encode([cases[0]["text"]], prompt_name="query", batch_size=1, convert_to_numpy=True)[0]
    drift = float(np.abs(reused - embeds[0]).max())
    cos = float(np.dot(reused, embeds[0]))
    print(f"real stateful drift (case 0 after case 5 on one model): max abs {drift}, cosine {cos}")

    model = copy.deepcopy(pristine)[0].auto_model
    trace = traced_real(model, cases[0]["ids"])

    # Rotary tables straight from the remote class, fresh per length.
    rot_cls = remote(model_dir, "modeling_hf_nomic_bert.NomicBertDynamicNTKRotaryEmbedding")
    cfg = model.config
    head_dim = cfg.n_embd // cfg.n_head
    lengths = [2048, 2049, 3000, 8192]
    rows = [0, 1, 7, 1000, 2047]
    inv, cos_rows, sin_rows = [], [], []
    for n in lengths:
        rot = rot_cls(
            dim=head_dim * cfg.rotary_emb_fraction,
            base=cfg.rotary_emb_base,
            scale_base=cfg.rotary_emb_scale_base,
            interleaved=cfg.rotary_emb_interleaved,
            rotary_scaling_factor=cfg.rotary_scaling_factor,
            max_position_embeddings=cfg.max_trained_positions,
        )
        rot(torch.zeros(1, n, 3, 1, head_dim), seqlen_offset=0)
        inv.append(rot.inv_freq.numpy().astype(np.float32))
        cos_rows.append(rot._cos_cached[rows].numpy().astype(np.float32))
        sin_rows.append(rot._sin_cached[rows].numpy().astype(np.float32))
    save_npy("real_rotary_inv_freq.npy", np.stack(inv))
    save_npy("real_rotary_cos_rows.npy", np.stack(cos_rows))
    save_npy("real_rotary_sin_rows.npy", np.stack(sin_rows))

    save_npy("real_embed.npy", np.stack(embeds))
    save_npy("real_trace_case0.npy", trace)
    # The pinned tokenizer (700 KB) so tokenizer parity runs without the
    # install; the GTE one is 17 MB and stays out of testdata.
    shutil.copyfile(model_dir / "tokenizer.json", OUT / "tokenizer.json")
    dump_json(
        OUT / "real_cases.json",
        {
            "repo": REPO,
            "revision": REV,
            "max_seq_length": pristine.max_seq_length,
            "cases": cases,
            "trace": {"case": 0, "rows": ["emb_ln out", "layer 0 out", "last_hidden_state"]},
            "rotary": {"lengths": lengths, "rows": rows},
            "stateful_reference_drift": {"after": 5, "case": 0, "max_abs": drift, "cosine": cos},
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
        for name in ["torch", "numpy", "safetensors", "transformers", "sentence-transformers", "tokenizers",
                     "einops"]
    }
    dump_json(
        OUT / "manifest.json",
        {
            "generator": "crates/embeddings/tools/gen_nomic_bert_fixtures.py",
            "packages": versions,
            "reference": "remote NomicBertModel (pinned .py, transformers 4.57.1, f32, CPU) + sentence-transformers",
            "pins": {
                "real": {"repo": REPO, "revision": REV},
                "tiny": "seeded random init (numpy default_rng(0)) of the remote classes, no download",
            },
            "remote_code_sha256": remote_hashes(model_dir),
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
