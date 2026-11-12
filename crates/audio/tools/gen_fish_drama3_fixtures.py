"""Golden fixtures for the Fish Audio Drama 3 port (`src/tts/fish_drama3/`).

The reference is mlx-audio at dad2087f790dfe96b07d5aac1405db4b578b411f
(Blaizzy/mlx-audio PR 1018), newer than the crate's normative e1b19b9, which
has no Drama 3. Fetch the pinned chunker and run this with an isolated
interpreter (the chunker imports only `re`):

    gh api "repos/Blaizzy/mlx-audio/contents/mlx_audio/tts/models/fish_drama3/chunking.py?ref=dad2087f790dfe96b07d5aac1405db4b578b411f" \
        -H "Accept: application/vnd.github.raw" > /path/outside/repo/chunking.py
    python3 -I crates/audio/tools/gen_fish_drama3_fixtures.py chunking /path/outside/repo/chunking.py \
        > crates/audio/testdata/fish_drama3/chunking.json

The tower fixtures need mlx-audio itself at that commit (tested with mlx
0.32.4), installed in a scratch environment outside the repo:

    uv venv --python 3.12 /path/outside/repo/venv
    uv pip install --python /path/outside/repo/venv/bin/python \
        "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@dad2087f790dfe96b07d5aac1405db4b578b411f"
    /path/outside/repo/venv/bin/python -I crates/audio/tools/gen_fish_drama3_fixtures.py \
        tower crates/audio/testdata/fish_drama3
    /path/outside/repo/venv/bin/python -I crates/audio/tools/gen_fish_drama3_fixtures.py \
        prompt crates/audio/testdata/fish_drama3
    /path/outside/repo/venv/bin/python -I crates/audio/tools/gen_fish_drama3_fixtures.py \
        prompt-real <drama-install-dir> > crates/audio/testdata/fish_drama3/prompt_real.json

Tests read the outputs; they never run Python or download.

The `teacher-forced` mode is not a fixture. It dumps the reference's own
numbers on the real 4-bit checkpoint for a Rust comparer to read
(`crates/runtime/examples/fish_drama3_parity.rs`). It holds about 24 GB of GPU
buffers, so run it alone on a 36 GB host and write under `target/`, never a
scratch directory that other sessions wipe:

    /path/outside/repo/venv/bin/python -I crates/audio/tools/gen_fish_drama3_fixtures.py \
        teacher-forced <mlx-4bit-dir> target/s5-parity/mlx <gen_q4-dir> \
        crates/audio/testdata/fish_drama3/prompt_real.json [variant,...] [case,...]

`<gen_q4-dir>` holds `<clip>.prompt.npy` and `<clip>.codes.npy` from an
earlier mlx-audio generation of the same checkpoint (the T7 evaluation's
`out/gen_q4`).
"""

import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

COMMIT = "dad2087f790dfe96b07d5aac1405db4b578b411f"

SPLIT_INPUTS = [
    "<|speaker:1|> " + "Das ist ein Satz mit Inhalt. " * 20,
    "wort " * 200,
    "Hallo. <whisper>" + "leise, " * 60 + "</whisper> Ende.",
    "<|speaker:0|> a. <|speaker:1|> b. <|speaker:2|> c. <|speaker:3|> d.",
    "hi <|speaker:4|> yo",
    "日本語の文章" * 40,
    "First sentence here, with a clause; and more. " * 9
    + "<|speaker:5|> [sighs] Fine. <whisper>quiet words</whisper> Then 3.5 percent.",
    "   ",
    "short",
    "<|speaker:2|>",
    "A" * 650,
    "x, " * 140 + "[" + "long cue " * 40 + "] tail.",
    "<|speaker:0|> " + "one. " * 70 + "<|speaker:1|> two. <|speaker:0|> three.",
    "Zahlen: 1.5, 2.25 und 3.75. Ende! Wirklich? Ja; gut… fertig\nneue Zeile。中文，句子：结束。",
    "<|speaker:0|> Are you coming tonight? <|speaker:1|> I don't know yet. <|speaker:0|> Let me know by six.",
]

UNCUTTABLE_INPUTS = [
    "a <|speaker:2|> b [laughs] c <em>x</em> <b-x>y</b-x> <a- x>z</a> <p>open",
    "[open <b>bold",
    "<|a|b|> <|ok|> [x[y]] <i>nest <i>in</i></i> <ü>u</ü> <a1_b>c</a1_b>",
    "<tag attr='1'>body\nline</tag><tag>",
]

VALIDATE_INPUTS = [
    "hi <|im_end|>",
    "<|semantic:7|>",
    "<|speaker:16|> hi",
    "  ",
    "\x1c",
    "　\xa0\n",
    "<|speaker:01|> a",
    "<|speaker:0|> a <|speaker:15|> b",
    "<|speaker:x|> a",
    "<|speaker a",
    "plain text",
    "<|voice|>",
    "<|interleave|> x",
]

TAG_INPUTS = ["hi", "  <|speaker:3|> hi", "<|speaker:12|>", "\n\t<|speaker:1|>x", " plain"]


def chunking(path: str) -> None:
    spec = importlib.util.spec_from_file_location("chunking", path)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)

    def validate(text: str) -> bool:
        try:
            c.validate_text(text, "text")
            return True
        except ValueError:
            return False

    out = {
        "commit": COMMIT,
        "split_text": [{"input": t, "output": c.split_text(t)} for t in SPLIT_INPUTS],
        "uncuttable": [
            {"input": t, "output": [m.group(0) for m in c.UNCUTTABLE.finditer(t)]}
            for t in UNCUTTABLE_INPUTS
        ],
        "validate_text": [{"input": t, "ok": validate(t)} for t in VALIDATE_INPUTS],
        "with_speaker_tag": [
            {"input": t, "output": c.with_speaker_tag(t)} for t in TAG_INPUTS
        ],
        "reference_transcript": [
            {"input": t, "output": c.reference_transcript(t)} for t in TAG_INPUTS
        ],
    }
    json.dump(out, sys.stdout, ensure_ascii=True, indent=1)
    sys.stdout.write("\n")


# -- tower ------------------------------------------------------------------

# A tiny tower with the real structure: enrichment and residual layers, GQA
# with gated attention, interleaved partial rope, 8-bit linears and BF16
# embeddings as the published conversion stores them.
TOWER = dict(
    dim=128,
    # Two query heads per KV head, so the GQA head-to-group mapping is
    # observable (with one KV head every mapping agrees).
    n_head=4,
    n_local_heads=2,
    head_dim=64,
    intermediate_size=128,
    vocab_size=16,
    num_codebooks=4,
    shared_enrichment_layers=2,
    vq_backbone_codebooks=1,
    norm_eps=1e-6,
    rope_base=1e7,
    rotary_percent=0.25,
)
SEMANTIC_BEGIN = 32
SEMANTIC_END = SEMANTIC_BEGIN + TOWER["vocab_size"] - 1
EOS = 2


def tower(out_dir: str) -> None:
    from pathlib import Path

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten

    from mlx_audio.tts.models.fish_drama3.config import TowerConfig
    from mlx_audio.tts.models.fish_drama3.tower import Tower

    out = Path(out_dir)
    mx.random.seed(7)
    cfg = TowerConfig.from_dict(TOWER)
    vocab, books, dim = cfg.vocab_size, cfg.num_codebooks, cfg.dim

    class Side(nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = Tower(cfg, SEMANTIC_BEGIN, SEMANTIC_END)
            self.semantic_head = nn.Linear(dim, vocab + 1, bias=False)

    side = Side()
    # Random weights in the converter's storage: zero-centered norms shifted
    # to 1 + w in float32, embeddings and linears bfloat16 before quantizing.
    params = []
    for name, value in tree_flatten(side.parameters()):
        if value.dtype == mx.float32 and name.endswith("weight") and value.ndim == 1:
            params.append((name, 1 + 0.1 * mx.random.normal(value.shape)))
        else:
            params.append((name, (0.08 * mx.random.normal(value.shape)).astype(mx.bfloat16)))
    side.load_weights(params)
    nn.quantize(
        side, group_size=64, bits=8, class_predicate=lambda _, m: isinstance(m, nn.Linear)
    )
    mx.eval(side.parameters())
    mx.save_safetensors(
        str(out / "tower_side.safetensors"), dict(tree_flatten(side.parameters()))
    )

    # A prompt of text and audio rows: [token, c0..c3]. Audio rows carry
    # c0 + SEMANTIC_BEGIN as their token.
    rng = np.random.default_rng(11)

    def audio_row():
        codes = rng.integers(0, vocab, size=books)
        return [int(codes[0]) + SEMANTIC_BEGIN, *map(int, codes)]

    prompt = [[5, 0, 0, 0, 0], audio_row(), audio_row(), [9, 0, 0, 0, 0], audio_row(), audio_row()]
    frames = mx.array(prompt, dtype=mx.int32)[None]
    hidden = (mx.random.normal((1, len(prompt), dim))).astype(mx.bfloat16)
    caches = side.tower.make_cache()
    last = frames.shape[1] - 1
    side.tower.prefill(hidden[:, :last], frames[:, :last], frames[:, 1:], caches)

    # Three decode frames: semantic, EOS (not semantic: codes masked), semantic.
    mains = [SEMANTIC_BEGIN + 5, EOS, SEMANTIC_BEGIN + 11]
    step_hidden = [hidden[:, last:]] + [
        mx.random.normal((1, 1, dim)).astype(mx.bfloat16) for _ in mains[1:]
    ]
    frame = frames[:, last:]
    steps = []
    for main, h in zip(mains, step_hidden):
        main_arr = mx.array([main])
        c0 = mx.clip(main_arr - SEMANTIC_BEGIN, 0, vocab - 1)
        semantic = side.tower.is_semantic(main_arr)[:, None]
        head_logits = []

        def greedy(logits):
            head_logits.append(np.array(logits[0, 0].astype(mx.float32)).tolist())
            return mx.argmax(logits[:, 0], axis=-1)[:, None]

        codes = side.tower.residual_codes(h, c0[:, None], semantic, frame, caches, greedy)
        new_frame = mx.concatenate([main_arr[:, None], *codes], axis=-1)[:, None, :].astype(mx.int32)
        steps.append(
            {
                "main": main,
                "hidden": np.array(h[0, 0].astype(mx.float32)).tolist(),
                "head_logits": head_logits,
                "frame": np.array(new_frame[0, 0]).tolist(),
            }
        )
        frame = new_frame

    # The backbone input mix for each prompt row, from a random text row.
    text = mx.random.normal((1, len(prompt), dim)).astype(mx.bfloat16)
    mixed = side.tower.embed_backbone(frames, text)
    record = {
        "commit": COMMIT,
        "tower": TOWER,
        "semantic_begin": SEMANTIC_BEGIN,
        "semantic_end": SEMANTIC_END,
        "eos": EOS,
        "prompt": prompt,
        "prompt_hidden": np.array(hidden[0].astype(mx.float32)).tolist(),
        "steps": steps,
        "embed_text": np.array(text[0].astype(mx.float32)).tolist(),
        "embed_mixed": np.array(mixed[0].astype(mx.float32)).tolist(),
    }
    (out / "tower.json").write_text(json.dumps(record))


# -- prompt -----------------------------------------------------------------

# The reference test's tiny WordLevel tokenizer (`write_tokenizer` in
# `test_fish_drama3.py`): structural and speaker tokens, the semantic band at
# PROMPT_SEMANTIC_BEGIN, and [UNK] for every plain word.
PROMPT_SEMANTIC_BEGIN = 32
PROMPT_VOCAB = 16
PROMPT_BOOKS = 3


def prompt(out_dir: str) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    import mlx.core as mx
    from tokenizers import Tokenizer, models, pre_tokenizers

    from mlx_audio.tts.models.fish_drama3.drama3 import Model

    out = Path(out_dir)
    special = ["<|im_start|>", "<|im_end|>", "<|voice|>"]
    special += [f"<|speaker:{i}|>" for i in range(16)]
    vocab = {"[UNK]": 0, **{token: i + 1 for i, token in enumerate(special)}}
    for code in range(PROMPT_VOCAB):
        vocab[f"<|semantic:{code}|>"] = PROMPT_SEMANTIC_BEGIN + code
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(special + [f"<|semantic:{c}|>" for c in range(PROMPT_VOCAB)])
    tokenizer.save(str(out / "prompt_tokenizer.json"))

    fake = SimpleNamespace(
        tokenizer=Tokenizer.from_file(str(out / "prompt_tokenizer.json")),
        semantic_begin=PROMPT_SEMANTIC_BEGIN,
        num_codebooks=PROMPT_BOOKS,
        tower=SimpleNamespace(vocab_size=PROMPT_VOCAB),
    )
    fake._check_codes = lambda codes, location: Model._check_codes(fake, codes, location)

    def codes(frames, seed):
        return [[(seed + 3 * f + 5 * b) % PROMPT_VOCAB for f in range(frames)] for b in range(PROMPT_BOOKS)]

    cases = [
        {"references": [], "history": [], "target": "Hallo du."},
        {
            "references": [["Guten Tag.", codes(3, 1)]],
            "history": [],
            "target": "<|speaker:0|> Eins. <|speaker:1|> Zwei.",
        },
        {
            "references": [["<|speaker:0|> Anna.", codes(2, 2)], ["  <|speaker:1|> Ben.", codes(4, 7)]],
            "history": [["Erster Teil.", codes(2, 4)], ["<|speaker:1|> Zweiter.", codes(1, 9)]],
            "target": "Dritter [laughs] Teil.",
        },
    ]
    for case in cases:
        frames, system_rows = Model.build_frames(
            fake,
            [(t, mx.array(c, dtype=mx.int32)) for t, c in case["references"]],
            [(t, mx.array(c, dtype=mx.int32)) for t, c in case["history"]],
            case["target"],
        )
        case["frames"] = frames.tolist()
        case["system_rows"] = system_rows
    record = {
        "commit": COMMIT,
        "semantic_begin": PROMPT_SEMANTIC_BEGIN,
        "vocab": PROMPT_VOCAB,
        "num_codebooks": PROMPT_BOOKS,
        "cases": cases,
    }
    (out / "prompt.json").write_text(json.dumps(record))


def prompt_real(install_dir: str) -> None:
    """Frames from the REAL tokenizer (`tokenizer.json` of the pinned
    `mlx-community/fish-audio-drama-3-4bit`), printed as JSON. The tokenizer is
    not committed, so the Rust test that reads this is opt-in."""
    from pathlib import Path
    from types import SimpleNamespace

    import mlx.core as mx
    from tokenizers import Tokenizer

    from mlx_audio.tts.models.fish_drama3.chunking import split_text
    from mlx_audio.tts.models.fish_drama3.drama3 import Model

    root = json.loads((Path(install_dir) / "config.json").read_text())
    books = root["audio_decoder_config"]["num_codebooks"]
    vocab = root["audio_decoder_config"]["vocab_size"]
    fake = SimpleNamespace(
        tokenizer=Tokenizer.from_file(str(Path(install_dir) / "tokenizer.json")),
        semantic_begin=root["semantic_start_token_id"],
        num_codebooks=books,
        tower=SimpleNamespace(vocab_size=vocab),
    )
    fake._check_codes = lambda codes, location: Model._check_codes(fake, codes, location)

    def codes(frames, seed):
        return [[(seed * 7919 + 131 * f + 17 * b) % (1024 if b else vocab) for f in range(frames)] for b in range(books)]

    cases = [
        {"references": [], "history": [], "target": "[sighs] Fine. If we miss it, it's on you."},
        {
            "references": [["  We asked over twenty different people.", codes(4, 3)]],
            "history": [["Erster Teil, mit Umlauten: aeoeue.", codes(3, 5)]],
            "target": "<|speaker:0|> Are you coming? <|speaker:1|> Give me a minute.",
        },
        {
            "references": [["<|speaker:1|> Ben spricht.", codes(2, 8)]],
            "history": [],
            "target": "Zahlen wie 3.14 und Emojis-freier Text, bitte <whisper>leise</whisper>.",
        },
        {
            # White space BEFORE the tag: the one shape where
            # `reference_transcript` differs from `with_speaker_tag`.
            "references": [["\n <|speaker:2|> Carla.", codes(3, 11)]],
            "history": [],
            "target": "<|speaker:2|> Hallo.",
        },
    ]
    # The two texts that run away at every seed tried (FISH_DRAMA3_PHASE0.md,
    # multi-speaker and cue texts), as `generate` builds them: through
    # `split_text`, whose rejoined turns carry a double space after each tag
    # and " \n" between turns.
    for text in (
        "<|speaker:0|> Are you coming tonight? <|speaker:1|> I don't know yet. "
        "<|speaker:0|> Let me know by six.",
        "[laughs] Oh, that is the best thing I have heard all week. [pause] Seriously.",
    ):
        (chunk,) = split_text(text)
        cases.append({"references": [], "history": [], "target": chunk})
    for case in cases:
        frames, system_rows = Model.build_frames(
            fake,
            [(t, mx.array(c, dtype=mx.int32)) for t, c in case["references"]],
            [(t, mx.array(c, dtype=mx.int32)) for t, c in case["history"]],
            case["target"],
        )
        case["frames"] = frames.tolist()
        case["system_rows"] = system_rows
    json.dump({"commit": COMMIT, "cases": cases}, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


# -- teacher-forced ---------------------------------------------------------

# Replayed recordings: mlx-audio's own 4-bit generations (the T7 evaluation's
# `out/gen_q4`), so every code sequence is one the reference produced. A spread
# of lengths plus the `[sighs]` cue clip. The prompt-real cases add speaker
# tags, history turns and reference rows, with no generated frames.
TF_CLIPS = ["s02_s0", "s03_s0", "s05_s0", "s07_s0", "s09_s1", "s13_s0", "s16_s1", "s19_s0"]
# ref: the checkpoint as mlx-audio runs it. router_int8: the router goes
# through the intake's INT8 quantizer. norm_bf16: the trunk's norm gammas are
# narrowed to BF16. convert: both of those plus the shared-expert gate in INT8,
# which is everything the intake does to the trunk's dense weights.
TF_VARIANTS = ("ref", "router_int8", "norm_bf16", "convert")
TF_TOP_K = 8
GATE_GROUP = 64


def _bf16_round(x):
    """Round-to-nearest-even through bfloat16, as `compute::f32_to_bf16`."""
    import numpy as np

    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def quantize_int8_rows(w):
    """`compute::quantize_int8_affine` per row, which is what the intake runs on
    the router. Returns (q uint8, scale, bias, dequantized f32), the scale and
    bias already rounded to bfloat16 as stored."""
    import numpy as np

    rows, cols = w.shape
    g = np.ascontiguousarray(w, dtype=np.float32).reshape(rows, cols // GATE_GROUP, GATE_GROUP)
    lo, hi = g.min(-1), g.max(-1)
    flat = hi == lo
    scale = np.where(flat, np.float32(1.0), (hi - lo) / np.float32(255.0)).astype(np.float32)
    s, b = _bf16_round(scale), _bf16_round(lo)
    inv = np.where(s == 0, np.float32(0.0), np.float32(1.0) / s).astype(np.float32)
    t = (g - b[..., None]) * inv[..., None]
    # Rust's f32::round is half away from zero, numpy's is half to even.
    q = np.clip(np.sign(t) * np.floor(np.abs(t) + np.float32(0.5)), 0, 255).astype(np.uint8)
    deq = (q.astype(np.float32) * s[..., None] + b[..., None]).reshape(rows, cols)
    return q.reshape(rows, cols), s, b, deq.astype(np.float32)


class _TfState:
    """Hook state shared by the patched sparse-MoE blocks."""

    def __init__(self, layers):
        self.layer_of = {id(layer.mlp): i for i, layer in enumerate(layers)}
        self.layers = len(layers)
        self.int8_router = None  # [layers] of mx f32 [experts, hidden]
        self.int8_shared = None  # [layers] of mx f32 [1, hidden]
        self.use_router_int8 = False
        self.use_shared_int8 = False
        self.isolate = False
        self.use_original = False
        self.pending = []
        self.chunks = {}

    def start_case(self):
        self.pending = []
        self.chunks = {}

    def push(self, layer, **arrays):
        self.pending.append((layer, arrays))

    def flush(self, *extra):
        import mlx.core as mx
        import numpy as np

        mx.eval(extra, [list(a.values()) for _, a in self.pending])
        for layer, arrays in self.pending:
            for key, value in arrays.items():
                rows = self.chunks.setdefault(key, [[] for _ in range(self.layers)])
                rows[layer].append(np.array(value)[0])
        self.pending = []

    def collect(self, rows):
        import numpy as np

        out = {}
        for key, per_layer in self.chunks.items():
            stacked = np.stack([np.concatenate(chunks, axis=0) for chunks in per_layer], axis=1)
            assert stacked.shape[0] == rows, (key, stacked.shape, rows)
            out[key] = stacked
        return out


def _tf_patch_moe(cls, state):
    """Replaces the sparse-MoE block's call with a copy of it that can swap in
    the INT8 router and record what routed. With every switch off it computes
    exactly what the original does (checked by `_tf_probe`)."""
    import mlx.core as mx

    original = cls.__call__

    def call(self, x):
        if state.use_original:
            return original(self, x)
        assert self.sharding_group is None
        layer = state.layer_of[id(self)]
        k = self.top_k
        if state.use_router_int8:
            logits = x.astype(mx.float32) @ state.int8_router[layer].T
        else:
            logits = self.gate(x)
        gates = mx.softmax(logits, axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)

        # The gap between the k-th and (k+1)-th probability: zero is a tie the
        # selection broke arbitrarily.
        top = mx.sort(gates.astype(mx.float32), axis=-1)[..., -(k + 1) :]
        record = {"used": inds, "margin": top[..., 1] - top[..., 0]}
        if state.isolate:
            # Three routers on the one input this layer saw: the BF16 gate as
            # run, the same BF16 weights computed in f32, and the INT8 weights
            # in f32. Their disagreement splits rounding from quantization.
            xf = x.astype(mx.float32)
            lf = xf @ self.gate.weight.astype(mx.float32).T
            li = xf @ state.int8_router[layer].T
            top_f = mx.sort(lf, axis=-1)[..., -(k + 1) :]
            record["ids_f32"] = mx.argpartition(lf, kth=-k, axis=-1)[..., -k:]
            record["ids_int8"] = mx.argpartition(li, kth=-k, axis=-1)[..., -k:]
            record["margin_f32"] = top_f[..., 1] - top_f[..., 0]
        state.push(layer, **record)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        if state.use_shared_int8:
            gate = mx.sigmoid(x.astype(mx.float32) @ state.int8_shared[layer].T).astype(x.dtype)
        else:
            gate = mx.sigmoid(self.shared_expert_gate(x))
        return y + gate * shared_y

    cls.__call__ = call
    return original


def _tf_cases(gen_dir, prompt_real_path, semantic_begin):
    import numpy as np

    gen = Path(gen_dir)
    system = {r["name"]: r["system_rows"] for r in json.loads((gen / "summary.json").read_text())["runs"]}
    cases = []
    for name in TF_CLIPS:
        prompt = np.load(gen / f"{name}.prompt.npy").astype(np.int32)
        codes = np.load(gen / f"{name}.codes.npy")
        # [main token, c0..c9] per generated frame, as `build_frames` lays out
        # an audio row.
        generated = np.concatenate([codes[:1] + semantic_begin, codes]).T.astype(np.int32)
        cases.append(
            {
                "name": name,
                "rows": np.concatenate([prompt, generated]),
                "prompt_rows": len(prompt),
                "system_rows": system[name],
            }
        )
    for i, case in enumerate(json.loads(Path(prompt_real_path).read_text())["cases"]):
        rows = np.array(case["frames"], dtype=np.int32)
        cases.append(
            {"name": f"pr{i}", "rows": rows, "prompt_rows": len(rows), "system_rows": case["system_rows"]}
        )
    return cases


def _tf_run(model, state, case):
    """One case, the way `Model._frames` runs it: the prompt in 256-row chunks
    (the system block first), then one row per step. The tower is teacher-forced
    with the recorded next row and its head logits are kept."""
    import mlx.core as mx
    import numpy as np

    from mlx_audio.tts.models.fish_drama3.drama3 import PREFILL_CHUNK

    rows, prompt_rows, system_rows = case["rows"], case["prompt_rows"], case["system_rows"]
    total = len(rows)
    frames = mx.array(rows)[None]
    bcache, tcache = model.backbone.make_cache(), model.tower.make_cache()
    state.start_case()
    hidden = []

    def caches():
        return [[c.state for c in bcache], [c.state for c in tcache]]

    def feed(begin, end):
        return model.backbone.model(
            None, cache=bcache, input_embeddings=model._embed(frames[:, begin:end])
        )

    for lo, hi in ((0, system_rows), (system_rows, prompt_rows)):
        for begin in range(lo, hi, PREFILL_CHUNK):
            end = min(begin + PREFILL_CHUNK, hi)
            h = feed(begin, end)
            labelled = min(end, prompt_rows - 1)
            if labelled > begin:
                model.tower.prefill(
                    h[:, : labelled - begin],
                    frames[:, begin:labelled],
                    frames[:, begin + 1 : labelled + 1],
                    tcache,
                )
            state.flush(h, caches())
            hidden.append(h[0])

    semantic, heads = [], []
    for t in range(prompt_rows - 1, total):
        h_t = hidden[-1][-1:][None]
        logits = model.semantic_head(h_t[:, 0]).astype(mx.float32)
        semantic.append(logits)
        if t + 1 == total:
            state.flush(logits)
            break
        nxt = frames[:, t + 1]
        main = nxt[:, 0]
        c0 = mx.clip(main - model.semantic_begin, 0, model.tower.vocab_size - 1)
        is_semantic = model.tower.is_semantic(main)[:, None]
        logs = []

        def sample(head_logits):
            logs.append(head_logits[:, 0].astype(mx.float32))
            return nxt[:, len(logs) + 1 : len(logs) + 2]

        model.tower.residual_codes(h_t, c0[:, None], is_semantic, frames[:, t : t + 1], tcache, sample)
        heads.append(mx.concatenate(logs, axis=0))
        h = feed(t + 1, t + 2)
        state.flush(h, logs, logits, caches())
        hidden.append(h[0])

    f32 = lambda a: np.array(a.astype(mx.float32))
    out = {
        "hidden": f32(mx.concatenate(hidden, axis=0)),
        "semantic": f32(mx.concatenate(semantic, axis=0)),
        "heads": f32(mx.stack(heads)) if heads else np.zeros((0, 9, model.tower.vocab_size), np.float32),
    }
    assert out["hidden"].shape[0] == total
    out.update(state.collect(total))
    return out


def _tf_probe(model, state, rows):
    """Hidden states of the first rows with the original block call and with the
    patched one: they must be equal, or the patch changed the model."""
    import mlx.core as mx

    frames = mx.array(rows[:24])[None]
    seen = []
    for original in (True, False):
        state.use_original = original
        state.start_case()
        cache = model.backbone.make_cache()
        h = model.backbone.model(None, cache=cache, input_embeddings=model._embed(frames))
        state.flush(h)
        seen.append(h)
    state.use_original = False
    return bool(mx.array_equal(seen[0], seen[1]).item())


def teacher_forced(model_dir, out_dir, gen_dir, prompt_real_path, variants=None, only=None):
    import mlx.core as mx
    import numpy as np

    from mlx_audio.lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    from mlx_audio.tts.models.fish_drama3.backbone import Fp32RMSNorm
    from mlx_audio.tts.utils import load_model

    out = Path(out_dir)
    chosen = variants.split(",") if variants else list(TF_VARIANTS)
    for v in chosen:
        assert v in TF_VARIANTS, v
    # The codec is not needed and the f32 copy is big; a small buffer cache
    # keeps the 4-bit model inside a 36 GB host.
    mx.set_cache_limit(1 << 30)
    model = load_model(Path(model_dir))
    model.codec = None
    gc.collect()
    mx.clear_cache()
    layers = model.backbone.model.layers
    state = _TfState(layers)
    _tf_patch_moe(Qwen3NextSparseMoeBlock, state)
    print(f"loaded: active {mx.get_active_memory() / 1e9:.1f} GB", flush=True)

    gate0 = layers[0].mlp.gate.weight
    assert gate0.dtype == mx.bfloat16, f"the router is {gate0.dtype}, expected the unquantized bfloat16"
    sim_dir = out / "gate_sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    state.int8_router, state.int8_shared = [], []
    for i, layer in enumerate(layers):
        w = np.array(layer.mlp.gate.weight.astype(mx.float32))
        q, s, b, deq = quantize_int8_rows(w)
        state.int8_router.append(mx.array(deq))
        ws = np.array(layer.mlp.shared_expert_gate.weight.astype(mx.float32))
        state.int8_shared.append(mx.array(quantize_int8_rows(ws)[3]))
        if i == 0:
            w.astype("<f4").tofile(sim_dir / "gate0.f32")
            q.tofile(sim_dir / "gate0_q.u8")
            s.astype("<f4").tofile(sim_dir / "gate0_scale.f32")
            b.astype("<f4").tofile(sim_dir / "gate0_bias.f32")

    norms = [m for _, m in model.backbone.named_modules() if isinstance(m, Fp32RMSNorm)]
    norm_originals = [m.weight for m in norms]
    assert all(w.dtype == mx.float32 for w in norm_originals)
    narrowed = [w.astype(mx.bfloat16).astype(mx.float32) for w in norm_originals]
    changed = sum(int((a != b).sum().item()) for a, b in zip(norm_originals, narrowed))
    total = sum(w.size for w in norm_originals)
    print(f"{len(norms)} trunk norms, {changed} of {total} gammas change in bfloat16", flush=True)

    cases = _tf_cases(gen_dir, prompt_real_path, model.semantic_begin)
    if only:
        # A short window: the named cases only. A finished case is skipped on a rerun.
        keep = only.split(",")
        cases = [c for c in cases if c["name"] in keep]
        assert cases, f"no case named {keep}"
    ok = _tf_probe(model, state, cases[0]["rows"])
    print(f"patched block equals the original: {ok}", flush=True)
    assert ok

    for variant in chosen:
        state.use_router_int8 = variant in ("router_int8", "convert")
        state.use_shared_int8 = variant == "convert"
        state.isolate = variant == "ref"
        for m, w in zip(norms, narrowed if variant in ("norm_bf16", "convert") else norm_originals):
            m.weight = w
        for case in cases:
            target = out / variant / case["name"]
            if (target / "meta.json").exists():
                continue
            started = time.perf_counter()
            result = _tf_run(model, state, case)
            target.mkdir(parents=True, exist_ok=True)
            for key, value in result.items():
                dtype = "<f4" if value.dtype.kind == "f" else "<i2"
                value.astype(dtype).tofile(target / f"{key}.bin")
            meta = {
                "commit": COMMIT,
                "variant": variant,
                "case": case["name"],
                "rows": case["rows"].tolist(),
                "prompt_rows": case["prompt_rows"],
                "system_rows": case["system_rows"],
                "dim": int(result["hidden"].shape[1]),
                "semantic_width": int(result["semantic"].shape[1]),
                "heads": int(result["heads"].shape[1]),
                "head_vocab": int(result["heads"].shape[2]),
                "layers": len(layers),
                "top_k": TF_TOP_K,
                "arrays": sorted(result),
            }
            (target / "meta.json").write_text(json.dumps(meta))
            print(
                f"{variant} {case['name']}: {len(case['rows'])} rows in "
                f"{time.perf_counter() - started:.1f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB",
                flush=True,
            )
    for m, w in zip(norms, norm_originals):
        m.weight = w


def main() -> None:
    mode, *args = sys.argv[1:]
    modes = {
        "chunking": chunking,
        "tower": tower,
        "prompt": prompt,
        "prompt-real": prompt_real,
        "teacher-forced": teacher_forced,
    }
    modes[mode](*args)


if __name__ == "__main__":
    main()
