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
"""

import importlib.util
import json
import sys

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


def main() -> None:
    mode, arg = sys.argv[1], sys.argv[2]
    {"chunking": chunking, "tower": tower, "prompt": prompt, "prompt-real": prompt_real}[mode](arg)


if __name__ == "__main__":
    main()
