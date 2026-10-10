#!/usr/bin/env python3
"""Regenerate the OuteTTS family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/outetts,
mlx_audio/lm/sample_utils.py, mlx 0.32.3). The reference modules are
imported directly and run against:

- a deterministic offline PreTrainedTokenizerFast (word-level vocabulary
  over the special tokens; c1 ids 200+i, c2 ids 1300+i, "<|im_end|>" at
  id 2) standing in for the network AutoTokenizer, so the reference
  PromptProcessor builds its real audio token maps;
- a seeded tiny two-book DAC (n_codebooks 2, codebook 64, hop 8, 24 kHz);
- a seeded tiny llama backbone driven by the reference
  `mlx_audio.lm.generate.generate_step` with the family sampler (greedy:
  temperature 0) and logits processors, stopping after the tokenizer eos
  the way `stream_generate` + the family loop append it.

The dac_interface multi-chunk decode at this commit is documented in the
family README: `mx.concatenate(recons, axis=-1)` joins the
`[1, samples, 1]` decode chunks along the size-1 channel axis, which
only works while the decode is a single chunk (up to 4096 codes). The
decode traces here run the same reference chunked `from_codes` + decode
steps with the chunks joined along samples (the single-chunk path the
reference takes) before the unchanged `process_audio_array` call, whose
squeeze branch receives `[1, 1, samples]` exactly as upstream.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_outetts_fixtures.py

Outputs into crates/audio/testdata/tts_outetts/:

- dac/model.safetensors + dac/config.json + dac/traces.json
    The tiny seeded codec, the c1/c2 audio token maps from the mock
    tokenizer, extract_audio_from_tokens cases on fixed ids, the chunked
    two-book decode waveform (joined along samples; the reference
    concat only works single-chunk), the loudness measurement, and the
    loudness-normalized waveforms (full and pad-path short inputs).
- prompt/traces.json
    text_normalizations, chunk_text, merge_speaker_text, and the
    create_codes / get_completion_prompt strings on a pinned speaker.
- generate/model.safetensors + generate/config.json + generate/traces.json
    Seeded tiny llama (vocab covering the mock tokenizer), the pinned
    prompt ids, two greedy scenarios (stop at eos, stop at max_tokens)
    with generated ids, extracted codebooks, and the stop scenario's
    loudness-normalized waveform.
- sampling/traces.json
    The sampler chain on fixed logits: logit bias, repetition penalty,
    log-softmax, top-p, min-p, top-k (the family's make_sampler order).
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.codec.models.descript.dac import DAC
from mlx_audio.lm.generate import generate_step
from mlx_audio.lm.models.llama import Model as LlamaModel, ModelArgs as LlamaArgs
from mlx_audio.lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    make_logits_processors,
    make_sampler,
)
from mlx_audio.dsp import integrated_loudness

from mlx_audio.tts.models.outetts.dac_interface import process_audio_array
from mlx_audio.tts.models.outetts.prompt_processor import PromptProcessor
from mlx_audio.tts.models.outetts.tokens import SpecialTokens

OUT = Path("../turbospark/crates/audio/testdata/tts_outetts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# Mock tokenizer ids (deterministic; the Rust tests read the maps from
# the traces, never these constants).
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
C1_BASE = 200
C2_BASE = 1300

# Tiny DAC geometry: two books, codebook 64, hop 2*4 = 8, 24 kHz like the
# deployed dac-speech-24khz-1.5kbps profile.
TINY_DAC = dict(
    encoder_dim=8,
    encoder_rates=[2, 4],
    latent_dim=None,
    decoder_dim=16,
    decoder_rates=[4, 2],
    n_codebooks=2,
    codebook_size=64,
    codebook_dim=4,
    sample_rate=24000,
)
CODEBOOK = 64
CODEBOOK_TOKENS = 1025

# Pinned prompt ids in the mock vocabulary: bos, text, text_end,
# audio_start, word_start, words, features, code, c1/c2 pairs, word_end,
# then a fresh word_start (the reference completion prompt ends there).
PROMPT_IDS = (
    [BOS_ID, 40, 41, 42, 41]
    + [3, 5]
    + [7, 41, 42, 41, 9, 10]
    + [200 + 3, 1300 + 5, 200 + 11, 1300 + 2]
    + [8, 7]
)

AMPLIFY_CODE = 30.0
AMPLIFY_EOS = 25.0
FIXTURE_PENALTY = 5.0
BACKBONE_SEED = 6


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level vocabulary over the special tokens; the c1/c2 ids land
    at 200+i and 1300+i so the tiny backbone can cover them."""
    vocab = {"[UNK]": UNK_ID, "<|im_start|>": BOS_ID, "<|im_end|>": EOS_ID}
    vocab["<|text_start|>"] = 3
    vocab["<|text_end|>"] = 4
    vocab["<|audio_start|>"] = 5
    vocab["<|audio_end|>"] = 6
    vocab["<|word_start|>"] = 7
    vocab["<|word_end|>"] = 8
    vocab["<|features|>"] = 9
    vocab["<|code|>"] = 10
    for i in range(CODEBOOK_TOKENS):
        vocab[f"<|c1_{i}|>"] = C1_BASE + i
        vocab[f"<|c2_{i}|>"] = C2_BASE + i
    for word in ("hello", "world", "snow", "light"):
        vocab[word] = 11 + len(vocab) % 12
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
    )


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def masked(arr) -> list:
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(arr, dtype=np.float32).ravel().tolist()
    ]


def tiny_dac() -> DAC:
    mx.random.seed(7)
    return DAC(**TINY_DAC)


def dump_dac(dac: DAC, case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    flat = dict(tree_flatten(dac.parameters()))
    mx.save_safetensors(str(case_dir / "model.safetensors"), flat)
    (case_dir / "config.json").write_text(json.dumps(TINY_DAC, indent=2) + "\n")


def reference_decode(dac: DAC, codebook1, codebook2) -> np.ndarray:
    """The DacInterface.decode chunk loop (chunk_length 4096) on the two
    books. MLX convs are channels-last, so each decode chunk comes back
    [1, samples, 1]; the reference concatenates its chunks along axis -1,
    which only works for a single chunk (a second chunk mismatches on the
    samples axis) before its swapaxes hands [1, 1, samples] to
    process_audio_array's squeeze branch. The chunks here are joined
    along the samples axis, which is the path the reference takes for
    any input up to the 4096-code chunk length and the clear intent
    beyond it (see the family README)."""
    codes = mx.array([list(codebook1), list(codebook2)])[None]
    chunks = []
    for start in range(0, codes.shape[-1], 4096):
        chunk = codes[..., start : start + 4096]
        z = dac.quantizer.from_codes(chunk)[0]
        chunks.append(np.array(dac.decode(z)))
    return np.concatenate(chunks, axis=1)[0, :, 0]


def build_decode_case(dac: DAC, processor: PromptProcessor) -> None:
    traces = {}
    # get_audio_token_map stores {token id: code index}; the traces carry
    # the inverted index -> token id tables the Rust AudioTokenMaps
    # constructor takes.
    c1_map = [None] * CODEBOOK_TOKENS
    for token_id, index in processor.c1.items():
        c1_map[index] = int(token_id)
    c2_map = [None] * CODEBOOK_TOKENS
    for token_id, index in processor.c2.items():
        c2_map[index] = int(token_id)
    traces["c1_map"] = c1_map
    traces["c2_map"] = c2_map

    # Extraction cases on fixed ids: interleaved codes, unknown ids, and
    # book-length truncation.
    known = (
        [C1_BASE + 3, C2_BASE + 5, 41, C1_BASE + 11, C2_BASE + 2]
        + [7, 9]
        + [C1_BASE + 1, C1_BASE + 4, C2_BASE + 6]
    )
    cases = [
        ("interleaved", known, [3, 11, 1], [5, 2, 6]),
        ("unknown_ids_skipped", [41, 42, C1_BASE + 7, C2_BASE + 9], [7], [9]),
        ("longer_book_truncated", [C1_BASE + 1, C1_BASE + 2, C2_BASE + 8], [1], [8]),
        ("empty", [41, 42, 7], [], []),
    ]
    records = []
    for name, tokens, want1, want2 in cases:
        got1, got2 = processor.extract_audio_from_tokens(list(tokens))
        assert ints(got1) == want1, name
        assert ints(got2) == want2, name
        records.append(
            {
                "name": name,
                "tokens": ints(tokens),
                "codebook1": ints(got1),
                "codebook2": ints(got2),
            }
        )
    traces["extract_cases"] = records

    # Chunked decode (4500 codes -> two chunks) and the loudness stage.
    flat_c1 = [(3 * i + 1) % CODEBOOK for i in range(4500)]
    flat_c2 = [(5 * i + 2) % CODEBOOK for i in range(4500)]
    wave = reference_decode(dac, flat_c1, flat_c2)
    assert wave.shape[0] >= 9600, "decode must clear one loudness block"
    traces["flat_c1"] = ints(flat_c1)
    traces["flat_c2"] = ints(flat_c2)
    traces["dac_wave"] = floats(wave)
    traces["loudness"] = float(integrated_loudness(wave.astype(np.float64), 24000, 0.4))
    processed = process_audio_array(mx.array(wave), 24000)
    traces["processed_wave"] = floats(processed)

    # The pad path: a short input padded to one block, measured,
    # normalized, and trimmed back.
    short = wave[:2400]
    processed_short = process_audio_array(mx.array(short), 24000)
    traces["short_wave"] = floats(short)
    traces["processed_wave_short"] = floats(processed_short)
    assert len(traces["processed_wave_short"]) == 2400

    case_dir = OUT / "dac"
    dump_dac(dac, case_dir)
    (case_dir / "traces.json").write_text(json.dumps(traces))


PINNED_SPEAKER = {
    "text": "The snow falls quietly.",
    "words": [
        {
            "word": "The",
            "duration": 0.2,
            "c1": [3, 11, 5],
            "c2": [8, 2, 9],
            "features": {"energy": 10, "spectral_centroid": 15, "pitch": 45},
        },
        {
            "word": "snow",
            "duration": 0.35,
            "c1": [7, 13],
            "c2": [1, 4],
            "features": {"energy": 32, "spectral_centroid": 40, "pitch": 51},
        },
        {
            "word": "falls",
            "duration": 0.31,
            "c1": [21],
            "c2": [17],
            "features": {"energy": 25, "spectral_centroid": 33, "pitch": 48},
        },
    ],
    "global_features": {"energy": 22, "spectral_centroid": 30, "pitch": 47},
}


def build_prompt_case(processor: PromptProcessor) -> None:
    import mlx_audio.tts.models.outetts.outetts as outetts_model

    traces = {"speaker": PINNED_SPEAKER}

    normalizations = [
        ("collapse_and_strip", "  Hello\t\tworld\n today  "),
        ("ellipsis", "wait for it\u2026 ok"),
        ("curly_quotes", "\u201cquoted\u201d and \u2018single\u2019"),
        ("dashes", "first \u2013 second \u2014 third"),
        ("control_removed", "a\x01b\x7fc"),
    ]
    records = []
    for name, text in normalizations:
        records.append(
            {
                "name": name,
                "input": text,
                "output": PromptProcessor.text_normalizations(text),
            }
        )
    traces["normalization_cases"] = records

    chunks = [
        ("sentences", "One two three four. Short! A longer sentence here yes?", 3),
        ("long_grouped", " ".join(["word"] * 35), 30),
        ("cjk_split", "\u3053\u3093\u306b\u3061\u306f\u3002\u4e16\u754c\uff01", 30),
    ]
    records = []
    for name, text, max_words in chunks:
        got = outetts_model.Model.chunk_text(None, text, max_words)
        records.append(
            {"name": name, "input": text, "max_words": max_words, "chunks": got}
        )
    traces["chunk_cases"] = records

    merges = [
        ("latin_period", "New text here", "The snow falls quietly."),
        ("no_ending_punct", "Another line", "spoken words"),
        ("question", "What next", "is that so?"),
        ("empty_speaker", "Only input", ""),
        ("cjk", "\u65b0\u3057\u3044\u30c6\u30ad\u30b9\u30c8", "\u4eca\u65e5\u306f\u6674\u308c"),
    ]
    records = []
    for name, input_text, speaker_text in merges:
        merged, suffix = processor.merge_speaker_text(input_text, speaker_text)
        records.append(
            {
                "name": name,
                "input": input_text,
                "speaker": speaker_text,
                "merged": merged,
                "suffix": suffix,
            }
        )
    traces["merge_cases"] = records

    speaker = json.loads(json.dumps(PINNED_SPEAKER))
    traces["create_codes"] = processor.create_codes(speaker["words"])
    traces["global_features"] = processor.get_global_features(
        speaker["global_features"]
    )

    prompt_cases = [
        (
            "with_speaker",
            "Hello world, the light grows.",
            json.loads(json.dumps(PINNED_SPEAKER)),
        ),
        (
            "needs_normalization",
            "  wait\u2026 for it\u2014ok  ",
            json.loads(json.dumps(PINNED_SPEAKER)),
        ),
    ]
    records = []
    for name, text, speaker in prompt_cases:
        prompt = processor.get_completion_prompt(text, speaker)
        records.append(
            {
                "name": name,
                "text": text,
                "speaker": speaker,
                "prompt": prompt,
                "last_word": speaker["words"][-1]["word"],
            }
        )
    traces["prompt_cases"] = records

    case_dir = OUT / "prompt"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps(traces))


def tiny_backbone(seed: int, vocab_size: int, amplify_eos: float):
    args = LlamaArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=vocab_size,
        head_dim=None,
        max_position_embeddings=256,
        num_key_value_heads=2,
        attention_bias=False,
        mlp_bias=False,
        rope_theta=10000.0,
        rope_traditional=False,
        rope_scaling=None,
        tie_word_embeddings=True,
        layer_types=None,
        sliding_window=None,
    )
    mx.random.seed(seed)
    model = LlamaModel(args)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith("norm.weight"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    embed = np.array(flat["model.embed_tokens.weight"], dtype=np.float32)
    for c in range(CODEBOOK):
        embed[C1_BASE + c] *= AMPLIFY_CODE
        embed[C2_BASE + c] *= AMPLIFY_CODE
    embed[EOS_ID] *= amplify_eos
    stored = {k: v.astype(mx.float16) for k, v in flat.items()}
    stored["model.embed_tokens.weight"] = mx.array(embed).astype(mx.float16)
    model.update(tree_unflatten({k: v.astype(mx.float32) for k, v in stored.items()}))
    return model, stored, args


def run_generate(model, prompt_ids, max_tokens):
    """The family loop through stream_generate: reference generate_step
    with the family sampler (greedy here) and logits processors; every
    sampled token is appended, including the eos response stream_generate
    emits last."""
    sampler = make_sampler(0.0, 0.9, min_p=0.05, top_k=40)
    processors = make_logits_processors(None, FIXTURE_PENALTY, 64)
    generated = []
    for token, _ in generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=processors,
    ):
        value = int(token)
        generated.append(value)
        if value == EOS_ID:
            break
    return generated


def build_generate_case(dac: DAC, processor: PromptProcessor, vocab_size: int) -> None:
    best = None
    for seed in range(64):
        model, stored, args = tiny_backbone(seed, vocab_size, AMPLIFY_EOS)
        stop_run = run_generate(model, PROMPT_IDS, 48)
        length_run = run_generate(model, PROMPT_IDS, 10)
        c1, c2 = processor.extract_audio_from_tokens(stop_run)
        len_c1, len_c2 = processor.extract_audio_from_tokens(length_run)
        stop_ok = (
            stop_run
            and stop_run[-1] == EOS_ID
            and len(c1) >= 7
            and len(c2) >= 7
            and min(c1) >= 0
            and max(c1) < CODEBOOK
            and min(c2) >= 0
            and max(c2) < CODEBOOK
        )
        # Ten draws cannot yield more than five pairs; the length
        # scenario only needs a non-eos trajectory with at least one
        # pair per book.
        length_ok = (
            EOS_ID not in length_run
            and len(length_run) == 10
            and len(len_c1) >= 1
            and len(len_c2) >= 1
            and max(len_c1) < CODEBOOK
            and max(len_c2) < CODEBOOK
        )
        if stop_ok and length_ok:
            best = (model, stored, args, stop_run, length_run, seed)
            print(f"backbone seed {seed}: acceptable greedy fixtures")
            break
    if best is None:
        raise SystemExit("no backbone seed produced acceptable greedy fixtures")

    model, stored, args, stop_run, length_run, seed = best
    if seed != BACKBONE_SEED:
        print(f"NOTE: search picked seed {seed}; BACKBONE_SEED is {BACKBONE_SEED}")
    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    config = dict(vars(args))
    config["tokenizer_name"] = None
    config["sample_rate"] = 24000
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # The stop scenario's waveform through the reference decode chain on
    # the extracted books (chunked from_codes decode + mono
    # process_audio_array).
    c1, c2 = processor.extract_audio_from_tokens(stop_run)
    wave = reference_decode(dac, c1, c2)
    processed = process_audio_array(mx.array(wave), 24000)

    traces = {
        "stop_token_id": EOS_ID,
        "prompt": PROMPT_IDS,
        "scenarios": [
            {
                "name": "stop",
                "max_tokens": 48,
                "repetition_penalty": FIXTURE_PENALTY,
                "generated": ints(stop_run),
                "codebook1": ints(c1),
                "codebook2": ints(c2),
                "stopped_by": "stop_token",
                "waveform": floats(processed),
            },
            {
                "name": "length",
                "max_tokens": 10,
                "repetition_penalty": FIXTURE_PENALTY,
                "generated": ints(length_run),
                "codebook1": ints(processor.extract_audio_from_tokens(length_run)[0]),
                "codebook2": ints(processor.extract_audio_from_tokens(length_run)[1]),
                "stopped_by": "max_tokens",
            },
        ],
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))


def build_sampling_case() -> None:
    cases = []

    def vector(n, offset):
        values = np.array(
            [
                float(np.sin(0.73 * (i + offset)) * 4.0 + np.cos(0.31 * i) * 1.5)
                for i in range(n)
            ],
            dtype=np.float32,
        )
        assert len(set(values.tolist())) == n, "fixed logits must have no ties"
        return values

    # Full family chain: logit bias + repetition penalty (defaults 1.1
    # over 64), log-softmax, top-p 0.9, min-p 0.05, top-k 40.
    logits = vector(160, 0)
    context = [5, 3, 5, 42, 7]
    bias = {2: 0.5, 9: -1.25}
    processors = make_logits_processors(bias, 1.1, 64)
    processed = mx.array(logits)[None, :]
    for processor in processors:
        processed = processor(mx.array(context), processed)
    processed = processed.squeeze(0)
    logprobs = processed - mx.logsumexp(processed, keepdims=True)
    top_p_out = apply_top_p(logprobs, 0.9)
    min_p_out = apply_min_p(top_p_out, 0.05, 1)
    top_k_out = apply_top_k(min_p_out, 40)
    cases.append(
        {
            "name": "chain",
            "logits": floats(logits),
            "context": context,
            "logit_bias": [[2, 0.5], [9, -1.25]],
            "repetition_penalty": 1.1,
            "repetition_context_size": 64,
            "processed_logits": floats(processed),
            "logprobs": floats(logprobs),
            "top_p": 0.9,
            "top_p_logprobs": masked(top_p_out),
            "min_p": 0.05,
            "min_tokens_to_keep": 1,
            "min_p_logprobs": masked(min_p_out),
            "top_k": 40,
            "top_k_logprobs": masked(top_k_out),
            "argmax": int(mx.argmax(top_k_out, axis=-1).item()),
        }
    )

    # min-p alone with min_tokens_to_keep = 3 so the guaranteed-survivor
    # path is exercised.
    logits = vector(96, 7)
    logprobs = mx.array(logits) - mx.logsumexp(mx.array(logits), keepdims=True)
    min_p_out = apply_min_p(logprobs, 0.25, 3)
    cases.append(
        {
            "name": "min_p_keep_3",
            "logits": floats(logits),
            "context": [],
            "logit_bias": [],
            "repetition_penalty": 0.0,
            "repetition_context_size": 64,
            "processed_logits": floats(logits),
            "logprobs": floats(logprobs),
            "top_p": 0.0,
            "top_p_logprobs": None,
            "min_p": 0.25,
            "min_tokens_to_keep": 3,
            "min_p_logprobs": masked(min_p_out),
            "top_k": 0,
            "top_k_logprobs": None,
            "argmax": int(mx.argmax(min_p_out, axis=-1).item()),
        }
    )

    # Repetition penalty with a context longer than the window plus
    # duplicates (the reference gathers and scatters once per token).
    # Context ids stay inside the logits width: the reference gather
    # silently zeroes out-of-range ids, while the Rust processor refuses
    # them, and a real trajectory never leaves the vocabulary.
    logits = vector(128, 29)
    context = [11] * 3 + [4] * 2 + list(range(12, 100))
    processor = make_logits_processors(None, 1.1, 64)[0]
    processed = processor(mx.array(context), mx.array(logits)[None, :]).squeeze(0)
    cases.append(
        {
            "name": "penalty_only",
            "logits": floats(logits),
            "context": context,
            "logit_bias": [],
            "repetition_penalty": 1.1,
            "repetition_context_size": 64,
            "processed_logits": floats(processed),
            "logprobs": floats(processed - mx.logsumexp(processed, keepdims=True)),
            "top_p": 0.0,
            "top_p_logprobs": None,
            "min_p": 0.0,
            "min_tokens_to_keep": 1,
            "min_p_logprobs": None,
            "top_k": 0,
            "top_k_logprobs": None,
            "argmax": int(mx.argmax(processed, axis=-1).item()),
        }
    )

    case_dir = OUT / "sampling"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps({"cases": cases}))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer()
    processor = PromptProcessor(tokenizer)
    dac = tiny_dac()
    build_decode_case(dac, processor)
    build_prompt_case(processor)
    build_generate_case(dac, processor, len(tokenizer.get_vocab()))
    build_sampling_case()

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_outetts_fixtures.py",
        "backbone_seed": BACKBONE_SEED,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
