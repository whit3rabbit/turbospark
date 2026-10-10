#!/usr/bin/env python3
"""Regenerate the soprano TTS family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/soprano,
mlx_audio/lm/sample_utils.py, mlx 0.32.3). The reference modules run
directly against a tiny seeded model:

- text.py `clean_text` and `Model._preprocess_text` on fixed inputs (the
  Rust port pins the same string pipeline);
- the reference `SopranoDecoder` (interpolate + VocosBackbone + ISTFTHead)
  on a pinned hidden matrix;
- the reference `Model.stream_generate` loop (greedy: temperature 0) over
  a seeded tiny qwen3 with an untied lm head, driven with pinned prompt
  ids, then the `generate` assembly (hidden concat, decoder,
  token-count trim) applied verbatim. Three scenarios fall out of one
  backbone: the length run stops at max_tokens, and its first and fourth
  draws become the stop ids of the immediate-stop and late-stop runs (so
  the stopping token itself is dropped, exactly like the reference break
  before the forward pass).

The generate() entry point needs a Hugging Face tokenizer; tokenization
is caller-owned in the Rust port, so the fixture pins prompt ids and the
loop, and pins the text pipeline separately.

Weights are saved in the raw checkpoint layout the Rust loader maps: the
LM under `model.*` (plus root `lm_head.weight` for the untied head) and
the decoder under `decoder.*`, exactly the tensors the reference
`Model.sanitize` reads (it strips `model.` and prefixes
`language_model.` at load; the decoder tensors pass through as float32).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_soprano_fixtures.py

Outputs into crates/audio/testdata/tts_soprano/:

- text/traces.json
    clean_text cases (numbers, abbreviations, symbols, punctuation) and
    _preprocess_text pieces with the merge rules.
- decoder/model.safetensors + decoder/config.json + decoder/traces.json
    The tiny seeded decoder and its waveform on a pinned hidden matrix.
- generate/model.safetensors + generate/config.json + generate/traces.json
    The tiny seeded qwen3 + decoder and the three greedy scenarios with
    generated ids and waveforms.
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

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.tts.models.soprano.soprano import (
    DecoderConfig,
    Model as SopranoTtsModel,
    ModelConfig,
)
from mlx_audio.tts.models.soprano.text import clean_text

OUT = Path("../turbospark/crates/audio/testdata/tts_soprano")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

VOCAB = 256
HIDDEN = 32
PROMPT_IDS = [10, 42, 7, 19, 3, 200, 15, 88, 5, 44]
STOP_ID = 1  # the mock pad token id the loop stops on
MAX_TOKENS = 8

# Tiny decoder geometry: 16-wide ConvNeXt (2 blocks), n_fft 32, hop 8,
# upscale 4, so one token covers 4 * 8 = 32 samples.
DECODER_CONFIG = {
    "decoder_num_layers": 2,
    "decoder_dim": 16,
    "decoder_intermediate_dim": 48,
    "hop_length": 8,
    "n_fft": 32,
    "upscale": 4,
    "input_kernel": 1,
    "dw_kernel": 3,
    "token_size": 32,
    "receptive_field": 4,
}

QWEN3_ARGS = {
    "model_type": "qwen3",
    "hidden_size": HIDDEN,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-5,
    "vocab_size": VOCAB,
    "max_position_embeddings": 256,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "rope_scaling": None,
}

BACKBONE_SEED = 11


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def build_model():
    config = ModelConfig(
        **QWEN3_ARGS,
        sample_rate=32000,
        decoder_config=DecoderConfig(**DECODER_CONFIG),
        model_path=None,
    )
    mx.random.seed(BACKBONE_SEED)
    model = SopranoTtsModel(config)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith(("norm.weight", "q_norm.weight", "k_norm.weight")):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    # Store as F16 and keep the upcast F32 view on the module, so the
    # traces see the same values the Rust f32 path loads.
    stored = {k: v.astype(mx.float16) for k, v in flat.items()}
    model.load_weights(list({k: v.astype(mx.float32) for k, v in stored.items()}.items()), strict=False)
    return model, stored


def run_stream(model, prompt_ids, max_tokens, stop_ids):
    """The reference stream_generate loop with the family stop condition
    applied where the reference applies it: on the sampled token, before
    the forward pass and its hidden-state yield (so the stopping token
    never reaches the stack)."""
    collected = []
    for token, hidden in model.stream_generate(
        mx.array(prompt_ids),
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=0.95,
    ):
        if token is not None:
            value = int(token.reshape(-1)[0])
            if value in stop_ids:
                break
        collected.append((token, hidden))
    return collected


def assemble(model, collected):
    """The generate() assembly after stream_generate: hidden concat,
    decoder, and the token-count trim."""
    all_hidden = [hidden for _, hidden in collected]
    hidden_states = mx.concatenate(all_hidden, axis=1)
    audio = model.decoder(hidden_states)
    token_count = len(collected)
    token_size = model.config.decoder_config.token_size
    audio_length = token_count * token_size - token_size
    if audio_length > 0:
        audio = audio[0, -audio_length:]
    else:
        audio = audio[0]
    return audio


def map_checkpoint_names(model) -> dict:
    """Raw checkpoint layout: the LM tensors under `model.*` (the HF
    names the reference sanitize strips), the untied head at the root,
    and the decoder under `decoder.*` (already shaped that way)."""
    flat = dict(tree_flatten(model.parameters()))
    stored = {}
    for name, weight in flat.items():
        if name.startswith("language_model.lm_head"):
            stored[name.removeprefix("language_model.")] = weight
        elif name.startswith("language_model."):
            stored["model." + name.removeprefix("language_model.")] = weight
        else:
            stored[name] = weight
    return stored


def checkpoint_config() -> dict:
    return {
        **QWEN3_ARGS,
        "sample_rate": 32000,
        "decoder_config": DECODER_CONFIG,
        "model_path": None,
    }


def build_text_case() -> None:
    traces = {}
    clean_inputs = [
        ("plain", "Hello world."),
        (
            "number_year",
            "In 1994 something happened, then 2000, 2005, and 1234 more.",
        ),
        ("number_ordinal", "The 1st, 2nd, 21st, 100th, and 1113rd items."),
        ("number_money", "It costs $1,234.56 or $5, and $1.2.3 is odd."),
        ("number_suffix", "Roughly 5K users and 3M total, #7 overall."),
        ("abbreviations", "Mr. and Mrs. Smith met Dr. Who at 5 kHz on 2 MB."),
        ("symbols", "A & B @ 50% costs +5 ~ here: ok; sure."),
        ("dedup_punct", "What?! Really..... now,,,, maybe.,. end"),
        ("whitespace", "  multiple   spaces\tand\nnewlines  "),
        ("case", "MiXeD CaSe TEXT."),
    ]
    records = []
    for name, text in clean_inputs:
        got = clean_text(text)
        records.append({"name": name, "input": text, "output": got})
    traces["clean_cases"] = records

    preprocess_inputs = [
        ("split", "One sentence here. Another follows! A third?", 30),
        (
            "merge_short",
            "Short. Also short. A considerably longer sentence that stands alone.",
            30,
        ),
        ("merge_last", "Tiny. Then a sentence long enough to stand by itself.", 30),
        ("no_split", "no terminal punctuation here", 30),
        ("trailing_space", "Ends with a period.   ", 30),
        ("min_length_zero", "Tiny. Off to a fresh start here indeed.", 0),
    ]
    records = []
    for name, text, min_length in preprocess_inputs:
        got = SopranoTtsModel._preprocess_text(None, [text], min_length)
        records.append(
            {
                "name": name,
                "input": text,
                "min_length": min_length,
                "prompts": [piece[0] for piece in got],
            }
        )
    traces["preprocess_cases"] = records

    case_dir = OUT / "text"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps(traces))


def build_decoder_case(model) -> None:
    np.random.seed(5)
    positions = 6
    hidden = np.random.uniform(-1.0, 1.0, (positions, HIDDEN)).astype(np.float32)
    audio = model.decoder(mx.array(hidden[None]))[0]
    traces = {
        "hidden": floats(hidden.reshape(-1)),
        "positions": positions,
        "waveform": floats(audio),
    }
    case_dir = OUT / "decoder"
    case_dir.mkdir(parents=True, exist_ok=True)
    stored = map_checkpoint_names(model)
    decoder_only = {k: v for k, v in stored.items() if k.startswith("decoder.")}
    mx.save_safetensors(str(case_dir / "model.safetensors"), decoder_only)
    (case_dir / "config.json").write_text(json.dumps(checkpoint_config(), indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces))


def build_generate_case(model) -> None:
    # The length run exhausts max_tokens; none of its draws may be the
    # mock pad id, and its first and fourth draws become the stop ids of
    # the immediate-stop and late-stop runs on the same backbone. The
    # collected list carries the prefill row first (the reference yields
    # it with a None token).
    length_run = run_stream(model, PROMPT_IDS, MAX_TOKENS, set())
    assert len(length_run) == MAX_TOKENS + 1, "length scenario must exhaust max_tokens"
    generated = [
        int(token.reshape(-1)[0]) for token, _ in length_run if token is not None
    ]
    assert len(generated) == MAX_TOKENS
    assert STOP_ID not in generated, "length run must not draw the pad id"
    immediate_stop_id = generated[0]
    late_stop_id = generated[3]

    immediate_run = run_stream(model, PROMPT_IDS, MAX_TOKENS, {immediate_stop_id})
    assert len(immediate_run) == 1, "immediate stop keeps only the prefill row"
    late_run = run_stream(model, PROMPT_IDS, MAX_TOKENS, {late_stop_id})
    assert len(late_run) == 4, "late stop fires after three tokens"

    scenarios = []
    for name, run, stop_ids, stopped in [
        ("immediate_stop", immediate_run, [immediate_stop_id], "stop_token"),
        ("late_stop", late_run, [late_stop_id], "stop_token"),
        ("length", length_run, [], "max_tokens"),
    ]:
        audio = assemble(model, run)
        scenarios.append(
            {
                "name": name,
                "prompt": PROMPT_IDS,
                "max_tokens": MAX_TOKENS,
                "stop_token_ids": stop_ids,
                "generated": [int(token.reshape(-1)[0]) for token, _ in run if token is not None],
                "stopped_by": stopped,
                "waveform": floats(audio),
            }
        )

    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), map_checkpoint_names(model))
    (case_dir / "config.json").write_text(json.dumps(checkpoint_config(), indent=2) + "\n")
    traces = {
        "stop_token_id": STOP_ID,
        "prompt": PROMPT_IDS,
        "scenarios": scenarios,
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, _ = build_model()
    build_text_case()
    build_decoder_case(model)
    build_generate_case(model)

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_soprano_fixtures.py",
        "backbone_seed": BACKBONE_SEED,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
