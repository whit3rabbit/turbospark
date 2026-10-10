#!/usr/bin/env python3
"""Regenerate the MOSS-TTS-Nano family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/moss_tts_nano, mlx 0.32.3). The true reference
module is imported and run against:

- a deterministic offline character-level tokenizer (encode(a + b) ==
  encode(a) + encode(b), so the reference string concatenation of
  continuation prompts matches the id concatenation the Rust port
  composes) standing in for the sentencepiece tokenizer;
- a seeded tiny 4-book mono MOSS audio tokenizer (8 kHz, codebook 16)
  saved into decode/ for the Rust codec path;
- greedy sampling (do_sample=False) so every draw is deterministic;
- an engineered continue/stop pair: after seeding, the local hidden of
  the first frame is computed and the `wte` rows of the two candidate
  tokens (assistant slot, audio end) are set along plus/minus that
  hidden, so one model keeps generating for the whole frame budget and
  a second stops on the first frame.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_moss_tts_nano_fixtures.py

Outputs into crates/audio/testdata/tts_moss_tts_nano/:

- decode/config.json + decode/model.safetensors + decode/traces.json
    The tiny seeded MOSS audio tokenizer, the reference encode codes
    for a seeded wave, and the reference decode_audio_codes waveform of
    the generated frames.
- generate/config.json + generate/model.safetensors + generate/traces.json
    Seeded tiny MOSS-TTS-Nano (2-layer global transformer, 1-layer
    local transformer, 4 VQ channels, codebook 16), the composed
    voice-clone prompt rows from `build_inference_input_ids`, the
    greedy frames from `generate_audio_token_ids`, and the decoded
    waveform through the reference codec wrapper.
- generate_stop/config.json + generate_stop/model.safetensors +
  generate_stop/traces.json
    The engineered-stop twin: the first continue/stop choice picks the
    audio-end candidate, the loop stops with no frames and no audio.
- prompt/traces.json
    The template piece ids (character-level), the composed voice-clone
    and continuation (with and without reference codes) prompt rows
    from the reference `build_inference_input_ids`.
- sampling/traces.json
    The family sampler on fixed logits: the clamping repetition
    penalty, top-k and top-p masks mapped back onto raw scores (null
    sentinels for negative infinities), the greedy candidate choice,
    and the argmax path.
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
from mlx.utils import tree_flatten, tree_unflatten

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.codec.models.moss_audio_tokenizer.moss_audio_tokenizer import (
    AudioTokenizerConfig,
    MossAudioTokenizer as ReferenceAudioTokenizer,
)

import mlx_audio.tts.models.moss_tts_nano.moss_tts_nano as nano
from mlx_audio.tts.models.moss_tts_nano.sampling import (
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample_assistant_text_token,
)
from mlx_audio.tts.models.moss_tts_nano.text import (
    ASSISTANT_ROLE_PREFIX,
    ASSISTANT_TURN_PREFIX,
    USER_ROLE_PREFIX,
    USER_TEMPLATE_AFTER_REFERENCE,
    USER_TEMPLATE_REFERENCE_PREFIX,
    USER_TEMPLATE_SUFFIX,
)

OUT = Path("../turbospark/crates/audio/testdata/tts_moss_tts_nano")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

TEMPLATE_PIECES = {
    "user_role_prefix": USER_ROLE_PREFIX,
    "user_template_reference_prefix": USER_TEMPLATE_REFERENCE_PREFIX,
    "user_template_after_reference": USER_TEMPLATE_AFTER_REFERENCE,
    "user_template_suffix": USER_TEMPLATE_SUFFIX,
    "assistant_turn_prefix": ASSISTANT_TURN_PREFIX,
    "assistant_role_prefix": ASSISTANT_ROLE_PREFIX,
    "none_reference": "None",
}

PROMPT_TEXT = "hello world today"
REF_TEXT = "reference words here"

# Tiny MOSS-TTS-Nano geometry: 4 VQ channels over an audio vocab of 16,
# 2-layer global transformer and 1-layer local transformer.
TINY_CONFIG = {
    "model_type": "moss_tts_nano",
    "n_vq": 4,
    "audio_vocab_size": 16,
    "audio_codebook_sizes": [16, 16, 16, 16],
    "audio_pad_token_id": 16,
    "pad_token_id": 3,
    "im_start_token_id": 4,
    "im_end_token_id": 5,
    "audio_start_token_id": 6,
    "audio_end_token_id": 7,
    "audio_user_slot_token_id": 8,
    "audio_assistant_slot_token_id": 9,
    "audio_tokenizer_type": "moss-audio-tokenizer-nano",
    "audio_tokenizer_sample_rate": 8000,
    "attn_implementation": "sdpa",
    "local_transformer_layers": 1,
    "gpt2_config": {
        "model_type": "gpt2",
        "vocab_size": 96,
        "n_positions": 512,
        "n_ctx": 512,
        "n_embd": 32,
        "n_layer": 2,
        "n_head": 4,
        "n_inner": 64,
        "activation_function": "gelu_new",
        "layer_norm_epsilon": 1e-5,
        "scale_attn_weights": True,
        "scale_attn_by_inverse_layer_idx": False,
        "position_embedding_type": "rope",
        "rope_base": 10000.0,
    },
}

# Tiny mono 4-book MOSS audio tokenizer (8 kHz, codebook 16); mirrors
# tools/gen_moss_fixtures.py with number_channels 1.
TINY_TOKENIZER = {
    "sample_rate": 8000,
    "downsample_rate": 8,
    "number_channels": 1,
    "enable_channel_interleave": True,
    "causal_transformer_context_duration": 10.0,
    "encoder_kwargs": [
        {"module_type": "PatchedPretransform", "patch_size": 4},
        {
            "module_type": "Transformer",
            "input_dimension": 4,
            "output_dimension": 32,
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 64,
            "causal": True,
            "context_duration": 0.001,
            "conv_layout": False,
            "positional_embedding": "rope",
            "max_period": 10000.0,
            "layer_scale": 0.1,
        },
    ],
    "decoder_kwargs": [
        {
            "module_type": "Transformer",
            "input_dimension": 32,
            "output_dimension": 4,
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 64,
            "causal": True,
            "context_duration": 0.001,
            "conv_layout": False,
            "positional_embedding": "rope",
            "max_period": 10000.0,
            "layer_scale": 0.1,
        },
        {"module_type": "PatchedPretransform", "patch_size": 4},
    ],
    "quantizer_type": "rlfq",
    "quantizer_kwargs": {
        "input_dim": 32,
        "rvq_dim": 32,
        "output_dim": 32,
        "num_quantizers": 4,
        "codebook_size": 16,
        "codebook_dim": 4,
    },
}

MAX_NEW_FRAMES = 4
CANDIDATE_SCALE = 50.0
CONTINUE_SEED = 12
STOP_SEED = 31


class CharTokenizer:
    """Character-level tokenizer over a fixed character inventory; the
    ids are deterministic and encode(a + b) == encode(a) + encode(b)."""

    def __init__(self):
        alphabet = set()
        for piece in TEMPLATE_PIECES.values():
            alphabet.update(piece)
        alphabet.update(PROMPT_TEXT + REF_TEXT)
        self.chars = sorted(alphabet)
        self.vocab = {char: 20 + index for index, char in enumerate(self.chars)}
        assert max(self.vocab.values()) < TINY_CONFIG["gpt2_config"]["vocab_size"]

    def encode(self, text: str, *args, **kwargs) -> list:
        return [self.vocab[char] for char in str(text)]

    def decode(self, token_ids, *args, **kwargs) -> str:
        inverse = {value: key for key, value in self.vocab.items()}
        return "".join(inverse[int(t)] for t in token_ids)


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def masked(arr) -> list:
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(arr, dtype=np.float32).ravel().tolist()
    ]


def tiny_audio_tokenizer() -> ReferenceAudioTokenizer:
    config = AudioTokenizerConfig.from_dict(TINY_TOKENIZER)
    mx.random.seed(501)
    model = ReferenceAudioTokenizer(config)
    flat = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(503)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in flat.items()
    }
    model.load_weights(list(seeded.items()))
    mx.eval(model)
    return model


def seeded_nano(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    flat = {}  # filled by the caller from the live model
    return flat, rng


def build_nano(seed: int) -> "nano.Model":
    mx.random.seed(seed)
    model = nano.Model(dict(TINY_CONFIG))
    rng = np.random.default_rng(seed)
    seeded = {}
    for name, value in dict(tree_flatten(model.parameters())).items():
        seeded[name] = mx.array(
            (rng.standard_normal(value.shape) * 0.3).astype(np.float32)
        )
    model.update(tree_unflatten(seeded))
    return model


def first_choice_hidden(model: "nano.Model", input_ids) -> np.ndarray:
    """The local-transformer hidden state the first continue/stop choice
    reads (`generate_audio_token_ids` frame 1, up to the text head)."""
    embeds = model._build_inputs_embeds(mx.array(input_ids)[None])
    hidden = model.transformer(inputs_embeds=embeds)
    local_hidden = model.local_transformer(inputs_embeds=hidden[:, -1:, :])
    return np.asarray(local_hidden[:, -1, :], dtype=np.float32)[0]


def engineer_candidates(model: "nano.Model", input_ids, continue_run: bool) -> None:
    """Sets the two candidate `wte` rows along plus/minus the first
    frame's local hidden, so the greedy choice deterministically picks
    the assistant slot (continue) or the audio end (stop)."""
    hidden = first_choice_hidden(model, input_ids)
    wte = np.asarray(model.transformer.wte.weight, dtype=np.float32)
    sign = 1.0 if continue_run else -1.0
    wte[9] = sign * CANDIDATE_SCALE * hidden
    wte[7] = -sign * CANDIDATE_SCALE * hidden
    model.transformer.wte.weight = mx.array(wte)
    model.transformer.wte.weight /= 1  # touch to force eval
    mx.eval(model.transformer.wte.weight)


def voice_clone_prompt(model: "nano.Model", tokenizer, codes) -> np.ndarray:
    input_ids, attention_mask = model.build_inference_input_ids(
        text=PROMPT_TEXT,
        tokenizer=tokenizer,
        mode="voice_clone",
        prompt_audio_codes=mx.array(codes),
    )
    assert int(attention_mask.sum()) == input_ids.shape[1], "unpadded batch"
    return np.asarray(input_ids, dtype=np.int32)[0]


def build_codec_case(codes: list, frames: list) -> None:
    tokenizer = tiny_audio_tokenizer()
    traces = {
        "wave": floats(WAVE),
        "encode_codes": [ints(row) for row in codes],
        "frames": [ints(frame) for frame in frames],
    }
    if frames:
        # The reference decode wrapper: [1, T, n_q] codes, n_q quantizers.
        audio = tokenizer.decode_audio_codes(
            mx.array(frames, dtype=mx.int32), num_quantizers=4
        )
        traces["waveform"] = floats(audio)

    case_dir = OUT / "decode"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(case_dir / "model.safetensors"),
        dict(tree_flatten(tokenizer.parameters())),
    )
    (case_dir / "config.json").write_text(json.dumps(TINY_TOKENIZER, indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces))


def build_prompt_case(model: "nano.Model", tokenizer) -> None:
    traces = {"pieces": {}, "cases": []}
    for name, piece in TEMPLATE_PIECES.items():
        traces["pieces"][name] = tokenizer.encode(piece)
    traces["text_ids"] = tokenizer.encode(PROMPT_TEXT)
    traces["ref_text_ids"] = tokenizer.encode(REF_TEXT)

    codes = mx.array(REFERENCE_CODES, dtype=mx.int32)
    records = []

    input_ids, _ = model.build_inference_input_ids(
        text=PROMPT_TEXT,
        tokenizer=tokenizer,
        mode="voice_clone",
        prompt_audio_codes=codes,
    )
    records.append(
        {
            "name": "voice_clone",
            "mode": "voice_clone",
            "rows": [ints(row) for row in input_ids[0]],
            "row_width": int(input_ids.shape[-1]),
        }
    )
    input_ids, _ = model.build_inference_input_ids(
        text=PROMPT_TEXT,
        tokenizer=tokenizer,
        mode="continuation",
    )
    records.append(
        {
            "name": "continuation_text_only",
            "mode": "continuation",
            "rows": [ints(row) for row in input_ids[0]],
            "row_width": int(input_ids.shape[-1]),
        }
    )
    input_ids, _ = model.build_inference_input_ids(
        text=PROMPT_TEXT,
        tokenizer=tokenizer,
        mode="continuation",
        prompt_text=REF_TEXT,
        prompt_audio_codes=codes,
    )
    records.append(
        {
            "name": "continuation_with_reference",
            "mode": "continuation",
            # The reference encodes prompt_text + text as ONE string; the
            # character tokenizer keeps encode(a + b) == encode(a) +
            # encode(b), which is the split the Rust composition needs.
            "rows": [ints(row) for row in input_ids[0]],
            "row_width": int(input_ids.shape[-1]),
        }
    )
    traces["cases"] = records

    case_dir = OUT / "prompt"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps(traces))


def build_generate_case(
    model: "nano.Model",
    tokenizer,
    audio_tokenizer: ReferenceAudioTokenizer,
    prompt_rows: np.ndarray,
    case_name: str,
    continue_run: bool,
) -> list:
    engineer_candidates(model, prompt_rows, continue_run)
    frames = model.generate_audio_token_ids(
        prompt_input_ids=mx.array(prompt_rows),
        max_new_frames=MAX_NEW_FRAMES,
        do_sample=False,
    )
    frames_np = np.asarray(frames, dtype=np.int32).reshape(-1, TINY_CONFIG["n_vq"])
    if continue_run:
        assert len(frames_np) == MAX_NEW_FRAMES, "continue run must fill the budget"
    else:
        assert len(frames_np) == 0, "engineered stop must produce no frames"
    traces = {
        "max_new_frames": MAX_NEW_FRAMES,
        "prompt_rows": prompt_rows.tolist(),
        "frames": frames_np.tolist(),
        "stopped_by": "max_frames" if continue_run else "end_of_text",
    }
    if len(frames_np) > 0:
        audio = audio_tokenizer.decode_audio_codes(
            mx.array(frames_np, dtype=mx.int32), num_quantizers=4
        )
        traces["waveform"] = floats(audio)

    case_dir = OUT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    stored = {
        key: value.astype(mx.float16)
        for key, value in dict(tree_flatten(model.parameters())).items()
    }
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces))
    return traces["frames"]


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

    # The clamping repetition penalty: negative and >= vocab ids clamp to
    # 0; duplicates penalize once.
    logits = vector(16, 3)
    context = [0, 0, 9, -3, 2]
    penalized = apply_repetition_penalty(mx.array(logits)[None, :], mx.array(context), 1.2)
    cases.append(
        {
            "name": "repetition_penalty_clamped",
            "logits": floats(logits),
            "context": context,
            "repetition_penalty": 1.2,
            "penalized": floats(penalized.squeeze(0)),
            "top_k": None,
            "top_k_scores": None,
            "top_p": None,
            "top_p_scores": None,
            "argmax": int(mx.argmax(penalized, axis=-1).item()),
        }
    )

    # top-k then top-p on the temperature-scaled scores, mapped back to
    # the raw values (the reference order).
    logits = vector(64, 11)
    scores = mx.array(logits) / 0.8
    top_k_scores = apply_top_k(scores, 25)
    top_p_scores = apply_top_p(scores, 0.95)
    chained = apply_top_p(top_k_scores, 0.95)
    cases.append(
        {
            "name": "audio_chain",
            "logits": floats(logits),
            "context": [],
            "repetition_penalty": 1.0,
            "temperature": 0.8,
            "penalized": floats(logits),
            "top_k": 25,
            "top_k_scores": masked(top_k_scores),
            "top_p": 0.95,
            "top_p_scores": masked(chained),
            "argmax": int(mx.argmax(chained, axis=-1).item()),
        }
    )

    # Disabled filters (top_k >= vocab, top_p >= 1) pass the scores
    # through unchanged.
    logits = vector(32, 23)
    scores = mx.array(logits) / 1.0
    kept = apply_top_k(scores, 32)
    kept = apply_top_p(kept, 1.0)
    cases.append(
        {
            "name": "filters_disabled",
            "logits": floats(logits),
            "context": [],
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "penalized": floats(logits),
            "top_k": 32,
            "top_k_scores": masked(kept),
            "top_p": 1.0,
            "top_p_scores": None,
            "argmax": int(mx.argmax(kept, axis=-1).item()),
        }
    )

    # The greedy candidate choice over [assistant slot, audio end].
    text_logits = np.zeros(32, dtype=np.float32)
    text_logits[9] = 0.4
    text_logits[7] = 1.7
    picked = sample_assistant_text_token(
        mx.array(text_logits),
        audio_assistant_slot_token_id=9,
        audio_end_token_id=7,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    cases.append(
        {
            "name": "assistant_choice_stops",
            "logits": floats(text_logits),
            "context": [],
            "repetition_penalty": 1.0,
            "picked": int(picked),
            "top_k": None,
            "top_k_scores": None,
            "top_p": None,
            "top_p_scores": None,
            "argmax": int(mx.argmax(mx.array(text_logits)).item()),
        }
    )
    text_logits[9] = 2.4
    picked = sample_assistant_text_token(
        mx.array(text_logits),
        audio_assistant_slot_token_id=9,
        audio_end_token_id=7,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    cases.append(
        {
            "name": "assistant_choice_continues",
            "logits": floats(text_logits),
            "context": [],
            "repetition_penalty": 1.0,
            "picked": int(picked),
            "top_k": None,
            "top_k_scores": None,
            "top_p": None,
            "top_p_scores": None,
            "argmax": int(mx.argmax(mx.array(text_logits)).item()),
        }
    )

    case_dir = OUT / "sampling"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps({"cases": cases}))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = CharTokenizer()
    audio_tokenizer = tiny_audio_tokenizer()

    # Reference audio for the voice-clone prompt: a short seeded wave
    # through the tiny tokenizer (40 samples -> 5 frames).
    global WAVE, REFERENCE_CODES
    rng = np.random.default_rng(507)
    WAVE = (rng.standard_normal(40) * 0.2).astype(np.float32)
    codes = np.asarray(
        audio_tokenizer.encode_audio(mx.array(WAVE), sample_rate=8000),
        dtype=np.int32,
    )
    REFERENCE_CODES = codes.tolist()

    prompt_model = build_nano(1)
    prompt_rows = voice_clone_prompt(prompt_model, tokenizer, REFERENCE_CODES)

    # Search for a continue seed: the engineering pins frame 1, later
    # frames depend on the hidden drift, which varies by seed.
    frames = None
    for seed in range(64):
        candidate = build_nano(seed)
        try:
            got = build_generate_case(
                candidate, tokenizer, audio_tokenizer, prompt_rows, "generate", True
            )
        except AssertionError as error:
            if "continue run" in str(error):
                continue
            raise
        frames = got
        global CONTINUE_SEED
        CONTINUE_SEED = seed
        print(f"continue seed {seed}: {len(got)} frames")
        break
    if frames is None:
        raise SystemExit("no seed produced a full continue run")

    stop_model = build_nano(STOP_SEED)
    build_generate_case(
        stop_model, tokenizer, audio_tokenizer, prompt_rows, "generate_stop", False
    )
    build_codec_case(REFERENCE_CODES, frames)
    build_prompt_case(prompt_model, tokenizer)
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
        "generator": "crates/audio/tools/gen_moss_tts_nano_fixtures.py",
        "continue_seed": CONTINUE_SEED,
        "stop_seed": STOP_SEED,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
