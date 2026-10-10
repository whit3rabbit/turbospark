#!/usr/bin/env python3
"""Regenerate the MOSS-TTS family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/moss_tts, mlx 0.32.3). The true reference modules
run in full: the delay-pattern variant (`generate_delay_pattern_ids`
with the delay bookkeeping, the keep/exclude text sets, and the per-book
heads), the v1.5 local-transformer variant (`generate_v15_local_ids`
over the qwen3 global transformer, the tied-out text/audio heads, and
the moss_tts_nano gpt2 local transformer), both processors
(`MossTTSDelayProcessor` unified codes with the delay pattern,
`MossTTSLocalV15Processor` row layout), the family sampler
(`moss_tts/sampling.py`), and the `_decode_generated_audio` segment
trim/decode through the MOSS audio tokenizer, with:

- a deterministic offline character-level tokenizer with multi-char
  special tokens (encode is additive over the composed content, so the
  reference string composition matches the id composition the Rust port
  builds) standing in for the checkpoint tokenizer;
- a seeded tiny 4-book mono MOSS audio tokenizer (8 kHz, codebook 16);
- greedy sampling throughout (temperature 0 for the delay variant,
  do_sample=False for the v1.5 local variant) so no draw is needed;
- engineered continue/stop pairs: the delay text-head rows for the
  special tokens are set along the prompt's last hidden state, and the
  v1.5 text-head candidate rows along the first frame's local hidden,
  so one model fills the frame budget (or cycles the delay bookkeeping
  to a clean stop) and the twin stops on the first choice.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_moss_tts_fixtures.py

Outputs into crates/audio/testdata/tts_moss_tts/:

- decode/  tiny codec + reference encode codes + decode waveform
- delay_generate/  delay variant, assistant-slot loop over the frame
    budget (keep-inside-audio text branch, every book sampled, penalty
    over the full per-book history), decoded waveform
- delay_stop/  delay variant, full bookkeeping to the stop (forced delay
    slots, audio_end, im_end, the pad row), decoded waveform
- delay_prompt/  template pieces and unified-code grids (generation with
    and without reference codes, continuation with truncation)
- delay_pattern/  apply_delay_pattern / apply_de_delay_pattern traces
- sampling/  the family sampler on fixed logits (penalty, top-k / top-p
    masks mapped back onto raw scores, greedy paths)
- v15_generate/  v1.5 local variant, four greedy frames with the per-book
    generated-history penalty, decoded waveform
- v15_stop/  engineered-stop twin (no frames, no audio)
- v15_prompt/  row layouts (voice clone with reference codes, generation
    without, continuation)
- manifest.json  file list with SHA-256 digests and generator metadata

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

import mlx_audio.tts.models.moss_tts.moss_tts as moss
from mlx_audio.tts.models.moss_tts.config import ModelConfig
from mlx_audio.tts.models.moss_tts.processor import (
    AUDIO_PLACEHOLDER,
    MossTTSDelayProcessor,
    MossTTSLocalV15Processor,
    apply_de_delay_pattern,
    apply_delay_pattern,
)
from mlx_audio.tts.models.moss_tts.sampling import (
    apply_repetition_penalty_delay_pattern,
    apply_top_k,
    apply_top_p,
    sample_token,
)

OUT = Path("../turbospark/crates/audio/testdata/tts_moss_tts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

PROMPT_TEXT = "hello world today"
REF_TEXT = "reference words here"

# Special token strings the character tokenizer round-trips; the ids are
# the tiny configs below.
SPECIAL_TOKENS = {
    3: "<|pad|>",
    4: "<|im_start|>",
    5: "<|im_end|>",
    6: "<|audio_start|>",
    7: "<|audio_end|>",
    8: "<|audio_user|>",
    9: "<|audio_assistant|>",
    10: "<|audio_delay|>",
}

TINY_QWEN3 = {
    "model_type": "qwen3",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-5,
    "vocab_size": 96,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}

# Tiny delay-pattern geometry: 4 VQ channels over an audio vocab of 16.
DELAY_CONFIG = {
    "model_type": "moss_tts_delay",
    "n_vq": 4,
    "audio_vocab_size": 16,
    "audio_codebook_sizes": [16, 16, 16, 16],
    "audio_pad_token_id": 16,
    "audio_pad_code": 16,
    "pad_token_id": 3,
    "im_start_token_id": 4,
    "im_end_token_id": 5,
    "audio_start_token_id": 6,
    "audio_end_token_id": 7,
    "audio_user_slot_token_id": 8,
    "audio_assistant_slot_token_id": 9,
    "audio_assistant_delay_slot_token_id": 10,
    "sampling_rate": 8000,
    "language_config": dict(TINY_QWEN3),
}

# Tiny v1.5 local-transformer geometry (the nano gpt2 local transformer).
V15_CONFIG = {
    "model_type": "moss_tts_local",
    "n_vq": 4,
    "audio_vocab_size": 16,
    "audio_codebook_sizes": [16, 16, 16, 16],
    "audio_pad_token_id": 16,
    "audio_pad_code": 16,
    "pad_token_id": 3,
    "im_start_token_id": 4,
    "im_end_token_id": 5,
    "audio_start_token_id": 6,
    "audio_end_token_id": 7,
    "audio_user_slot_token_id": 8,
    "audio_assistant_slot_token_id": 9,
    "audio_assistant_delay_slot_token_id": 10,
    "sampling_rate": 8000,
    "local_transformer_layers": 1,
    "local_text_head_mode": "full_vocab",
    "language_config": dict(TINY_QWEN3),
    "gpt2_config": {
        "model_type": "gpt2",
        "vocab_size": 96,
        "n_positions": 8,
        "n_ctx": 8,
        "n_embd": 32,
        "n_layer": 1,
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

DELAY_MAX_NEW_TOKENS = 6
V15_MAX_NEW_FRAMES = 4
CANDIDATE_SCALE = 50.0
STOP_SEED = 31


class CharTokenizer:
    """Character-level tokenizer with multi-char special tokens; encode
    is additive over composed strings and round-trips the special ids."""

    def __init__(self):
        alphabet = set()
        for text in (
            "user\n",
            "assistant\n",
            "\n",
            "<user_inst>\n- Reference(s):\n- Instruction:\n- Tokens:\n- Quality:\n"
            "- Sound Event:\n- Ambient Sound:\n- Language:\n- Text:\n</user_inst>",
            PROMPT_TEXT,
            REF_TEXT,
            "[S1]:",
            "None",
        ):
            alphabet.update(text)
        self.special_strings = dict(SPECIAL_TOKENS)
        alphabet.update("".join(self.special_strings.values()))
        self.chars = sorted(alphabet)
        self.vocab = {char: 20 + index for index, char in enumerate(self.chars)}
        for token_id, string in self.special_strings.items():
            self.vocab[string] = token_id
        assert max(self.vocab.values()) < 96, "tiny vocab overflow"

    def encode(self, text: str, *args, **kwargs) -> list:
        del args, kwargs
        text = str(text)
        specials = sorted(self.special_strings.values(), key=len, reverse=True)
        ids = []
        position = 0
        while position < len(text):
            for special in specials:
                if text.startswith(special, position):
                    ids.append(self.vocab[special])
                    position += len(special)
                    break
            else:
                ids.append(self.vocab[text[position]])
                position += 1
        return ids

    def decode(self, token_ids, *args, **kwargs) -> str:
        inverse = {value: key for key, value in self.vocab.items()}
        return "".join(inverse[int(t)] for t in token_ids)

    def convert_ids_to_tokens(self, token_id: int):
        inverse = {value: key for key, value in self.vocab.items()}
        return inverse.get(int(token_id), "")


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


def seed_model(model, seed: int):
    rng = np.random.default_rng(seed)
    seeded = {}
    for name, value in dict(tree_flatten(model.parameters())).items():
        seeded[name] = mx.array(
            (rng.standard_normal(value.shape) * 0.3).astype(np.float32)
        )
    model.update(tree_unflatten(seeded))
    return model


def save_model(case_dir: Path, model, config: dict) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    stored = {
        key: value.astype(mx.float16)
        for key, value in dict(tree_flatten(model.parameters())).items()
        if not key.startswith("audio_tokenizer.")
    }
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def write_traces(case_dir: Path, traces: dict) -> None:
    (case_dir / "traces.json").write_text(json.dumps(traces))


def delay_prompt(tokenizer, codes) -> np.ndarray:
    """The voice-clone generation prompt rows from the reference
    processor (user message with one reference block + generation
    prompt)."""
    config = ModelConfig.from_dict(dict(DELAY_CONFIG))
    processor = MossTTSDelayProcessor(tokenizer, config)
    conversation = [processor.build_user_message(text=PROMPT_TEXT, reference=[codes])]
    batch = processor([conversation], mode="generation")
    assert int(batch["attention_mask"].sum()) == batch["input_ids"].shape[1]
    return np.asarray(batch["input_ids"], dtype=np.int32)[0]


def probe_delay_hidden(model, input_ids) -> np.ndarray:
    """The prompt's last post-norm hidden state (cacheless probe)."""
    embeds = model._build_inputs_embeds(mx.array(input_ids)[None])
    dummy = mx.zeros(embeds.shape[:2], dtype=mx.int32)
    hidden = model.language_model(dummy, input_embeddings=embeds)
    return np.asarray(hidden[:, -1, :], dtype=np.float32)[0]


def engineer_delay_head(model, input_ids, stop_run: bool) -> None:
    """Sets the text-head rows for the special tokens along the prompt's
    last hidden state. The assistant loop seeds: audio_start, then (stop
    run) the delay bookkeeping to audio_end and im_end, or (generate run)
    the assistant slot for the whole budget."""
    hidden = probe_delay_hidden(model, input_ids)
    weight = np.asarray(model.lm_heads[0].weight, dtype=np.float32)
    if stop_run:
        # im_end outranks audio_start so the audio_end state falls through
        # to the stop instead of re-engaging the audio branch.
        weight[6] = 3.0 * hidden  # audio_start
        weight[10] = 2.6 * hidden  # delay slot beats the gen slot
        weight[9] = 2.5 * hidden
        weight[7] = 2.2 * hidden  # audio_end (forced when delayed == n_vq)
        weight[5] = 4.0 * hidden  # im_end
    else:
        weight[6] = 3.0 * hidden  # audio_start engages the audio branch
        weight[9] = 4.0 * hidden  # assistant slot wins the keep set
        weight[10] = 3.9 * hidden
    model.lm_heads[0].weight = mx.array(weight)
    mx.eval(model.lm_heads[0].weight)


def untrimmed_delay_wave(config, outputs, audio_tokenizer) -> list:
    """The trim-free decode of the extracted segments (the reference
    `_decode_generated_audio` without the first-segment trim), so the
    delay-path codec parity is pinned even when the trim ratio consumes
    the whole first segment."""
    _, generation_ids = outputs[0]
    n_vq = int(config["n_vq"])
    audio_codes = np.asarray(generation_ids, dtype=np.int32)[:, 1:]
    ded = np.asarray(
        moss.apply_de_delay_pattern(mx.array(audio_codes.tolist())), dtype=np.int32
    )
    is_pad = np.all(ded == config["audio_pad_code"], axis=1)
    non_pad = [i for i, v in enumerate(is_pad.tolist()) if not v]
    if not non_pad:
        return []
    breaks = [0]
    for i in range(1, len(non_pad)):
        if non_pad[i] != non_pad[i - 1] + 1:
            breaks.append(i)
    breaks.append(len(non_pad))
    audio_segments = []
    for break_start, break_end in zip(breaks[:-1], breaks[1:]):
        indices = non_pad[break_start:break_end]
        segment = ded[indices[0] : indices[-1] + 1]
        audio = audio_tokenizer.decode_audio_codes(
            mx.array(segment.tolist()), num_quantizers=n_vq
        )
        audio_segments.append(floats(audio))
    return [v for segment in audio_segments for v in segment]


def build_delay_case(
    seed: int,
    tokenizer,
    audio_tokenizer: ReferenceAudioTokenizer,
    prompt_rows: np.ndarray,
    case_name: str,
    stop_run: bool,
    max_new_tokens: int,
) -> dict:
    model = moss.Model(dict(DELAY_CONFIG))
    seed_model(model, seed)
    engineer_delay_head(model, prompt_rows, stop_run)
    outputs = model.generate_delay_pattern_ids(
        mx.array(prompt_rows)[None],
        max_new_tokens=max_new_tokens,
        text_temperature=0.0,
        audio_temperature=0.0,
        text_top_p=0.8,
        audio_top_p=0.8,
        text_top_k=50,
        audio_top_k=25,
        audio_repetition_penalty=1.2,
    )
    assert len(outputs) == 1
    start_length, generation_ids = outputs[0]
    rows = np.asarray(generation_ids, dtype=np.int32)
    text_tokens = [int(row[0]) for row in rows]
    traces = {
        "max_new_tokens": max_new_tokens,
        "prompt_rows": prompt_rows.tolist(),
        "start_length": int(start_length),
        "rows": rows.tolist(),
        "text_tokens": text_tokens,
    }
    if stop_run:
        assert 10 in text_tokens, "delay slot must engage"
        assert text_tokens[-1] == 5, "im_end is the final row (the loop breaks on it)"
        has_code = any(
            code != 16 for row in rows for code in row[1:]
        )
        assert has_code, "the stop run must sample real codes"
    else:
        assert 9 in text_tokens, "assistant slot must win the keep set"
        assert 5 not in text_tokens, "no stop inside the budget"
        has_code = any(
            code != 16 for row in rows for code in row[1:]
        )
        assert has_code, "the generate run must sample real codes"

    model.audio_tokenizer = audio_tokenizer
    audio, token_count = model._decode_generated_audio(outputs)
    traces["waveform"] = floats(audio)
    traces["token_count"] = int(token_count)
    assert token_count > 0
    traces["waveform_untrimmed"] = untrimmed_delay_wave(DELAY_CONFIG, outputs,
        audio_tokenizer)
    model.audio_tokenizer = None

    case_dir = OUT / case_name
    save_model(case_dir, model, DELAY_CONFIG)
    write_traces(case_dir, traces)
    return traces


def content_pieces(tokenizer, content: str, codes_list: list) -> list:
    """Splits a rendered message content on the `<|audio|>` placeholders
    into the recorded piece list the Rust prompt builder consumes."""
    parts = content.split(AUDIO_PLACEHOLDER)
    assert len(parts) == len(codes_list) + 1, "placeholder count"
    pieces = []
    for index, part in enumerate(parts):
        if part:
            pieces.append({"kind": "text", "ids": tokenizer.encode(part)})
        if index < len(codes_list):
            pieces.append({"kind": "audio", "codes": np.asarray(codes_list[index], dtype=np.int32).tolist()})
    return pieces


def build_delay_prompt_case(tokenizer) -> None:
    config = ModelConfig.from_dict(dict(DELAY_CONFIG))
    processor = MossTTSDelayProcessor(tokenizer, config)
    traces = {
        "special_ids": {str(k): v for k, v in SPECIAL_TOKENS.items()},
        "template": {
            "user_role": tokenizer.encode("user\n"),
            "assistant_role": tokenizer.encode("assistant\n"),
            "after_im_end": tokenizer.encode("\n"),
        },
        "cases": [],
    }
    traces["text_ids"] = tokenizer.encode(PROMPT_TEXT)

    codes = mx.array(REFERENCE_CODES, dtype=mx.int32)
    # Generation with one reference block.
    message = processor.build_user_message(text=PROMPT_TEXT, reference=[codes])
    conversation = [message]
    batch = processor([conversation], mode="generation")
    traces["cases"].append(
        {
            "name": "generation_reference",
            "mode": "generation",
            "messages": [
                {
                    "role": "user",
                    "pieces": content_pieces(tokenizer, message["content"], message["audio_codes_list"]),
                }
            ],
            "rows": np.asarray(batch["input_ids"], dtype=np.int32)[0].tolist(),
        }
    )
    # Generation without references: the all-pad grid.
    message = processor.build_user_message(text=PROMPT_TEXT)
    conversation = [message]
    batch = processor([conversation], mode="generation")
    traces["cases"].append(
        {
            "name": "generation_text_only",
            "mode": "generation",
            "messages": [
                {
                    "role": "user",
                    "pieces": content_pieces(tokenizer, message["content"], message["audio_codes_list"]),
                }
            ],
            "rows": np.asarray(batch["input_ids"], dtype=np.int32)[0].tolist(),
        }
    )
    # Continuation: user text message + assistant audio message; the
    # final block loses its last n_vq - 1 delayed rows.
    user_message = processor.build_user_message(text=REF_TEXT + PROMPT_TEXT)
    assistant_message = processor.build_assistant_message(audio_codes_list=[codes])
    conversation = [user_message, assistant_message]
    batch = processor([conversation], mode="continuation")
    traces["cases"].append(
        {
            "name": "continuation",
            "mode": "continuation",
            "messages": [
                {
                    "role": "user",
                    "pieces": content_pieces(
                        tokenizer, user_message["content"],
                        user_message["audio_codes_list"]
                    ),
                },
                {
                    "role": "assistant",
                    "pieces": content_pieces(
                        tokenizer, assistant_message["content"],
                        assistant_message["audio_codes_list"]
                    ),
                },
            ],
            "rows": np.asarray(batch["input_ids"], dtype=np.int32)[0].tolist(),
        }
    )

    case_dir = OUT / "delay_prompt"
    case_dir.mkdir(parents=True, exist_ok=True)
    write_traces(case_dir, traces)


def build_delay_pattern_case() -> None:
    rng = np.random.default_rng(707)
    codes = rng.integers(0, 16, size=(5, 4)).astype(np.int32)
    delayed = apply_delay_pattern(mx.array(codes), 16)
    restored = apply_de_delay_pattern(delayed)
    traces = {
        "codes": codes.tolist(),
        "pad_code": 16,
        "n_vq": 4,
        "delayed": np.asarray(delayed, dtype=np.int32).tolist(),
        "restored": np.asarray(restored, dtype=np.int32).tolist(),
    }
    assert np.array_equal(np.asarray(restored), codes), "round trip"
    case_dir = OUT / "delay_pattern"
    case_dir.mkdir(parents=True, exist_ok=True)
    write_traces(case_dir, traces)


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
    penalized = apply_repetition_penalty_delay_pattern(
        mx.array(logits)[None, :], mx.array(context), 1.2
    )
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
        }
    )

    # top-k then top-p on the caller-scaled scores, mapped back to the
    # raw values (the reference order).
    logits = vector(64, 11)
    scores = mx.array(logits) / 0.8
    top_k_scores = apply_top_k(scores, 25)
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
        }
    )

    # Disabled filters pass the scores through unchanged.
    logits = vector(32, 23)
    kept = apply_top_k(mx.array(logits), 32)
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
        }
    )

    # Greedy sample_token: the penalty shifts the argmax; the masked
    # keep/exclude path pins the surviving scores.
    logits = vector(24, 41)
    token = sample_token(
        mx.array(logits)[None, :],
        prev_tokens=mx.array([[0, 2]]),
        repetition_penalty=2.0,
        top_p=0.8,
        top_k=25,
        do_sample=False,
    )
    cases.append(
        {
            "name": "greedy_with_penalty",
            "logits": floats(logits),
            "context": [0, 2],
            "repetition_penalty": 2.0,
            "greedy_token": int(token.item()),
            "top_k": None,
            "top_k_scores": None,
            "top_p": None,
            "top_p_scores": None,
        }
    )

    case_dir = OUT / "sampling"
    case_dir.mkdir(parents=True, exist_ok=True)
    write_traces(case_dir, {"cases": cases})


def v15_prompt(tokenizer, codes) -> np.ndarray:
    """The voice-clone generation prompt rows from the v1.5 processor."""
    config = ModelConfig.from_dict(dict(V15_CONFIG))
    processor = MossTTSLocalV15Processor(tokenizer, config)
    conversation = [processor.build_user_message(text=PROMPT_TEXT, reference=[codes])]
    batch = processor([conversation], mode="generation")
    assert int(batch["attention_mask"].sum()) == batch["input_ids"].shape[1]
    return np.asarray(batch["input_ids"], dtype=np.int32)[0]


def probe_v15_hidden(model, input_ids) -> np.ndarray:
    """The first continue/stop choice's local hidden state."""
    embeds = model._build_v15_local_inputs_embeds(mx.array(input_ids)[None])
    dummy = mx.zeros(embeds.shape[:2], dtype=mx.int32)
    hidden = model.transformer(dummy, input_embeddings=embeds)
    local_hidden = model.local_transformer(inputs_embeds=hidden[:, -1:, :])
    return np.asarray(local_hidden[:, -1, :], dtype=np.float32)[0]


def engineer_v15_candidates(model, input_ids, continue_run: bool) -> None:
    """Sets the two candidate text-head rows along plus/minus the first
    frame's local hidden, so the greedy choice picks the assistant slot
    (continue) or the audio end (stop)."""
    hidden = probe_v15_hidden(model, input_ids)
    weight = np.asarray(model.text_lm_head.weight, dtype=np.float32)
    sign = 1.0 if continue_run else -1.0
    weight[9] = sign * CANDIDATE_SCALE * hidden
    weight[7] = -sign * CANDIDATE_SCALE * hidden
    model.text_lm_head.weight = mx.array(weight)
    mx.eval(model.text_lm_head.weight)


def build_v15_case(
    seed: int,
    tokenizer,
    audio_tokenizer: ReferenceAudioTokenizer,
    prompt_rows: np.ndarray,
    case_name: str,
    continue_run: bool,
) -> list:
    model = moss.Model(dict(V15_CONFIG))
    seed_model(model, seed)
    engineer_v15_candidates(model, prompt_rows, continue_run)
    outputs = model.generate_v15_local_ids(
        mx.array(prompt_rows)[None],
        max_new_tokens=V15_MAX_NEW_FRAMES,
        do_sample=False,
        text_temperature=1.0,
        audio_temperature=1.0,
        text_top_p=0.8,
        audio_top_p=0.8,
        text_top_k=50,
        audio_top_k=25,
        audio_repetition_penalty=1.2,
        use_kv_cache=True,
    )
    assert len(outputs) == 1
    start_length, generation_ids = outputs[0]
    rows = np.asarray(generation_ids, dtype=np.int32)
    traces = {
        "max_new_frames": V15_MAX_NEW_FRAMES,
        "prompt_rows": prompt_rows.tolist(),
        "start_length": int(start_length),
        "rows": rows.tolist(),
    }
    if continue_run:
        assert len(rows) - count_prompt_tail(prompt_rows) == V15_MAX_NEW_FRAMES, (
            "continue run must fill the budget"
        )
    else:
        assert len(rows) == count_prompt_tail(prompt_rows), (
            "engineered stop must produce no frames"
        )

    model.audio_tokenizer = audio_tokenizer
    audio, token_count = model._decode_generated_audio(outputs)
    traces["waveform"] = floats(audio)
    traces["token_count"] = int(token_count)
    model.audio_tokenizer = None

    case_dir = OUT / case_name
    save_model(case_dir, model, V15_CONFIG)
    write_traces(case_dir, traces)
    return traces


def count_prompt_tail(prompt_rows: np.ndarray) -> int:
    """Rows from the last audio_start position (the v1.5 tail start)."""
    column = np.asarray(prompt_rows)[:, 0]
    positions = np.where(column == 6)[0]
    return int(len(column) - positions[-1]) if len(positions) else 0


def build_v15_prompt_case(tokenizer) -> None:
    config = ModelConfig.from_dict(dict(V15_CONFIG))
    processor = MossTTSLocalV15Processor(tokenizer, config)
    from mlx_audio.tts.models.moss_tts.processor import (
        LOCAL_V15_ASSISTANT_ROLE_PREFIX,
        LOCAL_V15_ASSISTANT_TURN_PREFIX,
        LOCAL_V15_USER_ROLE_PREFIX,
        LOCAL_V15_USER_TEMPLATE_AFTER_REFERENCE_SUFFIX,
        LOCAL_V15_USER_TEMPLATE_REFERENCE_PREFIX,
        LOCAL_V15_USER_TEMPLATE_SUFFIX,
        _render_local_v15_user_prompt_after_reference,
    )

    traces = {
        "template": {
            "user_role_prefix": tokenizer.encode(LOCAL_V15_USER_ROLE_PREFIX),
            "user_template_reference_prefix": tokenizer.encode(
                LOCAL_V15_USER_TEMPLATE_REFERENCE_PREFIX
            ),
            "user_template_after_reference": tokenizer.encode(
                _render_local_v15_user_prompt_after_reference()
            ),
            "user_template_suffix": tokenizer.encode(LOCAL_V15_USER_TEMPLATE_SUFFIX),
            "assistant_turn_prefix": tokenizer.encode(LOCAL_V15_ASSISTANT_TURN_PREFIX),
            "assistant_role_prefix": tokenizer.encode(LOCAL_V15_ASSISTANT_ROLE_PREFIX),
            "none_reference": tokenizer.encode("None"),
        },
        "cases": [],
    }
    traces["text_ids"] = tokenizer.encode(PROMPT_TEXT)
    traces["ref_text_ids"] = tokenizer.encode(REF_TEXT)

    codes = mx.array(REFERENCE_CODES, dtype=mx.int32)
    conversation = [processor.build_user_message(text=PROMPT_TEXT, reference=[codes])]
    batch = processor([conversation], mode="generation")
    traces["cases"].append(
        {
            "name": "voice_clone",
            "mode": "generation",
            "rows": np.asarray(batch["input_ids"], dtype=np.int32)[0].tolist(),
        }
    )
    conversation = [processor.build_user_message(text=PROMPT_TEXT)]
    batch = processor([conversation], mode="generation")
    traces["cases"].append(
        {
            "name": "generation_text_only",
            "mode": "generation",
            "rows": np.asarray(batch["input_ids"], dtype=np.int32)[0].tolist(),
        }
    )
    conversation = [
        processor.build_user_message(text=REF_TEXT + PROMPT_TEXT),
        processor.build_assistant_message(audio_codes_list=[codes]),
    ]
    batch = processor([conversation], mode="continuation")
    traces["cases"].append(
        {
            "name": "continuation",
            "mode": "continuation",
            "rows": np.asarray(batch["input_ids"], dtype=np.int32)[0].tolist(),
        }
    )

    case_dir = OUT / "v15_prompt"
    case_dir.mkdir(parents=True, exist_ok=True)
    write_traces(case_dir, traces)


def build_codec_case(codes: list) -> None:
    tokenizer = tiny_audio_tokenizer()
    traces = {
        "wave": floats(WAVE),
        "encode_codes": [ints(row) for row in codes],
    }
    if codes:
        audio = tokenizer.decode_audio_codes(
            mx.array(codes, dtype=mx.int32), num_quantizers=4
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = CharTokenizer()
    audio_tokenizer = tiny_audio_tokenizer()

    # Reference audio for the voice-clone prompts: a short seeded wave
    # through the tiny tokenizer (40 samples -> 5 frames).
    global WAVE, REFERENCE_CODES
    rng = np.random.default_rng(507)
    WAVE = (rng.standard_normal(40) * 0.2).astype(np.float32)
    codes = np.asarray(
        audio_tokenizer.encode_audio(mx.array(WAVE), sample_rate=8000),
        dtype=np.int32,
    )
    REFERENCE_CODES = codes.tolist()

    delay_rows = delay_prompt(tokenizer, REFERENCE_CODES)
    delay_seed = None
    for seed in range(64):
        try:
            build_delay_case(
                seed, tokenizer, audio_tokenizer, delay_rows, "delay_generate", False,
                DELAY_MAX_NEW_TOKENS,
            )
        except AssertionError as error:
            if "must" in str(error):
                continue
            raise
        delay_seed = seed
        print(f"delay generate seed {seed}")
        break
    if delay_seed is None:
        raise SystemExit("no seed produced a delay generate run")

    stop_seed = None
    for seed in range(64):
        try:
            build_delay_case(
                seed, tokenizer, audio_tokenizer, delay_rows, "delay_stop", True,
                DELAY_MAX_NEW_TOKENS + 4,
            )
        except AssertionError as error:
            if "must" in str(error):
                continue
            raise
        stop_seed = seed
        print(f"delay stop seed {seed}")
        break
    if stop_seed is None:
        raise SystemExit("no seed produced a delay stop run")

    build_delay_prompt_case(tokenizer)
    build_delay_pattern_case()
    build_sampling_case()

    v15_rows = v15_prompt(tokenizer, REFERENCE_CODES)
    v15_seed = None
    for seed in range(64):
        try:
            build_v15_case(
                seed, tokenizer, audio_tokenizer, v15_rows, "v15_generate", True
            )
        except AssertionError as error:
            if "must" in str(error):
                continue
            raise
        v15_seed = seed
        print(f"v15 generate seed {seed}")
        break
    if v15_seed is None:
        raise SystemExit("no seed produced a v15 generate run")

    build_v15_case(
        STOP_SEED, tokenizer, audio_tokenizer, v15_rows, "v15_stop", False
    )
    build_v15_prompt_case(tokenizer)
    build_codec_case(REFERENCE_CODES)

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_moss_tts_fixtures.py",
        "delay_generate_seed": delay_seed,
        "delay_stop_seed": stop_seed,
        "v15_generate_seed": v15_seed,
        "v15_stop_seed": STOP_SEED,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
