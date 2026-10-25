#!/usr/bin/env python3
"""Regenerate the arktts (Audio8 TTS) family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/arktts, mlx 0.32.3). The true reference module is
imported and driven against:

- a deterministic offline PreTrainedTokenizerFast (word-level vocabulary
  carrying the Mistral chat control tokens) standing in for the network
  AutoTokenizer, injected via `model._tokenizer` (the reference consults
  it before any from_pretrained);
- tiny seeded codec instances built from subclasses of the reference
  `ArkttsCodec` classes that only shrink the hardcoded widths (the
  decoder strides stay (8, 8, 4, 2), the quantizer upsample stays two
  kernel-2 stride-2 stages, and every forward method is inherited
  reference code);
- greedy sampling (`do_sample=False`) for the deterministic generation
  cases, and a recorded `mx.random.uniform` stream for the sampled case
  (the exponential race consumes one uniform per vocabulary entry; mlx
  RNG is not reproduced on the Rust side, the stream is replayed);

Cases (testdata/tts_arktts/):

- codec/    decode-only codec parity: codes -> waveform through the
            residual VQ `from_codes`, the post window transformer, the
            upsample chain and the Snake decoder, with per-stage traces
            for bisection, plus a clamped-codes replay. Checkpoint holds
            only the tensors the decode path loads.
- pure_attention/  the 0.6b flavor (pure-attention slow stack, tied
            full-vocabulary semantic head) at tiny dims: prompt segment
            strings, greedy `generate_codes` end to end, and the codec
            waveform.
- sampled/  the same checkpoint family with `do_sample=True` and the
            recorded uniform stream (semantic normal + RAS-high draws of
            the full filtered width, one codebook-width draw per
            remaining codebook position).
- voice/    the reference-voice path: the tiny codec ENCODES a seeded
            wave into reference codes (recorded; the Rust encode is a
            documented refusal), and greedy generation runs with the
            reference codes riding the prompt.
- falcon/   the 0.1b flavor (falcon_h1 hybrid slow stack over the lm
            falcon backbone, compact codebook_size + 1 semantic head,
            runtime composite embedding multiplier): greedy end to end.
- eos/, falcon_eos/  seed-searched checkpoints whose first greedy
            semantic pick is EOS, so generation stops with no frames
            and the model-level generate refuses.
- sampling/traces.json  the sampler math on fixed logits: the legacy
            top-k/top-p filter (inclusive cumulative softmax, rank 0
            survives, -inf passes through) and the exponential race with
            recorded per-entry uniforms.
- prompt/traces.json  `_prepare_prompt` grids (no reference and with
            synthetic reference codes) and `_prompt_segments` part
            strings for both forms.

Every case is replayed on the Python side before freezing (same weights,
same streams, bitwise-identical outputs; the established recording
caveat). The script is deterministic: fixed seeds, no network access,
pinned to the CPU device (the Metal selective-scan kernels are not
exercised; the Mamba recurrence is shared math).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_arktts_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

import mlx_audio.tts.models.arktts.arktts as arktts
import mlx_audio.tts.models.arktts.codec as arktts_codec
from mlx_audio.tts.models.arktts.arktts import ModelConfig

OUT = Path("../turbospark/crates/audio/testdata/tts_arktts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# --- tiny TTS geometry (both flavors) ---------------------------------------
VOCAB = 512
DIM = 32
LAYERS = 2
HEADS = 4
LOCAL_HEADS = 2
HEAD_DIM = 8
INTERMEDIATE = 64
MAX_SEQ_LEN = 64
ROPE_BASE = 500000.0
CODEBOOK_SIZE = 64  # == codec semantic codebook size
NUM_CODEBOOKS = 3  # == 1 semantic + codec residual books
SEMANTIC_BEGIN = 400
SEMANTIC_END = SEMANTIC_BEGIN + CODEBOOK_SIZE - 1
EOS_ID = 499
PAD_ID = 0
FAST_LAYERS = 2
FAST_DIM = 24  # != DIM so the fast_project_in seam runs
FAST_HEADS = 3
FAST_LOCAL = 1
FAST_HEAD_DIM = 8
FAST_INTERMEDIATE = 48
RAS_WINDOW = 4

# --- tiny codec geometry (the Rust CodecConfig reads these keys) ------------
CODEC_HIDDEN = 64
CODEC_INPUT_DIM = 64
CODEC_DECODER_CHANNELS = 32
CODEC_SEMANTIC_SIZE = 64
CODEC_RESIDUAL_BOOKS = 2
CODEC_RESIDUAL_SIZE = 64
CODEBOOK_DIM = 8
POST_LAYERS = 2
POST_HEADS = 4
POST_LOCAL_HEADS = 2
POST_INTERMEDIATE = 96

TARGET_TEXT = "hello there friend"
REFERENCE_TEXT = "a calm voice speaks slowly"

CODEC_SEED = 7
PURE_SEED = 101
SAMPLED_SEED = 211
VOICE_SEED = 307
FALCON_SEED = 401
GREEDY_MAX_TOKENS = 10

WORDS = [
    "a",
    "calm",
    "convert",
    "following:",
    "friend",
    "hello",
    "provided",
    "reference",
    "slowly",
    "speech",
    "speaks",
    "text",
    "the",
    "there",
    "to",
    "voice",
    "Text:",
    "Speech:",
]
CONTROL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|audio|>",
    "<|speaker:0|>",
]


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def masked(arr) -> list:
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(arr, dtype=np.float32).ravel().tolist()
    ]


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level vocabulary with the Mistral chat control tokens the
    prompt segment strings use."""
    vocab = {"[UNK]": 1}
    next_id = 2
    for token in CONTROL_TOKENS:
        vocab[token] = next_id
        next_id += 1
    for word in WORDS:
        vocab[word] = next_id
        next_id += 1
    assert next_id < SEMANTIC_BEGIN, "tokenizer ids must stay below the semantic range"
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


class TinyEncoder(arktts_codec.ArkttsEncoder):
    """ArkttsEncoder with a tiny start width (dims double through the same
    four stride blocks; the last block keeps one transformer layer)."""

    def __init__(self):
        dim = 8
        modules = [arktts_codec.ArkttsCausalConv1d(1, dim, kernel_size=7)]
        for stride, transformer_layers in zip((2, 4, 8, 8), (0, 0, 0, 1)):
            dim *= 2
            modules.append(
                arktts_codec.ArkttsEncoderBlock(dim, stride, transformer_layers)
            )
        modules.extend(
            (
                arktts_codec.ArkttsSnake1d(dim),
                arktts_codec.ArkttsCausalConv1d(dim, CODEC_INPUT_DIM, kernel_size=3),
            )
        )
        self.block = modules


class TinyDownsampleQuantizer(arktts_codec.ArkttsDownsampleQuantizer):
    """ArkttsDownsampleQuantizer with tiny widths: the quantizers,
    upsample chain and post module are the ones decode runs; the downsample
    stages and pre module keep the structure at tiny dims (encode only)."""

    def __init__(self, config=None):
        self.semantic_quantizer = arktts_codec.ArkttsResidualQuantizer(
            CODEC_INPUT_DIM, 1, CODEC_SEMANTIC_SIZE, CODEBOOK_DIM
        )
        self.quantizer = arktts_codec.ArkttsResidualQuantizer(
            CODEC_INPUT_DIM, CODEC_RESIDUAL_BOOKS, CODEC_RESIDUAL_SIZE, CODEBOOK_DIM
        )
        self.downsample = [
            [
                arktts_codec.ArkttsCausalConv1d(
                    CODEC_HIDDEN, CODEC_HIDDEN, kernel_size=2, stride=2
                ),
                arktts_codec.ArkttsConvNeXtBlock(CODEC_HIDDEN),
            ],
            [
                arktts_codec.ArkttsCausalConv1d(
                    CODEC_HIDDEN, CODEC_HIDDEN, kernel_size=2, stride=2
                ),
                arktts_codec.ArkttsConvNeXtBlock(CODEC_HIDDEN),
            ],
        ]
        self.upsample = [
            [
                arktts_codec.ArkttsCausalConvTranspose1d(
                    CODEC_HIDDEN, CODEC_HIDDEN, kernel_size=2, stride=2
                ),
                arktts_codec.ArkttsConvNeXtBlock(CODEC_HIDDEN),
            ],
            [
                arktts_codec.ArkttsCausalConvTranspose1d(
                    CODEC_HIDDEN, CODEC_HIDDEN, kernel_size=2, stride=2
                ),
                arktts_codec.ArkttsConvNeXtBlock(CODEC_HIDDEN),
            ],
        ]
        pre_config = arktts_codec.ArkttsCodecTransformerConfig(
            n_layer=1, n_head=4, dim=CODEC_HIDDEN, intermediate_size=POST_INTERMEDIATE
        )
        post_config = arktts_codec.ArkttsCodecTransformerConfig(
            n_layer=POST_LAYERS,
            n_head=POST_HEADS,
            n_local_heads=POST_LOCAL_HEADS,
            dim=CODEC_HIDDEN,
            intermediate_size=POST_INTERMEDIATE,
        )
        self.pre_module = arktts_codec.ArkttsCodecWindowTransformer(
            pre_config, CODEC_HIDDEN, window_size=128
        )
        self.post_module = arktts_codec.ArkttsCodecWindowTransformer(
            post_config, CODEC_HIDDEN, window_size=128
        )


class TinyDecoder(arktts_codec.ArkttsDecoder):
    """ArkttsDecoder at decoder_channels 32; the stride structure (8, 8, 4, 2)
    and the block layout are the reference's."""

    def __init__(self):
        channels = CODEC_DECODER_CHANNELS
        modules = [
            arktts_codec.ArkttsCausalConv1d(
                CODEC_INPUT_DIM, channels, kernel_size=7
            )
        ]
        for index, stride in enumerate((8, 8, 4, 2)):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            modules.append(
                arktts_codec.ArkttsDecoderBlock(input_dim, output_dim, stride)
            )
        modules.extend(
            (
                arktts_codec.ArkttsSnake1d(output_dim),
                arktts_codec.ArkttsCausalConv1d(output_dim, 1, kernel_size=7),
            )
        )
        self.model = modules


class TinyCodec(arktts_codec.ArkttsCodec):
    """ArkttsCodec assembling the tiny submodules; every forward method is
    inherited reference code."""

    def __init__(self, config=None):
        self.encoder = TinyEncoder()
        self.quantizer = TinyDownsampleQuantizer(config)
        self.decoder = TinyDecoder()


def codec_config_dict() -> dict:
    """The codec_* config keys the Rust CodecConfig::from_tts_config reads;
    the published checkpoint carries none of them and keeps its defaults."""
    return {
        "codec_hidden": CODEC_HIDDEN,
        "codec_quantizer_input_dim": CODEC_INPUT_DIM,
        "codec_decoder_channels": CODEC_DECODER_CHANNELS,
        "codec_semantic_codebook_size": CODEC_SEMANTIC_SIZE,
        "codec_residual_books": CODEC_RESIDUAL_BOOKS,
        "codec_residual_codebook_size": CODEC_RESIDUAL_SIZE,
        "codec_codebook_dim": CODEBOOK_DIM,
        "codec_post_n_layer": POST_LAYERS,
        "codec_post_n_head": POST_HEADS,
        "codec_post_n_local_heads": POST_LOCAL_HEADS,
        "codec_post_intermediate_size": POST_INTERMEDIATE,
    }


PURE_CONFIG = {
    "model_type": "arktts",
    "slow_backbone": "arktts",
    "vocab_size": VOCAB,
    "dim": DIM,
    "n_layer": LAYERS,
    "n_head": HEADS,
    "n_local_heads": LOCAL_HEADS,
    "head_dim": HEAD_DIM,
    "intermediate_size": INTERMEDIATE,
    "max_seq_len": MAX_SEQ_LEN,
    "rope_base": ROPE_BASE,
    "norm_eps": 1e-6,
    "attention_qkv_bias": True,
    "attention_qk_norm": False,
    "attention_o_bias": False,
    "tie_word_embeddings": True,
    "codebook_size": CODEBOOK_SIZE,
    "num_codebooks": NUM_CODEBOOKS,
    "semantic_begin_id": SEMANTIC_BEGIN,
    "semantic_end_id": SEMANTIC_END,
    "n_fast_layer": FAST_LAYERS,
    "fast_dim": FAST_DIM,
    "fast_n_head": FAST_HEADS,
    "fast_n_local_heads": FAST_LOCAL,
    "fast_head_dim": FAST_HEAD_DIM,
    "fast_intermediate_size": FAST_INTERMEDIATE,
    "fast_attention_qkv_bias": False,
    "fast_attention_qk_norm": False,
    "fast_attention_o_bias": False,
    "norm_fastlayer_input": True,
    "ras_window_size": RAS_WINDOW,
    "ras_temperature": 1.0,
    "ras_top_p": 0.9,
    "eos_token_id": EOS_ID,
    "pad_token_id": PAD_ID,
    "sample_rate": 44100,
    **codec_config_dict(),
}

# The falcon flavor: unity muP multipliers EXCEPT the embedding multiplier,
# which stays runtime-applied to the composite (both sides refuse the rest).
FALCON_MULTIPLIERS = {
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_in_multiplier": 1.0,
    "attention_out_multiplier": 1.0,
    "key_multiplier": 1.0,
    "embedding_multiplier": 1.25,
    "lm_head_multiplier": 1.0,
    "mlp_bias": False,
    "mlp_multipliers": [1.0, 1.0],
    "projectors_bias": False,
    "ssm_in_multiplier": 1.0,
    "ssm_multipliers": [1.0, 1.0, 1.0, 1.0, 1.0],
    "ssm_out_multiplier": 1.0,
    "mamba_chunk_size": 8,
    "mamba_conv_bias": True,
    "mamba_d_conv": 4,
    "mamba_d_head": 4,
    "mamba_d_ssm": 16,
    "mamba_d_state": 8,
    "mamba_expand": 2,
    "mamba_n_groups": 2,
    "mamba_n_heads": 4,
    "mamba_norm_before_gate": False,
    "mamba_proj_bias": False,
    "mamba_rms_norm": False,
    "mamba_use_mlp": True,
}

FALCON_CONFIG = {
    **PURE_CONFIG,
    "slow_backbone": "falcon_h1",
    "norm_eps": 1e-5,
    "semantic_end_id": SEMANTIC_BEGIN + CODEBOOK_SIZE,
    **FALCON_MULTIPLIERS,
}


def seed_tree(module, seed: int, scale: float) -> dict:
    """Deterministic weights: small normals everywhere, RMS norms, layer
    scales and snake alphas positive so activations stay healthy."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, value in dict(tree_flatten(module.parameters())).items():
        if name.endswith(("norm.weight", "norm.bias", "gamma", "alpha")):
            out[name] = mx.array(
                rng.uniform(0.5, 1.5, size=value.shape).astype(np.float32)
            )
        else:
            out[name] = mx.array(
                (rng.standard_normal(value.shape) * scale).astype(np.float32)
            )
    module.update(tree_unflatten(out))
    return out


def build_model(config: dict, seed: int) -> "arktts.Model":
    """Constructs the reference Model offline with the tiny codec subclass
    and seeded weights."""
    allowed = {field.name for field in fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in config.items() if k in allowed})
    mx.random.seed(seed)
    model = arktts.Model(cfg)
    seed_tree(model.model, seed + 1, scale=0.35)
    seed_tree(model.codec, seed + 2, scale=0.1)
    model._tokenizer = TOKENIZER
    return model


def save_checkpoint(model: "arktts.Model", case_dir: Path, config: dict) -> None:
    """Saves the native f32 tree: the TTS model under `model.`, the codec
    decode-path tensors at the top level (encoder / downsample / pre module
    dropped; the Rust decode path never loads them)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    stored = {}
    for name, value in dict(tree_flatten(model.model.parameters())).items():
        stored[f"model.{name}"] = value.astype(mx.float32)
    for name, value in dict(tree_flatten(model.codec.parameters())).items():
        skip = (
            name.startswith("encoder.")
            or ".pre_module." in name
            or ".downsample." in name
        )
        if not skip:
            stored[name] = value.astype(mx.float32)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def codec_tensors(model: "arktts.Model") -> dict:
    return {
        name: value.astype(mx.float32)
        for name, value in dict(tree_flatten(model.codec.parameters())).items()
        if not (
            name.startswith("encoder.")
            or ".pre_module." in name
            or ".downsample." in name
        )
    }


def record_score_gaps(store: list):
    """Wraps `_processed_scores` to record the top-1 minus top-2 gap of
    every processed score vector (the greedy decision margin). The saved
    class-dict entry restores the original binding exactly."""
    original = arktts.ArkttsModel.__dict__["_processed_scores"]

    def wrapper(self, scores, top_k, top_p, temperature, semantic):
        out = original(self, scores, top_k, top_p, temperature, semantic)
        values = np.sort(np.asarray(out, dtype=np.float32).ravel())[::-1]
        if np.isfinite(values[0]) and np.isfinite(values[1]):
            store.append(float(values[0] - values[1]))
        return out

    arktts.ArkttsModel._processed_scores = wrapper
    return original


def record_race_gaps(store: list):
    """Wraps `_sample` to record the exponential-race winner margin (top-1
    minus top-2 of probabilities / noise)."""
    original = arktts.ArkttsModel.__dict__["_sample"]

    def wrapper(self, scores, rng_key):
        probabilities = mx.softmax(scores, axis=-1)
        random = mx.random.uniform(shape=probabilities.shape, key=rng_key)
        noise = -mx.log(random)
        ratios = np.asarray(probabilities / noise, dtype=np.float32).ravel()
        ordered = np.sort(ratios)[::-1]
        store.append(float(ordered[0] - ordered[1]))
        return mx.argmax(probabilities / noise, axis=-1).astype(mx.int64)

    arktts.ArkttsModel._sample = wrapper
    return original


class UniformRecorder:
    """Records every `mx.random.uniform` draw in consumption order."""

    def __init__(self):
        self.draws = []

    def __enter__(self):
        self.original = mx.random.uniform

        def wrapper(*args, **kwargs):
            out = self.original(*args, **kwargs)
            self.draws.append(np.asarray(out, dtype=np.float32).ravel().tolist())
            return out

        mx.random.uniform = wrapper
        return self

    def __exit__(self, *exc):
        mx.random.uniform = self.original
        return False


def greedy_generate(model, text, max_tokens=GREEDY_MAX_TOKENS, ref_audio=None, ref_text=None):
    """The reference Model.generate with greedy sampling; returns the
    GenerationResult (the loop is deterministic)."""
    return list(
        model.generate(
            text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=max_tokens,
            do_sample=False,
            seed=0,
        )
    )


def sampled_generate(model, max_tokens=GREEDY_MAX_TOKENS):
    with UniformRecorder() as recorder:
        results = list(
            model.generate(
                TARGET_TEXT,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_tokens=max_tokens,
                do_sample=True,
                seed=20250917,
            )
        )
    return results, recorder.draws


def verify_replay(case: str, first: dict, second: dict) -> None:
    """The established recording caveat: a fresh replay of the same weights
    and streams must agree with the recording before it freezes."""
    for key in first:
        a = np.asarray(first[key], dtype=np.float64)
        b = np.asarray(second[key], dtype=np.float64)
        assert a.shape == b.shape and np.array_equal(a, b), f"{case}: {key} replay diverged"


def find_eos_seed(config: dict, seed_base: int, attempts: int = 220) -> int:
    """Searches seeds for a checkpoint whose FIRST greedy semantic pick is
    EOS, so generation stops with no frames."""
    for offset in range(attempts):
        seed = seed_base + offset
        model = build_model(config, seed)
        try:
            greedy_generate(model, TARGET_TEXT, max_tokens=1)
        except RuntimeError:
            return seed
    raise SystemExit(f"no seed in [{seed_base}, {seed_base + attempts}) stops on the first frame")


def find_generation_seed(config: dict, seed_base: int, attempts: int = 64) -> int:
    """Searches seeds for a checkpoint whose greedy run emits several frames
    and stops through EOS before the cap (more loop coverage than a capped
    run)."""
    for offset in range(attempts):
        seed = seed_base + offset
        model = build_model(config, seed)
        try:
            result = greedy_generate(model, TARGET_TEXT)[0]
        except RuntimeError:
            continue
        count = int(result.token_count)
        if 5 <= count < GREEDY_MAX_TOKENS:
            return seed
    raise SystemExit(
        f"no seed in [{seed_base}, {seed_base + attempts}) gives an eos-stopped greedy run"
    )


def build_codec_case(model) -> None:
    """Decode-only codec parity with per-stage traces plus a clamped-codes
    replay. The checkpoint carries the decode-path tensors only."""
    codec = model.codec
    quantizer = codec.quantizer
    frames = 4
    codes = [[(3 * frame + book * 7 + 1) % CODEC_SEMANTIC_SIZE for frame in range(frames)] for book in range(1 + CODEC_RESIDUAL_BOOKS)]

    def decode_stages(code_grid):
        # from_codes takes (B, books, T); decode() slices the semantic and
        # residual books apart with the same clips.
        semantic = mx.clip(
            mx.array(code_grid[0])[None, None], 0, CODEC_SEMANTIC_SIZE - 1
        )
        residual = mx.clip(mx.array(code_grid[1:])[None], 0, CODEC_RESIDUAL_SIZE - 1)
        z = quantizer.semantic_quantizer.from_codes(semantic)
        z = z + quantizer.quantizer.from_codes(residual)
        post = quantizer.post_module(z)
        up = quantizer._run(quantizer.upsample, post)
        wave = np.asarray(codec.decode(mx.array(code_grid)[None])[0])
        return z, post, up, wave

    z, post, up, wave = decode_stages(codes)
    replay = decode_stages(codes)
    verify_replay(
        "codec",
        {"z": z, "post": post, "up": up, "wave": wave},
        {"z": replay[0], "post": replay[1], "up": replay[2], "wave": replay[3]},
    )
    assert float(np.abs(wave).max()) < 1.0, "bounded codec output expected"

    # One over-range code per book: both sides clamp to the top bin.
    clamped = [list(codes[0])]
    clamped[0][1] = CODEC_SEMANTIC_SIZE + 200
    clamped.append(list(codes[1]))
    clamped[1][2] = CODEC_RESIDUAL_SIZE + 5
    clamped.append(list(codes[2]))
    clamped_wave = np.asarray(codec.decode(mx.array(clamped)[None])[0])

    case_dir = OUT / "codec"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), codec_tensors(model))
    (case_dir / "config.json").write_text(json.dumps(codec_config_dict(), indent=2) + "\n")
    traces = {
        "frames": frames,
        "codes": codes,
        "from_codes": floats(z),
        "post_module": floats(post),
        "upsample": floats(up),
        "waveform": floats(wave),
        "clamped_codes": clamped,
        "clamped_waveform": floats(clamped_wave),
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))


def reference_prompt_ids():
    """The reference `_prompt_segments` ids for both prompt forms, plus the
    canonical part strings the Rust `prompt_segments` must return. The part
    lists are the reference's inline literals; the id cross-check keeps
    them honest against the real code path."""
    stub = arktts.Model.__new__(arktts.Model)
    stub._tokenizer = TOKENIZER
    plain_ids, empty_ids = stub._prompt_segments(TARGET_TEXT, None, False)
    voice_prefix_ids, voice_suffix_ids = stub._prompt_segments(
        TARGET_TEXT, REFERENCE_TEXT, True
    )
    plain_parts = [
        "<|im_start|>system\n",
        "convert the provided text to speech",
        "<|im_end|>\n",
        "<|im_start|>user\n",
        TARGET_TEXT,
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    voice_prefix_parts = [
        "<|im_start|>system\n",
        "convert the provided text to speech reference to the following:\n\nText:\n",
        f"<|speaker:0|>{REFERENCE_TEXT}",
        "\n\nSpeech:\n",
    ]
    voice_suffix_parts = [
        "<|im_end|>\n",
        "<|im_start|>user\n",
        TARGET_TEXT,
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    assert ENCODER.encode_parts(plain_parts) == ints(plain_ids), "plain parts drifted"
    assert (
        ENCODER.encode_parts(voice_prefix_parts) == ints(voice_prefix_ids)
    ), "voice prefix parts drifted"
    assert (
        ENCODER.encode_parts(voice_suffix_parts) == ints(voice_suffix_ids)
    ), "voice suffix parts drifted"
    assert ints(empty_ids) == [], "the plain suffix must be empty"
    return {
        "plain_parts": plain_parts,
        "plain_ids": ints(plain_ids),
        "voice_prefix_parts": voice_prefix_parts,
        "voice_suffix_parts": voice_suffix_parts,
        "voice_prefix_ids": ints(voice_prefix_ids),
        "voice_suffix_ids": ints(voice_suffix_ids),
    }


TOKENIZER = None


class StubTokenizer:
    """`encode(part, add_special_tokens=False)` per part, concatenated:
    the reference `encode_parts`."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_parts(self, parts) -> list:
        out = []
        for part in parts:
            out.extend(self.tokenizer.encode(part, add_special_tokens=False))
        return out


def build_generation_case(
    name: str, config: dict, seed_base: int, sampled: bool
) -> dict:
    """One end-to-end generation case; the seed search accepts a multi-frame
    run whose greedy decision margins stay at least 0.01 (the rope table's
    bf16 rounding is the one place Rust and mlx float paths can disagree,
    and the codes must be int-exact)."""
    seed = seed_base
    for offset in range(24):
        seed = seed_base + offset
        model = build_model(config, seed)
        gaps: list = []
        orig_processed = record_score_gaps(gaps)
        race_gaps: list = []
        orig_sample = record_race_gaps(race_gaps)
        try:
            if sampled:
                results, draws = sampled_generate(model)
                result = results[0]
            else:
                result = greedy_generate(model, TARGET_TEXT)[0]
                draws = None
        except RuntimeError:
            continue
        finally:
            arktts.ArkttsModel._processed_scores = orig_processed
            arktts.ArkttsModel._sample = orig_sample
        thin = (min(gaps) < 0.01) if gaps else False
        if sampled:
            if int(result.token_count) < 4:
                continue
        elif thin or int(result.token_count) < 4:
            continue
        if race_gaps and min(race_gaps) < 1e-4:
            continue
        break

    token_count = int(result.token_count)
    prompts = reference_prompt_ids()
    prefix_parts = prompts["plain_parts"]
    suffix_parts: list = []
    prefix_ids = prompts["plain_ids"]
    suffix_ids: list = []

    # The generated code grid comes from generate_codes directly (the
    # generate loop's audio is the codec decode of exactly these codes).
    if sampled:
        with UniformRecorder() as replay_recorder:
            codes_grid = model.model.generate_codes(
                mx.array(prefix_ids)[None],
                mx.array(suffix_ids)[None],
                None,
                None,
                max_new_tokens=GREEDY_MAX_TOKENS,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                seed=20250917,
            )
        assert replay_recorder.draws == draws, "uniform stream replay diverged"
    else:
        codes_grid = model.model.generate_codes(
            mx.array(prefix_ids)[None],
            mx.array(suffix_ids)[None],
            None,
            None,
            max_new_tokens=GREEDY_MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=False,
            seed=0,
        )
    codes_np = np.asarray(codes_grid)
    assert codes_np.shape[0] == 1 and codes_np.shape[1] == NUM_CODEBOOKS
    book_major = codes_np[0].astype(int).tolist()
    frame_major = [[book_major[book][t] for book in range(NUM_CODEBOOKS)] for t in range(codes_np.shape[2])]
    wave = np.asarray(result.audio, dtype=np.float32)
    assert token_count == codes_np.shape[2], "generate loop divergence"

    # Fresh replay: same weights, same streams, bitwise identical.
    model2 = build_model(config, seed)
    if sampled:
        results2, draws2 = sampled_generate(model2)
        assert draws2 == draws, "sampled draws not reproducible"
        wave2 = np.asarray(results2[0].audio, dtype=np.float32)
        assert np.array_equal(wave, wave2), "sampled waveform replay diverged"
    else:
        result2 = greedy_generate(model2, TARGET_TEXT)[0]
        wave2 = np.asarray(result2.audio, dtype=np.float32)
        assert np.array_equal(wave, wave2), "greedy waveform replay diverged"

    save_checkpoint(model, OUT / name, config)
    traces = {
        "seed": seed,
        "max_new_tokens": GREEDY_MAX_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": bool(sampled),
        "text": TARGET_TEXT,
        "prefix_parts": prefix_parts,
        "suffix_parts": suffix_parts,
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids,
        "codes_book_major": book_major,
        "codes": frame_major,
        "waveform": floats(wave),
    }
    if sampled:
        flat = [value for draw in draws for value in draw]
        assert all(0.0 < v < 1.0 for v in flat), "uniform draws must be in the open interval"
        traces["uniform_draws"] = draws
        traces["uniform_draw_widths"] = [len(draw) for draw in draws]
    (OUT / name / "traces.json").write_text(json.dumps(traces))
    return {
        "name": name,
        "seed": seed,
        "frames": len(frame_major),
        "min_score_gap": min(gaps) if gaps else None,
        "min_race_gap": min(race_gaps) if race_gaps else None,
    }


def build_voice_case(config: dict, seed_base: int) -> dict:
    """The reference-voice path: the tiny codec ENCODES a seeded wave into
    reference codes (the Rust encode is a documented refusal; the codes are
    recorded), then greedy generation rides the reference prompt. The seed
    search accepts a multi-frame run with a 0.01 greedy margin."""
    wave = None
    for offset in range(24):
        seed = seed_base + 2000 + offset
        model = build_model(config, seed)
        gaps: list = []
        orig_processed = record_score_gaps(gaps)
        try:
            sample_rate = 44100
            t = np.arange(4096, dtype=np.float32)
            wave = (
                0.4 * np.sin(2.0 * np.pi * 220.0 * t / sample_rate)
                + 0.15 * np.cos(2.0 * np.pi * 0.9 * t)
            ).astype(np.float32)
            codes, lengths = model.encode_reference(mx.array(wave))
            mx.eval(codes, lengths)
            result = greedy_generate(model, TARGET_TEXT, ref_audio=mx.array(wave), ref_text=REFERENCE_TEXT)[0]
        except RuntimeError:
            continue
        finally:
            arktts.ArkttsModel._processed_scores = orig_processed
        if int(result.token_count) < 4 or (gaps and min(gaps) < 0.01):
            continue
        break

    codes_np = np.asarray(codes)[0]  # (num_codebooks, T)
    reference_codes = codes_np.astype(int).tolist()
    reference_length = int(np.asarray(lengths)[0])
    prompts = reference_prompt_ids()
    prefix_parts = prompts["voice_prefix_parts"]
    suffix_parts = prompts["voice_suffix_parts"]
    prefix_ids = prompts["voice_prefix_ids"]
    suffix_ids = prompts["voice_suffix_ids"]

    codes_grid = model.model.generate_codes(
        mx.array(prefix_ids)[None],
        mx.array(suffix_ids)[None],
        [codes_np],
        mx.array([reference_length]),
        max_new_tokens=GREEDY_MAX_TOKENS,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=False,
        seed=0,
    )
    grid = np.asarray(codes_grid)
    assert grid.shape[1] == NUM_CODEBOOKS
    book_major = grid[0].astype(int).tolist()
    frame_major = [
        [book_major[book][t] for book in range(NUM_CODEBOOKS)] for t in range(grid.shape[2])
    ]
    wave_out = np.asarray(result.audio, dtype=np.float32)

    model2 = build_model(config, seed)
    encoded2, lengths2 = model2.encode_reference(mx.array(wave))
    assert np.array_equal(np.asarray(encoded2)[0], codes_np), "encode replay diverged"

    save_checkpoint(model, OUT / "voice", config)
    traces = {
        "seed": seed,
        "max_new_tokens": GREEDY_MAX_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": False,
        "text": TARGET_TEXT,
        "reference_text": REFERENCE_TEXT,
        "reference_wave": floats(wave),
        "reference_codes": reference_codes,
        "reference_length": reference_length,
        "prefix_parts": prefix_parts,
        "suffix_parts": suffix_parts,
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids,
        "codes_book_major": book_major,
        "codes": frame_major,
        "waveform": floats(wave_out),
    }
    (OUT / "voice" / "traces.json").write_text(json.dumps(traces))
    return {
        "name": "voice",
        "seed": seed,
        "frames": len(frame_major),
        "reference_frames": reference_length,
    }


def find_sampled_seed(config: dict, seed_base: int, attempts: int = 32) -> int:
    """Searches generation seeds for a sampled run that emits several frames
    (the sampled draw key differs from the greedy decision, so the greedy
    search seed does not transfer)."""
    for offset in range(attempts):
        seed = seed_base + offset
        model = build_model(config, seed + 1000)
        try:
            results, _ = sampled_generate(model)
        except RuntimeError:
            continue
        if int(results[0].token_count) >= 4:
            return seed + 1000
    raise SystemExit("no sampled seed produced a multi-frame run")


def find_voice_seed(config: dict, seed_base: int, attempts: int = 32) -> int:
    """Searches seeds for a voice run (reference codes in the prompt) that
    emits several frames."""
    for offset in range(attempts):
        seed = seed_base + offset
        model = build_model(config, seed + 2000)
        try:
            sample_rate = 44100
            t = np.arange(4096, dtype=np.float32)
            wave = (
                0.4 * np.sin(2.0 * np.pi * 220.0 * t / sample_rate)
                + 0.15 * np.cos(2.0 * np.pi * 0.9 * t)
            ).astype(np.float32)
            results = greedy_generate(
                model, TARGET_TEXT, ref_audio=mx.array(wave), ref_text=REFERENCE_TEXT
            )
        except RuntimeError:
            continue
        if int(results[0].token_count) >= 4:
            return seed + 2000
    raise SystemExit("no voice seed produced a multi-frame run")


def build_eos_case(name: str, config: dict, seed: int) -> None:
    """A checkpoint whose first greedy semantic pick is EOS: no frames, and
    the model-level generate refuses with no audio."""
    model = build_model(config, seed)
    try:
        greedy_generate(model, TARGET_TEXT, max_tokens=GREEDY_MAX_TOKENS)
        raise SystemExit(f"{name}: seed {seed} did not stop on the first frame")
    except RuntimeError as error:
        assert "no audio frames" in str(error), f"unexpected generate error {error}"

    prompts = reference_prompt_ids()
    prefix_parts = prompts["plain_parts"]
    prefix_ids = prompts["plain_ids"]
    codes_grid = model.model.generate_codes(
        mx.array(prefix_ids)[None],
        mx.array([])[None],
        None,
        None,
        max_new_tokens=GREEDY_MAX_TOKENS,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=False,
        seed=0,
    )
    assert np.asarray(codes_grid).shape[-1] == 0, "eos case must emit no frames"

    save_checkpoint(model, OUT / name, config)
    traces = {
        "seed": seed,
        "max_new_tokens": GREEDY_MAX_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": False,
        "text": TARGET_TEXT,
        "prefix_parts": prefix_parts,
        "suffix_parts": [],
        "prefix_ids": prefix_ids,
        "suffix_ids": [],
        "frames": [],
        "stopped_by": "eos_first_pick",
    }
    (OUT / name / "traces.json").write_text(json.dumps(traces))


def build_sampling_case() -> None:
    """The sampler math on fixed logits: the legacy filter and the
    exponential race with recorded uniforms (numpy f32 mirrors of the
    reference formulas)."""

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

    def legacy_mirror(scores, top_k, top_p):
        order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable")
        sorted_scores = np.asarray(scores, dtype=np.float32)[order]
        max_value = sorted_scores.max()
        exps = np.exp((sorted_scores - max_value).astype(np.float32))
        probs = exps / np.float32(exps.sum())
        cumulative = np.cumsum(probs)
        ranks = np.arange(len(scores))
        remove_sorted = (cumulative > np.float32(top_p)) | (ranks >= top_k)
        remove_sorted[0] = False
        remove = np.zeros_like(remove_sorted)
        remove[order] = remove_sorted
        out = np.where(remove, np.float32(-np.inf), np.asarray(scores, dtype=np.float32))
        return out

    def race_mirror(scores, uniforms):
        scores = np.asarray(scores, dtype=np.float32)
        max_value = scores.max()
        exps = np.exp((scores - max_value).astype(np.float32))
        total = np.float32(exps.sum())
        noise = -np.log(np.asarray(uniforms, dtype=np.float32))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = (exps / total) / noise
        winner = int(np.argmax(ratios))
        ordered = np.sort(ratios)[::-1]
        return winner, float(ordered[0] - ordered[1])

    cases = []

    # The legacy filter on a clean vector: top-k 50 / top-p 0.9, and the
    # rank-0 survival with a tiny top-p.
    logits = vector(96, 3)
    cases.append(
        {
            "name": "legacy_top_k_50_top_p_0_9",
            "logits": floats(logits),
            "top_k": 50,
            "top_p": 0.9,
            "filtered": masked(legacy_mirror(logits, 50, 0.9)),
        }
    )
    logits = vector(96, 11)
    cases.append(
        {
            "name": "legacy_top_k_5_top_p_0_3",
            "logits": floats(logits),
            "top_k": 5,
            "top_p": 0.3,
            "filtered": masked(legacy_mirror(logits, 5, 0.3)),
        }
    )
    # The semantic-filtered shape: most entries are -inf already and stay
    # -inf through the filter.
    filtered_like = np.full(96, -np.inf, dtype=np.float32)
    filtered_like[400 % 96 : 400 % 96 + 12] = vector(12, 17)
    filtered_like[95] = 2.0
    cases.append(
        {
            "name": "legacy_over_negative_infinity",
            "logits": masked(filtered_like),
            "top_k": 50,
            "top_p": 0.9,
            "filtered": masked(legacy_mirror(filtered_like, 50, 0.9)),
        }
    )
    # top_k wider than the vocab.
    logits = vector(24, 23)
    cases.append(
        {
            "name": "legacy_top_k_beyond_vocab",
            "logits": floats(logits),
            "top_k": 50,
            "top_p": 0.95,
            "filtered": masked(legacy_mirror(logits, 50, 0.95)),
        }
    )

    # The exponential race: recorded per-entry uniforms, winner plus margin.
    for name, offset, seed in [
        ("race_96", 5, 9001),
        ("race_with_negative_infinity", 9, 9002),
        ("race_10", 13, 9003),
    ]:
        if name == "race_10":
            scores = vector(10, offset)
        elif name == "race_with_negative_infinity":
            scores = np.full(96, -np.inf, dtype=np.float32)
            scores[:20] = vector(20, offset)
        else:
            scores = vector(96, offset)
        rng = np.random.default_rng(seed)
        uniforms = rng.uniform(0.01, 0.99, size=scores.shape).astype(np.float32)
        winner, margin = race_mirror(scores, uniforms)
        assert margin > 1e-4, f"thin race margin on {name}: {margin}"
        cases.append(
            {
                "name": name,
                "scores": masked(scores),
                "uniforms": floats(uniforms),
                "winner": winner,
                "margin": margin,
            }
        )

    # The RAS switch, mirroring `_sample_semantic`'s decision: a repeated
    # semantic in the window switches to the ras draw; a repeated EOS stays.
    pure_cfg = PURE_CONFIG
    begin = pure_cfg["semantic_begin_id"]
    end = pure_cfg["semantic_end_id"]
    eos = pure_cfg["eos_token_id"]
    ras_scores = np.full(VOCAB, -np.inf, dtype=np.float32)
    ras_scores[begin : end + 1] = vector(end - begin + 1, 31)
    ras_scores[eos] = 3.0
    ras_scores[begin + 2] += 6.0  # the normal winner
    ras_scores[begin + 3] += 3.0  # the ras winner (under ras_top_p 0.9 it stays in)
    rng = np.random.default_rng(9101)
    uniforms_normal = rng.uniform(0.05, 0.95, size=VOCAB).astype(np.float32)
    uniforms_high = rng.uniform(0.05, 0.95, size=VOCAB).astype(np.float32)
    normal_winner, _ = race_mirror(ras_scores / np.float32(0.7), uniforms_normal)
    high_winner, _ = race_mirror(ras_scores / np.float32(0.9), uniforms_high)
    assert begin <= normal_winner <= end, "expected a semantic normal winner"
    window_repeat = [0, 0, normal_winner, 0]
    window_clean = [1, 2, 3, 4]
    cases.append(
        {
            "name": "ras_switch",
            "config": "pure",
            "filtered_logits": masked(ras_scores),
            "uniforms_normal": floats(uniforms_normal),
            "uniforms_high": floats(uniforms_high),
            "normal_winner": normal_winner,
            "high_winner": high_winner,
            "windows": {
                "repeat": {"previous": window_repeat, "chosen": high_winner},
                "no_repeat": {"previous": window_clean, "chosen": normal_winner},
                "first_frame": {"previous": None, "chosen": normal_winner},
            },
        }
    )

    # A repeated EOS never switches: the race winner is EOS itself, the
    # window is full of EOS, and semantic_hit is false for EOS.
    eos_scores = np.full(VOCAB, -np.inf, dtype=np.float32)
    eos_scores[begin : end + 1] = vector(end - begin + 1, 37)
    eos_scores[eos] = 30.0  # EOS dominates both races
    eos_normal, _ = race_mirror(eos_scores / np.float32(0.7), uniforms_normal)
    eos_high, _ = race_mirror(eos_scores / np.float32(0.9), uniforms_high)
    assert eos_normal == eos and eos_high == eos, "expected an EOS race winner"
    cases.append(
        {
            "name": "ras_switch_eos",
            "config": "pure",
            "filtered_logits": masked(eos_scores),
            "uniforms_normal": floats(uniforms_normal),
            "uniforms_high": floats(uniforms_high),
            "normal_winner": eos_normal,
            "high_winner": eos_high,
            "windows": {
                "eos_repeat": {"previous": [eos, eos, eos, eos], "chosen": eos_normal},
            },
        }
    )

    case_dir = OUT / "sampling"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps({"cases": cases}))


def build_prompt_case(model) -> None:
    """`_prepare_prompt` grids and `_prompt_segments` part strings."""
    prompts = reference_prompt_ids()
    plain_parts = prompts["plain_parts"]
    prefix_ids = prompts["plain_ids"]
    voice_prefix = prompts["voice_prefix_parts"]
    voice_suffix = prompts["voice_suffix_parts"]
    voice_prefix_ids = prompts["voice_prefix_ids"]
    voice_suffix_ids = prompts["voice_suffix_ids"]

    prompt, mask = model.model._prepare_prompt(
        mx.array(prefix_ids)[None], mx.array([])[None], None, None
    )
    grid = np.asarray(prompt)[0].astype(int).tolist()
    mask_values = np.asarray(mask)[0].astype(int).tolist()

    reference_codes = [[5, 9], [3, 1], [7, 2]]
    prompt_ref, mask_ref = model.model._prepare_prompt(
        mx.array(voice_prefix_ids)[None],
        mx.array(voice_suffix_ids)[None],
        [mx.array(reference_codes)],
        mx.array([2]),
    )
    grid_ref = np.asarray(prompt_ref)[0].astype(int).tolist()
    mask_ref_values = np.asarray(mask_ref)[0].astype(int).tolist()

    case_dir = OUT / "prompt"
    case_dir.mkdir(parents=True, exist_ok=True)
    traces = {
        "plain": {
            "prefix_parts": plain_parts,
            "suffix_parts": [],
            "prefix_ids": prefix_ids,
            "suffix_ids": [],
            "grid": grid,
            "mask": mask_values,
        },
        "voice": {
            "prefix_parts": voice_prefix,
            "suffix_parts": voice_suffix,
            "prefix_ids": voice_prefix_ids,
            "suffix_ids": voice_suffix_ids,
            "reference_codes": reference_codes,
            "reference_length": 2,
            "grid": grid_ref,
            "mask": mask_ref_values,
        },
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))


def main() -> None:
    global TOKENIZER, ENCODER
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        TOKENIZER = build_tokenizer()
        ENCODER = StubTokenizer(TOKENIZER)

        # Patch the codec classes the reference Model constructs.
        arktts.ArkttsCodec = TinyCodec

        codec_model = build_model(PURE_CONFIG, CODEC_SEED)
        build_codec_case(codec_model)
        build_prompt_case(codec_model)

        summary = []
        summary.append(build_generation_case("pure_attention", PURE_CONFIG, PURE_SEED, sampled=False))
        summary.append(build_generation_case("sampled", PURE_CONFIG, SAMPLED_SEED + 1000, sampled=True))
        summary.append(build_voice_case(PURE_CONFIG, VOICE_SEED))
        summary.append(build_generation_case("falcon", FALCON_CONFIG, FALCON_SEED, sampled=False))

        eos_seed = find_eos_seed(PURE_CONFIG, 501)
        build_eos_case("eos", PURE_CONFIG, eos_seed)
        falcon_eos_seed = find_eos_seed(FALCON_CONFIG, 601)
        build_eos_case("falcon_eos", FALCON_CONFIG, falcon_eos_seed)

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
            "generator": "crates/audio/tools/gen_arktts_fixtures.py",
            "seeds": {
                "codec": CODEC_SEED,
                "eos": eos_seed,
                "falcon_eos": falcon_eos_seed,
            },
            "summary": summary,
            "files": files,
        }
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"wrote fixtures to {OUT}")
    finally:
        mx.set_default_device(previous_device)


if __name__ == "__main__":
    main()
