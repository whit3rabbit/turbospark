#!/usr/bin/env python3
"""Regenerate the Sesame (CSM) family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/sesame, mlx 0.32.3). The true reference module is
imported and driven against:

- a deterministic offline PreTrainedTokenizerFast (word-level vocabulary
  with the llama3 bos/eos template) standing in for the network
  AutoTokenizer, patched into `sesame.AutoTokenizer.from_pretrained`;
- a seeded tiny 4-book Mimi (quantizer_bins 16, 24 kHz, 12.5 fps)
  patched into `sesame.Mimi.from_pretrained`, replacing the network
  moshiko checkpoint;
- a raising `sesame.load_watermarker`, so the model takes the same
  `watermarker is None` path a checkpoint without silentcipher takes;
- greedy sampling (`make_sampler(0)`) so every draw is deterministic.

The end-to-end generate case runs the true `Model.generate` generator
(single prompt, no context, default voice prompt disabled, streaming
off), whose decode is the reference `generate_result` chunked
`MimiStreamingDecoder` path. The watermark stage is refused by the Rust
port (silentcipher downloads a separate checkpoint) and is not
exercised.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_sesame_fixtures.py

Outputs into crates/audio/testdata/tts_sesame/:

- decode/model.safetensors + decode/config.json + decode/traces.json
    The tiny seeded Mimi codec, the reference encode codes for a seeded
    wave, the `_tokenize_audio` frame layouts (with and without the eos
    frame), and the reference chunked streaming decode of 63 frames
    (two chunks, the generate_result join).
- generate/model.safetensors + generate/config.json + generate/traces.json
    Seeded tiny CSM (2-layer backbone, 1-layer depth decoder, 4
    codebooks, audio vocab 16), the prompt text ids and frame layout,
    four greedy frames from the reference loop, and the waveform from
    the true `Model.generate` (reference chunked decode included).
- generate_eos/model.safetensors + generate_eos/config.json +
  generate_eos/traces.json
    A second seeded backbone whose codebook heads are zeroed, so every
    sampler argmax is code 0 and the first frame is all zeros: the loop
    stops with no frames and no audio, exactly like the reference.
- sampling/traces.json
    The sampler chain on fixed raw logits (the family applies the
    sampler to un-normalized logits): greedy argmax, top-k 50, top-p
    0.8, and the top-p 0.9 then top-k 50 chain, with null sentinels for
    negative infinities.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.codec.models.mimi.mimi import Mimi, MimiStreamingDecoder, mimi_202407
from mlx_audio.lm.sample_utils import apply_top_k, apply_top_p, make_sampler

import mlx_audio.tts.models.sesame.sesame as sesame
from mlx_audio.tts.models.sesame.sesame import create_llama_model_args

OUT = Path("../turbospark/crates/audio/testdata/tts_sesame")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# Mock tokenizer ids (the Rust tests read ids from the traces).
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
SPEAKER_TOKEN = "[0]"
WORDS = ["speech", "grows", "loud", "and", "then", "fades", "soft"]
PROMPT_TEXT = "speech grows loud and then fades soft"

# Tiny CSM geometry: 4 codebooks over an audio vocab of 16 (the tiny Mimi
# quantizer_bins), 2-layer backbone and 1-layer depth decoder, both with
# the interleaved Llama3-scaled rope of attention.py.
TINY_CONFIG = {
    "model_type": "csm",
    "backbone_flavor": None,
    "decoder_flavor": None,
    "text_vocab_size": 64,
    "audio_vocab_size": 16,
    "audio_num_codebooks": 4,
    "audio_eos_token_id": 0,
    "audio_token_id": 0,
    "bos_token_id": 1,
    "codebook_eos_token_id": 0,
    "codebook_pad_token_id": 0,
    "head_dim": 16,
    "hidden_act": "silu",
    "hidden_size": 32,
    "initializer_range": 0.02,
    "intermediate_size": 64,
    "max_position_embeddings": 2048,
    "mlp_bias": False,
    "num_attention_heads": 4,
    "num_codebooks": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000,
    "tie_codebooks_embeddings": True,
    "tie_word_embeddings": True,
    "use_cache": True,
    "vocab_size": 64,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "speaker_prefix_space": True,
    "use_default_voice_prompt": False,
    "rope_scaling": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "depth_decoder_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "backbone_hidden_size": 32,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": 32,
        "initializer_range": 0.02,
        "intermediate_size": 64,
        "max_position_embeddings": 2048,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 4,
        "num_codebooks": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000,
        "rope_scaling": {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "use_cache": True,
        "vocab_size": 64,
    },
}

TINY_MIMI = dict(
    num_codebooks=4,
    seanet_dimension=16,
    seanet_nfilters=8,
    transformer_d_model=16,
    transformer_num_heads=4,
    transformer_dim_feedforward=32,
    quantizer_dim=8,
    quantizer_bins=16,
)

# 0.4 s at 24 kHz; the tiny Mimi pads to whole 12.5 fps frames.
WAVE_SECONDS = 0.4
SAMPLE_RATE = 24000

MAX_AUDIO_LENGTH_MS = 320.0  # 4 frames at 80 ms
BACKBONE_SEED = 0
EOS_SEED = 11


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level vocabulary with the llama3 bos/eos template of
    `load_llama3_tokenizer`."""
    vocab = {"[UNK]": UNK_ID, "<|begin_of_text|>": BOS_ID, "<|end_of_text|>": EOS_ID}
    vocab[SPEAKER_TOKEN] = 3
    for index, word in enumerate(WORDS):
        vocab[word] = 10 + index
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    # WhitespaceSplit keeps "[0]" whole (the default Whitespace splitter
    # would cut it into punctuation tokens).
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
    )
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(bos, tokenizer.bos_token_id), (eos, tokenizer.eos_token_id)],
    )
    return tokenizer


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def masked(arr) -> list:
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(arr, dtype=np.float32).ravel().tolist()
    ]


def tiny_mimi() -> Mimi:
    cfg = mimi_202407(TINY_MIMI["num_codebooks"])
    cfg.seanet.dimension = TINY_MIMI["seanet_dimension"]
    cfg.seanet.nfilters = TINY_MIMI["seanet_nfilters"]
    cfg.transformer.d_model = TINY_MIMI["transformer_d_model"]
    cfg.transformer.num_heads = TINY_MIMI["transformer_num_heads"]
    cfg.transformer.dim_feedforward = TINY_MIMI["transformer_dim_feedforward"]
    cfg.quantizer_dim = TINY_MIMI["quantizer_dim"]
    cfg.quantizer_bins = TINY_MIMI["quantizer_bins"]
    mx.random.seed(7)
    model = Mimi(cfg)
    # The 0.4 scale other codecs use makes the random-weight Mimi decoder
    # blow up (output magnitudes around 1e9); 0.05 keeps the decode
    # bounded so absolute waveform gates stay meaningful.
    rng = np.random.default_rng(41)
    seeded = {}
    for name, value in dict(tree_flatten(model.parameters())).items():
        seeded[name] = mx.array(
            (rng.standard_normal(value.shape) * 0.05).astype(np.float32)
        )
    load_and_refresh(model, seeded)
    return model


def load_and_refresh(model: Mimi, weights: dict) -> None:
    """load_weights plus the derived-state refresh load_pytorch_weights
    applies (the split-RVQ codebooks and every ConvTranspose1d expanded
    weight). Same contract as tools/gen_mimi_fixtures.py."""
    model.load_weights(list(weights.items()))
    for split in ("rvq_first", "rvq_rest"):
        vq = getattr(model.quantizer, split).vq
        for layer in vq.layers:
            layer.codebook.update_in_place()
    model.upsample.convtr.convtr.convtr.update_in_place()
    for layer in model.decoder.layers:
        layer.upsample.convtr.convtr.update_in_place()


def mimi_config_json(cfg) -> dict:
    return {
        "channels": cfg.channels,
        "sample_rate": cfg.sample_rate,
        "frame_rate": cfg.frame_rate,
        "renormalize": cfg.renormalize,
        "seanet": {
            "dimension": cfg.seanet.dimension,
            "channels": cfg.seanet.channels,
            "causal": cfg.seanet.causal,
            "nfilters": cfg.seanet.nfilters,
            "nresidual_layers": cfg.seanet.nresidual_layers,
            "ratios": cfg.seanet.ratios,
            "ksize": cfg.seanet.ksize,
            "residual_ksize": cfg.seanet.residual_ksize,
            "last_ksize": cfg.seanet.last_ksize,
            "dilation_base": cfg.seanet.dilation_base,
            "pad_mode": cfg.seanet.pad_mode,
            "true_skip": cfg.seanet.true_skip,
            "compress": cfg.seanet.compress,
        },
        "transformer": {
            "d_model": cfg.transformer.d_model,
            "num_heads": cfg.transformer.num_heads,
            "num_layers": cfg.transformer.num_layers,
            "causal": cfg.transformer.causal,
            "norm_first": cfg.transformer.norm_first,
            "bias_ff": cfg.transformer.bias_ff,
            "bias_attn": cfg.transformer.bias_attn,
            "layer_scale": cfg.transformer.layer_scale,
            "positional_embedding": cfg.transformer.positional_embedding,
            "use_conv_bias": cfg.transformer.use_conv_bias,
            "gating": cfg.transformer.gating,
            "norm": cfg.transformer.norm,
            "context": cfg.transformer.context,
            "max_period": cfg.transformer.max_period,
            "max_seq_len": cfg.transformer.max_seq_len,
            "kv_repeat": cfg.transformer.kv_repeat,
            "dim_feedforward": cfg.transformer.dim_feedforward,
            "conv_layout": cfg.transformer.conv_layout,
            "cross_attention": cfg.transformer.cross_attention,
            "conv_kernel_size": cfg.transformer.conv_kernel_size,
        },
        "quantizer_nq": cfg.quantizer_nq,
        "quantizer_bins": cfg.quantizer_bins,
        "quantizer_dim": cfg.quantizer_dim,
    }


def reference_decode(mimi: Mimi, frames: list) -> np.ndarray:
    """The `generate_result` decode verbatim: frames [F][K] stacked to
    [1, K, F] and decoded through the stateful MimiStreamingDecoder in
    62-frame chunks, concatenated along samples."""
    samples = [mx.array(frame, dtype=mx.int32)[None] for frame in frames]
    transposed = mx.transpose(mx.stack(samples), axes=[1, 2, 0])
    token_count = len(samples)
    tokens_per_batch = min(token_count, int(12.5 * 5))
    streaming = MimiStreamingDecoder(mimi)
    streaming.reset()
    all_audio = []
    for start in range(0, transposed.shape[2], tokens_per_batch):
        batch_tokens = transposed[:, :, start : start + tokens_per_batch]
        audio = streaming.decode_frames(batch_tokens).squeeze(0).squeeze(0)
        all_audio.append(audio)
    return np.concatenate([np.array(a) for a in all_audio], axis=0)


def seeded_weights(model, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    seeded = {}
    for name, value in dict(tree_flatten(model.parameters())).items():
        seeded[name] = mx.array(
            (rng.standard_normal(value.shape) * 0.4).astype(np.float32)
        )
        if name.endswith("norm.weight"):
            seeded[name] = mx.array(
                rng.uniform(0.5, 1.5, size=value.shape).astype(np.float32)
            )
    return seeded


def sanitize(weights: dict) -> dict:
    """The reference `Model.sanitize` (prefix + legacy renames); the tiny
    model only needs the `model.` prefix, but the real function runs."""
    return sesame.Model.sanitize(None, weights)


def build_family(config: dict, tokenizer, mimi: Mimi) -> "sesame.Model":
    """Constructs the reference `Model` offline: the network singletons
    are patched out at module level before this runs."""
    mx.random.seed(5)
    return sesame.Model(dict(config))


def save_family(model: "sesame.Model", case_dir: Path, config: dict) -> dict:
    """Saves the sanitized f16 checkpoint the Rust loader reads."""
    case_dir.mkdir(parents=True, exist_ok=True)
    stored = {
        key: value.astype(mx.float16)
        for key, value in sanitize(dict(tree_flatten(model.model.parameters()))).items()
    }
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return stored


def run_reference_frames(model: "sesame.Model", text: str, speaker: int, max_frames: int, sampler) -> list:
    """The `generate` frame loop verbatim (the streaming bookkeeping the
    loop interleaves does not touch the math). Returns [F][K] frames."""
    model.model.reset_caches()
    tokens, tokens_mask = model._tokenize_text_segment(text, speaker)
    curr_tokens = mx.expand_dims(tokens.astype(mx.int32), 0)
    curr_tokens_mask = mx.expand_dims(tokens_mask, 0)
    samples = []
    for _ in range(max_frames):
        sample = model.model.generate_frame(curr_tokens, curr_tokens_mask, None, sampler)
        if mx.all(sample == 0):
            break
        samples.append(sample)
        curr_tokens = mx.expand_dims(
            mx.concat([sample, mx.zeros((1, 1)).astype(mx.int32)], axis=1), axis=1
        )
        curr_tokens_mask = mx.expand_dims(
            mx.concat(
                [
                    mx.ones_like(sample).astype(mx.bool_),
                    mx.zeros((1, 1)).astype(mx.bool_),
                ],
                axis=1,
            ),
            axis=1,
        )
    return [ints(frame) for frame in samples]


def build_decode_case(mimi: Mimi, model: "sesame.Model") -> None:
    traces = {}

    # Reference encode of a seeded wave + the frame layouts.
    t = np.arange(int(WAVE_SECONDS * SAMPLE_RATE), dtype=np.float32)
    wave = 0.4 * np.sin(2.0 * np.pi * 220.0 * t / SAMPLE_RATE) + 0.2 * np.cos(
        2.0 * np.pi * 0.7 * t
    ).astype(np.float32)
    codes = [ints(book) for book in mimi.encode(mx.array(wave)[None, None, ...])[0]]
    traces["wave"] = floats(wave)
    traces["encode_codes"] = codes
    text_frame, text_mask = model._tokenize_audio(mx.array(wave), add_eos=False)
    audio_frame, audio_mask = model._tokenize_audio(mx.array(wave), add_eos=True)
    traces["audio_frames"] = ints(text_frame)
    traces["audio_frame_mask"] = np.asarray(text_mask, dtype=bool).ravel().tolist()
    traces["audio_frames_eos"] = ints(audio_frame)
    traces["audio_frame_mask_eos"] = np.asarray(audio_mask, dtype=bool).ravel().tolist()

    # 63 frames decode through two streaming chunks (the generate_result
    # join at the 62-frame batch cap).
    long_frames = [
        [(3 * frame + book) % 16 for book in range(4)] for frame in range(63)
    ]
    traces["long_frames"] = long_frames
    traces["long_wave"] = floats(reference_decode(mimi, long_frames))

    case_dir = OUT / "decode"
    case_dir.mkdir(parents=True, exist_ok=True)
    # f32, like tools/gen_mimi_fixtures.py (f16 storage of the random
    # weights shifts the decode far past the absolute gate).
    seeded = dict(tree_flatten(mimi.parameters()))
    mx.save_safetensors(str(case_dir / "model.safetensors"), seeded)
    (case_dir / "config.json").write_text(
        json.dumps(mimi_config_json(mimi.cfg), indent=2) + "\n"
    )
    (case_dir / "traces.json").write_text(json.dumps(traces))


def build_generate_case(mimi: Mimi, tokenizer) -> None:
    sampler = make_sampler(0)
    max_frames = int(MAX_AUDIO_LENGTH_MS / 80)
    # The string _tokenize_text_segment builds (the prefix is added
    # inside it, so the loop runs on the unprefixed prompt text).
    text = f"{SPEAKER_TOKEN} {PROMPT_TEXT}" if TINY_CONFIG["speaker_prefix_space"] else PROMPT_TEXT

    best = None
    for seed in range(64):
        model = build_family(TINY_CONFIG, tokenizer, mimi)
        model.model.update(tree_unflatten(seeded_weights(model.model, seed)))
        frames = run_reference_frames(model, PROMPT_TEXT, 0, max_frames, sampler)
        values = [code for frame in frames for code in frame]
        ok = (
            len(frames) == max_frames
            and min(values) >= 0
            and max(values) < 16
            and len(set(values)) >= 4
        )
        if ok:
            best = (model, seed, frames)
            print(f"backbone seed {seed}: acceptable greedy frames")
            break
    if best is None:
        raise SystemExit("no backbone seed produced acceptable greedy frames")
    model, seed, frames = best
    if seed != BACKBONE_SEED:
        print(f"NOTE: search picked seed {seed}; BACKBONE_SEED is {BACKBONE_SEED}")

    # The end-to-end waveform comes from the true Model.generate (the
    # reference loop plus the generate_result chunked decode).
    results = list(
        model.generate(
            PROMPT_TEXT,
            voice=None,
            speaker=0,
            context=[],
            sampler=sampler,
            max_audio_length_ms=MAX_AUDIO_LENGTH_MS,
            stream=False,
        )
    )
    assert len(results) == 1, "expected exactly one non-streaming result"
    result = results[0]
    assert int(result.token_count) == len(frames), "generate loop divergence"

    text_ids = ints(tokenizer(text, add_special_tokens=True)["input_ids"])
    segment_tokens, segment_mask = model._tokenize_text_segment(PROMPT_TEXT, 0)
    stored = save_family(model, OUT / "generate", TINY_CONFIG)
    traces = {
        "max_audio_length_ms": MAX_AUDIO_LENGTH_MS,
        "max_audio_frames": max_frames,
        "text": text,
        "text_ids": text_ids,
        "prompt_frames": ints(segment_tokens),
        "prompt_frame_mask": np.asarray(segment_mask, dtype=bool).ravel().tolist(),
        "frames": frames,
        "waveform": floats(np.array(result.audio, dtype=np.float32)),
    }
    (OUT / "generate" / "traces.json").write_text(json.dumps(traces))


def build_eos_case(mimi: Mimi, tokenizer) -> None:
    sampler = make_sampler(0)
    model = build_family(TINY_CONFIG, tokenizer, mimi)
    model.model.update(tree_unflatten(seeded_weights(model.model, EOS_SEED)))
    # Zero the codebook heads: every sampler argmax is code 0, so the
    # first frame is all zeros and the loop stops immediately.
    zeroed = seeded_weights(model.model, EOS_SEED)
    zeroed["codebook0_head.weight"] = mx.zeros(zeroed["codebook0_head.weight"].shape)
    zeroed["audio_head"] = mx.zeros(zeroed["audio_head"].shape)
    model.model.update(tree_unflatten(zeroed))
    stored = save_family(model, OUT / "generate_eos", TINY_CONFIG)

    text = f"{SPEAKER_TOKEN} {PROMPT_TEXT}" if TINY_CONFIG["speaker_prefix_space"] else PROMPT_TEXT
    frames = run_reference_frames(model, PROMPT_TEXT, 0, int(MAX_AUDIO_LENGTH_MS / 80), sampler)
    assert frames == [], "zeroed heads must stop on the first frame"
    results = list(
        model.generate(
            PROMPT_TEXT,
            voice=None,
            speaker=0,
            context=[],
            sampler=sampler,
            max_audio_length_ms=MAX_AUDIO_LENGTH_MS,
            stream=False,
        )
    )
    assert results == [], "an immediate eos yields no GenerationResult"
    text_ids = ints(tokenizer(text, add_special_tokens=True)["input_ids"])
    segment_tokens, segment_mask = model._tokenize_text_segment(PROMPT_TEXT, 0)
    traces = {
        "max_audio_length_ms": MAX_AUDIO_LENGTH_MS,
        "max_audio_frames": int(MAX_AUDIO_LENGTH_MS / 80),
        "text": text,
        "text_ids": text_ids,
        "prompt_frames": ints(segment_tokens),
        "prompt_frame_mask": np.asarray(segment_mask, dtype=bool).ravel().tolist(),
        "frames": frames,
        "stopped_by": "all_zero_frame",
    }
    (OUT / "generate_eos" / "traces.json").write_text(json.dumps(traces))


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

    # The family applies the sampler to RAW logits (make_sampler output
    # feeds mx.random.categorical / argmax directly). Greedy is the exact
    # argmax; top-k 50 and top-p 0.8 are the reference make_sampler order
    # (top-p first, then top-k).
    logits = vector(96, 0)
    cases.append(
        {
            "name": "greedy",
            "logits": floats(logits),
            "argmax": int(make_sampler(0)(mx.array(logits)[None, :]).item()),
            "top_k": None,
            "top_k_logprobs": None,
            "top_p": None,
            "top_p_logprobs": None,
        }
    )
    logits = vector(96, 7)
    cases.append(
        {
            "name": "top_k_50",
            "logits": floats(logits),
            "argmax": int(mx.argmax(mx.array(logits))),
            "top_k": 50,
            "top_k_logprobs": masked(apply_top_k(mx.array(logits), 50)),
            "top_p": None,
            "top_p_logprobs": None,
        }
    )
    logits = vector(96, 13)
    cases.append(
        {
            "name": "top_p_0_8",
            "logits": floats(logits),
            "argmax": int(mx.argmax(mx.array(logits))),
            "top_k": None,
            "top_k_logprobs": None,
            "top_p": 0.8,
            "top_p_logprobs": masked(apply_top_p(mx.array(logits), 0.8)),
        }
    )
    # The default sampler chain: make_sampler(0.9, top_k=50) applies
    # top-p (when enabled) before top-k on the same raw values.
    logits = vector(96, 29)
    top_p_out = apply_top_p(mx.array(logits), 0.9)
    cases.append(
        {
            "name": "chain_top_p_then_top_k",
            "logits": floats(logits),
            "argmax": int(mx.argmax(mx.array(logits))),
            "top_k": 50,
            "top_k_logprobs": masked(apply_top_k(top_p_out, 50)),
            "top_p": 0.9,
            "top_p_logprobs": masked(top_p_out),
        }
    )

    case_dir = OUT / "sampling"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps({"cases": cases}))


def build_flavor_case() -> None:
    flavors = {}
    for flavor in ("llama-1B", "llama-8B", "llama-100M", "llama-300M"):
        flavors[flavor] = asdict(create_llama_model_args(flavor))
    case_dir = OUT / "flavors"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps(flavors))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Patch the network singletons the reference Model.__init__ loads.
    tokenizer = build_tokenizer()
    mimi = tiny_mimi()
    sesame.AutoTokenizer.from_pretrained = lambda *args, **kwargs: tokenizer
    sesame.Mimi.from_pretrained = classmethod(lambda cls, *args, **kwargs: mimi)

    def _offline_watermarker(*args, **kwargs):
        raise RuntimeError("offline fixture run: no silentcipher")

    sesame.load_watermarker = _offline_watermarker

    build_decode_case(mimi, sesame.Model(dict(TINY_CONFIG)))
    build_generate_case(mimi, tokenizer)
    build_eos_case(mimi, tokenizer)
    build_sampling_case()
    build_flavor_case()

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_sesame_fixtures.py",
        "backbone_seed": BACKBONE_SEED,
        "eos_seed": EOS_SEED,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
