#!/usr/bin/env python3
"""Regenerate the Breeze TTS 2 family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/
breeze_tts, mlx 0.32.3). The reference module is imported directly and
the true Model builds at tiny seeded dims (qwen3 audio backbone, T5Gemma2
text encoder, llama depth decoder); the bundled tokenizer and the
Qwen3-TTS audio tokenizer are replaced by deterministic mocks, since the
portable crate consumes pre-tokenized ids and emits codebook frames (the
waveform decode needs the unported Qwen3-TTS speech tokenizer).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_breeze_tts_fixtures.py

Outputs into crates/audio/testdata/tts_breeze_tts/:

- generate/model.safetensors + generate/config.json
    Seeded tiny Breeze Model (2-layer qwen3 backbone at hidden 32 over
    4 codebooks of vocabulary 64, 3-layer T5Gemma2 text encoder at
    hidden 16 with a sliding/full layer pattern, 2-layer llama depth
    decoder at hidden 16). Weights are stored F16; the traces record the
    upcast F16 values the Rust f32 path reads back.
- generate/traces.json
    Prompt-embedding builds (plain, voice clone, instruct), greedy
    scenarios (EOS stop, max-tokens stop, classifier-free guidance) with
    the sampled frames, and the depth-head mapping logits.
- math/traces.json
    The reserved-logit mask, the _sample slice/argmax behavior, the
    sign-aware repetition penalty, and the T5Gemma2 text encoder
    forward (EOI replacement, bidirectional full and sliding layers).
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The
backbone seed at the bottom is pinned to the first seed of the search
below whose greedy runs satisfy the scenario shapes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

import mlx_audio.tts.models.breeze_tts.breeze_tts as breeze_tts
from mlx_audio.tts.models.breeze_tts.config import ModelConfig

OUT = Path("../turbospark/crates/audio/testdata/tts_breeze_tts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

K = 4
VOCAB = 64
BACKBONE_SEED = 1
EOS_BUDGET = 24
MAX_TOKENS_BUDGET = 6
CFG_SCALE = 2.0

CONFIG = {
    "model_type": "breeze",
    "num_codebooks": K,
    "vocab_size": VOCAB,
    "audio_embed_size": 8,
    "text_vocab_size": 128,
    "backbone_config": dict(
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-5,
        vocab_size=512,
        max_position_embeddings=256,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
    ),
    "text_encoder_config": dict(
        vocab_size=192,
        hidden_size=16,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=16,
        max_position_embeddings=128,
        sliding_window=4,
        _sliding_window_pattern=3,
        eoi_token_index=180,
    ),
    "depth_decoder_config": dict(
        vocab_size=64,
        num_codebooks=K,
        audio_embed_size=8,
        backbone_hidden_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-5,
        max_position_embeddings=33,
        rope_theta=500000.0,
    ),
    "codebook_eos_token_id": 0,
    "sample_rate": 24000,
}

SPECIAL = {"[S0]": 9, "<ins_bos>": 7, "<ins_eos>": 8}


class MockTokenizer:
    """Deterministic stand-in for the bundled fast tokenizer: the speaker
    and instruct markers map to fixed ids, every other chunk to a rolling
    hash. The Rust side consumes the recorded id sequences."""

    def __call__(self, text, add_special_tokens=True):
        ids = []
        for raw in re.split(r"(\[S0\]|<ins_bos>|<ins_eos>)", text):
            if not raw:
                continue
            if raw in SPECIAL:
                ids.append(SPECIAL[raw])
                continue
            ids.append(10 + (sum(ord(c) * (i + 1) for i, c in enumerate(raw)) % 100))
        return {"input_ids": ids}


class FakeAudioTokenizer:
    """Stand-in for the Qwen3-TTS speech tokenizer: a decode rate for the
    generate loop, fixed seeded codes for the reference-encode step, and
    a trivial decode (the Rust port ends at the frames)."""

    decode_upsample_rate = 48000

    def __init__(self, codes):
        self._codes = codes  # [1, K, T]

    def encode(self, audio):
        return self._codes

    def decode(self, codes):
        return mx.zeros((16,))


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def masked_floats(arr) -> list:
    """Negative infinities become null sentinels (strict JSON)."""
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(arr, dtype=np.float32).ravel().tolist()
    ]


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def build_model(seed: int, eos_bias: float = 1.0):
    config = ModelConfig.from_dict(CONFIG)
    mx.random.seed(seed)
    model = breeze_tts.Model(config)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if (
            name.endswith("norm.weight")
            or name.endswith("q_norm.weight")
            or name.endswith("k_norm.weight")
            or name.endswith("eoi_embedding")
        ):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    if eos_bias != 1.0:
        head = np.array(flat["lm_head.weight"], dtype=np.float32)
        head[VOCAB, :] *= eos_bias
        flat["lm_head.weight"] = mx.array(head)
    stored = {name: value.astype(mx.float16) for name, value in flat.items()}
    model.update(tree_unflatten({k: v.astype(mx.float32) for k, v in stored.items()}))
    mx.eval(model)
    return model, stored


def run_frames(model, text, instruct=None, ref_audio=None, ref_text=None,
               cfg_scale=None, max_tokens=EOS_BUDGET):
    """Greedy frame capture through the reference building blocks, with
    the same flow the reference generate drives."""
    model.tokenizer = MockTokenizer()
    cond = model._prompt_embeddings(
        text, voice=None, instruct=instruct, ref_audio=ref_audio, ref_text=ref_text
    )
    use_cfg = bool(instruct) and cfg_scale not in (None, 1.0)
    scale = 1.0 if cfg_scale is None else cfg_scale
    uncond = None
    if use_cfg:
        uncond = model._prompt_embeddings(
            text, voice=None, instruct=None, ref_audio=ref_audio, ref_text=ref_text
        )
    cond_cache = model.backbone_model.make_cache()
    cond_hidden = model.backbone_model(input_embeddings=cond, cache=cond_cache)[:, -1, :]
    uncond_hidden = None
    uncond_cache = None
    if use_cfg:
        uncond_cache = model.backbone_model.make_cache()
        uncond_hidden = model.backbone_model(
            input_embeddings=uncond, cache=uncond_cache
        )[:, -1, :]
    frames = []
    first_codes = []
    stopped = "max_tokens"
    for _ in range(max_tokens):
        logits = model.lm_head(cond_hidden)
        if use_cfg:
            uncond_logits = model.lm_head(uncond_hidden)
            logits = uncond_logits + scale * (logits - uncond_logits)
        logits = model._apply_repetition_penalty(
            logits, first_codes, 1.0
        )
        logits = model._mask_reserved_codec_logits(logits)
        first = model._sample(
            logits, temperature=0.0, top_p=1.0, top_k=0, allow_eos=True
        )
        if first == model.vocab_size:
            stopped = "eos"
            break
        tokens = [0, first]
        for _ in range(model.num_codebooks - 1):
            token_ids = mx.array(tokens, dtype=mx.int32)[None, :]
            step_logits = model.depth_decoder.next_logits(token_ids, cond_hidden)
            if use_cfg:
                uncond_step = model.depth_decoder.next_logits(token_ids, uncond_hidden)
                step_logits = uncond_step + scale * (step_logits - uncond_step)
            step_logits = model._mask_reserved_codec_logits(step_logits)
            tokens.append(
                model._sample(
                    step_logits, temperature=0.0, top_p=1.0, top_k=0, allow_eos=False
                )
            )
        frame = tokens[1:]
        frames.append(frame)
        first_codes.append(first)
        codebooks = mx.array(frame, dtype=mx.int32)[None, None, :]
        cond_hidden = model.backbone_model(input_ids=codebooks, cache=cond_cache)[:, -1, :]
        if use_cfg:
            uncond_hidden = model.backbone_model(
                input_ids=codebooks, cache=uncond_cache
            )[:, -1, :]
    return frames, stopped


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- backbone seed search --------------------------------------------
    best = None
    for seed in range(1, 200):
        model, stored = build_model(seed)
        frames, stopped = run_frames(model, "Hello there.")
        length_frames, _ = run_frames(model, "Hello there.", max_tokens=MAX_TOKENS_BUDGET)
        ok = (
            stopped == "eos"
            and 3 <= len(frames) <= 16
            and len(length_frames) == MAX_TOKENS_BUDGET
        )
        if ok:
            best = (model, stored, frames, seed)
            print(f"backbone seed {seed}: acceptable greedy fixtures")
            break
    if best is None:
        raise SystemExit("no backbone seed produced acceptable greedy fixtures")
    model, stored, frames, seed = best
    if seed != BACKBONE_SEED:
        print(f"NOTE: search picked seed {seed}; BACKBONE_SEED is {BACKBONE_SEED}")

    # --- prompt embedding records -----------------------------------------
    model.tokenizer = MockTokenizer()
    ref_codes = np.array(
        [[5, 11, 20, 3], [9, 2, 14, 7], [31, 8, 1, 22], [44, 17, 6, 12]], dtype=np.int32
    )
    model.audio_tokenizer = FakeAudioTokenizer(mx.array(ref_codes)[None])
    plain_ids = MockTokenizer()("[S0]Hello there.")["input_ids"]
    instruct_ids = MockTokenizer()(
        "[S0]<ins_bos>warm bright voice<ins_eos>Hello there."
    )["input_ids"]
    uncond_ids = MockTokenizer()("[S0]Hello there.")["input_ids"]
    ref_text_ids = MockTokenizer()("[S0]Ref voice.")["input_ids"]

    plain_emb = model._prompt_embeddings(
        "Hello there.", voice=None, instruct=None, ref_audio=None, ref_text=None
    )
    instruct_emb = model._prompt_embeddings(
        "Hello there.",
        voice=None,
        instruct="warm bright voice",
        ref_audio=None,
        ref_text=None,
    )
    ref_emb = model._prompt_embeddings(
        "Hello there.",
        voice=None,
        instruct=None,
        ref_audio=np.zeros(8, dtype=np.float32),
        ref_text="Ref voice.",
    )

    cfg_frames, cfg_stopped = run_frames(
        model,
        "Hello there.",
        instruct="warm bright voice",
        cfg_scale=CFG_SCALE,
    )

    # --- eos-first model -----------------------------------------------------
    eos_model, eos_stored = None, None
    for bias in (6.0, 3.0):
        candidate, candidate_stored = build_model(seed, eos_bias=bias)
        candidate.tokenizer = MockTokenizer()
        candidate.audio_tokenizer = FakeAudioTokenizer(mx.array(ref_codes))
        eos_frames, eos_stopped = run_frames(candidate, "Hello there.", max_tokens=4)
        if eos_stopped == "eos" and not eos_frames:
            eos_model, eos_stored = candidate, candidate_stored
            print(f"eos-first bias {bias}: acceptable")
            break
    if eos_model is None:
        raise SystemExit("no eos-first bias produced an acceptable fixture")

    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    eos_dir = OUT / "generate_eos"
    eos_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(eos_dir / "model.safetensors"), eos_stored)
    (eos_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    # --- depth head mapping ---------------------------------------------------
    rng = np.random.default_rng(733)
    backbone_hidden = (rng.standard_normal(32) * 0.5).astype(np.float32)
    depth_rows = []
    for length in range(2, K + 1):
        token_ids = [0] + [int(v) for v in rng.integers(0, VOCAB, size=length - 1)]
        logits = model.depth_decoder.next_logits(
            mx.array(token_ids, dtype=mx.int32)[None, :],
            mx.array(backbone_hidden)[None],
        )
        depth_rows.append(
            {
                "token_ids": token_ids,
                "backbone_hidden": floats(backbone_hidden),
                "logits": floats(logits),
            }
        )

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "backbone_seed": seed,
        "plain": {
            "ids": plain_ids,
            "prompt": "Hello there.",
            "speaker_ids": plain_ids,
            "embeddings": floats(plain_emb[0]),
        },
        "instruct": {
            "cond_ids": instruct_ids,
            "uncond_ids": uncond_ids,
            "cond_embeddings": floats(instruct_emb[0]),
            "uncond_embeddings": floats(
                model._prompt_embeddings(
                    "Hello there.",
                    voice=None,
                    instruct=None,
                    ref_audio=None,
                    ref_text=None,
                )[0]
            ),
            "cfg_scale": CFG_SCALE,
        },
        "reference": {
            "text_ids": ref_text_ids,
            "codes": [ints(row) for row in ref_codes],
            "embeddings": floats(ref_emb[0]),
        },
        "scenarios": [
            {
                "name": "eos",
                "max_tokens": EOS_BUDGET,
                "stopped_by": "eos",
                "frames": frames,
            },
            {
                "name": "max_tokens",
                "max_tokens": MAX_TOKENS_BUDGET,
                "stopped_by": "max_tokens",
                "frames": length_frames,
            },
            {
                "name": "cfg",
                "max_tokens": EOS_BUDGET,
                "cfg_scale": CFG_SCALE,
                "stopped_by": cfg_stopped,
                "cond_ids": instruct_ids,
                "uncond_ids": uncond_ids,
                "frames": cfg_frames,
            },
        ],
        "depth_head": depth_rows,
        "eos_first": {
            "ids": MockTokenizer()("[S0]Hello there.")["input_ids"],
            "frames": [],
            "stopped_by": "eos",
        },
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))
    eos_traces = {
        "reference_commit": REFERENCE_COMMIT,
        "ids": MockTokenizer()("[S0]Hello there.")["input_ids"],
        "frames": [],
        "stopped_by": "eos",
    }
    (eos_dir / "traces.json").write_text(json.dumps(eos_traces))

    # --- math case --------------------------------------------------------------
    math_traces = {}
    logits_rng = np.random.default_rng(811)

    def fixed_logits(n):
        values = np.array(
            [
                float(np.sin(0.71 * (i + 3)) * 4.0 + np.cos(0.37 * i) * 1.5)
                for i in range(n)
            ],
            dtype=np.float32,
        )
        assert len(set(values.tolist())) == n
        return values

    # Reserved mask + slice + log-softmax + argmax (allow_eos both ways).
    logits = fixed_logits(VOCAB + 1)
    masked = model._mask_reserved_codec_logits(mx.array(logits)[None, :])[0]
    math_traces["mask_reserved"] = {
        "logits": floats(logits),
        "masked": masked_floats(masked),
        "codec_vocab_size": model.config.codec_vocab_size,
    }
    for allow_eos in (True, False):
        sliced = masked[: model.vocab_size + int(allow_eos)]
        logprobs = sliced - mx.logsumexp(sliced)
        argmax = model._sample(
            mx.array(logits)[None, :],
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            allow_eos=allow_eos,
        )
        math_traces[f"sample_allow_eos_{allow_eos}"] = {
            "logprobs": masked_floats(logprobs),
            "argmax": int(argmax),
        }

    # Sign-aware repetition penalty over deduplicated prior tokens; the
    # out-of-range ids the reference filters away.
    logits = fixed_logits(VOCAB + 1)
    generated = [3, 3, 7, 1, 70, 5, -2, VOCAB + 9]
    penalized = model._apply_repetition_penalty(
        mx.array(logits)[None, :], generated, 1.7
    )[0]
    math_traces["repetition_penalty"] = {
        "logits": floats(logits),
        "generated": generated,
        "penalty": 1.7,
        "penalized": floats(penalized),
    }

    # T5Gemma2 text encoder: EOI replacement plus full and sliding
    # attention layers over an unpadded batch-of-one input.
    for name, ids in (
        ("text_encoder_short", [42, 180, 7]),
        ("text_encoder_long", [11, 42, 180, 7, 90, 33, 5, 180]),
    ):
        hidden = model.text_encoder(mx.array(ids)[None, :])
        math_traces[name] = {"ids": ids, "hidden": floats(hidden[0])}

    # Prompt embedding decomposition: the audio frame and EOS-codebook
    # embeddings the reference builder concatenates.
    math_traces["audio_frame_embed"] = {
        "codes": [5, 9, 31, 44],
        "embedding": floats(
            model.backbone_model.embed_tokens(
                mx.array([[[5, 9, 31, 44]]], dtype=mx.int32)
            )[0, 0]
        ),
    }
    math_dir = OUT / "math"
    math_dir.mkdir(parents=True, exist_ok=True)
    (math_dir / "traces.json").write_text(json.dumps(math_traces))

    # --- manifest ---------------------------------------------------------------
    files = {}
    for entry in sorted(OUT.rglob("*")):
        if entry.is_file() and entry.name != "manifest.json":
            files[str(entry.relative_to(OUT))] = hashlib.sha256(entry.read_bytes()).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_breeze_tts_fixtures.py",
        "backbone_seed": seed,
        "cfg_scale": CFG_SCALE,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
