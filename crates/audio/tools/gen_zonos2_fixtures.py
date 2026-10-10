#!/usr/bin/env python3
"""Regenerate the Zonos v2 family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/zonos2,
mlx 0.32.3). The reference Model builds at tiny seeded dims (4 layers,
dim 16, 3 codebooks over a vocabulary of 8 with a 2-bucket quality
feature, MoE on layers 1-2 with the layer-2 top-2 override) and the
generate loop runs through the reference building blocks: the byte-id
prompt builder with the silence tail, the fused-residual forward with
the gated attention and the routed SwitchGLU experts, the per-codebook
sampler, the delayed-frame bookkeeping, and the shear-up + DAC decode.

The speaker embedding arrives as a caller-supplied vector (the
reference audio->embedding extractor is a separate downloadable model).
Sampled draws are not pinned: the reference draws through mx.random
while the Rust port draws through the crate-local threefry RNG, so the
greedy scenarios run at temperature 0 and the sampled-path fixture pins
the deterministic invalid-fallback branch only.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_zonos2_fixtures.py

Outputs into crates/audio/testdata/tts_zonos2/:

- generate/config.json + model.safetensors
    The seeded tiny Zonos2 Model. Weights are stored F16; the traces
    record the upcast F16 values the Rust f32 path reads back.
- dac/config.json + model.safetensors
    A seeded tiny Descript DAC (3 codebooks over 64 entries, hop 8)
    for the decode leg.
- prompt/traces.json
    Prompt-row builds over conditioning combos, the shear pair, and
    the silence tail.
- forward/traces.json
    Hidden states and softcapped logits for prompt forwards with and
    without speaker injection.
- generate/traces.json
    Greedy scenarios (natural eos stop, speaker-conditioned, max
    tokens) with the delayed frames and eos frame.
- sampling/traces.json
    The per-frame sampler on fixed logits: temperature-0 argmax with
    the repetition penalty and the invalid-fallback greedy branch.
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
from mlx.utils import tree_flatten, tree_map

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.codec.models.descript import DAC
from mlx_audio.tts.models.zonos2.config import Zonos2Config
from mlx_audio.tts.models.zonos2.generation import (
    TTSSamplingParams,
    Zonos2GenerationState,
    sample_frame,
)
from mlx_audio.tts.models.zonos2.model import Model
from mlx_audio.tts.models.zonos2.prompt import (
    TTSPromptConfig,
    silence_prompt_tokens,
    shear,
    shear_up,
    text_to_byte_ids,
)

OUT = Path("../turbospark/crates/audio/testdata/tts_zonos2")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

SEED = 5
TOKEN_SEED = 17
HIDDEN_TOLERANCE = 2e-4

CONFIG = {
    "model_type": "zonos2",
    "n_layers": 4,
    "dim": 16,
    "head_dim": 8,
    "n_kv_heads": 1,
    "ffn_dim_multiplier": 1.5,
    "multiple_of": 8,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_seqlen": 128,
    "n_codebooks": 3,
    "codebook_size": 1024,
    "eoa_id": 1024,
    "audio_pad_id": 1025,
    "text_vocab": 455,
    "loss_softcap": 15.0,
    "sample_rate": 44100,
    "speaker_enabled": True,
    "speaker_embedding_dim": 8,
    "speaker_lda_dim": 4,
    "speaker_encoder_model_id": "unused",
    "speaker_encoder_path": None,
    "speaker_encoder_sample_rate": 24000,
    "speaking_rate_num_buckets": 2,
    "quality_num_buckets": 2,
    "quality_features": ["lufs"],
    "quality_buckets": {"lufs": ("a", "b")},
    "speaker_background_token_enabled": True,
    "accurate_mode_token_enabled": True,
    "moe_n_experts": 4,
    "moe_router_topk": 1,
    "special_topk_layers": {2: 2},
    "moe_router_dim": 8,
    "moe_start_from_layer": 1,
    "moe_end_from_layer": 1,
    "norm_topk_prob": False,
    "moe_balancing_strategy": "legacy",
}

DAC_CONFIG = {
    "encoder_dim": 8,
    "encoder_rates": [2, 4],
    "latent_dim": None,
    "decoder_dim": 16,
    "decoder_rates": [4, 2],
    "n_codebooks": 3,
    "codebook_size": 1024,
    "codebook_dim": 4,
    "sample_rate": 44100,
}

MAX_TOKENS = 24
SHORT_TOKENS = 4


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def build_model(seed: int):
    config = Zonos2Config.from_dict(CONFIG)
    mx.random.seed(seed)
    model = Model(config)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith("norm.weight") or name.endswith("rmsnorm_eda.weight"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
        if name.endswith("router_states_scale"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    stored = {name: value.astype(mx.float16) for name, value in flat.items()}
    # Load the F16-rounded values upcast to F32: the stored file keeps
    # the F16 bytes while the live model computes on the values the
    # Rust f32 path reads back.
    model.load_weights(
        [(k, v.astype(mx.float32)) for k, v in stored.items()], strict=True
    )
    mx.eval(model)
    return model, stored


def build_dac(seed: int):
    mx.random.seed(seed)
    dac = DAC(**DAC_CONFIG)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, dac.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith(".alpha"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    stored = {name: value.astype(mx.float16) for name, value in flat.items()}
    dac.load_weights(
        [(k, v.astype(mx.float32)) for k, v in stored.items()], strict=True
    )
    mx.eval(dac)
    return dac, stored


def run_greedy(model, text, params, speaker=None, speaking_rate_bucket=None,
               quality_buckets=None, clean_speaker_background=False,
               accurate_mode=True, seed=0):
    """The reference generate loop at temperature 0 (no draw)."""
    config = model.config
    prompt_rows, speaker_pos = model._build_prompt_rows(
        text,
        speaking_rate_bucket=speaking_rate_bucket,
        quality_buckets=model._resolve_quality_buckets(quality_buckets),
        speaker_conditioned=speaker is not None,
        clean_speaker_background=clean_speaker_background,
        accurate_mode=accurate_mode,
    )
    prompt = mx.array(prompt_rows, dtype=mx.int32)[None, :, :]
    from mlx_audio.lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(prompt, cache=cache, speaker_embedding=speaker,
                   speaker_token_position=speaker_pos)
    last_logits = logits[0, -1]
    state = Zonos2GenerationState(
        n_codebooks=config.n_codebooks,
        eoa_id=config.eoa_id,
        text_vocab=int(config.text_vocab),
    )
    # The reference generate loop iterates once per frame over the
    # TTSSamplingParams limit; the loop bound must come from params so a
    # short max_tokens scenario actually stops there.
    for _ in range(int(params.max_tokens)):
        frame = sample_frame(last_logits, state, params)
        state.append(frame, ignore_eos=params.ignore_eos)
        if state.finished:
            break
        next_ids = mx.array(frame, dtype=mx.int32)[None, None, :]
        logits = model(next_ids, cache=cache)
        last_logits = logits[0, -1]
    return state.generated, state.eos_frame


def decode(dac, delayed_rows, eos_frame, pad, codebook_size, hop):
    """The reference _decode_audio with the tiny DAC."""
    if not delayed_rows:
        return []
    codes = shear_up(mx.array(delayed_rows, dtype=mx.int32), pad)
    if eos_frame is not None:
        codes = codes[: max(0, int(eos_frame))]
    codes = mx.clip(codes, 0, codebook_size - 1)
    if codes.size == 0:
        return []
    z = dac.quantizer.from_codes(codes.T[None, :, :].astype(mx.int32))[0]
    audio = np.asarray(dac.decode(z), dtype=np.float32).reshape(-1)
    audio = audio[: int(codes.shape[0]) * hop]
    return floats(audio)


def main() -> None:
    global model_dac, dac_decode
    OUT.mkdir(parents=True, exist_ok=True)

    model, stored = build_model(SEED)
    config = model.config

    dac, dac_stored = build_dac(SEED + 1)
    dac_hop = int(np.prod(DAC_CONFIG["encoder_rates"]))

    rng = np.random.default_rng(TOKEN_SEED)
    speaker_embedding = (rng.standard_normal(config.speaker_embedding_dim) * 0.5).astype(
        np.float32
    )

    # --- prompt traces ----------------------------------------------------
    prompt_cases = []
    text = "Hi"
    combos = [
        ("plain", dict()),
        ("rate", dict(speaking_rate_bucket=1)),
        ("quality", dict(quality_buckets=[1])),
        ("rate_quality", dict(speaking_rate_bucket=0, quality_buckets=[None])),
        ("speaker", dict(speaker_conditioned=True, accurate_mode=True)),
        (
            "speaker_clean_no_accurate",
            dict(speaker_conditioned=True, clean_speaker_background=True,
                 accurate_mode=False),
        ),
    ]
    for name, kwargs in combos:
        rows, speaker_pos = model._build_prompt_rows(
            text,
            speaking_rate_bucket=kwargs.get("speaking_rate_bucket"),
            quality_buckets=model._resolve_quality_buckets(kwargs.get("quality_buckets")),
            speaker_conditioned=kwargs.get("speaker_conditioned", False),
            clean_speaker_background=kwargs.get("clean_speaker_background", False),
            accurate_mode=kwargs.get("accurate_mode", True),
        )
        prompt_cases.append(
            {
                "name": name,
                "text": text,
                "speaking_rate_bucket": kwargs.get("speaking_rate_bucket"),
                "quality_buckets": kwargs.get("quality_buckets"),
                "speaker_conditioned": kwargs.get("speaker_conditioned", False),
                "clean_speaker_background": kwargs.get("clean_speaker_background", False),
                "accurate_mode": kwargs.get("accurate_mode", True),
                "rows": [[int(v) for v in row] for row in rows],
                "speaker_position": speaker_pos,
            }
        )

    plain = [[1, 2, 3], [4, 5, 6], [7, 0, 2], [3, 1, 4]]
    sheared = shear(mx.array(plain, dtype=mx.int32), config.audio_pad_id)
    up = shear_up(sheared, config.audio_pad_id)
    silence = silence_prompt_tokens(
        TTSPromptConfig(
            n_codebooks=config.n_codebooks,
            audio_pad_id=config.audio_pad_id,
            text_vocab=int(config.text_vocab),
            speaking_rate_num_buckets=config.speaking_rate_num_buckets,
            quality_bucket_counts=config.quality_bucket_counts,
            speaker_background_num_buckets=config.speaker_background_num_buckets,
            accurate_mode_num_buckets=config.accurate_mode_num_buckets,
            prepend_silence=True,
        )
    )
    byte_ids = text_to_byte_ids("Hi")
    (OUT / "prompt").mkdir(parents=True, exist_ok=True)
    (OUT / "prompt" / "traces.json").write_text(
        json.dumps(
            {
                "reference_commit": REFERENCE_COMMIT,
                "cases": prompt_cases,
                "shear": {
                    "plain": plain,
                    "sheared": [[int(v) for v in row] for row in sheared],
                    "unsheared": [[int(v) for v in row] for row in up],
                },
                "silence": [[int(v) for v in row] for row in silence],
                "byte_ids": [int(v) for v in byte_ids],
            }
        )
    )

    # --- forward traces -----------------------------------------------------
    prompt_rows, speaker_pos = model._build_prompt_rows(
        text,
        speaking_rate_bucket=None,
        quality_buckets=None,
        speaker_conditioned=False,
        clean_speaker_background=False,
        accurate_mode=True,
    )
    prompt = mx.array(prompt_rows, dtype=mx.int32)
    hidden = model._forward_hidden(prompt[None], cache=None)
    logits = model.compute_logits(hidden)
    rows = prompt.shape[0]
    # Speaker injection on a synthetic flat input: the embedding
    # replaces the row at the given position after the embedding sum.
    cond_rows, cond_pos = model._build_prompt_rows(
        text,
        speaking_rate_bucket=None,
        quality_buckets=None,
        speaker_conditioned=True,
        clean_speaker_background=False,
        accurate_mode=True,
    )
    cond = mx.array(cond_rows, dtype=mx.int32)
    hidden_speaker = model._forward_hidden(
        cond[None],
        cache=None,
        speaker_embedding=mx.array(speaker_embedding)[None],
        speaker_token_position=cond_pos,
    )
    logits_speaker = model.compute_logits(hidden_speaker)
    (OUT / "forward").mkdir(parents=True, exist_ok=True)
    (OUT / "forward" / "traces.json").write_text(
        json.dumps(
            {
                "reference_commit": REFERENCE_COMMIT,
                "tolerance": HIDDEN_TOLERANCE,
                "cases": [
                    {
                        "name": "plain_prompt",
                        "frames": [[int(v) for v in row] for row in prompt_rows],
                        "hidden": floats(hidden),
                        "logits": floats(logits),
                        "tolerance": HIDDEN_TOLERANCE,
                    },
                    {
                        "name": "speaker_injected",
                        "frames": [[int(v) for v in row] for row in cond_rows],
                        "speaker_embedding": floats(speaker_embedding),
                        "speaker_position": cond_pos,
                        "hidden": floats(hidden_speaker),
                        "logits": floats(logits_speaker),
                        "tolerance": HIDDEN_TOLERANCE,
                    },
                ],
            }
        )
    )
    print(f"forward rows {rows}, hidden {hidden.shape}, logits {logits.shape}")

    # --- greedy scenarios -----------------------------------------------------
    params = TTSSamplingParams(
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        min_p=0.0,
        max_tokens=MAX_TOKENS,
        ignore_eos=False,
        repetition_window=50,
        repetition_penalty=1.2,
        repetition_codebooks=3,
    )
    short_params = TTSSamplingParams(
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        min_p=0.0,
        max_tokens=SHORT_TOKENS,
        ignore_eos=False,
        repetition_window=50,
        repetition_penalty=1.2,
        repetition_codebooks=3,
    )
    generated, eos_frame = run_greedy(model, "Hi", params)
    length, _ = run_greedy(model, "Hi", short_params)
    speaker_frames, speaker_eos = run_greedy(
        model,
        "Hi",
        params,
        speaker=mx.array(speaker_embedding)[None],
        accurate_mode=True,
    )
    rate_frames, rate_eos = run_greedy(
        model,
        "Hi there.",
        params,
        speaking_rate_bucket=1,
        quality_buckets=[0],
    )

    scenarios = [
        {
            "name": "natural",
            "text": "Hi",
            "params": {
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 0.0,
                "min_p": 0.0,
                "max_tokens": MAX_TOKENS,
                "ignore_eos": False,
                "repetition_window": 50,
                "repetition_penalty": 1.2,
                "repetition_codebooks": 3,
            },
            "generated": [[int(v) for v in row] for row in generated],
            "eos_frame": eos_frame,
        },
        {
            "name": "max_tokens",
            "text": "Hi",
            "params": {
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 0.0,
                "min_p": 0.0,
                "max_tokens": SHORT_TOKENS,
                "ignore_eos": False,
                "repetition_window": 50,
                "repetition_penalty": 1.2,
                "repetition_codebooks": 3,
            },
            "generated": [[int(v) for v in row] for row in length],
            "eos_frame": None,
        },
        {
            "name": "speaker",
            "text": "Hi",
            "params": {
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 0.0,
                "min_p": 0.0,
                "max_tokens": MAX_TOKENS,
                "ignore_eos": False,
                "repetition_window": 50,
                "repetition_penalty": 1.2,
                "repetition_codebooks": 3,
            },
            "speaker_embedding": floats(speaker_embedding),
            "clean_speaker_background": False,
            "accurate_mode": True,
            "generated": [[int(v) for v in row] for row in speaker_frames],
            "eos_frame": speaker_eos,
        },
        {
            "name": "rate_quality",
            "text": "Hi there.",
            "params": {
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 0.0,
                "min_p": 0.0,
                "max_tokens": MAX_TOKENS,
                "ignore_eos": False,
                "repetition_window": 50,
                "repetition_penalty": 1.2,
                "repetition_codebooks": 3,
            },
            "speaking_rate_bucket": 1,
            "quality_buckets": [0],
            "generated": [[int(v) for v in row] for row in rate_frames],
            "eos_frame": rate_eos,
        },
    ]

    # The decode leg through the tiny DAC for the natural scenario.
    audio = decode(dac, generated, eos_frame, config.audio_pad_id, config.codebook_size, dac_hop)
    (OUT / "generate").mkdir(parents=True, exist_ok=True)
    (OUT / "generate" / "traces.json").write_text(
        json.dumps(
            {
                "reference_commit": REFERENCE_COMMIT,
                "scenarios": scenarios,
                "decode": {
                    "scenario": "natural",
                    "audio_tolerance": 2e-4,
                    "audio": audio,
                    "samples_per_frame": dac_hop,
                },
            }
        )
    )

    # --- sampler math on fixed logits ---------------------------------------
    def fixed_logits(codebooks, vocab):
        values = np.array(
            [
                float(np.sin(0.83 * (i + 2)) * 3.0 + np.cos(0.51 * i) * 1.7)
                for i in range(codebooks * vocab)
            ],
            dtype=np.float32,
        )
        assert len(set(values.tolist())) == len(values)
        return values

    sampling_cases = []
    # Temperature-0 argmax with the repetition penalty over a history.
    history = [[1, 2, 3, 455], [4, 5, 6, 455], [1, 0, 2, 455]]
    state = Zonos2GenerationState(
        n_codebooks=config.n_codebooks,
        eoa_id=config.eoa_id,
        text_vocab=int(config.text_vocab),
    )
    for frame in history:
        state.append(frame, ignore_eos=False)
    zero_params = TTSSamplingParams(
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        min_p=0.0,
        max_tokens=1,
        ignore_eos=False,
        repetition_window=2,
        repetition_penalty=1.2,
        repetition_codebooks=3,
    )
    frame = sample_frame(fixed_logits(3, 1026).reshape(3, 1026), state, zero_params)
    sampling_cases.append(
        {
            "name": "temperature_zero",
            "logits": floats(fixed_logits(3, 1026)),
            "params": {
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 0.0,
                "min_p": 0.0,
                "max_tokens": 1,
                "ignore_eos": False,
                "repetition_window": 2,
                "repetition_penalty": 1.2,
                "repetition_codebooks": 3,
            },
            "generated_history": history,
            "frame": [int(v) for v in frame],
        }
    )
    # The invalid-fallback branch: min_p above 1 zeroes every lane, so
    # the sampler falls back to the greedy argmax of the filtered
    # logits.
    fallback_params = TTSSamplingParams(
        temperature=1.15,
        top_k=0,
        top_p=0.0,
        min_p=1.5,
        max_tokens=1,
        ignore_eos=False,
        repetition_window=2,
        repetition_penalty=1.2,
        repetition_codebooks=3,
    )
    frame = sample_frame(fixed_logits(3, 1026).reshape(3, 1026), state, fallback_params)
    sampling_cases.append(
        {
            "name": "invalid_fallback",
            "logits": floats(fixed_logits(3, 1026)),
            "params": {
                "temperature": 1.15,
                "top_k": 0,
                "top_p": 0.0,
                "min_p": 1.5,
                "max_tokens": 1,
                "ignore_eos": False,
                "repetition_window": 2,
                "repetition_penalty": 1.2,
                "repetition_codebooks": 3,
            },
            "generated_history": history,
            "frame": [int(v) for v in frame],
        }
    )
    (OUT / "sampling").mkdir(parents=True, exist_ok=True)
    (OUT / "sampling" / "traces.json").write_text(
        json.dumps({"reference_commit": REFERENCE_COMMIT, "cases": sampling_cases})
    )

    # --- write checkpoints ----------------------------------------------------
    mx.save_safetensors(str(OUT / "generate" / "model.safetensors"), stored)
    (OUT / "generate" / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (OUT / "dac").mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(OUT / "dac" / "model.safetensors"), dac_stored)
    (OUT / "dac" / "config.json").write_text(json.dumps(DAC_CONFIG, indent=2) + "\n")

    files = {}
    for entry in sorted(OUT.rglob("*")):
        if entry.is_file() and entry.name != "manifest.json":
            files[str(entry.relative_to(OUT))] = hashlib.sha256(entry.read_bytes()).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_zonos2_fixtures.py",
        "seed": SEED,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
