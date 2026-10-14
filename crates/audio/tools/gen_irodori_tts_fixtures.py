#!/usr/bin/env python3
"""Regenerate the irodori_tts family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/irodori_tts, mlx 0.32.3). The reference modules
run in full: the Rectified Flow DiT (gated joint attention over
self/text/speaker/caption branches with per-branch KV projections and
half-of-heads interleaved RoPE, low-rank AdaLN with tanh gates,
cond/delta_cond timestep modules), the scratch and ModernBERT
(v4 pretrained backbone + projectors) text encoders, the reference
latent speaker encoder, the token-sum duration predictor with AdaRN-Zero
conditioning, the Euler RF sampler (independent and joint CFG, sway and
linear schedules, cfg window, speaker-KV scaling with rollback,
temporal score rescaling inputs), the MeanFlow distilled sampler, and
the DACVAE codec decode, with:

- a deterministic offline mock tokenizer (BOS + word ids, right pad)
  standing in for the llm-jp tokenizer, attached as `model._tokenizer`
  and `model._caption_tokenizer`;
- seeded tiny stacks: a 2-layer DiT at model_dim 32 (4 heads,
  head_dim 8), a 3-layer tiny ModernBERT backbone (hidden 32, local
  window 8, global every 2) with linear/residual_mlp projectors, a
  1-layer speaker encoder at dim 32 with speaker_patch_size 2, a
  2-layer dual token-sum duration predictor at hidden 32, and the tiny
  DACVAE (codebook_dim 8, hop 8) from the codec fixtures as the
  bundled `dacvae/` checkpoint;
- every `mx.random.normal` draw (one per generate) recorded and
  replay-verified against a fresh `generate_latents` and a second
  replay.

Per-step traces come from a verbatim replay of the reference sampler
bodies (the loops have no hooks); each replay's final latents are
asserted identical to the reference sampler's output, and the reference
entry points are re-run a second time to pin reproducibility.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_irodori_tts_fixtures.py

Outputs into crates/audio/testdata/tts_irodori_tts/:

- duration/ (scratch encoder, speaker-only, duration predictor),
  generate_rf/ (pretrained ModernBERT, dual speaker+caption,
  independent CFG with the sway schedule and a cfg window that skips
  the last step), generate_joint/ (scratch encoder, speaker-only,
  joint CFG with speaker-KV scaling and rollback, manual seconds),
  meanflow/ (MeanFlow distillation with delta conditioning),
  codec/ (standalone DACVAE encode/decode through the family path)
    Each: config.json + model.safetensors (+ dacvae/ for the generate
    cases) + traces.json.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

sys.path.insert(0, str(Path.cwd()))

from mlx_audio.codec.models.dacvae.codec import DACVAE, DACVAEConfig
from mlx_audio.tts.models.irodori_tts.duration import build_duration_features
from mlx_audio.tts.models.irodori_tts.model import IrodoriDiT
from mlx_audio.tts.models.irodori_tts.sampling import (
    sample_euler_cfg,
    sample_euler_meanflow,
)
from mlx_audio.tts.models.irodori_tts.text import normalize_text

OUT = Path("../turbospark/crates/audio/testdata/tts_irodori_tts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

WEIGHT_SEED = 11
RNG_SEED = 23
NOMINAL_TEXT = "こんにちは世界、テスト123。"

MODERNBERT = {
    "vocab_size": 96,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "hidden_activation": "gelu",
    "norm_eps": 1e-5,
    "norm_bias": False,
    "attention_bias": False,
    "mlp_bias": False,
    "global_attn_every_n_layers": 2,
    "local_attention": 8,
    "global_rope_theta": 10000.0,
    "local_rope_theta": 1000.0,
    "max_position_embeddings": 512,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

DACVAE_TINY = dict(
    encoder_dim=8,
    encoder_rates=[2, 4],
    latent_dim=16,
    decoder_dim=72,
    decoder_rates=[4, 2],
    n_codebooks=4,
    codebook_size=64,
    codebook_dim=8,
    sample_rate=24000,
)
HOP = 8  # product of the encoder rates


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


class DrawRecorder:
    """Records every mx.random.normal draw (shape + values) in order."""

    def __init__(self):
        self.draws = []
        self._original = mx.random.normal

    def __enter__(self):
        recorder = self

        def patched(*args, **kwargs):
            out = recorder._original(*args, **kwargs)
            recorder.draws.append(
                (list(out.shape), np.array(out, dtype=np.float32).ravel().tolist())
            )
            return out

        mx.random.normal = patched
        return self

    def __exit__(self, *exc):
        mx.random.normal = self._original
        return False

    def flush(self) -> list:
        draws = self.draws
        self.draws = []
        return draws


class MockTokenizer:
    """Deterministic word-id tokenizer with BOS/pad semantics."""

    BOS = 1
    PAD = 0

    def _fallback(self, word: str) -> int:
        return 4 + (sum(ord(ch) for ch in word) % 90)

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        del add_special_tokens
        return [self._fallback(word) for word in text.split() if word]

    @property
    def bos_token_id(self) -> int:
        return self.BOS

    @property
    def pad_token_id(self) -> int:
        return self.PAD


def base_config(
    *,
    pretrained: bool,
    caption: bool,
    duration: bool,
    meanflow: bool,
    max_text_length: int = 24,
) -> dict:
    dit = {
        "flow_parameterization": "meanflow" if meanflow else "rf_velocity",
        "latent_dim": 8,
        "latent_patch_size": 1,
        "model_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "text_vocab_size": 96,
        "text_add_bos": True,
        "text_dim": 32,
        "text_layers": 2,
        "text_heads": 4,
        "text_encoder_type": "pretrained" if pretrained else "scratch",
        "pretrained_projector_type": "residual_mlp" if pretrained else "linear",
        "pretrained_projector_hidden_ratio": 2.0,
        "speaker_dim": 32,
        "speaker_layers": 1,
        "speaker_heads": 4,
        "speaker_patch_size": 2,
        "timestep_embed_dim": 32,
        "adaln_rank": 8,
        "norm_eps": 1e-5,
        "use_caption_condition": caption,
        "use_speaker_condition": True,
        "use_duration_predictor": duration,
        "duration_aux_dim": 14,
        "duration_hidden_dim": 32,
        "duration_layers": 2,
        "duration_dropout": 0.0,
        "duration_attention_heads": 4,
        "duration_architecture": (
            "token_sum_dual_adarn_zero_no_aux" if caption else "token_sum_adarn_zero_no_aux"
        ),
        "duration_token_init_frames": 9.0,
    }
    if caption:
        dit["caption_dim"] = 32
        dit["caption_layers"] = 2
        dit["caption_heads"] = 4
    if pretrained:
        dit["text_encoder_config"] = dict(MODERNBERT)
    return {
        "model_type": "irodori_tts",
        "sample_rate": 24000,
        "max_text_length": max_text_length,
        "max_caption_length": 24,
        "max_speaker_latent_length": 64,
        "ref_max_seconds": 30.0,
        "audio_downsample_factor": HOP,
        "dit": dit,
        "sampler": {
            "num_steps": 3,
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 5.0,
            "cfg_scale_caption": 3.0,
            "cfg_guidance_mode": "independent",
            "cfg_min_t": 0.4,
            "cfg_max_t": 1.0,
            "truncation_factor": None,
            "rescale_k": None,
            "rescale_sigma": None,
            "context_kv_cache": True,
            "speaker_kv_scale": None,
            "speaker_kv_min_t": None,
            "sequence_length": 16,
            "t_schedule_mode": "linear",
            "sway_coeff": -1.0,
            "duration_scale": 1.0,
            "min_seconds": 0.1,
            "max_seconds": 0.2,
        },
    }


def seeded_dit(config: dict, seed: int) -> IrodoriDiT:
    """Deterministic weights: small uniform everywhere, norm gains up."""
    from mlx_audio.tts.models.irodori_tts.config import IrodoriDiTConfig

    mx.random.seed(seed)
    dit = IrodoriDiT(IrodoriDiTConfig.from_dict(json.loads(json.dumps(config["dit"]))))

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, dit.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith("norm.weight") or name.endswith("_norm.weight"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    dit.update(tree_unflatten(flat))
    return dit


def seeded_dacvae(seed: int) -> DACVAE:
    mx.random.seed(seed)
    dac = DACVAE(DACVAEConfig(**json.loads(json.dumps(DACVAE_TINY))))

    def fill(weight):
        return mx.random.uniform(low=-0.3, high=0.3, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, dac.parameters())))
    dac.update(tree_unflatten(flat))
    return dac


def attach_tokenizer(dit: IrodoriDiT) -> None:
    """The checkpoint tokenizer stands in on the DiT wrapper; the Model
    level lazy loaders are bypassed."""
    dit._tokenizer = MockTokenizer()
    dit._caption_tokenizer = MockTokenizer()


def tokenize(text: str, max_length: int, add_bos: bool) -> tuple[list, list]:
    """The reference `encode_text` over the mock tokenizer."""
    tokenizer = MockTokenizer()
    ids = list(tokenizer.encode(text))
    if add_bos:
        ids.insert(0, tokenizer.bos_token_id)
    ids = ids[:max_length]
    mask = [True] * len(ids) + [False] * (max_length - len(ids))
    ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))
    return ids, mask


def build_model(config: dict) -> tuple[IrodoriDiT, DACVAE]:
    dit = seeded_dit(config, WEIGHT_SEED)
    attach_tokenizer(dit)
    dac = seeded_dacvae(WEIGHT_SEED + 1)
    return dit, dac


def reference_audio(samples: int = 256) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.arange(samples, dtype=np.float32) / DACVAE_TINY["sample_rate"]
    return (0.2 * np.sin(2 * np.pi * 300.0 * t) + 0.05 * rng.standard_normal(samples)).astype(
        np.float32
    )


def encode_ref(dac: DACVAE, audio: np.ndarray) -> tuple[np.ndarray, int]:
    """The reference `_encode_ref_audio` on an already-loaded waveform."""
    latent = dac.encode(mx.array(audio)[None, :, None])  # (1, D, T)
    latent = np.array(mx.transpose(latent, (0, 2, 1)), dtype=np.float32)[0]
    actual_t = audio.shape[0] // HOP
    return latent[:actual_t], actual_t


def patch_rows(latent: np.ndarray, mask: list, patch_size: int) -> tuple[np.ndarray, list]:
    if patch_size <= 1:
        return latent, mask
    usable = (latent.shape[0] // patch_size) * patch_size
    groups = usable // patch_size
    patched = latent[:usable].reshape(groups, -1)
    patched_mask = [all(mask[g * patch_size : (g + 1) * patch_size]) for g in range(groups)]
    return patched, patched_mask


def encode_conditions(
    dit: IrodoriDiT,
    config: dict,
    text_ids: list,
    text_mask: list,
    ref_latent: np.ndarray,
    ref_mask: list,
    caption_ids: list | None,
    caption_mask: list | None,
):
    """The reference `encode_conditions_full` on numpy inputs."""
    cfg = config["dit"]
    text_state = dit.text_norm(
        dit._encode_text(mx.array(text_ids)[None, :], mx.array(text_mask)[None, :])
    )
    speaker_state = None
    speaker_mask = None
    if cfg.get("use_speaker_condition", True) and cfg.get("use_speaker_condition") is not False:
        patched, patched_mask = patch_rows(
            ref_latent, ref_mask, cfg["speaker_patch_size"]
        )
        state = dit.speaker_norm(
            dit.speaker_encoder(
                mx.array(patched)[None, :], mx.array(patched_mask)[None, :]
            )
        )
        speaker_state = np.array(state, dtype=np.float32)[0]
        speaker_mask = patched_mask
    caption_state = None
    cap_mask_out = None
    if cfg.get("use_caption_condition"):
        state = dit.caption_norm(
            dit._encode_caption(
                mx.array(caption_ids)[None, :], mx.array(caption_mask)[None, :]
            )
        )
        caption_state = np.array(state, dtype=np.float32)[0]
        cap_mask_out = list(caption_mask)
    return {
        "text_state": np.array(text_state, dtype=np.float32)[0],
        "text_mask": list(text_mask),
        "speaker_state": speaker_state,
        "speaker_mask": speaker_mask,
        "caption_state": caption_state,
        "caption_mask": cap_mask_out,
    }


def run_sampler(
    dit: IrodoriDiT,
    config: dict,
    conditions,
    patched_steps: int,
    sampler_overrides: dict,
    capture: bool,
):
    """Verbatim replay of the reference sampler bodies with per-step
    capture. Returns (sink, x_t, draws)."""
    cfg = dict(config["sampler"])
    cfg.update(sampler_overrides)
    latent_dim = config["dit"]["latent_dim"]
    batch = 1
    num_steps = cfg["num_steps"]

    with DrawRecorder() as recorder:
        mx.random.seed(RNG_SEED)
        if config["dit"]["flow_parameterization"] == "meanflow":
            kv_text = [
                b.attention.get_kv_cache_text(mx.array(conditions["text_state"])[None, :])
                for b in dit.blocks
            ]
            kv_speaker = (
                [
                    b.attention.get_kv_cache_speaker(
                        mx.array(conditions["speaker_state"])[None, :]
                    )
                    for b in dit.blocks
                ]
                if conditions["speaker_state"] is not None
                else None
            )
            kv_caption = None
            x_t = mx.random.normal((batch, patched_steps, latent_dim))
            sink = []
            t_schedule = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float32)
            for i in range(num_steps):
                t = float(t_schedule[i])
                t_next = float(t_schedule[i + 1])
                delta = float(t - t_next)
                velocity = dit.forward_with_conditions(
                    x_t=x_t,
                    t=mx.full((batch,), t),
                    text_state=mx.array(conditions["text_state"])[None, :],
                    text_mask=mx.array(conditions["text_mask"])[None, :],
                    speaker_state=(
                        mx.array(conditions["speaker_state"])[None, :]
                        if conditions["speaker_state"] is not None
                        else None
                    ),
                    speaker_mask=(
                        mx.array(conditions["speaker_mask"])[None, :]
                        if conditions["speaker_mask"] is not None
                        else None
                    ),
                    kv_text=kv_text,
                    kv_speaker=kv_speaker,
                    caption_state=None,
                    caption_mask=None,
                    kv_caption=None,
                    delta_t=mx.full((batch,), delta),
                )
                if capture:
                    sink.append(
                        {"index": i, "t": t, "delta": delta, "velocity": floats(velocity), "x_t": floats(x_t)}
                    )
                x_t = x_t + velocity * (t_next - t)
                mx.eval(x_t)
            draws = recorder.flush()
            return sink, np.array(x_t, dtype=np.float32)[0], draws

        # RF with CFG.
        use_spk = conditions["speaker_state"] is not None
        use_cap = conditions["caption_state"] is not None
        is_dual = use_spk and use_cap
        cfg_scale_text = float(cfg["cfg_scale_text"])
        cfg_scale_speaker = float(cfg["cfg_scale_speaker"])
        cfg_scale_caption = float(cfg["cfg_scale_caption"])
        cfg_scale_context = (
            cfg_scale_caption if (use_cap and not is_dual) else cfg_scale_speaker
        )
        has_text_cfg = cfg_scale_text > 0
        has_speaker_cfg = cfg_scale_speaker > 0 and use_spk
        has_caption_cfg = (
            cfg_scale_caption > 0
            and use_cap
            and any(conditions["caption_mask"] or [])
        )
        has_context_cfg = cfg_scale_context > 0 if not is_dual else False

        def zeros_like(arr):
            return None if arr is None else np.zeros_like(arr)

        text_state_uncond = zeros_like(conditions["text_state"])
        text_mask_uncond = [False] * len(conditions["text_mask"])
        speaker_state_uncond = zeros_like(conditions["speaker_state"])
        speaker_mask_uncond = (
            [False] * len(conditions["speaker_mask"])
            if conditions["speaker_mask"] is not None
            else None
        )
        caption_state_uncond = zeros_like(conditions["caption_state"])
        caption_mask_uncond = (
            [False] * len(conditions["caption_mask"])
            if conditions["caption_mask"] is not None
            else None
        )

        kv_text = [
                b.attention.get_kv_cache_text(mx.array(conditions["text_state"])[None, :])
                for b in dit.blocks
            ]
        kv_speaker = (
            [b.attention.get_kv_cache_speaker(mx.array(conditions["speaker_state"])[None, :]) for b in dit.blocks]
            if use_spk
            else None
        )
        kv_caption = (
            [b.attention.get_kv_cache_caption(mx.array(conditions["caption_state"])[None, :]) for b in dit.blocks]
            if use_cap
            else None
        )
        speaker_kv_scale = cfg.get("speaker_kv_scale")
        if speaker_kv_scale is not None and kv_speaker is not None:
            kv_speaker = [(k * speaker_kv_scale, v * speaker_kv_scale) for k, v in kv_speaker]
        speaker_kv_active = speaker_kv_scale is not None

        x_t = mx.random.normal((batch, patched_steps, latent_dim))
        init_scale = 0.999
        mode = str(cfg["t_schedule_mode"]).strip().lower()
        if mode == "sway":
            u = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
            u = u + float(cfg["sway_coeff"]) * (np.cos(0.5 * np.pi * u) + u - 1.0)
            u = np.clip(u, 0.0, 1.0)
            t_schedule = (1.0 - u) * init_scale
        else:
            t_schedule = np.linspace(1.0 * init_scale, 0.0, num_steps + 1, dtype=np.float32)

        sink = []
        for i in range(num_steps):
            t = float(t_schedule[i])
            t_next = float(t_schedule[i + 1])
            use_cfg = (has_text_cfg or has_speaker_cfg or has_caption_cfg) and (
                float(cfg["cfg_min_t"]) <= t <= float(cfg["cfg_max_t"])
            )
            if use_cfg:
                if cfg["cfg_guidance_mode"] == "independent":
                    if not is_dual:
                        bundles = []
                        if has_text_cfg:
                            bundles.append((text_state_uncond, text_mask_uncond, speaker_state_uncond if False else None, None, "text"))
                        if has_context_cfg:
                            bundles.append((None, None, speaker_state_uncond, speaker_mask_uncond, "context"))
                        v_pred = None
                        v_cond = dit.forward_with_conditions(
                            x_t=x_t,
                            t=mx.full((batch,), t),
                            text_state=mx.array(conditions["text_state"])[None, :],
                            text_mask=mx.array(conditions["text_mask"])[None, :],
                            speaker_state=(
                                mx.array(conditions["speaker_state"])[None, :]
                                if conditions["speaker_state"] is not None
                                else None
                            ),
                            speaker_mask=(
                                mx.array(conditions["speaker_mask"])[None, :]
                                if conditions["speaker_mask"] is not None
                                else None
                            ),
                            kv_text=kv_text,
                            kv_speaker=kv_speaker,
                        )
                        v_pred = np.array(v_cond, dtype=np.float32)
                        for bundle in bundles:
                            kind = bundle[4]
                            if kind == "text":
                                v_uncond = dit.forward_with_conditions(
                                    x_t=x_t,
                                    t=mx.full((batch,), t),
                                    text_state=mx.array(text_state_uncond)[None, :],
                                    text_mask=mx.array(text_mask_uncond)[None, :],
                                    speaker_state=(
                                        mx.array(conditions["speaker_state"])[None, :]
                                        if conditions["speaker_state"] is not None
                                        else None
                                    ),
                                    speaker_mask=(
                                        mx.array(conditions["speaker_mask"])[None, :]
                                        if conditions["speaker_mask"] is not None
                                        else None
                                    ),
                                    kv_text=kv_text,
                                    kv_speaker=kv_speaker,
                                )
                                v_pred = v_pred + cfg_scale_text * (
                                    np.array(v_cond, dtype=np.float32) - np.array(v_uncond, dtype=np.float32)
                                )
                            else:
                                v_uncond = dit.forward_with_conditions(
                                    x_t=x_t,
                                    t=mx.full((batch,), t),
                                    text_state=mx.array(conditions["text_state"])[None, :],
                                    text_mask=mx.array(conditions["text_mask"])[None, :],
                                    speaker_state=mx.array(speaker_state_uncond)[None, :],
                                    speaker_mask=mx.array(speaker_mask_uncond)[None, :],
                                    kv_text=kv_text,
                                    kv_speaker=kv_speaker,
                                )
                                v_pred = v_pred + cfg_scale_context * (
                                    np.array(v_cond, dtype=np.float32) - np.array(v_uncond, dtype=np.float32)
                                )
                    else:
                        v_cond = dit.forward_with_conditions(
                            x_t=x_t,
                            t=mx.full((batch,), t),
                            text_state=mx.array(conditions["text_state"])[None, :],
                            text_mask=mx.array(conditions["text_mask"])[None, :],
                            speaker_state=mx.array(conditions["speaker_state"])[None, :],
                            speaker_mask=mx.array(conditions["speaker_mask"])[None, :],
                            kv_text=kv_text,
                            kv_speaker=kv_speaker,
                            caption_state=mx.array(conditions["caption_state"])[None, :],
                            caption_mask=mx.array(conditions["caption_mask"])[None, :],
                            kv_caption=kv_caption,
                        )
                        v_pred = np.array(v_cond, dtype=np.float32)
                        if has_text_cfg:
                            v_uncond = dit.forward_with_conditions(
                                x_t=x_t,
                                t=mx.full((batch,), t),
                                text_state=mx.array(text_state_uncond)[None, :],
                                text_mask=mx.array(text_mask_uncond)[None, :],
                                speaker_state=mx.array(conditions["speaker_state"])[None, :],
                                speaker_mask=mx.array(conditions["speaker_mask"])[None, :],
                                kv_text=kv_text,
                                kv_speaker=kv_speaker,
                                caption_state=mx.array(conditions["caption_state"])[None, :],
                                caption_mask=mx.array(conditions["caption_mask"])[None, :],
                                kv_caption=kv_caption,
                            )
                            v_pred = v_pred + cfg_scale_text * (
                                np.array(v_cond, dtype=np.float32) - np.array(v_uncond, dtype=np.float32)
                            )
                        if has_speaker_cfg:
                            v_uncond = dit.forward_with_conditions(
                                x_t=x_t,
                                t=mx.full((batch,), t),
                                text_state=mx.array(conditions["text_state"])[None, :],
                                text_mask=mx.array(conditions["text_mask"])[None, :],
                                speaker_state=mx.array(speaker_state_uncond)[None, :],
                                speaker_mask=mx.array(speaker_mask_uncond)[None, :],
                                kv_text=kv_text,
                                kv_speaker=kv_speaker,
                                caption_state=mx.array(conditions["caption_state"])[None, :],
                                caption_mask=mx.array(conditions["caption_mask"])[None, :],
                                kv_caption=kv_caption,
                            )
                            v_pred = v_pred + cfg_scale_speaker * (
                                np.array(v_cond, dtype=np.float32) - np.array(v_uncond, dtype=np.float32)
                            )
                        if has_caption_cfg:
                            v_uncond = dit.forward_with_conditions(
                                x_t=x_t,
                                t=mx.full((batch,), t),
                                text_state=mx.array(conditions["text_state"])[None, :],
                                text_mask=mx.array(conditions["text_mask"])[None, :],
                                speaker_state=mx.array(conditions["speaker_state"])[None, :],
                                speaker_mask=mx.array(conditions["speaker_mask"])[None, :],
                                kv_text=kv_text,
                                kv_speaker=kv_speaker,
                                caption_state=mx.array(caption_state_uncond)[None, :],
                                caption_mask=mx.array(caption_mask_uncond)[None, :],
                                kv_caption=kv_caption,
                            )
                            v_pred = v_pred + cfg_scale_caption * (
                                np.array(v_cond, dtype=np.float32) - np.array(v_uncond, dtype=np.float32)
                            )
                else:  # joint
                    if is_dual:
                        active_scales = [
                            s
                            for s, a in [
                                (cfg_scale_text, has_text_cfg),
                                (cfg_scale_speaker, has_speaker_cfg),
                                (cfg_scale_caption, has_caption_cfg),
                            ]
                            if a
                        ]
                        joint_scale = active_scales[0] if active_scales else cfg_scale_text
                    elif has_text_cfg and has_context_cfg:
                        joint_scale = cfg_scale_text
                    elif has_text_cfg:
                        joint_scale = cfg_scale_text
                    else:
                        joint_scale = cfg_scale_context
                    v_cond = dit.forward_with_conditions(
                        x_t=x_t,
                        t=mx.full((batch,), t),
                        text_state=mx.array(conditions["text_state"])[None, :],
                        text_mask=mx.array(conditions["text_mask"])[None, :],
                        speaker_state=(
                            mx.array(conditions["speaker_state"])[None, :]
                            if conditions["speaker_state"] is not None
                            else None
                        ),
                        speaker_mask=(
                            mx.array(conditions["speaker_mask"])[None, :]
                            if conditions["speaker_mask"] is not None
                            else None
                        ),
                        kv_text=kv_text,
                        kv_speaker=kv_speaker,
                    )
                    if use_spk:
                        kv_text_u = [
                            b.attention.get_kv_cache_text(mx.array(text_state_uncond)[None, :])
                            for b in dit.blocks
                        ]
                        kv_speaker_u = [
                            b.attention.get_kv_cache_speaker(
                                mx.array(speaker_state_uncond)[None, :]
                            )
                            for b in dit.blocks
                        ]
                    else:
                        kv_text_u = [
                            b.attention.get_kv_cache_text(mx.array(text_state_uncond)[None, :])
                            for b in dit.blocks
                        ]
                        kv_speaker_u = None
                    v_uncond = dit.forward_with_conditions(
                        x_t=x_t,
                        t=mx.full((batch,), t),
                        text_state=mx.array(text_state_uncond)[None, :],
                        text_mask=mx.array(text_mask_uncond)[None, :],
                        speaker_state=(
                            mx.array(speaker_state_uncond)[None, :] if use_spk else None
                        ),
                        speaker_mask=(
                            mx.array(speaker_mask_uncond)[None, :] if use_spk else None
                        ),
                        kv_text=kv_text_u,
                        kv_speaker=kv_speaker_u,
                    )
                    v_pred = np.array(v_cond, dtype=np.float32) + joint_scale * (
                        np.array(v_cond, dtype=np.float32) - np.array(v_uncond, dtype=np.float32)
                    )
            else:
                v_cond = dit.forward_with_conditions(
                    x_t=x_t,
                    t=mx.full((batch,), t),
                    text_state=mx.array(conditions["text_state"])[None, :],
                    text_mask=mx.array(conditions["text_mask"])[None, :],
                    speaker_state=(
                        mx.array(conditions["speaker_state"])[None, :]
                        if conditions["speaker_state"] is not None
                        else None
                    ),
                    speaker_mask=(
                        mx.array(conditions["speaker_mask"])[None, :]
                        if conditions["speaker_mask"] is not None
                        else None
                    ),
                    kv_text=kv_text,
                    kv_speaker=kv_speaker,
                    caption_state=(
                        mx.array(conditions["caption_state"])[None, :]
                        if conditions["caption_state"] is not None
                        else None
                    ),
                    caption_mask=(
                        mx.array(conditions["caption_mask"])[None, :]
                        if conditions["caption_mask"] is not None
                        else None
                    ),
                    kv_caption=kv_caption,
                )
                v_pred = np.array(v_cond, dtype=np.float32)

            if speaker_kv_active and cfg.get("speaker_kv_min_t") is not None:
                min_t = float(cfg["speaker_kv_min_t"])
                if t_next < min_t <= t and kv_speaker is not None:
                    inv = 1.0 / speaker_kv_scale
                    kv_speaker = [(k * inv, v * inv) for k, v in kv_speaker]
                    speaker_kv_active = False

            if capture:
                sink.append(
                    {
                        "index": i,
                        "t": t,
                        "t_next": t_next,
                        "use_cfg": bool(use_cfg),
                        "v_pred": floats(v_pred),
                        "x_t": floats(x_t),
                    }
                )
            x_t = mx.array(x_t) + mx.array(v_pred) * (t_next - t)
            mx.eval(x_t)

        draws = recorder.flush()
        return sink, np.array(x_t, dtype=np.float32)[0], draws


def save_case(name: str, config: dict, dit: IrodoriDiT, dac: DACVAE | None, traces: dict) -> None:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    raw = {"model." + k: w for k, w in tree_flatten(dit.parameters())}
    mx.save_safetensors(str(case_dir / "model.safetensors"), raw)
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if dac is not None:
        dac_dir = case_dir / "dacvae"
        dac_dir.mkdir(exist_ok=True)
        (dac_dir / "config.json").write_text(
            json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in DACVAE_TINY.items()}, indent=2) + "\n"
        )
        mx.save_safetensors(
            str(dac_dir / "model.safetensors"), dict(tree_flatten(dac.parameters()))
        )
    (case_dir / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")


def run_duration_case() -> None:
    config = base_config(pretrained=False, caption=False, duration=True, meanflow=False)
    dit, _ = build_model(config)
    text = NOMINAL_TEXT
    normalized = normalize_text(text).strip()
    text_ids, text_mask = tokenize(normalized, config["max_text_length"], True)
    rng = np.random.default_rng(31)
    ref_latent = rng.uniform(-1.0, 1.0, size=(6, 8)).astype(np.float32)
    ref_mask = [True] * 6

    features = np.array(
        build_duration_features(
            [normalized],
            token_counts=[sum(text_mask)],
            max_text_len=config["max_text_length"],
            has_speaker=[True],
        ),
        dtype=np.float32,
    )[0]

    conditions = encode_conditions(
        dit, config, text_ids, text_mask, ref_latent, ref_mask, None, None
    )
    has_speaker_arr = mx.array([True], dtype=mx.bool_)
    pred_log = dit.predict_duration_log_frames(
        text_state=mx.array(conditions["text_state"])[None, :],
        text_mask=mx.array(conditions["text_mask"])[None, :],
        speaker_state=mx.array(conditions["speaker_state"])[None, :],
        speaker_mask=mx.array(conditions["speaker_mask"])[None, :],
        duration_features=mx.array(features)[None, :],
        has_speaker=has_speaker_arr,
    )
    pred_log_value = float(pred_log[0].item())
    pred_frames = float(np.expm1(pred_log_value))
    duration_scale = 1.3
    min_seconds, max_seconds = 0.1, 0.2
    min_frames = max(1, math.ceil(min_seconds * config["sample_rate"] / HOP))
    max_frames = max(1, math.floor(max_seconds * config["sample_rate"] / HOP))
    scaled = pred_frames * duration_scale
    latent_steps = int(round(scaled))
    latent_steps = max(min_frames, min(max_frames, latent_steps))

    traces = {
        "mode": "duration",
        "weight_seed": WEIGHT_SEED,
        "text": text,
        "normalized_text": normalized,
        "prompt": {
            "ids": text_ids,
            "mask": text_mask,
            "ref_latent": floats(ref_latent),
            "ref_rows": int(ref_latent.shape[0]),
            "ref_mask": ref_mask,
        },
        "features": floats(features),
        "conditions": {
            "text_state": floats(conditions["text_state"]),
            "speaker_state": floats(conditions["speaker_state"]),
            "speaker_mask": list(conditions["speaker_mask"]),
        },
        "pred_log_frames": pred_log_value,
        "pred_frames": pred_frames,
        "duration_scale": duration_scale,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "latent_steps": latent_steps,
    }
    save_case("duration", config, dit, None, traces)
    print(f"duration: features ok, pred {pred_frames:.3f} -> {latent_steps} frames")


def run_generate_rf_case() -> None:
    config = base_config(pretrained=True, caption=True, duration=True, meanflow=False)
    config["sampler"]["t_schedule_mode"] = "sway"
    config["sampler"]["sway_coeff"] = -1.0
    # The sway schedule lands near t = 0.999, 0.865, 0.5; a 0.6 floor
    # makes the last step skip CFG, exercising both paths.
    config["sampler"]["cfg_min_t"] = 0.6
    dit, dac = build_model(config)
    text = NOMINAL_TEXT
    normalized = normalize_text(text).strip()
    caption = "落ち着いた女性の声。"
    text_ids, text_mask = tokenize(normalized, config["max_text_length"], True)
    caption_ids, caption_mask = tokenize(caption.strip(), config["max_caption_length"], True)

    rng = np.random.default_rng(41)
    ref_latent = rng.uniform(-1.0, 1.0, size=(6, 8)).astype(np.float32)
    ref_mask = [True] * 6

    features = np.array(
        build_duration_features(
            [normalized],
            token_counts=[sum(text_mask)],
            max_text_len=config["max_text_length"],
            has_speaker=[True],
        ),
        dtype=np.float32,
    )[0]

    conditions = encode_conditions(
        dit, config, text_ids, text_mask, ref_latent, ref_mask, caption_ids, caption_mask
    )
    pred_log = dit.predict_duration_log_frames(
        text_state=mx.array(conditions["text_state"])[None, :],
        text_mask=mx.array(conditions["text_mask"])[None, :],
        speaker_state=mx.array(conditions["speaker_state"])[None, :],
        speaker_mask=mx.array(conditions["speaker_mask"])[None, :],
        duration_features=mx.array(features)[None, :],
        has_speaker=mx.array([True], dtype=mx.bool_),
        caption_state=mx.array(conditions["caption_state"])[None, :],
        caption_mask=mx.array(conditions["caption_mask"])[None, :],
        has_caption=mx.array([True], dtype=mx.bool_),
    )
    pred_frames = float(np.expm1(float(pred_log[0].item())))
    min_frames = max(1, math.ceil(config["sampler"]["min_seconds"] * config["sample_rate"] / HOP))
    max_frames = max(1, math.floor(config["sampler"]["max_seconds"] * config["sample_rate"] / HOP))
    latent_steps = max(min_frames, min(max_frames, int(round(pred_frames))))
    patched_steps = math.ceil(latent_steps / config["dit"]["latent_patch_size"])

    sink, latent, draws = run_sampler(dit, config, conditions, patched_steps, {}, capture=True)

    # The reference sampler entry point must reproduce the replay.
    with DrawRecorder() as recorder:
        mx.random.seed(RNG_SEED)
        reference = sample_euler_cfg(
            model=dit,
            text_input_ids=mx.array(text_ids)[None, :],
            text_mask=mx.array(text_mask)[None, :],
            ref_latent=mx.array(ref_latent)[None, :],
            ref_mask=mx.array(ref_mask)[None, :],
            caption_input_ids=mx.array(caption_ids)[None, :],
            caption_mask=mx.array(caption_mask)[None, :],
            latent_dim=config["dit"]["latent_dim"],
            rng_seed=RNG_SEED,
            sequence_length=patched_steps,
            num_steps=config["sampler"]["num_steps"],
            cfg_scale_text=config["sampler"]["cfg_scale_text"],
            cfg_scale_speaker=config["sampler"]["cfg_scale_speaker"],
            cfg_scale_caption=config["sampler"]["cfg_scale_caption"],
            cfg_guidance_mode="independent",
            cfg_min_t=config["sampler"]["cfg_min_t"],
            cfg_max_t=config["sampler"]["cfg_max_t"],
            context_kv_cache=True,
            t_schedule_mode="sway",
            sway_coeff=-1.0,
        )
        reference_draws = recorder.flush()
    assert reference_draws == draws, "sampler draw replay diverged"
    # The reference batches the CFG bundles; kernel fusion drift
    # accumulates over the Euler steps, so agreement holds to 1e-4 on the
    # unit-scale latents. The Rust parity gates run against the
    # SEQUENTIAL replay, which is the exact single-row semantics.
    diff = float(np.abs(np.array(reference)[0] - latent).max())
    assert diff < 1e-4, f"sampler replay diverged: {diff}"
    print(f"  sampler replay agreement {diff:.2e}")

    # Decode through the DACVAE and trim (the reference generate tail).
    latent_for_decode = mx.transpose(mx.array(latent)[None, :], (0, 2, 1))
    audio = np.array(dac.decode(latent_for_decode), dtype=np.float32)[0]
    target_samples = latent_steps * HOP
    silence = find_silence_point(latent)
    trim = min(target_samples, silence * HOP) if silence > 0 else target_samples
    waveform = audio[:trim]

    assert any(not step["use_cfg"] for step in sink), "expected a skipped-CFG step"
    traces = {
        "mode": "generate_rf",
        "weight_seed": WEIGHT_SEED,
        "rng_seed": RNG_SEED,
        "text": text,
        "normalized_text": normalized,
        "caption": caption,
        "prompt": {
            "ids": text_ids,
            "mask": text_mask,
            "caption_ids": caption_ids,
            "caption_mask": caption_mask,
            "ref_latent": floats(ref_latent),
            "ref_rows": int(ref_latent.shape[0]),
            "ref_mask": ref_mask,
        },
        "features": floats(features),
        "conditions": {
            "text_state": floats(conditions["text_state"]),
            "text_mask": list(conditions["text_mask"]),
            "speaker_state": floats(conditions["speaker_state"]),
            "speaker_mask": list(conditions["speaker_mask"]),
            "caption_state": floats(conditions["caption_state"]),
            "caption_mask": list(conditions["caption_mask"]),
        },
        "pred_frames": pred_frames,
        "latent_steps": latent_steps,
        "patched_steps": patched_steps,
        "noise_draws": [
            {"shape": shape, "values": values} for shape, values in draws
        ],
        "steps": sink,
        "latent": floats(latent),
        "waveform": floats(waveform),
        "trim": int(trim),
        "sample_rate": int(config["sample_rate"]),
    }
    save_case("generate_rf", config, dit, dac, traces)
    print(f"generate_rf: {latent_steps} frames ({patched_steps} patches), {trim} samples")


def find_silence_point(latent: np.ndarray, window_size: int = 20) -> int:
    padded = np.concatenate([latent, np.zeros((window_size, latent.shape[-1]) if latent.ndim > 1 else (window_size,), dtype=latent.dtype)], axis=0)
    for i in range(int(padded.shape[0] - window_size)):
        window = padded[i : i + window_size]
        if float(window.std()) < 0.05 and abs(float(window.mean())) < 0.1:
            return i
    return int(latent.shape[0])


def run_generate_joint_case() -> None:
    config = base_config(pretrained=False, caption=False, duration=False, meanflow=False)
    config["sampler"]["cfg_guidance_mode"] = "joint"
    config["sampler"]["cfg_scale_text"] = 2.0
    config["sampler"]["cfg_scale_speaker"] = 2.0
    config["sampler"]["cfg_min_t"] = 0.3
    config["sampler"]["speaker_kv_scale"] = 0.9
    config["sampler"]["speaker_kv_min_t"] = 0.7
    dit, dac = build_model(config)
    text = "テスト、joint。"
    normalized = normalize_text(text).strip()
    text_ids, text_mask = tokenize(normalized, config["max_text_length"], True)
    rng = np.random.default_rng(51)
    ref_latent = rng.uniform(-1.0, 1.0, size=(6, 8)).astype(np.float32)
    ref_mask = [True] * 6

    # Manual duration: 0.12 s -> 360 frames at hop 8... keep it small:
    # 0.12 * 24000 / 8 = 360. Use seconds clamped to min 0.1.
    seconds = 0.12
    clamped = min(0.2, max(0.1, seconds))
    latent_steps = math.ceil(clamped * config["sample_rate"] / HOP)
    patched_steps = math.ceil(latent_steps / config["dit"]["latent_patch_size"])

    conditions = encode_conditions(
        dit, config, text_ids, text_mask, ref_latent, ref_mask, None, None
    )
    sink, latent, draws = run_sampler(dit, config, conditions, patched_steps, {}, capture=True)
    assert any(step["t_next"] < 0.7 <= step["t"] for step in sink), "expected the speaker-KV rollback"

    with DrawRecorder() as recorder:
        mx.random.seed(RNG_SEED)
        reference = sample_euler_cfg(
            model=dit,
            text_input_ids=mx.array(text_ids)[None, :],
            text_mask=mx.array(text_mask)[None, :],
            ref_latent=mx.array(ref_latent)[None, :],
            ref_mask=mx.array(ref_mask)[None, :],
            latent_dim=config["dit"]["latent_dim"],
            rng_seed=RNG_SEED,
            sequence_length=patched_steps,
            num_steps=config["sampler"]["num_steps"],
            cfg_scale_text=2.0,
            cfg_scale_speaker=2.0,
            cfg_guidance_mode="joint",
            cfg_min_t=0.3,
            cfg_max_t=1.0,
            context_kv_cache=True,
            speaker_kv_scale=0.9,
            speaker_kv_min_t=0.7,
            t_schedule_mode="linear",
        )
        reference_draws = recorder.flush()
    assert reference_draws == draws, "sampler draw replay diverged"
    diff = float(np.abs(np.array(reference)[0] - latent).max())
    assert diff < 1e-4, f"sampler replay diverged: {diff}"
    print(f"  sampler replay agreement {diff:.2e}")

    latent_for_decode = mx.transpose(mx.array(latent)[None, :], (0, 2, 1))
    audio = np.array(dac.decode(latent_for_decode), dtype=np.float32)[0]
    trim = latent_steps * HOP
    waveform = audio[:trim]

    traces = {
        "mode": "generate_joint",
        "weight_seed": WEIGHT_SEED,
        "rng_seed": RNG_SEED,
        "text": text,
        "normalized_text": normalized,
        "prompt": {
            "ids": text_ids,
            "mask": text_mask,
            "ref_latent": floats(ref_latent),
            "ref_rows": int(ref_latent.shape[0]),
            "ref_mask": ref_mask,
        },
        "seconds": seconds,
        "clamped_seconds": clamped,
        "conditions": {
            "text_state": floats(conditions["text_state"]),
            "text_mask": list(conditions["text_mask"]),
            "speaker_state": floats(conditions["speaker_state"]),
            "speaker_mask": list(conditions["speaker_mask"]),
        },
        "latent_steps": latent_steps,
        "patched_steps": patched_steps,
        "noise_draws": [
            {"shape": shape, "values": values} for shape, values in draws
        ],
        "steps": sink,
        "latent": floats(latent),
        "waveform": floats(waveform),
        "trim": int(trim),
        "sample_rate": int(config["sample_rate"]),
    }
    save_case("generate_joint", config, dit, dac, traces)
    print(f"generate_joint: {latent_steps} frames, {trim} samples")


def run_meanflow_case() -> None:
    config = base_config(pretrained=False, caption=False, duration=False, meanflow=True)
    dit, dac = build_model(config)
    text = "平均フローテスト。"
    normalized = normalize_text(text).strip()
    text_ids, text_mask = tokenize(normalized, config["max_text_length"], True)
    rng = np.random.default_rng(61)
    ref_latent = rng.uniform(-1.0, 1.0, size=(6, 8)).astype(np.float32)
    ref_mask = [True] * 6

    seconds = 0.1
    latent_steps = math.ceil(seconds * config["sample_rate"] / HOP)
    patched_steps = math.ceil(latent_steps / config["dit"]["latent_patch_size"])

    conditions = encode_conditions(
        dit, config, text_ids, text_mask, ref_latent, ref_mask, None, None
    )
    sink, latent, draws = run_sampler(dit, config, conditions, patched_steps, {}, capture=True)

    with DrawRecorder() as recorder:
        mx.random.seed(RNG_SEED)
        reference = sample_euler_meanflow(
            model=dit,
            text_input_ids=mx.array(text_ids)[None, :],
            text_mask=mx.array(text_mask)[None, :],
            ref_latent=mx.array(ref_latent)[None, :],
            ref_mask=mx.array(ref_mask)[None, :],
            latent_dim=config["dit"]["latent_dim"],
            rng_seed=RNG_SEED,
            sequence_length=patched_steps,
            num_steps=config["sampler"]["num_steps"],
        )
        reference_draws = recorder.flush()
    assert reference_draws == draws, "sampler draw replay diverged"
    # MeanFlow runs one row per step, so this one is bit-exact.
    assert np.array_equal(np.array(reference)[0], latent), "sampler replay diverged"

    latent_for_decode = mx.transpose(mx.array(latent)[None, :], (0, 2, 1))
    audio = np.array(dac.decode(latent_for_decode), dtype=np.float32)[0]
    trim = latent_steps * HOP
    waveform = audio[:trim]

    traces = {
        "mode": "meanflow",
        "weight_seed": WEIGHT_SEED,
        "rng_seed": RNG_SEED,
        "text": text,
        "normalized_text": normalized,
        "prompt": {
            "ids": text_ids,
            "mask": text_mask,
            "ref_latent": floats(ref_latent),
            "ref_rows": int(ref_latent.shape[0]),
            "ref_mask": ref_mask,
        },
        "conditions": {
            "text_state": floats(conditions["text_state"]),
            "text_mask": list(conditions["text_mask"]),
            "speaker_state": floats(conditions["speaker_state"]),
            "speaker_mask": list(conditions["speaker_mask"]),
        },
        "latent_steps": latent_steps,
        "patched_steps": patched_steps,
        "noise_draws": [
            {"shape": shape, "values": values} for shape, values in draws
        ],
        "steps": sink,
        "latent": floats(latent),
        "waveform": floats(waveform),
        "trim": int(trim),
        "sample_rate": int(config["sample_rate"]),
    }
    save_case("meanflow", config, dit, dac, traces)
    print(f"meanflow: {latent_steps} frames, {trim} samples")


def run_codec_case() -> None:
    config = base_config(pretrained=False, caption=False, duration=False, meanflow=False)
    dit, dac = build_model(config)
    audio = reference_audio(256)
    latent, actual_t = encode_ref(dac, audio)
    decoded = np.array(dac.decode(mx.array(latent.T)[None, :]), dtype=np.float32)[0]

    traces = {
        "mode": "codec",
        "weight_seed": WEIGHT_SEED,
        "audio": floats(audio),
        "audio_samples": int(audio.shape[0]),
        "ref_latent": floats(latent),
        "ref_rows": int(actual_t),
        "latent_dim": int(DACVAE_TINY["codebook_dim"]),
        "decoded": floats(decoded),
        "sample_rate": int(config["sample_rate"]),
        "audio_downsample_factor": HOP,
    }
    save_case("codec", config, dit, dac, traces)
    print(f"codec: {actual_t} latent frames, {decoded.shape[0]} samples")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_duration_case()
    run_generate_rf_case()
    run_generate_joint_case()
    run_meanflow_case()
    run_codec_case()

    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "generator": "tools/gen_irodori_tts_fixtures.py",
        "files": {
            str(path.relative_to(OUT)): "sha256:"
            + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures under {OUT}")


if __name__ == "__main__":
    main()
