#!/usr/bin/env python3
"""Regenerate the voxcpm2 (VoxCPM2 2B tokenizer-free 48 kHz) family
Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/voxcpm2, mlx 0.32.3). The reference modules run
in full: the MiniCPM2 backbone (use_mup=True, kv_channels-free tiny
heads, optional no_rope residual LM), the CLS patch encoder, the
ScalarQuantizationLayer, the fusion_concat_proj residual mix, the
VoxCPMLocDiTV2 denoiser (two-token mu block, delta-time token, mean
mode off) inside the UnifiedCFM Euler solver (CFG zero-star, sway
sampling), the stop predictor with the `real_steps > min_tokens` gate,
the causal-conv AudioVAE V2 with the per-block scale_bias
sample-rate conditioning (bucket of 48000 under [20000, 30000,
40000]), and the reference `Model.generate` loop with:

- a deterministic offline mock tokenizer (`encode` word ids) standing
  in for the checkpoint tokenizer, injected as `model.tokenizer`
  (the `_tokenize` wrapper is a tokenizer concern the caller owns);
- seeded tiny stacks: 2-layer backbones at hidden 32, 4 query / 2 kv
  heads, a 1-block AudioVAE V2 (encoder rates [2], decoder rates [4],
  latent 8, sr buckets 4) at 16 kHz encode / 48 kHz output;
- every `mx.random.normal` draw recorded (one flow draw per generate
  step) and replay-verified across two full runs of each case.

Four prompt modes: `zero_shot` (text only, stop fires),
`length` (zero-shot to max_tokens), `ref_clone` (reference audio
prefix [103 | patches | 104]), `continuation` (prompt audio suffix,
warmup=0, decode front trim). The stop-biased cases bias `stop_proj`
so class 1 dominates and the run ends at `real_steps == min_tokens +
1`; the never-stop cases zero both stop rows.

Per-step traces come from a verbatim replay of the reference generate
body (the loop has no hooks); the replay's waveform and draw sequence
are asserted identical to a fresh `model.generate` call, and the
replay is re-run a second time to pin reproducibility.

Run from inside the mlx-audio checkout root so `mlx_audio` imports
from the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_voxcpm2_fixtures.py

Outputs into crates/audio/testdata/tts_voxcpm2/:
zero_shot/, length/, ref_clone/, continuation/ (config.json +
model.safetensors + traces.json each) and manifest.json.

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

import mlx_audio.tts.models.voxcpm2.audio_vae as audio_vae_mod
import mlx_audio.tts.models.voxcpm2.voxcpm2 as voxcpm2_mod
from mlx_audio.tts.models.voxcpm2.config import ModelArgs
from mlx_audio.tts.models.voxcpm2.voxcpm2 import Model

OUT = Path("../turbospark/crates/audio/testdata/tts_voxcpm2")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

MOCK_IDS = {
    "hello": 4,
    "world": 6,
    "ptext": 8,
    "rtext": 10,
}


class MockTokenizer:
    """Deterministic word-id tokenizer."""

    @staticmethod
    def _fallback(word: str) -> int:
        """Stable id for words the table does not know (the reference
        concatenates prompt_text/ref_text with the text before
        encoding, which merges boundary words)."""
        return 12 + 2 * (sum(ord(ch) for ch in word) % 24)

    def tokenize(self, text: str) -> list:
        # Word-level tokens; the multi-char Chinese split in `_tokenize`
        # is a no-op on these.
        return text.split(" ")

    def convert_tokens_to_ids(self, tokens: list) -> list:
        return [
            MOCK_IDS.get(token, self._fallback(token))
            for token in tokens
            if token
        ]

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        del add_special_tokens
        return [
            MOCK_IDS.get(word, self._fallback(word))
            for word in text.split(" ")
            if word
        ]


LM_BASE = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 128,
    "rope_theta": 10000.0,
    "rope_scaling_type": "longrope",
    "rope_long_factor": [1.1, 1.3, 1.5, 1.7],
    "rope_short_factor": [1.0, 1.2, 1.4, 1.6],
    "scale_emb": 12,
    "dim_model_base": 256,
    "scale_depth": 1.4,
    "original_max_position_embeddings": 4096,
    "max_position_embeddings": 4096,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "use_mup": True,
    "kv_channels": None,
    "no_rope": False,
}

ENCODER = {
    "hidden_dim": 32,
    "ffn_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "kv_channels": None,
}

DIT = {
    "hidden_dim": 32,
    "ffn_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "kv_channels": None,
    "dit_mean_mode": False,
    "cfm_config": {
        "sigma_min": 1e-6,
        "solver": "euler",
        "t_scheduler": "log-norm",
        "inference_cfg_rate": 2.0,
    },
}

AUDIO_VAE = {
    "encoder_dim": 16,
    "encoder_rates": [2],
    "latent_dim": 8,
    "decoder_dim": 32,
    "decoder_rates": [4],
    "depthwise": True,
    "sample_rate": 16000,
    "out_sample_rate": 48000,
    "use_noise_block": False,
    "sr_bin_boundaries": [20000, 30000, 40000],
    "cond_type": "scale_bias",
    "cond_dim": 128,
    "cond_out_layer": False,
}

MODEL_TOP = {
    "patch_size": 2,
    "feat_dim": 8,
    "scalar_quantization_latent_dim": 16,
    "scalar_quantization_scale": 9,
    "residual_lm_num_layers": 2,
    "residual_lm_no_rope": False,
    "max_length": 8192,
}

INFERENCE_TIMESTEPS = 4
CFG_VALUE = 2.0
MIN_TOKENS = 2
WARMUP = 2


def make_config() -> dict:
    return {
        "lm_config": dict(LM_BASE),
        "encoder_config": dict(ENCODER),
        "dit_config": json.loads(json.dumps(DIT)),
        "audio_vae_config": dict(AUDIO_VAE),
        **dict(MODEL_TOP),
    }


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


def seeded_model(config: dict, seed: int, stop_bias: float) -> Model:
    """Deterministic weights: small uniform everywhere, norms biased
    up, the stop predictor biased so class 1 dominates (or both rows
    zeroed for a never-stop run)."""
    mx.random.seed(seed)
    model = Model(ModelArgs.from_dict(json.loads(json.dumps(config))))

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith(
            ("norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight")
        ):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    hidden = config["lm_config"]["hidden_size"]
    flat["stop_proj.bias"] = mx.full((hidden,), 8.0)
    stop_weight = mx.zeros((2, hidden))
    if stop_bias > 0.0:
        stop_weight = mx.concatenate(
            [mx.zeros((1, hidden)), mx.full((1, hidden), stop_bias)], axis=0
        )
    flat["stop_head.weight"] = stop_weight
    model.update(tree_unflatten(flat))
    model.tokenizer = MockTokenizer()
    return model


def reference_audio(samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(samples, dtype=np.float32) / AUDIO_VAE["sample_rate"]
    return (0.2 * np.sin(2 * np.pi * 300.0 * t) + 0.05 * rng.standard_normal(samples)).astype(
        np.float32
    )


def save_case(name: str, model: Model, config: dict) -> None:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(case_dir / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def reference_generate_body(
    model: Model,
    text: str,
    max_tokens: int,
    min_tokens: int,
    warmup: int,
    ref_audio,
    prompt_audio,
    prompt_text,
    capture: bool,
):
    """Verbatim replay of the reference `Model.generate` body with
    per-step capture. Returns (sink, pred_feat_seq, waveform, prompt)."""
    scale_emb = (
        model.args.lm_config.scale_emb if model.args.lm_config.use_mup else 1.0
    )
    latent_dim = model.audio_vae.latent_dim

    has_ref = ref_audio is not None
    has_prompt = prompt_audio is not None
    sink = []

    if has_ref:
        text_ids = model.tokenizer.encode(text)
        text_token = mx.array(text_ids + [model.audio_start_token], dtype=mx.int32)
        text_length = text_token.shape[0]
        ref_feat = model._encode_wav(mx.array(ref_audio), padding_mode="right")
        ref_len = ref_feat.shape[0]
        import scipy.signal as _sp
        _num = int(len(ref_audio) * model.args.audio_vae_config.sample_rate
                   / model.args.audio_vae_config.out_sample_rate)
        ref_resampled = _sp.resample(ref_audio, _num)
        latent_dim_vae = model.audio_vae.latent_dim
        z1 = mx.zeros((1, model.patch_size, latent_dim_vae))
        ref_tokens = mx.concatenate(
            [
                mx.array([model.ref_audio_start_token], dtype=mx.int32),
                mx.zeros(ref_len, dtype=mx.int32),
                mx.array([model.ref_audio_end_token], dtype=mx.int32),
            ]
        )
        ref_feats = mx.concatenate([z1, ref_feat, z1], axis=0)
        ref_t_mask = mx.concatenate(
            [
                mx.array([1], dtype=mx.float32),
                mx.zeros(ref_len, dtype=mx.float32),
                mx.array([1], dtype=mx.float32),
            ]
        )
        ref_a_mask = mx.concatenate(
            [
                mx.array([0], dtype=mx.float32),
                mx.ones(ref_len, dtype=mx.float32),
                mx.array([0], dtype=mx.float32),
            ]
        )
        text_pad_feat = mx.zeros((text_length, model.patch_size, latent_dim_vae))
        text_token_full = mx.concatenate([ref_tokens, text_token])
        audio_feat = mx.concatenate([ref_feats, text_pad_feat], axis=0)
        text_mask = mx.concatenate([ref_t_mask, mx.ones(text_length, dtype=mx.float32)])
        audio_mask = mx.concatenate([ref_a_mask, mx.zeros(text_length, dtype=mx.float32)])
    elif has_prompt:
        text_ids = model.tokenizer.encode(prompt_text + text)
        text_token = mx.array(text_ids + [model.audio_start_token], dtype=mx.int32)
        text_length = text_token.shape[0]
        prompt_feat = model._encode_wav(mx.array(prompt_audio), padding_mode="left")
        prompt_len = prompt_feat.shape[0]
        text_pad_feat = mx.zeros((text_length, model.patch_size, latent_dim))
        prompt_pad_token = mx.zeros(prompt_len, dtype=mx.int32)
        text_token_full = mx.concatenate([text_token, prompt_pad_token])
        audio_feat = mx.concatenate([text_pad_feat, prompt_feat], axis=0)
        text_mask = mx.concatenate(
            [mx.ones(text_length, dtype=mx.float32), mx.zeros(prompt_len, dtype=mx.float32)]
        )
        audio_mask = mx.concatenate(
            [mx.zeros(text_length, dtype=mx.float32), mx.ones(prompt_len, dtype=mx.float32)]
        )
    else:
        text_ids = model.tokenizer.encode(text)
        text_token = mx.array(text_ids + [model.audio_start_token], dtype=mx.int32)
        text_length = text_token.shape[0]
        text_token_full = text_token
        audio_feat = mx.zeros((text_length, model.patch_size, latent_dim))
        text_mask = mx.ones(text_length, dtype=mx.float32)
        audio_mask = mx.zeros(text_length, dtype=mx.float32)

    token_count = len(text_ids)

    text_token_b = text_token_full[None, :]
    audio_feat_b = audio_feat[None, :, :, :]
    text_mask_b = text_mask[None, :]
    audio_mask_b = audio_mask[None, :]

    feat_embed = model.feat_encoder(audio_feat_b)
    feat_embed = model.enc_to_lm_proj(feat_embed)
    text_embed = model.base_lm.embed_tokens(text_token_b) * scale_emb
    combined_embed = (
        text_mask_b[:, :, None] * text_embed + audio_mask_b[:, :, None] * feat_embed
    )
    prefix_feat_cond = audio_feat_b[:, -1, :, :]

    prompt = {
        "input_ids": ints(text_token_full),
        "audio_mask": floats(audio_mask),
        "combined_embed": floats(combined_embed),
        "prefix_feat_cond": floats(prefix_feat_cond),
        "token_count": token_count,
        "feat_embed": floats(feat_embed),
    }
    if has_ref:
        prompt["ref_resampled"] = floats(ref_resampled)
        prompt["ref_patches"] = floats(ref_feat)

    enc_outputs, lm_cache = model.base_lm(combined_embed)
    enc_outputs = (
        model.fsq_layer(enc_outputs) * audio_mask_b[:, :, None]
        + enc_outputs * text_mask_b[:, :, None]
    )
    lm_hidden = enc_outputs[:, -1, :]
    residual_input = model.fusion_concat_proj(
        mx.concatenate([enc_outputs, audio_mask_b[:, :, None] * feat_embed], axis=-1)
    )
    residual_outputs, res_cache = model.residual_lm(residual_input)
    residual_hidden = residual_outputs[:, -1, :]

    prompt["lm_hidden"] = floats(lm_hidden)
    prompt["residual_hidden"] = floats(residual_hidden)

    has_continuation = audio_mask[-1].item() == 1.0
    if has_continuation:
        mask_np = np.array(audio_mask)
        audio_indices = np.nonzero(mask_np > 0)[0]
        context_len = min(4 - 1, len(audio_indices))
        last_indices = audio_indices[-context_len:]
        pred_feat_seq = [
            audio_feat_b[:, int(idx), :, :][:, None, :, :] for idx in last_indices
        ]
    else:
        pred_feat_seq = []

    warmup = 0 if has_continuation else warmup

    for i in range(max_tokens + warmup):
        dit_h1 = model.lm_to_dit_proj(lm_hidden)
        dit_h2 = model.res_to_dit_proj(residual_hidden)
        dit_h = mx.concatenate([dit_h1, dit_h2], axis=-1)

        cond_in = prefix_feat_cond.transpose(0, 2, 1)
        pred_feat = model.feat_decoder.sample(
            mu=dit_h,
            n_timesteps=INFERENCE_TIMESTEPS,
            patch_size=model.patch_size,
            cond=cond_in,
            cfg_value=CFG_VALUE,
        )
        t_span = mx.linspace(1, 0, INFERENCE_TIMESTEPS + 1)
        t_span = t_span + 1.0 * (mx.cos(math.pi / 2 * t_span) - 1 + t_span)
        sink_tspan = floats(t_span)

        pred_feat = pred_feat.transpose(0, 2, 1)
        if i >= warmup:
            pred_feat_seq.append(pred_feat[:, None, :, :])

        curr_embed = model.feat_encoder(pred_feat[:, None, :, :])
        curr_embed = model.enc_to_lm_proj(curr_embed)

        stop_logits = model.stop_head(nn.silu(model.stop_proj(lm_hidden)))
        stop_flag = mx.argmax(stop_logits, axis=-1).item()
        real_steps = i - warmup
        if capture:
            sink.append(
                {
                    "index": i,
                    "latent": floats(pred_feat),
                    "collected": i >= warmup,
                    "mu": floats(dit_h),
                    "curr_embed": floats(curr_embed),
                    "stop_logits": floats(stop_logits),
                    "stop_flag": int(stop_flag),
                    "lm_hidden_in": floats(lm_hidden),
                    "residual_hidden_in": floats(residual_hidden),
                    "t_span": sink_tspan,
                }
            )
        if real_steps > min_tokens and stop_flag == 1:
            break

        new_lm_out, lm_cache = model.base_lm(inputs_embeds=curr_embed, cache=lm_cache)
        lm_hidden = new_lm_out[:, -1, :]
        lm_hidden = model.fsq_layer(lm_hidden)

        curr_residual_input = model.fusion_concat_proj(
            mx.concatenate([lm_hidden[:, None, :], curr_embed], axis=-1)
        )
        new_res_out, res_cache = model.residual_lm(
            inputs_embeds=curr_residual_input, cache=res_cache
        )
        residual_hidden = new_res_out[:, -1, :]

        prefix_feat_cond = pred_feat

    all_feats = mx.concatenate(pred_feat_seq, axis=1)
    B = all_feats.shape[0]
    all_feats_flat = all_feats.reshape(B, -1, model.feat_dim)
    waveform = model.audio_vae.decode(all_feats_flat).flatten()
    if has_continuation:
        decode_patch_len = model.patch_size * model.audio_vae.decode_chunk_size
        trim_audio_samples = decode_patch_len * (4 - 1)
        if trim_audio_samples < len(waveform):
            waveform = waveform[trim_audio_samples:]
    return sink, all_feats_flat, waveform, prompt


def run_case(
    name: str,
    model: Model,
    config: dict,
    text: str,
    max_tokens: int,
    stop_bias: float,
    ref_audio=None,
    prompt_audio=None,
    prompt_text=None,
    warmup: int = 0,
    expect_steps: str = "",
) -> None:
    save_case(name, model, config)

    # ---- run 1: verbatim body replay with capture + draws ----------
    with DrawRecorder() as recorder:
        mx.random.seed(101)
        sink, decode_input, waveform, prompt = reference_generate_body(
            model, text, max_tokens, MIN_TOKENS, warmup, ref_audio, prompt_audio,
            prompt_text, True
        )
        noise_draws = recorder.flush()

    # ---- run 2: the true reference generate must agree exactly -----
    with DrawRecorder() as recorder:
        mx.random.seed(101)
        kwargs = dict(
            text=text,
            max_tokens=max_tokens,
            min_tokens=MIN_TOKENS,
            inference_timesteps=INFERENCE_TIMESTEPS,
            cfg_value=CFG_VALUE,
            warmup_patches=warmup,
        )
        if ref_audio is not None:
            kwargs.update(ref_audio=mx.array(ref_audio))
        if prompt_audio is not None:
            kwargs.update(prompt_text=prompt_text, prompt_audio=mx.array(prompt_audio))
        result = list(model.generate(**kwargs))[0]
        generate_draws = recorder.flush()
    assert generate_draws == noise_draws, "generate draw replay diverged"
    assert np.array_equal(np.array(result.audio), np.array(waveform)), (
        "the replay body and generate disagree"
    )

    # ---- run 3: fresh replay of run 1 (reproducibility pin) --------
    with DrawRecorder() as recorder:
        mx.random.seed(101)
        sink2, decode_input2, waveform2, prompt2 = reference_generate_body(
            model, text, max_tokens, MIN_TOKENS, warmup, ref_audio, prompt_audio,
            prompt_text, True
        )
        replay_draws = recorder.flush()
    assert replay_draws == noise_draws, "second replay diverged"
    assert np.array_equal(np.array(waveform2), np.array(waveform)), "replay waveform"
    for a, b in zip(sink, sink2):
        for key in a:
            if key == "index":
                continue
            assert a[key] == b[key], f"replay step {a['index']} field {key} diverged"
    assert np.array_equal(np.array(decode_input2), np.array(decode_input)), (
        "replay decode input"
    )

    collected = [step for step in sink if step["collected"]]
    if stop_bias > 0.0:
        assert sink[-1]["stop_flag"] == 1 and len(collected) == MIN_TOKENS + 2, (
            f"{name}: expected the stop head at min_tokens + 2, got {len(collected)}"
        )
    else:
        assert len(collected) == max_tokens, f"{name}: expected {max_tokens} steps"

    traces = {
        "mode": name,
        "generate": {
            "text": text,
            "max_tokens": max_tokens,
            "min_tokens": MIN_TOKENS,
            "warmup_patches": warmup,
            "inference_timesteps": INFERENCE_TIMESTEPS,
            "cfg_value": CFG_VALUE,
            "streaming_prefix_len": 4,
        },
        "prompt": prompt,
        "noise_draws": [
            {"shape": shape, "values": values} for shape, values in noise_draws
        ],
        "steps": sink,
        "waveform": floats(waveform),
        "sample_rate": int(model.sample_rate),
        # The full decode input: the continuation prefix patches (if
        # any) plus every collected generated patch.
        "latents": floats(decode_input[0]),
    }
    if ref_audio is not None:
        traces["ref"] = {"ref_audio": floats(ref_audio)}
    if prompt_audio is not None:
        traces["continuation"] = {"prompt_audio": floats(prompt_audio)}
    (OUT / name / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(
        f"{name}: {len(sink)} loop steps, {len(collected)} collected, "
        f"{len(waveform)} samples, stopped "
        f"{'stop_head' if stop_bias > 0.0 else 'max_steps'} {expect_steps}"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = make_config()

    # Zero-shot, stop head fires.
    model = seeded_model(config, 11, 0.5)
    run_case("zero_shot", model, config, text="hello world", max_tokens=10, stop_bias=0.5)

    # Zero-shot, stop silenced, runs to max_tokens.
    model = seeded_model(config, 11, 0.0)
    run_case("length", model, config, text="hello world", max_tokens=8, stop_bias=0.0)

    # Reference cloning: [103 | patches | 104] prefix, stop head fires.
    model = seeded_model(config, 11, 0.5)
    run_case(
        "ref_clone",
        model,
        config,
        text="rtext hello",
        max_tokens=10,
        stop_bias=0.5,
        ref_audio=reference_audio(20, 17),
    )

    # Continuation: prompt audio suffix (left pad), warmup=0, decode
    # front trim.
    model = seeded_model(config, 11, 0.0)
    run_case(
        "continuation",
        model,
        config,
        text="hello",
        max_tokens=8,
        stop_bias=0.0,
        prompt_audio=reference_audio(20, 23),
        prompt_text="ptext ",
    )

    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "generator": "tools/gen_voxcpm2_fixtures.py",
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
