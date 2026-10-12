#!/usr/bin/env python3
"""Regenerate the voxcpm (VoxCPM 1.5) family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/voxcpm, mlx 0.32.3). The reference modules run in
full: the MiniCPM backbone with the LongRoPE factor tables (base LM with
the embedding table, residual LM, patch encoder, and DiT decoder as
vocab-free variants), the VoxCPMLocEnc CLS patch encoder, the
ScalarQuantizationLayer, the VoxCPMLocDiT denoiser inside the
UnifiedCFM Euler solver (CFG zero-star, sway sampling), the two-linear
stop predictor, the causal-conv AudioVAE (depthwise, snake, weight
norm), and the reference `Model.generate` loop, with:

- a deterministic offline mock tokenizer (`encode` word ids) standing
  in for the checkpoint tokenizer, injected as `model.tokenizer`;
- seeded tiny stacks: 2-layer backbones at hidden 32, 4 query / 2 kv
  heads, a 2+2-block AudioVAE (encoder rates [2, 4], decoder rates
  [4, 2], hop 8) at 24 kHz;
- every `mx.random.normal` draw recorded (one flow draw per generate
  step) and replay-verified across two full runs of each case.

Two stop-head regimes: `generate`/`voice_clone` stop after step 5 (the
reference `i > 5` guard) via a large positive `stop_proj` bias so the
class-1 logit dominates; `length`/`longrope` have both stop weights
zeroed and run to `max_tokens`. The `longrope` case crosses
`original_max_position_embeddings` = 16, so the LongRoPE factor table
flips from `short_factor` to `long_factor` inside the same run.

Per-step traces come from a verbatim replay of the reference generate
body (the loop has no hooks); the replay's waveform and draw sequence
are asserted identical to a fresh `model.generate` call, and the
replay is itself re-run a second time to pin reproducibility.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_voxcpm_fixtures.py

Outputs into crates/audio/testdata/tts_voxcpm/:

- generate/ (zero-shot, stop head fires), length/ (zero-shot, no stop),
  voice_clone/ (reference audio + ref text), longrope/ (long-factor
  rope flip, no stop)
    Each: config.json + model.safetensors + traces.json.
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

import mlx_audio.tts.models.voxcpm.audio_vae as audio_vae_mod
import mlx_audio.tts.models.voxcpm.voxcpm as voxcpm_mod
from mlx_audio.tts.models.voxcpm.config import ModelArgs
from mlx_audio.tts.models.voxcpm.voxcpm import Model

OUT = Path("../turbospark/crates/audio/testdata/tts_voxcpm")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

AUDIO_START_TOKEN = 101

MOCK_IDS = {
    "hello": 4,
    "world": 6,
    "ref": 8,
    "clip": 10,
}
MOCK_IDS.update({f"w{i}": 12 + 2 * i for i in range(24)})


class MockTokenizer:
    """Deterministic word-id tokenizer."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        del add_special_tokens
        return [MOCK_IDS[word] for word in text.split(" ")]


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
    "original_max_position_embeddings": 512,
    "max_position_embeddings": 4096,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "use_mup": False,
}

LM_LONGROPE = dict(LM_BASE)
LM_LONGROPE.update(
    original_max_position_embeddings=16,
    max_position_embeddings=64,
)

ENCODER = {
    "hidden_dim": 32,
    "ffn_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
}

DIT = {
    "hidden_dim": 32,
    "ffn_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "cfm_config": {
        "sigma_min": 1e-6,
        "solver": "euler",
        "t_scheduler": "log-norm",
        "inference_cfg_rate": 2.0,
    },
}

AUDIO_VAE = {
    "encoder_dim": 16,
    "encoder_rates": [2, 4],
    "latent_dim": 8,
    "decoder_dim": 32,
    "decoder_rates": [4, 2],
    "sample_rate": 24000,
}

MODEL_TOP = {
    "patch_size": 2,
    "feat_dim": 8,
    "scalar_quantization_latent_dim": 16,
    "scalar_quantization_scale": 9,
    "residual_lm_num_layers": 2,
    "max_length": 8192,
}

INFERENCE_TIMESTEPS = 4
CFG_VALUE = 2.0


def make_config(lm: dict) -> dict:
    return {
        "lm_config": dict(lm),
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
    """Deterministic weights: small uniform everywhere, norms biased up,
    the stop predictor biased so class 1 dominates (or the class weights
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
    # The stop path: stop_proj bias +8 makes every silu output positive
    # and large, so a positive class-1 stop_head row produces a huge
    # margin over the zero class-0 row. The never-stop cases zero both
    # rows so argmax is always class 0.
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


def reference_audio() -> np.ndarray:
    rng = np.random.default_rng(53)
    t = np.arange(32, dtype=np.float32) / AUDIO_VAE["sample_rate"]
    return (0.2 * np.sin(2 * np.pi * 300.0 * t) + 0.05 * rng.standard_normal(32)).astype(
        np.float32
    )


def save_case(name: str, model: Model, config: dict) -> None:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(case_dir / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def encode_prompt_audio(model: Model, audio: np.ndarray | mx.array) -> mx.array:
    """The reference `_encode_prompt_audio` on an already-loaded waveform."""
    return model._encode_prompt_audio(mx.asarray(audio, dtype=mx.float32))


def reference_generate_body(
    model: Model,
    text: str,
    max_tokens: int,
    ref_audio: np.ndarray | None,
    ref_text: str | None,
    capture: bool,
):
    """Verbatim replay of the reference `Model.generate` body with
    per-step capture. Returns (sink, all_feats, waveform)."""
    scale_emb = (
        model.args.lm_config.scale_emb if not model.args.lm_config.use_mup else 1.0
    )

    sink = []
    text_mask = None
    audio_mask = None
    feat_embed = None

    if ref_audio is not None and ref_text is not None:
        combined_text = ref_text + text
        input_ids = model.tokenizer.encode(combined_text)
        input_ids = mx.array(input_ids)
        input_ids = mx.concatenate([input_ids, mx.array([AUDIO_START_TOKEN])])
        text_length = len(input_ids)
        audio_feat = encode_prompt_audio(model, ref_audio)
        audio_length = audio_feat.shape[0]
        text_pad_token = mx.zeros(audio_length, dtype=mx.int32)
        text_token = mx.concatenate([input_ids, text_pad_token])
        audio_pad_feat = mx.zeros(
            (text_length, model.patch_size, model.feat_dim), dtype=mx.float32
        )
        audio_feat = mx.concatenate([audio_pad_feat, audio_feat], axis=0)
        text_mask = mx.concatenate(
            [mx.ones(text_length, dtype=mx.float32), mx.zeros(audio_length)]
        )
        audio_mask = mx.concatenate(
            [mx.zeros(text_length), mx.ones(audio_length, dtype=mx.float32)]
        )
        text_token = text_token[None, :]
        audio_feat = audio_feat[None, :, :, :]
        text_mask = text_mask[None, :]
        audio_mask = audio_mask[None, :]
        feat_embed = model.feat_encoder(audio_feat)
        feat_embed = model.enc_to_lm_proj(feat_embed)
        text_embed = model.base_lm.embed_tokens(text_token) * scale_emb
        combined_embed = (
            text_mask[:, :, None] * text_embed + audio_mask[:, :, None] * feat_embed
        )
        prefix_feat_cond = audio_feat[:, -1, :, :]
        token_count = len(input_ids)
    else:
        input_ids = model.tokenizer.encode(text)
        input_ids = mx.array(input_ids)
        token_count = len(input_ids)
        start_token = mx.array([AUDIO_START_TOKEN])
        input_ids = mx.concatenate([input_ids, start_token])
        combined_embed = model.base_lm.embed_tokens(input_ids[None, :]) * scale_emb
        prefix_feat_cond = mx.zeros((1, model.patch_size, model.feat_dim))

    if text_mask is None:
        audio_feat = mx.zeros(
            (1, combined_embed.shape[1], model.patch_size, model.feat_dim)
        )
        text_token_full = input_ids[None, :]
    else:
        text_token_full = None
    prompt = {
        "text_token": None if text_token_full is None else ints(text_token_full),
        "text_mask": None if text_mask is None else floats(text_mask),
        "audio_mask": None if audio_mask is None else floats(audio_mask),
        "audio_feat": floats(audio_feat),
        "combined_embed": floats(combined_embed),
        "prefix_feat_cond": floats(prefix_feat_cond),
        "token_count": token_count,
    }

    enc_outputs, lm_cache = model.base_lm(combined_embed)
    if text_mask is not None and audio_mask is not None:
        enc_outputs = (
            model.fsq_layer(enc_outputs) * audio_mask[:, :, None]
            + enc_outputs * text_mask[:, :, None]
        )
    lm_hidden = enc_outputs[:, -1, :]
    if text_mask is None:
        lm_hidden = model.fsq_layer(lm_hidden)
    if text_mask is not None and audio_mask is not None:
        residual_input = enc_outputs + audio_mask[:, :, None] * feat_embed
    else:
        residual_input = enc_outputs
    residual_outputs, res_cache = model.residual_lm(residual_input)
    residual_hidden = residual_outputs[:, -1, :]

    prompt["enc_outputs"] = floats(enc_outputs)
    prompt["residual_input"] = floats(residual_input)
    prompt["lm_hidden"] = floats(lm_hidden)
    prompt["residual_hidden"] = floats(residual_hidden)

    pred_feat_seq = []
    euler_tspans = []
    for i in range(max_tokens):
        dit_h1 = model.lm_to_dit_proj(lm_hidden)
        dit_h2 = model.res_to_dit_proj(residual_hidden)
        dit_h = dit_h1 + dit_h2

        cond_in = prefix_feat_cond.transpose(0, 2, 1)

        pred_feat = model.feat_decoder.sample(
            mu=dit_h,
            n_timesteps=INFERENCE_TIMESTEPS,
            patch_size=model.patch_size,
            cond=cond_in,
            cfg_value=CFG_VALUE,
        )
        # Record the sway-adjusted t_span the solver consumed.
        t_span = mx.linspace(1, 0, INFERENCE_TIMESTEPS + 1)
        t_span = t_span + 1.0 * (mx.cos(math.pi / 2 * t_span) - 1 + t_span)
        euler_tspans.append(floats(t_span))

        pred_feat = pred_feat.transpose(0, 2, 1)
        pred_feat_seq.append(pred_feat)

        curr_embed = model.feat_encoder(pred_feat[:, None, :, :])
        curr_embed = model.enc_to_lm_proj(curr_embed)

        stop_logits = model.stop_head(nn.silu(model.stop_proj(lm_hidden)))
        stop_flag = mx.argmax(stop_logits, axis=-1).item()
        if capture:
            sink.append(
                {
                    "index": i,
                    "latent": floats(pred_feat),
                    "mu": floats(dit_h),
                    "cond": floats(cond_in),
                    "curr_embed": floats(curr_embed),
                    "stop_logits": floats(stop_logits),
                    "stop_flag": int(stop_flag),
                    "lm_hidden_in": floats(lm_hidden),
                    "residual_hidden_in": floats(residual_hidden),
                }
            )
        if i > 5 and stop_flag == 1:
            break

        new_lm_out, lm_cache = model.base_lm(
            inputs_embeds=curr_embed, cache=lm_cache
        )
        lm_hidden_next = new_lm_out[:, -1, :]
        lm_hidden_next = model.fsq_layer(lm_hidden_next)

        res_in = lm_hidden_next[:, None, :] + curr_embed
        new_res_out, res_cache = model.residual_lm(
            inputs_embeds=res_in, cache=res_cache
        )
        residual_hidden = new_res_out[:, -1, :]

        lm_hidden = lm_hidden_next
        prefix_feat_cond = pred_feat

    all_feats = mx.concatenate(pred_feat_seq, axis=1)
    waveform = model.audio_vae.decode(all_feats).flatten()
    return sink, all_feats, waveform, prompt, euler_tspans


def run_case(
    name: str,
    model: Model,
    config: dict,
    max_tokens: int,
    stop_bias: float,
    text: str = "hello world",
    ref_audio: np.ndarray | None = None,
    ref_text: str | None = None,
) -> None:
    save_case(name, model, config)

    # ---- capture the AudioVAE decode stages of the final decode ----
    vae_decode_trace = []
    original_dec_call = model.audio_vae.decoder.__call__
    block_records = []
    original_dec_block = audio_vae_mod.CausalDecoderBlock.__call__

    def recorded_dec_block(self, x):
        out = original_dec_block(self, x)
        block_records.append(floats(out))
        return out

    def recorded_decoder_call(self, x):
        block_records.clear()
        out = original_dec_call(self, x)
        vae_decode_trace.append(
            {"z": floats(x), "blocks": list(block_records), "pre_tanh": floats(out)}
        )
        return out

    # ---- capture the first DiT estimator calls --------------------
    dit_calls = []
    original_dit_call = model.feat_decoder.estimator.__call__

    def recorded_dit_call(self, x, mu, t, cond, dt):
        out = original_dit_call(self, x, mu, t, cond, dt)
        if len(dit_calls) < 6:
            dit_calls.append(
                {
                    "x": floats(x),
                    "mu": floats(mu),
                    "t": floats(t),
                    "dt": floats(dt),
                    "cond": floats(cond),
                    "out": floats(out),
                }
            )
        return out

    model.feat_decoder.estimator.__call__ = recorded_dit_call

    # ---- run 1: verbatim body replay with capture + draws ----------
    audio_vae_mod.CausalDecoderBlock.__call__ = recorded_dec_block
    model.audio_vae.decoder.__call__ = recorded_decoder_call
    with DrawRecorder() as recorder:
        mx.random.seed(101)
        sink, all_feats, waveform, prompt, euler_tspans = reference_generate_body(
            model, text, max_tokens, ref_audio, ref_text, capture=True
        )
        noise_draws = recorder.flush()
    model.audio_vae.decoder.__call__ = original_dec_call
    audio_vae_mod.CausalDecoderBlock.__call__ = original_dec_block

    # ---- run 2: the true reference generate must agree exactly -----
    with DrawRecorder() as recorder:
        mx.random.seed(101)
        kwargs = dict(
            text=text,
            max_tokens=max_tokens,
            inference_timesteps=INFERENCE_TIMESTEPS,
            cfg_value=CFG_VALUE,
        )
        if ref_audio is not None:
            kwargs.update(ref_text=ref_text, ref_audio=mx.array(ref_audio))
        result = list(model.generate(**kwargs))[0]
        generate_draws = recorder.flush()
    assert generate_draws == noise_draws, "generate draw replay diverged"
    assert np.array_equal(np.array(result.audio), np.array(waveform)), (
        "the replay body and generate disagree"
    )

    # ---- run 3: fresh replay of run 1 (reproducibility pin) --------
    with DrawRecorder() as recorder:
        mx.random.seed(101)
        sink2, _, waveform2, _, _ = reference_generate_body(
            model, text, max_tokens, ref_audio, ref_text, capture=True
        )
        replay_draws = recorder.flush()
    assert replay_draws == noise_draws, "second replay diverged"
    assert np.array_equal(np.array(waveform2), np.array(waveform)), "replay waveform"
    for a, b in zip(sink, sink2):
        for key in a:
            if key == "index":
                continue
            assert a[key] == b[key], f"replay step {a['index']} field {key} diverged"

    stopped_early = any(step["stop_flag"] == 1 for step in sink) and len(sink) < max_tokens
    if stop_bias > 0.0:
        assert stopped_early and sink[-1]["index"] == 6, (
            f"{name}: expected the stop head to fire at step 6, got {len(sink)}"
        )
    else:
        assert len(sink) == max_tokens, f"{name}: expected {max_tokens} steps"

    traces = {
        "mode": "voice_clone" if ref_audio is not None else "zero_shot",
        "generate": {
            "text": text,
            "ref_text": ref_text,
            "max_tokens": max_tokens,
            "inference_timesteps": INFERENCE_TIMESTEPS,
            "cfg_value": CFG_VALUE,
            "audio_start_token": AUDIO_START_TOKEN,
        },
        "prompt": prompt,
        "noise_draws": [
            {"shape": shape, "values": values} for shape, values in noise_draws
        ],
        "steps": sink,
        "euler_tspans": euler_tspans,
        "dit_calls": dit_calls,
        "vae_decode": vae_decode_trace,
        "waveform": floats(waveform),
        "sample_rate": int(model.sample_rate),
    }
    if ref_audio is not None:
        traces["voice_clone"] = {
            "ref_audio": floats(ref_audio),
            "prompt_audio_feat": floats(encode_prompt_audio(model, ref_audio)),
        }
    (OUT / name / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(
        f"{name}: mode {traces['mode']}, {len(sink)} steps, "
        f"{len(waveform)} samples, stopped "
        f"{'stop_head' if stopped_early else 'max_steps'}"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Zero-shot, stop head fires after step 5 (7 patches).
    config = make_config(LM_BASE)
    model = seeded_model(config, 7, 0.5)
    run_case("generate", model, config, max_tokens=10, stop_bias=0.5)

    # Zero-shot, stop silenced, runs to max_tokens.
    model = seeded_model(config, 7, 0.0)
    run_case("length", model, config, max_tokens=8, stop_bias=0.0)

    # Voice clone: reference audio + ref text, stop head fires.
    model = seeded_model(config, 7, 0.5)
    run_case(
        "voice_clone",
        model,
        config,
        max_tokens=10,
        stop_bias=0.5,
        ref_audio=reference_audio(),
        ref_text="ref clip",
    )

    # LongRoPE flip: the 21-token prompt crosses
    # original_max_position_embeddings = 16, so the factor table
    # switches from short to long inside the same run.
    long_config = make_config(LM_LONGROPE)
    model = seeded_model(long_config, 7, 0.0)
    long_text = " ".join(f"w{i}" for i in range(20))
    run_case(
        "longrope",
        model,
        long_config,
        max_tokens=8,
        stop_bias=0.0,
        text=long_text,
    )

    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "generator": "tools/gen_voxcpm_fixtures.py",
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
