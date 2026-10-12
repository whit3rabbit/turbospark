#!/usr/bin/env python3
"""Regenerate the DramaBox family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/dramabox, mlx 0.32.3). The true reference modules
run in full: the text conditioner (FeatureExtractorV2 + register
reorder + the rope'd 1D connector), the audio-only LTX transformer
(AdaLN-single, gated split-rope attention blocks, prompt-AdaLN cross
attention), the multi-modal guider with the std rescale, the LTX2
sigma schedule with the Euler step, and the reference
`guided_euler_loop` driving `X0Model`, with:

- engineered Gemma hidden states standing in for the separate
  downloadable `mlx-community/gemma-3-12b-it-8bit` text encoder (the
  conditioner consumes the stacked hidden states directly, exactly as
  the reference `DramaboxTextConditioner` would);
- the f32 dtype the Rust port computes in (`create_initial_state` is
  driven directly with `dtype=mx.float32` instead of the bf16 the
  reference `generate` wrapper hardcodes; every module on this path is
  dtype-generic);
- seeded tiny stacks: a 2-block DiT (heads 4, head_dim 8, inner 32,
  cross dim 32, patch in 16 = 4 channels x 4 mels), a 1-block
  connector (heads 2, head_dim 4, 4 learnable registers over an
  8-token grid), 4 flow steps;
- every `mx.random.normal` draw recorded in reference draw order (one
  initial latent noise per run; the scheduler is deterministic) and
  replayed by the Rust test.

Two cases: `generate` (cfg 2.0, stg 1.0 on block 1, auto rescale, so
the cond, uncond-text, and STG-perturbed passes all run) and `plain`
(cfg 1.0, stg 0.0, the single-pass path). The AudioVAE decode and the
vocoder are NOT run: the Rust port ends at the mel latents by design
(see the family README for that refusal). A Python-side replay check
rebuilds each case from the saved safetensors plus the recorded draw
stream and asserts the latents match before anything is frozen.

Run from inside the mlx-audio checkout root so `mlx_audio` imports
from the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_dramabox_fixtures.py

Outputs into crates/audio/testdata/tts_dramabox/<case>/ (config.json +
model.safetensors + traces.json, digests in manifest.json). The script
is deterministic: fixed seeds, no network access.
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

from mlx_audio.tts.models.dramabox.config import AudioConfig, ModelConfig
from mlx_audio.tts.models.dramabox.guidance import MultiModalGuiderParams, auto_rescale_for_cfg
from mlx_audio.tts.models.dramabox.latent import AudioLatentTools, AudioPatchifier
from mlx_audio.tts.models.dramabox.sampling import (
    guided_euler_loop,
    patch_long_clip_silence_prior,
    target_shape_for_duration,
)
from mlx_audio.tts.models.dramabox.text_conditioning import DramaboxTextConditioner
from mlx_audio.tts.models.dramabox.transformer import AudioOnlyLTXModel, Modality, X0Model

OUT = Path("../turbospark/crates/audio/testdata/tts_dramabox")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"
REPLAY_GATE = 2e-5

TOKENS = 8
CONTEXT_TOKENS = 8

TINY_CONFIG = {
    "model_type": "dramabox-tts",
    "sample_rate": 8000,
    "text_encoder": "tiny-gemma",
    "text_encoder_hidden_size": 8,
    "num_gemma_layers": 3,
    "transformer": {
        "num_layers": 2,
        "audio_num_attention_heads": 4,
        "audio_attention_head_dim": 8,
        "audio_in_channels": 16,
        "audio_out_channels": 16,
        "audio_cross_attention_dim": 32,
        "norm_eps": 1e-6,
        "attention_type": "default",
        "positional_embedding_theta": 10000.0,
        "audio_positional_embedding_max_pos": [20.0],
        "timestep_scale_multiplier": 1000,
        "use_middle_indices_grid": True,
        "rope_type": "split",
        "frequencies_precision": "float64",
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "caption_proj_before_connector": True,
        "audio_connector_attention_head_dim": 8,
        "audio_connector_num_attention_heads": 4,
        "connector_num_layers": 1,
        "connector_positional_embedding_max_pos": [4096],
        "connector_num_learnable_registers": 4,
    },
    "audio": {
        "sample_rate": 8000,
        "latent_sample_rate": 16000,
        "hop_length": 160,
        "latent_downsample_factor": 4,
        "vae_channels": 4,
        "mel_bins": 4,
        "fps": 25.0,
    },
    "inference_defaults": {
        "cfg_scale": 2.0,
        "stg_scale": 1.0,
        "stg_block": 1,
        "rescale_scale": "auto",
        "modality_scale": 1.0,
        "duration_multiplier": 1.1,
        "seed": 42,
        "steps": 30,
    },
}

CASES = {
    "generate": {"cfg_scale": 2.0, "stg_scale": 1.0, "stg_block": 1, "steps": 4},
    "plain": {"cfg_scale": 1.0, "stg_scale": 0.0, "stg_block": 1, "steps": 4},
}


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


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


class DrawReplayer:
    """Replays recorded draws in order instead of drawing."""

    def __init__(self, draws: list):
        self.draws = draws
        self.cursor = 0
        self._original = mx.random.normal

    def __enter__(self):
        replayer = self

        def patched(*args, **kwargs):
            del args, kwargs
            entry = replayer.draws[replayer.cursor]
            replayer.cursor += 1
            return mx.array(
                np.array(entry["values"], dtype=np.float32).reshape(entry["shape"])
            )

        mx.random.normal = patched
        return self

    def __exit__(self, *exc):
        mx.random.normal = self._original
        return False


def seed_model(model, seed: int):
    rng = np.random.default_rng(seed)
    seeded = {}
    for name, value in dict(tree_flatten(model.parameters())).items():
        seeded[name] = mx.array(
            (rng.standard_normal(value.shape) * 0.3).astype(np.float32)
        )
    model.update(tree_unflatten(seeded))
    return model


def round_trip_f16(model) -> None:
    """Casts every parameter through f16 and back, so the traces come
    from exactly the weights the Rust fixture loads."""
    flat = dict(tree_flatten(model.parameters()))
    f16 = {
        key: value.astype(mx.float16).astype(mx.float32)
        for key, value in flat.items()
    }
    model.update(tree_unflatten(f16))


def build_model() :
    config = ModelConfig.from_dict(dict(TINY_CONFIG))
    model = AudioOnlyLTXModel(config.transformer)
    return model, config


def run_case(model, config, ctx, neg_ctx, params, x0_calls=None) -> tuple:
    """The reference generate() zero-reference flow in f32, returning
    the denoised, silence-patched latents."""
    tools = AudioLatentTools(AudioPatchifier(), latent_shape(config))
    state = tools.create_initial_state(dtype=mx.float32)
    state = _add_gaussian_noise(state)
    inner_model = X0Model(model)
    if x0_calls is not None:
        class RecordingX0:
            def __call__(self, modality, stg_blocks=None):
                out = inner_model(modality, stg_blocks=stg_blocks)
                x0_calls.append(np.asarray(out, dtype=np.float32).ravel().tolist())
                return out
        x0_model = RecordingX0()
    else:
        x0_model = inner_model
    guider = MultiModalGuiderParams(
        cfg_scale=params["cfg_scale"],
        stg_scale=params["stg_scale"],
        stg_blocks=(params["stg_block"],),
        rescale_scale=auto_rescale_for_cfg(params["cfg_scale"]),
        modality_scale=1.0,
    )
    denoised_state = guided_euler_loop(
        state=state,
        x0_model=x0_model,
        context=ctx,
        negative_context=neg_ctx,
        context_mask=None,
        negative_context_mask=None,
        steps=params["steps"],
        guider_params=guider,
    )
    denoised_state = tools.clear_conditioning(denoised_state)
    latents = tools.unpatchify(denoised_state).latent
    latents = patch_long_clip_silence_prior(latents)
    return latents


def latent_shape(config):
    from mlx_audio.tts.models.dramabox.latent import AudioLatentShape

    return AudioLatentShape(batch=1, channels=config.audio.vae_channels, frames=TOKENS, mel_bins=config.audio.mel_bins)


def _add_gaussian_noise(state):
    """`add_gaussian_noise` with the noise draw already recorded: the
    recorder patched mx.random.normal, so call the reference directly."""
    from mlx_audio.tts.models.dramabox.latent import add_gaussian_noise

    return add_gaussian_noise(state, seed=42, noise_scale=1.0)


def build_case(seed: int, case_name: str) -> None:
    mx.random.seed(1100)
    model, config = build_model()
    seed_model(model, seed)
    round_trip_f16(model)

    conditioner_inputs = engineered_hidden_states(seed)

    conditioner_seed = seed + 1
    # The conditioner is a separate module in the reference Model; build
    # and seed it so the recorded context matches the Rust port's.
    conditioner = DramaboxTextConditioner(
        embedding_dim=config.text_encoder_hidden_size,
        audio_inner_dim=config.transformer.audio_cross_attention_dim,
        num_gemma_layers=TINY_CONFIG["num_gemma_layers"],
        connector_layers=config.transformer.connector_num_layers,
        connector_heads=config.transformer.audio_connector_num_attention_heads,
        connector_head_dim=config.transformer.audio_connector_attention_head_dim,
        connector_num_learnable_registers=config.transformer.connector_num_learnable_registers,
    )
    seed_model(conditioner, conditioner_seed)
    round_trip_f16(conditioner)
    hidden = mx.array(
        np.array(conditioner_inputs, dtype=np.float32).reshape(
            1, CONTEXT_TOKENS, config.text_encoder_hidden_size, 3
        )
    )
    mask = mx.ones((1, CONTEXT_TOKENS), dtype=mx.int32)
    features = conditioner.feature_extractor(hidden, mask)
    context, _ = conditioner(hidden, mask)
    negative, _ = conditioner(
        mx.array(
            np.array(conditioner_inputs[::-1], dtype=np.float32).reshape(
                1, CONTEXT_TOKENS, config.text_encoder_hidden_size, 3
            )
        ),
        mask,
    )
    context = context.astype(mx.float32)
    negative = negative.astype(mx.float32)

    params = CASES[case_name]

    # Intermediates for the Rust debugging gate: the sigma schedule and
    # the first cond x0 pass.
    from mlx_audio.tts.models.dramabox.scheduler import ltx2_sigmas, to_denoised
    from mlx_audio.tts.models.dramabox.transformer import TransformerArgsPreprocessor

    sigmas = ltx2_sigmas(steps=params["steps"], latent=mx.zeros((1, TOKENS, config.audio.vae_channels * config.audio.mel_bins)))
    tools_probe = AudioLatentTools(AudioPatchifier(), latent_shape(config))
    state_probe = tools_probe.create_initial_state(dtype=mx.float32)
    positions = state_probe.positions
    timesteps0 = state_probe.denoise_mask[..., 0] * sigmas[0].reshape(1, 1)
    modality0 = Modality(
        latent=state_probe.latent,
        sigma=sigmas[0].reshape(-1),
        timesteps=timesteps0,
        positions=positions,
        context=context,
        context_mask=None,
        attention_mask=state_probe.attention_mask,
    )
    velocity0 = model(modality0, None)
    cond0 = X0Model(model)(modality0, None)

    prep = TransformerArgsPreprocessor(
        patchify_proj=model.audio_patchify_proj,
        adaln=model.audio_adaln_single,
        inner_dim=model.audio_inner_dim,
        max_pos=config.transformer.audio_positional_embedding_max_pos,
        num_attention_heads=config.transformer.audio_num_attention_heads,
        use_middle_indices_grid=config.transformer.use_middle_indices_grid,
        timestep_scale_multiplier=config.transformer.timestep_scale_multiplier,
        double_precision_rope=config.transformer.frequencies_precision == "float64",
        positional_embedding_theta=config.transformer.positional_embedding_theta,
        rope_type=config.transformer.rope_type,
        prompt_adaln=model.audio_prompt_adaln_single,
    )
    args0 = prep.prepare(modality0)
    cos0, sin0 = args0.positional_embeddings

    # Per-block probes.
    args_full = prep.prepare(modality0)
    xs = []
    for block in model.transformer_blocks:
        args_full = block(args_full)
        xs.append(args_full.x)
    out_norm = model.audio_norm_out(xs[1])
    shift_head = model.audio_scale_shift_table[0][None, None, :] + args_full.embedded_timestep
    scale_head = model.audio_scale_shift_table[1][None, None, :] + args_full.embedded_timestep
    out_scaled = out_norm * (1 + scale_head) + shift_head
    out_head = model.audio_proj_out(out_scaled)

    block0 = model.transformer_blocks[0]
    import numpy as _np

    # Self-attention stage only: reproduce the block0 up to attn1.
    from mlx_audio.tts.models.dramabox.layers import rms_norm as _rms_norm

    shift_msa, scale_msa, gate_msa = (
        v.squeeze(2)
        for v in block0._ada_values(block0.audio_scale_shift_table, args0.x.shape[0], args0.timesteps, 0, 3)
    )
    norm_x = _rms_norm(args0.x, eps=block0.norm_eps)
    norm_x = norm_x * (1 + scale_msa) + shift_msa
    attn1_out = block0.audio_attn1(norm_x, pe=args0.positional_embeddings, mask=None)
    after_attn1 = args0.x + attn1_out * gate_msa

    # Cross-attention stage.
    shift_q, scale_q, gate_c = (
        v.squeeze(2)
        for v in block0._ada_values(block0.audio_scale_shift_table, args0.x.shape[0], args0.timesteps, 6, 9)
    )
    prompt_values = block0.audio_prompt_scale_shift_table[None, None] + args0.prompt_timestep.reshape(
        args0.x.shape[0], args0.prompt_timestep.shape[1], 2, -1
    )
    shift_kv, scale_kv = (v.squeeze(2) for v in mx.split(prompt_values, 2, axis=2))
    attn_input = _rms_norm(after_attn1, eps=block0.norm_eps) * (1 + scale_q) + shift_q
    context_mod = args0.context * (1 + scale_kv) + shift_kv
    cross = block0.audio_attn2(attn_input, context=context_mod, mask=None)
    after_cross = after_attn1 + cross * gate_c

    recorder = DrawRecorder()
    x0_calls = []
    with recorder:
        latents = run_case(model, config, context, negative, params, x0_calls)
    draws = recorder.flush()

    traces = {
        "tokens": TOKENS,
        "context_tokens": CONTEXT_TOKENS,
        "cfg_scale": params["cfg_scale"],
        "stg_scale": params["stg_scale"],
        "stg_block": params["stg_block"],
        "steps": params["steps"],
        "conditioner_inputs": floats(conditioner_inputs),
        "negative_conditioner_inputs": floats(conditioner_inputs[::-1]),
        "attention_mask": [1] * CONTEXT_TOKENS,
        "context": floats(context),
        "features": floats(features),
        "sigmas": floats(sigmas),
        "cond0": floats(cond0),
        "velocity0": floats(velocity0),
        "positions": floats(positions),
        "timestep_out": floats(args0.timesteps),
        "embedded_timestep": floats(args0.embedded_timestep),
        "prompt_timestep": floats(args0.prompt_timestep),
        "audio_cos": floats(cos0),
        "audio_sin": floats(sin0),
        "patched": floats(state_probe.latent),
        "after_attn1": floats(after_attn1),
        "cross": floats(cross),
        "after_cross": floats(after_cross),
        "block0_out": floats(xs[0]),
        "block1_out": floats(xs[1]),
        "out_norm": floats(out_norm),
        "shift_head": floats(shift_head),
        "scale_head": floats(scale_head),
        "out_scaled": floats(out_scaled),
        "out_head": floats(out_head),
        "attn1_out": floats(attn1_out),
        "draws": [{"shape": shape, "values": values} for shape, values in draws],
        "latents": floats(latents),
        "x0_calls": x0_calls,
    }

    case_dir = OUT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    # One combined save: transformer + conditioner parameters (the
    # Rust loader reads both from the same shard set).
    stored = {}
    for key, value in dict(tree_flatten(model.parameters())).items():
        stored[f"transformer.{key}"] = value.astype(mx.float16)
    for key, value in dict(tree_flatten(conditioner.parameters())).items():
        stored[f"text_conditioner.{key}"] = value.astype(mx.float16)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces))

    python_replay_check(case_dir, traces)
    print(f"{case_name}: {len(draws)} draws, replay ok")


def engineered_hidden_states(seed: int) -> list:
    rng = np.random.default_rng(seed + 100)
    values = rng.standard_normal((CONTEXT_TOKENS, TINY_CONFIG["text_encoder_hidden_size"], 3))
    return (values * 0.5).astype(np.float32).ravel().tolist()


def reload_model_from(case_dir: Path):
    weights = mx.load(str(case_dir / "model.safetensors"))
    model, config = build_model()
    model.update(
        tree_unflatten(
            {
                key[len("transformer."):]: mx.array(value).astype(mx.float32)
                for key, value in weights.items()
                if key.startswith("transformer.")
            }
        )
    )
    conditioner = DramaboxTextConditioner(
        embedding_dim=config.text_encoder_hidden_size,
        audio_inner_dim=config.transformer.audio_cross_attention_dim,
        num_gemma_layers=TINY_CONFIG["num_gemma_layers"],
        connector_layers=config.transformer.connector_num_layers,
        connector_heads=config.transformer.audio_connector_num_attention_heads,
        connector_head_dim=config.transformer.audio_connector_attention_head_dim,
        connector_num_learnable_registers=config.transformer.connector_num_learnable_registers,
    )
    conditioner.update(
        tree_unflatten(
            {
                key[len("text_conditioner."):]: mx.array(value).astype(mx.float32)
                for key, value in weights.items()
                if key.startswith("text_conditioner.")
            }
        )
    )
    return model, conditioner, config


def python_replay_check(case_dir: Path, traces: dict) -> None:
    """The recording caveat gate: fresh weights plus the recorded draw
    stream must reproduce the recorded latents before anything
    freezes."""
    model, conditioner, config = reload_model_from(case_dir)
    params = {key: traces[key] for key in ("cfg_scale", "stg_scale", "stg_block", "steps")}
    hidden = mx.array(
        np.array(traces["conditioner_inputs"], dtype=np.float32).reshape(
            1, CONTEXT_TOKENS, config.text_encoder_hidden_size, 3
        )
    )
    mask = mx.ones((1, CONTEXT_TOKENS), dtype=mx.int32)
    context, _ = conditioner(hidden, mask)
    negative, _ = conditioner(
        mx.array(
            np.array(traces["negative_conditioner_inputs"], dtype=np.float32).reshape(
                1, CONTEXT_TOKENS, config.text_encoder_hidden_size, 3
            )
        ),
        mask,
    )
    with DrawReplayer(traces["draws"]):
        latents = run_case(model, config, context.astype(mx.float32), negative.astype(mx.float32), params)
    diff = float(
        np.abs(
            np.asarray(latents, dtype=np.float32).ravel()
            - np.array(traces["latents"], dtype=np.float32)
        ).max()
    )
    assert diff <= REPLAY_GATE, f"python replay mismatch {diff} in {case_dir.name}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_case(21, "generate")
    build_case(23, "plain")

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_dramabox_fixtures.py",
        "generate_seed": 21,
        "plain_seed": 23,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
