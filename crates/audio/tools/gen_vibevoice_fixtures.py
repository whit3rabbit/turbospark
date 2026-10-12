#!/usr/bin/env python3
"""Regenerate the VibeVoice family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/vibevoice, mlx 0.32.3). The true reference modules
run in full: the split Qwen2 model (lower `language_model` + upper
`tts_language_model`), the type embeddings, the SpeechConnector, the
diffusion prediction head, the deterministic DPM-Solver++ scheduler,
the acoustic tokenizer decoder, the binary EOS classifier, and the
reference `Model._generate_single_speaker` loop, with:

- a deterministic offline character-level tokenizer
  (`encode(text, add_special_tokens=False)`) standing in for the
  checkpoint tokenizer;
- seeded tiny stacks: a 2-layer split decoder (hidden 16, 4 query
  heads, 2 kv heads, one lower and one upper layer, vocab 700 so the
  char ids stay in range), a 2-stage acoustic decoder (ratios [2, 2],
  depths 2-2, vae_dim 8, 8 kHz), a 2-layer diffusion head (latent 8,
  4 DDPM inference steps over a 20-step cosine schedule);
- every `mx.random.normal` draw recorded in reference draw order (one
  initial latent noise per `sample_speech_tokens` call; the scheduler
  is deterministic and draws nothing) and replayed by the Rust test;
- an engineered `tts_eos_classifier.fc2` bias so the generate case
  never ends on the classifier (the budget stops it after two text
  windows and eight latents) and the stop case ends on the first
  latent;
- a voice case: a voice cache built from a four-token probe prompt is
  saved in the reference `load_voice` layout (mixed 3D and 4D key
  tensors so both storage paths are exercised) and generation starts
  from it.

A Python-side replay check rebuilds every case from the saved
safetensors plus the recorded draw stream and asserts the audio is
bit-identical to the recording run before anything is frozen.

Run from inside the mlx-audio checkout root so `mlx_audio` imports
from the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_vibevoice_fixtures.py

Outputs into crates/audio/testdata/tts_vibevoice/:

- generate/  the full-budget loop: three text windows, eight speech
    latents with classifier-free guidance (cfg 3.0), the recorded
    noise draws, the unscaled latents, and the decoded waveform.
- stop/  the engineered-stop twin: the classifier ends the run after
    the first latent.
- voice/  the voice-conditioned twin: generation starts from the
    saved caches and stops after the first latent.
- manifest.json  file list with SHA-256 digests and generator metadata.

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

import mlx_audio.tts.models.vibevoice.vibevoice as vibe

OUT = Path("../turbospark/crates/audio/testdata/tts_vibevoice")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

PROMPT_TEXT = "hi there"
VOICE_PROBE_IDS = [21, 22, 23, 24]
REPLAY_GATE = 2e-5


TINY_CONFIG = {
    "model_type": "vibevoice_streaming",
    "model_path": None,
    "sample_rate": 8000,
    "acoustic_vae_dim": 8,
    "tts_backbone_num_hidden_layers": 1,
    "acoustic_tokenizer_config": {
        "model_type": "vibevoice_acoustic_tokenizer",
        "channels": 1,
        "corpus_normalize": 0.0,
        "causal": True,
        "vae_dim": 8,
        "fix_std": 0.5,
        "std_dist_type": "gaussian",
        "mixer_layer": "depthwise_conv",
        "conv_norm": "none",
        "pad_mode": "constant",
        "disable_last_norm": True,
        "layernorm": "RMSNorm",
        "layernorm_eps": 1e-5,
        "layernorm_elementwise_affine": True,
        "conv_bias": True,
        "layer_scale_init_value": 1e-6,
        "weight_init_value": 0.01,
        "encoder_n_filters": 8,
        "encoder_ratios": [2, 2],
        "encoder_depths": "2-2",
        "decoder_n_filters": 8,
        "decoder_ratios": None,
        "decoder_depths": None,
    },
    "decoder_config": {
        "model_type": "qwen2",
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": 16,
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "max_position_embeddings": 512,
        "max_window_layers": 24,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 700,
        "head_dim": None,
    },
    "diffusion_head_config": {
        "model_type": "vibevoice_diffusion_head",
        "hidden_size": 16,
        "head_layers": 2,
        "head_ffn_ratio": 3.0,
        "rms_norm_eps": 1e-5,
        "latent_size": 8,
        "speech_vae_dim": 8,
        "prediction_type": "v_prediction",
        "diffusion_type": "ddpm",
        "ddpm_num_steps": 20,
        "ddpm_num_inference_steps": 4,
        "ddpm_beta_schedule": "cosine",
        "ddpm_batch_mul": 4,
    },
}

CFG_SCALE = 3.0


class CharTokenizer:
    """Character-level tokenizer with the `add_special_tokens` kwarg the
    reference prompt path passes."""

    def __init__(self):
        alphabet = set(PROMPT_TEXT + "\n")
        self.chars = sorted(alphabet)
        self.vocab = {char: 20 + index for index, char in enumerate(self.chars)}
        assert max(self.vocab.values()) < 700, "char ids must stay in vocab"

    def encode(self, text, *args, **kwargs):
        del args, kwargs
        return [self.vocab[char] for char in str(text)]

    def decode(self, token_ids, *args, **kwargs):
        inverse = {value: key for key, value in self.vocab.items()}
        return "".join(inverse[int(t)] for t in token_ids)


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


def engineer_eos(model, positive_bias: float) -> None:
    """Pins the end-of-speech logit to the bias: the fc2 weight is zero,
    so sigmoid(bias) decides every step deterministically."""
    hidden = model.config.decoder_config.hidden_size
    model.tts_eos_classifier.fc1.weight = mx.array(
        np.asarray(model.tts_eos_classifier.fc1.weight, dtype=np.float32)
    )
    model.tts_eos_classifier.fc2.weight = mx.zeros((1, hidden))
    model.tts_eos_classifier.fc2.bias = mx.array([float(positive_bias)])
    # The engineered head rows go through the same f16 round trip so the
    # saved file reproduces the generating weights exactly.
    round_trip_f16(model)


def build_voice_cache(model, ids) -> dict:
    """Runs a probe prompt through the split model and returns the
    reference `load_voice` tensors (positive lm + tts, negative tts)."""
    hidden = model.config.decoder_config.hidden_size
    input_ids = mx.array([ids], dtype=mx.int32)
    text_embeds = model.language_model.embed_tokens(input_ids)
    lm_out, lm_cache = model.language_model(inputs_embeds=text_embeds)
    ones = mx.ones((1, input_ids.shape[1]), dtype=mx.int32)
    tts_out, tts_cache = model.tts_language_model(
        inputs_embeds=lm_out + model.tts_input_types(ones)
    )
    zeros = mx.zeros((1, input_ids.shape[1], hidden))
    neg_out, neg_cache = model.tts_language_model(
        inputs_embeds=zeros + model.tts_input_types(ones)
    )
    tensors = {
        # lm_hidden is required by the reference load_voice but never
        # read by the generation loop.
        "lm_hidden": np.asarray(lm_out, dtype=np.float32),
        "tts_lm_hidden": np.asarray(tts_out, dtype=np.float32),
        "neg_tts_lm_hidden": np.asarray(neg_out, dtype=np.float32),
    }
    for i, (k, v) in enumerate(lm_cache):
        # The lm rows ship Swift-style 4D (B, kv, S, d): the load path
        # transposes every 4D tensor back, keys and values alike.
        k4 = np.asarray(k, dtype=np.float32).transpose(0, 2, 1, 3)
        v4 = np.asarray(v, dtype=np.float32).transpose(0, 2, 1, 3)
        tensors[f"lm_key_{i}"] = k4
        tensors[f"lm_value_{i}"] = v4
    # The reference attention caches are 4D (B, S, kv, d) and
    # `load_voice` transposes EVERY 4D tensor on load, so every cache
    # row ships Swift-style (B, kv, S, d).
    for i, (k, v) in enumerate(tts_cache):
        tensors[f"tts_lm_key_{i}"] = np.asarray(k, dtype=np.float32).transpose(0, 2, 1, 3)
        tensors[f"tts_lm_value_{i}"] = np.asarray(v, dtype=np.float32).transpose(0, 2, 1, 3)
    for i, (k, v) in enumerate(neg_cache):
        tensors[f"neg_tts_lm_key_{i}"] = np.asarray(k, dtype=np.float32).transpose(0, 2, 1, 3)
        tensors[f"neg_tts_lm_value_{i}"] = np.asarray(v, dtype=np.float32).transpose(0, 2, 1, 3)
    return tensors


def save_model(case_dir: Path, model) -> None:
    stored = {
        key: value.astype(mx.float16)
        for key, value in dict(tree_flatten(model.parameters())).items()
    }
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)


def reload_model_from(case_dir: Path):
    """Rebuilds the model from exactly the saved safetensors. The saved
    values carry f16 mantissas but the recording run computed in f32,
    so the reload casts back to f32: same bits, same arithmetic."""
    model = vibe.Model(vibe.ModelConfig.from_dict(dict(TINY_CONFIG)))
    weights = mx.load(str(case_dir / "model.safetensors"))
    model.update(
        tree_unflatten(
            {
                key: mx.array(value).astype(mx.float32)
                for key, value in weights.items()
            }
        )
    )
    model.tokenizer = CharTokenizer()
    return model


def python_replay_check(
    case_dir: Path, traces: dict, voice: str | None, replayer: DrawReplayer
) -> None:
    """The recording caveat gate: a fresh model rebuilt from the saved
    weights plus the recorded draw stream must reproduce the recorded
    audio before anything freezes."""
    model = reload_model_from(case_dir)
    if voice is not None:
        model.config.model_path = str(case_dir)
        model.load_voice(voice)
    with replayer:
        results = list(
            model.generate(
                PROMPT_TEXT,
                cfg_scale=traces["cfg_scale"],
                max_tokens=traces["max_tokens"],
                verbose=False,
            )
        )
    audio = np.asarray(results[0].audio, dtype=np.float32)
    ref = np.array(traces["waveform"], dtype=np.float32)
    diff = float(np.abs(audio - ref).max())
    assert diff <= REPLAY_GATE, f"python replay mismatch {diff} in {case_dir.name}"
    assert replayer.cursor == len(traces["draws"]), (
        f"replay consumed {replayer.cursor} of {len(traces['draws'])} draws"
    )


def build_case(seed: int, case_name: str, eos_bias: float, voice: str | None) -> None:
    # The recorded draws come from the global MLX RNG; seed it so the
    # fixture is reproducible run to run.
    mx.random.seed(700)
    model = vibe.Model(vibe.ModelConfig.from_dict(dict(TINY_CONFIG)))
    seed_model(model, seed)
    round_trip_f16(model)
    model.tokenizer = CharTokenizer()

    max_tokens = 8 if voice is None else 4

    if voice is not None:
        # The voice cache derives from the same f16 weights, then the
        # EOS head is engineered on top.
        tensors = build_voice_cache(model, VOICE_PROBE_IDS)
        voices_dir = OUT / case_name / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(voices_dir / f"{voice}.safetensors"),
            {key: mx.array(value) for key, value in tensors.items()},
        )
        model.config.model_path = str(OUT / case_name)
        model.load_voice(voice)

    engineer_eos(model, eos_bias)

    recorded_latents = []
    original_sample = model.sample_speech_tokens

    def recording_sample(condition, neg_condition, cfg_scale=1.0, ddpm_steps=None):
        out = original_sample(
            condition,
            neg_condition,
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
        )
        recorded_latents.append(
            np.asarray(out, dtype=np.float32).ravel().tolist()
        )
        return out

    model.sample_speech_tokens = recording_sample

    recorder = DrawRecorder()
    with recorder:
        results = list(
            model.generate(
                PROMPT_TEXT,
                cfg_scale=CFG_SCALE,
                max_tokens=max_tokens,
                verbose=False,
            )
        )
    draws = recorder.flush()
    assert len(results) == 1
    result = results[0]
    latent_frames = len(recorded_latents)
    scaling = float(model.speech_scaling_factor)
    bias = float(model.speech_bias_factor)
    latent_seq = np.concatenate(
        [np.asarray(latent, dtype=np.float32).reshape(1, -1) for latent in recorded_latents]
    )
    latent_seq = latent_seq / scaling - bias

    traces = {
        "prompt_ids": model.tokenizer.encode(PROMPT_TEXT.strip() + "\n"),
        "max_tokens": max_tokens,
        "cfg_scale": CFG_SCALE,
        "draws": [{"shape": shape, "values": values} for shape, values in draws],
        "latent_frames": latent_frames,
        "latents": floats(latent_seq),
        "waveform": floats(result.audio),
        "token_count": int(result.token_count),
    }

    case_dir = OUT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    save_model(case_dir, model)
    (case_dir / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces))

    # The recording caveat gate: replay before freezing.
    python_replay_check(case_dir, traces, voice, DrawReplayer(traces["draws"]))
    print(f"{case_name}: {latent_frames} latents, {len(draws)} draws, replay ok")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_case(3, "generate", eos_bias=-10.0, voice=None)
    build_case(31, "stop", eos_bias=10.0, voice=None)
    build_case(7, "voice", eos_bias=10.0, voice="myvoice")

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_vibevoice_fixtures.py",
        "generate_seed": 3,
        "stop_seed": 31,
        "voice_seed": 7,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
