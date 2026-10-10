#!/usr/bin/env python3
"""Regenerate the KugelAudio family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/kugelaudio plus the vibevoice modules it reuses,
mlx 0.32.3). The true reference modules run in full: the vibevoice
Qwen2 decoder, the SpeechConnector, the diffusion prediction head, the
SDE-DPM-Solver++ scheduler, the acoustic tokenizer decoder, and the
reference `Model._generate_impl` loop, with:

- a deterministic offline character-level tokenizer
  (`encode(text, add_special_tokens=False)`) standing in for the
  checkpoint tokenizer;
- seeded tiny stacks: a 2-layer qwen2 decoder (hidden 16, vocab 151656
  so the repurposed Qwen2.5 speech token ids stay in range), a 2-stage
  acoustic decoder (ratios [2, 2], depths 2-2, vae_dim 8, 8 kHz), and a
  2-layer diffusion head (latent 8, 4 DDPM inference steps over a
  20-step cosine schedule);
- every `mx.random.normal` draw recorded in reference draw order (the
  initial latent noise per `sample_speech_tokens` call, then one draw
  per scheduler step) and replayed by the Rust test;
- engineered `lm_head` rows along the prompt's last hidden state so the
  greedy constrained argmax deterministically emits speech diffusion
  tokens for the whole budget (generate case) or stops on the first
  choice (stop case).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_kugelaudio_fixtures.py

Outputs into crates/audio/testdata/tts_kugelaudio/:

- generate/  the AR loop over the full budget: four speech-diffusion
    steps with classifier-free guidance (cfg 3.0), the recorded noise
    draws, the unscaled latents, and the decoded, peak-normalized
    waveform.
- stop/  the engineered-stop twin: speech_end wins the first
    constrained argmax with no latents yet, so the run finishes with no
    audio.
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

import mlx_audio.tts.models.kugelaudio.kugelaudio as kugel

OUT = Path("../turbospark/crates/audio/testdata/tts_kugelaudio")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

PROMPT_TEXT = "hello world today"

TINY_CONFIG = {
    "model_type": "kugelaudio",
    "sample_rate": 8000,
    "acoustic_vae_dim": 8,
    "semantic_vae_dim": 8,
    "tie_word_embeddings": False,
    "acoustic_tokenizer_config": {
        "model_type": "kugelaudio_acoustic_tokenizer",
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
        "hidden_size": 16,
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "max_position_embeddings": 512,
        "max_window_layers": 28,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_mrope": False,
        "use_sliding_window": False,
        "vocab_size": 151656,
        "head_dim": None,
    },
    "diffusion_head_config": {
        "model_type": "kugelaudio_diffusion_head",
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
        "ddpm_algorithm_type": "sde-dpmsolver++",
        "ddpm_batch_mul": 4,
    },
}

MAX_TOKENS = 4
CFG_SCALE = 3.0


SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)


def text_section(formatted_text: str) -> str:
    return f" Text input:\n {formatted_text}\n Speech output:\n"


class CharTokenizer:
    """Character-level tokenizer with the `add_special_tokens` kwarg the
    reference prompt builder passes."""

    def __init__(self):
        alphabet = set(SYSTEM_PROMPT + text_section(PROMPT_TEXT) + "Speaker 0:")
        self.chars = sorted(alphabet)
        self.vocab = {char: 20 + index for index, char in enumerate(self.chars)}
        assert max(self.vocab.values()) < 151643, "char ids must stay below EOS"

    def encode(self, text: str, *args, **kwargs) -> list:
        del args, kwargs
        return [self.vocab[char] for char in str(text)]

    def decode(self, token_ids, *args, **kwargs) -> str:
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


def seed_model(model, seed: int):
    rng = np.random.default_rng(seed)
    seeded = {}
    for name, value in dict(tree_flatten(model.parameters())).items():
        seeded[name] = mx.array(
            (rng.standard_normal(value.shape) * 0.3).astype(np.float32)
        )
    model.update(tree_unflatten(seeded))
    return model


def probe_hidden(model, prompt_ids) -> np.ndarray:
    """The prompt's last post-norm hidden state (cacheless probe)."""
    hidden, _ = model.language_model(input_ids=mx.array([prompt_ids]))
    return np.asarray(hidden[:, -1, :], dtype=np.float32)[0]


def engineer_lm_head(model, prompt_ids, stop_run: bool) -> None:
    """Sets the four constrained speech-token head rows along the
    prompt's last hidden state: the generate case keeps speech diffusion
    winning the whole budget, the stop case hands the first choice to
    speech end."""
    hidden = probe_hidden(model, prompt_ids)
    weight = np.asarray(model.lm_head.weight, dtype=np.float32)
    if stop_run:
        weight[kugel.SPEECH_END_ID] = 6.0 * hidden
        weight[kugel.SPEECH_DIFFUSION_ID] = 5.0 * hidden
        weight[kugel.EOS_TOKEN_ID] = 4.0 * hidden
        weight[kugel.SPEECH_START_ID] = 3.0 * hidden
    else:
        weight[kugel.SPEECH_DIFFUSION_ID] = 6.0 * hidden
        weight[kugel.SPEECH_END_ID] = 5.0 * hidden
        weight[kugel.EOS_TOKEN_ID] = 4.0 * hidden
        weight[kugel.SPEECH_START_ID] = 3.0 * hidden
    model.lm_head.weight = mx.array(weight)
    mx.eval(model.lm_head.weight)


def round_trip_f16(model) -> None:
    """Casts every parameter through f16 and back, so the traces come
    from exactly the weights the Rust fixture loads."""
    flat = dict(tree_flatten(model.parameters()))
    f16 = {
        key: value.astype(mx.float16).astype(mx.float32)
        for key, value in flat.items()
    }
    model.update(tree_unflatten(f16))


def build_case(seed: int, case_name: str, stop_run: bool) -> None:
    # The recorded draws come from the global MLX RNG; seed it so the
    # fixture is reproducible run to run.
    mx.random.seed(700)
    model = kugel.Model(kugel.ModelConfig.from_dict(dict(TINY_CONFIG)))
    seed_model(model, seed)
    round_trip_f16(model)
    model.tokenizer = CharTokenizer()

    prompt_tokens = model._build_prompt_tokens(PROMPT_TEXT)
    assert prompt_tokens[-1] == kugel.SPEECH_START_ID
    prompt_ids = [int(t) for t in prompt_tokens]

    engineer_lm_head(model, prompt_ids, stop_run)
    # The engineered head rows go through the same f16 round trip so the
    # saved file reproduces the generating weights exactly.
    round_trip_f16(model)

    recorded_latents = []
    original_sample = model.sample_speech_tokens

    def recording_sample(condition, cfg_scale=1.0, neg_condition=None):
        out = original_sample(
            condition, cfg_scale=cfg_scale, neg_condition=neg_condition
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
                max_tokens=MAX_TOKENS,
                verbose=False,
            )
        )
    draws = recorder.flush()
    assert len(results) == 1
    result = results[0]

    if stop_run:
        assert len(recorded_latents) == 0, "the stop case must not sample latents"
        assert int(np.asarray(result.audio).shape[-1]) == 0
    else:
        assert len(recorded_latents) == MAX_TOKENS, (
            "the generate case must fill the budget"
        )
        scaling = float(model.speech_scaling_factor)
        bias = float(model.speech_bias_factor)
        latent_seq = np.concatenate(
            [np.asarray(latent, dtype=np.float32) for latent in recorded_latents]
        )
        latent_seq = latent_seq / scaling - bias

    traces = {
        "prompt_ids": prompt_ids,
        "max_tokens": MAX_TOKENS,
        "cfg_scale": CFG_SCALE,
        "draws": [{"shape": shape, "values": values} for shape, values in draws],
    }
    if stop_run:
        traces["latent_frames"] = 0
        traces["token_count"] = int(result.token_count)
    else:
        traces["latent_frames"] = int(len(recorded_latents))
        traces["latents"] = floats(latent_seq)
        traces["waveform"] = floats(result.audio)
        traces["token_count"] = int(result.token_count)

    case_dir = OUT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    stored = {
        key: value.astype(mx.float16)
        for key, value in dict(tree_flatten(model.parameters())).items()
    }
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_case(3, "generate", False)
    build_case(31, "stop", True)

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_kugelaudio_fixtures.py",
        "generate_seed": 3,
        "stop_seed": 31,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
