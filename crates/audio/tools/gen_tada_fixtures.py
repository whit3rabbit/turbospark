#!/usr/bin/env python3
"""Regenerate the TADA family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/tada, mlx 0.32.3). The true reference modules run
in full: the TADA Llama backbone, the acoustic/mask/duration
embeddings, the flow-matching prediction head, the logSNR time
schedule with the scheduled two-half CFG, the gray-code duration
decode, the de-normalization and the zero-shot skip math, and the
reference `Model.generate` -> `_generate_loop` zero-shot path, with:

- a deterministic offline tokenizer with the Llama special ids the
  chat prefix needs (bos, eot, header markers, pad) standing in for
  the Llama-3.2 tokenizer;
- seeded tiny stacks: a 2-layer llama (hidden 32, 4 query heads, 2 kv
  heads, head_dim 8, vocab 200, tied head), acoustic_dim 8,
  num_time_classes 16 (4 gray bits per duration), shift_acoustic 2,
  a 2-layer diffusion head (latent 16), 4 flow-matching steps;
- the tada-codec decoder is constructed (it rides the config) but
  never run: the port ends at the de-normalized acoustic features and
  durations by design (the codec ships in the separate
  HumeAI/tada-codec download);
- every `mx.random.normal` draw recorded in reference draw order (one
  per generation step, before the flow integration) and replayed by
  the Rust test.

Two cases: `generate` (acoustic cfg 1.6, cosine cfg schedule over the
logSNR grid) and `nocfg` (acoustic cfg 1.0, the single-row forward
path). A Python-side replay check rebuilds each case from the saved
safetensors plus the recorded draw stream and asserts the features
match before anything is frozen.

Run from inside the mlx-audio checkout root so `mlx_audio` imports
from the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_tada_fixtures.py

Outputs into crates/audio/testdata/tts_tada/<case>/ (config.json +
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

import mlx_audio.tts.models.tada.tada as tada

OUT = Path("../turbospark/crates/audio/testdata/tts_tada")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

PROMPT_TEXT = "hello world"
REPLAY_GATE = 2e-5

# The tiny tokenizer's special ids (mirrored in the Rust tests).
BOS = 3
EOT = 4
START_HEADER = 5
END_HEADER = 6
PAD = 7


TINY_CONFIG = {
    "model_type": "tada",
    "sample_rate": 24000,
    # Llama backbone
    "vocab_size": 200,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 64,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True,
    # TADA-specific
    "acoustic_dim": 8,
    "num_time_classes": 16,
    "shift_acoustic": 2,
    "head_layers": 2,
    "head_ffn_ratio": 4.0,
    "bottleneck_dim": None,
    "context_window": 8,
    "acoustic_mean": 0.0,
    "acoustic_std": 1.5,
    "diffusion_head_type": "vibevoice",
    # Decoder config (constructed, never run)
    "decoder_hidden_dim": 32,
    "decoder_d_model": 8,
    "decoder_embed_dim": 8,
    "decoder_strides": [2, 2],
    "decoder_num_attn_layers": 1,
    "decoder_num_attn_heads": 2,
    "decoder_attn_dim_feedforward": 16,
    "decoder_block_attention": "v2",
    # Encoder config (constructed in the reference hook, not here)
    "encoder_hidden_dim": 32,
    "encoder_d_model": 8,
    "encoder_embed_dim": 8,
    "encoder_strides": [2, 2],
    "encoder_num_attn_layers": 1,
    "encoder_num_attn_heads": 2,
    "encoder_attn_dim_feedforward": 16,
    "encoder_block_attention": "v2",
    "encoder_std": 0.5,
    # EOS tokens
    "eos_token_id": 100,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
}

PARAMS = {
    "generate": {
        "acoustic_cfg_scale": 1.6,
        "duration_cfg_scale": 1.0,
        "cfg_schedule": "cosine",
        "noise_temperature": 0.9,
        "num_flow_matching_steps": 4,
        "time_schedule": "logsnr",
        "num_transition_steps": 5,
        "max_tokens": 64,
        "num_extra_steps": 0,
    },
    "nocfg": {
        "acoustic_cfg_scale": 1.0,
        "duration_cfg_scale": 1.0,
        "cfg_schedule": "cosine",
        "noise_temperature": 0.9,
        "num_flow_matching_steps": 4,
        "time_schedule": "logsnr",
        "num_transition_steps": 5,
        "max_tokens": 64,
        "num_extra_steps": 0,
    },
}


class TinyTokenizer:
    """Character-level tokenizer with the Llama special-token surface
    the reference generate path touches."""

    def __init__(self):
        # normalize_text capitalizes the first letter, so the alphabet
        # must cover the capitalized prompt.
        alphabet = set(PROMPT_TEXT.capitalize() + " systemassistant")
        self.chars = sorted(alphabet)
        self.vocab = {char: 20 + index for index, char in enumerate(self.chars)}
        assert max(self.vocab.values()) < 200, "char ids must stay in vocab"

    @property
    def bos_token_id(self) -> int:
        return BOS

    def encode(self, text, *args, **kwargs):
        del args, kwargs
        return [self.vocab[char] for char in str(text)]

    def convert_tokens_to_ids(self, token: str) -> int:
        table = {
            "<|eot_id|>": EOT,
            "<|start_header_id|>": START_HEADER,
            "<|end_header_id|>": END_HEADER,
            "<|finetune_right_pad_id|>": PAD,
        }
        return table[token]


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


def build_input_ids(model, params) -> tuple:
    """The reference generate() id assembly with the tiny tokenizer."""
    text = tada.normalize_text(PROMPT_TEXT)
    tokenizer = model._tokenizer
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    start_header = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    system_tokens = tokenizer.encode("system", add_special_tokens=False)
    assistant_tokens = tokenizer.encode("assistant", add_special_tokens=False)
    prefix_text_tokens = (
        [start_header]
        + system_tokens
        + [end_header, eot_id, start_header]
        + assistant_tokens
        + [end_header]
    )
    prefix = [tokenizer.bos_token_id] + prefix_text_tokens
    eos_suffix = [eot_id] * model.num_eos_tokens
    ids = prefix + text_tokens + eos_suffix
    prefix_len = len(prefix_text_tokens)
    del params
    return ids, prefix_len


def build_case(seed: int, case_name: str) -> None:
    mx.random.seed(900)
    model = tada.Model(tada.ModelConfig.from_dict(dict(TINY_CONFIG)))
    seed_model(model, seed)
    round_trip_f16(model)
    model._tokenizer = TinyTokenizer()

    params = PARAMS[case_name]
    input_ids, prefix_len = build_input_ids(model, params)

    recorded_steps = []
    original_solve = model._solve_flow_matching

    def recording_solve(speech, cond, neg_cond, num_steps=20, **kwargs):
        out = original_solve(speech, cond, neg_cond, num_steps=num_steps, **kwargs)
        recorded_steps.append(
            np.asarray(out, dtype=np.float32).ravel().tolist()
        )
        return out

    model._solve_flow_matching = recording_solve

    recorder = DrawRecorder()
    with recorder:
        results = list(
            model.generate(
                PROMPT_TEXT,
                acoustic_cfg_scale=params["acoustic_cfg_scale"],
                duration_cfg_scale=params["duration_cfg_scale"],
                cfg_schedule=params["cfg_schedule"],
                noise_temperature=params["noise_temperature"],
                num_flow_matching_steps=params["num_flow_matching_steps"],
                time_schedule=params["time_schedule"],
                num_transition_steps=params["num_transition_steps"],
                max_tokens=params["max_tokens"],
                verbose=False,
            )
        )
    draws = recorder.flush()
    assert len(results) == 1
    audio = np.asarray(results[0].audio, dtype=np.float32)
    assert audio.size > 0, "the reference decode must produce audio on the tiny codec"

    # Rebuild the port-side outputs from the recording: the final
    # de-normalized, skip-trimmed features and durations. The reference
    # decoded them through the tiny codec, so replay the loop outputs.
    # all_acoustic_features after skip and de-normalization are exactly
    # the codec latents.
    shift = model.config.shift_acoustic
    num_steps = min(len(input_ids), params["max_tokens"])
    features_per_step = recorded_steps  # not used directly; see below

    # Recompute the loop bookkeeping to slice the recorded per-step
    # speech vectors the way _generate_loop does.
    collected = []
    times = []
    for step in range(shift, num_steps):
        collected.append(features_per_step[step][: model.config.acoustic_dim])
        # time_before comes from the gray decode of the step sample.
        bits = model.num_time_bits
        tail = features_per_step[step][model.config.acoustic_dim:]
        import mlx_audio.tts.models.tada.gray_code as gray

        before = int(gray.gray_code_to_int(tail[:bits]))
        times.append(before)
    times.append(times[-1])

    features = np.array(collected, dtype=np.float32)
    features = features * model.config.acoustic_std + model.config.acoustic_mean
    zero_shot_prompt_tokens = max(0, prefix_len - params["num_transition_steps"])
    skip = zero_shot_prompt_tokens + params["num_transition_steps"] - 1
    assert 0 < skip < features.shape[0], "skip must trim on the tiny fixture"
    features = features[skip:]
    times = times[skip:]

    traces = {
        "prompt_ids": [int(t) for t in input_ids],
        "prefix_len": prefix_len,
        "params": params,
        "draws": [{"shape": shape, "values": values} for shape, values in draws],
        "feature_frames": int(features.shape[0]),
        "features": floats(features),
        "time_before": [int(t) for t in times],
    }

    case_dir = OUT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    stored = {
        key: value.astype(mx.float16)
        for key, value in dict(tree_flatten(model.parameters())).items()
    }
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces))

    python_replay_check(case_dir, traces)
    print(f"{case_name}: {features.shape[0]} frames, {len(draws)} draws, replay ok")


def reload_model_from(case_dir: Path):
    """Rebuilds the model from exactly the saved safetensors (f16
    mantissas, f32 arithmetic like the recording run)."""
    model = tada.Model(tada.ModelConfig.from_dict(dict(TINY_CONFIG)))
    weights = mx.load(str(case_dir / "model.safetensors"))
    model.update(
        tree_unflatten(
            {
                key: mx.array(value).astype(mx.float32)
                for key, value in weights.items()
            }
        )
    )
    model._tokenizer = TinyTokenizer()
    return model


def python_replay_check(case_dir: Path, traces: dict) -> None:
    """The recording caveat gate: a fresh model rebuilt from the saved
    weights plus the recorded draw stream must reproduce the recorded
    features before anything freezes."""
    import mlx_audio.tts.models.tada.gray_code as gray

    model = reload_model_from(case_dir)
    params = traces["params"]
    recorded = []
    original_solve = model._solve_flow_matching

    def recording_solve(speech, cond, neg_cond, num_steps=20, **kwargs):
        out = original_solve(speech, cond, neg_cond, num_steps=num_steps, **kwargs)
        recorded.append(np.asarray(out, dtype=np.float32).ravel().tolist())
        return out

    model._solve_flow_matching = recording_solve
    with DrawReplayer(traces["draws"]):
        list(
            model.generate(
                PROMPT_TEXT,
                acoustic_cfg_scale=params["acoustic_cfg_scale"],
                duration_cfg_scale=params["duration_cfg_scale"],
                cfg_schedule=params["cfg_schedule"],
                noise_temperature=params["noise_temperature"],
                num_flow_matching_steps=params["num_flow_matching_steps"],
                time_schedule=params["time_schedule"],
                num_transition_steps=params["num_transition_steps"],
                max_tokens=params["max_tokens"],
                verbose=False,
            )
        )
    assert len(recorded) >= len(traces["draws"]), "replay produced fewer steps"

    # Re-derive the port-side outputs from the replayed solves and
    # compare against the frozen traces.
    input_ids, prefix_len = build_input_ids(model, params)
    shift = model.config.shift_acoustic
    num_steps = min(len(input_ids), params["max_tokens"])
    collected = []
    times = []
    for step in range(shift, num_steps):
        collected.append(recorded[step][: model.config.acoustic_dim])
        bits = model.num_time_bits
        tail = recorded[step][model.config.acoustic_dim:]
        times.append(int(gray.gray_code_to_int(tail[:bits])))
    times.append(times[-1])
    features = np.array(collected, dtype=np.float32)
    features = features * model.config.acoustic_std + model.config.acoustic_mean
    zero_shot_prompt_tokens = max(0, prefix_len - params["num_transition_steps"])
    skip = zero_shot_prompt_tokens + params["num_transition_steps"] - 1
    features = features[skip:]
    times = times[skip:]
    want = np.array(traces["features"], dtype=np.float32).reshape(features.shape)
    diff = float(np.abs(features - want).max())
    assert diff <= REPLAY_GATE, f"python replay mismatch {diff} in {case_dir.name}"
    assert times == traces["time_before"], "python replay time mismatch"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_case(11, "generate")
    build_case(13, "nocfg")

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_tada_fixtures.py",
        "generate_seed": 11,
        "nocfg_seed": 13,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
