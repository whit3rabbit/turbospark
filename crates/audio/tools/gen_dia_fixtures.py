#!/usr/bin/env python3
"""Regenerate the Dia family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/dia plus the mlx_audio.lm sampler subset and the
Descript Audio Codec, mlx 0.32.3). The true reference modules run in
full: the DiaModel encoder/decoder, `_prepare_text_input` byte
tokenization, the classifier-free-guidance decode loop with the delay
pattern, the post-EOS countdown bookkeeping, `codebook_to_audio`, and
the DAC decode, with:

- a seeded tiny two-layer DiaModel (encoder hidden 32 with 4 heads,
  decoder hidden 32 with 8 GQA query heads over 2 kv heads, 4 cross
  heads, vocab 1028, nine channels, delay [0, 8, ..., 15], text_length
  128);
- a seeded tiny nine-codebook DAC (codebook 1024 x 4, hop 8, 44.1 kHz)
  standing in for the downloadable 44 kHz checkpoint;
- all weights round-tripped through f16 so the traces come from exactly
  the values the Rust fixture loads;
- greedy decoding (temperature 0) throughout, with a scripted EOS
  override injected at the `_sample_next_token` boundary where the case
  needs a specific EOS step (a seeded tiny model cannot be relied on to
  emit EOS naturally at a usable step); the natural argmax is recorded
  at every step so the Rust test still verifies its own selection math
  before applying the same override;
- no network access anywhere (`Model.__init__`'s
  `DAC.from_pretrained` is patched to the local tiny DAC).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_dia_fixtures.py

Outputs into crates/audio/testdata/tts_dia/:

- text/      the byte tokenizer (S1/S2 replacement, pad to
             text_length, padding mask) and `_split_turns` on a
             three-pair script.
- delay/     `apply_audio_delay` / `revert_audio_delay` around trips
             and `codebook_to_audio` with invalid codes clamped, over a
             synthetic code grid through the tiny DAC.
- generate/  greedy end to end: 51 steps with the EOS override at step
             20, final code grid, DAC waveform, token count.
- budget/    natural argmax throughout, budget stop at 45 steps, code
             grid and waveform.
- midstop/   EOS at step 30 with the budget stop at 45: the countdown
             adds land inside the sliced grid and ride the clamp into
             the decode.
- voice/     reference-audio prompt: DAC-encoded delayed prompt codes,
             the prefill, 45 further steps, code grid and waveform.
- sampling/  fixed step-0 CFG logits through the real sampler chain
             (temperature 1.3, top-p 0.95, top-k 35): filtered logits
             and drawn tokens, plus the temperature-0 argmax.
- manifest.json  file list with SHA-256 digests and generator metadata.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.lm.sample_utils import apply_top_k, apply_top_p, make_sampler
from mlx_audio.tts.models.dia import audio as dia_audio
from mlx_audio.tts.models.dia import dia as dia_module
from mlx_audio.codec.models.descript import DAC

OUT = Path("../turbospark/crates/audio/testdata/tts_dia")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

TINY_CONFIG = {
    "version": "1.0",
    "model": {
        "encoder": {
            "n_layer": 2,
            "n_embd": 32,
            "n_hidden": 64,
            "n_head": 4,
            "head_dim": 8,
            "mlp_activations": ["silu", "linear"],
            "use_pre_norm": False,
        },
        "decoder": {
            "n_layer": 2,
            "n_embd": 32,
            "n_hidden": 64,
            "gqa_query_heads": 8,
            "kv_heads": 2,
            "gqa_head_dim": 8,
            "cross_query_heads": 4,
            "cross_head_dim": 8,
            "mlp_activations": ["silu", "linear"],
            "use_pre_norm": False,
        },
        "src_vocab_size": 128,
        "tgt_vocab_size": 1028,
        "dropout": 0.0,
        "normalization_layer_epsilon": 1.0e-5,
        "weight_dtype": "float32",
        "rope_min_timescale": 1,
        "rope_max_timescale": 10000,
        "sample_rate": 44100,
    },
    "training": {"dtype": "float32", "logits_dot_in_fp32": False},
    "data": {
        "text_length": 128,
        "audio_length": 128,
        "channels": 9,
        "text_pad_value": 0,
        "audio_eos_value": 1024,
        "audio_pad_value": 1025,
        "audio_bos_value": 1026,
        "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15],
    },
}

DAC_CONFIG = {
    "encoder_dim": 8,
    "encoder_rates": [2, 4],
    "latent_dim": None,
    "decoder_dim": 16,
    "decoder_rates": [4, 2],
    "n_codebooks": 9,
    "codebook_size": 1024,
    "codebook_dim": 4,
    "sample_rate": 44100,
}

SEGMENT_TEXT = "[S1] Hello there. [S2] Hi back, friend!"
SPLIT_TEXT = "[S1] a alpha [S2] b beta [S1] c gamma [S2] d delta [S1] e eps [S2] f foxtrot"
REF_TEXT = "hello there"
MAX_TOKENS = 60
BUDGET_TOKENS = 45
VOICE_TOKENS = 70
VOICE_EOS_STEP = 30
VOICE_CURRENT_STEP = 8  # the BOS row plus the 8 prompt frames
EOS_OVERRIDE_STEP = 20


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return np.asarray(arr).ravel().astype(int).tolist()


def nested(arr) -> list:
    """Keeps the array shape (e.g. a [C][T] code grid)."""
    return np.asarray(arr).astype(int).tolist()


def save_npy(path: Path, arr) -> None:
    np.save(path, np.ascontiguousarray(arr, dtype=np.float32))


def seed_model(model, seed: int) -> None:
    rng = np.random.default_rng(seed)
    flat = dict(tree_flatten(model.parameters()))
    filled = {}
    for name, value in flat.items():
        if name.endswith("norm.weight"):
            filled[name] = mx.array(
                rng.uniform(0.5, 1.5, value.shape).astype(np.float32)
            )
        else:
            filled[name] = mx.array(
                rng.uniform(-0.35, 0.35, value.shape).astype(np.float32)
            )
    model.load_weights(list(filled.items()), strict=True)


def round_trip_f16(model) -> None:
    flat = dict(tree_flatten(model.parameters()))
    f16 = {key: value.astype(mx.float16).astype(mx.float32) for key, value in flat.items()}
    model.load_weights(list(f16.items()), strict=True)


def build_dac(seed: int) -> DAC:
    mx.random.seed(seed)
    dac = DAC(**DAC_CONFIG)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, dac.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith(".alpha"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    stored = {name: value.astype(mx.float16) for name, value in flat.items()}
    dac.load_weights([(k, v.astype(mx.float32)) for k, v in stored.items()], strict=True)
    mx.eval(dac)
    return dac, stored


class StepRecorder:
    """Wraps `dia._sample_next_token`, recording the natural selection
    and applying an optional override. `override(index, natural)`
    returns the full pred list to use, or None to keep the natural
    choice."""

    def __init__(self, override):
        self.steps = []
        self.override = override
        self._original = dia_module._sample_next_token

    def __enter__(self):
        recorder = self

        def patched(logits_BCxV, temperature, sampler):
            natural = recorder._original(logits_BCxV, temperature, sampler)
            natural_list = ints(natural)
            pred = natural
            if recorder.override is not None:
                overridden = recorder.override(len(recorder.steps), natural_list)
                if overridden is not None:
                    pred = mx.array(np.asarray(overridden, dtype=np.int32))
            recorder.steps.append(
                {"natural": natural_list, "pred": ints(pred)}
            )
            return pred

        dia_module._sample_next_token = patched
        return self

    def __exit__(self, *exc):
        dia_module._sample_next_token = self._original
        return False


def eos_at(step, channel, value, offset=0):
    """Override rule: force `value` on `channel` at absolute decode
    step `step` (recorder index = absolute step - offset)."""
    def rule(index, natural):
        if index + offset == step:
            pred = list(natural)
            pred[channel] = value
            return pred
        return None
    return rule


def suppress_eos(channel, replacement):
    """Override rule: keep EOS off `channel` entirely."""
    def rule(index, natural):
        del index
        if natural[channel] == 1024:
            pred = list(natural)
            pred[channel] = replacement
            return pred
        return None
    return rule


class GridRecorder:
    """Wraps `dia.codebook_to_audio`, recording the sliced code grid."""

    def __init__(self):
        self.grids = []
        self._original = dia_module.codebook_to_audio

    def __enter__(self):
        recorder = self

        def patched(generated_codes, model, delay_pattern, B=1, T=2600, C=9):
            recorder.grids.append(nested(generated_codes))
            return recorder._original(generated_codes, model, delay_pattern, B=B, T=T, C=C)

        dia_module.codebook_to_audio = patched
        return self

    def __exit__(self, *exc):
        dia_module.codebook_to_audio = self._original
        return False


def natural_steps(traces, current_step):
    """How many recorded steps have clean (in-vocabulary) decoder
    inputs. From the first post-EOS add onward the reference embeds
    out-of-vocabulary ids whose rows are undefined (MLX gather past the
    buffer), so parity is pinned by replay there instead."""
    override = traces.get("override") or {}
    if "step" not in override:
        return len(traces["steps"])
    detect = override["step"]
    first_add = detect + 15
    if len(traces["steps"]) <= first_add:
        return len(traces["steps"])
    return first_add - current_step + 1


def write_case(name: str, config: dict, weights: dict, traces: dict, dac, dac_weights: dict) -> None:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), weights)
    mx.save_safetensors(str(case_dir / "dac.safetensors"), dac_weights)
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (case_dir / "dac_config.json").write_text(json.dumps(DAC_CONFIG, indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dac, dac_weights = build_dac(41)

    # The Dia Model constructor downloads the DAC checkpoint; patch the
    # module-level DAC symbol to hand back the local tiny codec.
    class LocalDac:
        @classmethod
        def from_pretrained(cls, name):
            del name
            return dac

    dia_module.DAC = LocalDac

    mx.random.seed(700)
    model = dia_module.Model(dict(TINY_CONFIG))
    seed_model(model, 3)
    round_trip_f16(model)
    weights = {
        name: value.astype(mx.float16)
        for name, value in dict(tree_flatten(model.parameters())).items()
    }

    # --- text case ----------------------------------------------------
    src_tokens, src_positions, src_padding_mask, _ = model._prepare_text_input(SEGMENT_TEXT)
    traces_text = {
        "script": SEGMENT_TEXT,
        "text_length": int(TINY_CONFIG["data"]["text_length"]),
        "src_tokens": ints(src_tokens),
        "src_positions": ints(src_positions),
        "padding_mask": [bool(v) for v in np.asarray(src_padding_mask).ravel()],
        "split_script": SPLIT_TEXT,
        "split_turns": model._split_turns(SPLIT_TEXT),
    }
    write_case("text", TINY_CONFIG, weights, traces_text, dac, dac_weights)

    # --- delay case ---------------------------------------------------
    rng = np.random.default_rng(9)
    grid = rng.integers(0, 64, size=(40, 9)).astype(np.int32)
    grid[2, 0] = 1024
    grid[3, 4] = 1026
    grid[4, 8] = -5
    grid[5, 1] = 2050
    delay = TINY_CONFIG["data"]["delay_pattern"]
    precomp = dia_audio.build_delay_indices(1, grid.shape[0], 9, delay)
    delayed = dia_audio.apply_audio_delay(
        mx.array(grid)[None],
        pad_value=TINY_CONFIG["data"]["audio_pad_value"],
        bos_value=TINY_CONFIG["data"]["audio_bos_value"],
        precomp=precomp,
    )
    revert_precomp = dia_audio.build_revert_indices(1, grid.shape[0], 9, delay)
    reverted = dia_audio.revert_audio_delay(
        mx.array(grid)[None],
        pad_value=0,
        precomp=revert_precomp,
        T=grid.shape[0],
    )
    audio = dia_audio.codebook_to_audio(mx.array(grid).T, dac, delay, B=1, T=64, C=9)
    traces_delay = {
        "input_grid": nested(grid),
        "delayed_grid": nested(delayed[0]),
        "reverted_grid": nested(reverted[0]),
        "codebook_input": nested(grid.T),
        "waveform": floats(audio),
        "max_tokens": 64,
    }
    write_case("delay", TINY_CONFIG, weights, traces_delay, dac, dac_weights)

    # --- generate case (scripted EOS at step 20) ----------------------
    recorder = StepRecorder(eos_at(EOS_OVERRIDE_STEP, 0, 1024))
    grid_rec = GridRecorder()
    with recorder, grid_rec:
        audio, token_count = model._generate(
            SEGMENT_TEXT,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
    traces_generate = {
        "prompt": SEGMENT_TEXT,
        "max_tokens": MAX_TOKENS,
        "override": {"step": EOS_OVERRIDE_STEP, "channel": 0, "value": 1024},
        "cfg_scale": 3.0,
        "temperature": 0.0,
        "top_p": 0.95,
        "use_cfg_filter": True,
        "cfg_filter_top_k": 35,
        "steps": recorder.steps,
        "codes": grid_rec.grids[0],
        "waveform": floats(audio),
        "token_count": int(token_count),
    }
    traces_generate["natural_steps"] = natural_steps(traces_generate, 0)
    write_case("generate", TINY_CONFIG, weights, traces_generate, dac, dac_weights)

    # --- budget case (natural argmax, EOS suppressed on channel 0) ----
    recorder = StepRecorder(suppress_eos(0, 1023))
    grid_rec = GridRecorder()
    with recorder, grid_rec:
        audio, token_count = model._generate(
            SEGMENT_TEXT,
            max_tokens=BUDGET_TOKENS,
            temperature=0.0,
        )
    traces_budget = {
        "prompt": SEGMENT_TEXT,
        "max_tokens": BUDGET_TOKENS,
        "override": {"suppress_eos_channel": 0, "replacement": 1023},
        "steps": recorder.steps,
        "codes": grid_rec.grids[0],
        "waveform": floats(audio),
        "token_count": int(token_count),
    }
    traces_budget["natural_steps"] = natural_steps(traces_budget, 0)
    write_case("budget", TINY_CONFIG, weights, traces_budget, dac, dac_weights)

    # --- midstop case (EOS at step 30, budget exit at 45) -------------
    recorder = StepRecorder(eos_at(30, 0, 1024))
    grid_rec = GridRecorder()
    with recorder, grid_rec:
        audio, token_count = model._generate(
            SEGMENT_TEXT, max_tokens=BUDGET_TOKENS, temperature=0.0
        )
    traces_midstop = {
        "prompt": SEGMENT_TEXT,
        "max_tokens": BUDGET_TOKENS,
        "override": {"step": 30, "channel": 0, "value": 1024},
        "steps": recorder.steps,
        "codes": grid_rec.grids[0],
        "waveform": floats(audio),
        "token_count": int(token_count),
    }
    traces_midstop["natural_steps"] = natural_steps(traces_midstop, 0)
    write_case("midstop", TINY_CONFIG, weights, traces_midstop, dac, dac_weights)

    # --- voice case (reference-audio prompt prefill) ------------------
    rng = np.random.default_rng(23)
    ref_audio = (rng.standard_normal(64) * 0.1).astype(np.float32)
    prompt_codes = dia_audio.audio_to_codebook(
        dac,
        mx.array(ref_audio)[None, None, :],
        dia_module.DiaConfig.load_dict(dict(TINY_CONFIG)).data,
    )
    # The self-attention cache is sized max_tokens and the prefill
    # consumes its rows, so the voice path must leave the loop through
    # the EOS countdown before the budget: EOS at step 30 breaks at 60.
    # The voice loop starts at the prefill length, so the override
    # target is expressed as an absolute decode step.
    recorder = StepRecorder(eos_at(VOICE_EOS_STEP, 0, 1024, offset=VOICE_CURRENT_STEP))
    grid_rec = GridRecorder()
    with recorder, grid_rec:
        audio, token_count = model._generate(
            SEGMENT_TEXT,
            temperature=0.0,
            max_tokens=VOICE_TOKENS,
            ref_audio=mx.array(ref_audio),
            ref_text=REF_TEXT,
        )
    traces_voice = {
        "prompt": SEGMENT_TEXT,
        "ref_text": REF_TEXT,
        "ref_audio": floats(ref_audio),
        "prompt_codes": nested(prompt_codes[0]),
        "max_tokens": VOICE_TOKENS,
        "override": {"step": VOICE_EOS_STEP, "channel": 0, "value": 1024, "offset": VOICE_CURRENT_STEP},
        "steps": recorder.steps,
        "codes": grid_rec.grids[0],
        "waveform": floats(audio),
        "token_count": int(token_count),
    }
    traces_voice["natural_steps"] = natural_steps(traces_voice, 8)
    write_case("voice", TINY_CONFIG, weights, traces_voice, dac, dac_weights)

    # --- sampling case (fixed logits, real sampler chain) -------------
    # One decode step at temperature 0 supplies the step-0 CFG logits;
    # the capture aborts the run so the sampler chain itself is driven
    # directly below.
    cfg = None

    class CaptureDone(Exception):
        pass

    def capture(logits_BCxV, temperature, sampler):
        nonlocal cfg
        cfg = np.asarray(logits_BCxV, dtype=np.float32).copy()
        raise CaptureDone()

    original_sample = dia_module._sample_next_token
    dia_module._sample_next_token = capture
    try:
        model._generate(SEGMENT_TEXT, max_tokens=4, temperature=0.0)
    except CaptureDone:
        pass
    finally:
        dia_module._sample_next_token = original_sample
    assert cfg is not None
    sampler = make_sampler(1.3, 0.95, top_k=35)
    drawn = np.asarray(sampler(mx.array(cfg)), dtype=np.int64).ravel().tolist()
    filtered_p = apply_top_p(mx.array(cfg), 0.95)
    filtered_pk = apply_top_k(filtered_p, 35)
    traces_sampling = {
        "temperature": 1.3,
        "top_p": 0.95,
        "top_k": 35,
        "drawn": drawn,
        "argmax": ints(original_sample(mx.array(cfg), 0.0, None)),
    }
    (OUT / "sampling").mkdir(parents=True, exist_ok=True)
    save_npy(OUT / "sampling" / "logits.npy", cfg)
    save_npy(OUT / "sampling" / "filtered_top_p.npy", filtered_p)
    save_npy(OUT / "sampling" / "filtered_top_p_top_k.npy", filtered_pk)
    (OUT / "sampling" / "traces.json").write_text(json.dumps(traces_sampling))

    # --- manifest ------------------------------------------------------
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_dia_fixtures.py",
        "seeds": {"dia": 3, "dac": 41, "case_rng": [9, 23, 700]},
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
