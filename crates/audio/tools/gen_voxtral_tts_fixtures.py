#!/usr/bin/env python3
"""Regenerate the voxtral_tts (Voxtral TTS 4B) family Rust parity
fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/voxtral_tts, mlx 0.32.3). The reference modules run
in full: the Mistral-geometry LM backbone (mlx_audio.lm llama with
traditional/interleaved rope, tied embeddings, no biases), the voice
embedding replacement over the [AUDIO] prompt positions, the flow
matching acoustic head (masked semantic argmax over the padded head, the
8-step Euler CFG loop over the 3-token bidirectional transformer), the
per-codebook-offset audio embedding feedback, and the codec decoder (EMA
semantic codebook, FSQ acoustic quantizer, weight-normed causal and
transposed convs with replicate/reflect pads, ALiBi sliding-window
transformer stages, patch output projection), with:

- a stubbed speech tokenizer (`model._encode_text` returns the canned
  prompt ids; the real path needs mistral-common's tekken speech
  tokenizer), the voice table loaded from the case's
  `voice_embedding/casual_male.safetensors` through the reference
  `_get_voice_embedding`, and `model.tokenizer` set to a stub so the
  reference `generate` entry point runs;
- seeded tiny stacks: a 2-layer LM at hidden 32 (4 query / 2 kv heads,
  head_dim 8, vocab 128), a 2-layer acoustic head at dim 32, and a
  4-stage codec decoder (strides 1,2,2,2, kernels 3,4,4,4) at dim 32
  with a 12-sample patch (96 samples per frame);
- the checkpoint saved under the raw consolidated Mistral names
  (`tok_embeddings`, `layers.N.attention.wq`, `attention_norm`, the
  `mm_audio_embeddings.` prefix on the shared audio table), exercising
  the Rust loader's name remap;
- every `mx.random.normal` draw (one per generated frame) recorded and
  replay-verified against a fresh `model.generate` and a second replay.

Stop control: every semantic head row except the nominal code (5) and
the end token (1) is shrunk, and the end row is `nominal + eps * v`.
The LM hidden trajectory is fixed by the seed, the weights, and the
recorded draws (it does not depend on `v` while no stop fires), so `v`
is chosen offline against the recorded hidden states: the `generate`
case flips the argmax to the end token exactly at step 3 (3 frames), and
the `length` case never flips (runs to max_tokens). A full re-run after
choosing `v` asserts the trajectory, stop step, and waveform are
unchanged from the pre-choice run.

Per-step traces come from a verbatim replay of the reference generate
body (the loop has no hooks); the replay's waveform and draw sequence
are asserted identical to a fresh `model.generate` call, and the replay
is itself re-run a second time to pin reproducibility.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_voxtral_tts_fixtures.py

Outputs into crates/audio/testdata/tts_voxtral_tts/:

- generate/ (voice-conditioned, end token fires at step 3),
  length/ (voice-conditioned, runs to max_tokens),
  codec/ (standalone codec decode of a random code grid)
    Each: config.json + model.safetensors + traces.json, plus
    voice_embedding/casual_male.safetensors for the first two.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

sys.path.insert(0, str(Path.cwd()))

from mlx_audio.lm.models.cache import make_prompt_cache
from mlx_audio.tts.models.voxtral_tts.voxtral_tts import Model

OUT = Path("../turbospark/crates/audio/testdata/tts_voxtral_tts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

WEIGHT_SEED = 7
DRAW_SEED = 101
MIN_FSQ_MARGIN = 5e-3
VOICE = "casual_male"
PROMPT_IDS = [4, 5, 6, 24, 24, 24, 25]
NOMINAL_CODE = 5
END_CODE = 1
EPS = 0.05
N_STEPS = 8
CFG_ALPHA = 1.2
WINDOW_SIZES = [2, 4, 8, 16]

CONFIG = {
    "model_type": "voxtral_tts",
    # LM backbone.
    "dim": 32,
    "n_layers": 2,
    "head_dim": 8,
    "hidden_dim": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "vocab_size": 128,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "tied_embeddings": True,
    "use_biases": False,
    "max_position_embeddings": 128,
    "sample_rate": 24000,
    # Audio model.
    "semantic_codebook_size": 32,
    "acoustic_codebook_size": 5,
    "n_acoustic_codebook": 6,
    "frame_rate": 12.5,
    "num_codebooks": 7,
    # Acoustic transformer (flow matching head).
    "acoustic_dim": 32,
    "acoustic_n_layers": 2,
    "acoustic_head_dim": 8,
    "acoustic_hidden_dim": 64,
    "acoustic_n_heads": 4,
    "acoustic_n_kv_heads": 2,
    "acoustic_rope_theta": 10000.0,
    "acoustic_sigma": 1e-5,
    "acoustic_sigma_max": 1.0,
    # Audio tokenizer (codec decoder).
    "tokenizer_dim": 32,
    "tokenizer_hidden_dim": 64,
    "tokenizer_n_heads": 4,
    "tokenizer_n_kv_heads": 2,
    "tokenizer_head_dim": 8,
    "tokenizer_patch_size": 12,
    "tokenizer_patch_proj_kernel_size": 7,
    "tokenizer_semantic_dim": 8,
    "tokenizer_acoustic_dim": 6,
    "tokenizer_norm_eps": 0.01,
    "tokenizer_decoder_transformer_lengths_str": "2,2,1,1",
    "tokenizer_decoder_convs_kernels_str": "3,4,4,4",
    "tokenizer_decoder_convs_strides_str": "1,2,2,2",
    # Special tokens.
    "bos_token_id": 1,
    "audio_token_id": 24,
    "begin_audio_token_id": 25,
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


def seeded_model(seed: int) -> Model:
    """Deterministic weights: small uniform everywhere, norm gains biased
    up, the semantic codebook usages positive. The constructor-derived
    buffers (none are registered parameters in this build) stay put."""
    return _build_model(seed)


def _build_model(seed: int) -> Model:
    from mlx_audio.tts.models.voxtral_tts.voxtral_tts import ModelConfig

    mx.random.seed(seed)
    model = Model(ModelConfig.from_dict(json.loads(json.dumps(CONFIG))))

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith("norm.weight"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    usage_key = "audio_tokenizer.quantizer.semantic_codebook.cluster_usage"
    flat[usage_key] = mx.random.uniform(low=0.5, high=1.5, shape=flat[usage_key].shape)
    model.update(tree_unflatten(flat))
    return model


def pristine_semantic_row(model: Model) -> np.ndarray:
    """The refilled semantic head weight, before any race shaping."""
    flat = dict(tree_flatten(model.parameters()))
    key = "acoustic_transformer.semantic_codebook_output.weight"
    return np.array(flat[key], dtype=np.float32)


def install_race_rows(model: Model, pristine: np.ndarray, direction: np.ndarray) -> None:
    """End row = nominal row + eps * vector."""
    flat = dict(tree_flatten(model.parameters()))
    key = "acoustic_transformer.semantic_codebook_output.weight"
    sem = pristine.copy()
    for row in range(sem.shape[0]):
        if row not in (NOMINAL_CODE, END_CODE):
            sem[row] *= 0.05
    sem[NOMINAL_CODE] = 30.0 * direction
    flat[key] = mx.array(sem)
    model.update(tree_unflatten(flat))


def set_end_row(model: Model, vector: np.ndarray, eps: float) -> None:
    """End row = nominal row + eps * vector."""
    flat = dict(tree_flatten(model.parameters()))
    key = "acoustic_transformer.semantic_codebook_output.weight"
    sem = np.array(flat[key], dtype=np.float32)
    sem[END_CODE] = sem[NOMINAL_CODE] + eps * vector
    flat[key] = mx.array(sem)
    model.update(tree_unflatten(flat))


def quantization_margin(model: Model, sink: list, draw_seed: int) -> float:
    """The smallest distance of any frame's final Euler state to an FSQ
    quantization boundary (the pre-round value vs the half-integers the
    rounding flips at). f32 noise below the margin cannot flip a code."""
    size = model.config.acoustic_codebook_size
    worst = float("inf")
    for step in sink:
        trace, _ = frame_flow_trace(model, step["hidden"], draw_seed, step["index"])
        x = np.asarray(trace["x_t_final"], dtype=np.float32)
        scaled = (np.clip(x, -1.0, 1.0) + 1.0) * (size - 1) / 2.0
        distances = np.abs(scaled - np.floor(scaled) - 0.5)
        worst = min(worst, float(distances.min()))
    return worst


def choose_stop_vector(
    model: Model,
    max_tokens: int,
    stop_step: int | None,
    draw_seed: int,
) -> np.ndarray:
    """Shapes the semantic head into a two-way race and finds the
    end-row direction with the wanted stop behavior.

    The nominal row is pointed along the running hidden trajectory until
    it provably wins every checked step (the trajectory is then the
    all-nominal one), after which the end row is `nominal + eps * v` and
    the argmax flips exactly when `dot(h, v)` turns positive. Candidates
    whose decision margin is under 0.02 are rejected so f32 noise cannot
    move the stop step.

    stop_step = 3 -> the argmax flips to the end token exactly at step 3.
    stop_step = None -> the end token never wins within max_tokens.
    """
    rng = np.random.default_rng(1234)
    dim = model.config.dim
    needed = max_tokens if stop_step is None else stop_step + 1
    pristine = pristine_semantic_row(model)

    direction = pristine[NOMINAL_CODE].copy()
    hiddens = None
    for _ in range(32):
        install_race_rows(model, pristine, direction)
        # Probe end rows strictly below the nominal row: a random
        # direction, small negative step. A probe whose end token wins
        # early is skipped.
        for probe in range(8):
            probe_vector = rng.standard_normal(dim).astype(np.float32)
            probe_vector /= np.linalg.norm(probe_vector)
            set_end_row(model, probe_vector, -EPS)
            try:
                sink, _, _ = run_replay(model, max_tokens, capture=True, draw_seed=draw_seed)
            except RuntimeError:
                # The end token won at step 0; try the next probe.
                continue
            if len(sink) >= needed and all(
                step["semantic"] == NOMINAL_CODE for step in sink[:needed]
            ):
                hiddens = np.stack(
                    [
                        np.asarray(step["hidden"], dtype=np.float32)
                        for step in sink[:needed]
                    ]
                )
                break
        if hiddens is not None:
            break
        mean = np.sum(
            [np.asarray(step["hidden"], dtype=np.float32) for step in sink[:needed]],
            axis=0,
        )
        norm = np.linalg.norm(mean)
        if norm == 0:
            break
        direction = (mean / norm).astype(np.float32)
    if hiddens is None:
        raise RuntimeError("the nominal row never stabilized")

    for attempt in range(100000):
        vector = rng.standard_normal(dim).astype(np.float32)
        vector /= np.linalg.norm(vector)
        signs = hiddens @ vector
        if np.min(np.abs(signs)) < 0.02:
            continue
        if stop_step is None:
            if np.all(signs < 0):
                return vector
        else:
            if np.all(signs[:stop_step] < 0) and signs[stop_step] > 0:
                return vector
    raise RuntimeError("no stop vector found")


def attach_voice(model: Model, voice_path: Path) -> None:
    """The reference voice loading: generate reads
    `voice_embedding/<name>.safetensors` through `_get_voice_embedding`.
    The tekken speech tokenizer (mistral-common) is stubbed: the canned
    prompt ids stand in for `encode_speech_request`."""
    model._voice_embedding_files = {VOICE: voice_path}
    model._voice_embeddings = {}
    model.tokenizer = object()
    model._encode_text = lambda text, voice_name: list(PROMPT_IDS)


def replay_generate(model: Model, max_tokens: int, capture: bool):
    """Verbatim replay of the reference `Model.generate` body with
    per-step capture. Returns (sink, all_codes, waveform)."""
    input_ids_mx = mx.array(PROMPT_IDS)[None, :]
    input_embeddings = model._build_input_embeddings(input_ids_mx, VOICE)
    lm_backbone = model.language_model.model.model
    cache = make_prompt_cache(model.language_model.model)
    hidden = lm_backbone(input_ids_mx, cache=cache, input_embeddings=input_embeddings)

    all_codes = []
    sink = []
    for i in range(max_tokens):
        h = hidden[:, -1, :]
        codes = model.acoustic_transformer.decode_one_frame(h)
        semantic_code = int(codes[0, 0].item())
        if capture:
            sink.append(
                {
                    "index": i,
                    "hidden": floats(h),
                    "codes": ints(codes),
                    "semantic": semantic_code,
                }
            )
        if semantic_code <= 1:
            break
        all_codes.append(codes[:, None, :])
        global_codes = model._codes_to_global_indices(codes)
        code_embeddings = model.audio_codebook_embeddings["embeddings"](global_codes)
        next_embedding = code_embeddings.sum(axis=1, keepdims=True)
        dummy_input = mx.array([[model.config.audio_token_id]])
        hidden = lm_backbone(dummy_input, cache=cache, input_embeddings=next_embedding)

    if not all_codes:
        raise RuntimeError("No audio frames generated")
    audio_codes = mx.concatenate(all_codes, axis=1)
    waveform = model.audio_tokenizer.decode(audio_codes).squeeze(0)
    return sink, all_codes, waveform


def run_replay(model: Model, max_tokens: int, capture: bool = True, draw_seed: int = DRAW_SEED):
    """One replay under a fresh draw seed. Returns (sink, waveform,
    draws)."""
    with DrawRecorder() as recorder:
        mx.random.seed(draw_seed)
        sink, all_codes, waveform = replay_generate(model, max_tokens, capture)
        draws = recorder.flush()
    return sink, np.array(waveform, dtype=np.float32), draws


def frame_flow_trace(model: Model, hidden: list, draw_seed: int, index: int) -> dict:
    """The flow internals of frame `index`, aligned with the recorded
    draw stream (the seed replays, `index` draws are consumed, and the
    next draw is the frame's own)."""
    n_acoustic = model.config.n_acoustic_codebook
    with DrawRecorder() as recorder:
        mx.random.seed(draw_seed)
        for _ in range(index):
            mx.random.normal((1, n_acoustic))
        trace = flow_stage_trace(model, hidden)
        draws = recorder.flush()
    return trace, draws


def flow_stage_trace(model: Model, hidden: list) -> dict:
    """Replays the flow head for one frame with its internals exposed
    (same seeded draw as the loop's frame at the same step)."""
    head = model.acoustic_transformer
    args = head.args
    h = mx.array(np.asarray(hidden, dtype=np.float32))[None, :]
    B = h.shape[0]
    x_t = mx.random.normal((B, args.n_acoustic_codebook)) * args.sigma_max
    n_steps = args.n_denoising_steps
    timesteps = [i / (n_steps - 1) for i in range(n_steps)]
    llm_uncond = mx.zeros_like(h)
    steps = []
    for step in range(n_steps - 1):
        t_val = timesteps[step]
        dt = timesteps[step + 1] - t_val
        t = mx.full((B,), t_val)
        x_t_batch = mx.concatenate([x_t, x_t], axis=0)
        t_batch = mx.concatenate([t, t], axis=0)
        llm_batch = mx.concatenate([h, llm_uncond], axis=0)
        v_both = head._predict_velocity(x_t_batch, t_batch, llm_batch)
        v_cond, v_uncond = v_both[:B], v_both[B:]
        v = args.cfg_alpha * v_cond + (1.0 - args.cfg_alpha) * v_uncond
        x_t = x_t + v * dt
        steps.append(
            {
                "t": float(t_val),
                "dt": float(dt),
                "v_cond": floats(v_cond),
                "v_uncond": floats(v_uncond),
                "x_t": floats(x_t),
            }
        )
    quantized = (
        mx.clip(
            mx.round(
                (mx.clip(x_t, -1.0, 1.0) + 1.0) * (args.acoustic_codebook_size - 1) / 2.0
            ),
            0,
            args.acoustic_codebook_size - 1,
        ).astype(mx.int32)
        + 2
    )
    semantic = int(head._predict_semantic(h).item())
    return {
        "steps": steps,
        "x_t_final": floats(x_t),
        "codes": [semantic] + ints(quantized),
    }


def save_case(name: str, model: Model, traces: dict) -> None:
    case_dir = OUT / name
    (case_dir / "voice_embedding").mkdir(parents=True, exist_ok=True)

    # Raw consolidated Mistral names, the layout the reference sanitize
    # consumes.
    suffix_map = {
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }
    raw = {}
    for name, weight in tree_flatten(model.parameters()):
        if name.startswith("language_model.model.model."):
            short = name[len("language_model.model.model.") :]
            match = re.match(r"layers\.(\d+)\.(.*)", short)
            if match:
                layer_idx, suffix = match.groups()
                short = f"layers.{layer_idx}.{suffix_map.get(suffix, suffix)}"
            elif short == "embed_tokens.weight":
                short = "tok_embeddings.weight"
            raw[short] = weight
        elif name.startswith("audio_codebook_embeddings."):
            raw["mm_audio_embeddings." + name] = weight
        elif name == "language_model.model.lm_head.weight":
            continue  # tied
        else:
            raw[name] = weight
    mx.save_safetensors(str(case_dir / "model.safetensors"), raw)
    (case_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")


def voice_embedding(dim: int) -> np.ndarray:
    rng = np.random.default_rng(53)
    return rng.uniform(-0.5, 0.5, size=(3, dim)).astype(np.float32)


def run_generation_case(name: str, model: Model, max_tokens: int, stop_step: int | None) -> None:
    """Tunes the stop row, records the traces, and verifies the replay
    against a fresh `model.generate` and a second replay."""
    case_dir = OUT / name
    voice_dir = case_dir / "voice_embedding"
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice_path = voice_dir / f"{VOICE}.safetensors"
    embedding = voice_embedding(model.config.dim)
    mx.save_safetensors(str(voice_path), {"embedding": mx.array(embedding)})
    attach_voice(model, voice_path)

    # The Euler flow's f32 noise cannot be made to vanish, so the draw
    # seed is searched until every frame's final state sits well clear of
    # an FSQ quantization boundary (min margin MIN_FSQ_MARGIN); codes are
    # then exact under the Rust replay and the exactness gate means
    # something, not that the values happened to land near a boundary.
    min_margin = 0.0
    chosen_seed = None
    vector = None
    for candidate in (DRAW_SEED, 202, 303, 404, 505, 606, 707, 808):
        vector = choose_stop_vector(model, max_tokens, stop_step, candidate)
        set_end_row(model, vector, EPS)
        sink, waveform, draws = run_replay(model, max_tokens, capture=True, draw_seed=candidate)
        if stop_step is None:
            frames_ok = len(sink) == max_tokens and all(
                step["semantic"] == NOMINAL_CODE for step in sink
            )
        else:
            frames_ok = (
                len(sink) == stop_step + 1
                and all(step["semantic"] == NOMINAL_CODE for step in sink[:stop_step])
                and sink[stop_step]["semantic"] == END_CODE
            )
        if not frames_ok:
            continue
        min_margin = quantization_margin(model, sink, candidate)
        if min_margin >= MIN_FSQ_MARGIN:
            chosen_seed = candidate
            break
    if chosen_seed is None:
        raise RuntimeError(f"{name}: no draw seed cleared the FSQ margin")
    sink, waveform, noise_draws = run_replay(model, max_tokens, capture=True, draw_seed=chosen_seed)

    # Stage gate: the flow internals of frame 0 must reproduce the
    # loop's first codes under the same seeded draw.
    flow, flow_draws = frame_flow_trace(model, sink[0]["hidden"], chosen_seed, 0)
    assert flow_draws[0] == (
        noise_draws[0][0],
        noise_draws[0][1],
    ), "flow stage draw misaligned"
    assert flow["codes"] == sink[0]["codes"], "flow stage replay diverged"

    if stop_step is None:
        expected_semantics = [NOMINAL_CODE] * max_tokens
        frames = max_tokens
        assert len(sink) == max_tokens
    else:
        expected_semantics = [NOMINAL_CODE] * stop_step + [END_CODE]
        frames = stop_step
        assert len(sink) == stop_step + 1
    assert [step["semantic"] for step in sink] == expected_semantics, (
        f"{name}: semantics {[step['semantic'] for step in sink]}"
    )

    # ---- run 2: fresh reference generate ---------------------------
    with DrawRecorder() as recorder:
        mx.random.seed(chosen_seed)
        result = list(
            model.generate(
                text="ignored", voice=VOICE, max_tokens=max_tokens, verbose=False
            )
        )[0]
        generate_draws = recorder.flush()
    assert generate_draws == noise_draws, "generate draw replay diverged"
    assert np.array_equal(np.array(result.audio), waveform), "generate disagrees"

    # ---- run 3: fresh replay of run 1 (reproducibility pin) --------
    sink2, waveform2, replay_draws = run_replay(model, max_tokens, capture=True, draw_seed=chosen_seed)
    assert replay_draws == noise_draws, "second replay diverged"
    assert np.array_equal(waveform2, waveform), "replay waveform"
    for a, b in zip(sink, sink2):
        for key in a:
            if key == "index":
                continue
            assert a[key] == b[key], f"replay step {a['index']} field {key}"

    input_ids_mx = mx.array(PROMPT_IDS)[None, :]
    input_embeddings = np.array(
        model._build_input_embeddings(input_ids_mx, VOICE), dtype=np.float32
    )[0]

    traces = {
        "mode": name,
        "weight_seed": WEIGHT_SEED,
        "draw_seed": chosen_seed,
        "min_fsq_margin": min_margin,
        "generate": {
            "prompt_ids": PROMPT_IDS,
            "voice": VOICE,
            "max_tokens": max_tokens,
            "n_denoising_steps": N_STEPS,
            "cfg_alpha": CFG_ALPHA,
            "sigma_max": CONFIG["acoustic_sigma_max"],
        },
        "prompt": {
            "ids": PROMPT_IDS,
            "voice_embedding": floats(embedding),
            "input_embeddings": floats(input_embeddings),
            "rows": int(input_embeddings.shape[0]),
        },
        "end_row_vector": floats(vector),
        "noise_draws": [
            {"shape": shape, "values": values} for shape, values in noise_draws
        ],
        "steps": sink,
        "flow_stage": flow,
        "codes": ints(np.concatenate([np.asarray(s["codes"]) for s in sink[:-1]]))
        if stop_step is not None
        else ints(np.concatenate([np.asarray(s["codes"]) for s in sink])),
        "frames": frames,
        "stopped": stop_step is not None,
        "waveform": floats(waveform),
        "sample_rate": int(model.config.sample_rate),
    }
    save_case(name, model, traces)
    print(
        f"{name}: {frames} frames, {waveform.shape[0]} samples, "
        f"stopped={'end_token' if stop_step is not None else 'max_steps'}, "
        f"draw_seed={chosen_seed}, min_fsq_margin={min_margin:.4f}"
    )


def run_codec_case(name: str, model: Model) -> None:
    """Standalone codec decode of a random code grid, with the per-stage
    hidden states."""
    tok = model.audio_tokenizer
    rng = np.random.default_rng(11)
    frames = 8
    codes = np.concatenate(
        [
            rng.integers(2, 2 + CONFIG["semantic_codebook_size"], size=(1, frames, 1)),
            rng.integers(2, 2 + CONFIG["acoustic_codebook_size"], size=(1, frames, 6)),
        ],
        axis=-1,
    ).astype(np.int32)
    codes_mx = mx.array(codes)

    x = tok.quantizer.decode(codes_mx)
    stages = [("quantizer", x)]
    blocks = tok.decoder_blocks
    for i in range(0, len(blocks), 2):
        conv_block = blocks[i]
        xformer_block = blocks[i + 1]
        stage_idx = i // 2
        stride = tok._strides[stage_idx]
        x = conv_block.conv(x, stride=stride, transpose=stride > 1)
        stages.append(("conv", x))
        x = xformer_block(x, tok._alibi_slopes, window_size=WINDOW_SIZES[stage_idx])
        stages.append(("transformer", x))
    x = tok.output_proj.conv(x, stride=1, transpose=False)
    waveform = np.array(x.reshape(1, -1), dtype=np.float32)[0]

    # The reference decode entry point must agree with the instrumented
    # loop.
    reference = np.array(
        tok.decode(codes_mx).reshape(1, -1), dtype=np.float32
    )[0]
    assert np.array_equal(reference, waveform), "codec loop diverged"

    traces = {
        "mode": name,
        "weight_seed": WEIGHT_SEED,
        "codes": ints(codes),
        "frames": frames,
        "num_codebooks": 7,
        "stages": [
            {
                "stage": stage,
                "rows": int(np.asarray(values).shape[1]),
                "width": int(np.asarray(values).shape[2]),
                "values": floats(values),
            }
            for stage, values in stages
        ],
        "waveform": floats(waveform),
        "sample_rate": int(model.config.sample_rate),
    }
    save_case(name, model, traces)
    print(f"{name}: {frames} frames, {waveform.shape[0]} samples")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Voice-conditioned generation, end token fires at step 3 (3 frames).
    model = seeded_model(WEIGHT_SEED)
    run_generation_case("generate", model, max_tokens=6, stop_step=3)

    # Voice-conditioned generation, no stop, runs to max_tokens.
    model = seeded_model(WEIGHT_SEED)
    run_generation_case("length", model, max_tokens=5, stop_step=None)

    # Standalone codec decode over a random code grid.
    model = seeded_model(WEIGHT_SEED)
    run_codec_case("codec", model)

    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "generator": "tools/gen_voxtral_tts_fixtures.py",
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
