#!/usr/bin/env python3
"""Regenerate the chatterbox_turbo family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/chatterbox_turbo and the mlx_audio S3 tokenizer it
consumes; mlx 0.32.3). The true reference modules are imported and
driven against:

- tiny seeded T3 geometry: the reference ``create_gpt2_config`` is
  swapped for a 2-layer hidden-32 GPT-2 (learned positions kept), and
  the T3 config shrinks the text table; every forward method is
  inherited reference code;
- tiny seeded S3Gen meanflow geometry (conformer 2+2 blocks at width
  48, U-Net estimator at width 64 with 2 mid blocks and 2 transformer
  blocks per stage) with the flow start windows recorded per call (the
  reference draws fresh normals at every call; the Rust loader takes
  them as explicit inputs);
- the HiFT vocoder and CAMPPlus keep the reference structure at shrunk
  widths (the vocoder geometry is constructor-configuration; CAMPPlus
  shrinks growth/init/embedding but keeps the full FCM head and the
  real 12/24/16 dense-block depth, loaded under the turbo key tree);
- a tiny seeded S3TokenizerV2 (128 mel bins, hidden 128) standing in
  for the network S3TokenizerV2 checkpoint;
- greedy generation: the reference ``inference_turbo`` always samples,
  so the deterministic case drives the reference forward methods
  (``prepare_input_embeds``, ``tfmr``, ``speech_head``,
  ``_apply_repetition_penalty``) with an argmax pick over the
  post-penalty, post-filter logits - the explicit greedy mode the Rust
  port exposes (top-k/top-p never mask the maximum, so the filters
  cannot move the argmax);
- the sampled case runs the unmodified ``t3.inference_turbo`` from a
  seeded ``mx.random`` stream (the crate RNG is MLX-compatible
  threefry, so the Rust replay consumes the same key sequence; one
  uniform per sample);
- the HiFT sine randomness (the eight harmonic phase draws, the sine
  noise bed, and the discarded source-module noise bed) and the two
  flow noise windows (``noised_mels``, then the CFM ``z``) are
  recorded and replayed.

Cases (testdata/tts_chatterbox_turbo/): the one combined checkpoint
holds every submodule under the reference prefixes (t3. / s3gen. /
ve.), plus s3_tokenizer.safetensors next to it.

- t3/       conditioning rows, the composite prompt embeds, the greedy
            loop end to end (ids + per-step decision margins), and the
            sampled loop with the seeded categorical stream (per-step
            head logits recorded for both).
- flow/     token+prompt -> mel through the meanflow solve (2 Euler
            steps, no CFG) with the recorded start windows.
- hift/     mel -> 24 kHz waveform through the vocoder with the
            recorded sine draws, plus f0; the 20 ms trim fade is
            deterministic math applied on both sides.
- end_to_end/  greedy T3 ids -> the < 6561 crop -> three 4299 silence
            tokens -> flow -> vocoder -> fade, the full generate path
            on one realization.
- campplus/  the float64 Kaldi fbank (strict-edge mel banks) with the
            per-bin mean removal, and the embedding through the full
            CAMPPlus structure.
- sampling/traces.json  the filter math on fixed logits: the unique-token
            repetition penalty, the temperature divide, the top-k
            k-th-value threshold, the descending top-p shift mask, the
            categorical draw with a recorded uniform, and the greedy
            argmax (numpy mirrors of the reference formulas).

Every case is replayed on the Python side before freezing (same
weights, same streams, bitwise-identical outputs; the established
recording caveat). The script is deterministic: fixed seeds, no network
access, pinned to the CPU device.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_chatterbox_turbo_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

# Prefer the checkout the script is run from over any installed copy.
import sys

sys.path.insert(0, str(Path.cwd()))

import mlx_audio.tts.models.chatterbox_turbo.models.t3.t3_config as t3_config_module
import mlx_audio.tts.models.chatterbox_turbo.models.t3.t3 as t3_module
from mlx_audio.tts.models.chatterbox_turbo.models.t3.cond_enc import T3Cond
from mlx_audio.tts.models.chatterbox_turbo.models.t3.gpt2 import GPT2Config
from mlx_audio.tts.models.chatterbox_turbo.models.t3.t3 import T3
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen.decoder import (
    ConditionalDecoder,
)
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen.flow_matching import (
    CausalConditionalCFM,
)
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen.hifigan import (
    F0Predictor,
    HiFTGenerator,
)
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen.encoder import (
    UpsampleConformerEncoder,
)
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen.s3gen import (
    S3GEN_SIL,
    SPEECH_VOCAB_SIZE,
    S3Token2Mel,
)
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen.xvector import (
    CAMPPlus,
    extract_fbank_features,
)
from mlx_audio.tts.models.chatterbox_turbo.models.voice_encoder.config import (
    VoiceEncConfig,
)
from mlx_audio.tts.models.chatterbox_turbo.models.voice_encoder.voice_encoder import (
    VoiceEncoder,
)
from mlx_audio.tts.models.chatterbox_turbo.models.s3tokenizer import (
    log_mel_spectrogram,
)
from mlx_audio.codec.models.s3.model_v2 import S3TokenizerV2

OUT = Path("../turbospark/crates/audio/testdata/tts_chatterbox_turbo")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# --- tiny T3 geometry --------------------------------------------------------
T3_SEED = 97001
T3_LAYERS = 2
T3_DIM = 32
T3_HEADS = 4
T3_VOCAB = 64  # backbone vocab (never read; inputs are embeddings)
T3_POSITIONS = 64
T3_TEXT_DICT = 64
T3_SPEECH_DICT = 6563
T3_MAX_TEXT = 64
T3_MAX_SPEECH = 64
T3_MAX_GEN_LEN = 12
T3_SAMPLED_SEED = 77001
SPEAKER_EMBED_SIZE = 256
REP_PENALTY = 1.2
TOP_K = 1000
TOP_P = 0.95
TEMPERATURE = 0.8

TINY_GPT2 = {
    "model_type": "gpt2",
    "vocab_size": T3_VOCAB,
    "n_ctx": T3_POSITIONS,
    "n_positions": T3_POSITIONS,
    "n_embd": T3_DIM,
    "n_layer": T3_LAYERS,
    "n_head": T3_HEADS,
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-5,
}

# --- tiny S3Gen flow geometry ------------------------------------------------
FLOW_SEED = 97101
FLOW_WIDTH = 48
FLOW_HEADS = 4
FLOW_UNITS = 96
FLOW_BLOCKS = 2
FLOW_UP_BLOCKS = 2
DEC_WIDTH = 64
DEC_HEADS = 4
DEC_HEAD_DIM = 32
DEC_N_BLOCKS = 2
DEC_MID_BLOCKS = 2
SPK_DIM = 48
PROMPT_TOKENS = 12
GEN_TOKENS = 8
N_TIMESTEPS = 2  # the turbo meanflow default

# --- HiFT geometry (reference structure, shrunk widths) ----------------------
HIFT_SEED = 97201
HIFT_BASE = 64
HIFT_F0_CHANNELS = 64

# --- CAMPPlus geometry (full structure at shrunk widths) ---------------------
CAMPPLUS_SEED = 97301
CAMPPLUS_GROWTH = 8
CAMPPLUS_INIT = 16

# --- voice encoder geometry --------------------------------------------------
VE_SEED = 97401
VE_HIDDEN = 32
VE_SPEAKER_EMBED = 32

# --- conditioning wavebands --------------------------------------------------
COND_SEED = 97501
COND_WAVE_24K = 14400  # 0.6 s at 24 kHz
COND_WAVE_16K = 9600  # 0.6 s at 16 kHz

TEXT_TOKENS = [12, 44, 3, 41, 27, 5, 30, 61, 17, 6, 25, 40, 9, 8]


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def save_npy(name: str, arr: np.ndarray) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr, dtype=np.float32))


def save_i32(name: str, arr: np.ndarray) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr, dtype=np.int32))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_tiny_patches() -> None:
    """Swap the reference GPT-2 medium constructor for the tiny config.
    The `T3Config.n_channels` property reads the module-level
    ``GPT2_MEDIUM_CONFIG`` dict rather than the constructor, so the
    hidden width is patched there too."""
    t3_module.create_gpt2_config = lambda: GPT2Config(
        vocab_size=T3_VOCAB,
        n_positions=T3_POSITIONS,
        n_embd=T3_DIM,
        n_layer=T3_LAYERS,
        n_head=T3_HEADS,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
    )
    t3_config_module.GPT2_MEDIUM_CONFIG["hidden_size"] = T3_DIM


NORM_SUFFIXES = (
    ".norm.weight",
    ".norm.bias",
    ".norm1.weight",
    ".norm1.bias",
    ".norm3.weight",
    ".norm3.bias",
    ".norm_mha.weight",
    ".norm_mha.bias",
    ".norm_ff.weight",
    ".norm_ff.bias",
    ".after_norm.weight",
    ".after_norm.bias",
    ".ln_1.weight",
    ".ln_1.bias",
    ".ln_2.weight",
    ".ln_2.bias",
    ".ln_f.weight",
    ".ln_f.bias",
)


def seed_tree(module, seed: int, scale: float) -> dict:
    """Deterministic weights: small normals everywhere, norm gains and
    batchnorm stats positive so activations stay healthy."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, value in dict(tree_flatten(module.parameters())).items():
        if name.endswith("pos_enc.pe"):
            # The espnet positional table is derived at init (pure
            # sin/cos math); keep the module's own table so the fixture
            # matches a real checkpoint (which stores the derived
            # values) and the Rust port's recomputation.
            continue
        if name.endswith("stft_window") or name.endswith("trim_fade"):
            # Same derived-math rule: the iSTFT Hann window and the 20 ms
            # trim fade are pure formulas built at init. Seeding them
            # would desync the fixture from the Rust port's
            # recomputation.
            continue
        if "running_var" in name:
            out[name] = mx.array(
                (0.3 + np.abs(rng.standard_normal(value.shape)) * 0.4).astype(np.float32)
            )
        elif "running_mean" in name or name.endswith(NORM_SUFFIXES):
            out[name] = mx.array(rng.uniform(0.5, 1.5, size=value.shape).astype(np.float32))
        else:
            out[name] = mx.array((rng.standard_normal(value.shape) * scale).astype(np.float32))
    module.update(tree_unflatten(out))
    return out


class DrawPatch:
    """Records (or replays) the mx.random draws a call makes."""

    def __init__(self, replay: list[np.ndarray] | None = None):
        self.replay = replay
        self.draws: list[np.ndarray] = []
        self._orig = {}

    def __enter__(self) -> "DrawPatch":
        def wrap(orig):
            def fn(*args, **kwargs):
                if self.replay is not None:
                    out = mx.array(self.replay[len(self.draws)])
                else:
                    out = orig(*args, **kwargs)
                self.draws.append(np.asarray(out, dtype=np.float32))
                return out

            return fn

        for name in ("uniform", "normal"):
            self._orig[name] = getattr(mx.random, name)
            setattr(mx.random, name, wrap(self._orig[name]))
        return self

    def __exit__(self, *exc) -> None:
        for name, orig in self._orig.items():
            setattr(mx.random, name, orig)


def build_t3(seed: int) -> T3:
    apply_tiny_patches()
    hp = t3_module.T3Config(
        text_tokens_dict_size=T3_TEXT_DICT,
        max_text_tokens=T3_MAX_TEXT,
        speech_tokens_dict_size=T3_SPEECH_DICT,
        max_speech_tokens=T3_MAX_SPEECH,
        speaker_embed_size=SPEAKER_EMBED_SIZE,
    )
    t3 = T3(hp)
    seed_tree(t3, seed, scale=0.3)
    t3.eval()
    return t3


def make_t3_cond(seed: int) -> T3Cond:
    rng = np.random.default_rng(seed)
    speaker = rng.standard_normal((1, SPEAKER_EMBED_SIZE)).astype(np.float32) * 0.5
    tokens = rng.integers(0, SPEECH_VOCAB_SIZE, size=(1, 9)).astype(np.int32)
    return T3Cond(
        speaker_emb=mx.array(speaker),
        cond_prompt_speech_tokens=mx.array(tokens),
    )


def text_tokens_row() -> mx.array:
    return mx.array(TEXT_TOKENS)[None]


def filter_mirror(logits: np.ndarray, generated: list[int], temperature: float) -> np.ndarray:
    """The reference `_sample_token` filter chain on one row (penalty
    over unique generated ids, temperature divide, top-k k-th-value
    threshold, the descending top-p shift mask)."""
    x = np.asarray(logits, dtype=np.float32).ravel().copy()
    if REP_PENALTY != 1.0 and generated:
        uniq = np.unique(np.array(generated, dtype=np.int64))
        uniq = uniq[(uniq >= 0) & (uniq < x.shape[-1])]
        selected = x[uniq]
        x[uniq] = np.where(selected < 0, selected * REP_PENALTY, selected / REP_PENALTY)
    if temperature > 0 and temperature != 1.0:
        x = x / np.float32(temperature)
    if TOP_K > 0:
        k = min(TOP_K, x.shape[-1])
        kth = np.sort(x, axis=-1)[..., -k]
        x = np.where(x >= kth, x, np.float32(-np.inf))
    if TOP_P < 1.0:
        order = np.argsort(-x, axis=-1, kind="stable")
        sorted_logits = np.take_along_axis(x, order, axis=-1)
        probs = np.exp(sorted_logits - sorted_logits.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        cumulative = np.cumsum(probs, axis=-1)
        remove = cumulative > TOP_P
        remove = np.concatenate([np.zeros(1, dtype=bool), remove[:-1]], axis=-1)
        sorted_logits = np.where(remove, np.float32(-np.inf), sorted_logits)
        x = np.empty_like(sorted_logits)
        np.put_along_axis(x, order, sorted_logits, axis=-1)
    return x


def greedy_pick(logits_row: np.ndarray, generated: list[int]) -> int:
    """The explicit greedy mode: the argmax of the filtered logits
    (the filters never mask the maximum, so this equals the
    post-penalty argmax). `logits_row` is `(vocab,)`."""
    filtered = filter_mirror(logits_row, generated, 0.0)
    return int(np.argmax(filtered))


def run_greedy_loop(t3: T3, cond: T3Cond) -> tuple[list[int], list[int]]:
    """The reference forward methods driven with argmax picks. Returns
    the popped id stream (the reference return value) and the raw
    stream including a trailing stop token."""
    text_tokens = text_tokens_row()
    start = mx.ones((1, 1), dtype=mx.int32) * t3.hp.start_speech_token
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=cond, text_tokens=text_tokens, speech_tokens=start
    )
    hidden, cache = t3.tfmr(inputs_embeds=embeds, cache=None)
    logits = t3.speech_head(hidden[:, -1:, :])
    generated: list[int] = []
    nxt = greedy_pick(np.asarray(logits)[:, -1, :], generated)
    generated.append(nxt)
    for _ in range(T3_MAX_GEN_LEN):
        if nxt == int(t3.hp.stop_speech_token):
            break
        emb = t3.speech_emb(mx.array([[nxt]], dtype=mx.int32))
        hidden, cache = t3.tfmr(inputs_embeds=emb, cache=cache)
        logits = t3.speech_head(hidden)
        nxt = greedy_pick(np.asarray(logits)[:, -1, :], generated)
        generated.append(nxt)
    raw = list(generated)
    if generated and generated[-1] == int(t3.hp.stop_speech_token):
        generated.pop()
    return generated, raw


def head_logits_over_stream(t3: T3, cond: T3Cond, ids: list[int]) -> np.ndarray:
    """Re-runs the reference forward pass over a known id stream and
    records the speech-head output at every sampling point (the model
    is deterministic, so the logits equal the ones the generation loop
    saw; the greedy self-check below proves it)."""
    stop = int(t3.hp.stop_speech_token)
    text_tokens = text_tokens_row()
    start = mx.ones((1, 1), dtype=mx.int32) * t3.hp.start_speech_token
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=cond, text_tokens=text_tokens, speech_tokens=start
    )
    hidden, cache = t3.tfmr(inputs_embeds=embeds, cache=None)
    calls = [np.asarray(t3.speech_head(hidden[:, -1:, :]), dtype=np.float32)]
    for k in range(1, len(ids)):
        if ids[k - 1] == stop:
            break
        emb = t3.speech_emb(mx.array([[ids[k - 1]]], dtype=mx.int32))
        hidden, cache = t3.tfmr(inputs_embeds=emb, cache=cache)
        calls.append(np.asarray(t3.speech_head(hidden), dtype=np.float32))
    return np.stack(calls)


def margins_over_calls(calls: np.ndarray, ids: list[int], penalty: float) -> list[float]:
    """The post-penalty top-2 gap at every step (the decision margins
    the greedy argmax and the categorical draw read)."""
    out = []
    vocab = calls.shape[-1]
    for step in range(calls.shape[0]):
        row = calls[step, 0, -1, :].astype(np.float64)
        context = ids[:step]
        if context:
            uniq = np.unique(np.array(context, dtype=np.int64))
            uniq = uniq[(uniq >= 0) & (uniq < vocab)]
            selected = row[uniq]
            row = row.copy()
            row[uniq] = np.where(selected < 0, selected * penalty, selected / penalty)
        values = np.sort(row)[::-1]
        out.append(float(values[0] - values[1]))
    return out


def run_t3_case() -> dict:
    t3 = build_t3(T3_SEED)
    cond = make_t3_cond(T3_SEED + 1)

    # Conditioning rows and the composite prompt embeds.
    cond_rows = t3.prepare_conditioning(cond)
    mx.eval(cond_rows)
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=cond, text_tokens=text_tokens_row(), speech_tokens=mx.ones((1, 1), dtype=mx.int32) * t3.hp.start_speech_token
    )
    mx.eval(embeds)

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "text_ids": TEXT_TOKENS,
        "repetition_penalty": REP_PENALTY,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "max_gen_len": T3_MAX_GEN_LEN,
        "start_speech_token": int(t3.hp.start_speech_token),
        "stop_speech_token": int(t3.hp.stop_speech_token),
        "speaker_emb": floats(cond.speaker_emb),
        "cond_prompt_speech_tokens": ints(cond.cond_prompt_speech_tokens),
        "cond_rows": floats(cond_rows),
        "prompt_embeds_shape": list(np.asarray(embeds).shape),
        "prompt_embeds": floats(embeds),
    }

    # Greedy loop: the ids carry the choices; the per-step logits are
    # collected afterwards by re-running the reference forward pass
    # over the frozen stream (deterministic weights, so the logits are
    # the ones the loop read; the self-check proves it).
    greedy_ids, greedy_raw = run_greedy_loop(t3, cond)
    traces["greedy_ids"] = greedy_ids
    greedy_calls = head_logits_over_stream(t3, cond, greedy_raw)
    assert greedy_calls.shape[0] == len(greedy_raw), (
        greedy_calls.shape,
        len(greedy_raw),
    )
    # Self-check: the post-penalty, post-filter argmax over the
    # collected logits must reproduce the stream, and the recorded
    # decision margins (a thin margin would let the f32 reordering
    # between mlx and the crate flip a choice) stay wide.
    for step in range(greedy_calls.shape[0]):
        row = greedy_calls[step, 0, -1, :]
        pick = greedy_pick(row, greedy_raw[:step])
        assert pick == greedy_raw[step], f"greedy self-check step {step}"
    margins = margins_over_calls(greedy_calls, greedy_raw, REP_PENALTY)
    traces["greedy_margins"] = margins
    assert min(margins) > 0.01, "greedy margins too thin"

    # Sampled loop with a seeded stream; the categorical draw is the
    # only RNG consumer, one per sampled token.
    mx.random.seed(T3_SAMPLED_SEED)
    sampled_ids = ints(
        t3.inference_turbo(
            t3_cond=cond,
            text_tokens=text_tokens_row(),
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REP_PENALTY,
            max_gen_len=T3_MAX_GEN_LEN,
        ).reshape(-1)
    )
    traces["sampled_seed"] = T3_SAMPLED_SEED
    traces["sampled_ids"] = sampled_ids
    # The raw sampled stream includes a trailing stop when the loop hit
    # it (the reference pops it before returning).
    if len(sampled_ids) < T3_MAX_GEN_LEN + 1:
        sampled_raw = list(sampled_ids) + [int(t3.hp.stop_speech_token)]
    else:
        sampled_raw = list(sampled_ids)
    traces["sampled_ids_full"] = sampled_raw
    sampled_calls = head_logits_over_stream(t3, cond, sampled_raw)
    assert sampled_calls.shape[0] == len(sampled_raw), (
        sampled_calls.shape,
        len(sampled_raw),
    )
    # The post-penalty decision margins of the sampled steps stay
    # wide.
    s_margins = margins_over_calls(sampled_calls, sampled_raw, REP_PENALTY)
    traces["sampled_margins"] = s_margins
    assert min(s_margins) > 1e-4, "sampled margins too thin"

    # Python-side replay: a fresh build must reproduce both streams and
    # the collected logits.
    t3b = build_t3(T3_SEED)
    greedy_b, greedy_raw_b = run_greedy_loop(t3b, make_t3_cond(T3_SEED + 1))
    assert greedy_b == greedy_ids, "greedy replay diverged"
    assert greedy_raw_b == greedy_raw, "greedy raw stream diverged"
    greedy_calls_b = head_logits_over_stream(t3b, make_t3_cond(T3_SEED + 1), greedy_raw_b)
    assert np.array_equal(greedy_calls_b, greedy_calls), "greedy head logits diverged"
    mx.random.seed(T3_SAMPLED_SEED)
    sampled_b = ints(
        t3b.inference_turbo(
            t3_cond=make_t3_cond(T3_SEED + 1),
            text_tokens=text_tokens_row(),
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REP_PENALTY,
            max_gen_len=T3_MAX_GEN_LEN,
        ).reshape(-1)
    )
    assert sampled_b == sampled_ids, "sampled replay diverged"

    case_dir = OUT / "t3"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps(traces))
    save_npy("t3/greedy_head_calls.npy", greedy_calls)
    save_npy("t3/sampled_head_calls.npy", sampled_calls)
    return {"name": "t3", "greedy_tokens": len(greedy_ids), "sampled_tokens": len(sampled_ids)}


# --- tiny S3Gen meanflow -----------------------------------------------------
class TinyS3Token2Mel(S3Token2Mel):
    """S3Token2Mel at the tiny geometry; every forward method is
    inherited reference code."""

    def __init__(self):
        self.meanflow = True
        self.input_embedding = t3_module.nn.Embedding(SPEECH_VOCAB_SIZE, FLOW_WIDTH)
        self.speaker_encoder = CAMPPlus(
            feat_dim=80,
            embedding_size=SPK_DIM,
            growth_rate=CAMPPLUS_GROWTH,
            bn_size=4,
            init_channels=CAMPPLUS_INIT,
        )
        self.spk_embed_affine_layer = t3_module.nn.Linear(SPK_DIM, 80)
        self.encoder = UpsampleConformerEncoder(
            input_size=FLOW_WIDTH,
            output_size=FLOW_WIDTH,
            attention_heads=FLOW_HEADS,
            linear_units=FLOW_UNITS,
            num_blocks=FLOW_BLOCKS,
            dropout_rate=0.1,
        )
        self.encoder_proj = t3_module.nn.Linear(FLOW_WIDTH, 80)
        estimator = ConditionalDecoder(
            in_channels=80 * 4,
            out_channels=80,
            causal=True,
            channels=[DEC_WIDTH],
            dropout=0.0,
            attention_head_dim=DEC_HEAD_DIM,
            n_blocks=DEC_N_BLOCKS,
            num_mid_blocks=DEC_MID_BLOCKS,
            num_heads=DEC_HEADS,
            meanflow=True,
        )
        self.decoder = CausalConditionalCFM(
            in_channels=240,
            spk_emb_dim=80,
            sigma_min=1e-6,
            t_scheduler="cosine",
            inference_cfg_rate=0.7,
            estimator=estimator,
        )
        self.token_mel_ratio = 2
        self.pre_lookahead_len = 3


class TinyS3Token2Wav(TinyS3Token2Mel):
    """Adds the tiny HiFT vocoder and the trim fade, replicating the
    S3Token2Wav constructor at shrunk widths."""

    def __init__(self):
        super().__init__()
        self.mel2wav = HiFTGenerator(
            sampling_rate=24000,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            base_channels=HIFT_BASE,
            in_channels=80,
            istft_params={"n_fft": 16, "hop_len": 4},
            f0_predictor=F0Predictor(
                in_channels=80, hidden_channels=HIFT_F0_CHANNELS, num_layers=5
            ),
        )
        n_trim = 24000 // 50
        trim_fade = np.zeros(2 * n_trim, dtype=np.float32)
        trim_fade[n_trim:] = (
            (np.cos(np.linspace(np.pi, 0, n_trim)) + 1) / 2
        ).astype(np.float32)
        self.trim_fade = mx.array(trim_fade)

    def inference(self, speech_tokens, ref_dict, n_cfm_timesteps=N_TIMESTEPS):
        output_mels = S3Token2Mel.__call__(
            self, speech_tokens, ref_dict, n_cfm_timesteps=n_cfm_timesteps, finalize=True
        )
        output_mels = output_mels.transpose(0, 2, 1)  # (B, T, 80) for HiFT
        output_wavs, _ = self.mel2wav.inference(
            output_mels, mx.zeros((1, 1, 0))
        )
        fade_len = len(self.trim_fade)
        if output_wavs.shape[1] >= fade_len:
            faded_start = output_wavs[:, :fade_len] * self.trim_fade
            output_wavs = mx.concatenate(
                [faded_start, output_wavs[:, fade_len:]], axis=1
            )
        return output_wavs


def attention_margin_seed(s3gen, start: int, threshold: float = 5e-4) -> int:
    """Picks a flow-input seed whose smallest top-2 attention score gap
    (over every encoder and estimator attention head, row, and Euler
    step) clears `threshold`. The crate and mlx sum the attention
    projections in different orders (~1e-5 score noise at this
    geometry), so a smaller gap lets an f32 reordering flip an
    attention weight and scramble everything downstream."""
    import mlx.nn as nn

    class EncRec(nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.attn = attn
            self.min_gap = 1e30

        def __call__(self, x, mask, pos_emb):
            a = self.attn
            B, T, D = x.shape
            H, dk = a.n_head, a.d_k
            q = a.linear_q(x).reshape(B, T, H, dk)
            k = a.linear_k(x).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
            qu = (q + a.pos_bias_u).transpose(0, 2, 1, 3)
            qv = (q + a.pos_bias_v).transpose(0, 2, 1, 3)
            p = a.linear_pos(pos_emb).reshape(1, 2 * T - 1, H, dk).transpose(0, 2, 1, 3)
            ac = qu @ k.transpose(0, 1, 3, 2)
            bd = qv @ p.transpose(0, 1, 3, 2)
            if ac.shape != bd.shape:
                zero_pad = mx.zeros((B, H, T, 1))
                xp = mx.concatenate([zero_pad, bd], axis=-1).reshape(B, H, 2 * T, T)
                bd = xp[:, :, 1:, :].reshape(B, H, T, 2 * T - 1)[:, :, :, :T]
            scores = (ac + bd) * a.scale
            srt = mx.sort(scores, axis=-1)
            gap = srt[:, :, :, -1] - srt[:, :, :, -2]
            self.min_gap = min(self.min_gap, float(gap.min()))
            return a(x, mask, pos_emb)

    class DecRec(nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.attn = attn
            self.min_gap = 1e30

        def __call__(self, x, mask=None):
            a = self.attn
            B, T, C = x.shape
            H, dh = a.num_heads, a.head_dim
            q = a.to_q(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            k = a.to_k(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) * a.scale
            srt = mx.sort(scores, axis=-1)
            gap = srt[:, :, :, -1] - srt[:, :, :, -2]
            self.min_gap = min(self.min_gap, float(gap.min()))
            return a(x, mask)

    saved = []
    recs = []
    for layer in s3gen.encoder.encoders:
        saved.append((layer, "self_attn", layer.self_attn))
        r = EncRec(layer.self_attn)
        layer.self_attn = r
        recs.append(r)
    for layer in s3gen.encoder.up_encoders:
        saved.append((layer, "self_attn", layer.self_attn))
        r = EncRec(layer.self_attn)
        layer.self_attn = r
        recs.append(r)
    estimator = s3gen.decoder.estimator
    blocks = [estimator.down_blocks[0]] + list(estimator.mid_blocks) + [estimator.up_blocks[0]]
    for blk in blocks:
        for tb in blk.transformer_blocks:
            saved.append((tb, "attn1", tb.attn1))
            r = DecRec(tb.attn1)
            tb.attn1 = r
            recs.append(r)
    try:
        for offset in range(start, start + 500):
            for r in recs:
                r.min_gap = 1e30
            inputs = flow_inputs_at(s3gen_seed_offset=offset)
            ref_dict = ref_dict_from(inputs)
            s3gen(mx.array(inputs[1])[None], ref_dict=ref_dict, n_cfm_timesteps=N_TIMESTEPS)
            gap = min(r.min_gap for r in recs)
            if gap > threshold:
                return offset, gap
    finally:
        for layer_owner, name, original in saved:
            setattr(layer_owner, name, original)
    raise SystemExit("no flow seed gives a comfortable attention margin")


def flow_inputs_at(s3gen_seed_offset: int):
    return flow_inputs(FLOW_SEED + s3gen_seed_offset)


def flow_inputs(seed: int):
    rng = np.random.default_rng(seed)
    prompt_tokens = rng.integers(0, SPEECH_VOCAB_SIZE, PROMPT_TOKENS).astype(np.int32)
    gen_tokens = rng.integers(0, SPEECH_VOCAB_SIZE, GEN_TOKENS).astype(np.int32)
    prompt_feat = (
        rng.standard_normal((PROMPT_TOKENS * 2, 80)).astype(np.float32) * 0.5 - 2.0
    )
    embedding = rng.standard_normal(SPK_DIM).astype(np.float32) * 0.5
    return prompt_tokens, gen_tokens, prompt_feat, embedding


def build_s3gen(seed: int) -> TinyS3Token2Wav:
    s3gen = TinyS3Token2Wav()
    seed_tree(s3gen, seed, scale=0.2)
    # The HiFT resblock chain is high-gain at the shared 0.2 scale: the
    # stacked conv activations reach 1e10 before the exp/sin iSTFT head
    # and the waveform pins at the 0.99 audio limit, which carries no
    # parity information. Re-seed the vocoder at 0.04 so the fixture
    # waveform stays unsaturated.
    seed_tree(s3gen.mel2wav, seed + 1, scale=0.04)
    s3gen.eval()
    # Voiced f0: hold the f0 classifier bias near 60 Hz.
    s3gen.mel2wav.f0_predictor.classifier.bias = mx.array(
        (60.0 + 0.5 * np.random.default_rng(seed + 2).standard_normal((1,))).astype(np.float32)
    )
    # Snake alphas near one so the activation keeps its scale.
    flat = dict(tree_flatten(s3gen.mel2wav.parameters()))
    for name, value in flat.items():
        if name.endswith(".alpha"):
            alpha = 1.0 + 0.25 * np.random.default_rng(seed + 3).standard_normal(value.shape)
            s3gen.mel2wav.update(tree_unflatten({name: mx.array(alpha.astype(np.float32))}))
    return s3gen


def ref_dict_from(inputs) -> dict:
    prompt_tokens, gen_tokens_unused, prompt_feat, embedding = inputs
    _ = gen_tokens_unused
    return dict(
        prompt_token=mx.array(prompt_tokens)[None],
        prompt_token_len=mx.array([PROMPT_TOKENS], dtype=mx.int32),
        prompt_feat=mx.array(prompt_feat)[None],
        prompt_feat_len=mx.array([PROMPT_TOKENS * 2], dtype=mx.int32),
        embedding=mx.array(embedding)[None],
    )


def flow_noise_from(draws: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The two flow start windows in reference draw order: the
    generation-span `noised_mels`, then the CFM `z` whose prompt head
    survives the splice."""
    noised_mels = draws[0].reshape(80, -1)
    z = draws[1].reshape(80, -1)
    prompt_len = z.shape[1] - noised_mels.shape[1]
    return noised_mels, z[:, :prompt_len]


def run_flow_case(s3gen: TinyS3Token2Wav, flow_offset: int) -> dict:
    inputs = flow_inputs(FLOW_SEED + flow_offset)
    ref_dict = ref_dict_from(inputs)
    _, gen_tokens, _, _ = inputs
    with DrawPatch() as recorder:
        mel = s3gen(mx.array(gen_tokens)[None], ref_dict=ref_dict, n_cfm_timesteps=N_TIMESTEPS)
    mel_np = np.asarray(mel)
    assert mel_np.shape == (1, 80, GEN_TOKENS * 2), mel_np.shape
    assert len(recorder.draws) == 2, [d.shape for d in recorder.draws]
    noised_mels, z_prompt = flow_noise_from(recorder.draws)

    # Python-side replay from the recorded windows.
    with DrawPatch(replay=recorder.draws) as replay_patch:
        mel_b = s3gen(mx.array(gen_tokens)[None], ref_dict=ref_dict, n_cfm_timesteps=N_TIMESTEPS)
    assert np.array_equal(mel_np, np.asarray(mel_b)), "flow replay diverged"
    assert replay_patch.draws[0].shape == recorder.draws[0].shape

    case_dir = OUT / "flow"
    case_dir.mkdir(parents=True, exist_ok=True)
    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "n_timesteps": N_TIMESTEPS,
        "prompt_tokens": ints(inputs[0]),
        "gen_tokens": ints(gen_tokens),
        "prompt_feat_shape": list(inputs[2].shape),
        "embedding_shape": [SPK_DIM],
        "flow_mel_shape": [GEN_TOKENS * 2, 80],
        "flow_mel_abs_max": float(np.abs(mel_np).max()),
        "noised_mels_shape": [80, int(noised_mels.shape[1])],
        "z_prompt_shape": [80, int(z_prompt.shape[1])],
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))
    save_npy("flow/prompt_feat.npy", inputs[2])
    save_npy("flow/embedding.npy", inputs[3])
    save_i32("flow/prompt_tokens.npy", inputs[0])
    save_i32("flow/gen_tokens.npy", gen_tokens)
    save_npy("flow/noised_mels.npy", noised_mels)
    save_npy("flow/z_prompt.npy", z_prompt)
    save_npy("flow/flow_mel.npy", mel_np[0].T)
    return {
        "name": "flow",
        "mel_frames": int(mel_np.shape[2]),
        "mel": mel_np,
        "draws": recorder.draws,
    }


def run_hift_case(s3gen: TinyS3Token2Wav, mel_np: np.ndarray) -> dict:
    hift = s3gen.mel2wav
    f0 = np.asarray(hift.f0_predictor(mx.array(mel_np)))
    save_npy("hift/f0.npy", f0[0])

    with DrawPatch() as recorder:
        wav, _ = hift.inference(mx.array(mel_np).transpose(0, 2, 1), mx.zeros((1, 1, 0)))
    wav_np = np.asarray(wav)[0]
    assert len(recorder.draws) == 3, [d.shape for d in recorder.draws]
    phase = recorder.draws[0].reshape(-1).copy()
    assert phase.shape[0] == 8, phase.shape  # harmonics 1..8 only
    sine_noise = recorder.draws[1][0]  # [harmonics + 1, T_wav]
    source_noise = recorder.draws[2].reshape(-1)
    # The sine beds run at the full pre-truncation waveform length; the
    # min_len fusion clamp shortens the rendered waveform.
    assert sine_noise.shape[1] == mel_np.shape[2] * 480, (
        sine_noise.shape,
        mel_np.shape,
    )
    assert source_noise.shape[0] == mel_np.shape[2] * 480

    # Replay the recorded draws: the waveform must be identical.
    with DrawPatch(replay=recorder.draws) as replay_patch:
        wav_b, _ = hift.inference(mx.array(mel_np).transpose(0, 2, 1), mx.zeros((1, 1, 0)))
    assert np.array_equal(wav_np, np.asarray(wav_b)[0]), "hift replay diverged"
    assert np.array_equal(source_noise, replay_patch.draws[2].reshape(-1))

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "mel_frames": int(mel_np.shape[2]),
        "phase_shape": [int(phase.shape[0])],
        "sine_noise_shape": list(sine_noise.shape),
        "wav_len": int(wav_np.shape[0]),
        "wav_abs_max": float(np.abs(wav_np).max()),
        "source_noise_discarded": True,
    }
    (OUT / "hift").mkdir(parents=True, exist_ok=True)
    (OUT / "hift" / "traces.json").write_text(json.dumps(traces))
    save_npy("hift/phase.npy", phase)
    save_npy("hift/sine_noise.npy", sine_noise)
    save_npy("hift/source_noise.npy", source_noise)
    save_npy("hift/wav.npy", wav_np)
    return {"name": "hift", "wav_len": int(wav_np.shape[0])}


def crop_speech_tokens(ids: list[int]) -> np.ndarray:
    """The turbo crop: keep the ids strictly below the vocabulary bound
    and append the three silence tokens."""
    tokens = np.array([t for t in ids if t < SPEECH_VOCAB_SIZE], dtype=np.int32)
    silence = np.array([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=np.int32)
    return np.concatenate([tokens, silence])


def run_end_to_end_case(s3gen: TinyS3Token2Wav, ref_dict: dict) -> dict:
    """The full generate path on one realization: greedy T3 -> crop ->
    silence tail -> flow -> vocoder -> trim fade."""
    t3 = build_t3(T3_SEED)
    ids, _raw = run_greedy_loop(t3, make_t3_cond(T3_SEED + 1))
    speech_tokens = crop_speech_tokens(ids)
    with DrawPatch() as recorder:
        wav = s3gen.inference(mx.array(speech_tokens)[None], ref_dict=ref_dict)
    wav_np = np.asarray(wav)[0]
    # Draw order: the two flow windows, then the three vocoder draws.
    assert len(recorder.draws) == 5, [d.shape for d in recorder.draws]
    noised_mels, z_prompt = flow_noise_from(recorder.draws[:2])
    phase = recorder.draws[2].reshape(-1).copy()
    sine_noise = recorder.draws[3][0]
    source_noise = recorder.draws[4].reshape(-1)

    # Replay.
    with DrawPatch(replay=recorder.draws) as replay_patch:
        wav_b = s3gen.inference(mx.array(speech_tokens)[None], ref_dict=ref_dict)
    assert np.array_equal(wav_np, np.asarray(wav_b)[0]), "end-to-end replay diverged"
    assert replay_patch.draws[4].shape == recorder.draws[4].shape

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "text_ids": TEXT_TOKENS,
        "ids": ids,
        "speech_tokens": ints(speech_tokens),
        "phase": floats(phase),
        "sine_noise_shape": list(sine_noise.shape),
        "noised_mels_shape": [80, int(noised_mels.shape[1])],
        "z_prompt_shape": [80, int(z_prompt.shape[1])],
        "wav_len": int(wav_np.shape[0]),
        "wav_abs_max": float(np.abs(wav_np).max()),
    }
    (OUT / "end_to_end").mkdir(parents=True, exist_ok=True)
    (OUT / "end_to_end" / "traces.json").write_text(json.dumps(traces))
    save_npy("end_to_end/noised_mels.npy", noised_mels)
    save_npy("end_to_end/z_prompt.npy", z_prompt)
    save_npy("end_to_end/sine_noise.npy", sine_noise)
    save_npy("end_to_end/source_noise.npy", source_noise)
    save_npy("end_to_end/wav.npy", wav_np)
    return {"name": "end_to_end", "speech_tokens": int(speech_tokens.shape[0])}


def run_campplus_case(model_campplus, seed: int) -> dict:
    rng = np.random.default_rng(seed + 3)
    wave = (rng.standard_normal(COND_WAVE_16K) * 0.1).astype(np.float32)
    fbank = np.asarray(extract_fbank_features(mx.array(wave)))[0]
    fbank = fbank - fbank.mean(axis=0, keepdims=True)
    embedding = np.asarray(model_campplus(mx.array(fbank)[None]))
    assert embedding.shape == (1, SPK_DIM), embedding.shape
    # Replay.
    embedding_b = np.asarray(model_campplus(mx.array(fbank)[None]))
    assert np.array_equal(embedding, embedding_b), "campplus replay diverged"

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "wave_len": int(wave.shape[0]),
        "fbank_shape": list(fbank.shape),
        "embedding_shape": [SPK_DIM],
        "embedding_abs_max": float(np.abs(embedding).max()),
    }
    (OUT / "campplus").mkdir(parents=True, exist_ok=True)
    (OUT / "campplus" / "traces.json").write_text(json.dumps(traces))
    save_npy("campplus/wave.npy", wave)
    save_npy("campplus/fbank.npy", fbank)
    save_npy("campplus/embedding.npy", embedding[0])
    return {"name": "campplus", "fbank_frames": int(fbank.shape[0])}


def build_sampling_case() -> None:
    """The filter math on fixed logits: the unique-token repetition
    penalty, the temperature divide, the top-k k-th-value threshold,
    the descending top-p shift mask, the categorical draw with a
    recorded uniform, and the greedy argmax (numpy mirrors of the
    reference formulas; the mlx functions agree to f32 rounding and the
    Rust sampler ports the same math)."""

    def vector(n, offset):
        values = np.array(
            [
                float(np.sin(0.71 * (i + offset)) * 4.0 + np.cos(0.29 * i) * 1.5)
                for i in range(n)
            ],
            dtype=np.float32,
        )
        assert len(set(values.tolist())) == n, "fixed logits must have no ties"
        return values

    def masked(values):
        # JSON has no infinity: filtered entries carry None, the Rust
        # reader maps them back to -inf.
        return [
            float(v) if np.isfinite(v) else None
            for v in np.asarray(values, dtype=np.float32).ravel().tolist()
        ]

    def penalty_mirror(logits, context, penalty):
        out = np.asarray(logits, dtype=np.float32).copy()
        tokens = np.unique(np.asarray(context, dtype=np.int64))
        tokens = tokens[(tokens >= 0) & (tokens < out.shape[-1])]
        selected = out[tokens]
        out[tokens] = np.where(selected < 0, selected * penalty, selected / penalty)
        return out

    def temperature_mirror(logits, temperature):
        return np.asarray(logits, dtype=np.float32) / np.float32(temperature)

    def top_k_mirror(logits, top_k):
        x = np.asarray(logits, dtype=np.float32)
        k = min(top_k, x.shape[-1])
        kth = np.sort(x, axis=-1)[..., -k]
        return np.where(x >= kth, x, np.float32(-np.inf))

    def top_p_mirror(logits, top_p):
        x = np.asarray(logits, dtype=np.float32)[None, :]
        order = np.argsort(-x, axis=-1, kind="stable")
        sorted_logits = np.take_along_axis(x, order, axis=-1)
        probs = np.exp(sorted_logits - sorted_logits.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        cumulative = np.cumsum(probs, axis=-1)
        remove = cumulative > top_p
        remove = np.concatenate(
            [np.zeros((x.shape[0], 1), dtype=bool), remove[:, :-1]], axis=-1
        )
        sorted_logits = np.where(remove, np.float32(-np.inf), sorted_logits)
        out = np.empty_like(sorted_logits)
        np.put_along_axis(out, order, sorted_logits, axis=-1)
        return out[0]

    def categorical_mirror(logits, uniform):
        weights = np.exp(np.asarray(logits, dtype=np.float32) - np.max(logits)).astype(np.float64)
        weights[~np.isfinite(weights)] = 0.0
        total = weights.sum()
        u = np.float64(uniform) * total
        cumulative = 0.0
        count = 0
        for w in weights:
            if cumulative > u:
                break
            count += 1
            cumulative += w
        return min(count - 1, len(weights) - 1)

    vocab = 96
    cases = []

    # Repetition penalty with duplicate context tokens (np.unique
    # first, so duplicates are idempotent).
    base = vector(vocab, 5)
    context = [5, 5, 40, 12]
    penalized = penalty_mirror(base, context, REP_PENALTY)
    cases.append(
        {
            "name": "repetition_penalty_duplicates",
            "logits": floats(base),
            "context": context,
            "penalty": REP_PENALTY,
            "penalized": floats(penalized),
        }
    )

    # The temperature divide.
    temped = temperature_mirror(base, TEMPERATURE)
    cases.append(
        {
            "name": "temperature_divide",
            "logits": floats(base),
            "temperature": TEMPERATURE,
            "scaled": floats(temped),
        }
    )

    # The top-k k-th-value threshold on the unnormalized logits (ties
    # at the threshold survive).
    raw = vector(vocab, 13)
    cases.append(
        {
            "name": "top_k_threshold",
            "logits": floats(raw),
            "top_k": 16,
            "filtered": masked(top_k_mirror(raw, 16)),
        }
    )
    narrow = vector(24, 19)
    cases.append(
        {
            "name": "top_k_all_kept",
            "logits": floats(narrow),
            "top_k": 32,
            "filtered": masked(top_k_mirror(narrow, 32)),
        }
    )

    # The descending top-p shift mask on the raw logits.
    cases.append(
        {
            "name": "top_p_shift_0_9",
            "logits": floats(raw),
            "top_p": 0.9,
            "filtered": masked(top_p_mirror(raw, 0.9)),
        }
    )
    cases.append(
        {
            "name": "top_p_shift_0_5",
            "logits": floats(narrow),
            "top_p": 0.5,
            "filtered": masked(top_p_mirror(narrow, 0.5)),
        }
    )

    # The full chain (penalty, temperature, top-k, top-p), then the
    # categorical draw with a recorded uniform and a decisive margin.
    after_penalty = penalty_mirror(raw, [3, 3, 40], REP_PENALTY)
    after_temp = temperature_mirror(after_penalty, TEMPERATURE)
    after_top_k = top_k_mirror(after_temp, 16)
    after_top_p = top_p_mirror(after_top_k, 0.9)
    rng = np.random.default_rng(9601)
    uniforms = rng.uniform(0.02, 0.98, size=6).astype(np.float32)
    winner = categorical_mirror(after_top_p, uniforms[0])
    scaled = np.asarray(after_top_p, dtype=np.float32).astype(np.float64)
    weights = np.exp(scaled - scaled.max())
    ordered = np.sort(weights / weights.sum())[::-1]
    margin = float(ordered[0] - ordered[1])
    assert margin > 1e-4, f"thin categorical margin {margin}"
    cases.append(
        {
            "name": "filtered_categorical",
            "logits": masked(after_top_p),
            "uniform": floats(np.array([uniforms[0]])),
            "winner": int(winner),
            "prob_margin": margin,
        }
    )

    # Greedy: the plain post-penalty argmax (the filters never mask the
    # maximum).
    penalized_greedy = penalty_mirror(base, [0, 3, 3, 40], REP_PENALTY)
    cases.append(
        {
            "name": "greedy_argmax",
            "logits": floats(penalized_greedy),
            "argmax": int(np.argmax(penalized_greedy)),
        }
    )

    case_dir = OUT / "sampling"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps({"cases": cases}))


def main() -> None:
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    OUT.mkdir(parents=True, exist_ok=True)
    apply_tiny_patches()
    summary = []

    # Build the tiny family once; one combined checkpoint.
    t3 = build_t3(T3_SEED)
    s3gen = build_s3gen(FLOW_SEED)
    # The turbo VoiceEncConfig is a plain attribute class (not a
    # dataclass): set the shrunk widths on the instance.
    ve_hp = VoiceEncConfig()
    ve_hp.num_mels = 40
    ve_hp.ve_hidden_size = VE_HIDDEN
    ve_hp.speaker_embed_size = VE_SPEAKER_EMBED
    ve = VoiceEncoder(ve_hp)
    seed_tree(ve, VE_SEED, scale=0.3)
    ve.eval()

    # --- combined checkpoint (model.safetensors) ------------------------
    weights: dict[str, mx.array] = {}
    for name, value in dict(tree_flatten(t3.parameters())).items():
        weights[f"t3.{name}"] = value.astype(mx.float32)
    for name, value in dict(tree_flatten(s3gen.parameters())).items():
        weights[f"s3gen.{name}"] = value.astype(mx.float32)
    for name, value in dict(tree_flatten(ve.parameters())).items():
        weights[f"ve.{name}"] = value.astype(mx.float32)
    mx.save_safetensors(str(OUT / "model.safetensors"), weights)

    # --- config.json -----------------------------------------------------
    config = {
        "model_type": "chatterbox_turbo",
        "s3_sr": 16000,
        "s3gen_sr": 24000,
        "sample_rate": 24000,
        "enc_cond_len": 15 * 16000,
        "dec_cond_len": 10 * 24000,
        "t3_config": {
            "text_tokens_dict_size": T3_TEXT_DICT,
            "start_text_token": 255,
            "stop_text_token": 0,
            "max_text_tokens": T3_MAX_TEXT,
            "speech_tokens_dict_size": T3_SPEECH_DICT,
            "start_speech_token": 6561,
            "stop_speech_token": 6562,
            "max_speech_tokens": T3_MAX_SPEECH,
            "speech_cond_prompt_len": 375,
            "encoder_type": "voice_encoder",
            "speaker_embed_size": SPEAKER_EMBED_SIZE,
            "use_perceiver_resampler": False,
            "emotion_adv": False,
            "gpt2_config": TINY_GPT2,
        },
        "flow_config": {
            "input_size": FLOW_WIDTH,
            "output_size": 80,
            "spk_embed_dim": SPK_DIM,
            "vocab_size": int(SPEECH_VOCAB_SIZE),
            "attention_heads": FLOW_HEADS,
            "linear_units": FLOW_UNITS,
            "num_blocks": FLOW_BLOCKS,
            "upsample_stride": 2,
            "pre_lookahead_len": 3,
            "n_timesteps": N_TIMESTEPS,
            "inference_cfg_rate": 0.7,
            "meanflow": True,
            "decoder_channels": DEC_WIDTH,
            "decoder_n_blocks": DEC_N_BLOCKS,
            "decoder_num_mid_blocks": DEC_MID_BLOCKS,
            "decoder_num_heads": DEC_HEADS,
            "decoder_head_dim": DEC_HEAD_DIM,
        },
        "hift_config": {
            "base_channels": HIFT_BASE,
            "in_channels": 80,
            "f0_channels": HIFT_F0_CHANNELS,
        },
        "campplus_config": {
            "feat_dim": 80,
            "embedding_size": SPK_DIM,
            "fcm_channels": 32,
            "init_channels": CAMPPLUS_INIT,
            "growth_rate": CAMPPLUS_GROWTH,
            "bn_size": 4,
            "num_layers": [12, 24, 16],
        },
        "ve_config": {
            "num_mels": 40,
            "sample_rate": 16000,
            "speaker_embed_size": VE_SPEAKER_EMBED,
            "ve_hidden_size": VE_HIDDEN,
            "ve_partial_frames": 160,
            "ve_final_relu": True,
        },
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # --- tiny S3 tokenizer ------------------------------------------------
    cfg2 = dict(
        n_mels=128,
        n_audio_ctx=64,
        n_audio_state=128,
        n_audio_head=2,
        n_audio_layer=2,
        n_codebook_size=3**8,
    )
    mx.random.seed(111)
    tiny_s3tok = S3TokenizerV2("speech_tokenizer_v2", type("C", (), cfg2)())
    seed_model_seed = 113
    flat = dict(tree_flatten(tiny_s3tok.parameters()))
    rng = np.random.default_rng(seed_model_seed)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in flat.items()
    }
    tiny_s3tok.load_weights(list(seeded.items()))
    tiny_s3tok.eval()
    mx.eval(tiny_s3tok.parameters())
    mx.save_safetensors(
        str(OUT / "s3_tokenizer.safetensors"),
        {name: value.astype(mx.float32) for name, value in seeded.items()},
    )
    (OUT / "s3_tokenizer_config.json").write_text(json.dumps(cfg2, indent=2) + "\n")

    # --- cases ------------------------------------------------------------
    summary.append(run_t3_case())
    # The flow input seed needs a comfortable attention margin so the
    # crate's summation order cannot flip a weight (see
    # attention_margin_seed).
    flow_offset, flow_gap = attention_margin_seed(s3gen, start=4)
    summary.append(
        {"name": "flow_seed_margin", "offset": flow_offset, "min_gap": flow_gap}
    )
    inputs = flow_inputs(FLOW_SEED + flow_offset)
    ref_dict = ref_dict_from(inputs)
    flow_result = run_flow_case(s3gen, flow_offset)
    summary.append({k: v for k, v in flow_result.items() if k not in ("mel", "draws")})
    # The hift and end-to-end cases consume the flow realization the
    # fixture froze (the recorded-draw run), not a second noised run.
    mel = flow_result["mel"]
    summary.append(run_hift_case(s3gen, mel))
    summary.append(run_end_to_end_case(s3gen, ref_dict))
    summary.append(run_campplus_case(s3gen.speaker_encoder, CAMPPLUS_SEED))
    build_sampling_case()

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = digest(path)
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_chatterbox_turbo_fixtures.py",
        "seeds": {
            "t3": T3_SEED,
            "t3_sampled": T3_SAMPLED_SEED,
            "flow": FLOW_SEED,
            "hift": HIFT_SEED,
            "campplus": CAMPPLUS_SEED,
            "ve": VE_SEED,
            "cond": COND_SEED,
        },
        "summary": summary,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote fixtures to {OUT}")
    mx.set_default_device(previous_device)


if __name__ == "__main__":
    main()
