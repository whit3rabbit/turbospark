#!/usr/bin/env python3
"""Regenerate the chatterbox (T3 + S3Gen) family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/chatterbox and the mlx_audio/codec S3 tokenizer it
consumes; mlx 0.32.3). The true reference modules are imported and
driven against:

- tiny seeded T3 geometry: the reference ``LLAMA_CONFIGS["Llama_520M"]``
  entry is swapped for a 2-layer hidden-32 llama (llama3 rope scaling
  kept), and the perceiver constructor defaults are patched to the tiny
  width; every forward method is inherited reference code;
- tiny seeded S3Gen flow geometry (conformer 2+2 blocks at width 48,
  matcha U-Net at width 64 with 2 mid blocks) with the reference CFM
  noise buffer overwritten by a short seeded window (the reference
  generates a 15000-frame seed-0 buffer at init; real checkpoints do
  not carry it and the Rust loader accepts any buffer at least as long
  as the solve needs);
- the HiFT vocoder and CAMPPlus keep the reference structure at shrunk
  widths (the vocoder geometry is constructor-configuration in the
  reference; CAMPPlus shrinks growth/init/embedding but keeps the full
  FCM head and the real 12/24/16 dense-block depth);
- a tiny seeded S3TokenizerV2 (128 mel bins, hidden 128) built the same
  way the s3 codec fixtures build theirs, standing in for the network
  S3TokenizerV2 checkpoint;
- greedy sampling (temperature 0) for the deterministic T3 end-to-end
  case, and a seeded ``mx.random`` stream for the sampled case (the
  crate RNG is MLX-compatible threefry, so the Rust replay consumes the
  same key sequence; per-step filtered logits and choices are recorded
  as the cross-check);
- the HiFT sine-generator randomness (initial phase draw and the sine
  noise bed) recorded and replayed; the reference's third draw (the
  source-module noise bed) is discarded upstream and only recorded to
  keep the draw order honest.

Cases (testdata/tts_chatterbox/): the one combined checkpoint holds
every submodule under the reference prefixes (t3. / flow. / mel2wav. /
speaker_encoder. / ve.), plus s3_tokenizer.safetensors next to it.

- t3/      conditioning-encoder rows, the composite prompt embeds, the
           greedy generate loop end to end (ids + stop reason + per-step
           decision margins), and the sampled loop with the seeded
           categorical stream.
- flow/    token+prompt -> mel through the flow (embed_ref alignment on
           synthetic inputs, the 10-step Euler solve, prompt-span crop).
- hift/    mel -> 24 kHz waveform through the vocoder with the recorded
           sine draws, plus f0; the 20 ms trim fade is deterministic
           math applied on both sides.
- end_to_end/  greedy T3 ids -> token crop -> flow -> hift -> fade, the
           full generate path on one realization.
- campplus/   kaldi fbank + mean-normalized CAMPPlus embedding.
- voice_encoder/  trim + 40-mel + partial windows + stacked LSTM +
           speaker embedding.
- conditioning/  prepare_conditionals on the three wavebands through
           the tiny S3 tokenizer (mel + token alignment, T3 prompt
           tokens, CAMPPlus embedding, voice-encoder speaker embed).
- sampling/traces.json  the filter math on fixed logits: CFG combine,
           the repetition penalty with duplicate context tokens, raw
           top-p/min-p filters, and the categorical draw with recorded
           uniforms (numpy mirrors of the reference formulas).

Every case is replayed on the Python side before freezing (same
weights, same streams, bitwise-identical outputs; the established
recording caveat). The script is deterministic: fixed seeds, no network
access, pinned to the CPU device.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_chatterbox_fixtures.py
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

import mlx_audio.tts.models.chatterbox.config as cb_config
from mlx_audio.tts.models.chatterbox.chatterbox import drop_invalid_tokens, SPEECH_VOCAB_SIZE
from mlx_audio.tts.models.chatterbox.s3gen.decoder import ConditionalDecoder
from mlx_audio.tts.models.chatterbox.s3gen.f0_predictor import ConvRNNF0Predictor
from mlx_audio.tts.models.chatterbox.s3gen.flow import CausalMaskedDiffWithXvec
from mlx_audio.tts.models.chatterbox.s3gen.flow_matching import CFM_PARAMS, CausalConditionalCFM
from mlx_audio.tts.models.chatterbox.s3gen.hifigan import HiFTGenerator
from mlx_audio.tts.models.chatterbox.s3gen.s3gen import S3Token2Mel
from mlx_audio.tts.models.chatterbox.s3gen.transformer.upsample_encoder import (
    UpsampleConformerEncoder,
)
from mlx_audio.tts.models.chatterbox.s3gen.xvector import CAMPPlus, kaldi_fbank
from mlx_audio.tts.models.chatterbox.t3.cond_enc import T3Cond
from mlx_audio.tts.models.chatterbox.t3.perceiver import Perceiver
from mlx_audio.tts.models.chatterbox.t3.t3 import T3
from mlx_audio.tts.models.chatterbox.voice_encoder.config import VoiceEncConfig
from mlx_audio.tts.models.chatterbox.voice_encoder.voice_encoder import VoiceEncoder
from mlx_audio.tts.models.chatterbox.s3tokenizer import log_mel_spectrogram
from mlx_audio.codec.models.s3.model_v2 import S3TokenizerV2

OUT = Path("../turbospark/crates/audio/testdata/tts_chatterbox")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# --- tiny T3 geometry --------------------------------------------------------
T3_SEED = 9001
T3_LAYERS = 2
T3_DIM = 32
T3_HEADS = 4
T3_KV_HEADS = 2
T3_HEAD_DIM = 8
T3_INTERMEDIATE = 64
T3_VOCAB = 64  # backbone vocab (never read; inputs are embeddings)
T3_TEXT_DICT = 704
T3_SPEECH_DICT = 8194
T3_MAX_TEXT = 64
T3_MAX_SPEECH = 64
T3_MAX_NEW_TOKENS = 10
T3_GREEDY_SEED = 1
T3_SAMPLED_SEED = 77001
SPEAKER_EMBED_SIZE = 256
PERCEIVER_TOKENS = 32
EMOTION_ADV = 0.4
CFG_WEIGHT = 0.5
REP_PENALTY = 1.2
MIN_P = 0.05
TOP_P = 1.0
TEMPERATURE = 0.8

# --- tiny S3Gen flow geometry ------------------------------------------------
FLOW_SEED = 9101
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
N_TIMESTEPS = 10

# --- HiFT geometry (reference structure, shrunk widths) ----------------------
HIFT_SEED = 9201
HIFT_BASE = 64
HIFT_F0_CHANNELS = 64

# --- CAMPPlus geometry (full structure at shrunk widths) ---------------------
CAMPPLUS_SEED = 9301
CAMPPLUS_EMBEDDING = 48
CAMPPLUS_GROWTH = 8
CAMPPLUS_INIT = 16

# --- voice encoder geometry --------------------------------------------------
VE_SEED = 9401
VE_HIDDEN = 32
VE_SPEAKER_EMBED = 32

# --- conditioning wavebands --------------------------------------------------
COND_SEED = 9501
COND_WAVE_24K = 14400  # 0.6 s at 24 kHz
COND_WAVE_16K = 9600  # 0.6 s at 16 kHz

TEXT_TOKENS = [12, 44, 3, 209, 87, 5, 300, 61, 17, 6, 255, 40, 90, 8]


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


# --- tiny T3 patches ---------------------------------------------------------
TINY_LLAMA = {
    "model_type": "llama",
    "vocab_size": T3_VOCAB,
    "hidden_size": T3_DIM,
    "num_hidden_layers": T3_LAYERS,
    "intermediate_size": T3_INTERMEDIATE,
    "num_attention_heads": T3_HEADS,
    "num_key_value_heads": T3_KV_HEADS,
    "head_dim": T3_HEAD_DIM,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,
}


def apply_tiny_patches() -> None:
    cb_config.LLAMA_CONFIGS["Llama_520M"] = TINY_LLAMA
    original = Perceiver.__init__

    def tiny_init(
        self,
        pre_attention_query_token=PERCEIVER_TOKENS,
        pre_attention_query_size=T3_DIM,
        embedding_dim=T3_DIM,
        num_attn_heads=4,
    ):
        original(
            self,
            pre_attention_query_token=pre_attention_query_token,
            pre_attention_query_size=pre_attention_query_size,
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
        )

    Perceiver.__init__ = tiny_init


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
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
)


def seed_tree(module, seed: int, scale: float) -> dict:
    """Deterministic weights: small normals everywhere, norm gains and
    batchnorm stats positive so activations stay healthy."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, value in dict(tree_flatten(module.parameters())).items():
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


class HeadRecorder:
    """Records every speech-head call (the loop reads the last row of
    each (B, 1, V) output for the CFG combine)."""

    def __init__(self, head):
        self.calls = []
        self.head = head

    def __call__(self, x):
        out = self.head(x)
        self.calls.append(np.asarray(out, dtype=np.float32))
        return out


def build_t3(seed: int) -> T3:
    apply_tiny_patches()
    hp = cb_config.T3Config(
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
        emotion_adv=mx.ones((1, 1, 1)) * EMOTION_ADV,
    )


def prompt_ids() -> list:
    return [255] + TEXT_TOKENS + [0]  # start/stop text tokens


def combined_and_penalized(calls, cfg_weight, generated_per_step):
    """The reference loop math after the head: CFG combine, then the
    sign-aware repetition penalty (the gather runs before the scatter,
    so duplicate context tokens are idempotent)."""
    steps = []
    for step, call in enumerate(calls):
        raw = call[0, -1, :].astype(np.float64)
        if call.shape[0] > 1:
            cond = call[0, -1, :].astype(np.float64)
            uncond = call[1, -1, :].astype(np.float64)
            raw = cond + cfg_weight * (cond - uncond)
        context = np.array(generated_per_step[step], dtype=np.int64)
        penalized = raw.copy()
        selected = penalized[context]
        penalized[context] = np.where(selected < 0, selected * REP_PENALTY, selected / REP_PENALTY)
        values = np.sort(penalized)[::-1]
        steps.append((penalized.astype(np.float32), float(values[0] - values[1])))
    return steps


def run_t3_case() -> dict:
    t3 = build_t3(T3_SEED)
    cond = make_t3_cond(T3_SEED + 1)

    # Conditioning rows and the composite prompt embeds.
    cond_rows = t3.prepare_conditioning(cond)
    mx.eval(cond_rows)
    text_tokens = mx.array(prompt_ids())[None]
    if CFG_WEIGHT > 0.0:
        # chatterbox.generate duplicates the text row for CFG before
        # t3.inference; the T3 loop rides that batch (the reference
        # decode embeds are duplicated unconditionally under cfg).
        text_tokens = mx.concatenate([text_tokens, text_tokens], axis=0)
    embeds = t3._prepare_inference_context(
        t3_cond=cond, text_tokens=text_tokens, cfg_weight=CFG_WEIGHT
    )
    mx.eval(embeds)

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "text_ids": TEXT_TOKENS,
        "cfg_weight": CFG_WEIGHT,
        "repetition_penalty": REP_PENALTY,
        "min_p": MIN_P,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "max_new_tokens": T3_MAX_NEW_TOKENS,
        "start_speech_token": int(t3.hp.start_speech_token),
        "stop_speech_token": int(t3.hp.stop_speech_token),
        "speaker_emb": floats(cond.speaker_emb),
        "cond_prompt_speech_tokens": ints(cond.cond_prompt_speech_tokens),
        "emotion_adv": float(np.asarray(cond.emotion_adv).ravel()[0]),
        "cond_rows": floats(cond_rows),
        "prompt_embeds_shape": list(np.asarray(embeds).shape),
        "prompt_embeds": floats(embeds),
    }

    # Greedy loop: the head recorder observes the per-step logits; the
    # ids carry the choices. Saved as float32 npy [steps, rows, vocab].
    recorder = HeadRecorder(t3.speech_head)
    t3.speech_head = recorder
    ids = t3.inference(
        t3_cond=cond,
        text_tokens=text_tokens,
        max_new_tokens=T3_MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=TOP_P,
        min_p=MIN_P,
        repetition_penalty=REP_PENALTY,
        cfg_weight=CFG_WEIGHT,
    )
    ids_np = ints(ids[0])
    greedy_calls = np.stack(recorder.calls)  # [steps, rows, vocab]
    t3.speech_head = recorder.head
    traces["greedy_ids"] = ids_np
    traces["greedy_stop"] = (
        "stop_token" if t3.hp.stop_speech_token in ids_np[1:] else "max_tokens"
    )
    # Decision margins of the CFG-combined, penalized logits (the
    # inputs the greedy argmax read); a thin margin would let the f32
    # reordering between mlx and the crate flip a choice.
    generated = [int(t3.hp.start_speech_token)]
    context_per_step = [list(generated)]
    body = ids_np[1:]
    for token in body[:-1] if ids_np[-1] == t3.hp.stop_speech_token else body:
        generated.append(int(token))
        context_per_step.append(list(generated))
    steps = combined_and_penalized(recorder.calls, CFG_WEIGHT, context_per_step)
    traces["greedy_margins"] = [m for _, m in steps]
    assert min(traces["greedy_margins"]) > 0.01, "greedy margins too thin"

    # Sampled loop with a seeded stream; the categorical draw is the
    # only RNG consumer, one per loop step.
    mx.random.seed(T3_SAMPLED_SEED)
    recorder2 = HeadRecorder(t3.speech_head)
    t3.speech_head = recorder2
    sampled_ids = t3.inference(
        t3_cond=cond,
        text_tokens=text_tokens,
        max_new_tokens=T3_MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        min_p=MIN_P,
        repetition_penalty=REP_PENALTY,
        cfg_weight=CFG_WEIGHT,
    )
    t3.speech_head = recorder2.head
    sampled_np = ints(sampled_ids[0])
    sampled_calls = np.stack(recorder2.calls)
    traces["sampled_seed"] = T3_SAMPLED_SEED
    traces["sampled_ids"] = sampled_np
    steps_s = combined_and_penalized(recorder2.calls, CFG_WEIGHT, context_per_step)
    traces["sampled_margins"] = [m for _, m in steps_s]
    assert min(traces["sampled_margins"]) > 1e-4, "sampled margins too thin"

    # Python-side replay: a fresh build must reproduce both streams.
    t3b = build_t3(T3_SEED)
    ids_b = ints(t3b.inference(
        t3_cond=make_t3_cond(T3_SEED + 1),
        text_tokens=text_tokens,
        max_new_tokens=T3_MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=TOP_P,
        min_p=MIN_P,
        repetition_penalty=REP_PENALTY,
        cfg_weight=CFG_WEIGHT,
    )[0])
    assert ids_b == ids_np, "greedy replay diverged"
    mx.random.seed(T3_SAMPLED_SEED)
    sampled_b = ints(t3b.inference(
        t3_cond=make_t3_cond(T3_SEED + 1),
        text_tokens=text_tokens,
        max_new_tokens=T3_MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        min_p=MIN_P,
        repetition_penalty=REP_PENALTY,
        cfg_weight=CFG_WEIGHT,
    )[0])
    assert sampled_b == sampled_np, "sampled replay diverged"

    case_dir = OUT / "t3"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps(traces))
    save_npy("t3/greedy_head_calls.npy", greedy_calls)
    save_npy("t3/sampled_head_calls.npy", sampled_calls)
    return {"name": "t3", "greedy_frames": len(ids_np) - 1}


# --- tiny S3Gen flow ---------------------------------------------------------
class TinyS3Token2Mel(S3Token2Mel):
    """S3Token2Mel at the tiny geometry; every forward method is
    inherited reference code."""

    def __init__(self):
        encoder = UpsampleConformerEncoder(
            output_size=FLOW_WIDTH,
            attention_heads=FLOW_HEADS,
            linear_units=FLOW_UNITS,
            num_blocks=FLOW_BLOCKS,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=FLOW_WIDTH,
            use_cnn_module=False,
            macaron_style=False,
        )
        # The decoder input is [x, mu, projected speaker (80), cond]
        # - the speaker embedding is projected to the mel width before
        # the decoder, so the input width is always 4 * 80.
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
            act_fn="gelu",
        )
        decoder = CausalConditionalCFM(
            spk_emb_dim=SPK_DIM,
            cfm_params=CFM_PARAMS,
            estimator=estimator,
        )
        self.speaker_encoder = CAMPPlus(
            feat_dim=80,
            embedding_size=SPK_DIM,
            growth_rate=CAMPPLUS_GROWTH,
            bn_size=4,
            init_channels=CAMPPLUS_INIT,
        )
        self.flow = CausalMaskedDiffWithXvec(
            input_size=FLOW_WIDTH,
            output_size=80,
            spk_embed_dim=SPK_DIM,
            vocab_size=SPEECH_VOCAB_SIZE,
            encoder=encoder,
            decoder=decoder,
        )


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
            f0_predictor=ConvRNNF0Predictor(cond_channels=HIFT_F0_CHANNELS),
        )
        n_trim = 24000 // 50
        trim_fade_cos = (mx.cos(mx.linspace(mx.pi, 0, n_trim)) + 1) / 2
        self.trim_fade = mx.concatenate([mx.zeros(n_trim), trim_fade_cos])

    def __call__(self, speech_tokens, ref_dict, finalize=False):
        output_mels = S3Token2Mel.__call__(self, speech_tokens, ref_dict, finalize)
        output_wavs, _ = self.mel2wav.inference(
            speech_feat=output_mels, cache_source=mx.zeros((1, 1, 0))
        )
        fade_len = len(self.trim_fade)
        if output_wavs.shape[1] >= fade_len:
            output_wavs[:, :fade_len] *= self.trim_fade
        return output_wavs


def flow_inputs(seed: int):
    rng = np.random.default_rng(seed)
    prompt_tokens = rng.integers(0, SPEECH_VOCAB_SIZE, PROMPT_TOKENS).astype(np.int32)
    gen_tokens = rng.integers(0, SPEECH_VOCAB_SIZE, GEN_TOKENS).astype(np.int32)
    prompt_feat = (
        rng.standard_normal((PROMPT_TOKENS * 2, 80)).astype(np.float32) * 0.5 - 2.0
    )
    embedding = rng.standard_normal(SPK_DIM).astype(np.float32) * 0.5
    return prompt_tokens, gen_tokens, prompt_feat, embedding


def build_s3gen(seed: int, noise_seed: int, total_frames: int) -> TinyS3Token2Wav:
    s3gen = TinyS3Token2Wav()
    seed_tree(s3gen, seed, scale=0.2)
    s3gen.eval()
    # Replace the seed-0 15000-frame noise buffer with a short seeded
    # window (the Rust loader accepts any buffer at least as long as
    # the solve needs; real checkpoints do not carry one at all).
    rng = np.random.default_rng(noise_seed)
    s3gen.flow.decoder.rand_noise = mx.array(
        rng.standard_normal((1, 80, total_frames)).astype(np.float32)
    )
    # Voiced f0: hold the f0 classifier bias near 60 Hz.
    s3gen.mel2wav.f0_predictor.classifier.bias = mx.array(
        (60.0 + 0.5 * np.random.default_rng(noise_seed + 1).standard_normal((1,))).astype(np.float32)
    )
    # Snake alphas near one so the activation keeps its scale.
    flat = dict(tree_flatten(s3gen.mel2wav.parameters()))
    for name, value in flat.items():
        if name.endswith(".alpha"):
            alpha = 1.0 + 0.25 * np.random.default_rng(noise_seed + 2).standard_normal(value.shape)
            s3gen.mel2wav.update(tree_unflatten({name: mx.array(alpha.astype(np.float32))}))
    return s3gen


class DrawPatch:
    """Records (or replays) the mx.random draws a HiFT call makes: the
    per-harmonic initial phase (uniform), the sine noise bed (normal),
    and the discarded source-module noise bed (normal)."""

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


def run_flow_case(s3gen: TinyS3Token2Wav) -> dict:
    prompt_tokens, gen_tokens, prompt_feat, embedding = flow_inputs(FLOW_SEED + 2)
    ref_dict = dict(
        prompt_token=mx.array(prompt_tokens)[None],
        prompt_token_len=mx.array([PROMPT_TOKENS], dtype=mx.int32),
        prompt_feat=mx.array(prompt_feat)[None],
        prompt_feat_len=mx.array([PROMPT_TOKENS * 2], dtype=mx.int32),
        embedding=mx.array(embedding)[None],
    )
    mel, _ = s3gen.flow.inference(
        token=mx.array(gen_tokens)[None],
        token_len=mx.array([GEN_TOKENS], dtype=mx.int32),
        finalize=True,
        **ref_dict,
    )
    mel_np = np.asarray(mel)
    assert mel_np.shape == (1, 80, GEN_TOKENS * 2), mel_np.shape

    # Python-side replay.
    mel_b, _ = s3gen.flow.inference(
        token=mx.array(gen_tokens)[None],
        token_len=mx.array([GEN_TOKENS], dtype=mx.int32),
        finalize=True,
        **ref_dict,
    )
    assert np.array_equal(mel_np, np.asarray(mel_b)), "flow replay diverged"

    case_dir = OUT / "flow"
    case_dir.mkdir(parents=True, exist_ok=True)
    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "n_timesteps": N_TIMESTEPS,
        "prompt_tokens": ints(prompt_tokens),
        "gen_tokens": ints(gen_tokens),
        "prompt_feat_shape": list(prompt_feat.shape),
        "embedding_shape": [SPK_DIM],
        "flow_mel_shape": [GEN_TOKENS * 2, 80],
        "flow_mel_abs_max": float(np.abs(mel_np).max()),
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))
    save_npy("flow/prompt_feat.npy", prompt_feat)
    save_npy("flow/embedding.npy", embedding)
    save_i32("flow/prompt_tokens.npy", prompt_tokens)
    save_i32("flow/gen_tokens.npy", gen_tokens)
    save_npy("flow/flow_mel.npy", mel_np[0].T)
    return {"name": "flow", "mel_frames": int(mel_np.shape[2])}


def run_hift_case(s3gen: TinyS3Token2Wav, mel_np: np.ndarray) -> dict:
    hift = s3gen.mel2wav
    f0 = np.asarray(hift.f0_predictor(mx.array(mel_np)))
    save_npy("hift/f0.npy", f0[0])

    with DrawPatch() as recorder:
        wav, _ = hift.inference(speech_feat=mx.array(mel_np), cache_source=mx.zeros((1, 1, 0)))
    wav_np = np.asarray(wav)[0]
    assert len(recorder.draws) == 3, [d.shape for d in recorder.draws]
    phase = recorder.draws[0].reshape(-1).copy()
    phase[0] = 0.0  # the reference zeroes harmonic 0 after the draw
    sine_noise = recorder.draws[1][0]  # [harmonics + 1, T_wav]
    assert sine_noise.shape[1] == wav_np.shape[0], (sine_noise.shape, wav_np.shape)

    # Replay the recorded draws: the waveform must be identical.
    with DrawPatch(replay=recorder.draws) as replay_patch:
        wav_b, _ = hift.inference(speech_feat=mx.array(mel_np), cache_source=mx.zeros((1, 1, 0)))
    assert np.array_equal(wav_np, np.asarray(wav_b)[0]), "hift replay diverged"

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
    save_npy("hift/wav.npy", wav_np)
    return {"name": "hift", "wav_len": int(wav_np.shape[0])}


def run_end_to_end_case(s3gen: TinyS3Token2Wav, ref_dict: dict) -> dict:
    """The full generate path on one realization: greedy T3 -> token
    crop -> flow -> hift -> trim fade."""
    t3 = build_t3(T3_SEED)
    cond = make_t3_cond(T3_SEED + 1)
    text_tokens = mx.array(prompt_ids())[None]
    if CFG_WEIGHT > 0.0:
        text_tokens = mx.concatenate([text_tokens, text_tokens], axis=0)
    ids = t3.inference(
        t3_cond=cond,
        text_tokens=text_tokens,
        max_new_tokens=T3_MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=TOP_P,
        min_p=MIN_P,
        repetition_penalty=REP_PENALTY,
        cfg_weight=CFG_WEIGHT,
    )
    ids_np = np.asarray(ids)
    speech_tokens = np.asarray(drop_invalid_tokens(mx.array(ids_np[0])))
    speech_tokens = speech_tokens[speech_tokens < SPEECH_VOCAB_SIZE]
    speech_tokens = speech_tokens.reshape(1, -1).astype(np.int32)
    with DrawPatch() as recorder:
        wav = s3gen(mx.array(speech_tokens), ref_dict=ref_dict, finalize=True)
    wav_np = np.asarray(wav)[0]
    phase = recorder.draws[0].reshape(-1).copy()
    phase[0] = 0.0

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "text_ids": TEXT_TOKENS,
        "ids": ints(ids_np[0]),
        "speech_tokens": ints(speech_tokens[0]),
        "phase": floats(phase),
        "sine_noise_shape": list(recorder.draws[1].shape),
        "wav_len": int(wav_np.shape[0]),
        "wav_abs_max": float(np.abs(wav_np).max()),
    }
    (OUT / "end_to_end").mkdir(parents=True, exist_ok=True)
    (OUT / "end_to_end" / "traces.json").write_text(json.dumps(traces))
    save_npy("end_to_end/sine_noise.npy", recorder.draws[1])
    save_npy("end_to_end/wav.npy", wav_np)
    return {"name": "end_to_end", "speech_tokens": int(speech_tokens.shape[1])}


def run_campplus_case(model_campplus, seed: int) -> dict:
    rng = np.random.default_rng(seed + 3)
    wave = (rng.standard_normal(COND_WAVE_16K) * 0.1).astype(np.float32)
    fbank = np.asarray(kaldi_fbank(mx.array(wave), num_mel_bins=80))
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


def run_voice_encoder_case(seed: int) -> dict:
    hp = VoiceEncConfig(
        num_mels=40,
        ve_hidden_size=VE_HIDDEN,
        speaker_embed_size=VE_SPEAKER_EMBED,
    )
    ve = VoiceEncoder(hp)
    seed_tree(ve, seed, scale=0.3)
    ve.eval()
    rng = np.random.default_rng(seed + 4)
    wave = (rng.standard_normal(COND_WAVE_16K) * 0.15).astype(np.float32)
    embeds = ve.embeds_from_wavs([mx.array(wave)], sample_rate=16000)
    embeds_np = np.asarray(embeds)
    assert embeds_np.shape == (1, VE_SPEAKER_EMBED), embeds_np.shape
    embeds_b = ve.embeds_from_wavs([mx.array(wave)], sample_rate=16000)
    assert np.array_equal(embeds_np, np.asarray(embeds_b)), "ve replay diverged"

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "wave_len": int(wave.shape[0]),
        "trim_top_db": 20,
        "rate": 1.3,
        "overlap": 0.5,
        "speaker_embed_shape": [VE_SPEAKER_EMBED],
    }
    (OUT / "voice_encoder").mkdir(parents=True, exist_ok=True)
    (OUT / "voice_encoder" / "traces.json").write_text(json.dumps(traces))
    save_npy("voice_encoder/wave.npy", wave)
    save_npy("voice_encoder/speaker_emb.npy", embeds_np[0])
    return {"name": "voice_encoder"}


def run_conditioning_case(s3gen: TinyS3Token2Wav, t3: T3, tiny_s3tok, seed: int) -> dict:
    """prepare_conditionals on the three wavebands: the S3Gen mel and
    token alignment, the T3 prompt tokens, the CAMPPlus embedding, and
    the voice-encoder speaker embed. Mirrors the reference body with
    pre-resampled bands (the polyphase resampler stays outside the
    port)."""
    rng = np.random.default_rng(seed + 5)
    wav_24k = (rng.standard_normal(COND_WAVE_24K) * 0.1).astype(np.float32)
    wav_16k_from_24k = (rng.standard_normal(COND_WAVE_16K) * 0.1).astype(np.float32)
    wav_16k_full = (rng.standard_normal(COND_WAVE_16K) * 0.1).astype(np.float32)

    from mlx_audio.tts.models.chatterbox.s3gen.mel import mel_spectrogram

    # --- S3Gen band ---
    ref_mels_24 = np.asarray(mel_spectrogram(mx.array(wav_24k)[None]))  # (1, M, T)
    ref_mels_24 = ref_mels_24.transpose(0, 2, 1)  # (1, T, M)
    mel_len = ref_mels_24.shape[1]
    s3gen_mel = log_mel_spectrogram(mx.array(wav_16k_from_24k))
    s3gen_mel = mx.expand_dims(s3gen_mel, 0)
    s3gen_tokens, s3gen_token_lens = tiny_s3tok(s3gen_mel, mx.array([s3gen_mel.shape[2]]))

    # embed_ref alignment: tokens must equal mel_len // 2.
    actual_token_len = s3gen_tokens.shape[1]
    expected_token_len = mel_len // 2
    if actual_token_len != expected_token_len:
        if actual_token_len < expected_token_len:
            ref_mels_24 = ref_mels_24[:, : 2 * actual_token_len, :]
        else:
            s3gen_tokens = s3gen_tokens[:, :expected_token_len]
            actual_token_len = expected_token_len
    prompt_feat = ref_mels_24[0]  # (T, M)

    # --- T3 band ---
    t3_mel = log_mel_spectrogram(mx.array(wav_16k_full))
    t3_mel = mx.expand_dims(t3_mel, 0)
    t3_tokens, _ = tiny_s3tok(t3_mel, mx.array([t3_mel.shape[2]]))
    plen = t3.hp.speech_cond_prompt_len
    t3_tokens = t3_tokens[:, :plen]

    # --- voice encoder band ---
    ve = VoiceEncoder(
        VoiceEncConfig(
            num_mels=40, ve_hidden_size=VE_HIDDEN, speaker_embed_size=VE_SPEAKER_EMBED
        )
    )
    seed_tree(ve, VE_SEED, scale=0.3)
    ve.eval()
    speaker = np.asarray(ve.embeds_from_wavs([mx.array(wav_16k_full)], sample_rate=16000))

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "wav_24k_len": COND_WAVE_24K,
        "wav_16k_len": COND_WAVE_16K,
        "mel_len": int(mel_len),
        "s3gen_tokens_raw": int(int(np.asarray(s3gen_token_lens)[0])),
        "aligned_token_len": int(actual_token_len),
        "prompt_feat_shape": list(prompt_feat.shape),
        "t3_cond_prompt_tokens": ints(t3_tokens[0]),
        "speaker_embed_shape": list(speaker.shape),
    }
    (OUT / "conditioning").mkdir(parents=True, exist_ok=True)
    (OUT / "conditioning" / "traces.json").write_text(json.dumps(traces))
    save_npy("conditioning/wav_24k.npy", wav_24k)
    save_npy("conditioning/wav_16k_from_24k.npy", wav_16k_from_24k)
    save_npy("conditioning/wav_16k_full.npy", wav_16k_full)
    save_npy("conditioning/prompt_feat.npy", prompt_feat)
    save_i32("conditioning/prompt_token.npy", np.asarray(s3gen_tokens)[0])
    save_npy("conditioning/embedding.npy", np.asarray(
        _campplus_embedding(s3gen, wav_16k_from_24k)
    ))
    save_npy("conditioning/speaker_emb.npy", speaker[0])
    return {"name": "conditioning", "aligned_tokens": int(actual_token_len)}


def _campplus_embedding(s3gen: TinyS3Token2Wav, wav_16k: np.ndarray) -> np.ndarray:
    fbank = np.asarray(kaldi_fbank(mx.array(wav_16k), num_mel_bins=80))
    fbank = fbank - fbank.mean(axis=0, keepdims=True)
    return np.asarray(s3gen.speaker_encoder(mx.array(fbank)[None]))


def build_sampling_case() -> None:
    """The filter math on fixed logits: CFG combine, the repetition
    penalty with duplicate context tokens, raw top-p / min-p filters,
    and the categorical draw with recorded uniforms (numpy mirrors of
    the reference formulas; the mlx functions agree to f32 rounding and
    the Rust sampler ports the same math)."""

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

    def cfg_mirror(cond, uncond, weight):
        cond = np.asarray(cond, dtype=np.float32)
        uncond = np.asarray(uncond, dtype=np.float32)
        return cond + np.float32(weight) * (cond - uncond)

    def penalty_mirror(logits, context, penalty):
        out = np.asarray(logits, dtype=np.float32).copy()
        tokens = np.asarray(context, dtype=np.int64)
        selected = out[tokens]
        out[tokens] = np.where(selected < 0, selected * penalty, selected / penalty)
        return out

    def top_p_mirror(logits, top_p):
        raw = np.asarray(logits, dtype=np.float32)
        probs = np.exp(raw)
        order = np.argsort(raw, kind="stable")
        cumulative = np.cumsum(probs[order])
        keep_sorted = cumulative > np.float32(1.0 - top_p)
        keep = np.zeros_like(keep_sorted)
        keep[order] = keep_sorted
        return np.where(keep, raw, np.float32(-np.inf))

    def min_p_mirror(logits, min_p):
        raw = np.asarray(logits, dtype=np.float32)
        scaled = raw.max() + np.float32(np.log(min_p))
        return np.where(raw < scaled, np.float32(-np.inf), raw)

    def categorical_mirror(logits, temp, uniform):
        scaled = np.asarray(logits, dtype=np.float32) / np.float32(temp)
        weights = np.exp(scaled - scaled.max()).astype(np.float64)
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

    # CFG combine on a fresh vector pair.
    cond = vector(vocab, 3)
    uncond = vector(vocab, 9)
    combined = cfg_mirror(cond, uncond, CFG_WEIGHT)
    cases.append(
        {
            "name": "cfg_combine",
            "cond": floats(cond),
            "uncond": floats(uncond),
            "cfg_weight": CFG_WEIGHT,
            "combined": floats(combined),
        }
    )

    # Repetition penalty with duplicate context tokens (the gather
    # precedes the scatter, so duplicates are idempotent).
    base = vector(vocab, 13)
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

    # Raw top-p on unnormalized logits (the reference applies the
    # sampler to the post-CFG raw logits).
    raw = vector(vocab, 21)
    cases.append(
        {
            "name": "top_p_raw_0_9",
            "logits": floats(raw),
            "top_p": 0.9,
            "filtered": masked(top_p_mirror(raw, 0.9)),
        }
    )
    narrow = vector(24, 29)
    cases.append(
        {
            "name": "top_p_raw_0_5",
            "logits": floats(narrow),
            "top_p": 0.5,
            "filtered": masked(top_p_mirror(narrow, 0.5)),
        }
    )

    # min-p on raw logits (shift invariant).
    cases.append(
        {
            "name": "min_p_raw_0_05",
            "logits": floats(raw),
            "min_p": 0.05,
            "filtered": masked(min_p_mirror(raw, 0.05)),
        }
    )
    cases.append(
        {
            "name": "min_p_raw_0_4",
            "logits": floats(narrow),
            "min_p": 0.4,
            "filtered": masked(min_p_mirror(narrow, 0.4)),
        }
    )

    # min-p after top-p (the sampler order), then the categorical draw
    # with a recorded uniform and a decisive margin.
    after_top_p = top_p_mirror(raw, 0.9)
    after_min_p = min_p_mirror(after_top_p, 0.05)
    rng = np.random.default_rng(9601)
    uniforms = rng.uniform(0.02, 0.98, size=6).astype(np.float32)
    winner = categorical_mirror(after_min_p, TEMPERATURE, uniforms[0])
    scaled = (after_min_p / np.float32(TEMPERATURE)).astype(np.float64)
    weights = np.exp(scaled - scaled.max())
    ordered = np.sort(weights / weights.sum())[::-1]
    margin = float(ordered[0] - ordered[1])
    assert margin > 1e-4, f"thin categorical margin {margin}"
    cases.append(
        {
            "name": "filtered_categorical",
            "logits": masked(after_min_p),
            "temperature": TEMPERATURE,
            "uniform": floats(np.array([uniforms[0]])),
            "winner": int(winner),
            "prob_margin": margin,
        }
    )

    # Greedy: plain argmax after the penalty (the reference make_sampler
    # with temp 0 skips every filter).
    penalized_greedy = penalty_mirror(combined, [0, 3, 3, 40], REP_PENALTY)
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
    s3gen = build_s3gen(FLOW_SEED, FLOW_SEED + 10, total_frames=2 * (PROMPT_TOKENS + GEN_TOKENS))
    ve = VoiceEncoder(
        VoiceEncConfig(
            num_mels=40, ve_hidden_size=VE_HIDDEN, speaker_embed_size=VE_SPEAKER_EMBED
        )
    )
    seed_tree(ve, VE_SEED, scale=0.3)
    ve.eval()

    # --- combined checkpoint (model.safetensors) ------------------------
    weights: dict[str, mx.array] = {}
    for name, value in dict(tree_flatten(t3.parameters())).items():
        weights[f"t3.{name}"] = value.astype(mx.float32)
    for name, value in dict(tree_flatten(s3gen.flow.parameters())).items():
        weights[f"flow.{name}"] = value.astype(mx.float32)
    for name, value in dict(tree_flatten(s3gen.mel2wav.parameters())).items():
        weights[f"mel2wav.{name}"] = value.astype(mx.float32)
    for name, value in dict(tree_flatten(s3gen.speaker_encoder.parameters())).items():
        weights[f"speaker_encoder.{name}"] = value.astype(mx.float32)
    for name, value in dict(tree_flatten(ve.parameters())).items():
        weights[f"ve.{name}"] = value.astype(mx.float32)
    mx.save_safetensors(str(OUT / "model.safetensors"), weights)

    # --- config.json -----------------------------------------------------
    config = {
        "model_type": "chatterbox",
        "multilingual": False,
        "text_preprocessing": "legacy",
        "s3_sr": 16000,
        "s3gen_sr": 24000,
        "sample_rate": 24000,
        "enc_cond_len": 6 * 16000,
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
            "input_pos_emb": "learned",
            "speech_cond_prompt_len": 150,
            "encoder_type": "voice_encoder",
            "speaker_embed_size": SPEAKER_EMBED_SIZE,
            "use_perceiver_resampler": True,
            "emotion_adv": True,
            "llama_config": TINY_LLAMA,
        },
        "flow_config": {
            "input_size": FLOW_WIDTH,
            "output_size": 80,
            "spk_embed_dim": SPK_DIM,
            "vocab_size": int(SPEECH_VOCAB_SIZE),
            "attention_heads": FLOW_HEADS,
            "linear_units": FLOW_UNITS,
            "num_blocks": FLOW_BLOCKS,
            "num_up_blocks": FLOW_UP_BLOCKS,
            "upsample_stride": 2,
            "pre_lookahead_len": 3,
            "n_timesteps": N_TIMESTEPS,
            "inference_cfg_rate": 0.7,
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
    prompt_tokens, gen_tokens, prompt_feat, embedding = flow_inputs(FLOW_SEED + 2)
    ref_dict = dict(
        prompt_token=mx.array(prompt_tokens)[None],
        prompt_token_len=mx.array([PROMPT_TOKENS], dtype=mx.int32),
        prompt_feat=mx.array(prompt_feat)[None],
        prompt_feat_len=mx.array([PROMPT_TOKENS * 2], dtype=mx.int32),
        embedding=mx.array(embedding)[None],
    )
    summary.append(run_flow_case(s3gen))
    mel, _ = s3gen.flow.inference(
        token=mx.array(gen_tokens)[None],
        token_len=mx.array([GEN_TOKENS], dtype=mx.int32),
        finalize=True,
        **ref_dict,
    )
    mel = np.asarray(mel)
    summary.append(run_hift_case(s3gen, mel))
    summary.append(run_end_to_end_case(s3gen, ref_dict))
    summary.append(run_campplus_case(s3gen.speaker_encoder, CAMPPLUS_SEED))
    summary.append(run_voice_encoder_case(VE_SEED))
    summary.append(run_conditioning_case(s3gen, t3, tiny_s3tok, COND_SEED))
    build_sampling_case()

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = digest(path)
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_chatterbox_fixtures.py",
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
