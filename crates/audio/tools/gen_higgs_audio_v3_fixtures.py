#!/usr/bin/env python3
"""Regenerate the Higgs Audio v3 TTS family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/
higgs_audio_v3, mlx 0.32.3). The reference modules are imported directly;
the TTS model builds the true v3 Model over a seeded tiny qwen3 backbone,
the true HiggsAudioV3PromptBuilder and generation.step run the prompt and
sampler paths, and the tiny codec stands in for the network
HiggsAudioTokenizer with the same decode stack (RVQ sum, fc2, acoustic
decoder) the codec family fixture pins.

This is the TTS family fixture generator; the codec fixture lives in
gen_higgs_fixtures.py.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_higgs_audio_v3_fixtures.py

Outputs into crates/audio/testdata/tts_higgs_audio_v3/:

- generate/model.safetensors + generate/config.json
    Seeded tiny v3 Model (2 qwen3 layers, hidden 32, 4 heads over 2 kv
    heads, head_dim 8, text vocab 64, 4 codebooks of 10 with BOC 8 and
    EOC 9). Weights are stored F16 (the traces are recorded after
    upcasting the same stored values, so the Rust f32 path sees
    identical inputs). Tensor names carry the MLX module tree
    (`backbone.*`, `multimodal_embedding`).
- generate/traces.json
    The pinned voice-clone prompt (special ids, mock-tokenized text,
    delayed reference codes, prompt token ids with placeholders, the
    full embedding block), a forward pin (last hidden row and audio
    logits), and two greedy scenarios: an EOC ramp stop and a max-steps
    stop, each with the raw delayed frames, the aligned codes, the stop
    reason, and the faded tiny-codec waveform.
- codec/model.safetensors + codec/config.json
    Seeded tiny codec in the HiggsAudioTokenizer layout (3 RVQ books of
    10 classes, latent 8, decoder input 6), the same geometry the codec
    family fixture uses.
- delay/traces.json
    apply_delay_pattern and reverse_delay_pattern on fixed id grids.
- sampling/traces.json
    sample_independent greedy draws on fixed logit blocks and the
    top-k/top-p masked SCALED logits (the reference `_apply_top_k` /
    `_apply_top_p` links: log-softmax, filter, map the -inf entries
    back). The temperature > 0 categorical draw is reference-only: the
    Rust categorical draw is the crate-local threefry path and is not
    bit-reproducible against mx.random.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The
backbone seed at the bottom is pinned to the first seed of the search
below whose greedy runs satisfy both scenario shapes; regeneration
reproduces the same fixtures.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.codec.models.dacvae.codec import WNConv1d
from mlx_audio.codec.models.higgs_audio.dac import (
    AcousticDecoderBlock,
    AcousticEncoderBlock,
    ResidualVectorQuantizer,
    Snake1d,
    VectorQuantizer,
)
from mlx_audio.codec.models.higgs_audio.higgs_audio import (
    AcousticDecoder,
    AcousticEncoder,
)
from mlx_audio.lm.models.cache import make_prompt_cache
from mlx_audio.tts.models.higgs_audio_v3.config import HiggsAudioV3Config
from mlx_audio.tts.models.higgs_audio_v3.generation import (
    HiggsSamplerState,
    apply_delay_pattern,
    reverse_delay_pattern,
    step,
)
from mlx_audio.tts.models.higgs_audio_v3.model import Model
from mlx_audio.tts.models.higgs_audio_v3.prompt import (
    HiggsAudioV3PromptBuilder,
    ReferenceCodes,
)
from mlx_audio.lm.sample_utils import apply_top_k as _apply_top_k_logprobs
from mlx_audio.lm.sample_utils import apply_top_p as _apply_top_p_logprobs

OUT = Path("../turbospark/crates/audio/testdata/tts_higgs_audio_v3")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

K = 4
CODEBOOK_SIZE = 10
BOC = 8
EOC = 9

SAMPLE_RATE = 24000
FADE_IN_MS = 30.0
FADE_OUT_MS = 15.0

# The EOC-row amplification and the pinned backbone seed: first seed of
# the search below whose greedy runs satisfy both scenario shapes.
AMPLIFY_EOC = 1.8
BACKBONE_SEED = 9
EOC_BUDGET = 60
MAXSTEPS_BUDGET = 8

TEXT_CONFIG = dict(
    model_type="qwen3",
    hidden_size=32,
    num_hidden_layers=2,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    rope_theta=10000.0,
    head_dim=8,
    rms_norm_eps=1e-6,
    vocab_size=64,
    tie_word_embeddings=True,
    rope_scaling=None,
)

CONFIG = {
    "model_type": "higgs_audio_v3",
    "text_config": TEXT_CONFIG,
    "audio_token_id": -100,
    "audio_num_codebooks": K,
    "audio_codebook_size": CODEBOOK_SIZE,
    "audio_boc_token_id": BOC,
    "audio_eoc_token_id": EOC,
    "use_delay_pattern": True,
    "sample_rate": SAMPLE_RATE,
}

SPECIAL = {
    "<|tts|>": 1,
    "<|ref_audio|>": 2,
    "<|text|>": 3,
    "<|audio|>": 4,
    "<|ref_text|>": 5,
}

REF_TEXT = "Ref voice."
TARGET_TEXT = "Hello there."


class FakeTokenizer:
    """The mock tokenizer: specials map to fixed ids, every other
    character to a deterministic id. The Rust side consumes the
    recorded id sequences."""

    def __init__(self):
        self.vocab = dict(SPECIAL)

    def get_added_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10 + (ord(ch) % 20) for ch in text]


class TinyAcousticDecoder(AcousticDecoder):
    """AcousticDecoder.__init__ at decoder_hidden=32, decoder input 6."""

    _STRIDES = [8, 5, 4, 2, 3]
    _IN_CHANNELS = [32, 16, 8, 4, 2]
    _OUT_CHANNELS = [16, 8, 4, 2, 1]

    def __init__(self):
        self.conv1 = WNConv1d(6, self._IN_CHANNELS[0], kernel_size=7, padding=3, norm="none")
        self.block = [
            AcousticDecoderBlock(self._IN_CHANNELS[i], self._OUT_CHANNELS[i], self._STRIDES[i])
            for i in range(len(self._STRIDES))
        ]
        self.snake1 = Snake1d(self._OUT_CHANNELS[-1])
        self.conv2 = WNConv1d(self._OUT_CHANNELS[-1], 1, kernel_size=7, padding=3, norm="none")


class TinyAcousticEncoder(AcousticEncoder):
    """AcousticEncoder.__init__ at encoder_hidden=4 (weight-tree layout
    parity only; the semantic encode path is out of contract)."""

    _STRIDES = [8, 5, 4, 2, 3]
    _CHANNELS = [4, 8, 16, 32, 64, 128]

    def __init__(self):
        c = self._CHANNELS
        self.conv1 = WNConv1d(1, c[0], kernel_size=7, padding=3, norm="none")
        self.block = [
            AcousticEncoderBlock(c[i], c[i + 1], self._STRIDES[i])
            for i in range(len(self._STRIDES))
        ]
        self.snake1 = Snake1d(c[-1])
        self.conv2 = WNConv1d(c[-1], 6, kernel_size=3, padding=1, norm="none")


class TinyVq(VectorQuantizer):
    def __init__(self):
        super().__init__(latent_dim=8, codebook_size=CODEBOOK_SIZE, codebook_dim=4)


class TinyRvq(ResidualVectorQuantizer):
    def __init__(self):
        # One quantizer per family codebook so the aligned diagonal
        # decodes (the deployed v3 tokenizer matches its codec books to
        # the codebook count).
        self.quantizers = [TinyVq() for _ in range(K)]


class TinyCodec:
    """Stand-in for the HiggsAudioTokenizer: decode through the real tiny
    stacks in consumption order (RVQ sum, fc2, acoustic decoder). The
    reference `_decode_audio` passes the [T, K] aligned codes directly."""

    decode_upsample_rate = 24000

    def __init__(self, dec, fc2, rvq):
        self.dec = dec
        self.fc2 = fc2
        self.rvq = rvq

    def decode(self, tokens):
        z = self.rvq.decode(tokens[None])
        return self.dec(self.fc2(z))[0, :, 0]


def seed_weights(model, seed: int, scale: float = 0.3) -> dict:
    flat = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(seed)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * scale).astype(np.float32))
        for name, v in flat.items()
    }
    model.load_weights(list(seeded.items()))
    mx.eval(model)
    return seeded


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def masked_floats(arr) -> list:
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(arr, dtype=np.float32).ravel().tolist()
    ]


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def build_model(seed: int):
    config = HiggsAudioV3Config.from_dict(CONFIG)
    mx.random.seed(seed)
    model = Model(config)
    seed_weights(model, seed)
    # Amplify the per-codebook EOC rows of the fused embedding table (the
    # as_linear head reads output class j from row j) so the greedy EOC
    # scenario lands inside a short budget.
    table = np.array(model.multimodal_embedding.weight, dtype=np.float32)
    for codebook in range(K):
        table[codebook * CODEBOOK_SIZE + EOC, :] *= AMPLIFY_EOC
    model.multimodal_embedding.weight = mx.array(table)
    mx.eval(model)
    stored = {
        name: value.astype(mx.float16) for name, value in tree_flatten(model.parameters())
    }
    stored["multimodal_embedding.weight"] = mx.array(table).astype(mx.float16)
    # Reload the upcast F16 stored values so every trace records exactly
    # what the Rust f32 path reads back from the file (llama fixture
    # convention).
    model.update(
        tree_unflatten({k: v.astype(mx.float32) for k, v in stored.items()})
    )
    mx.eval(model)
    return model, stored


def raw_frames(model, prompt_embeds, limit):
    """The raw delayed rows through the true prefill, step sampler, and
    backbone decode (the reference `generate` loop, batch of one).
    Returns the rows and whether the EOC countdown closed the loop."""
    cache = make_prompt_cache(model)
    dummy = mx.zeros((1, prompt_embeds.shape[1]), dtype=mx.int32)
    hidden = model.backbone(dummy, cache=cache, input_embeddings=prompt_embeds)
    last_hidden = hidden[:, -1, :]
    state = HiggsSamplerState(num_codebooks=K)
    rows = []
    stopped_eoc = False
    for _ in range(limit):
        logits = model._audio_logits(last_hidden)[0]
        codes = step(
            logits,
            state,
            temperature=0.0,
            top_p=None,
            top_k=None,
            boc_id=BOC,
            eoc_id=EOC,
        )
        rows.append(ints(codes))
        if state.generation_done:
            stopped_eoc = True
            break
        next_embed = model._embed_audio_codes(codes)[None]
        hidden = model.backbone(
            mx.zeros((1, 1), dtype=mx.int32), cache=cache, input_embeddings=next_embed
        )
        last_hidden = hidden[:, -1, :]
    return rows, stopped_eoc


def fades(pcm: np.ndarray) -> np.ndarray:
    out = pcm.copy()
    n_in = int(FADE_IN_MS * SAMPLE_RATE / 1000.0)
    n_out = int(FADE_OUT_MS * SAMPLE_RATE / 1000.0)
    if n_in > 0 and out.size > n_in:
        out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 0 and out.size > n_out:
        out[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return out


def decode_audio(codec, delayed_rows):
    """The reference `_decode_audio` for one sequence."""
    if not delayed_rows:
        return np.zeros((0,), dtype=np.float32)
    delayed = mx.array(delayed_rows, dtype=mx.int32)
    if delayed.shape[0] < K:
        return np.zeros((0,), dtype=np.float32)
    raw_codes = reverse_delay_pattern(delayed)
    return np.asarray(codec.decode(raw_codes), dtype=np.float32).reshape(-1)


def scenario_record(model, codec, prompt_embeds, limit, name):
    frames, stopped_eoc = raw_frames(model, prompt_embeds, limit)
    stopped = "eoc-ramp" if stopped_eoc else "max-steps"
    pcm = decode_audio(codec, frames)
    if frames:
        aligned = [ints(row) for row in reverse_delay_pattern(mx.array(frames, dtype=mx.int32))]
    else:
        aligned = []
    return {
        "name": name,
        "max_new_frames": limit,
        "num_frames_raw": len(frames),
        "num_frames_aligned": len(aligned),
        "stop_reason": stopped,
        "frames": frames,
        "codes": aligned,
        "waveform_faded": floats(fades(pcm)),
    }


def build_prompt(model, tokenizer, ref_raw):
    """The reference prompt construction over a pre-delayed reference."""
    builder = HiggsAudioV3PromptBuilder(tokenizer)
    model._tokenizer = tokenizer
    model._prompt_builder = builder
    delayed = apply_delay_pattern(mx.array(ref_raw, dtype=mx.int32), boc_id=BOC, eoc_id=EOC)
    references = [ReferenceCodes(codes=delayed, text=REF_TEXT)]
    parts = builder.build_prompt(TARGET_TEXT, references=references)
    embeds, prompt_tokens = model._build_prompt_embeddings(TARGET_TEXT, references)
    return builder, delayed, parts, embeds, prompt_tokens


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- tiny codec (independent of the backbone search) ----------------
    dec = TinyAcousticDecoder()
    enc = TinyAcousticEncoder()
    rvq = TinyRvq()
    fc2 = nn.Linear(8, 6, bias=True)
    codec_weights = {}
    codec_weights.update(
        {f"acoustic_decoder.{k}": v for k, v in seed_weights(dec, 313, 0.1).items()}
    )
    codec_weights.update(
        {f"acoustic_encoder.{k}": v for k, v in seed_weights(enc, 311, 0.1).items()}
    )
    codec_weights.update({f"quantizer.{k}": v for k, v in seed_weights(rvq, 315, 0.1).items()})
    flat_fc2 = dict(tree_flatten(fc2.parameters()))
    rng = np.random.default_rng(317)
    fc2_pairs = [
        (n, mx.array((rng.standard_normal(v.shape) * 0.1).astype(np.float32)))
        for n, v in flat_fc2.items()
    ]
    fc2.load_weights(fc2_pairs)
    mx.eval(fc2)
    codec_weights.update({f"fc2.{n}": v for n, v in fc2_pairs})
    codec = TinyCodec(dec, fc2, rvq)

    codec_dir = OUT / "codec"
    codec_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(codec_dir / "model.safetensors"), codec_weights)
    codec_config = {
        "sample_rate": 24000,
        "codebook_size": CODEBOOK_SIZE,
        "codebook_dim": 4,
        "downsample_factor": 320,
        "dac_num_codebooks": K,
        "dac_encoder_ratios": [8, 5, 4, 2, 3],
        "dac_encoder_hidden": 4,
        "dac_decoder_hidden": 32,
        "semantic_sample_rate": 16000,
        "quantizer_latent_dim": 8,
        "decoder_input_dim": 6,
    }
    (codec_dir / "config.json").write_text(json.dumps(codec_config, indent=2) + "\n")

    # --- backbone search --------------------------------------------------
    ref_raw = np.array(
        [[1, 3, 5, 7], [2, 4, 6, 8], [3, 5, 7, 9], [0, 2, 4, 6]], dtype=np.int32
    )
    tokenizer = FakeTokenizer()
    best = None
    for seed in range(1, 200):
        model, stored = build_model(seed)
        _builder, delayed, parts, embeds, prompt_tokens = build_prompt(
            model, tokenizer, ref_raw
        )
        eos_run = scenario_record(model, codec, embeds, EOC_BUDGET, "eoc_ramp")
        maxsteps_run = scenario_record(model, codec, embeds, MAXSTEPS_BUDGET, "max_steps")
        eos_ok = (
            eos_run["stop_reason"] == "eoc-ramp"
            and eos_run["num_frames_raw"] >= K + 3
            and eos_run["num_frames_aligned"] >= 2
            and all(len(row) == K for row in eos_run["frames"])
            and all(all(0 <= v < CODEBOOK_SIZE for v in row) for row in eos_run["codes"])
        )
        maxsteps_ok = maxsteps_run["stop_reason"] == "max-steps"
        if eos_ok and maxsteps_ok:
            best = (model, stored, delayed, parts, embeds, prompt_tokens, eos_run, maxsteps_run, seed)
            print(f"backbone seed {seed}: acceptable greedy fixtures")
            break
    if best is None:
        raise SystemExit("no backbone seed produced acceptable greedy fixtures")
    model, stored, delayed, parts, embeds, prompt_tokens, eos_run, maxsteps_run, seed = best
    if seed != BACKBONE_SEED:
        print(f"NOTE: search picked seed {seed}; BACKBONE_SEED is {BACKBONE_SEED}")

    # --- forward pin -------------------------------------------------------
    cache = make_prompt_cache(model)
    dummy = mx.zeros((1, embeds.shape[1]), dtype=mx.int32)
    hidden = model.backbone(dummy, cache=cache, input_embeddings=embeds)
    last_hidden = hidden[:, -1, :]
    logits = model._audio_logits(last_hidden)[0]

    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    text_ids = tokenizer.encode(TARGET_TEXT)
    ref_text_ids = tokenizer.encode(REF_TEXT)
    # Segment start: <|tts|> + optional <|ref_text|> + transcript + <|ref_audio|>.
    segment_start = 1 + 1 + len(ref_text_ids) + 1
    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "prompt": {
            "tts_id": SPECIAL["<|tts|>"],
            "ref_audio_id": SPECIAL["<|ref_audio|>"],
            "text_id": SPECIAL["<|text|>"],
            "audio_id": SPECIAL["<|audio|>"],
            "ref_text_id": SPECIAL["<|ref_text|>"],
            "target_text": TARGET_TEXT,
            "ref_text": REF_TEXT,
            "text_ids": text_ids,
            "ref_text_ids": ref_text_ids,
            "ref_raw": ints(mx.array(ref_raw)),
            "ref_delayed": [ints(row) for row in delayed],
            "segment_start": segment_start,
        },
        "prompt_token_ids": parts.token_ids,
        "prompt_tokens": int(prompt_tokens),
        "prompt_embeds": floats(embeds[0]),
        "probe_frame": [3, 5, 7, 9],
        "probe_frame2": [9, 7, 5, 3],
        "forward_last_hidden": floats(last_hidden[0]),
        "forward_audio_logits": floats(logits),
        "scenarios": [eos_run, maxsteps_run],
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))

    # --- delay case ---------------------------------------------------------
    delay_traces = {
        "boc_id": BOC,
        "eoc_id": EOC,
        "apply_input": [ints(row) for row in mx.array(ref_raw)],
        "apply_output": [ints(row) for row in delayed],
        "reverse_input": [ints(row) for row in delayed],
        "reverse_output": [ints(row) for row in reverse_delay_pattern(delayed)],
    }
    grid = (np.arange(K * 6).reshape(K, 6) % CODEBOOK_SIZE).astype(np.int32).T
    grid_delayed = apply_delay_pattern(mx.array(grid), boc_id=BOC, eoc_id=EOC)
    delay_traces["apply_input_2"] = [ints(row) for row in grid]
    delay_traces["apply_output_2"] = [ints(row) for row in grid_delayed]
    delay_traces["reverse_input_2"] = [ints(row) for row in grid_delayed]
    delay_traces["reverse_output_2"] = [ints(row) for row in reverse_delay_pattern(grid_delayed)]
    delay_dir = OUT / "delay"
    delay_dir.mkdir(parents=True, exist_ok=True)
    (delay_dir / "traces.json").write_text(json.dumps(delay_traces))

    # --- sampling case --------------------------------------------------------
    cases = []
    logits_rng = np.random.default_rng(531)
    for name, top_k, top_p in (
        ("greedy", None, None),
        ("top_k", 4, None),
        ("top_p", None, 0.5),
        ("top_k_ties", 3, None),
    ):
        block = logits_rng.standard_normal((K, CODEBOOK_SIZE)).astype(np.float32)
        if name == "top_k_ties":
            block[1] = np.array([2.0, 2.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.0, -0.5, -1.0], dtype=np.float32)
        greedy = ints(mx.argmax(mx.array(block), axis=-1))
        record = {
            "name": name,
            "codebooks": K,
            "top_k": top_k or 0,
            "top_p": top_p if top_p is not None else 0.0,
            "logits": floats(block),
            "greedy": greedy,
        }
        if top_k is not None:
            # The reference `_apply_top_k` link on raw logits (temperature
            # 1.0): log-softmax, filter, map the masked entries back.
            scaled = block.copy()
            for row in range(K):
                logprobs = nn.log_softmax(mx.array(scaled[row]), axis=-1)
                masked = _apply_top_k_logprobs(logprobs, int(top_k))
                scaled[row] = mx.where(masked == -mx.inf, -mx.inf, scaled[row])
            record["top_k_masked"] = masked_floats(scaled)
        if top_p is not None:
            scaled = block.copy()
            for row in range(K):
                logprobs = nn.log_softmax(mx.array(scaled[row]), axis=-1)
                masked = _apply_top_p_logprobs(logprobs, float(top_p))
                scaled[row] = mx.where(masked == -mx.inf, -mx.inf, scaled[row])
            record["top_p_masked"] = masked_floats(scaled)
        cases.append(record)
    # One temperature>0 draw record; the Rust path draws through the
    # crate threefry RNG and is not bit-reproducible, so only the greedy
    # sample and the filter masks are pinned as gates.
    draw_rng = np.random.default_rng(533)
    block = draw_rng.standard_normal((K, CODEBOOK_SIZE)).astype(np.float32)
    cases.append(
        {
            "name": "categorical_record_only",
            "codebooks": K,
            "top_k": 0,
            "top_p": 0.0,
            "temperature": 0.8,
            "logits": floats(block),
            "greedy": ints(mx.argmax(mx.array(block), axis=-1)),
        }
    )
    sampling_dir = OUT / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    (sampling_dir / "traces.json").write_text(json.dumps({"cases": cases}))

    # --- manifest ----------------------------------------------------------------
    files = {}
    for entry in sorted(OUT.rglob("*")):
        if entry.is_file() and entry.name != "manifest.json":
            files[str(entry.relative_to(OUT))] = hashlib.sha256(entry.read_bytes()).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_higgs_audio_v3_fixtures.py",
        "backbone_seed": seed,
        "amplify_eoc": AMPLIFY_EOC,
        "fade_in_ms": FADE_IN_MS,
        "fade_out_ms": FADE_OUT_MS,
        "sample_rate": SAMPLE_RATE,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
