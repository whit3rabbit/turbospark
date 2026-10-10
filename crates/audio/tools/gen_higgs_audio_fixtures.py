#!/usr/bin/env python3
"""Regenerate the Higgs Audio v2 TTS family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/
higgs_audio, mlx 0.32.3). The reference modules are imported directly;
the TTS model builds the true HiggsAudioModel at tiny seeded dims, the
prompt builders run the true serve.py steps, and the tiny codec stands
in for the network HiggsAudioTokenizer with the same decode stack (RVQ
sum, fc2, acoustic decoder) the codec family fixture pins.

This is the TTS family fixture generator; the codec fixture lives in
gen_higgs_fixtures.py.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_higgs_audio_fixtures.py

Outputs into crates/audio/testdata/tts_higgs_audio/:

- generate/model.safetensors + generate/config.json
    Seeded tiny HiggsAudioModel (2 dual-FFN layers, hidden 32, text
    vocab 96, 3 codebooks of 16 + stream BOS/EOS, llama3 rope scaling).
    Weights are stored F16 (the traces are recorded after upcasting the
    same stored values, so the Rust f32 path sees identical inputs).
- generate/traces.json
    The pinned voice-clone prompt (ids, reference codes, embeddings,
    routing mask), a forward pin over those positions (text + audio
    logits with a mixed audio mask), and two greedy scenarios: an EOS
    ramp-out stop and a max-frames stop, each with the raw delayed
    frames, the aligned codes, the stop reason, the tiny-codec
    waveform, and the faded waveform.
- codec/model.safetensors + codec/config.json
    Seeded tiny codec in the HiggsAudioTokenizer layout (3 RVQ books,
    latent 8, decoder input 6), the same geometry the codec family
    fixture uses.
- delay/traces.json
    revert_delay_pattern, apply_delay_pattern, and
    build_delay_pattern_mask on fixed id grids plus
    lookup_audio_embedding on a fixed id block.
- sampling/traces.json
    greedy_sample_audio on fixed logit blocks and the top-k (kth-value,
    ties kept) and top-p (descending-cumsum) filter intermediates. The
    temperature > 0 Gumbel draw is recorded for reference only: the
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
import re
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
from mlx_audio.tts.models.higgs_audio.config import HiggsAudioConfig
from mlx_audio.tts.models.higgs_audio.generation import (
    apply_delay_pattern,
    build_delay_pattern_mask,
    greedy_sample_audio,
    lookup_audio_embedding,
    revert_delay_pattern,
)
from mlx_audio.tts.models.higgs_audio.higgs_audio import HiggsAudioModel

OUT = Path("../turbospark/crates/audio/testdata/tts_higgs_audio")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

K = 3
CODEBOOK_SIZE = 16
BOS = 16
EOS = 17
STRIDE = CODEBOOK_SIZE + 2

SAMPLE_RATE = 24000
FADE_IN_MS = 30.0
FADE_OUT_MS = 15.0

# The stream-EOS column amplification and the pinned backbone seed:
# first seed of the search below whose greedy runs satisfy both
# scenario shapes.
AMPLIFY_EOS = 1.6
BACKBONE_SEED = 1
EOS_BUDGET = 24
MAXFRAMES_BUDGET = 6

TEXT_CONFIG = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=64,
    vocab_size=96,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    rope_scaling={
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
)

CONFIG = {
    "text_config": TEXT_CONFIG,
    "audio_num_codebooks": K,
    "audio_codebook_size": CODEBOOK_SIZE,
    "audio_stream_bos_id": BOS,
    "audio_stream_eos_id": EOS,
    "audio_dual_ffn_layers": [0, 1],
    "use_audio_out_self_attention": False,
    "audio_decoder_proj_num_layers": 0,
    "use_delay_pattern": True,
    "sample_rate": SAMPLE_RATE,
}

SPECIAL = {
    "<|begin_of_text|>": 1,
    "<|start_header_id|>": 2,
    "<|end_header_id|>": 3,
    "<|eot_id|>": 4,
    "<|audio_out_bos|>": 5,
    "<|audio_eos|>": 6,
}


def encode(text: str) -> list[int]:
    """The mock tokenizer: special tokens map to fixed ids, every other
    chunk to a deterministic rolling-hash id. The Rust side consumes the
    recorded id sequences."""
    ids = []
    for raw in re.split(r"(<\|[^>]+\|>)", text):
        if not raw:
            continue
        if raw in SPECIAL:
            ids.append(SPECIAL[raw])
            continue
        ids.append(10 + (sum(ord(c) * (i + 1) for i, c in enumerate(raw)) % 80))
    return ids


PREFIX_TEXT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nRef voice.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>"
MIDDLE_TEXT = "<|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello there.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>"


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
        super().__init__(latent_dim=8, codebook_size=16, codebook_dim=4)


class TinyRvq(ResidualVectorQuantizer):
    def __init__(self):
        self.quantizers = [TinyVq() for _ in range(3)]


class TinyCodec:
    """Stand-in for the HiggsAudioTokenizer: decode through the real tiny
    stacks in consumption order (RVQ sum, fc2, acoustic decoder)."""

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
    config = HiggsAudioConfig.from_dict(CONFIG)
    mx.random.seed(seed)
    model = HiggsAudioModel(config)
    seed_weights(model, seed)
    # Amplify the per-codebook stream-EOS rows of the audio head (the
    # MLX Linear stores [out, in], so output class j is row j) so the
    # greedy EOS scenario lands inside a short budget.
    head = np.array(model.audio_decoder_proj.audio_lm_head.weight, dtype=np.float32)
    for codebook in range(K):
        head[codebook * STRIDE + EOS, :] *= AMPLIFY_EOS
    model.audio_decoder_proj.audio_lm_head.weight = mx.array(head)
    mx.eval(model)
    stored = {
        name: value.astype(mx.float16) for name, value in tree_flatten(model.parameters())
    }
    stored["audio_decoder_proj.audio_lm_head.weight"] = mx.array(head).astype(mx.float16)
    # Reload the upcast F16 stored values so every trace records exactly
    # what the Rust f32 path reads back from the file (llama fixture
    # convention).
    model.update(
        tree_unflatten({k: v.astype(mx.float32) for k, v in stored.items()})
    )
    mx.eval(model)
    return model, stored


def build_voice_clone_prompt(model, ref_codes):
    """The serve.py voice-clone prompt through the true functions."""
    prefix_ids = encode(PREFIX_TEXT)
    middle_ids = encode(MIDDLE_TEXT)
    bos_col = mx.full((K, 1), BOS, dtype=mx.int32)
    eos_col = mx.full((K, 1), EOS, dtype=mx.int32)
    wrapped = mx.concatenate([bos_col, mx.array(ref_codes, dtype=mx.int32), eos_col], axis=1)
    delayed = build_delay_pattern_mask(wrapped, BOS, EOS)
    prefix_emb = model.embed_tokens(mx.array([prefix_ids]))[0]
    audio_emb = lookup_audio_embedding(model.audio_codebook_embeddings, delayed, STRIDE)
    middle_emb = model.embed_tokens(mx.array([middle_ids]))[0]
    full_embeds = mx.concatenate([prefix_emb, audio_emb, middle_emb], axis=0)[None]
    audio_out_mask = mx.concatenate(
        [
            mx.zeros((len(prefix_ids),), dtype=mx.bool_),
            mx.ones((delayed.shape[1],), dtype=mx.bool_),
            mx.zeros((len(middle_ids),), dtype=mx.bool_),
        ],
        axis=0,
    )[None]
    return {
        "prefix_ids": prefix_ids,
        "middle_ids": middle_ids,
        "prefix_emb": prefix_emb,
        "audio_emb": audio_emb,
        "t_ref": int(ref_codes.shape[1]),
        "t_ref_delayed": int(delayed.shape[1]),
        "full_embeds": full_embeds,
        "audio_out_mask": audio_out_mask,
    }


def raw_frames(model, prompt, max_new_frames):
    """The raw delayed frames through the true frame loop."""
    return [
        tok
        for tok, _meta in model._generate_raw_frames(
            prompt["full_embeds"],
            prompt["audio_out_mask"],
            max_new_frames=max_new_frames,
            temperature=0.0,
            top_p=0.95,
            top_k=None,
            ras_win_len=7,
            ras_max_repeat=2,
            sampling_warmup_frames=0,
        )
    ]


def fades(pcm: np.ndarray) -> np.ndarray:
    out = pcm.copy()
    n_in = int(FADE_IN_MS * SAMPLE_RATE / 1000.0)
    n_out = int(FADE_OUT_MS * SAMPLE_RATE / 1000.0)
    if n_in > 0 and out.size > n_in:
        out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 0 and out.size > n_out:
        out[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return out


def scenario_record(model, codec, prompt, max_new_frames, name):
    frames = raw_frames(model, prompt, max_new_frames)
    aligned, info = model.generate(
        inputs_embeds=prompt["full_embeds"],
        audio_out_mask=prompt["audio_out_mask"],
        max_new_frames=max_new_frames,
        temperature=0.0,
        top_p=0.95,
        top_k=None,
        ras_win_len=7,
        ras_max_repeat=2,
        sampling_warmup_frames=0,
    )
    pcm = np.asarray(codec.decode(aligned.T), dtype=np.float32).reshape(-1)
    return {
        "name": name,
        "max_new_frames": max_new_frames,
        "num_frames_raw": int(info["num_frames_raw"]),
        "num_frames_aligned": int(info["num_frames_aligned"]),
        "stop_reason": info["stop_reason"],
        "frames": [ints(frame) for frame in frames],
        "aligned": [ints(row) for row in aligned],
        "waveform": floats(pcm),
        "waveform_faded": floats(fades(pcm)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- tiny codec (independent of the backbone search) ----------------
    dec = TinyAcousticDecoder()
    enc = TinyAcousticEncoder()
    rvq = TinyRvq()
    fc2 = nn.Linear(8, 6, bias=True)
    codec_weights = {}
    codec_weights.update(
        {f"acoustic_decoder.{k}": v for k, v in seed_weights(dec, 303, 0.1).items()}
    )
    codec_weights.update(
        {f"acoustic_encoder.{k}": v for k, v in seed_weights(enc, 301, 0.1).items()}
    )
    codec_weights.update({f"quantizer.{k}": v for k, v in seed_weights(rvq, 305, 0.1).items()})
    flat_fc2 = dict(tree_flatten(fc2.parameters()))
    rng = np.random.default_rng(307)
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
        "codebook_size": 16,
        "codebook_dim": 4,
        "downsample_factor": 320,
        "dac_num_codebooks": 3,
        "dac_encoder_ratios": [8, 5, 4, 2, 3],
        "dac_encoder_hidden": 4,
        "dac_decoder_hidden": 32,
        "semantic_sample_rate": 16000,
        "quantizer_latent_dim": 8,
        "decoder_input_dim": 6,
    }
    (codec_dir / "config.json").write_text(json.dumps(codec_config, indent=2) + "\n")

    # --- backbone search --------------------------------------------------
    ref_codes = np.array([[3, 5, 7, 9], [1, 2, 3, 4], [15, 13, 11, 9]], dtype=np.int32)
    best = None
    for seed in range(1, 200):
        model, stored = build_model(seed)
        prompt = build_voice_clone_prompt(model, ref_codes)
        eos_run = scenario_record(model, codec, prompt, EOS_BUDGET, "eos_ramp")
        maxframes_run = scenario_record(model, codec, prompt, MAXFRAMES_BUDGET, "max_frames")
        eos_ok = (
            eos_run["stop_reason"].startswith("eos-ramp-complete")
            and eos_run["num_frames_raw"] >= K + 3
            and eos_run["num_frames_aligned"] >= 2
            and all(len(row) == K for row in eos_run["frames"])
        )
        maxframes_ok = maxframes_run["stop_reason"].startswith("max-frames")
        if eos_ok and maxframes_ok:
            best = (model, stored, prompt, eos_run, maxframes_run, seed)
            print(f"backbone seed {seed}: acceptable greedy fixtures")
            break
    if best is None:
        raise SystemExit("no backbone seed produced acceptable greedy fixtures")
    model, stored, prompt, eos_run, maxframes_run, seed = best
    if seed != BACKBONE_SEED:
        print(f"NOTE: search picked seed {seed}; BACKBONE_SEED is {BACKBONE_SEED}")

    # --- forward pin -------------------------------------------------------
    text_logits, audio_logits = model(
        inputs_embeds=prompt["full_embeds"], audio_out_mask=prompt["audio_out_mask"]
    )

    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    (case_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "prompt": {
            "prefix_text": PREFIX_TEXT,
            "middle_text": MIDDLE_TEXT,
            "prefix_ids": prompt["prefix_ids"],
            "middle_ids": prompt["middle_ids"],
            "ref_codes": [ints(row) for row in ref_codes],
            "t_ref": prompt["t_ref"],
            "t_ref_delayed": prompt["t_ref_delayed"],
            "prefix_emb": floats(prompt["prefix_emb"]),
            "audio_emb": floats(prompt["audio_emb"]),
        },
        "prompt_embeds": floats(prompt["full_embeds"][0]),
        "prompt_mask": [bool(v) for v in prompt["audio_out_mask"][0]],
        "forward_text_logits": floats(text_logits[0]),
        "forward_audio_logits": floats(audio_logits[0]),
        "scenarios": [eos_run, maxframes_run],
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))

    # --- delay case ---------------------------------------------------------
    delay_traces = {}
    grid = (np.arange(K * 7).reshape(K, 7) % 20).astype(np.int32)
    reverted = revert_delay_pattern(mx.array(grid))
    delay_traces["revert_input"] = [ints(row) for row in grid]
    delay_traces["revert_output"] = [ints(row) for row in reverted]
    reapplied = apply_delay_pattern(reverted, BOS)
    delay_traces["apply_input"] = [ints(row) for row in reverted]
    delay_traces["apply_output"] = [ints(row) for row in reapplied]
    aligned_in = (np.arange(K * 4).reshape(K, 4) % 12).astype(np.int32)
    mask_out = build_delay_pattern_mask(mx.array(aligned_in), BOS, EOS)
    delay_traces["mask_input"] = [ints(row) for row in aligned_in]
    delay_traces["mask_output"] = [ints(row) for row in mask_out]
    lookup_ids = (np.arange(K * 5).reshape(K, 5) % 18).astype(np.int32)
    lookup_out = lookup_audio_embedding(
        model.audio_codebook_embeddings, mx.array(lookup_ids), STRIDE
    )
    delay_traces["lookup_ids"] = [ints(row) for row in lookup_ids]
    delay_traces["lookup_rows"] = floats(lookup_out)
    delay_dir = OUT / "delay"
    delay_dir.mkdir(parents=True, exist_ok=True)
    (delay_dir / "traces.json").write_text(json.dumps(delay_traces))

    # --- sampling case --------------------------------------------------------
    cases = []
    logits_rng = np.random.default_rng(521)
    for name, top_k, top_p in (
        ("greedy", None, None),
        ("top_k", 6, None),
        ("top_p", None, 0.5),
        ("top_k_ties", 5, None),
    ):
        block = logits_rng.standard_normal((K, CODEBOOK_SIZE + 2)).astype(np.float32)
        if name == "top_k_ties":
            block[1] = np.array(
                [
                    2.0, 2.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.0, -0.5, -1.0, -1.5,
                    -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0,
                ],
                dtype=np.float32,
            )
        record = {
            "name": name,
            "top_k": top_k or 0,
            "top_p": top_p if top_p is not None else 0.0,
            "logits": floats(block),
            "greedy": ints(greedy_sample_audio(mx.array(block)[None, None])),
        }
        if top_k is not None:
            # The reference sample_audio top-k: mask below the kth value
            # (ties at the threshold survive).
            scaled = block.copy()
            for row in range(K):
                kth = np.sort(scaled[row])[-top_k]
                scaled[row][scaled[row] < kth] = -np.inf
            record["top_k_masked"] = masked_floats(scaled)
        if top_p is not None:
            # The reference sample_audio top-p: descending softmax
            # cumsum, keep where the PREVIOUS mass is below top_p.
            masked = np.empty_like(block)
            for row in range(K):
                order = np.argsort(-block[row], kind="stable")
                sorted_logits = block[row][order]
                probs = mx.softmax(mx.array(sorted_logits))
                cumulative = np.cumsum(np.asarray(probs))
                shifted = np.concatenate([[0.0], cumulative[:-1]])
                keep = shifted < top_p
                out = np.full_like(block[row], -np.inf)
                for position, index in enumerate(order):
                    if keep[position]:
                        out[index] = block[row][index]
                masked[row] = out
            record["top_p_masked"] = masked_floats(masked)
        cases.append(record)
    # One temperature>0 Gumbel draw record; the Rust path draws through
    # the crate threefry RNG and is not bit-reproducible, so only the
    # greedy sample and the filter masks are pinned as gates.
    draw_rng = np.random.default_rng(523)
    block = draw_rng.standard_normal((K, CODEBOOK_SIZE + 2)).astype(np.float32)
    cases.append(
        {
            "name": "gumbel_record_only",
            "top_k": 0,
            "top_p": 0.0,
            "temperature": 0.8,
            "logits": floats(block),
            "greedy": ints(greedy_sample_audio(mx.array(block)[None, None])),
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
        "generator": "crates/audio/tools/gen_higgs_audio_fixtures.py",
        "backbone_seed": seed,
        "amplify_eos": AMPLIFY_EOS,
        "fade_in_ms": FADE_IN_MS,
        "fade_out_ms": FADE_OUT_MS,
        "sample_rate": SAMPLE_RATE,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
