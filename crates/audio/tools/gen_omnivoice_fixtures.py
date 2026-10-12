#!/usr/bin/env python3
"""Regenerate the OmniVoice family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/omnivoice,
mlx 0.32.3). The true reference iterative_unmask runs on a tiny seeded
bidirectional qwen3 backbone (hidden 32 / 2 layers / 4 heads / 2 kv
heads / rope 10000 / tied embeddings, text vocab 128) with 8 audio
codebooks of vocab 17 (mask id 16), and a tiny HiggsAudioV2 codec
stand-in (8 RVQ books, latent 8, decoder input 6) provides the
tokenizer.decode path the reference generate calls.

Determinism: class_temperature 0 pins the token draw to the greedy
argmax over the CFG-combined log probs; the position Gumbel draws are
recorded from the seeded mx.random stream and replayed by the Rust
(the crate-local threefry stream documented in the family README is
not bit-reproducible against mx.random). Text tokenization is
caller-owned: the style and wrapped-text strings are assembled by the
reference string rules and tokenized with a deterministic mock (CRC
buckets), and the ids are recorded for the Rust intake.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_omnivoice_fixtures.py

Outputs into crates/audio/testdata/tts_omnivoice/:

- generate/model.safetensors + generate/config.json
    Seeded tiny OmniVoice model in the converted naming
    (backbone.*, audio_embeddings.{i}, audio_heads.{i}).
- codec/model.safetensors + codec/config.json
    Seeded tiny codec in the HiggsAudioTokenizer layout (8 RVQ books,
    codebook 16, latent 8, decoder input 6), the same geometry the
    codec family fixture uses.
- traces.json
    Both cases (plain and voice-clone): style ids, text ids, target
    length, the assembled cond id grid and audio mask, the recorded
    position-Gumbel uniforms per step, the final token grid, and the
    codec waveform.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path.cwd()))

import mlx_audio.tts.models.omnivoice.generation as generation_mod
from mlx_audio.codec.models.higgs_audio.dac import (
    AcousticDecoderBlock,
    AcousticEncoderBlock,
    ResidualVectorQuantizer,
    Snake1d,
    VectorQuantizer,
)
from mlx_audio.codec.models.higgs_audio.higgs_audio import AcousticDecoder
from mlx_audio.codec.models.dacvae.codec import WNConv1d
from mlx_audio.tts.models.omnivoice.omnivoice import Model, OmniVoiceConfig
from mlx_audio.tts.models.omnivoice.backbone import BackboneConfig, OmniVoiceBackbone

OUT = Path("../turbospark/crates/audio/testdata/tts_omnivoice")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# Patched geometry.
LLM = {
    "model_type": "qwen3",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 128,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 512,
    "tie_word_embeddings": True,
}
AUDIO_VOCAB_SIZE = 17  # 16 codec tokens + the mask at 16
AUDIO_MASK_ID = 16
CODEBOOKS = 8
TARGET_LEN = 6
NUM_STEPS = 4
GUIDANCE_SCALE = 2.0
CLASS_TEMPERATURE = 0.0
POSITION_TEMPERATURE = 5.0
LAYER_PENALTY = 5.0
T_SHIFT = 0.1
REF_FRAMES = 4


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


class MockTokenizer:
    """Deterministic stand-in for the HF text tokenizer: whitespace
    words bucket by CRC into 4..127; special-tag strings bucket the
    whole string the same way."""

    def __call__(self, text: str, return_tensors: str | None = None,
                 add_special_tokens: bool = True):
        ids: list[int] = []
        for word in text.split(" "):
            if not word:
                continue
            ids.append(4 + zlib.crc32(word.encode("utf-8")) % 120)
        if return_tensors == "np":
            class Out:
                input_ids = [np.array(ids, dtype=np.int64)]

            return Out()
        return type("Out", (), {"input_ids": ids})


class TinyAcousticDecoder(AcousticDecoder):
    """AcousticDecoder.__init__ at decoder_hidden=32, decoder input 6
    (inherits the reference __call__)."""

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


class TinyAcousticEncoder(nn.Module):
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
        self.quantizers = [TinyVq() for _ in range(CODEBOOKS)]


class TinyCodec:
    """Stand-in for the network HiggsAudioTokenizer with the same decode
    stack the real decode runs (RVQ sum, fc2, acoustic decoder)."""

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


def build_model() -> tuple[Model, dict]:
    rng = np.random.default_rng(501)
    config = OmniVoiceConfig.from_dict({
        "model_type": "omnivoice",
        "audio_vocab_size": AUDIO_VOCAB_SIZE,
        "audio_mask_id": AUDIO_MASK_ID,
        "num_audio_codebook": CODEBOOKS,
        "sample_rate": 24000,
        "llm_config": dict(LLM),
    })
    model = Model(config)

    weights: dict = {}
    # Backbone: embed + 2 decoder layers + final norm.
    weights["backbone.embed_tokens.weight"] = mx.array(
        (rng.standard_normal((LLM["vocab_size"], LLM["hidden_size"])) * 0.3).astype(np.float32))
    for layer in range(LLM["num_hidden_layers"]):
        p = f"backbone.layers.{layer}."
        h, hd = LLM["hidden_size"], LLM["head_dim"]
        weights[p + "self_attn.q_proj.weight"] = mx.array(
            (rng.standard_normal((4 * hd, h)) * 0.3).astype(np.float32))
        weights[p + "self_attn.k_proj.weight"] = mx.array(
            (rng.standard_normal((2 * hd, h)) * 0.3).astype(np.float32))
        weights[p + "self_attn.v_proj.weight"] = mx.array(
            (rng.standard_normal((2 * hd, h)) * 0.3).astype(np.float32))
        weights[p + "self_attn.o_proj.weight"] = mx.array(
            (rng.standard_normal((h, 4 * hd)) * 0.3).astype(np.float32))
        weights[p + "self_attn.q_norm.weight"] = mx.array(
            (rng.uniform(0.5, 1.5, hd)).astype(np.float32))
        weights[p + "self_attn.k_norm.weight"] = mx.array(
            (rng.uniform(0.5, 1.5, hd)).astype(np.float32))
        weights[p + "mlp.gate_proj.weight"] = mx.array(
            (rng.standard_normal((LLM["intermediate_size"], h)) * 0.3).astype(np.float32))
        weights[p + "mlp.up_proj.weight"] = mx.array(
            (rng.standard_normal((LLM["intermediate_size"], h)) * 0.3).astype(np.float32))
        weights[p + "mlp.down_proj.weight"] = mx.array(
            (rng.standard_normal((h, LLM["intermediate_size"])) * 0.3).astype(np.float32))
        weights[p + "input_layernorm.weight"] = mx.array(
            (rng.uniform(0.5, 1.5, h)).astype(np.float32))
        weights[p + "post_attention_layernorm.weight"] = mx.array(
            (rng.uniform(0.5, 1.5, h)).astype(np.float32))
    weights["backbone.norm.weight"] = mx.array(
        (rng.uniform(0.5, 1.5, LLM["hidden_size"])).astype(np.float32))
    # Audio tables and heads.
    for i in range(CODEBOOKS):
        weights[f"audio_embeddings.{i}.weight"] = mx.array(
            (rng.standard_normal((AUDIO_VOCAB_SIZE, LLM["hidden_size"])) * 0.3).astype(np.float32))
        weights[f"audio_heads.{i}.weight"] = mx.array(
            (rng.standard_normal((AUDIO_VOCAB_SIZE, LLM["hidden_size"])) * 0.3).astype(np.float32))

    pairs = [(name, weights[name]) for name in weights]
    model.load_weights(pairs)
    mx.eval(model.parameters())
    return model, weights


def tokenize_style(text: str) -> list[int]:
    return MockTokenizer()(text).input_ids


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gen_dir = OUT / "generate"
    codec_dir = OUT / "codec"
    gen_dir.mkdir(exist_ok=True)
    codec_dir.mkdir(exist_ok=True)

    # --- 1. The tiny model. -------------------------------------------
    model, stored = build_model()
    mx.save_safetensors(str(gen_dir / "model.safetensors"), stored)
    config_json = {
        "model_type": "omnivoice",
        "audio_vocab_size": AUDIO_VOCAB_SIZE,
        "audio_mask_id": AUDIO_MASK_ID,
        "num_audio_codebook": CODEBOOKS,
        "sample_rate": 24000,
        "llm_config": dict(LLM),
        "dtype": "float32",
    }
    (gen_dir / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")

    # --- 2. The tiny codec. -------------------------------------------
    dec = TinyAcousticDecoder()
    enc = TinyAcousticEncoder()
    rvq = TinyRvq()
    fc2 = nn.Linear(8, 6, bias=True)
    codec_weights = {}
    codec_weights.update(
        {f"acoustic_decoder.{k}": v for k, v in seed_weights(dec, 503, 0.1).items()})
    # The encoder tree is present for checkpoint parity only; the
    # encode path stays out of contract for the fixture.
    codec_weights.update(
        {f"acoustic_encoder.{k}": v for k, v in seed_weights(enc, 504, 0.1).items()})
    codec_weights.update({f"quantizer.{k}": v for k, v in seed_weights(rvq, 505, 0.1).items()})
    flat_fc2 = dict(tree_flatten(fc2.parameters()))
    rng = np.random.default_rng(507)
    fc2_pairs = [
        (n, mx.array((rng.standard_normal(v.shape) * 0.1).astype(np.float32)))
        for n, v in flat_fc2.items()
    ]
    fc2.load_weights(fc2_pairs)
    mx.eval(fc2)
    codec_weights.update({f"fc2.{n}": v for n, v in fc2_pairs})
    codec = TinyCodec(dec, fc2, rvq)
    mx.save_safetensors(str(codec_dir / "model.safetensors"), codec_weights)
    codec_config = {
        "sample_rate": 24000,
        "codebook_size": 16,
        "codebook_dim": 4,
        "downsample_factor": 320,
        "dac_num_codebooks": CODEBOOKS,
        "dac_encoder_ratios": [8, 5, 4, 2, 3],
        "dac_encoder_hidden": 4,
        "dac_decoder_hidden": 32,
        "semantic_sample_rate": 16000,
        "quantizer_latent_dim": 8,
        "decoder_input_dim": 6,
    }
    (codec_dir / "config.json").write_text(json.dumps(codec_config, indent=2) + "\n")

    # --- 3. Run the two generation cases. ------------------------------
    # Record the position-Gumbel uniforms: the wrapper keeps the exact
    # reference math while capturing the seeded mx.random stream.
    draws: list = []
    original_gumbel = generation_mod._gumbel_noise

    def recording_gumbel(x, temperature):
        u = mx.random.uniform(shape=x.shape)
        draws.append(floats(u))
        gumbel = -mx.log(-mx.log(u + 1e-10) + 1e-10)
        return x / temperature + gumbel

    generation_mod._gumbel_noise = recording_gumbel

    ref_tokens = mx.array(
        np.asarray(
            [[(7 * r + 3 * c) % 16 for c in range(CODEBOOKS)] for r in range(REF_FRAMES)],
            dtype=np.int32,
        ))
    cases = [
        {
            "name": "plain",
            "text": "Hello there friend",
            "language": "en",
            "ref_tokens": None,
        },
        {
            "name": "voice_clone",
            "text": " Hi  there\tCJK test 中文 test.",
            "language": "zh",
            "ref_tokens": ref_tokens,
        },
    ]

    trace_cases = []
    for case in cases:
        has_ref = case["ref_tokens"] is not None
        # The reference string assembly (verified by the Rust unit tests
        # on the same rules); the mock tokenizer supplies the ids.
        from mlx_audio.tts.models.omnivoice.omnivoice import (
            _combine_text,
            _tokenize_with_nonverbal_tags,
        )

        style_text = ("<|denoise|>" if has_ref else "") + (
            f"<|lang_start|>{case['language']}<|lang_end|>"
            "<|instruct_start|>None<|instruct_end|>"
        )
        style_ids = tokenize_style(style_text)
        wrapped = f"<|text_start|>{_combine_text(case['text'])}<|text_end|>"
        text_ids = ints(_tokenize_with_nonverbal_tags(wrapped, MockTokenizer()))

        inputs = model._prepare_inference_inputs(
            mx.array(style_ids, dtype=mx.int32),
            mx.array(text_ids, dtype=mx.int32),
            TARGET_LEN,
            case["ref_tokens"],
        )
        cond_input_ids = inputs["input_ids"]
        cond_audio_mask = inputs["audio_mask"]
        draws.clear()
        mx.random.seed(0)
        tokens = generation_mod.iterative_unmask(
            model,
            cond_input_ids,
            cond_audio_mask,
            T=TARGET_LEN,
            num_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            class_temperature=CLASS_TEMPERATURE,
            position_temperature=POSITION_TEMPERATURE,
            layer_penalty_factor=LAYER_PENALTY,
            t_shift=T_SHIFT,
        )
        token_grid = np.asarray(tokens, dtype=np.int32)
        wave = np.asarray(codec.decode(tokens), dtype=np.float32)
        assert np.abs(wave).max() < 1.0, "codec waveform must stay unclipped"
        trace_cases.append({
            "name": case["name"],
            "text": case["text"],
            "language": case["language"],
            "denoise": has_ref,
            "style_ids": style_ids,
            "text_ids": text_ids,
            "target_len": TARGET_LEN,
            "cond_input_ids": ints(cond_input_ids),
            "cond_audio_mask": [int(v) for v in np.asarray(cond_audio_mask).ravel()],
            "gumbel_draws": draws,
            "tokens": ints(token_grid),
            "waveform": floats(wave),
            "sample_rate": 24000,
        })
    generation_mod._gumbel_noise = original_gumbel

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "num_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "class_temperature": CLASS_TEMPERATURE,
        "position_temperature": POSITION_TEMPERATURE,
        "layer_penalty_factor": LAYER_PENALTY,
        "t_shift": T_SHIFT,
        "cases": trace_cases,
    }
    (OUT / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")

    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "generator": "tools/gen_omnivoice_fixtures.py",
        "files": {
            str(path.relative_to(OUT)): "sha256:"
            + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"omnivoice: {len(trace_cases)} cases, target {TARGET_LEN} frames; "
          f"wrote fixtures under {OUT}")


if __name__ == "__main__":
    main()
