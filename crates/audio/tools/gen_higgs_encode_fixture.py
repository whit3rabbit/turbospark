#!/usr/bin/env python3
"""Regenerate the Higgs Audio encode-path Rust parity fixture.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/codec/models/higgs_audio, mlx 0.32.3).

The released OmniVoice ``audio_tokenizer/`` folder ships the
Transformers format: PyTorch conv layouts, ``codebook.embed``, snake
``[1, C, 1]``, the Wav2Vec2 semantic tower under ``semantic_model.``,
and the config with the DAC geometry nested under
``acoustic_model_config``. This fixture mirrors that layout at shrunk
dims (the same trick gen_higgs_fixtures.py uses for the decode path)
and records the reference encode pipeline for two waveforms: the sinc
resample, the tower hidden-state mean, the semantic encoder, the
acoustic features, the fused embeddings, and the RVQ codes.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_higgs_encode_fixture.py

Outputs into crates/audio/testdata/higgs_audio_encode/ and is
deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.higgs_audio.config import HiggsAudioConfig
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
    HiggsAudioTokenizer,
    _sinc_resample,
)
from mlx_audio.codec.models.dacvae.codec import WNConv1d
from mlx_audio.stt.models.wav2vec.wav2vec import Wav2Vec2Model

OUT = Path("../turbospark/crates/audio/testdata/higgs_audio_encode")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# The inline semantic_model_config dict, at shrunk dims. conv strides
# multiply to 4, so downsample_factor is 4 and the tower runs at 4 kHz
# equivalent frame rate; the acoustic hop stays 960 (25 fps), so the
# fusion downsample factor is 160 - an extreme but exact ratio that
# exercises the stride slicing hard.
SEMANTIC_CONFIG = {
    "model_type": "hubert",
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 8,
    "layer_norm_eps": 1e-05,
    "conv_dim": [4, 4],
    "conv_stride": [2, 2],
    "conv_kernel": [3, 3],
    "num_feat_extract_layers": 2,
    "conv_bias": False,
    "feat_extract_norm": "group",
    "do_stable_layer_norm": False,
    "hidden_act": "gelu",
    "num_conv_pos_embeddings": 5,
    "num_conv_pos_embedding_groups": 2,
    "vocab_size": 32,
}

# The released acoustic config declares nine books for the full
# Transformers model while the RVQ the LM consumes carries eight; the
# fixture uses eight to match the released quantizer.quantizers count.
NUM_BOOKS = 8


class TinyAcousticEncoder(AcousticEncoder):
    """AcousticEncoder.__init__ at encoder_hidden=4, strides [8,5,4,2,3]."""

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
        # The reference hardcodes the acoustic feature width at 256
        # (the fusion constant); the tiny ladder keeps that width.
        self.conv2 = WNConv1d(c[-1], 256, kernel_size=3, padding=1, norm="none")


class TinyAcousticDecoder(AcousticDecoder):
    """AcousticDecoder.__init__ at decoder_hidden=32, decoder input 6."""

    _STRIDES = [8, 5, 4, 2, 3]
    _IN_CHANNELS = [32, 16, 8, 4, 2]
    _OUT_CHANNELS = [16, 8, 4, 2, 1]

    def __init__(self):
        self.conv1 = WNConv1d(256, self._IN_CHANNELS[0], kernel_size=7, padding=3, norm="none")
        self.block = [
            AcousticDecoderBlock(self._IN_CHANNELS[i], self._OUT_CHANNELS[i], self._STRIDES[i])
            for i in range(len(self._STRIDES))
        ]
        self.snake1 = Snake1d(self._OUT_CHANNELS[-1])
        self.conv2 = WNConv1d(self._OUT_CHANNELS[-1], 1, kernel_size=7, padding=3, norm="none")


class TinyVq(VectorQuantizer):
    def __init__(self):
        # The latent is the fusion width: acoustic 256 + tower hidden 16.
        super().__init__(latent_dim=272, codebook_size=16, codebook_dim=4)


class TinyRvq(ResidualVectorQuantizer):
    def __init__(self):
        self.quantizers = [TinyVq() for _ in range(NUM_BOOKS)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def seed(model, seed_value: int) -> dict:
    flat = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(seed_value)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in flat.items()
    }
    model.load_weights(list(seeded.items()))
    mx.eval(model)
    return seeded


def to_pytorch_naming(flat: dict) -> dict:
    """Rewrite MLX-layout weights into the released PyTorch layout.

    Conv weights [out, K, in] -> [out, in, K]; ConvTranspose weights
    [out, K, in] -> [in, out, K]; snake alpha [1, 1, C] -> [1, C, 1];
    the positional-conv weight_v [out, K, in/g] -> [out, in/g, K];
    the codebook `.weight` renames to `.embed`. The feature-extractor
    convs are plain ``nn.Conv1d`` upstream, so no weight norm is folded;
    the positional conv keeps its ``weight_g``/``weight_v`` pair.
    """
    out = {}
    for name, value in flat.items():
        w = np.asarray(value, dtype=np.float32)
        if name.endswith(".weight") and w.ndim == 3 and "conv_t1" in name:
            w = w.transpose(2, 0, 1)
        elif name.endswith(".weight") and w.ndim == 3:
            w = w.transpose(0, 2, 1)
        elif (
            name.endswith(".weight_v") or name.endswith("parametrizations.weight.original1")
        ) and w.ndim == 3:
            w = w.transpose(0, 2, 1)
        elif name.endswith(".alpha") and w.ndim == 3:
            w = w.transpose(0, 2, 1)
        elif ".codebook.weight" in name:
            name = name.replace(".codebook.weight", ".codebook.embed")
        out[name] = np.ascontiguousarray(w)
    return out


def rvq_margins(rvq: TinyRvq, z: np.ndarray) -> float:
    """Smallest best-vs-second margin across books for these rows."""
    residual = z.copy()
    min_margin = np.inf
    for vq in rvq.quantizers:
        zq = np.asarray(vq.project_in(mx.array(residual)), dtype=np.float32)
        w = np.asarray(vq.codebook.weight, dtype=np.float32)
        d = (zq**2).sum(-1, keepdims=True) - 2.0 * zq @ w.T + (w**2).sum(-1)[None, :]
        order = np.argsort(d, axis=-1, kind="stable")
        best = np.take_along_axis(d, order[:, :1], axis=-1)
        second = np.take_along_axis(d, order[:, 1:2], axis=-1)
        min_margin = min(min_margin, float((second - best).min()))
        idx = np.argmin(d, axis=-1)
        recon = np.asarray(vq.decode_codes(mx.array(idx)), dtype=np.float32)
        residual = residual - recon
    return min_margin


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files: list[str] = []

    # The reference tokenizer with its acoustic modules swapped for the
    # tiny ones; _init_encode_modules builds the real Wav2Vec2 tower,
    # SemanticEncoder, and fusion fc from the inline dict.
    config = HiggsAudioConfig(
        sample_rate=24000,
        codebook_size=16,
        codebook_dim=4,
        downsample_factor=4,
        dac_num_codebooks=NUM_BOOKS,
        dac_encoder_ratios=[8, 5, 4, 2, 3],
        dac_encoder_hidden=4,
        dac_decoder_hidden=32,
        semantic_sample_rate=16000,
        strides=[1, 1],
        block_dilations=[1, 1],
        channel_ratios=[1, 1],
        kernel_size=3,
        unit_kernel_size=3,
        semantic_model_config=dict(SEMANTIC_CONFIG),
    )
    tokenizer = HiggsAudioTokenizer(config)
    tokenizer.acoustic_encoder = TinyAcousticEncoder()
    tokenizer.acoustic_decoder = TinyAcousticDecoder()
    tokenizer.quantizer = TinyRvq()
    tokenizer.fc2 = nn.Linear(272, 256, bias=True)
    tokenizer._init_encode_modules()

    seeded = {}
    seeded.update({f"acoustic_encoder.{k}": v for k, v in seed(tokenizer.acoustic_encoder, 401).items()})
    seeded.update({f"acoustic_decoder.{k}": v for k, v in seed(tokenizer.acoustic_decoder, 402).items()})
    seeded.update({f"quantizer.{k}": v for k, v in seed(tokenizer.quantizer, 403).items()})
    seeded.update({f"semantic_model.{k}": v for k, v in seed(tokenizer.semantic_model, 405).items()})
    seeded.update({f"encoder_semantic.{k}": v for k, v in seed(tokenizer.encoder_semantic, 407).items()})
    rng = np.random.default_rng(409)
    for prefix, module in (("fc2", tokenizer.fc2), ("fc", tokenizer.fc)):
        for name, v in dict(tree_flatten(module.parameters())).items():
            seeded[f"{prefix}.{name}"] = mx.array(
                (rng.standard_normal(v.shape) * 0.3).astype(np.float32)
            )
        module.load_weights(
            [(n[len(prefix) + 1:], v) for n, v in seeded.items() if n.startswith(f"{prefix}.")]
        )
    mx.eval(tokenizer.parameters())

    pytorch_weights = to_pytorch_naming(seeded)
    mx.save_safetensors(str(OUT / "model.safetensors"), {k: mx.array(v) for k, v in pytorch_weights.items()})

    config_out = {
        "model_type": "higgs_audio_v2_tokenizer",
        "acoustic_model_config": {
            "model_type": "dac",
            "codebook_size": 16,
            "codebook_dim": 4,
            "downsampling_ratios": [8, 5, 4, 2, 3],
            "encoder_hidden_size": 4,
            "decoder_hidden_size": 32,
            "hidden_size": 256,
            "hop_length": 960,
            "sampling_rate": 16000,
        },
        "quantizer_latent_dim": 272,
        "codebook_size": 16,
        "codebook_dim": 4,
        "downsample_factor": 4,
        "sample_rate": 24000,
        "semantic_sample_rate": 16000,
        "strides": [1, 1],
        "block_dilations": [1, 1],
        "channel_ratios": [1, 1],
        "kernel_size": 3,
        "unit_kernel_size": 3,
        "semantic_model_config": dict(SEMANTIC_CONFIG),
    }
    (OUT / "config.json").write_text(json.dumps(config_out, indent=2))

    # The recorded pipeline mirrors HiggsAudioTokenizer.encode step by
    # step so every stage lands in a file the Rust test can gate on.
    semantic_model = tokenizer.semantic_model
    encoder_semantic = tokenizer.encoder_semantic
    acoustic_encoder = tokenizer.acoustic_encoder
    fc = tokenizer.fc
    quantizer = tokenizer.quantizer
    dsf = config.semantic_downsample_factor
    pad = config.downsample_factor // 2
    traces["downsample_factor"] = config.downsample_factor
    traces["dsf"] = int(dsf)
    traces["hubert_pad"] = pad
    traces["books"] = NUM_BOOKS

    wave_rng = np.random.default_rng(411)
    for tag, n in (("a", 1920), ("b", 3840)):
        wave = wave_rng.standard_normal(n).astype(np.float32) * 0.2
        save_npy(f"{tag}_wave.npy", wave)
        trace_files.append(f"{tag}_wave.npy")

        resampled = np.asarray(
            _sinc_resample(wave, 24000, 16000), dtype=np.float32
        )
        save_npy(f"{tag}_resampled.npy", resampled)
        trace_files.append(f"{tag}_resampled.npy")

        tower_in = np.concatenate(
            [np.zeros(pad, dtype=np.float32), resampled, np.zeros(pad, dtype=np.float32)]
        )
        out = semantic_model(mx.array(tower_in)[None], output_hidden_states=True, return_dict=True)
        states = np.stack([np.asarray(h, dtype=np.float32)[0] for h in out.hidden_states])
        mean = states.mean(axis=0)
        save_npy(f"{tag}_semantic_mean.npy", mean)
        trace_files.append(f"{tag}_semantic_mean.npy")
        traces[f"{tag}_tower_steps"] = int(mean.shape[0])
        traces[f"{tag}_hidden_states_count"] = int(states.shape[0])

        sliced = np.ascontiguousarray(mean[::dsf])
        sem_enc = np.asarray(encoder_semantic(mx.array(sliced)[None]), dtype=np.float32)[0]
        save_npy(f"{tag}_semantic_encoded.npy", sem_enc)
        trace_files.append(f"{tag}_semantic_encoded.npy")

        acoustic = np.asarray(
            acoustic_encoder(mx.array(wave)[None, :, None]), dtype=np.float32
        )[0]  # [T', 256], channels-last like the reference
        save_npy(f"{tag}_acoustic_rows.npy", acoustic)
        trace_files.append(f"{tag}_acoustic_rows.npy")

        steps = min(sem_enc.shape[0], acoustic.shape[0])
        emb = np.concatenate([acoustic[:steps], sem_enc[:steps]], axis=-1)
        fused = np.asarray(fc(mx.array(emb)[None]), dtype=np.float32)[0]
        save_npy(f"{tag}_fused.npy", fused)
        trace_files.append(f"{tag}_fused.npy")

        codes = np.asarray(
            quantizer.encode(mx.array(fused)[None]), dtype=np.int32
        )[0]  # [T, books]
        np.save(OUT / f"{tag}_codes.npy", np.ascontiguousarray(codes))
        trace_files.append(f"{tag}_codes.npy")
        traces[f"{tag}_frames"] = int(codes.shape[0])
        traces[f"{tag}_rvq_min_margin"] = rvq_margins(quantizer, fused.astype(np.float32))

        # The unmodified reference encode must reproduce the staged codes.
        direct = np.asarray(tokenizer.encode(mx.array(wave)[None, :, None]), dtype=np.int32)[0]
        assert (direct == codes).all(), f"{tag}: staged pipeline diverges from the reference encode"

    (OUT / "traces.json").write_text(json.dumps(traces, indent=2))

    files = [
        "config.json",
        "model.safetensors",
        "traces.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_higgs_encode_fixture.py",
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
