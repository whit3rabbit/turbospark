#!/usr/bin/env python3
"""Regenerate the Echo-TTS family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/echo_tts, mlx_audio/codec/models/fish_s1_dac,
mlx 0.32.3). The reference modules run in full: the EchoDiT denoiser
(text/speaker encoders, joint attention with latent-prefix KV, low-rank
AdaLN), both euler samplers (whole-sequence and blockwise with the
speaker-KV scale/restore crossing), the PCA latent bridge, and the Fish
S1 DAC codec encode_zq/decode_zq path, with:

- the true byte-level text intake (normalization + BOS) exercised on
  several recorded strings;
- a seeded tiny EchoDiT (latent 8, model 32, 2 blocks, text/speaker
  stacks of 1 layer at model 32) and a seeded tiny fish DAC
  (encoder_dim 8, hop 4, latent 16, dual codebooks 16/12, 2x2
  downsample, window-4 pre/post transformers);
- every `mx.random.normal` draw recorded (one per euler pass) and
  replay-verified across the two runs of each case.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_echo_tts_fixtures.py

Outputs into crates/audio/testdata/tts_echo_tts/:

- greedy_generate/   default sampler, no reference audio; per-step
                     guided velocities recorded
- blockwise_generate/ block_sizes [8, 8, 8] with rescale and
                     speaker-KV scale/restore enabled in the case config
- voice_clone/       reference-audio speaker latent through the codec
                     and PCA
- codec_zq/          standalone codec encode_zq/decode_zq roundtrip
  Each case: config.json + model.safetensors + pca_state.safetensors +
  codec/{config.json,model.safetensors} + traces.json
- text_cases.json    normalization + byte-id table for the recorded
                     strings
- manifest.json      file list with SHA-256 digests

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

sys.path.insert(0, str(Path.cwd()))

from mlx_audio.codec.models.fish_s1_dac import fish_s1_dac as F
from mlx_audio.tts.models.echo_tts.audio import (
    ae_decode,
    ae_encode,
    crop_audio_to_flattening_point,
    find_flattening_point,
    get_speaker_latent_and_mask,
    load_pca_state,
)
from mlx_audio.tts.models.echo_tts.config import ModelConfig
from mlx_audio.tts.models.echo_tts.echo_tts import Model
from mlx_audio.tts.models.echo_tts.sampling import (
    sample_blockwise_euler_cfg_independent_guidances,
    sample_euler_cfg_independent_guidances,
)
from mlx_audio.tts.models.echo_tts.text import (
    get_text_input_ids_and_mask,
    normalize_text_prompt,
)

OUT = Path("../turbospark/crates/audio/testdata/tts_echo_tts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

DIT = {
    "latent_size": 8,
    "model_size": 32,
    "num_layers": 2,
    "num_heads": 4,
    "intermediate_size": 64,
    "norm_eps": 1e-5,
    "text_vocab_size": 256,
    "text_model_size": 32,
    "text_num_layers": 1,
    "text_num_heads": 4,
    "text_intermediate_size": 64,
    "speaker_patch_size": 2,
    "speaker_model_size": 32,
    "speaker_num_layers": 1,
    "speaker_num_heads": 4,
    "speaker_intermediate_size": 64,
    "timestep_embed_size": 16,
    "adaln_rank": 8,
}

SAMPLER_BASE = {
    "num_steps": 5,
    "cfg_scale_text": 3.0,
    "cfg_scale_speaker": 8.0,
    "cfg_min_t": 0.5,
    "cfg_max_t": 1.0,
    "truncation_factor": None,
    "rescale_k": None,
    "rescale_sigma": None,
    "speaker_kv_scale": None,
    "speaker_kv_max_layers": None,
    "speaker_kv_min_t": None,
    "sequence_length": 24,
}

SAMPLER_BLOCKWISE = dict(
    SAMPLER_BASE,
    num_steps=6,
    cfg_min_t=0.4,
    rescale_k=0.7,
    rescale_sigma=1.1,
    speaker_kv_scale=1.3,
    speaker_kv_max_layers=1,
    speaker_kv_min_t=0.7,
)

MODEL_BASE = {
    "model_type": "echo_tts",
    "sample_rate": 8000,
    "max_text_length": 64,
    "max_speaker_latent_length": 8,
    "audio_downsample_factor": 2048,
    "normalize_text": True,
    "delete_blockwise_modules": False,
    "pca_filename": "pca_state.safetensors",
    "fish_codec_repo": "tiny-local",
    "dit": DIT,
}

LATENT = DIT["latent_size"]
CODEC_LATENT = 16
PCA_SCALE = 0.53


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


class TinyArgs(F.ModelArgs):
    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head


def make_tiny_dac() -> F.DAC:
    """build_ae() at encoder_dim 8, hop 4, latent 16, dual codebooks."""
    q_config = TinyArgs(
        block_size=256,
        n_layer=1,
        n_head=2,
        dim=16,
        intermediate_size=32,
        head_dim=8,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )

    def make_transformer():
        return F.WindowLimitedTransformer(
            causal=True,
            window_size=4,
            input_dim=16,
            config=q_config,
        )

    def transformer_general_config(**kw):
        dim = kw.get("dim", 16)
        return TinyArgs(
            block_size=kw.get("block_size", 256),
            n_layer=kw.get("n_layer", 1),
            n_head=2,
            dim=dim,
            intermediate_size=kw.get("intermediate_size", 32),
            n_local_heads=kw.get("n_local_heads", -1),
            head_dim=8,
            rope_base=kw.get("rope_base", 10000),
            norm_eps=kw.get("norm_eps", 1e-5),
            dropout_rate=kw.get("dropout_rate", 0.1),
            attn_dropout_rate=kw.get("attn_dropout_rate", 0.1),
            channels_first=True,
        )

    quantizer = F.DownsampleResidualVectorQuantize(
        input_dim=16,
        n_codebooks=2,
        codebook_size=12,
        codebook_dim=4,
        quantizer_dropout=0.5,
        downsample_factor=(2, 2),
        semantic_codebook_size=16,
        pre_module=make_transformer(),
        post_module=make_transformer(),
    )
    return F.DAC(
        encoder_dim=8,
        encoder_rates=[2, 2],
        latent_dim=16,
        decoder_dim=16,
        decoder_rates=[2, 2],
        quantizer=quantizer,
        sample_rate=8000,
        causal=True,
        encoder_transformer_layers=[0, 0],
        decoder_transformer_layers=[0, 0],
        transformer_general_config=transformer_general_config,
    )


def seeded_dac(seed: int) -> F.DAC:
    dac = make_tiny_dac()
    flat = dict(tree_flatten(dac.parameters()))
    derived = [k for k in flat if "freqs_cis" in k or "causal_mask" in k]
    trainable = {k: v for k, v in flat.items() if k not in derived}
    rng = np.random.default_rng(seed)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in trainable.items()
    }
    dac.load_weights(list(seeded.items()), strict=False)
    mx.eval(dac)
    return dac


def dac_config_json() -> dict:
    return {
        "sample_rate": 8000,
        "encoder_dim": 8,
        "encoder_rates": [2, 2],
        "latent_dim": 16,
        "decoder_dim": 16,
        "decoder_rates": [2, 2],
        "encoder_transformer_layers": [0, 0],
        "encoder_transformer": {
            "window_size": 16,
            "n_layer": 1,
            "n_head": 2,
            "dim": 32,
            "intermediate_size": 64,
            "head_dim": 8,
            "rope_base": 10000.0,
            "norm_eps": 1e-5,
        },
        "quantizer": {
            "input_dim": 16,
            "n_codebooks": 2,
            "codebook_size": 12,
            "semantic_codebook_size": 16,
            "codebook_dim": 4,
            "downsample_factor": [2, 2],
            "pre": {
                "window_size": 4,
                "n_layer": 1,
                "n_head": 2,
                "dim": 16,
                "intermediate_size": 32,
                "head_dim": 8,
                "rope_base": 10000.0,
                "norm_eps": 1e-5,
            },
            "post": {
                "window_size": 4,
                "n_layer": 1,
                "n_head": 2,
                "dim": 16,
                "intermediate_size": 32,
                "head_dim": 8,
                "rope_base": 10000.0,
                "norm_eps": 1e-5,
            },
        },
    }


def make_model(sampler: dict) -> Model:
    config = dict(MODEL_BASE, sampler=dict(sampler))
    model = Model(ModelConfig.from_dict(config))
    # Deterministic weights: small uniform everywhere, norm weights in
    # [0.5, 1.5] so the activations stay sane.
    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith("norm.weight"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    model.update(tree_unflatten(flat))
    return model


def make_pca() -> dict:
    rng = np.random.default_rng(31)
    # Orthonormal-ish components so the PCA is well conditioned.
    raw = rng.standard_normal((LATENT, CODEC_LATENT)).astype(np.float32)
    q, _r = np.linalg.qr(raw.T)
    components = np.ascontiguousarray(q.T[:LATENT, :CODEC_LATENT]).astype(np.float32)
    mean = (rng.standard_normal(CODEC_LATENT) * 0.1).astype(np.float32)
    return {
        "pca_components": mx.array(components),
        "pca_mean": mx.array(mean),
        "latent_scale": mx.array(PCA_SCALE),
    }


def reference_audio() -> np.ndarray:
    rng = np.random.default_rng(97)
    t = np.arange(4608, dtype=np.float32) / 16000.0
    return (0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * rng.standard_normal(4608)).astype(
        np.float32
    )


def save_case(name: str, model: Model, config: dict, dac: F.DAC) -> None:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(case_dir / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    codec_dir = case_dir / "codec"
    codec_dir.mkdir(exist_ok=True)
    flat = dict(tree_flatten(dac.parameters()))
    derived = [k for k in flat if "freqs_cis" in k or "causal_mask" in k]
    mx.save_safetensors(
        str(codec_dir / "model.safetensors"),
        {k: v for k, v in flat.items() if k not in derived},
    )
    (codec_dir / "config.json").write_text(json.dumps(dac_config_json(), indent=2) + "\n")


def record_text(text: str) -> dict:
    ids, mask = get_text_input_ids_and_mask([text], None, True, False)
    return {
        "raw": text,
        "normalized": normalize_text_prompt(text),
        "ids": ints(ids),
        "mask": [bool(v) for v in np.asarray(mask).ravel()],
    }


def run_generation_case(
    name: str,
    model: Model,
    config: dict,
    dac: F.DAC,
    text: str,
    block_sizes=None,
    ref_audio=None,
) -> None:
    save_case(name, model, config, dac)
    pca = load_pca_state(str(OUT / name / "pca_state.safetensors"))
    latent = DIT["latent_size"]

    # Stage trace: the encoded text state (post text encoder + norm).
    ids, mask, _ = get_text_input_ids_and_mask(
        [text], config["max_text_length"], True, True, False
    )
    text_state = model.model.text_encoder(ids, mask)
    from mlx_audio.tts.models.echo_tts.model import RMSNorm as RefRMSNorm

    text_norm = RefRMSNorm(config["dit"]["text_model_size"], config["dit"]["norm_eps"])
    text_norm.weight = model.model.text_norm.weight
    text_state = text_norm(text_state)

    # Stage trace: the speaker latent actually used. The chunk size is
    # shrunk to the tiny codec's frame length (the reference generate
    # passes the 640 * 2048 default, which presumes the real codec's
    # 2048-sample latent stride).
    if ref_audio is not None:
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(
            dac,
            pca,
            mx.array(ref_audio)[None],
            max_speaker_latent_length=config["max_speaker_latent_length"],
            audio_chunk_size=4096,
            divis_by_patch_size=config["dit"]["speaker_patch_size"],
        )
    else:
        patch = config["dit"]["speaker_patch_size"]
        speaker_latent = mx.zeros((1, patch, latent))
        speaker_mask = mx.zeros((1, patch), dtype=mx.bool_)

    # Per-step velocity trace on the first euler pass.
    velocity_trace = []
    original_call = model.model.__class__.__call__

    def recorded_call(mself, x, t, text_mask, speaker_mask, kv_cache_text,
                      kv_cache_speaker, start_pos=None, kv_cache_latent=None):
        out = original_call(
            mself, x, t, text_mask, speaker_mask, kv_cache_text,
            kv_cache_speaker, start_pos, kv_cache_latent,
        )
        velocity_trace.append(
            {
                "t": float(t[0]),
                "x": floats(x),
                "out": floats(out),
            }
        )
        return out

    model.model.__class__.__call__ = recorded_call
    with DrawRecorder() as recorder:
        latent_out = model.generate_latents(
            text=text,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            rng_seed=17,
            block_sizes=block_sizes,
        )
        noise_draws = recorder.flush()
    model.model.__class__.__call__ = original_call

    # Replay determinism: a second run re-seeds inside the sampler.
    with DrawRecorder() as recorder:
        latent_out2 = model.generate_latents(
            text=text,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            rng_seed=17,
            block_sizes=block_sizes,
        )
        replay_draws = recorder.flush()
    assert replay_draws == noise_draws, "draw replay diverged"
    assert np.allclose(np.array(latent_out), np.array(latent_out2), atol=0, rtol=0), (
        "latent replay diverged"
    )

    # Blockwise stage traces: per-block prefix latents and the first
    # block's latent-prefix KV (all three batch rows, layer 0).
    latent_blocks_trace = []
    if block_sizes:
        rows_so_far = 0
        for bindex, bsize in enumerate(block_sizes):
            prefix = latent_out[:, :rows_so_far, :] if rows_so_far > 0 else mx.zeros(
                (1, 0, latent)
            )
            full_prefix = mx.concatenate([prefix] * 3, axis=0)
            kv = model.model.get_kv_cache_latent(full_prefix)
            latent_blocks_trace.append(
                {
                    "block": int(bindex),
                    "rows": int(rows_so_far),
                    "prefix": floats(prefix),
                    "kv_keys": int(kv[0][0].shape[1]),
                    "kv_k": floats(kv[0][0]),
                    "batch": 3,
                }
            )
            rows_so_far += bsize

    audio_out = ae_decode(dac, pca, latent_out)
    fp = find_flattening_point(latent_out[0])
    cropped = crop_audio_to_flattening_point(audio_out, latent_out[0])

    # The speaker latent path through the codec, for the stage gates.
    if ref_audio is not None:
        speaker_zq = ae_encode(dac, pca, mx.array(ref_audio[:4096])[None, None, :])
        speaker_frames = int(speaker_zq.shape[1])
    else:
        speaker_zq, speaker_frames = None, 0

    traces = {
        "ref_audio": floats(ref_audio) if ref_audio is not None else None,
        "text": {
            "normalized": normalize_text_prompt(text),
            "ids": ints(ids),
            "mask": [bool(v) for v in np.asarray(mask).ravel()],
            "max_text_length": int(config["max_text_length"]),
            "token_count": int(ids.shape[1]),
        },
        "text_state": floats(text_state),
        "speaker": {
            "latent": floats(speaker_latent),
            "mask": [bool(v) for v in np.asarray(speaker_mask).ravel()],
            "shape": list(speaker_latent.shape),
            "encode_zq": floats(speaker_zq) if speaker_zq is not None else None,
            "encode_frames": int(speaker_frames),
        },
        "sampler": dict(config["sampler"]),
        "block_sizes": list(block_sizes) if block_sizes else None,
        "noise_draws": [
            {"shape": shape, "values": values} for shape, values in noise_draws
        ],
        "diT_calls": velocity_trace,
        "latent_blocks": latent_blocks_trace,
        "latents": floats(latent_out),
        "latent_frames": int(latent_out.shape[1]),
        "flattening_point": int(fp),
        "waveform_full": floats(audio_out),
        "waveform": floats(cropped),
        "sample_rate": int(config["sample_rate"]),
    }
    (OUT / name / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(
        f"{name}: {traces['latent_frames']} latent frames, "
        f"fp {fp}, {len(traces['waveform'])} samples"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pca = make_pca()

    # Text intake table.
    text_cases = [
        "Hello world, this is Echo.",
        "Line one\nLine two; more: done",
        "A quoted \u2019pause\u201d and an ellipsis\u2026 plus an em\u2014dash",
        "(Parenthetical speaker tag stays)",
        "[S1] Already tagged.",
        "S2 speaks here.",
    ]
    (OUT / "text_cases.json").write_text(
        json.dumps(
            {
                "cases": [record_text(text) for text in text_cases],
                "byte_model": "utf8 with zero BOS",
            },
            indent=2,
        )
        + "\n"
    )

    def install_pca(name: str) -> None:
        mx.save_safetensors(str(OUT / name / "pca_state.safetensors"), pca)

    # Case 1: greedy whole-sequence generation, no reference audio.
    name = "greedy_generate"
    dac = seeded_dac(41)
    model = make_model(SAMPLER_BASE)
    save_case(name, model, dict(MODEL_BASE, sampler=dict(SAMPLER_BASE)), dac)
    install_pca(name)
    run_generation_case(
        name, model, dict(MODEL_BASE, sampler=dict(SAMPLER_BASE)), dac,
        "Hello world, this is Echo.",
    )

    # Case 2: blockwise generation with rescale and speaker-KV scaling.
    name = "blockwise_generate"
    dac = seeded_dac(43)
    model = make_model(SAMPLER_BLOCKWISE)
    config2 = dict(MODEL_BASE, sampler=dict(SAMPLER_BLOCKWISE))
    save_case(name, model, config2, dac)
    install_pca(name)
    run_generation_case(
        name, model, config2, dac,
        "Testing blocks; voice carried over.",
        block_sizes=[8, 8, 8],
    )

    # Case 3: voice clone through the reference-audio path.
    name = "voice_clone"
    dac = seeded_dac(47)
    model = make_model(SAMPLER_BASE)
    config3 = dict(MODEL_BASE, sampler=dict(SAMPLER_BASE))
    save_case(name, model, config3, dac)
    install_pca(name)
    run_generation_case(
        name, model, config3, dac,
        "Cloning a tiny reference voice.",
        ref_audio=reference_audio(),
    )

    # Case 4: standalone codec roundtrip (encode_zq / decode_zq).
    name = "codec_zq"
    dac = seeded_dac(53)
    save_case(name, make_model(SAMPLER_BASE), dict(MODEL_BASE, sampler=dict(SAMPLER_BASE)), dac)
    install_pca(name)
    rng = np.random.default_rng(59)
    wave = (rng.standard_normal(64) * 0.2).astype(np.float32)  # 4 frames
    zq = dac.encode_zq(mx.array(wave)[None])
    zq_frames = int(zq.shape[2])
    decoded = dac.decode_zq(zq.astype(mx.float32))
    codes, code_len = dac.encode(mx.array(wave)[None])
    # Crop math with a flattening tail: tiny random latents whose last
    # frames are exactly zero, so find_flattening_point fires inside.
    rng_crop = np.random.default_rng(61)
    tail_rows = 20
    tail_latent = np.concatenate(
        [
            (rng_crop.standard_normal((14, LATENT)) * 0.8).astype(np.float32),
            np.zeros((6, LATENT), dtype=np.float32),
        ],
        axis=0,
    )
    from mlx_audio.tts.models.echo_tts.audio import PCAState

    tail_pca = PCAState(
        pca_components=pca["pca_components"],
        pca_mean=pca["pca_mean"],
        latent_scale=float(pca["latent_scale"].item()),
    )
    tail_fp = find_flattening_point(mx.array(tail_latent))
    tail_decoded = ae_decode(dac, tail_pca, mx.array(tail_latent)[None])
    tail_cropped = crop_audio_to_flattening_point(tail_decoded, mx.array(tail_latent))
    traces = {
        "wave": floats(wave),
        "zq": floats(zq),
        "zq_frames": int(zq_frames),
        "zq_shape": list(zq.shape),
        "decoded": floats(decoded),
        "codes": ints(codes),
        "codes_shape": list(codes.shape),
        "code_len": int(code_len),
        "frame_length": int(dac.frame_length),
        "crop": {
            "latent": floats(tail_latent),
            "latent_rows": int(tail_rows),
            "flattening_point": int(tail_fp),
            "decoded": floats(tail_decoded),
            "cropped": floats(tail_cropped),
        },
    }
    (OUT / name / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(f"codec_zq: {zq_frames} frames, {np.asarray(decoded).shape[-1]} samples")

    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "generator": "tools/gen_echo_tts_fixtures.py",
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
