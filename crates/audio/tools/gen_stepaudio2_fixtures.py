#!/usr/bin/env python3
"""Regenerate the StepAudio2 token-to-wav Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/codec/models/stepaudio2 and the chatterbox s3gen modules it
imports; mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_stepaudio2_fixtures.py

Outputs into crates/audio/testdata/stepaudio2/:

- tiny_config.json + tiny_weights.safetensors
    One seeded checkpoint holding all three submodels under the
    prefixes the Rust loader reads: `flow.*` (CausalMaskedDiffWithXvec
    with a reduced encoder/DiT), `hift.*` (StepAudio2HiFTGenerator at
    the full reference geometry), and `campplus.*` (StepAudio2CAMPPlus
    at the full reference geometry). Weights are stored in the MLX
    parameter-tree layouts the reference uses; the Rust loaders own the
    layout conversions.
- traces.json + npy files
    Per-stage goldens: kaldi fbank + CAMPPlus embedding, the 24 kHz
    chatterbox prompt mel plus the length-matched prompt_feat, the flow
    output mel for the generated span, the HiFT f0, and the HiFT and
    end-to-end waveforms. The HiFT sine-generator randomness (initial
    phase draw and sine-branch noise) is recorded while running the
    reference and replayed for the end-to-end call, so both waveforms
    share one noise realization; the Rust port takes those tensors as
    explicit inputs.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.stepaudio2.decoder_dit import DiT
from mlx_audio.codec.models.stepaudio2.flow import CausalMaskedDiffWithXvec
from mlx_audio.codec.models.stepaudio2.flow_matching import CausalConditionalCFM
from mlx_audio.codec.models.stepaudio2.hift import StepAudio2HiFTGenerator
from mlx_audio.codec.models.stepaudio2.speaker import StepAudio2CAMPPlus
from mlx_audio.codec.models.stepaudio2.token2wav import StepAudio2Token2Wav
from mlx_audio.codec.models.stepaudio2.upsample_encoder_v2 import (
    UpsampleConformerEncoderV2,
)
from mlx_audio.tts.models.chatterbox.s3gen.mel import mel_spectrogram
from mlx_audio.tts.models.chatterbox.s3gen.xvector import kaldi_fbank

OUT = Path("../turbospark/crates/audio/testdata/stepaudio2")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

N_MELS = 80
VOCAB = 6561
SPK_DIM = 192
N_TIMESTEPS = 10
PROMPT_TOKENS_LEN = 12
GEN_TOKENS_LEN = 8

# Keys match the reference keyword names the Rust config parser reads.
TINY_FLOW = dict(
    input_size=64,
    output_size=N_MELS,
    spk_embed_dim=SPK_DIM,
    vocab_size=VOCAB,
    linear_units=128,
    attention_heads=4,
    num_blocks=2,
    num_up_blocks=2,
    pre_lookahead_len=3,
    up_stride=2,
    up_scale_factor=2,
    dit_hidden=64,
    dit_depth=2,
    dit_heads=4,
    dit_head_dim=16,
    dit_mlp_ratio=4.0,
    inference_cfg_rate=0.7,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def save_i32(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.int32))


def build_flow(cfg: dict) -> CausalMaskedDiffWithXvec:
    encoder = UpsampleConformerEncoderV2(
        input_size=cfg["input_size"],
        output_size=cfg["input_size"],
        input_layer="linear",
        pre_lookahead_len=cfg["pre_lookahead_len"],
        num_blocks=cfg["num_blocks"],
        num_up_blocks=cfg["num_up_blocks"],
        up_stride=cfg["up_stride"],
        up_scale_factor=cfg["up_scale_factor"],
        attention_heads=cfg["attention_heads"],
        pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn",
        key_bias=True,
        linear_units=cfg["linear_units"],
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
    )
    decoder = CausalConditionalCFM(
        estimator=DiT(
            in_channels=4 * cfg["output_size"],
            out_channels=cfg["output_size"],
            mlp_ratio=cfg["dit_mlp_ratio"],
            depth=cfg["dit_depth"],
            num_heads=cfg["dit_heads"],
            head_dim=cfg["dit_head_dim"],
            hidden_size=cfg["dit_hidden"],
        ),
        inference_cfg_rate=cfg["inference_cfg_rate"],
    )
    return CausalMaskedDiffWithXvec(
        input_size=cfg["input_size"],
        output_size=cfg["output_size"],
        spk_embed_dim=cfg["spk_embed_dim"],
        vocab_size=cfg["vocab_size"],
        encoder=encoder,
        decoder=decoder,
    )


class DrawPatch:
    """Records (or replays) the mx.random draws a HiFT call makes.

    The reference draws randomness inside SineGen at inference time; the
    Rust port takes those tensors as explicit inputs, so the fixture
    records the exact arrays the reference used and replays them for the
    end-to-end waveform.
    """

    def __init__(self, replay: list[np.ndarray] | None = None):
        self.replay = replay
        self.draws: list[np.ndarray] = []
        self.pre_clip_wav: np.ndarray | None = None
        self._orig: dict = {}

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
        self._orig["clip"] = mx.clip

        def clip(a, a_min, a_max, *args, **kwargs):
            if (
                self.pre_clip_wav is None
                and a.ndim == 2
                and a_min == -0.99
                and a_max == 0.99
            ):
                self.pre_clip_wav = np.asarray(a)
            return self._orig["clip"](a, a_min, a_max, *args, **kwargs)

        mx.clip = clip
        return self

    def __exit__(self, *exc) -> None:
        for name, orig in self._orig.items():
            setattr(mx.random if name in ("uniform", "normal") else mx, name, orig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files: list[str] = []

    # Seeded submodels. The flow uses the reduced geometry in TINY_FLOW;
    # the HiFT vocoder and CAMPPlus speaker encoder keep the full
    # reference geometry (their structure is not configurable).
    mx.random.seed(401)
    flow = build_flow(TINY_FLOW)
    hift = StepAudio2HiFTGenerator()
    campplus = StepAudio2CAMPPlus()
    models = {"flow": flow, "hift": hift, "campplus": campplus}

    rng = np.random.default_rng(403)
    # Per-prefix weight scales: the HiFT residual stream plus the exp()
    # magnitude head amplify quickly, so the vocoder uses a gentler
    # scale that keeps the pre-exp values small (documented in
    # traces.json); the waveform still has unit scale because exp(0)=1
    # anchors the magnitude branch.
    weight_scales = {"flow": 0.2, "hift": 0.02, "campplus": 0.2}
    weights: dict[str, mx.array] = {}
    weights_manifest: dict[str, list] = {}
    alpha_names: list[str] = []
    for prefix, model in models.items():
        model.eval()
        flat = dict(tree_flatten(model.parameters()))
        scale = weight_scales[prefix]
        for name, value in flat.items():
            if "running_var" in name:
                # BatchNorm variance must stay positive; mirror the
                # positive-by-construction checkpoint statistic.
                seeded = 0.3 + np.abs(rng.standard_normal(value.shape)) * 0.4
            elif name.endswith("pos_enc.pe"):
                # The positional-encoding buffer is derived at init and
                # convert.py only ever saves the derived values (the
                # torch state has no pe); keep the deterministic table
                # so the fixture matches a real checkpoint.
                seeded = np.asarray(value, dtype=np.float32)
            else:
                seeded = rng.standard_normal(value.shape) * scale
            weights[f"{prefix}.{name}"] = mx.array(seeded.astype(np.float32))
            if ".alpha" in name:
                alpha_names.append(f"{prefix}.{name}")
        weights_manifest[prefix] = [
            {"name": n, "shape": list(v.shape)} for n, v in sorted(flat.items())
        ]

    # Deliberate overrides, documented in traces.json:
    # - Snake alphas stay positive and near one so the activation keeps
    #   its reference scale under random weights.
    # - The f0 predictor classifier bias holds the f0 estimate in the
    #   voiced range (above the threshold of 10) so the NSF sine branch
    #   is exercised.
    for name in alpha_names:
        weights[name] = mx.array(
            (1.0 + 0.25 * rng.standard_normal(weights[name].shape)).astype(
                np.float32
            )
        )
    weights["hift.f0_predictor.classifier.bias"] = mx.array(
        (60.0 + 5.0 * rng.standard_normal((1,))).astype(np.float32)
    )
    traces["weight_scales"] = weight_scales

    # Push the seeded weights back into the models (per-prefix names).
    for prefix, model in models.items():
        model.load_weights(
            [
                (name[len(prefix) + 1 :], value)
                for name, value in weights.items()
                if name.startswith(prefix + ".")
            ],
            strict=True,
        )
        mx.eval(model.parameters())

    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), weights)
    (OUT / "tiny_config.json").write_text(json.dumps(TINY_FLOW, indent=2))
    (OUT / "weights_manifest.json").write_text(
        json.dumps(weights_manifest, indent=2)
    )
    trace_files += ["tiny_config.json", "weights_manifest.json"]

    # ------------------------------------------------------------------
    # Inputs: one 16 kHz prompt waveform (0.6 s) for the speaker encoder
    # and one 24 kHz waveform for the prompt mel. They are independent
    # draws; the Rust prompt-prep contract takes the two resampled
    # waveforms (file loading and resampling stay outside the port, like
    # every other codec family). Random speech tokens in the FSQ range.
    wave_16k = (np.random.default_rng(407).standard_normal(9600) * 0.1).astype(
        np.float32
    )
    wave_24k = (np.random.default_rng(409).standard_normal(14400) * 0.1).astype(
        np.float32
    )
    save_npy("audio_16k.npy", wave_16k)
    save_npy("audio_24k.npy", wave_24k)
    prompt_tokens = np.random.default_rng(411).integers(0, VOCAB, PROMPT_TOKENS_LEN)
    gen_tokens = np.random.default_rng(413).integers(0, VOCAB, GEN_TOKENS_LEN)
    save_i32("prompt_tokens.npy", prompt_tokens)
    save_i32("gen_tokens.npy", gen_tokens)
    trace_files += ["audio_16k.npy", "audio_24k.npy", "prompt_tokens.npy", "gen_tokens.npy"]

    # ------------------------------------------------------------------
    # Stage: CAMPPlus speaker encoder. kaldi fbank, per-feature mean
    # removal, batch-of-one forward.
    # Raw fbank gate first; the embedding consumes the mean-removed
    # variant exactly as the reference inference does.
    fbank = np.asarray(kaldi_fbank(mx.array(wave_16k), num_mel_bins=N_MELS))
    save_npy("campplus_fbank.npy", fbank)
    fbank = fbank - fbank.mean(axis=0, keepdims=True)
    embedding = np.asarray(campplus(mx.array(fbank)[None]))
    assert embedding.shape == (1, SPK_DIM), embedding.shape
    save_npy("embedding.npy", embedding[0])
    traces["campplus_fbank_shape"] = list(fbank.shape)
    traces["embedding_shape"] = list(embedding[0].shape)
    traces["embedding_abs_max"] = float(np.abs(embedding).max())
    trace_files += ["campplus_fbank.npy", "embedding.npy"]

    # ------------------------------------------------------------------
    # Stage: prompt mel (chatterbox s3gen mel at 24 kHz) and the
    # length-matched prompt_feat exactly as prepare_prompt builds it.
    prompt_mels = np.asarray(mel_spectrogram(mx.array(wave_24k)[None]))
    # (B, M, T') -> (T', M) row-major for the fixture reader.
    prompt_mels = prompt_mels.transpose(0, 2, 1)
    save_npy("prompt_mel.npy", prompt_mels[0])
    traces["prompt_mel_shape"] = list(prompt_mels[0].shape)
    trace_files.append("prompt_mel.npy")

    target_mel_len = PROMPT_TOKENS_LEN * flow.up_rate
    feats = prompt_mels.copy()
    if feats.shape[1] < target_mel_len:
        pad_len = target_mel_len - feats.shape[1]
        tail = np.broadcast_to(feats[:, -1:, :], (1, pad_len, feats.shape[2]))
        feats = np.concatenate([feats, tail], axis=1)
    elif feats.shape[1] > target_mel_len:
        feats = feats[:, :target_mel_len, :]
    assert feats.shape[1] == target_mel_len
    save_npy("prompt_feat.npy", feats[0])
    traces["prompt_feat_shape"] = list(feats[0].shape)
    traces["up_rate"] = int(flow.up_rate)
    traces["target_mel_len"] = int(target_mel_len)
    trace_files.append("prompt_feat.npy")

    # ------------------------------------------------------------------
    # Stage: flow inference. Tokens + prompt -> mel for the generated
    # span, row-major (mel_len2, 80).
    prompt = dict(
        prompt_token=mx.array(prompt_tokens.astype(np.int32))[None],
        prompt_token_len=mx.array([PROMPT_TOKENS_LEN], dtype=mx.int32),
        prompt_feat=mx.array(feats),
        prompt_feat_len=mx.array([feats.shape[1]], dtype=mx.int32),
        embedding=mx.array(embedding),
    )
    mel = flow.inference(
        mx.array(gen_tokens.astype(np.int32))[None],
        mx.array([GEN_TOKENS_LEN], dtype=mx.int32),
        n_timesteps=N_TIMESTEPS,
        **prompt,
    )
    mel = np.asarray(mel)  # (1, 80, mel_len2)
    mel_len2 = mel.shape[2]
    assert mel_len2 == GEN_TOKENS_LEN * flow.up_rate, mel.shape
    save_npy("flow_mel.npy", mel[0].T)
    traces["flow_mel_shape"] = list(mel[0].T.shape)
    traces["flow_mel_abs_max"] = float(np.abs(mel).max())
    trace_files.append("flow_mel.npy")

    # ------------------------------------------------------------------
    # Stage: HiFT standalone. f0 golden first, then the waveform with
    # the recorded sine-generator randomness.
    f0 = np.asarray(hift.f0_predictor(mx.array(mel)))
    save_npy("f0.npy", f0[0])
    traces["f0_shape"] = list(f0[0].shape)
    traces["f0_min"] = float(f0.min())
    trace_files.append("f0.npy")

    with DrawPatch() as recorder:
        wav, _ = hift.inference(speech_feat=mx.array(mel))
    wav = np.asarray(wav)[0]
    assert len(recorder.draws) >= 2, recorder.draws
    assert recorder.pre_clip_wav is not None, "clip hook did not fire"
    pre_clip = recorder.pre_clip_wav[0]
    # The reference zeroes the first harmonic's drawn initial phase
    # before use; save the processed tensor so the Rust input matches.
    rand_ini_saved = recorder.draws[0].copy()
    rand_ini_saved[..., 0] = 0.0
    save_npy("sine_rand_ini.npy", rand_ini_saved)
    save_npy("sine_noise.npy", recorder.draws[1])
    save_npy("hift_wav_raw.npy", pre_clip)
    save_npy("hift_wav.npy", wav)
    traces["sine_rand_ini_shape"] = list(recorder.draws[0].shape)
    traces["sine_noise_shape"] = list(recorder.draws[1].shape)
    traces["hift_wav_len"] = int(wav.shape[0])
    traces["hift_wav_raw_abs_max"] = float(np.abs(pre_clip).max())
    traces["hift_wav_clip_fraction"] = float(
        (np.abs(wav) >= 0.9899).mean()
    )
    trace_files += [
        "sine_rand_ini.npy",
        "sine_noise.npy",
        "hift_wav_raw.npy",
        "hift_wav.npy",
    ]

    # ------------------------------------------------------------------
    # Stage: end-to-end decode with the same noise realization.
    orchestrator = StepAudio2Token2Wav(
        flow=flow, hift=hift, speech_tokenizer=None, speaker_encoder=campplus
    )
    with DrawPatch(replay=recorder.draws) as replay_patch:
        wav2 = orchestrator.decode(
            mx.array(gen_tokens.astype(np.int32))[None],
            prompt,
            n_timesteps=N_TIMESTEPS,
        )
    wav2 = np.asarray(wav2)[0]
    assert wav2.shape == wav.shape, (wav2.shape, wav.shape)
    save_npy("full_wav_raw.npy", replay_patch.pre_clip_wav[0])
    save_npy("full_wav.npy", wav2)
    traces["full_wav_len"] = int(wav2.shape[0])
    traces["full_vs_hift_max_abs_diff"] = float(np.abs(wav2 - wav).max())
    trace_files += ["full_wav_raw.npy", "full_wav.npy"]

    (OUT / "traces.json").write_text(json.dumps(traces, indent=2))

    files = ["tiny_weights.safetensors", "traces.json", *trace_files]
    manifest = {
        "generator": "tools/gen_stepaudio2_fixtures.py",
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "notes": [
            "snake alphas reseeded positive ~1.0",
            "f0 classifier bias reseeded to the voiced range",
            "sine-generator randomness recorded and replayed",
        ],
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 1} fixture files to {OUT}")
    for key in (
        "embedding_abs_max",
        "flow_mel_abs_max",
        "f0_min",
        "hift_wav_raw_abs_max",
        "hift_wav_clip_fraction",
        "full_vs_hift_max_abs_diff",
    ):
        print(f"  {key}: {traces[key]}")


if __name__ == "__main__":
    main()
