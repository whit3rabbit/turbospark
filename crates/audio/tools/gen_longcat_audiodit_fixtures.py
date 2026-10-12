#!/usr/bin/env python3
"""Regenerate the LongCat-AudioDiT family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/longcat_audiodit plus the UMT5 encoder, the DiT
transformer, and the WAV-VAE codec, mlx 0.32.3). The true reference
modules run in full: `Model.encode_text` with the text_norm_feat /
text_add_embed post-processing, `_approx_duration` and
`_normalize_text`, `encode_prompt_audio`, the Euler ODE under
classifier-free guidance and APG (including the repa-layer long-skip
add inside the transformer), and the VAE encode/decode, with:

- a seeded tiny stack: a 2-layer UMT5 encoder (d_model 32, 2 heads,
  d_ff 64, vocab 96), a 2-layer DiT (dim 32, 4 heads, ff 128, global
  AdaLN, qk norms, ConvNeXtV2 text conv, long skip, repa layer 1 so
  the mid-stack add is exercised, latent 16), and a tiny WAV-VAE
  (channels 16, c_mults [2, 4], strides [2, 4], hop 8, 8 kHz);
- every `mx.random.normal` draw recorded in reference draw order (the
  VAE reparameterization draw of a voice prompt, then the initial
  latent noise) and replayed by the Rust test;
- a deterministic offline character tokenizer standing in for the
  downloadable `google/umt5-base` tokenizer (text tokenization is
  caller-owned in the port);
- all weights round-tripped through f16 so the traces come from
  exactly the values the Rust fixture loads.

Run from inside the mlx-audio checkout root so `mlx_audio` imports
from the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_longcat_audiodit_fixtures.py

Outputs into crates/audio/testdata/tts_longcat_audiodit/:

- text/      `normalize_text` and `_approx_duration` on sample inputs.
- dit/       the transformer forward on fixed inputs (masked text
             tail, latent conditioning, repa long-skip add).
- generate/  end to end without a prompt: cfg 4.0, 4 Euler steps, the
             recorded noise draw, the pred latent, and the waveform.
- voice/     reference-audio prompt: the padded WAV-VAE encode (with
             its recorded draw), the prompt-frame interpolation, and
             the waveform.
- apg/       the twin of generate under guidance_method="apg".
- manifest.json  file list with SHA-256 digests and generator metadata.

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

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.tts.models.longcat_audiodit import longcat_audiodit as longcat

OUT = Path("../turbospark/crates/audio/testdata/tts_longcat_audiodit")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

TINY_CONFIG = {
    "dit_dim": 32,
    "dit_depth": 2,
    "dit_heads": 4,
    "dit_ff_mult": 4.0,
    "dit_text_dim": 32,
    "dit_dropout": 0.0,
    "dit_bias": True,
    "dit_cross_attn": True,
    "dit_adaln_type": "global",
    "dit_adaln_use_text_cond": True,
    "dit_long_skip": True,
    "dit_text_conv": True,
    "dit_qk_norm": True,
    "dit_cross_attn_norm": False,
    "dit_eps": 1e-6,
    "dit_use_latent_condition": True,
    "repa_dit_layer": 1,
    "latent_dim": 16,
    "sigma": 0.0,
    "sampling_rate": 8000,
    "latent_hop": 8,
    "max_wav_duration": 0.05,
    "text_add_embed": True,
    "text_norm_feat": True,
    "vae_config": {
        "in_channels": 1,
        "channels": 16,
        "c_mults": [2, 4],
        "strides": [2, 4],
        "latent_dim": 16,
        "encoder_latent_dim": 32,
        "use_snake": True,
        "downsample_shortcut": "averaging",
        "upsample_shortcut": "duplicating",
        "out_shortcut": "averaging",
        "in_shortcut": "duplicating",
        "final_tanh": False,
        "downsampling_ratio": 8,
        "sample_rate": 8000,
        "scale": 0.71,
    },
    "text_encoder_config": {
        "vocab_size": 96,
        "d_model": 32,
        "d_kv": 16,
        "d_ff": 64,
        "num_layers": 2,
        "num_heads": 2,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "dropout_rate": 0.0,
        "layer_norm_epsilon": 1e-6,
        "is_gated_act": True,
        "dense_act_fn": "gelu_new",
    },
}

PROMPT_TEXT = "Hello, World!"
REF_TEXT = "hi there"
STEPS = 4
CFG = 4.0


class CharTokenizer:
    """Character-level tokenizer under the tiny vocab."""

    def __init__(self):
        alphabet = sorted(
            set(longcat._normalize_text(PROMPT_TEXT) + " "
                + longcat._normalize_text(REF_TEXT))
        )
        self.vocab = {char: 5 + index for index, char in enumerate(alphabet)}
        assert max(self.vocab.values()) < 96

    def encode(self, text: str, *args, **kwargs) -> list:
        del args, kwargs
        return [self.vocab[char] for char in str(text)]

    def __call__(self, texts, padding=None, return_tensors=None, **kwargs):
        del padding, return_tensors, kwargs
        ids = [self.encode(t) for t in texts]
        longest = max(len(row) for row in ids)
        input_ids = np.array(
            [row + [0] * (longest - len(row)) for row in ids], dtype=np.int64
        )
        attention_mask = np.array(
            [[1.0] * len(row) + [0.0] * (longest - len(row)) for row in ids],
            dtype=np.float32,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


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


def seed_model(model, seed: int) -> None:
    rng = np.random.default_rng(seed)
    flat = dict(tree_flatten(model.parameters()))
    filled = {}
    for name, value in flat.items():
        if name.endswith("norm.weight") or name.endswith(
            "layer_norm.weight"
        ) or name.endswith("layer_norm_sa.weight") or name.endswith(
            "layer_norm_ff.weight"
        ) or name.endswith("final_layer_norm.weight") or name.endswith(
            "q_norm.weight"
        ) or name.endswith("k_norm.weight"):
            filled[name] = mx.array(
                rng.uniform(0.5, 1.5, value.shape).astype(np.float32)
            )
        else:
            filled[name] = mx.array(
                rng.uniform(-0.35, 0.35, value.shape).astype(np.float32)
            )
    model.update(tree_unflatten(filled))


def round_trip_f16(model) -> None:
    flat = dict(tree_flatten(model.parameters()))
    f16 = {key: value.astype(mx.float16).astype(mx.float32) for key, value in flat.items()}
    model.update(tree_unflatten(f16))


def build_model(seed: int) -> longcat.Model:
    mx.random.seed(seed)
    model = longcat.Model(longcat.ModelConfig.from_dict(dict(TINY_CONFIG)))
    seed_model(model, seed)
    round_trip_f16(model)
    return model


def write_case(name: str, weights: dict, traces: dict) -> None:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), weights)
    (case_dir / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2) + "\n")
    (case_dir / "traces.json").write_text(json.dumps(traces))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = build_model(11)
    tokenizer = CharTokenizer()
    weights = {
        name: value.astype(mx.float16)
        for name, value in dict(tree_flatten(model.parameters())).items()
    }

    # The reference generate downloads the `google/umt5-base`
    # tokenizer; stand in a local character tokenizer instead.
    import types

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda name: tokenizer
    )
    sys.modules["transformers"] = fake_transformers

    # --- text case -----------------------------------------------------
    samples = ["Hello, World!", "  mixed   spaces\tand\nnewlines  ",
               "\u201cquoted\u201d and \u2018single\u2019", "Chinese text test",
               "nums 1234 !?"]
    traces_text = {
        "inputs": samples,
        "normalized": [longcat._normalize_text(s) for s in samples],
        "durations": [
            float(longcat._approx_duration(longcat._normalize_text(s), 30.0))
            for s in samples
        ],
        "duration_capped": [
            float(longcat._approx_duration(longcat._normalize_text(s), 0.05))
            for s in samples
        ],
    }
    write_case("text", weights, traces_text)

    # --- dit case ------------------------------------------------------
    rng = np.random.default_rng(5)
    seq = 6
    x = (rng.standard_normal((seq, 16)) * 0.5).astype(np.float32)
    text_cond = (rng.standard_normal((5, 32)) * 0.5).astype(np.float32)
    latent_cond = np.zeros((seq, 16), dtype=np.float32)
    output = model.transformer(
        x=mx.array(x[None]),
        text=mx.array(text_cond[None]),
        text_len=mx.array([3.0]),
        time=mx.array([0.3]),
        mask=mx.ones((1, seq), dtype=mx.bool_),
        cond_mask=mx.array([[True] * 3 + [False] * 2]),
        return_ith_layer=TINY_CONFIG["repa_dit_layer"],
        latent_cond=mx.array(latent_cond[None]),
    )
    traces_dit = {
        "x": floats(x),
        "text": floats(text_cond),
        "text_len": 3.0,
        "time": 0.3,
        "latent_cond": floats(latent_cond),
        "output": floats(output["last_hidden_state"]),
    }
    write_case("dit", weights, traces_dit)

    # --- generate case (no prompt, cfg) --------------------------------
    ids = tokenizer.encode(longcat._normalize_text(PROMPT_TEXT))
    mask = [1.0] * len(ids)
    recorder = DrawRecorder()
    with recorder:
        results = list(
            model.generate(
                PROMPT_TEXT,
                ref_audio=None,
                steps=STEPS,
                cfg_strength=CFG,
                verbose=False,
            )
        )
    assert len(results) == 1
    draws = recorder.flush()
    assert len(draws) == 1, "the no-prompt path draws once"
    traces_generate = {
        "prompt": PROMPT_TEXT,
        "token_prompt": longcat._normalize_text(PROMPT_TEXT),
        "input_ids": ids,
        "attention_mask": mask,
        "steps": STEPS,
        "cfg_strength": CFG,
        "guidance_method": "cfg",
        "draws": [{"shape": shape, "values": values} for shape, values in draws],
        "waveform": floats(results[0].audio),
        "sample_rate": int(TINY_CONFIG["sampling_rate"]),
    }
    write_case("generate", weights, traces_generate)

    # --- voice case (reference-audio prompt) ---------------------------
    rng = np.random.default_rng(23)
    ref_audio = (rng.standard_normal(64) * 0.1).astype(np.float32)
    ref_ids = tokenizer.encode(longcat._normalize_text(f"{REF_TEXT} {PROMPT_TEXT}"))
    # The prompt encode outside the recorded generate (which reseeds
    # mx.random to 1024 internally), with its own draw record: the
    # Rust test replays it for encode_prompt_audio parity.
    mx.random.seed(1024)
    encode_recorder = DrawRecorder()
    with encode_recorder:
        prompt_latent, prompt_dur = model.encode_prompt_audio(mx.array(ref_audio)[None])
    encode_draws = encode_recorder.flush()
    recorder = DrawRecorder()
    with recorder:
        results = list(
            model.generate(
                PROMPT_TEXT,
                ref_audio=ref_audio,
                ref_text=REF_TEXT,
                steps=STEPS,
                cfg_strength=CFG,
                verbose=False,
            )
        )
    draws = recorder.flush()
    assert len(draws) == 2, "the voice path draws for the VAE then the latent"
    assert draws[0][1] == encode_draws[0][1], "the reseeds align the VAE draw"
    traces_voice = {
        "prompt": PROMPT_TEXT,
        "ref_text": REF_TEXT,
        "ref_audio": floats(ref_audio),
        "input_ids": ref_ids,
        "attention_mask": [1.0] * len(ref_ids),
        "steps": STEPS,
        "cfg_strength": CFG,
        "guidance_method": "cfg",
        "draws": [{"shape": shape, "values": values} for shape, values in draws],
        "prompt_latent": floats(prompt_latent),
        "prompt_frames": int(prompt_dur),
        "waveform": floats(results[0].audio),
        "sample_rate": int(TINY_CONFIG["sampling_rate"]),
    }
    write_case("voice", weights, traces_voice)

    # --- apg case -------------------------------------------------------
    recorder = DrawRecorder()
    with recorder:
        results = list(
            model.generate(
                PROMPT_TEXT,
                ref_audio=None,
                steps=STEPS,
                cfg_strength=CFG,
                guidance_method="apg",
                verbose=False,
            )
        )
    draws = recorder.flush()
    traces_apg = {
        "prompt": PROMPT_TEXT,
        "token_prompt": longcat._normalize_text(PROMPT_TEXT),
        "input_ids": ids,
        "attention_mask": mask,
        "steps": STEPS,
        "cfg_strength": CFG,
        "guidance_method": "apg",
        "draws": [{"shape": shape, "values": values} for shape, values in draws],
        "waveform": floats(results[0].audio),
        "sample_rate": int(TINY_CONFIG["sampling_rate"]),
    }
    write_case("apg", weights, traces_apg)

    # --- manifest -------------------------------------------------------
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_longcat_audiodit_fixtures.py",
        "seeds": {"model": 11, "dit": 5, "voice": 23, "generate_seed": 1024},
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
