#!/usr/bin/env python3
"""Regenerate the CLAP (laion/clap-htsat-unfused) Rust parity fixtures.

Normative reference: Hugging Face transformers 4.57.1 (`ClapModel`,
`ClapFeatureExtractor`, the fast RoBERTa tokenizer from tokenizer.json),
torch CPU, fp32. The real checkpoint is the safetensors conversion of
laion/clap-htsat-unfused at refs/pr/3
(79b58ed25fc00386262a2bea4b19fd21dc4310a0). `pytorch_model.bin` is never
loaded: `from_pretrained(..., use_safetensors=True)`.

Three fixture groups land in crates/embeddings/testdata/clap/:

- features/<clip>.npy
    `ClapFeatureExtractor` output (`input_features[0, 0]`, [1001, 64] f32)
    for deterministic synthetic 48 kHz clips, using the REAL
    preprocessor_config.json (rand_trunc + repeatpad, Slaney mel, fmin 50,
    fmax 14000). Drives the default-suite feature parity test.
- tiny/
    A seeded random-init tiny ClapModel (Swin depths [2, 2, 2], window 4,
    spec_size 64, 16 mel bins; RoBERTa 2 layers, hidden 32, vocab 100)
    saved with save_pretrained (config.json + model.safetensors), plus
    golden intermediates for two mel inputs and three id sequences.
- real/
    Final audio and text embeddings, logits, and tokenizer ids from the
    real pinned checkpoint, for the `#[ignore]`d checkpoint parity tests.

clips.json holds the synthesis recipe the Rust tests replay (no audio is
stored), and manifest.json records versions, the pin, and the SHA-256 of
every fixture file.

rand_trunc is random in the reference (`np.random.randint`). The long
clip is extracted under `np.random.seed(RAND_TRUNC_SEED)`; the offset
that draw produces is replayed here and written to clips.json so the Rust
side crops at the same sample. The script asserts that the cropped clip
re-extracted with no truncation reproduces the reference features.

Run with the pinned venv interpreter (isolated mode):

    <venv>/bin/python -I crates/embeddings/tools/gen_clap_fixtures.py \
        --model-dir ~/models/embeddings/clap-htsat-unfused

Deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch

import transformers
from transformers import AutoTokenizer, ClapConfig, ClapFeatureExtractor, ClapModel

REPO = "laion/clap-htsat-unfused"
REVISION = "79b58ed25fc00386262a2bea4b19fd21dc4310a0"
SR = 48_000
RAND_TRUNC_SEED = 1234

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "clap"

# Synthesis recipe. Mirrored by `synth_clip` in src/clap/tests.rs; keep the
# arithmetic order identical. Noise is an exact xorshift32 stream mapped
# to k * 2^-23 - 1, which is representable in f32 in both languages.
CLIPS = [
    {"name": "tone_3s", "kind": "tone", "samples": 3 * SR},
    {"name": "chirp_noise_10s", "kind": "chirp_noise", "samples": 10 * SR, "seed": 1},
    {"name": "burst_0p7s", "kind": "burst", "samples": 33_600, "seed": 2},
    {"name": "long_12s", "kind": "long", "samples": 12 * SR, "seed": 3},
]

LABELS = [
    "a dog barking",
    "a pure musical tone",
    "white noise",
    "a person speaking",
    "a rising sine sweep",
]

TOKENIZER_PROBES = LABELS + [
    "This is a sound of a dog barking.",
    "  leading and trailing spaces  ",
    "caf\u00e9 na\u00efve \u00fcber",
    "a <mask> in the middle",
    "UPPER lower 12345 !?",
]


def xorshift_noise(n: int, seed: int) -> np.ndarray:
    x = np.uint32(seed if seed != 0 else 0x9E3779B9)
    out = np.empty(n, dtype=np.float64)
    x = int(x)
    for i in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = float(x >> 8) * (2.0**-23) - 1.0
    return out


def synth_clip(spec: dict) -> np.ndarray:
    n = spec["samples"]
    t = np.arange(n, dtype=np.float64) / SR
    kind = spec["kind"]
    if kind == "tone":
        x = 0.3 * np.sin(2.0 * math.pi * 440.0 * t) + 0.15 * np.sin(2.0 * math.pi * 1320.0 * t)
    elif kind == "chirp_noise":
        dur = n / SR
        phase = 2.0 * math.pi * (200.0 * t + (6000.0 - 200.0) * t * t / (2.0 * dur))
        x = 0.25 * np.sin(phase) + 0.05 * xorshift_noise(n, spec["seed"])
    elif kind == "burst":
        x = 0.2 * xorshift_noise(n, spec["seed"]) * np.exp(-4.0 * t)
    elif kind == "long":
        env = 0.5 + 0.5 * np.sin(2.0 * math.pi * 0.25 * t)
        x = 0.2 * np.sin(2.0 * math.pi * 330.0 * t) * env + 0.05 * xorshift_noise(n, spec["seed"])
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


def save_npy(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(np.asarray(arr)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- features


def gen_features(model_dir: Path) -> tuple[dict, dict]:
    fe = ClapFeatureExtractor.from_pretrained(model_dir)
    assert fe.truncation == "rand_trunc" and fe.padding == "repeatpad", (fe.truncation, fe.padding)
    clips_out = []
    feats = {}
    for spec in CLIPS:
        pcm = synth_clip(spec)
        entry = dict(spec)
        entry["checksum_sum"] = float(np.sum(pcm.astype(np.float64)))
        entry["head"] = [float(v) for v in pcm[:8]]
        max_len = fe.nb_max_samples
        if len(pcm) > max_len:
            np.random.seed(RAND_TRUNC_SEED)
            out = fe(pcm, sampling_rate=SR, return_tensors="np")
            np.random.seed(RAND_TRUNC_SEED)
            offset = int(np.random.randint(0, len(pcm) - max_len + 1))
            # The crop is the only randomness: re-extracting the cropped
            # clip (exactly max_len samples, no truncation, no padding)
            # must reproduce the reference features bit for bit.
            again = fe(pcm[offset : offset + max_len], sampling_rate=SR, return_tensors="np")
            assert np.array_equal(out["input_features"], again["input_features"])
            entry["crop_offset"] = offset
        else:
            out = fe(pcm, sampling_rate=SR, return_tensors="np")
            entry["crop_offset"] = 0
        mel = out["input_features"][0, 0].astype(np.float32)
        entry["is_longer"] = bool(out["is_longer"][0][0])
        entry["frames"] = int(mel.shape[0])
        save_npy(OUT / "features" / f"{spec['name']}.npy", mel)
        feats[spec["name"]] = out
        clips_out.append(entry)
    return {"sample_rate": SR, "rand_trunc_seed": RAND_TRUNC_SEED, "clips": clips_out}, feats


# ---------------------------------------------------------------- tiny model


TINY_AUDIO = dict(
    window_size=4,
    num_mel_bins=16,
    spec_size=64,
    patch_size=4,
    patch_stride=[4, 4],
    hidden_size=32,
    depths=[2, 2, 2],
    num_attention_heads=[1, 2, 4],
    patch_embeds_hidden_size=8,
    enable_fusion=False,
    mlp_ratio=4.0,
    layer_norm_eps=1e-5,
    hidden_act="gelu",
    qkv_bias=True,
)
TINY_TEXT = dict(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=40,
    type_vocab_size=1,
    pad_token_id=1,
    layer_norm_eps=1e-12,
    hidden_act="gelu",
)
TINY_TEXT_IDS = [
    [0, 5, 17, 99, 42, 2],
    [0, 3, 8, 13, 21, 34, 55, 89, 44, 33, 22, 11, 10, 9, 7, 6, 4, 2],
    # A pad id (1) mid-sequence: RoBERTa gives it position padding_idx and
    # does not advance the position counter for it.
    [0, 7, 1, 9, 2],
]


def randomize(model: ClapModel, seed: int) -> None:
    """Overwrite every parameter and BatchNorm statistic from a numpy RNG so
    the tiny weights do not depend on torch's init order or RNG."""
    rng = np.random.default_rng(seed)
    sd = model.state_dict()
    with torch.no_grad():
        for name in sorted(sd):
            t = sd[name]
            if not torch.is_floating_point(t) or t.ndim == 0 and "logit_scale" in name:
                continue
            shape = tuple(t.shape)
            if name.endswith("running_var"):
                v = rng.uniform(150.0, 450.0, shape)
            elif name.endswith("running_mean"):
                v = rng.uniform(-45.0, -25.0, shape)
            elif "relative_position_bias_table" in name:
                v = rng.normal(0.0, 1.0, shape)
            elif ("norm" in name.lower()) and name.endswith("weight"):
                v = 1.0 + 0.2 * rng.normal(0.0, 1.0, shape)
            elif name.endswith("bias"):
                v = 0.1 * rng.normal(0.0, 1.0, shape)
            elif t.ndim >= 2:
                fan_in = int(np.prod(shape[1:]))
                v = rng.normal(0.0, 1.0 / math.sqrt(fan_in), shape)
            else:
                v = rng.normal(0.0, 0.5, shape)
            t.copy_(torch.from_numpy(v.astype(np.float32)))
        model.logit_scale_a.copy_(torch.tensor(math.log(10.0)))
        model.logit_scale_t.copy_(torch.tensor(math.log(30.0)))


def capture_audio(model: ClapModel, feats: np.ndarray) -> dict:
    enc = model.audio_model.audio_encoder
    got: dict[str, np.ndarray] = {}
    hooks = []

    def keep(name, fn):
        def hook(_m, _i, out):
            got[name] = fn(out)

        return hook

    hooks.append(
        enc.batch_norm.register_forward_hook(
            keep("batch_norm", lambda o: o[0][:, :, 0].T.numpy().copy())
        )
    )
    hooks.append(enc.patch_embed.register_forward_hook(keep("patch_embed", lambda o: o[0].numpy().copy())))
    for i, layer in enumerate(enc.layers):
        hooks.append(layer.register_forward_hook(keep(f"stage{i}_blocks", lambda o: o[1][0].numpy().copy())))
        hooks.append(layer.register_forward_hook(keep(f"stage{i}_out", lambda o: o[0][0].numpy().copy())))
    hooks.append(enc.norm.register_forward_hook(keep("norm", lambda o: o[0].numpy().copy())))
    orig = enc.reshape_mel2img

    def wrapped(x):
        y = orig(x)
        got["mel2img"] = y[0, 0].numpy().copy()
        return y

    enc.reshape_mel2img = wrapped
    x = torch.from_numpy(feats)[None, None]
    with torch.no_grad():
        out = model.audio_model(input_features=x)
        got["pooled"] = out.pooler_output[0].numpy().copy()
        got["embed"] = model.get_audio_features(input_features=x)[0].numpy().copy()
    enc.reshape_mel2img = orig
    for h in hooks:
        h.remove()
    return got


def capture_text(model: ClapModel, ids: list[int]) -> dict:
    tm = model.text_model
    got: dict[str, np.ndarray] = {}
    hooks = []

    def first(o):
        return o[0] if isinstance(o, tuple) else o

    hooks.append(tm.embeddings.register_forward_hook(lambda _m, _i, o: got.__setitem__("embeddings", o[0].numpy().copy())))
    for i, layer in enumerate(tm.encoder.layer):
        hooks.append(
            layer.register_forward_hook(
                lambda _m, _i, o, i=i: got.__setitem__(f"layer{i}", first(o)[0].numpy().copy())
            )
        )
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = tm(input_ids=x)
        got["pooled"] = out.pooler_output[0].numpy().copy()
        got["embed"] = model.get_text_features(input_ids=x)[0].numpy().copy()
    for h in hooks:
        h.remove()
    return got


def gen_tiny() -> dict:
    cfg = ClapConfig(
        text_config=TINY_TEXT,
        audio_config=TINY_AUDIO,
        projection_dim=16,
        projection_hidden_act="relu",
    )
    torch.manual_seed(0)
    model = ClapModel(cfg).eval()
    randomize(model, seed=20251013)
    tiny = OUT / "tiny"
    tiny.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tiny, safe_serialization=True)
    # Reload from disk so the goldens come from exactly the saved bytes.
    model = ClapModel.from_pretrained(tiny, use_safetensors=True).eval()

    rng = np.random.default_rng(7)
    cases = {
        # 200 frames < spec_width 256: exercises the bicubic stretch.
        "audio_a": (rng.normal(-35.0, 15.0, (200, 16))).astype(np.float32),
        # Exactly spec_width frames: no interpolation.
        "audio_b": (rng.normal(-35.0, 15.0, (256, 16))).astype(np.float32),
    }
    audio_embeds = []
    meta = {"audio": {}, "text": {}}
    for name, feats in cases.items():
        save_npy(tiny / f"{name}_input.npy", feats)
        got = capture_audio(model, feats)
        for k, v in got.items():
            save_npy(tiny / f"{name}_{k}.npy", v.astype(np.float32))
        meta["audio"][name] = {k: list(v.shape) for k, v in got.items()}
        audio_embeds.append(got["embed"])
    text_embeds = []
    for i, ids in enumerate(TINY_TEXT_IDS):
        got = capture_text(model, ids)
        for k, v in got.items():
            save_npy(tiny / f"text{i}_{k}.npy", v.astype(np.float32))
        meta["text"][f"text{i}"] = {"ids": ids, **{k: list(v.shape) for k, v in got.items()}}
        text_embeds.append(got["embed"])
    a = np.stack(audio_embeds)
    t = np.stack(text_embeds)
    scale_a = float(model.logit_scale_a.detach().exp())
    logits = (a @ t.T) * scale_a
    # Cross-check against the reference forward (unpadded texts one at a
    # time give the same per-pair logits as a padded batch).
    with torch.no_grad():
        for ai, (name, feats) in enumerate(cases.items()):
            for ti, ids in enumerate(TINY_TEXT_IDS):
                o = model(
                    input_ids=torch.tensor([ids]),
                    input_features=torch.from_numpy(feats)[None, None],
                )
                assert abs(float(o.logits_per_audio[0, 0]) - float(logits[ai, ti])) < 1e-4
    save_npy(tiny / "logits_per_audio.npy", logits.astype(np.float32))
    meta["logit_scale_a_exp"] = scale_a
    meta["logit_scale_t_exp"] = float(model.logit_scale_t.detach().exp())
    (tiny / "goldens.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


# ---------------------------------------------------------------- real model


def gen_real(model_dir: Path, feats: dict) -> dict:
    model = ClapModel.from_pretrained(model_dir, use_safetensors=True).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    real = OUT / "real"
    real.mkdir(parents=True, exist_ok=True)
    out = {"clips": [c["name"] for c in CLIPS], "labels": LABELS}

    audio = []
    with torch.no_grad():
        for spec in CLIPS:
            f = torch.from_numpy(feats[spec["name"]]["input_features"])
            audio.append(model.get_audio_features(input_features=f)[0].numpy())
        pooled0 = model.audio_model(
            input_features=torch.from_numpy(feats[CLIPS[0]["name"]]["input_features"])
        ).pooler_output[0].numpy()
    audio = np.stack(audio).astype(np.float32)

    # Text the way the zero-shot pipeline batches it: padded with a mask.
    batch = tok(LABELS, padding=True, return_tensors="pt")
    with torch.no_grad():
        text = model.get_text_features(**batch).numpy().astype(np.float32)
        single = np.stack(
            [
                model.get_text_features(**tok([l], return_tensors="pt"))[0].numpy()
                for l in LABELS
            ]
        )
        o = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            input_features=torch.from_numpy(
                np.concatenate([feats[c["name"]]["input_features"] for c in CLIPS])
            ),
        )
        logits = o.logits_per_audio.numpy().astype(np.float32)
        probs = o.logits_per_audio.softmax(dim=-1).numpy().astype(np.float32)
    out["padded_vs_unpadded_text_max_abs"] = float(np.max(np.abs(text - single)))
    out["logit_scale_a_exp"] = float(model.logit_scale_a.detach().exp())
    out["logit_scale_t_exp"] = float(model.logit_scale_t.detach().exp())
    out["label_ids"] = [tok(l)["input_ids"] for l in LABELS]
    out["tokenizer_probes"] = [{"text": s, "ids": tok(s)["input_ids"]} for s in TOKENIZER_PROBES]
    out["logits_per_audio"] = logits.tolist()
    out["probs"] = probs.tolist()
    save_npy(real / "audio_embeds.npy", audio)
    save_npy(real / "text_embeds.npy", single.astype(np.float32))
    save_npy(real / "audio_pooled_0.npy", pooled0.astype(np.float32))
    (real / "goldens.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    args = ap.parse_args()
    model_dir = args.model_dir.expanduser()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    clips, feats = gen_features(model_dir)
    (OUT / "clips.json").write_text(json.dumps(clips, indent=1) + "\n")
    gen_tiny()
    gen_real(model_dir, feats)

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "generator": "crates/embeddings/tools/gen_clap_fixtures.py",
        "reference": "transformers ClapModel / ClapFeatureExtractor (fp32, CPU)",
        "repo": REPO,
        "revision": REVISION,
        "checkpoint_sha256": {
            name: sha256(model_dir / name)
            for name in ["model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json"]
        },
        "versions": {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "safetensors": __import__("safetensors").__version__,
        },
        "files": {str(p.relative_to(OUT)): sha256(p) for p in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {len(files) + 1} files, {total} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
