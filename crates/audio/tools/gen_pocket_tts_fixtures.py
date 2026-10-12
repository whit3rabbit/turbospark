#!/usr/bin/env python3
"""Regenerate the pocket_tts TTS family Rust fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/pocket_tts, mlx 0.32.3). The reference FlowLMModel
is constructed directly (`Model.__init__` would build the full Mimi
adapter and load a sentencepiece tokenizer; the sentencepiece processor
is stubbed and the Mimi side never runs because the port ends at the
scaled latents). Seeded tiny dims: d_model 32, 2 layers, 4 heads,
lookup dim 16 with an output projection, latent dim 8, flow dim 32
depth 2.

The generation loop runs the true `_run_flow_lm` chain:
conditioning prefill, text prefill, then per-step
`_sample_next_latent` (NaN-seeded bos replacement, temperature-scaled
noise, lsd_decode Euler, out_eos threshold). Noise draws are recorded
per step from controlled numpy streams and replayed by the Rust tests.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_pocket_tts_fixtures.py

Outputs into crates/audio/testdata/tts_pocket_tts/:

- generate/model.safetensors + generate/config.json
    Seeded tiny FlowLM; a seed search pins scenarios where every EOS
    decision holds a margin against cross-engine accumulation noise.
- generate/traces.json
    Conditioning rows, text token ids, per scenario: noise draws in
    consumption order, per-step EOS scores and flags, latents, scaled
    decoding-input latents, eos_step, and the stop reason.
- flow/traces.json
    `lsd_decode` on a fixed condition and start vector with 3 Euler
    steps: per-step s, t, flow direction, and integrated state.
- text/traces.json
    `prepare_text_prompt` normalization cases.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest.mock as mock
from pathlib import Path

import mlx.core as mx
import mlx.core.random as mx_random
import numpy as np
from mlx.utils import tree_flatten, tree_map

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

OUT = Path("../turbospark/crates/audio/testdata/tts_pocket_tts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

N_BINS = 24
LDIM = 8
D_MODEL = 32
FRAME_RATE = 12.5
SAMPLE_RATE = 24000


def stub_tokens(text: str) -> list[int]:
    total = sum(ord(ch) for ch in text)
    return [1 + total % 20, 2 + len(text) % 12]


class StubSentencePiece:
    """The sentencepiece surface LUTConditioner touches."""

    def __init__(self, path):
        pass

    def vocab_size(self):
        return N_BINS

    def encode(self, text, out_type=int):
        return stub_tokens(text)


def build_flow_lm(seed: int):
    from mlx_audio.tts.models.pocket_tts.config import (
        FlowConfig,
        FlowLMConfig,
        FlowLMTransformerConfig,
        LookupTable,
    )
    from mlx_audio.tts.models.pocket_tts.flow_lm import FlowLMModel

    config = FlowLMConfig(
        dtype="float32",
        flow=FlowConfig(dim=D_MODEL, depth=2),
        transformer=FlowLMTransformerConfig(
            hidden_scale=2,
            max_period=10000,
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
        ),
        lookup_table=LookupTable(
            dim=16,
            n_bins=N_BINS,
            tokenizer="stub",
            tokenizer_path="stub.model",
        ),
        weights_path=None,
    )
    mx.random.seed(seed)
    with mock.patch("sentencepiece.SentencePieceProcessor", StubSentencePiece):
        model = FlowLMModel.from_config(config, latent_dim=LDIM)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    # The timestep embedders' `freqs` tables are derived state; keep
    # their true init values out of the seeded weights.
    flat = {
        name: weight
        for name, weight in flat.items()
        if "freqs" not in name
    }
    for name, weight in list(flat.items()):
        if "out_norm" in name or name.endswith("norm.weight") or name.endswith("norm.bias"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
        if "in_ln" in name:
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
        if "emb_std" in name:
            flat[name] = mx.random.uniform(low=0.8, high=1.2, shape=weight.shape)
        if "emb_mean" in name:
            flat[name] = mx.random.uniform(low=-0.2, high=0.2, shape=weight.shape)
        if "final_layer.norm_final" in name:
            # Affine-free norm carries no parameters.
            pass
    model.load_weights(list(flat.items()), strict=False)
    mx.eval(model)
    return model, flat


class NoopModel:
    """The pocket Model surface the generate loop needs, over a bare
    FlowLMModel (the Mimi side never runs in the latent path)."""


def record_normal():
    """Patch mx.random.normal to draw from a controlled numpy stream."""
    calls: list[list[float]] = []
    rng = np.random.default_rng(700)

    def wrapper(*args, **kwargs):
        shape = kwargs.get("shape") or (1,)
        width = 1
        for dim in shape:
            width *= int(dim)
        draw = rng.standard_normal(width).astype(np.float32)
        calls.append(draw.ravel().tolist())
        return mx.array(draw.reshape(shape))

    return calls, mock.patch.object(mx_random, "normal", wrapper)


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


class RecordLinear:
    """Wraps the out_eos head and records every score it produces."""

    def __init__(self, inner):
        self.inner = inner
        self.scores = []

    def __call__(self, x):
        out = self.inner(x)
        self.scores.append(float(out[0, 0]))
        return out


class RecordNorm:
    """Wraps out_norm and records the transformer output rows it sees."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __call__(self, x):
        out = self.inner(x)
        self.calls.append(np.asarray(x)[0])
        return out


def run_generate(model, conditioning, text_tokens, scenario):
    """The reference `_generate_audio_stream_short_text` loop."""
    model.temp = scenario["temperature"]
    model.lsd_decode_steps = scenario["lsd_decode_steps"]
    model.noise_clamp = scenario.get("noise_clamp")
    model.eos_threshold = scenario["eos_threshold"]
    recorder = RecordLinear(model.out_eos)
    model.out_eos = recorder
    norm_recorder = RecordNorm(model.out_norm)
    model.out_norm = norm_recorder
    cache = model.make_cache()
    # Conditioning prefill: the empty text and empty latent inputs
    # contribute nothing; the cache holds the conditioning frames.
    text_embeddings = mx.zeros((1, 0, D_MODEL))
    audio = mx.array(conditioning)[None]
    text_embeddings = mx.concatenate([text_embeddings, audio], axis=1)
    model._sample_next_latent(
        sequence=mx.zeros((1, 0, LDIM)),
        text_embeddings=text_embeddings,
        cache=cache,
        lsd_decode_steps=1,
        temp=1.0,
        noise_clamp=None,
        eos_threshold=-1e9,
    )
    # Text prefill.
    text_embeddings = mx.array(conditioner_tokens(model, text_tokens))[None]
    model._sample_next_latent(
        sequence=mx.zeros((1, 0, LDIM)),
        text_embeddings=text_embeddings,
        cache=cache,
        lsd_decode_steps=1,
        temp=1.0,
        noise_clamp=None,
        eos_threshold=-1e9,
    )

    draws, patch = record_normal()
    latents = []
    decoding_input = []
    eos_flags = []
    eos_step = None
    max_steps = scenario["max_steps"]
    recorder.scores = []
    with patch:
        for step in range(max_steps):
            backbone_input = (
                mx.full((1, 1, LDIM), float("nan"))
                if not latents
                else mx.array(latents[-1])[None, None, :]
            )
            next_latent, is_eos = model._sample_next_latent(
                sequence=backbone_input,
                text_embeddings=mx.zeros((1, 0, D_MODEL)),
                cache=cache,
                lsd_decode_steps=scenario["lsd_decode_steps"],
                temp=scenario["temperature"],
                noise_clamp=scenario.get("noise_clamp"),
                eos_threshold=scenario["eos_threshold"],
            )
            value = np.asarray(next_latent)[0]
            latents.append(floats(value))
            decoding_input.append(
                floats(
                    np.asarray(value * np.array(flat_std(model)) + np.array(flat_mean(model)))
                )
            )
            flags = bool(is_eos.item())
            eos_flags.append(flags)
            if flags and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + scenario["frames_after_eos"]:
                break
    assert len(recorder.scores) >= len(latents), "eos head ran per step"
    eos_scores = recorder.scores[len(recorder.scores) - len(latents):]
    # The out_norm calls: conditioning prefill rows, text prefill rows,
    # then one row per generation step.
    transformer_outputs = [floats(rows.ravel()) for rows in norm_recorder.calls]
    return {
        "noise_calls": draws,
        "latents": latents,
        "decoding_input": decoding_input,
        "eos_scores": eos_scores,
        "eos_flags": eos_flags,
        "eos_step": eos_step,
        "transformer_outputs": transformer_outputs,
    }


def conditioner_tokens(model, tokens):
    """Conditioner embeddings for the recorded token ids."""
    from mlx_audio.tts.models.pocket_tts.conditioners import TokenizedText

    embeddings = model.conditioner(TokenizedText(mx.array(tokens, dtype=mx.int32)[None]))
    return np.asarray(embeddings)[0]


def flat_std(model):
    return np.asarray(model.emb_std).ravel()


def flat_mean(model):
    return np.asarray(model.emb_mean).ravel()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    best = None
    for seed in range(40):
        model, flat = build_flow_lm(seed)
        rng = np.random.default_rng(500)
        conditioning = (rng.standard_normal((6, D_MODEL)) * 0.5).astype(np.float32)
        text_tokens = stub_tokens("hello world")
        scenarios = [
            {
                "name": "eos",
                "temperature": 0.7,
                "lsd_decode_steps": 1,
                "eos_threshold": -4.0,
                "frames_after_eos": 3 + 2,
                "max_steps": 12,
            },
            {
                "name": "max_steps",
                "temperature": 0.7,
                "lsd_decode_steps": 1,
                "eos_threshold": 1e9,
                "frames_after_eos": 99,
                "max_steps": 8,
            },
            {
                "name": "lsd3",
                "temperature": 0.4,
                "lsd_decode_steps": 3,
                "eos_threshold": -4.0,
                "frames_after_eos": 3 + 2,
                "max_steps": 12,
                "noise_clamp": 2.0,
            },
        ]
        records = {}
        ok = True
        for scenario in scenarios:
            record = run_generate(model, conditioning, text_tokens, scenario)
            # EOS decision margins: every score must sit at least 1.0
            # away from the threshold so the f32 port cannot flip it.
            # A never-fires threshold (the max_steps scenario) skips the
            # check.
            if abs(scenario["eos_threshold"]) < 1e8:
                for score, flag in zip(record["eos_scores"], record["eos_flags"]):
                    if not np.isfinite(score):
                        ok = False
                        break
                    distance = score - scenario["eos_threshold"] if flag else (
                        scenario["eos_threshold"] - score
                    )
                    if distance < 1.0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
            records[scenario["name"]] = record
        if ok:
            best = (model, flat, conditioning, text_tokens, scenarios, records, seed)
            print(f"pocket seed {seed}: acceptable scenarios")
            break
    if best is None:
        raise SystemExit("no seed produced acceptable pocket scenarios")
    model, flat, conditioning, text_tokens, scenarios, records, seed = best

    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    # The real Model nests the flow LM under `flow_lm.`; re-prefix the
    # bare module's keys to that layout.
    prefixed = {f"flow_lm.{name}": weight for name, weight in flat.items()}
    mx.save_safetensors(str(case_dir / "model.safetensors"), prefixed)
    config = {
        "model_type": "pocket_tts",
        "flow_lm": {
            "dtype": "float32",
            "flow": {"dim": D_MODEL, "depth": 2},
            "transformer": {
                "hidden_scale": 2,
                "max_period": 10000,
                "d_model": D_MODEL,
                "num_heads": 4,
                "num_layers": 2,
            },
            "lookup_table": {
                "dim": 16,
                "n_bins": N_BINS,
                "tokenizer": "stub",
                "tokenizer_path": "stub.model",
            },
        },
        "mimi": {
            "dtype": "float32",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "frame_rate": FRAME_RATE,
            "quantizer": {"dimension": LDIM, "output_dimension": LDIM},
        },
    }
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    traces = {
        "conditioning": floats(conditioning),
        "text_tokens": text_tokens,
        "frames_after_eos_default": 3 + 2,
        "scenarios": {},
    }
    for scenario in scenarios:
        name = scenario["name"]
        traces["scenarios"][name] = {
            "temperature": scenario["temperature"],
            "lsd_decode_steps": scenario["lsd_decode_steps"],
            "eos_threshold": scenario["eos_threshold"],
            "frames_after_eos": scenario["frames_after_eos"],
            "max_steps": scenario["max_steps"],
            "noise_clamp": scenario.get("noise_clamp"),
            **records[name],
        }
    (case_dir / "traces.json").write_text(json.dumps(traces))

    # lsd_decode trajectory on a fixed condition and start vector.
    rng = np.random.default_rng(760)
    condition = (rng.standard_normal(D_MODEL) * 0.5).astype(np.float32)
    x0 = (rng.standard_normal(LDIM) * 0.3).astype(np.float32)
    current = x0.copy()
    steps = []
    num_steps = 3
    for index in range(num_steps):
        s = index / num_steps
        t = (index + 1) / num_steps
        # Per-stage trace mirroring SimpleMLPAdaLN.__call__.
        fn = model.flow_net
        c = mx.array(condition)[None]
        ts = [mx.full((1, 1, 1), s), mx.full((1, 1, 1), t)]
        ts_embed = []
        ts_silu = []
        for i in range(2):
            embedder = fn.time_embed[i]
            args = ts[i].astype(mx.float32) * embedder.freqs[None, :]
            embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
            linear1 = embedder.mlp[0](embedding)
            silu = linear1 / (1.0 + mx.exp(-linear1))
            normalized = embedder.mlp[3](embedder.mlp[2](silu))
            ts_embed.append(np.asarray(normalized)[0].ravel())
            ts_silu.append(np.asarray(silu)[0].ravel())
        x_in = fn.input_proj(mx.array(current)[None, None, :])
        projected = np.asarray(x_in)[0]
        t_combined = mx.zeros((x_in.shape[0], D_MODEL))
        for i in range(2):
            t_combined = t_combined + fn.time_embed[i](ts[i])
        t_combined = t_combined / 2
        y = t_combined + fn.cond_embed(c)
        block_outs = []
        block_stages = []
        for block in fn.res_blocks:
            # Per-stage trace of the reference ResBlock.__call__:
            # silu first, then the linear.
            modulation = y / (1.0 + mx.exp(-y))
            modulation = block.adaLN_modulation[1](modulation)
            shift_mlp, scale_mlp, gate_mlp = mx.split(modulation, 3, axis=-1)
            h = block.in_ln(x_in)
            h = h * (1 + scale_mlp) + shift_mlp
            h = block.mlp[0](h)
            h = h / (1.0 + mx.exp(-h))
            h = block.mlp[2](h)
            modulated = x_in + gate_mlp * h
            block_stages.append(
                {
                    "modulation": floats(modulation[0]),
                    "normed": floats(np.asarray(block.in_ln(x_in))[0]),
                    "modulated": floats(h[0]),
                    "out": floats(modulated[0]),
                }
            )
            x_in = modulated
            block_outs.append(floats(x_in[0]))
        fl = fn.final_layer
        modulation = y / (1.0 + mx.exp(-y))
        modulation = fl.adaLN_modulation[1](modulation)
        shift, scale = mx.split(modulation, 2, axis=-1)
        normed = fl.norm_final(x_in)
        modulated = normed * (1 + scale) + shift
        final_trace = {
            "modulation": floats(np.asarray(modulation)[0]),
            "normed": floats(np.asarray(normed)[0]),
            "modulated": floats(np.asarray(modulated)[0]),
            "out": floats(np.asarray(fl.linear(modulated))[0]),
        }
        final = fn.final_layer(x_in, y)
        flow_dir = np.asarray(final)[0]
        current = current + flow_dir / num_steps
        steps.append(
            {
                "s": s,
                "t": t,
                "time_embed": [floats(e) for e in ts_embed],
                "time_embed_silu": [floats(e) for e in ts_silu],
                "condition": floats(y[0]),
                "projected": floats(projected),
                "block_outs": block_outs,
                "block_stages": block_stages,
                "final_stages": final_trace,
                "flow_dir": floats(flow_dir),
                "current": floats(current),
            }
        )
    flow_dir = OUT / "flow"
    flow_dir.mkdir(parents=True, exist_ok=True)
    (flow_dir / "traces.json").write_text(
        json.dumps(
            {
                "condition": floats(condition),
                "x0": floats(x0),
                "num_steps": num_steps,
                "steps": steps,
            }
        )
    )

    # prepare_text_prompt string cases.
    from mlx_audio.tts.models.pocket_tts.pocket_tts import prepare_text_prompt

    cases = []
    for text in (
        "hello world",
        "  spaced   out  text  ",
        "lower start.",
        "no end punct",
        "one",
    ):
        normalized, guess = prepare_text_prompt(text)
        cases.append({"input": text, "normalized": normalized, "guess": guess})
    text_dir = OUT / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "traces.json").write_text(json.dumps({"cases": cases}))

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_pocket_tts_fixtures.py",
        "model_seed": seed,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
