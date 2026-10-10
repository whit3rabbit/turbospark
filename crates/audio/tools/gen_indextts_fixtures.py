#!/usr/bin/env python3
"""Regenerate the indextts family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/indextts,
mlx 0.32.3). The reference modules are imported directly: the true
indextts `Model` (gpt2 backbone with identity embedding patches,
Conformer + PerceiverResampler conditioning, local ecapa_tdnn speaker
encoder, conditioned BigVGAN vocoder) at tiny seeded dims, the true
`prepare_input_embedding` and generate loop, and the true
`log_mel_spectrogram`.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_indextts_fixtures.py

Outputs into crates/audio/testdata/tts_indextts/:

- generate/model.safetensors + generate/config.json
    Seeded tiny Model (gpt2 2 layers over dim 32 / 4 heads, conformer
    2 blocks at output 32, perceiver 8 latents, 16 mel codes, 24 text
    tokens, two-stage conditioned BigVGAN). Weights are stored F16 and
    reloaded upcast before the traces run (llama fixture convention),
    except the computed `pos_enc._pe` buffer, which stays f32 because
    the Rust recomputes the table instead of reading it.
- generate/traces.json
    The reference waveform, its log-mel rows, the conditioning rows,
    the prompt embedding, and two greedy scenarios: a stop-token stop
    (the mel head's stop row amplified) and a max-tokens stop, each
    with the sampled ids, the per-step latent states, and the vocoder
    waveform.
- sampler/traces.json
    Greedy argmax rows and the top-k masked logits (make_sampler's
    kth-value mask with ties kept) on fixed logit rows.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access (the
sentencepiece tokenizer is trained locally in a temp directory). The
backbone seed at the bottom is pinned to the first seed of the search
below whose greedy runs satisfy both scenario shapes.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import sys

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

import sentencepiece as spm

from mlx_audio.tts.models.indextts.indextts import Model, ModelArgs
from mlx_audio.tts.models.indextts.mel import log_mel_spectrogram
from mlx_audio.tts.models.indextts import normalize
from mlx_audio.lm.sample_utils import make_sampler

OUT = Path("../turbospark/crates/audio/testdata/tts_indextts")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

DIM = 32
HEADS = 4
LAYERS = 2
MEL_CODES = 16
START_MEL = 14
STOP_MEL = 15
TEXT_TOKENS = 24
START_TEXT = 22
STOP_TEXT = 23
N_LATENTS = 8
MEL_HOP = 256

AMPLIFY_STOP = 1.8
BACKBONE_SEED = 30
STOP_BUDGET = 60
MAXTOKENS_BUDGET = 6

TEXT = "Hello there."

CONFIG = {
    "bigvgan": {
        "num_mels": 100,
        "upsample_rates": [8, 8],
        "upsample_kernel_sizes": [16, 16],
        "upsample_initial_channel": 32,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7],
        "resblock_dilation_sizes": [[1, 3], [1, 3]],
        "activation": "snakebeta",
        "snake_logscale": True,
        "use_bias_at_final": True,
        "use_tanh_at_final": True,
        "gpt_dim": DIM,
        "speaker_embedding_dim": 4,
        "cond_d_vector_in_each_upsampling_layer": True,
    },
    "gpt": {
        "model_dim": DIM,
        "heads": HEADS,
        "layers": LAYERS,
        "max_mel_tokens": 48,
        "max_text_tokens": 48,
        "number_text_tokens": TEXT_TOKENS,
        "number_mel_codes": MEL_CODES,
        "start_mel_token": START_MEL,
        "stop_mel_token": STOP_MEL,
        "start_text_token": START_TEXT,
        "stop_text_token": STOP_TEXT,
        "use_mel_codes_as_input": True,
        "mel_length_compression": 1024,
        "condition_type": "conformer_perceiver",
        "max_conditioning_inputs": 1,
        "condition_num_latent": N_LATENTS,
        "condition_module": {
            "input_size": 100,
            "output_size": 32,
            "num_blocks": 2,
            "linear_units": 64,
            "attention_heads": 4,
            "pos_enc_layer_type": "rel_pos",
            "input_layer": "conv2d2",
            "cnn_module_kernel": 5,
            "pos_emb_max_len": 256,
            "use_bias": True,
            "xscaling": True,
            "macaron_style": False,
            "perceiver_mult": 2,
        },
    },
    "tokenizer_name": "fixture",
    "sample_rate": 24000,
}


def train_tokenizer(directory: Path) -> Path:
    """A tiny deterministic sentencepiece model so Model.__init__ can
    load a tokenizer without network access; the Rust port never reads
    it (tokenization is caller-owned)."""
    corpus = directory / "corpus.txt"
    corpus.write_text(
        "hello there general kenobi\n"
        "the quick brown fox jumps over the lazy dog\n"
        "index text to speech synthesis model\n"
    )
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(directory / "tokenizer"),
        vocab_size=32,
        character_coverage=0.9995,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )
    return directory


def seed_weights(model, seed: int, scale: float = 0.08) -> dict:
    flat = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(seed)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * scale).astype(np.float32))
        for name, v in flat.items()
    }
    # The small scale keeps the attentive-statistics pooling softmax
    # numerically stable (at the 0.3 default the seeded logits saturate
    # the softmax into one-hot weights, and a 1e-3 f32 ordering
    # difference flips the argmax, moving the pooled context by tens).
    # BatchNorm running variances must stay positive (the real
    # checkpoints carry trained variances); a seeded negative would NaN
    # the sqrt in both implementations.
    for name, value in list(seeded.items()):
        if "running_var" in name:
            seeded[name] = mx.array(np.abs(np.array(value)) + 1.0)
    model.load_weights(list(seeded.items()))
    mx.eval(model)
    return seeded


def store(model, seed: int, amplify_stop: float | None):
    """Seed, optionally amplify the stop row, and round to the F16 save
    layout (computed position tables stay f32). Returns the stored
    dict."""
    seeded = seed_weights(model, seed)
    if amplify_stop is not None:
        head = np.array(model.mel_head.weight, dtype=np.float32)
        head[STOP_MEL, :] *= amplify_stop
        model.mel_head.weight = mx.array(head)
        mx.eval(model)
    stored = {}
    for name, value in tree_flatten(model.parameters()):
        if "pos_enc" in name:
            stored[name] = value
        else:
            stored[name] = value.astype(mx.float16)
    model.update(
        tree_unflatten(
            {
                k: (v if "pos_enc" in k else v.astype(mx.float32))
                for k, v in stored.items()
            }
        )
    )
    mx.eval(model)
    return stored, seeded


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def masked_floats(arr) -> list:
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(arr, dtype=np.float32).ravel().tolist()
    ]


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def raw_loop(model, embedding, sampler, max_tokens):
    """The true generate loop body (batch of one), returning the ids,
    the latent rows, and whether the stop token closed the run."""
    from mlx_audio.lm.models.cache import KVCache

    cache = [KVCache() for _ in range(model.args.gpt.layers)]
    inputs = embedding
    generated = []
    latent_states = []
    stopped = False
    mel_position = 0
    for _ in range(max_tokens):
        hidden_states = model.gpt(inputs, cache=cache)
        hidden_states = model.final_norm(hidden_states)
        latent_states.append(np.asarray(hidden_states[:, -1:, :], dtype=np.float32))
        mel_logits = model.mel_head(hidden_states[:, -1:, :])
        next_token = sampler(mel_logits)
        if int(next_token.item()) == model.args.gpt.stop_mel_token:
            stopped = True
            break
        generated.append(int(next_token.item()))
        mel_emb = model.mel_embedding(next_token) + model.mel_pos_embedding(
            next_token, embedding.shape[1] + mel_position
        )
        inputs = mel_emb
        mel_position += 1
    return generated, np.concatenate(latent_states, axis=-2), stopped


def scenario(model, ref_mel, tokenized, sampler, max_tokens, name):
    # The model is already in the stored (F16-upcast, stop-amplified)
    # state the search left it in.
    embedding = model.prepare_input_embedding([TEXT], None, ref_mel)
    mx.eval(embedding)
    generated, latents, stopped = raw_loop(model, embedding, sampler, max_tokens)
    audio = model.bigvgan(
        mx.array(latents).transpose(0, 2, 1), ref_mel.transpose(0, 2, 1)
    )
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return {
        "name": name,
        "max_tokens": max_tokens,
        "amplify_stop": AMPLIFY_STOP,
        "stop_reason": "stop-mel-token" if stopped else "max-tokens",
        "tokens": generated,
        "latent_states": floats(latents[0]),
        "num_latent_rows": int(latents.shape[1]),
        "waveform": floats(audio),
        "prompt_embedding": floats(embedding[0]),
        "prompt_length": int(embedding.shape[1]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer_dir = train_tokenizer(Path(tmp))
        CONFIG["tokenizer_name"] = str(tokenizer_dir)

        # --- reference waveform ------------------------------------------------
        rng = np.random.default_rng(4242)
        ref_audio = (
            0.3 * np.sin(2 * np.pi * 220.0 * np.arange(8448) / 24000.0)
            + 0.05 * rng.standard_normal(8448)
        ).astype(np.float32)
        ref_mel = log_mel_spectrogram(mx.array(ref_audio))
        mx.eval(ref_mel)
        normalized = normalize.tokenize_by_CJK_char(normalize.normalize(TEXT))
        print(f"normalized text: {normalized!r}")

        # --- backbone search -----------------------------------------------------
        sampler_greedy = make_sampler(temp=0.0)
        best = None
        for seed in range(1, 120):
            args = from_dict_args(CONFIG)
            mx.random.seed(seed)
            model = Model(args)
            model.eval()
            model._fixture_seed = seed
            # Probe with the amplified stop row: the greedy run must hit
            # the stop token inside the budget with real content before
            # it, and a short budget must stop on max-tokens.
            stored, _ = store(model, seed, AMPLIFY_STOP)
            embedding = model.prepare_input_embedding([TEXT], None, ref_mel)
            mx.eval(embedding)
            generated, latents, stopped = raw_loop(model, embedding, sampler_greedy, STOP_BUDGET)
            stop_ok = (
                stopped
                and 2 <= len(generated) <= STOP_BUDGET - 1
                and all(0 <= t < MEL_CODES and t not in (START_MEL,) for t in generated[: len(generated)])
            )
            short = raw_loop(model, embedding, sampler_greedy, MAXTOKENS_BUDGET)
            max_ok = not short[2]
            if stop_ok and max_ok:
                best = (model, stored, embedding, generated, latents, stopped, seed)
                print(f"backbone seed {seed}: acceptable greedy fixtures")
                break
        if best is None:
            raise SystemExit("no backbone seed produced acceptable greedy fixtures")
        model, stored, embedding, generated, latents, stopped, chosen_seed = best
        if chosen_seed != BACKBONE_SEED:
            print(f"NOTE: search picked seed {chosen_seed}; BACKBONE_SEED is {BACKBONE_SEED}")

        # Rerun the two pinned scenarios through the scenario recorder so
        # the stored file and the traces share one save path.
        stop_run = scenario(model, ref_mel, None, sampler_greedy, STOP_BUDGET, "stop_token")
        max_run = scenario(model, ref_mel, None, sampler_greedy, MAXTOKENS_BUDGET, "max_tokens")

        # --- forward pin -----------------------------------------------------
        text_ids = model.tokenizer.encode(normalized)
        conditioning = model.get_conditioning(ref_mel)
        mx.eval(conditioning)

        case_dir = OUT / "generate"
        case_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
        # The saved config keeps the placeholder tokenizer name: the
        # trained tokenizer lives in a random tempdir and the Rust port
        # never reads it (tokenization is caller-owned).
        saved_config = json.loads(json.dumps(CONFIG))
        saved_config["tokenizer_name"] = "fixture"
        (case_dir / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")
        traces = {
            "reference_commit": REFERENCE_COMMIT,
            "text": TEXT,
            "normalized": normalized,
            "text_ids": text_ids,
            "ref_audio": floats(ref_audio),
            "ref_mel": floats(ref_mel[0]),
            "ref_mel_rows": int(ref_mel.shape[1]),
            "conditioning": floats(conditioning[0]),
            "prompt_embedding": floats(embedding[0]),
            "prompt_length": int(embedding.shape[1]),
            "scenarios": [stop_run, max_run],
        }
        (case_dir / "traces.json").write_text(json.dumps(traces))

    # --- sampler case --------------------------------------------------------
    cases = []
    logits_rng = np.random.default_rng(541)
    for name, top_k in (("greedy", None), ("top_k", 5), ("top_k_ties", 4)):
        row = logits_rng.standard_normal(MEL_CODES).astype(np.float32)
        if name == "top_k_ties":
            row = np.array(
                [2.0, 2.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0],
                dtype=np.float32,
            )
        sampler = make_sampler(temp=0.0) if top_k is None else make_sampler(temp=0.0, top_k=top_k)
        record = {
            "name": name,
            "top_k": top_k or 0,
            "logits": floats(row),
            "greedy": int(np.argmax(row)),
        }
        if top_k is not None:
            masked = row.copy()
            kth = np.sort(masked)[-top_k]
            masked[masked < kth] = -np.inf
            record["top_k_masked"] = masked_floats(masked)
        cases.append(record)
    sampler_dir = OUT / "sampler"
    sampler_dir.mkdir(parents=True, exist_ok=True)
    (sampler_dir / "traces.json").write_text(json.dumps({"cases": cases}))

    # --- manifest ------------------------------------------------------------
    files = {}
    for entry in sorted(OUT.rglob("*")):
        if entry.is_file() and entry.name != "manifest.json":
            files[str(entry.relative_to(OUT))] = hashlib.sha256(entry.read_bytes()).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_indextts_fixtures.py",
        "backbone_seed": chosen_seed,
        "amplify_stop": AMPLIFY_STOP,
        "sample_rate": 24000,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


def from_dict_args(config):
    from mlx_audio.utils import from_dict

    return from_dict(ModelArgs, config)


if __name__ == "__main__":
    main()
