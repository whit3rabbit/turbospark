#!/usr/bin/env python3
"""Regenerate the llama (Orpheus-style) TTS family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/llama,
mlx_audio/lm/sample_utils.py, mlx 0.32.3). The reference module is
imported directly with its network SNAC singleton patched out, then the
tiny seeded codec is swapped in, so parse_output, codes_to_layers,
decode_audio_from_codes, and encode_audio_to_codes run the true
reference code. The generate loop runs the reference
`mlx_audio.lm.generate.generate_step` with the family's sampler and
logits processors (greedy: temperature 0).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_llama_fixtures.py

Outputs into crates/audio/testdata/tts_llama/:

- generate/model.safetensors + generate/config.json
    Seeded tiny llama backbone (2 layers, hidden 32, vocab 128307 so the
    real special-token ids fit). The embed rows of the start-of-speech,
    end-of-speech, and audio token ids are amplified so greedy argmax
    draws from the audio vocabulary; weights are stored as F16 (the
    traces are recorded after upcasting the same stored values, so the
    Rust f32 path sees identical inputs).
- generate/traces.json
    The pinned prompt, two greedy scenarios (stop at the end-of-speech
    token, stop at max_tokens), their generated ids, the parsed codes,
    and the final waveform through the tiny codec.
- decode/model.safetensors + decode/config.json
    Seeded tiny SNAC (noise-free, depthwise, codebook 64), the same
    geometry the codec family's tiny_a fixture uses.
- decode/traces.json
    Flat-code parse cases on fixed id sequences, the flat-code to layer
    split, the codes-to-waveform decode, the encode inverse on a seeded
    wave, and the zeroprompt id layout.
- sampling/traces.json
    The deterministic sampler math on fixed logit vectors: logits
    processors (logit bias, sign-aware repetition penalty), the
    log-softmax normalization, and the top-p / top-k filters. No RNG.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The
backbone seed search at the bottom picks the first seed whose greedy
runs satisfy the shape assertions, then the chosen seed is pinned
below; regeneration reproduces the same fixtures.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest.mock as mock
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.codec.models.snac.snac import SNAC
from mlx_audio.lm.generate import generate_step
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.llama import Model as LlamaModel, ModelArgs as LlamaArgs
from mlx_audio.lm.sample_utils import (
    apply_top_k,
    apply_top_p,
    make_logits_processors,
    make_sampler,
)

# Import the reference TTS module with the network codec disabled, then
# point its module-level snac_model at the tiny seeded instance.
with mock.patch(
    "mlx_audio.codec.models.snac.SNAC.from_pretrained",
    return_value=mock.MagicMock(),
):
    import mlx_audio.tts.models.llama.llama as llama_tts

OUT = Path("../turbospark/crates/audio/testdata/tts_llama")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# Special tokens from llama.py.
SOH = 128259
EOT = 128009
EOH = 128260
SOS = 128257
EOS_OF_SPEECH = 128258
SOA = 128261
EOA = 128262
AUDIO_OFFSET = 128266

# Tiny SNAC geometry: the codec family's noise-free tiny_a layout, so the
# decode path is deterministic (no NoiseBlock draws).
TINY_SNAC = dict(
    sampling_rate=44100,
    encoder_dim=8,
    encoder_rates=[2, 2, 2],
    latent_dim=None,
    decoder_dim=32,
    decoder_rates=[2, 2, 2],
    attn_window_size=None,
    codebook_size=64,
    codebook_dim=4,
    vq_strides=[4, 2, 1],
    noise=False,
    depthwise=True,
)

# Pinned "text" ids and the audio codes the fixtures use.
TEXT_IDS = [40, 314, 11, 271, 55]
NUM_AUDIO_CODES = 41  # ids 128266 .. 128306
VOCAB_SIZE = 128307
AMPLIFY_SOS = 25.0
AMPLIFY_EOS = 18.0
AMPLIFY_CODE = 16.0
# The greedy runs need a stronger repetition penalty than the family
# default (1.3) so the tiny model's argmax leaves single-token traps;
# the value is recorded per scenario and the Rust test passes the same
# value through GenerateParams.
FIXTURE_PENALTY = 5.0

# The prompt carries the start-of-speech token and 14 flat audio ids so
# the greedy runs have decodable codes; the family accepts such prompts
# through the zeroprompt path (generate_from_prompt).
PROMPT_CODES = 14

# The pinned backbone seed: first seed of the search below whose greedy
# runs satisfy the scenario shape assertions.
BACKBONE_SEED = 1


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def masked(arr) -> list:
    """Filtered log-prob vectors carry negative infinities, which Python's
    json module writes as invalid strict JSON; emit null sentinels."""
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(arr, dtype=np.float32).ravel().tolist()
    ]


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def tiny_backbone(seed: int, amplify_sos=AMPLIFY_SOS, amplify_eos=AMPLIFY_EOS,
                  amplify_code=AMPLIFY_CODE, vocab_size=None):
    """Seeded tiny llama with amplified audio-token rows, stored as F16
    while the module keeps the upcast F32 view for the traces."""
    args = LlamaArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=vocab_size or VOCAB_SIZE,
        head_dim=None,
        max_position_embeddings=256,
        num_key_value_heads=2,
        attention_bias=False,
        mlp_bias=False,
        rope_theta=10000.0,
        rope_traditional=False,
        rope_scaling=None,
        tie_word_embeddings=True,
        layer_types=None,
        sliding_window=None,
    )
    mx.random.seed(seed)
    model = LlamaModel(args)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith(("norm.weight", "q_norm.weight", "k_norm.weight")):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    embed = np.array(flat["model.embed_tokens.weight"], dtype=np.float32)
    embed[SOS] *= amplify_sos
    embed[EOS_OF_SPEECH] *= amplify_eos
    for c in range(NUM_AUDIO_CODES):
        embed[AUDIO_OFFSET + c] *= amplify_code
    stored = {k: v.astype(mx.float16) for k, v in flat.items()}
    stored["model.embed_tokens.weight"] = mx.array(embed).astype(mx.float16)
    model.update(tree_unflatten({k: v.astype(mx.float32) for k, v in stored.items()}))
    return model, stored, args


def run_generate(model, prompt_ids, max_tokens, penalty=FIXTURE_PENALTY):
    """The TTS family's segment loop: reference generate_step with the
    family sampler (greedy here) and logits processors, stopping after
    the end-of-speech token."""
    sampler = make_sampler(0.0, 0.8, top_k=-1)
    processors = make_logits_processors(None, penalty, 20)
    generated = []
    for token, _ in generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=processors,
    ):
        value = int(token)
        generated.append(value)
        if value == EOS_OF_SPEECH:
            break
    return generated


def scenario(model, prompt_ids, max_tokens, name):
    generated = run_generate(model, prompt_ids, max_tokens)
    all_ids = list(prompt_ids) + generated
    code_lists = llama_tts.Model.parse_output(None, mx.array([all_ids]))
    codes = ints(code_lists[0])
    stopped = "end_of_speech" if EOS_OF_SPEECH in generated else "max_tokens"
    return {
        "name": name,
        "max_tokens": max_tokens,
        "repetition_penalty": FIXTURE_PENALTY,
        "stopped_by": stopped,
        "generated": generated,
        "codes": codes,
    }


def build_generate_case(snac):
    llama_tts.snac_model = snac
    best = None
    for seed in range(64):
        model, stored, args = tiny_backbone(seed)
        prompt = (
            [SOH]
            + TEXT_IDS
            + [EOT, EOH]
            + [SOS]
            + [AUDIO_OFFSET + ((3 * i + 1) % 64) for i in range(PROMPT_CODES)]
        )
        eos_run = scenario(model, prompt, 48, "eos")
        length_run = scenario(model, prompt, 10, "length")
        codes_ok = (
            eos_run["stopped_by"] == "end_of_speech"
            and len(eos_run["codes"]) >= 14
            and min(eos_run["codes"]) >= 0
        )
        length_ok = (
            length_run["stopped_by"] == "max_tokens"
            and len(length_run["codes"]) >= 7
            and min(length_run["codes"]) >= 0
        )
        if codes_ok and length_ok:
            best = (model, stored, args, prompt, eos_run, length_run, seed)
            print(f"backbone seed {seed}: acceptable greedy fixtures")
            break
    if best is None:
        raise SystemExit("no backbone seed produced acceptable greedy fixtures")

    model, stored, args, prompt, eos_run, length_run, seed = best
    if seed != BACKBONE_SEED:
        print(f"NOTE: search picked seed {seed}; BACKBONE_SEED is {BACKBONE_SEED}")
    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    config = dict(vars(args))
    config["tokenizer_name"] = "mlx-community/orpheus-3b-0.1-ft-bf16"
    config["sample_rate"] = 24000
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    traces = {
        "prompt": prompt,
        "scenarios": [eos_run, length_run],
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))
    build_generate_eos_case(snac)


def build_generate_eos_case(snac):
    """A second seeded backbone whose very first greedy draw is the
    end-of-speech token. The loop then appends EOS, stops, and parses the
    prompt's own codes, which are constructed slot-valid (L1, L2, L3
    ranges), so the final waveform decodes exactly instead of through the
    reference's silent MLX negative-index wraparound."""
    llama_tts.snac_model = snac
    prompt_codes = []
    for group in range(2):
        for slot in range(7):
            prompt_codes.append(slot * 4096 + ((3 * group + slot) % 64))
    prompt = (
        [SOH]
        + TEXT_IDS
        + [EOT, EOH]
        + [SOS]
        + [AUDIO_OFFSET + code for code in prompt_codes]
    )
    best = None
    for seed in range(64):
        model, stored, args = tiny_backbone(
            seed,
            amplify_sos=16.0,
            amplify_eos=55.0,
            amplify_code=20.0,
            # The prompt carries slot-valid flat codes up to 6*4096+63, so
            # the vocabulary must cover their token ids.
            vocab_size=AUDIO_OFFSET + 6 * 4096 + 64,
        )
        generated = run_generate(model, prompt, 8)
        if generated == [EOS_OF_SPEECH]:
            best = (model, stored, args, seed)
            print(f"eos-first backbone seed {seed}: acceptable")
            break
    if best is None:
        raise SystemExit("no eos-first backbone seed produced an acceptable fixture")

    model, stored, args, seed = best
    case_dir = OUT / "generate_eos"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    config = dict(vars(args))
    config["tokenizer_name"] = "mlx-community/orpheus-3b-0.1-ft-bf16"
    config["sample_rate"] = 24000
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    waveform = llama_tts.decode_audio_from_codes(prompt_codes)
    traces = {
        "prompt": prompt,
        "repetition_penalty": FIXTURE_PENALTY,
        "max_tokens": 8,
        "generated": [EOS_OF_SPEECH],
        "codes": prompt_codes,
        "waveform": floats(waveform),
        "eos_first_seed": seed,
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))


def build_decode_case(snac):
    llama_tts.snac_model = snac
    traces = {}

    # Parse cases: fixed id sequences through the reference parse_output.
    prefix = [SOH] + TEXT_IDS + [EOT, EOH]
    codes_a = [AUDIO_OFFSET + ((3 * i + 1) % 64) for i in range(14)]
    cases = [
        (
            "after_last_sos",
            prefix + [SOS] + codes_a + [EOS_OF_SPEECH],
        ),
        (
            "second_sos_wins",
            [SOS] + codes_a[:7] + [EOS_OF_SPEECH, SOS] + codes_a[7:],
        ),
        (
            "no_sos_trims_below_one_group",
            [5, 9, EOS_OF_SPEECH, 12],
        ),
        (
            "no_sos_negative_codes_survive",
            [5, 9, 12, 14, 17, 3, 8, 21],
        ),
        (
            "tail_trimmed_to_groups",
            prefix + [SOS] + codes_a + codes_a[:3],
        ),
        (
            "end_of_speech_removed_between_codes",
            prefix + [SOS] + codes_a[:7] + [EOS_OF_SPEECH] + codes_a[7:14],
        ),
        ("empty_after_last_sos", prefix + codes_a[:7] + [SOS]),
    ]
    records = []
    for name, ids in cases:
        parsed = llama_tts.Model.parse_output(None, mx.array([ids]))
        records.append(
            {"name": name, "input_ids": ints(ids), "codes": ints(parsed[0])}
        )
    traces["parse_cases"] = records

    # Flat codes to SNAC levels and the codes-to-waveform decode.
    flat = []
    for g in range(4):
        flat.append((5 * g + 3) % 64)
        flat.append(4096 + ((5 * g + 4) % 64))
        flat.append(2 * 4096 + ((5 * g + 5) % 64))
        flat.append(3 * 4096 + ((5 * g + 6) % 64))
        flat.append(4 * 4096 + ((5 * g + 0) % 64))
        flat.append(5 * 4096 + ((5 * g + 1) % 64))
        flat.append(6 * 4096 + ((5 * g + 2) % 64))
    layers = llama_tts.codes_to_layers(flat)
    traces["flat_codes"] = ints(flat)
    traces["layers"] = [ints(layer) for layer in layers]
    audio = llama_tts.decode_audio_from_codes(flat)
    traces["waveform"] = floats(audio)

    # Encode inverse on a deterministic wave (no codec randomness).
    wave = np.random.default_rng(3).standard_normal(256).astype(np.float32) * 0.1
    encoded = llama_tts.encode_audio_to_codes(mx.array(wave))
    traces["encode_wave"] = floats(wave)
    traces["encode_flat_codes"] = ints(encoded)

    # Zeroprompt layout with a pinned transcript.
    transcript = [40, 314, 11]
    audio_ids = mx.array(encoded).reshape(1, -1) + AUDIO_OFFSET
    zeroprompt = mx.concatenate(
        [
            mx.array([[SOH]]),
            mx.array([transcript]),
            mx.array([[EOT, EOH]]),
            mx.array([[SOA, SOS]]),
            audio_ids,
            mx.array([[EOS_OF_SPEECH, EOA]]),
        ],
        axis=1,
    )
    traces["zeroprompt_transcript_ids"] = transcript
    traces["zeroprompt"] = ints(zeroprompt)

    case_dir = OUT / "decode"
    case_dir.mkdir(parents=True, exist_ok=True)
    dump_snac(snac, case_dir)
    (case_dir / "traces.json").write_text(json.dumps(traces))


def dump_snac(snac, case_dir):
    flat = tree_flatten(snac.parameters())
    mx.save_safetensors(str(case_dir / "model.safetensors"), dict(flat))
    (case_dir / "config.json").write_text(json.dumps(TINY_SNAC, indent=2) + "\n")


def build_sampling_case():
    cases = []

    def vector(n, offset):
        values = np.array(
            [float(np.sin(0.73 * (i + offset)) * 4.0 + np.cos(0.31 * i) * 1.5)
             for i in range(n)],
            dtype=np.float32,
        )
        assert len(set(values.tolist())) == n, "fixed logits must have no ties"
        return values

    # Full chain: logit bias + repetition penalty, log-softmax, top-p,
    # top-k. Mirrors the family step order exactly.
    logits = vector(64, 0)
    context = [5, 3, 5, 42, 7]
    bias = {2: 0.5, 9: -1.25}
    processors = make_logits_processors(bias, 1.3, 20)
    processed = mx.array(logits)[None, :]
    for processor in processors:
        processed = processor(mx.array(context), processed)
    processed = processed.squeeze(0)
    logprobs = processed - mx.logsumexp(processed, keepdims=True)
    top_p_out = apply_top_p(logprobs, 0.8)
    top_k_out = apply_top_k(top_p_out, 5)
    cases.append(
        {
            "name": "chain",
            "logits": floats(logits),
            "context": context,
            "logit_bias": [[2, 0.5], [9, -1.25]],
            "repetition_penalty": 1.3,
            "repetition_context_size": 20,
            "processed_logits": floats(processed),
            "logprobs": floats(logprobs),
            "top_p": 0.8,
            "top_p_logprobs": masked(top_p_out),
            "top_k": 5,
            "top_k_logprobs": masked(top_k_out),
            "argmax": int(mx.argmax(top_k_out, axis=-1).item()),
        }
    )

    # Repetition penalty alone, with a context longer than the window so
    # the truncation is exercised, plus duplicate tokens in the window
    # (the reference gather/scatter applies the penalty once per unique
    # token).
    logits = vector(48, 7)
    context = [11] * 3 + [4] * 2 + list(range(12, 40))
    processor = make_logits_processors(None, 1.3, 20)[0]
    processed = processor(mx.array(context), mx.array(logits)[None, :]).squeeze(0)
    cases.append(
        {
            "name": "penalty_only",
            "logits": floats(logits),
            "context": context,
            "logit_bias": [],
            "repetition_penalty": 1.3,
            "repetition_context_size": 20,
            "processed_logits": floats(processed),
            "logprobs": floats(processed - mx.logsumexp(processed, keepdims=True)),
            "top_p": 0.0,
            "top_p_logprobs": None,
            "top_k": 0,
            "top_k_logprobs": None,
            "argmax": int(mx.argmax(processed, axis=-1).item()),
        }
    )

    # Wide top-p: keeps nearly everything except the flattened tail.
    logits = vector(96, 13)
    logprobs = mx.array(logits) - mx.logsumexp(mx.array(logits), keepdims=True)
    top_p_out = apply_top_p(logprobs, 0.5)
    cases.append(
        {
            "name": "top_p_only",
            "logits": floats(logits),
            "context": [],
            "logit_bias": [],
            "repetition_penalty": 0.0,
            "repetition_context_size": 20,
            "processed_logits": floats(logits),
            "logprobs": floats(logprobs),
            "top_p": 0.5,
            "top_p_logprobs": masked(top_p_out),
            "top_k": 0,
            "top_k_logprobs": None,
            "argmax": int(mx.argmax(top_p_out, axis=-1).item()),
        }
    )

    # Top-k alone on a negative-leaning vector.
    logits = vector(64, 29) - 6.0
    logprobs = mx.array(logits) - mx.logsumexp(mx.array(logits), keepdims=True)
    top_k_out = apply_top_k(logprobs, 3)
    cases.append(
        {
            "name": "top_k_only",
            "logits": floats(logits),
            "context": [],
            "logit_bias": [],
            "repetition_penalty": 0.0,
            "repetition_context_size": 20,
            "processed_logits": floats(logits),
            "logprobs": floats(logprobs),
            "top_p": 0.0,
            "top_p_logprobs": None,
            "top_k": 3,
            "top_k_logprobs": masked(top_k_out),
            "argmax": int(mx.argmax(top_k_out, axis=-1).item()),
        }
    )

    case_dir = OUT / "sampling"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps({"cases": cases}))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mx.random.seed(42)
    snac = SNAC(**TINY_SNAC)
    build_generate_case(snac)
    build_decode_case(snac)
    build_sampling_case()

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_llama_fixtures.py",
        "backbone_seed": BACKBONE_SEED,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
