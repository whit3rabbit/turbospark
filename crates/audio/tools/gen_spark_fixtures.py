#!/usr/bin/env python3
"""Regenerate the Spark TTS family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/spark,
mlx 0.32.3). The reference Model builds at tiny seeded dims (2-layer
qwen2 backbone) and the generate loop runs through the same building
blocks the reference stream_generate drives: prefill of every prompt
token but the last, the make_logits_processors chain (logit bias,
sign-aware repetition penalty over the accumulated token context), the
log-softmax, and the make_sampler chain (greedy argmax at temperature
0, else top-p then top-k then the categorical draw).

The bundled fast tokenizer is replaced by a deterministic mock that
maps every special marker (task tokens, content/style blocks,
bicodec_semantic_i / bicodec_global_i, and the semantic-end token) to
a fixed id; the port consumes the recorded id sequences and reverse
code tables. Sampled draws are not pinned: the reference draws through
mx.random.state while the Rust port draws through the crate-local
threefry RNG, so only greedy runs pin the loop end to end and the
filtered distributions are pinned on fixed logits.

Run from inside the mlx-audio checkout root so `mlx_audio` imports
from the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_spark_fixtures.py

Outputs into crates/audio/testdata/tts_spark/:

- generate/config.json + model.safetensors
    The seeded tiny Spark Model (2-layer qwen2 at hidden 32 over a
    512-entry vocabulary with the tied head the reference defaults to)
    plus the audio token maps the mock tokenizer realizes. Weights are
    stored F16; the traces record the upcast F16 values the Rust f32
    path reads back.
- generate/traces.json
    Greedy scenarios (semantic-end stop, eos stop, max-tokens stop)
    with the prompt ids, sampled tokens, and the extracted
    semantic/global codes.
- math/traces.json
    The logits-processor and sampler chain on fixed logits: logit
    bias, the repetition penalty over the truncated context, the
    log-softmax, and the top-p/top-k filtered distributions.
- prompts.json
    The process_prompt / process_prompt_control strings.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The
backbone seed at the bottom is pinned to the first seed of the search
whose greedy runs satisfy the scenario shapes.
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

import mlx_audio.tts.models.spark.spark as spark
from mlx_audio.tts.models.spark.spark import ModelConfig
from mlx_audio.tts.models.spark.utils.token_parser import (
    GENDER_MAP,
    LEVELS_MAP,
    TASK_TOKEN_MAP,
)

OUT = Path("../turbospark/crates/audio/testdata/tts_spark")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

VOCAB = 512
BACKBONE_SEED = 14
SEMANTIC_CODES = 8
GLOBAL_CODES = 4
EOS_BUDGET = 24
MAX_TOKENS_BUDGET = 6

SEMANTIC_END = 300
EOS_ID = 511
SEMANTIC_BASE = 100
GLOBAL_BASE = 40

CONFIG = {
    "model_type": "spark",
    "sample_rate": 16000,
    "bos_token_id": 151643,
    "eos_token_id": EOS_ID,
    "hidden_size": 32,
    "intermediate_size": 64,
    "max_position_embeddings": 256,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "vocab_size": VOCAB,
}

SPECIAL = {
    TASK_TOKEN_MAP["tts"]: 1,
    TASK_TOKEN_MAP["controllable_tts"]: 2,
    "<|start_content|>": 3,
    "<|end_content|>": 4,
    "<|start_global_token|>": 5,
    "<|end_global_token|>": 6,
    "<|start_semantic_token|>": 7,
    "<|start_style_label|>": 8,
    "<|end_style_label|>": 9,
    f"<|gender_{GENDER_MAP['male']}|>": 10,
    f"<|gender_{GENDER_MAP['female']}|>": 11,
}
for level, label in LEVELS_MAP.items():
    SPECIAL[f"<|pitch_label_{label}|>"] = 20 + label
    SPECIAL[f"<|speed_label_{label}|>"] = 30 + label


class MockTokenizer:
    """Deterministic stand-in for the bundled fast tokenizer: special
    markers map to fixed ids, `bicodec_semantic_i` to 100 + i,
    `bicodec_global_i` to 40 + i, the semantic-end token to 300, and
    every other chunk to a rolling hash. The Rust side consumes the
    recorded id sequences and the reverse code tables."""

    def __call__(self, text, add_special_tokens=True):
        import re

        pattern = (
            r"(<\|task_[a-z_]+\|>|<\|start_[a-z_]+\|>|<\|end_[a-z_]+\|>"
            r"|<\|gender_\d+\|>|<\|pitch_label_\d+\|>|<\|speed_label_\d+\|>"
            r"|<\|bicodec_semantic_\d+\|>|<\|bicodec_global_\d+\|>)"
        )
        ids = []
        for raw in re.split(pattern, text):
            if not raw:
                continue
            if raw in SPECIAL:
                ids.append(SPECIAL[raw])
                continue
            match = re.fullmatch(r"<\|bicodec_semantic_(\d+)\|>", raw)
            if match:
                ids.append(SEMANTIC_BASE + int(match.group(1)))
                continue
            match = re.fullmatch(r"<\|bicodec_global_(\d+)\|>", raw)
            if match:
                ids.append(GLOBAL_BASE + int(match.group(1)))
                continue
            if raw == "<|end_semantic_token|>":
                ids.append(SEMANTIC_END)
                continue
            ids.append(12 + (sum(ord(c) * (i + 1) for i, c in enumerate(raw)) % 400))
        return {"input_ids": ids}


def build_maps():
    semantic = [SEMANTIC_BASE + i for i in range(SEMANTIC_CODES)]
    global_ids = [GLOBAL_BASE + i for i in range(GLOBAL_CODES)]
    return {
        "semantic": semantic,
        "global": global_ids,
        "semantic_codes": SEMANTIC_CODES,
        "global_codes": GLOBAL_CODES,
    }


def build_model(seed: int, semantic_end_bias: float = 1.0):
    config = ModelConfig.from_dict(CONFIG)
    mx.random.seed(seed)
    model = spark.Model(config)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith("norm.weight"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    stored = {name: value.astype(mx.float16) for name, value in flat.items()}
    # Load the F16-rounded values upcast to F32: the stored file keeps
    # the F16 bytes while the live model computes on the values the
    # Rust f32 path reads back.
    model.load_weights(
        [(k, v.astype(mx.float32)) for k, v in stored.items()], strict=True
    )
    mx.eval(model)
    return model, stored


def run_greedy(model, prompt_ids, max_tokens):
    """Greedy through the reference stream_generate building blocks."""
    from mlx_audio.lm.models.cache import make_prompt_cache
    from mlx_audio.lm.sample_utils import make_repetition_penalty

    tokenizer = MockTokenizer()
    eos_ids = {CONFIG["eos_token_id"]}
    cache = make_prompt_cache(model.model)

    def step(input_tokens, context):
        logits = model.model(input_tokens[None], cache=cache)[:, -1, :]
        # The make_logits_processors chain with the reference defaults:
        # no logit bias, repetition penalty 1.3 over 20 context tokens.
        penalty = make_repetition_penalty(1.3, context_size=20)
        logits = penalty(mx.array(context), logits)
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        return logprobs.squeeze(0)

    # Prefill every prompt token but the last (single chunk at these
    # lengths), exactly like generate_step.
    total = len(prompt_ids)
    processed = 0
    while total - processed > 1:
        count = min(2048, total - processed - 1)
        model.model(mx.array(prompt_ids[processed : processed + count])[None], cache=cache)
        processed += count
    context = [prompt_ids[-1]]
    logprobs = step(mx.array([prompt_ids[-1]]), context)
    tokens = []
    stopped = "max_tokens"
    for _ in range(max_tokens):
        token = int(mx.argmax(logprobs).item())
        if token in eos_ids:
            stopped = "end_of_text"
            break
        context.append(token)
        tokens.append(token)
        if token == SEMANTIC_END:
            stopped = "semantic_end"
            break
        logprobs = step(mx.array([token]), context)
    return tokens, stopped


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- backbone seed search --------------------------------------------
    control_prompt = (
        TASK_TOKEN_MAP["controllable_tts"]
        + "<|start_content|>Hello there.<|end_content|>"
        + "<|start_style_label|>"
        + f"<|gender_{GENDER_MAP['male']}|><|pitch_label_{LEVELS_MAP['moderate']}|>"
        + f"<|speed_label_{LEVELS_MAP['high']}|><|end_style_label|>"
    )
    control_ids = MockTokenizer()(control_prompt)["input_ids"]

    best = None
    for seed in range(1, 200):
        model, stored = build_model(seed)
        tokens, stopped = run_greedy(model, control_ids, EOS_BUDGET)
        length_tokens, _ = run_greedy(model, control_ids, MAX_TOKENS_BUDGET)
        ok = (
            stopped == "semantic_end"
            and 3 <= len(tokens) <= 16
            and len(length_tokens) == MAX_TOKENS_BUDGET
        )
        if ok:
            best = (model, stored, tokens, seed)
            print(f"backbone seed {seed}: acceptable greedy fixtures")
            break
    if best is None:
        raise SystemExit("no backbone seed produced acceptable greedy fixtures")
    model, stored, tokens, seed = best
    if seed != BACKBONE_SEED:
        print(f"NOTE: search picked seed {seed}; BACKBONE_SEED is {BACKBONE_SEED}")

    # --- eos-first model ---------------------------------------------------
    eos_stored = None
    for bias in (3.0, 2.0, 1.5):
        candidate, candidate_stored = build_model(seed)
        head = np.array(candidate_stored["model.embed_tokens.weight"], dtype=np.float32)
        # Flip the eos row sign and crush every other row, so the
        # first sampled token is the eos stop (the stream breaks before
        # yielding it).
        head[EOS_ID, :] *= -bias
        for row in range(head.shape[0]):
            if row != EOS_ID:
                head[row, :] *= 1.0 / bias
        candidate_stored = dict(candidate_stored)
        candidate_stored["model.embed_tokens.weight"] = mx.array(head).astype(mx.float16)
        candidate.load_weights(
            [(k, v.astype(mx.float32)) for k, v in candidate_stored.items()],
            strict=True,
        )
        mx.eval(candidate)
        eos_tokens, eos_stopped = run_greedy(candidate, control_ids, 8)
        if eos_stopped == "end_of_text" and not eos_tokens:
            eos_stored = candidate_stored
            print(f"eos-first bias {bias}: acceptable")
            break
    if eos_stored is None:
        raise SystemExit("no eos bias produced an acceptable fixture")

    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)
    config_out = dict(CONFIG)
    config_out["audio_token_maps"] = build_maps()
    config_out["reference_commit"] = REFERENCE_COMMIT
    (case_dir / "config.json").write_text(json.dumps(config_out, indent=2) + "\n")

    eos_dir = OUT / "generate_eos"
    eos_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(eos_dir / "model.safetensors"), eos_stored)
    (eos_dir / "config.json").write_text(json.dumps(config_out, indent=2) + "\n")

    maps = build_maps()

    def codes_from(tokens, base, count):
        wanted = {base + i for i in range(count)}
        return [t - base for t in tokens if t in wanted]

    length_tokens, _ = run_greedy(model, control_ids, MAX_TOKENS_BUDGET)
    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "backbone_seed": seed,
        "semantic_end_token_id": SEMANTIC_END,
        "scenarios": [
            {
                "name": "semantic_end",
                "case": "generate",
                "prompt_ids": control_ids,
                "params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repetition_penalty": 1.3,
                    "repetition_context_size": 20,
                    "max_tokens": EOS_BUDGET,
                    "stop_token_ids": [],
                    "semantic_end_token_id": SEMANTIC_END,
                },
                "tokens": tokens,
                "semantic_codes": codes_from(tokens, SEMANTIC_BASE, SEMANTIC_CODES),
                "global_codes": codes_from(tokens, GLOBAL_BASE, GLOBAL_CODES),
                "stopped_by": "semantic_end",
            },
            {
                "name": "max_tokens",
                "case": "generate",
                "prompt_ids": control_ids,
                "params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repetition_penalty": 1.3,
                    "repetition_context_size": 20,
                    "max_tokens": MAX_TOKENS_BUDGET,
                    "stop_token_ids": [],
                    "semantic_end_token_id": SEMANTIC_END,
                },
                "tokens": length_tokens,
                "semantic_codes": codes_from(
                    length_tokens, SEMANTIC_BASE, SEMANTIC_CODES
                ),
                "global_codes": codes_from(length_tokens, GLOBAL_BASE, GLOBAL_CODES),
                "stopped_by": "max_tokens",
            },
            {
                "name": "end_of_text",
                "case": "generate_eos",
                "prompt_ids": control_ids,
                "params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repetition_penalty": 1.3,
                    "repetition_context_size": 20,
                    "max_tokens": 8,
                    "stop_token_ids": [],
                    "semantic_end_token_id": SEMANTIC_END,
                },
                "tokens": [],
                "semantic_codes": [],
                "global_codes": [],
                "stopped_by": "end_of_text",
            },
        ],
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))

    # --- sampler math on fixed logits -------------------------------------
    def fixed_logits(n):
        values = np.array(
            [
                float(np.sin(0.71 * (i + 3)) * 4.0 + np.cos(0.37 * i) * 1.5)
                for i in range(n)
            ],
            dtype=np.float32,
        )
        assert len(set(values.tolist())) == n
        return values

    from mlx_audio.lm.sample_utils import (
        apply_top_k,
        apply_top_p,
        make_repetition_penalty,
    )

    math_cases = []
    for name, top_p, top_k in (
        ("default_filters", 0.95, 50),
        ("no_top_k", 0.95, 0),
        ("no_filters", 1.0, 0),
        ("tight_top_p", 0.4, 0),
    ):
        logits = fixed_logits(VOCAB)
        context = [12, 7, 7, 403, 5]
        penalized = make_repetition_penalty(1.3, context_size=20)(
            mx.array(context), mx.array(logits)[None, :]
        )
        logprobs = penalized - mx.logsumexp(penalized, keepdims=True)
        filtered = logprobs.squeeze(0)
        if top_p > 0.0 and top_p < 1.0:
            filtered = apply_top_p(filtered, top_p)
        if top_k > 0:
            filtered = apply_top_k(filtered, top_k)
        masked = [
            float(v) if np.isfinite(v) else None
            for v in np.asarray(filtered, dtype=np.float32).ravel().tolist()
        ]
        math_cases.append(
            {
                "name": name,
                "logits": [float(v) for v in logits.tolist()],
                "context": context,
                "repetition_penalty": 1.3,
                "repetition_context_size": 20,
                "top_p": top_p,
                "top_k": top_k,
                "filtered": masked,
                "argmax": int(mx.argmax(filtered).item()),
            }
        )

    # Logit-bias processor on fixed logits.
    logits = fixed_logits(VOCAB)
    pairs = [(9, 2.5), (300, -3.0), (511, 8.0)]
    biased = mx.array(logits)
    for token, value in pairs:
        biased = biased.at[token].add(value)
    biased = biased - mx.logsumexp(biased, keepdims=True)
    math_cases.append(
        {
            "name": "logit_bias",
            "logits": [float(v) for v in logits.tolist()],
            "logit_bias": [[t, v] for t, v in pairs],
            "context": [],
            "repetition_penalty": 1.0,
            "repetition_context_size": 20,
            "top_p": 1.0,
            "top_k": 0,
            "filtered": [
                float(v) for v in np.asarray(biased, dtype=np.float32).ravel().tolist()
            ],
            "argmax": int(mx.argmax(biased).item()),
        }
    )
    (OUT / "math").mkdir(parents=True, exist_ok=True)
    (OUT / "math" / "traces.json").write_text(
        json.dumps({"reference_commit": REFERENCE_COMMIT, "cases": math_cases})
    )

    # --- prompt strings -----------------------------------------------------
    # The reference process_prompt needs the live audio tokenizer
    # (wav2vec2 encode), so the clone strings are the exact list
    # concatenations the reference builds around the token strings.
    prompts = {
        "clone": {
            "string": (
                TASK_TOKEN_MAP["tts"]
                + "<|start_content|>Ref voice.Hello there.<|end_content|>"
                + "<|start_global_token|><|bicodec_global_5|><|bicodec_global_9|><|end_global_token|>"
                + "<|start_semantic_token|><|bicodec_semantic_3|><|bicodec_semantic_1|><|bicodec_semantic_4|>"
            )
        },
        "clone_no_text": {
            "string": (
                TASK_TOKEN_MAP["tts"]
                + "<|start_content|>Hello there.<|end_content|>"
                + "<|start_global_token|><|bicodec_global_7|><|end_global_token|>"
            )
        },
        "control": {"string": control_prompt},
    }
    (OUT / "prompts.json").write_text(json.dumps(prompts, indent=2))

    # --- manifest ---------------------------------------------------------------
    files = {}
    for entry in sorted(OUT.rglob("*")):
        if entry.is_file() and entry.name != "manifest.json":
            files[str(entry.relative_to(OUT))] = hashlib.sha256(entry.read_bytes()).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_spark_fixtures.py",
        "backbone_seed": seed,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
