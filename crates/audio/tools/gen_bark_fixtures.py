#!/usr/bin/env python3
"""Regenerate the bark TTS family Rust fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/bark; the recording interpreter's mlx version is
written to manifest.json). The reference Pipeline runs with
`use_kv_caching=True` over the true semantic, coarse, and fine GPT
stacks at tiny seeded dims; the BERT tokenizer is a stub, the EnCodec
singleton is the tiny seeded codec reloaded from
testdata/encodec/tiny_weights.safetensors, and the semantic and coarse
`mx.random.categorical` draws (plus the fine stage's sampled path at
temperature 0.7) are replaced with a deterministic inverse-CDF stand-in:
one recorded uniform per sampled row, mirrored by the Rust replay. Each
uniform is resampled until its decision is boundary-safe - the distance
from the uniform to both adjacent distinct cdf boundaries must stay over
MARGIN_GATE times the boundary value - so the Rust f32 port cannot flip
a choice through accumulation noise (relative logit drift of order
1e-5..1e-4 moves a boundary by that fraction of its value; a uniform
that sat 6e-5 from a boundary of 0.147 once flipped). The gate is
relative because wide slices with near-uniform probability have cdf
steps far below any absolute margin.

The reference `Pipeline.__call__` passes `use_kv_caching` positionally
into `generate_coarse`'s `max_coarse_history` slot, so the recorded
coarse stage runs with `max_coarse_history = True` (= 1), the default
sliding window of 60, and NO kv cache: every step re-forwards the full
growing window. That wiring is part of the pinned behavior the fixtures
freeze.

History note: an earlier recording produced a self-inconsistent
scenario a because `capture_stages` wrapped the pipeline methods
cumulatively - the second scenario's run passed through the first
scenario's wrapper and overwrote the first scenario's captured dict in
place (its prompt fields stayed from scenario a while its semantic and
coarse outputs, encoded text, and uniforms came from scenario b, which
also made the first scenario's own 445-token run look lost). The
wrappers now restore the original methods after each run, and the
recording verifies rerun equality and prompt-to-encoding consistency
before writing.

Root cause of the inconsistent ids, confirmed by spying on
`pipe.tokenizer.encode` during a rebuilt run: the reference pipeline
encodes the raw scenario text in exactly ONE encode call with no
normalization or sentence splitting, so [310, 17, 3] is the only
possible encoding of "hello there" under the stub tokenizer and
[153, 17, 3] (the ids the broken recording carried for scenario a) is
`stub_encode("the second one")` - scenario b's text. Nothing about the
tokenizer was wrong; the capture chaining alone mixed the scenarios.
`run_scenario` therefore now derives `text_token_ids` from the
pipeline's OWN captured encoding of the same run (strip the text pads
and the encoding offset) and builds the prefill probe from those same
derived ids, so every field of a scenario describes one identical run
by construction. A tokenizer spy inside the recording asserts the
pipeline consumed the raw scenario text, and `matches_prompt` keeps
cross-checking the captured encoding against an independent
`stub_encode` call.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter. PYTHONHASHSEED is
pinned: the seeded weight fill walks the parameter tree in an
order that Python hash randomization perturbs across processes, and the
fixture must record one reproducible run.

    cd ../mlx-audio
    PYTHONHASHSEED=0 .venv/bin/python \
        ../turbospark/crates/audio/tools/gen_bark_fixtures.py

Outputs into crates/audio/testdata/tts_bark/:

- generate/model.safetensors + generate/config.json
    Seeded tiny bark (2 layers per stack, n_embd 16, 2 heads; semantic
    input vocab 129600 so the real special-token ids fit, coarse input
    12096, fine input 1056 with output vocab 64 so every fine code is
    decodable by the tiny codec).
- generate/traces.json
    The stub text token ids and, per scenario: the semantic tokens, the
    coarse codes, the fine codes, the encoded text ids, the boundary
    margin stats, and the decoded waveform.
- manifest.json
    File list with SHA-256 digests and generator metadata.

Phase 1 searches model seeds and per-scenario streams for a run that is
structurally acceptable and reproduces itself in-process; phase 2
re-records the winner in a fresh interpreter (long-running search
processes do not reproduce; MLX kernel state accumulates) and verifies
rerun equality there before writing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest.mock as mock
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

OUT = Path("../turbospark/crates/audio/testdata/tts_bark")
ENCODEC_TESTDATA = Path("../turbospark/crates/audio/testdata/encodec")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# Minimum distance from a sampled uniform to both adjacent distinct cdf
# boundaries, as a fraction of the boundary value. The corrected
# uncached coarse path puts the MLX-vs-Rust difference back at f32 noise
# (about 2e-6 on the logits), which moves a boundary by under ~1e-5 of
# its value; 3e-4 leaves two orders of magnitude of headroom. Larger
# gates are unsatisfiable in dense cdf regions, where the local step is
# a small fraction of the boundary value and no uniform is safe.
MARGIN_GATE = 3e-4
MARGIN_RETRIES = 64

TEXTS = ["hello there", "the second one"]


def stub_encode(text: str) -> list[int]:
    total = sum(ord(ch) for ch in text)
    return [10 + total % 400, 17, 3]


class StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return stub_encode(text)


def bark_encodec():
    """A bark-local tiny EnCodec: the shared tiny geometry with a
    1024-entry codebook so the coarse codes riding fine books 0 and 1
    stay decodable (the reference decodes them through silent MLX
    wraparound on the shared 64-entry tiny codec). Seeded weights."""
    from mlx.utils import tree_flatten

    from mlx_audio.codec.models.encodec.encodec import Encodec, EncodecConfig

    config_value = json.loads((ENCODEC_TESTDATA / "tiny_config.json").read_text())
    config_value["codebook_size"] = 1024
    config = EncodecConfig(**config_value)
    mx.random.seed(700)
    codec = Encodec(config)

    def fill(weight):
        return mx.random.uniform(low=-0.3, high=0.3, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, codec.parameters())))
    codec.load_weights(list(flat.items()), strict=False)
    mx.eval(codec)
    return codec, flat, config_value


CODEC_DIR = OUT / "codec"


def replay_margin(cdf: np.ndarray, u: float, drawn: int) -> float:
    """Distance from the uniform to the nearest cdf boundary the replay
    decision can flip on, relative to that boundary's value. `drawn` is
    the first index with cdf > u (searchsorted side='right'), so the
    lower boundary is cdf[drawn-1]; the upper boundary is the next
    DISTINCT value, because a drift that pushes a flat run's value below
    the uniform jumps the draw past the whole run, not one cell. The
    relative form keeps the gate satisfiable on wide slices whose cdf
    steps are far below any absolute margin."""
    worst = float("inf")
    if drawn > 0:
        left = float(cdf[drawn - 1])
        if left > 0.0:
            worst = min(worst, (u - left) / left)
    value = float(cdf[drawn])
    j = drawn + 1
    while j < len(cdf) and float(cdf[j]) <= value:
        j += 1
    if j < len(cdf):
        above = float(cdf[j])
        worst = min(worst, (above - u) / above)
    else:
        worst = min(worst, 1.0 - u)
    return worst


def inverse_cdf_sampler(stream_seed):
    """Replace mx.random.categorical with a deterministic inverse-CDF
    draw over the softmax of the scaled slice: one recorded uniform per
    row, resampled while the decision sits within MARGIN_GATE of a cdf
    boundary. Returns (uniforms, stats, patch)."""
    us: list[float] = []
    stats = {"count": 0, "min_margin": float("inf"), "resamples": 0}
    rng = np.random.default_rng(stream_seed)

    def wrapper(logits, axis=None, shape=None, num_samples=None):
        values = np.asarray(logits, dtype=np.float64)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        weights = np.exp(values - values.max(axis=-1, keepdims=True))
        probs = weights / weights.sum(axis=-1, keepdims=True)
        draw = np.empty(values.shape[0], dtype=np.int32)
        for row in range(values.shape[0]):
            cdf = np.cumsum(probs[row])
            u = rng.uniform(1e-6, 1.0)
            index = min(int(np.searchsorted(cdf, u, side="right")), len(cdf) - 1)
            margin = replay_margin(cdf, u, index)
            retries = 0
            while margin < MARGIN_GATE:
                u = rng.uniform(1e-6, 1.0)
                index = min(int(np.searchsorted(cdf, u, side="right")), len(cdf) - 1)
                margin = replay_margin(cdf, u, index)
                retries += 1
                if retries >= MARGIN_RETRIES:
                    raise RuntimeError(
                        "no boundary-safe uniform for one decision "
                        f"(stream {stream_seed}, row {row})"
                    )
            stats["resamples"] += retries
            stats["count"] += 1
            stats["min_margin"] = min(stats["min_margin"], margin)
            us.append(float(u))
            draw[row] = index
        if single:
            return mx.array(draw[:1])
        return mx.array(draw)

    return us, stats, mock.patch.object(mx.random, "categorical", wrapper)


def tiny_config():
    from mlx_audio.tts.models.bark.bark import (
        CodecConfig,
        CoarseAcousticsConfig,
        FineAcousticsConfig,
        ModelConfig,
        SemanticConfig,
    )

    return ModelConfig(
        codec_config=CodecConfig(),
        semantic_config=SemanticConfig(
            block_size=1024,
            input_vocab_size=129600,
            output_vocab_size=10016,
            n_layer=2,
            n_head=2,
            n_embd=16,
            bias=False,
        ),
        coarse_acoustics_config=CoarseAcousticsConfig(
            block_size=1024,
            input_vocab_size=12096,
            output_vocab_size=12048,
            n_layer=2,
            n_head=2,
            n_embd=16,
            bias=False,
        ),
        fine_acoustics_config=FineAcousticsConfig(
            block_size=1024,
            input_vocab_size=1056,
            output_vocab_size=64,
            n_layer=2,
            n_head=2,
            n_embd=16,
            n_codes_total=8,
            n_codes_given=1,
        ),
        codec_path="mlx-community/encodec-24khz-float32",
        sample_rate=24000,
    )


def build_model(seed: int):
    from mlx_audio.tts.models.bark.bark import Model

    config = tiny_config()
    mx.random.seed(seed)
    model = Model(config)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if "layernorm" in name and name.endswith(".weight"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    # The reference ties fine books 1..7's input embeddings to the lm
    # heads ([output_vocab, n_embd] tables); at tiny output_vocab 64 the
    # coarse codes riding book 1 would index out of bounds, so the
    # fixture stores independent [input_vocab, n_embd] tables instead.
    for index in range(1, 8):
        name = f"fine_acoustics.input_embeds_layers.{index}.weight"
        rng = np.random.default_rng(9000 + index)
        flat[name] = mx.array(
            (rng.standard_normal((1056, 16)) * 0.35).astype(np.float32)
        )
    model.load_weights(list(flat.items()), strict=False)
    mx.eval(model)
    return model, flat


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    if isinstance(arr, (int, np.integer)):
        return [int(arr)]
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def ints2d(arr) -> list:
    return [ints(row) for row in np.asarray(arr)]


def make_pipeline(model, codec):
    from mlx_audio.codec.models.encodec.encodec import Encodec
    from mlx_audio.tts.models.bark.pipeline import Pipeline

    with mock.patch.object(Encodec, "from_pretrained", lambda *a, **k: (codec, None)):
        return Pipeline(model=model, tokenizer=StubTokenizer(), config=tiny_config())


def capture_stages(pipe, captured):
    """Wrap the pipeline stages so their outputs land in `captured`, and
    return a restore callable. The restore matters: re-wrapping on the
    next scenario would chain the wrappers and let a later scenario's
    run overwrite this scenario's captured dict (the bug that produced
    the original self-inconsistent scenario a)."""
    orig = (
        pipe.generate_text_semantic,
        pipe.generate_coarse,
        pipe.generate_fine,
    )

    def rec_semantic(*args, **kwargs):
        out, encoded = orig[0](*args, **kwargs)
        captured["semantic_tokens"] = ints(out)
        captured["encoded_text_ids"] = ints(encoded)
        return out, encoded

    def rec_coarse(*args, **kwargs):
        out = orig[1](*args, **kwargs)
        captured["coarse_codes"] = ints2d(out)
        return out

    def rec_fine(*args, **kwargs):
        out = orig[2](*args, **kwargs)
        captured["fine_codes"] = ints2d(out)
        return out

    pipe.generate_text_semantic = rec_semantic
    pipe.generate_coarse = rec_coarse
    pipe.generate_fine = rec_fine

    def restore():
        pipe.generate_text_semantic = orig[0]
        pipe.generate_coarse = orig[1]
        pipe.generate_fine = orig[2]

    return restore


def run_scenario(pipe, codec, text, stream_seed):
    captured: dict = {}
    restore = capture_stages(pipe, captured)
    # Spy on the tokenizer: the recording must prove the pipeline encoded
    # the raw scenario text (one call, no splitting or normalization), so
    # the derived prompt ids describe what this run actually consumed.
    orig_encode = pipe.tokenizer.encode
    encode_calls: list[tuple[str, list[int]]] = []

    def spy_encode(value, *args, **kwargs):
        ids = orig_encode(value, *args, **kwargs)
        encode_calls.append((value, [int(v) for v in ids]))
        return ids

    pipe.tokenizer.encode = spy_encode
    try:
        from mlx_audio.codec.models.encodec.encodec import Encodec

        us, stats, sampler = inverse_cdf_sampler(stream_seed)
        with mock.patch.object(Encodec, "from_pretrained", lambda *a, **k: (codec, None)):
            with sampler:
                results = list(
                    pipe(text, voice=None, temperature=0.7, use_kv_caching=True, verbose=False)
                )
    finally:
        restore()
        pipe.tokenizer.encode = orig_encode
    if not results:
        raise RuntimeError("pipeline produced no result")
    if len(encode_calls) != 1 or encode_calls[0][0] != text:
        raise RuntimeError(
            f"pipeline consumed unexpected tokenizer input: {encode_calls!r}"
        )
    captured["tokenizer_text"] = encode_calls[0][0]
    # Self-consistency by construction: the scenario's prompt ids come
    # from the pipeline's OWN encoding of this run (strip the text pads
    # and the encoding offset), never from an independent stub_encode
    # call. The broken recording filled this field from a separate
    # stub_encode while the capture chaining let scenario b's run
    # overwrite a's captured encoding, so scenario a carried b's ids and
    # a replay that re-encodes from text_token_ids could never match.
    captured["text_token_ids"] = [
        int(v) - 10048 for v in captured["encoded_text_ids"] if v != 129595
    ]

    # The reference semantic prefill logits for the merged prompt of this
    # same run: rebuilt from the derived ids so the recorded probe chain
    # and the recorded generation share one prompt by construction.
    model = pipe.model
    encoded = mx.array(captured["text_token_ids"]) + 10048
    encoded = mx.pad(encoded, (0, 256 - len(encoded)), constant_values=129595)
    history = mx.full((256,), 10000, dtype=mx.int32)
    x = mx.concatenate([encoded, history, mx.array([129599])])[None]
    stage = {}

    def rec(name, arr):
        stage[name] = floats(np.asarray(arr).ravel())

    sem = model.semantic
    tok_emb = sem.input_embeds_layer(x[:, :256]) + sem.input_embeds_layer(
        x[:, 256:512]
    )
    tail = sem.input_embeds_layer(x[:, 512:])
    merged = mx.concatenate([tok_emb, tail], axis=1)
    rec("embeds", merged[0])
    positions = mx.arange(0, merged.shape[1])
    pos_emb = sem.position_embeds_layer(positions)[None]
    h = merged + pos_emb
    rec("after_embeds", h[0])
    for index, block in enumerate(sem.layers):
        normed = block.layernorm_1(h)
        qkv = block.attn.att_proj(normed)
        rec(f"block{index}.qkv", qkv[0])
        attn_out = block.attn(normed, past_kv=None, use_cache=False)[0]
        rec(f"block{index}.attn", attn_out[0])
        h = h + attn_out
        rec(f"block{index}.attn_out", h[0])
        h = h + block.mlp(block.layernorm_2(h))
        rec(f"block{index}.out", h[0])
    h = sem.layernorm_final(h)
    rec("final_norm", h[0])
    logits = sem.lm_head(h[:, -1:, :])[0, 0]
    rec("logits", logits)
    captured["prefill_stages"] = stage
    captured["prefill_logits"] = floats(logits)
    audio = np.asarray(results[0].audio).ravel()
    captured["waveform"] = floats(audio)
    captured["uniforms"] = us
    captured["margin_gate"] = MARGIN_GATE
    captured["margin_count"] = stats["count"]
    captured["min_margin"] = stats["min_margin"]
    captured["margin_resamples"] = stats["resamples"]
    return captured


def accept(captured, min_semantic: int) -> bool:
    """Structural acceptance plus the boundary-margin gate."""
    semantic = captured.get("semantic_tokens", [])
    if not (min_semantic <= len(semantic) <= 480):
        return False
    coarse = captured.get("coarse_codes", [])
    if not isinstance(coarse, list) or len(coarse) != 2:
        return False
    if not coarse[0] or len(coarse[1]) != len(coarse[0]):
        return False
    fine = captured.get("fine_codes", [])
    if len(fine) != 8 or len(fine[0]) != len(coarse[0]):
        return False
    if not captured.get("waveform"):
        return False
    if captured.get("min_margin", 0.0) < MARGIN_GATE:
        return False
    return True


def semantic_ok(probe, min_semantic: int) -> bool:
    return (
        min_semantic <= len(probe["semantic_tokens"]) <= 480
        and probe["min_margin"] >= MARGIN_GATE
    )


def probe_semantic(pipe, codec, text, stream_seed):
    """Run only the semantic stage for one candidate stream. The full
    run re-creates the same rng, so the accepted probe stream reproduces
    the same semantic tokens inside run_scenario. The coarse stage is
    quadratic in the reference (no kv cache through the pipeline), so
    rejecting bad streams before it matters for search time."""
    us, stats, sampler = inverse_cdf_sampler(stream_seed)
    restore = capture_stages(pipe, {})
    try:
        from mlx_audio.codec.models.encodec.encodec import Encodec

        with mock.patch.object(
            Encodec, "from_pretrained", lambda *a, **k: (codec, None)
        ):
            with sampler:
                out, _ = pipe.generate_text_semantic(text, None, 0.7, True, verbose=False)
    finally:
        restore()
    return {
        "semantic_tokens": ints(out),
        "draws": len(us),
        "min_margin": stats["min_margin"],
        "resamples": stats["resamples"],
    }


def search_stream(pipe, codec, text, base, count, min_semantic):
    for offset in range(count):
        stream = base + offset
        try:
            probe = probe_semantic(pipe, codec, text, stream)
        except Exception:  # noqa: BLE001 - stream search
            continue
        if not semantic_ok(probe, min_semantic):
            continue
        try:
            captured = run_scenario(pipe, codec, text, stream)
        except Exception:  # noqa: BLE001 - stream search
            continue
        if not accept(captured, min_semantic):
            continue
        if not rerun_matches(pipe, codec, text, stream, captured):
            print(f"stream {stream}: rerun disagrees, skip")
            continue
        print(
            f"stream {stream}: semantic {len(captured['semantic_tokens'])} steps, "
            f"min margin {captured['min_margin']:.3g}, "
            f"resamples {captured['margin_resamples']}"
        )
        return stream, captured
    return None, None


def search_fixtures() -> None:
    codec, _, _ = bark_encodec()
    for seed in range(40):
        model, _ = build_model(seed)
        pipe = make_pipeline(model, codec)
        try:
            stream_a, captured_a = search_stream(
                pipe, codec, TEXTS[0], 500, 200, min_semantic=40
            )
        except Exception:  # noqa: BLE001 - seed search
            continue
        if captured_a is None:
            print(f"seed {seed}: no scenario a stream, skip")
            continue
        try:
            stream_b, captured_b = search_stream(
                pipe, codec, TEXTS[1], 900, 200, min_semantic=1
            )
        except Exception:  # noqa: BLE001 - seed search
            continue
        if captured_b is None:
            print(f"seed {seed}: no scenario b stream, skip")
            continue
        print(
            f"bark seed {seed}: semantic a {len(captured_a['semantic_tokens'])} steps, "
            f"semantic b {len(captured_b['semantic_tokens'])} steps, streams "
            f"{stream_a}/{stream_b}, min margins "
            f"{captured_a['min_margin']:.4g}/{captured_b['min_margin']:.4g}"
        )
        print(f"search picked seed {seed}; re-execing a fresh recording process")
        os.execve(
            sys.executable,
            [sys.executable, __file__],
            {
                **os.environ,
                "BARK_RECORD_SEED": str(seed),
                "BARK_RECORD_STREAM_A": str(stream_a),
                "BARK_RECORD_STREAM_B": str(stream_b),
                "PYTHONHASHSEED": "0",
            },
        )
    raise SystemExit("no seed produced acceptable bark fixtures")


def matches_prompt(captured, text: str) -> bool:
    """The scenario's recorded encoded text must still be the stub
    tokenizer's encoding of its own prompt. `run_scenario` derives the
    prompt ids from the captured encoding, so this is the independent
    cross-check that the stub tokenizer governs the pipeline (guards a
    future tokenizer swap silently changing the encoding)."""
    expected = [v + 10048 for v in stub_encode(text)]
    return captured["encoded_text_ids"][: len(expected)] == expected


def rerun_matches(pipe, codec, text, stream, captured) -> bool:
    again = run_scenario(pipe, codec, text, stream)
    fields = (
        "text_token_ids",
        "encoded_text_ids",
        "semantic_tokens",
        "coarse_codes",
        "fine_codes",
        "prefill_logits",
        "uniforms",
        "waveform",
    )
    return all(captured[field] == again[field] for field in fields)


def write_fixtures(flat, codec_flat, codec_config_value, seed, captured_a, captured_b, stream_a, stream_b) -> None:
    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), flat)
    codec_dir = CODEC_DIR
    codec_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(codec_dir / "model.safetensors"), codec_flat)
    (codec_dir / "config.json").write_text(
        json.dumps(codec_config_value, indent=2) + "\n"
    )
    config = tiny_config()
    config_dict = {
        "semantic_config": dict(vars(config.semantic_config)),
        "coarse_acoustics_config": dict(vars(config.coarse_acoustics_config)),
        "fine_acoustics_config": dict(vars(config.fine_acoustics_config)),
        "codec_path": config.codec_path,
        "sample_rate": config.sample_rate,
    }
    (case_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    traces = {
        "texts": TEXTS,
        "model_seed": seed,
        "stream_a": stream_a,
        "stream_b": stream_b,
        "margin_gate": MARGIN_GATE,
        "semantic_pad_token": 10000,
        "text_encoding_offset": 10048,
        "text_pad_token": 129595,
        "semantic_infer_token": 129599,
        "coarse_semantic_pad_token": 12048,
        "coarse_infer_token": 12050,
        "scenarios": {"a": captured_a, "b": captured_b},
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))


def record_fixtures(seed: int, stream_a: int, stream_b: int) -> None:
    """Fresh-process recording: a new interpreter reproduces exactly one
    run per scenario; long-running search processes do not (MLX kernel
    state accumulates), so the final fixtures are recorded here. Rerun
    equality and prompt consistency are verified before writing."""
    codec, codec_flat, codec_config_value = bark_encodec()
    model, flat = build_model(seed)
    pipe = make_pipeline(model, codec)
    captured_a = run_scenario(pipe, codec, TEXTS[0], stream_a)
    captured_b = run_scenario(pipe, codec, TEXTS[1], stream_b)
    for name, captured, text, stream in (
        ("a", captured_a, TEXTS[0], stream_a),
        ("b", captured_b, TEXTS[1], stream_b),
    ):
        if not matches_prompt(captured, text):
            raise SystemExit(f"scenario {name} encoded text does not match its prompt")
        if not rerun_matches(pipe, codec, text, stream, captured):
            raise SystemExit(f"scenario {name} recording rerun disagrees")
    write_fixtures(
        flat, codec_flat, codec_config_value, seed, captured_a, captured_b, stream_a, stream_b
    )
    print(
        f"recorded: seed {seed}, streams {stream_a}/{stream_b}, "
        f"semantic a {len(captured_a['semantic_tokens'])} steps, "
        f"semantic b {len(captured_b['semantic_tokens'])} steps, "
        f"min margins {captured_a['min_margin']:.4g}/{captured_b['min_margin']:.4g}, "
        f"resamples {captured_a['margin_resamples']}/{captured_b['margin_resamples']}"
    )


def main() -> None:
    record = os.environ.get("BARK_RECORD_SEED")
    if record:
        record_fixtures(
            int(record),
            int(os.environ.get("BARK_RECORD_STREAM_A", "529")),
            int(os.environ.get("BARK_RECORD_STREAM_B", "903")),
        )
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        search_fixtures()
    _finalize_manifest()


def _finalize_manifest() -> None:
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_bark_fixtures.py",
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
