#!/usr/bin/env python3
"""Regenerate the rumik-oss TTS family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/rumik_oss,
mlx_audio/lm/sample_utils.py, mlx 0.32.3). The reference module runs in
full: the cohere2 backbone with its stop predictor, the audio logits
processor (min-tokens gate, stop-predictor forcing, unit-token
restriction), the reference `mlx_audio.lm.generate.generate_step` loop,
the round-robin frame assembler, and the Mimi decode, all driven through
the reference methods with:

- a deterministic offline mock tokenizer (BOS + a word id per space
  token) standing in for the network CohereTokenizer, injected as
  `model._tokenizer`;
- a seeded tiny 8-book Mimi (quantizer_bins 64, 24 kHz, 12.5 fps)
  injected as `model._mimi`;
- seeded tiny cohere2 backbones whose stop-predictor head is re-biased
  per scenario: one that always fires (forcing `</audio>` after
  min_tokens) and one that never fires (max_tokens stop).

The sliced audio head (the reference default) is cross-checked against
the full-head processor path on both generate scenarios; the traces
record the ids both paths agree on.

The generate cases use num_quantizers 2 so the greedy unit-token walk
completes codec frames by parity alone; the code-major/quantizer-minor
mapping, the resync rules, and the level-major code layout are pinned at
the reference-shaped num_quantizers 8 by the decode case's crafted
sequences, which drive the reference `push_audio_token` and
`audio_tokens_to_codes` directly.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_rumik_oss_fixtures.py

Outputs into crates/audio/testdata/tts_rumik_oss/:

- decode/model.safetensors + decode/config.json + decode/traces.json
    Seeded tiny rumik model (4 cohere2 layers, hidden 32) configured at
    num_quantizers 8 / codebook_size 64 plus the tiny Mimi. Traces: the
    round-robin assembler states on crafted id sequences (in-order,
    out-of-order resync, non-unit break, end-token stop), the level-major
    codes, the no-frame error message, the derived last_unit_id for the
    reference-shaped 2048/8 config, the build_prompt/resolve_voice
    cases, the stop-predictor probabilities on fixed hidden rows, and
    the Mimi waveform for the decoded codes.
- generate_forced/... and generate_length/...
    End-to-end greedy runs (temperature 0) over the reference generate
    loop: prompt string and mock-tokenizer ids, generated ids, stop
    reason, complete frames, level-major codes, and the final waveform
    through the reference `_decode_frames`. `generate_forced` re-biases
    the stop predictor to always fire (the reference force path emits
    `</audio>` at the first eligible step); `generate_length` re-biases
    it to never fire and stops at max_tokens with a dropped partial tail.
- sampling/traces.json
    The reference `make_audio_logits_processor` on fixed full-vocabulary
    logits for the three step regimes (below min_tokens, stop fires, stop
    silent), the log-softmax normalization, the greedy argmax, and the
    top-p/top-k filters of `make_sampler`. No RNG draws are recorded.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The
backbone seed search at the bottom pins the first seed whose greedy runs
satisfy the frame-count assertions; regeneration reproduces it.
"""

from __future__ import annotations

import dataclasses
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

from mlx_audio.codec.models.mimi import Mimi, MimiStreamingDecoder
from mlx_audio.codec.models.mimi.mimi import mimi_202407
from mlx_audio.lm.generate import generate_step
from mlx_audio.lm.sample_utils import apply_top_k, apply_top_p, make_sampler

with mock.patch(
    "mlx_audio.codec.models.mimi.Mimi.from_pretrained",
    return_value=mock.MagicMock(),
):
    import mlx_audio.tts.models.rumik_oss.rumik_oss as rumik

OUT = Path("../turbospark/crates/audio/testdata/tts_rumik_oss")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# Tiny cohere2 geometry (window 4 over prompts longer than the window,
# three sliding layers plus one global layer, tied head over logit_scale).
COHERE2 = dict(
    model_type="rumik_oss",
    hidden_size=32,
    head_dim=8,
    num_hidden_layers=4,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    rope_theta=50000.0,
    layer_norm_eps=1e-5,
    logit_scale=0.0625,
    attention_bias=False,
    layer_norm_bias=False,
    sliding_window=4,
    sliding_window_pattern=4,
)

# Audio vocabulary for the generation cases: num_quantizers 2 / codebook 64
# so greedy unit draws complete frames by parity; ids beyond the units are
# the audio bookkeeping tokens the config carries.
GEN_UNITS_FIRST = 24
GEN_QUANTIZERS = 2
GEN_CODEBOOK = 64
GEN_UNITS_LAST = GEN_UNITS_FIRST + GEN_CODEBOOK * GEN_QUANTIZERS - 1  # 151
GEN_END = 152
GEN_START = 153
GEN_TEXT_START = 154
GEN_VOCAB = 200

# The decode case keeps the reference's 8 quantizers (codebook 64 to fit
# the tiny Mimi's quantizer_bins).
DECODE_UNITS_FIRST = 24
DECODE_QUANTIZERS = 8
DECODE_CODEBOOK = 64
DECODE_UNITS_LAST = DECODE_UNITS_FIRST + DECODE_CODEBOOK * DECODE_QUANTIZERS - 1
DECODE_END = 536
DECODE_VOCAB = 560

# Reference-shaped audio config for the __post_init__ derivation trace
# (real defaults; no backbone is built for it).
REALSHAPED_NUM_QUANTIZERS = 8
REALSHAPED_CODEBOOK = 2048
REALSHAPED_FIRST = 24

MOCK_BOS = 1

# The Rust Mimi loader runs the transformer at the SEANET width (the
# reference in/out projections are the identity at these sizes), so the
# dimension and d_model stay equal like the sesame fixtures. The decode
# case uses the reference-shaped 8-book codec; the generate cases use a
# 2-book codec matching their num_quantizers (the reference split
# quantizer sums only the provided books, and the fixture keeps that
# reduction explicit instead of patching the codec).
TINY_MIMI = dict(
    num_codebooks=8,
    seanet_dimension=16,
    seanet_nfilters=8,
    transformer_d_model=16,
    transformer_num_heads=4,
    transformer_dim_feedforward=32,
    quantizer_dim=8,
    quantizer_bins=64,
)

# Pinned backbone seed: first seed of the search below whose greedy runs
# satisfy the frame-count assertions.
BACKBONE_SEED = 0


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def masked(arr) -> list:
    """Processor outputs carry negative infinities, which Python's json
    module writes as invalid strict JSON; emit null sentinels."""
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(arr, dtype=np.float32).ravel().tolist()
    ]


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


class MockTokenizer:
    """Deterministic BOS + word-id stand-in for the CohereTokenizer."""

    def __init__(self):
        self.vocab = {}

    def id_for(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = 10 + 2 * len(self.vocab)
        return self.vocab[word]

    def __call__(self, text: str):
        ids = [MOCK_BOS]
        for word in text.split(" "):
            ids.append(self.id_for(word))
        return {"input_ids": ids}


def model_config(**overrides) -> "rumik.ModelConfig":
    fields = dict(COHERE2)
    fields.update(overrides)
    return rumik.ModelConfig(**fields)


def seeded_model(seed: int, **audio) -> "rumik.Model":
    """Deterministic rumik model: small uniform weights, norms biased up,
    exactly the lm fixture convention."""
    vocab_size = audio.pop("vocab_size")
    config = model_config(vocab_size=vocab_size, **audio)
    mx.random.seed(seed)
    model = rumik.Model(config)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith("norm.weight") or name.endswith("layers.0.weight"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    model.update(tree_unflatten(flat))
    model._tokenizer = MockTokenizer()
    return model


def rebiased(model: "rumik.Model", bias: float) -> "rumik.Model":
    """Zeroes the stop-predictor readout weight and pins the bias, so the
    forced scenario always fires (positive) and the length scenario never
    does (negative). The projector keeps its seeded weights."""
    flat = dict(tree_flatten(model.parameters()))
    flat["stop_predictor.layers.3.weight"] = mx.zeros(
        flat["stop_predictor.layers.3.weight"].shape
    )
    flat["stop_predictor.layers.3.bias"] = mx.array([bias], dtype=mx.float32)
    model.update(tree_unflatten(flat))
    return model


def tiny_mimi(num_codebooks: int) -> Mimi:
    config = mimi_202407(num_codebooks)
    config.seanet.dimension = TINY_MIMI["seanet_dimension"]
    config.seanet.nfilters = TINY_MIMI["seanet_nfilters"]
    config.transformer.d_model = TINY_MIMI["transformer_d_model"]
    config.transformer.num_heads = TINY_MIMI["transformer_num_heads"]
    config.transformer.dim_feedforward = TINY_MIMI["transformer_dim_feedforward"]
    config.quantizer_dim = TINY_MIMI["quantizer_dim"]
    config.quantizer_bins = TINY_MIMI["quantizer_bins"]
    mx.random.seed(7)
    model = Mimi(config)
    # The 0.05 scale keeps the random-weight Mimi decoder bounded (see the
    # sesame fixtures for the same note).
    rng = np.random.default_rng(41)
    seeded = {}
    for name, value in dict(tree_flatten(model.parameters())).items():
        seeded[name] = mx.array(
            (rng.standard_normal(value.shape) * 0.05).astype(np.float32)
        )
    model.load_weights(list(seeded.items()))
    for split in ("rvq_first", "rvq_rest"):
        vq = getattr(model.quantizer, split).vq
        for layer in vq.layers:
            layer.codebook.update_in_place()
    model.upsample.convtr.convtr.convtr.update_in_place()
    for layer in model.decoder.layers:
        layer.upsample.convtr.convtr.update_in_place()
    return model


def save_case(name: str, model: "rumik.Model") -> None:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(case_dir / "model.safetensors"), weights)
    config = dataclasses.asdict(model.config)
    # model_path is injected by the loader, not part of the checkpoint.
    config.pop("model_path", None)
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def save_mimi(case: str, mimi: Mimi) -> None:
    """Writes the tiny codec next to the family model so the Rust tests
    load the decode side through the same files."""
    case_dir = OUT / case
    config = {
        "channels": mimi.cfg.channels,
        "sample_rate": mimi.cfg.sample_rate,
        "frame_rate": mimi.cfg.frame_rate,
        "renormalize": mimi.cfg.renormalize,
        "seanet": {
            "dimension": mimi.cfg.seanet.dimension,
            "channels": mimi.cfg.seanet.channels,
            "causal": mimi.cfg.seanet.causal,
            "nfilters": mimi.cfg.seanet.nfilters,
            "nresidual_layers": mimi.cfg.seanet.nresidual_layers,
            "ratios": mimi.cfg.seanet.ratios,
            "ksize": mimi.cfg.seanet.ksize,
            "residual_ksize": mimi.cfg.seanet.residual_ksize,
            "last_ksize": mimi.cfg.seanet.last_ksize,
            "dilation_base": mimi.cfg.seanet.dilation_base,
            "pad_mode": mimi.cfg.seanet.pad_mode,
            "true_skip": mimi.cfg.seanet.true_skip,
            "compress": mimi.cfg.seanet.compress,
        },
        "transformer": {
            "d_model": mimi.cfg.transformer.d_model,
            "num_heads": mimi.cfg.transformer.num_heads,
            "num_layers": mimi.cfg.transformer.num_layers,
            "causal": mimi.cfg.transformer.causal,
            "norm_first": mimi.cfg.transformer.norm_first,
            "bias_ff": mimi.cfg.transformer.bias_ff,
            "bias_attn": mimi.cfg.transformer.bias_attn,
            "layer_scale": mimi.cfg.transformer.layer_scale,
            "positional_embedding": mimi.cfg.transformer.positional_embedding,
            "use_conv_bias": mimi.cfg.transformer.use_conv_bias,
            "gating": mimi.cfg.transformer.gating,
            "norm": mimi.cfg.transformer.norm,
            "context": mimi.cfg.transformer.context,
            "max_period": mimi.cfg.transformer.max_period,
            "max_seq_len": mimi.cfg.transformer.max_seq_len,
            "kv_repeat": mimi.cfg.transformer.kv_repeat,
            "dim_feedforward": mimi.cfg.transformer.dim_feedforward,
            "conv_layout": mimi.cfg.transformer.conv_layout,
            "cross_attention": mimi.cfg.transformer.cross_attention,
            "conv_kernel_size": mimi.cfg.transformer.conv_kernel_size,
        },
        "quantizer_nq": mimi.cfg.quantizer_nq,
        "quantizer_bins": mimi.cfg.quantizer_bins,
        "quantizer_dim": mimi.cfg.quantizer_dim,
    }
    (case_dir / "codec_config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.save_safetensors(
        str(case_dir / "codec.safetensors"), dict(tree_flatten(mimi.parameters()))
    )


def write_manifest() -> None:
    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "generator": "tools/gen_rumik_oss_fixtures.py",
        "files": {
            str(path.relative_to(OUT)): "sha256:"
            + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def codes_array(codes) -> list:
    """audio_tokens_to_codes returns [1, Q, T]; the traces carry [Q][T]."""
    return np.array(codes, dtype=np.int32)[0].tolist()


# --------------------------------------------------------------------------
# decode case: assembler math, prompt building, stop probabilities, Mimi.
# --------------------------------------------------------------------------


def assembler_trace(model: "rumik.Model", name: str, token_ids: list) -> dict:
    """Steps the reference push_audio_token one id at a time, recording
    the accumulator after every token plus the completed frames."""
    frame: list = []
    steps = []
    frames = []
    for token in token_ids:
        frame, done = model.push_audio_token(token, frame)
        steps.append({"token": int(token), "frame": ints(frame)})
        if done is not None:
            frames.append(ints(done))
    return {
        "name": name,
        "tokens": ints(token_ids),
        "steps": steps,
        "frames": frames,
        "codes": codes_array(model.audio_tokens_to_codes(token_ids)),
    }


def _error_message(model: "rumik.Model", token_ids: list) -> str:
    try:
        model.audio_tokens_to_codes(token_ids)
    except ValueError as error:
        return str(error)
    raise SystemExit("expected audio_tokens_to_codes to refuse a frameless input")


def prompt_traces(model: "rumik.Model") -> list:
    cases = [
        {"text": "namaste", "voice": None, "description": None},
        {"text": "hello world", "voice": "aisha", "description": None},
        {"text": "  padded text  ", "voice": "ZOYA", "description": "happy, fast"},
        {
            "text": '<description="inline"> kept',
            "voice": None,
            "description": "ignored",
        },
        {"text": "tagged", "voice": None, "description": None, "instruct": "calm"},
    ]
    traces = []
    for case in cases:
        entry = dict(case)
        description = case.get("description") or case.get("instruct")
        prompt = model.build_prompt(case["text"], case["voice"], description)
        entry["prompt"] = prompt
        entry["prompt_ids"] = ints(model.encode_prompt(prompt))
        traces.append(entry)
    try:
        model.resolve_voice("unknown-speaker")
    except ValueError as error:
        traces.append({"voice": "unknown-speaker", "error": str(error)})
    return traces


def stop_traces(model: "rumik.Model") -> list:
    hidden = np.random.default_rng(11).standard_normal((3, model.config.hidden_size))
    rows = []
    for index in range(hidden.shape[0]):
        row = mx.array(hidden[index : index + 1], dtype=mx.float32)
        rows.append(
            {
                "hidden": floats(row),
                "probability": float(model.stop_probability(row).item()),
            }
        )
    return rows


def build_decode_case(mimi: Mimi) -> None:
    model = seeded_model(
        BACKBONE_SEED,
        vocab_size=DECODE_VOCAB,
        num_quantizers=DECODE_QUANTIZERS,
        codebook_size=DECODE_CODEBOOK,
        audio_start_token_id=DECODE_END + 1,
        audio_end_token_id=DECODE_END,
        text_start_token_id=DECODE_END + 2,
        first_unit_id=DECODE_UNITS_FIRST,
    )
    save_case("decode", model)
    model._mimi = mimi
    save_mimi("decode", mimi)

    first = DECODE_UNITS_FIRST
    q = DECODE_QUANTIZERS

    def unit(code: int, quantizer: int) -> int:
        return first + code * q + quantizer

    in_order = (
        [unit(1, i) for i in range(q)]
        + [unit(2, i) for i in range(q)]
        + [DECODE_END]
        + [unit(3, i) for i in range(3)]  # tail after the end token
    )
    resync = (
        [unit(4, 0), unit(4, 1), unit(4, 3)]  # quantizer jump drops the pair
        + [unit(5, 2)]  # still out of order: dropped
        + [unit(6, i) for i in range(q)]
        + [42]  # non-unit id: drops the partial frame
        + [unit(7, 0), unit(7, 1)]
    )
    incomplete = [unit(8, i) for i in range(q)] + [unit(9, 0), unit(9, 1), unit(9, 0)]
    no_frame = [unit(10, 1), unit(10, 0), unit(10, 1)]

    traces = {
        "audio": {
            "first_unit_id": DECODE_UNITS_FIRST,
            "last_unit_id": DECODE_UNITS_LAST,
            "audio_end_token_id": DECODE_END,
            "num_quantizers": DECODE_QUANTIZERS,
            "codebook_size": DECODE_CODEBOOK,
        },
        "assembler": [
            assembler_trace(model, "in_order", in_order),
            assembler_trace(model, "resync", resync),
            assembler_trace(model, "incomplete_tail", incomplete),
        ],
        "no_frame_error": _error_message(model, no_frame),
        # The reference __post_init__ derivation at the real config
        # proportions (2048 x 8).
        "derived_last_unit_id": int(
            model_config(
                vocab_size=1,
                num_quantizers=REALSHAPED_NUM_QUANTIZERS,
                codebook_size=REALSHAPED_CODEBOOK,
                first_unit_id=REALSHAPED_FIRST,
            ).last_unit_id
        ),
        "prompts": prompt_traces(model),
        "stop_probability": stop_traces(model),
    }

    # The reference _decode_frames path for the in-order codes.
    frames = traces["assembler"][0]["frames"]
    audio = model._decode_frames([list(map(int, frame)) for frame in frames])
    traces["waveform"] = floats(audio)
    traces["waveform_sample_rate"] = int(model.config.sample_rate)

    (OUT / "decode" / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")


# --------------------------------------------------------------------------
# generate cases: end-to-end greedy runs over the reference generate loop.
# --------------------------------------------------------------------------


def generate_audio_fields() -> dict:
    return dict(
        vocab_size=GEN_VOCAB,
        num_quantizers=GEN_QUANTIZERS,
        codebook_size=GEN_CODEBOOK,
        audio_start_token_id=GEN_START,
        audio_end_token_id=GEN_END,
        text_start_token_id=GEN_TEXT_START,
        first_unit_id=GEN_UNITS_FIRST,
    )


def greedy_ids(model, prompt_ids, max_tokens: int, min_tokens: int) -> list:
    """The reference generate loop's ids: generate_step with the family's
    audio logits processor and a greedy sampler, stopping at </audio>."""
    sampler = make_sampler(0.0)
    processors = [model.make_audio_logits_processor(min_tokens)]
    generated = []
    for token, _ in generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=processors,
    ):
        generated.append(int(token))
        if token == model.audio_end_id:
            break
    return generated


def sliced_ids(model, prompt_ids, max_tokens: int, min_tokens: int) -> list:
    """The same loop through the reference sliced audio head (the
    generate default path)."""
    sampler = model.make_sliced_sampler(make_sampler(0.0))
    processors = [model.make_sliced_logits_processor(min_tokens)]
    generated = []
    with model.audio_head(True):
        for token, _ in generate_step(
            mx.array(prompt_ids),
            model,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors,
        ):
            generated.append(int(token))
            if token == model.audio_end_id:
                break
    return generated


def run_generate(model, mimi, max_tokens: int, min_tokens: int) -> dict:
    model._mimi = mimi
    prompt_string = model.build_prompt("namaste world", "aisha", "happy, steady")
    prompt_ids = ints(model.encode_prompt(prompt_string))
    generated = greedy_ids(model, prompt_ids, max_tokens, min_tokens)
    if generated != sliced_ids(model, prompt_ids, max_tokens, min_tokens):
        raise SystemExit("sliced and full-head ids diverge")
    stopped = "audio_end" if generated[-1] == GEN_END else "max_tokens"

    frame: list = []
    frames = []
    for token in generated:
        frame, done = model.push_audio_token(token, frame)
        if done is not None:
            frames.append(ints(done))
    codes = codes_array(model.audio_tokens_to_codes(generated))
    # The reference _consume final decode (non-streaming).
    waveform = model._decode_frames(frames)
    return {
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "prompt": prompt_string,
        "prompt_ids": prompt_ids,
        "generated": generated,
        "stopped_by": stopped,
        "frames": frames,
        "codes": codes,
        "waveform": floats(waveform),
        "sample_rate": int(model.sample_rate),
    }


def build_generate_case(name: str, bias: float, mimi) -> dict:
    model = rebiased(seeded_model(BACKBONE_SEED, **generate_audio_fields()), bias)
    save_case(name, model)
    save_mimi(name, mimi)
    trace = run_generate(model, mimi, max_tokens=24, min_tokens=16)
    (OUT / name / "traces.json").write_text(json.dumps(trace, indent=2) + "\n")
    return trace


# --------------------------------------------------------------------------
# sampling traces: the processor and the sampler filters on fixed logits.
# --------------------------------------------------------------------------


def build_sampling_traces(model: "rumik.Model") -> None:
    vocab = GEN_VOCAB
    logits = mx.array(
        (np.random.default_rng(23).standard_normal(vocab) * 2.0).astype(np.float32)
    )
    hidden = mx.array(
        np.random.default_rng(29).standard_normal((1, model.config.hidden_size)),
        dtype=mx.float32,
    )
    hidden = mx.array(
        np.random.default_rng(29).standard_normal((1, model.config.hidden_size)),
        dtype=mx.float32,
    )
    # The processor consults the model's own stop head, so each branch
    # runs on the model re-biased for it: the forced head always fires,
    # the silenced one never does. below_min_tokens never reaches the
    # predictor (either head agrees).
    silent_model = rebiased(
        seeded_model(BACKBONE_SEED, **generate_audio_fields()), -20.0
    )
    forced_p = float(model.stop_probability(hidden).item())
    silent_p = float(silent_model.stop_probability(hidden).item())

    cases = []
    for name, step, min_tokens, stop, case_model in [
        ("below_min_tokens", 0, 8, forced_p, model),
        ("stop_fires", 8, 8, forced_p, model),
        ("stop_silent", 8, 8, silent_p, silent_model),
    ]:
        case_model._last_hidden = hidden
        processor = case_model.make_audio_logits_processor(min_tokens)
        # Step the processor's internal counter forward to the target step.
        for _ in range(step):
            processor(None, logits[None])
        processed = processor(None, logits[None])[0]
        logprobs = processed - mx.logsumexp(processed, keepdims=True)
        case = {
            "name": name,
            "step": step,
            "min_tokens": min_tokens,
            "stop_probability": stop,
            "logits": floats(logits),
            "processed": masked(processed),
            "logprobs": masked(logprobs),
            "argmax": int(mx.argmax(logprobs).item()),
        }
        # The make_sampler filters on the normalized log-probs (top-p
        # first, then top-k). The categorical draw itself is the
        # documented Rust divergence; the filters and the greedy argmax
        # are what the fixtures pin.
        case["top_p_logprobs"] = masked(apply_top_p(logprobs, 0.9))
        case["top_k_logprobs"] = masked(apply_top_k(logprobs, 30))
        cases.append(case)

    traces = {
        "audio": {
            "first_unit_id": GEN_UNITS_FIRST,
            "last_unit_id": GEN_UNITS_LAST,
            "audio_end_token_id": GEN_END,
        },
        "stop_hidden": floats(hidden),
        "stop_probability": forced_p,
        "cases": cases,
    }
    (OUT / "sampling").mkdir(parents=True, exist_ok=True)
    (OUT / "sampling" / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")


def main() -> None:
    global BACKBONE_SEED
    OUT.mkdir(parents=True, exist_ok=True)
    mimi = tiny_mimi(TINY_MIMI["num_codebooks"])
    gen_mimi = tiny_mimi(GEN_QUANTIZERS)

    # Seed search: the forced run gets 16 parity unit tokens and needs two
    # complete frames; the length run needs at least one.
    forced = length = None
    for seed in range(64):
        BACKBONE_SEED = seed
        forced = build_generate_case("generate_forced", bias=20.0, mimi=gen_mimi)
        length = build_generate_case("generate_length", bias=-20.0, mimi=gen_mimi)
        if len(forced["frames"]) >= 2 and len(length["frames"]) >= 1:
            break
    else:
        raise SystemExit("no backbone seed produced acceptable greedy runs")
    print(f"backbone seed: {BACKBONE_SEED}")
    print(
        f"forced: {len(forced['frames'])} frames, stopped {forced['stopped_by']}"
    )
    print(
        f"length: {len(length['frames'])} frames, stopped {length['stopped_by']}"
    )

    build_decode_case(mimi)

    # The sampling traces reuse the forced model's real stop head.
    forced_model = rebiased(
        seeded_model(BACKBONE_SEED, **generate_audio_fields()), 20.0
    )
    build_sampling_traces(forced_model)

    write_manifest()
    print(f"wrote fixtures under {OUT}")


if __name__ == "__main__":
    main()
