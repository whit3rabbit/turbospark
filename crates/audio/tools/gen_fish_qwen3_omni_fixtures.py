#!/usr/bin/env python3
"""Regenerate the fish_qwen3_omni (fish_speech) TTS family Rust fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/tts/models/fish_qwen3_omni, mlx 0.32.3). The reference module
is imported with `mx.compile` neutered (so `_sample_logits` runs eagerly
and its `mx.random.uniform` draws are observable); the network tokenizer
and codec singletons from `post_load_hook` are replaced with a stub
tokenizer and the tiny seeded Fish S1 DAC (the same geometry and weights
as testdata/fish_s1_dac/tiny_weights.safetensors, loaded from that
file). The single-sequence generate path (`_generate_codes_for_batch`
over the assembled conversation) runs the true reference code.

Sampling noise comes from `controlled_uniforms`: every scenario draws
from its own fresh numpy stream, because the RAS high-temperature
resample consumes the global MLX random state and would otherwise not be
reproducible across calls. The drawn arrays are recorded per call and
replayed by the Rust tests (one call = one full-vocabulary array, the
reference draw shape); greedy calls draw nothing, so Rust consumption
matches when the control flow matches.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_fish_qwen3_omni_fixtures.py

Outputs into crates/audio/testdata/tts_fish_qwen3_omni/:

- generate/model.safetensors + generate/config.json
    Seeded tiny DualAR model (2 slow + 2 fast layers, text dim 32,
    fast dim 24 so the projector exists, vocab 300, 4 codebooks, audio
    vocab 16). The embedding rows of one semantic token and the im_end
    token are amplified; a seed search pins the scenarios below.
- generate/traces.json
    The stub-encoded header/text ids, the encoded prompt matrices
    (first prompt, prompt with the assistant header, and the second
    prompt carrying the first assistant VQ reply), per scenario:
    generated codes, the recorded uniform draws in consumption order,
    and the decoded waveforms through the tiny codec.
- sampling/traces.json
    `_sample_logits` on fixed logits: greedy argmax and sampled draws
    with recorded uniforms, plus RAS decisions on fixed histories.
- prompt/traces.json
    Conversation encoding with VQ parts (header, text, codebook rows).
- speed/traces.json
    `_adjust_speed` on a fixed waveform at 1.0, 0.8, and 1.5.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
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


def import_fish_speech():
    """Import fish_speech with mx.compile neutered so the decorated
    `_sample_logits` runs eagerly and its uniform draws are observable."""
    import mlx_audio.tts.models.fish_qwen3_omni.fish_speech as module

    def fake_compile(fun=None, *args, **kwargs):
        return fun

    with mock.patch.object(mx, "compile", fake_compile):
        module = importlib.reload(module)
    return module


F = import_fish_speech()

# Greedy-argmax decision margins: every temperature-0 `_sample_logits`
# call records the top1-top2 gap of its input, and every sampled call
# records the winner-over-runner-up ratio of the Gumbel scores; the seed
# search requires comfortable minimums so the Rust f32 port
# (accumulation order differs from MLX by a few ulps) cannot flip a
# decision.
MARGINS: list[float] = []
SAMPLED_MARGINS: list[float] = []
_ORIGINAL_SAMPLE = F._sample_logits


def _softmax(values):
    shifted = values - values.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def _margin_sample(logits, temperature, top_p, top_k):
    if temperature <= 0:
        values = np.sort(np.asarray(logits, dtype=np.float32).ravel())
        MARGINS.append(float(values[-1] - values[-2]))
        return _ORIGINAL_SAMPLE(logits, temperature, top_p, top_k)
    token = _ORIGINAL_SAMPLE(logits, temperature, top_p, top_k)
    values = np.asarray(logits, dtype=np.float32).ravel()
    vocab = len(values)
    order = np.argsort(-values, kind="stable")
    top_k_eff = vocab if top_k <= 0 or top_k > vocab else top_k
    cumulative = np.cumsum(_softmax(values)[order])
    remove = (cumulative > top_p) | (np.arange(vocab) >= top_k_eff)
    remove[order[0]] = False
    filtered = np.where(remove, -np.inf, values)
    probs = _softmax(filtered / max(temperature, 1e-5))
    # The uniform wrapper appends this call's draws before the original
    # function computes the noise, so the last array is this call's.
    draws = np.asarray(UNIFORM_CALLS[-1], dtype=np.float32)
    scores = probs / -np.log(np.clip(draws, 1e-6, None))
    winners = np.argsort(-scores)[:2]
    if np.isfinite(scores[winners[1]]) and scores[winners[0]] > 0:
        SAMPLED_MARGINS.append(float(scores[winners[0]] / scores[winners[1]]))
    return token


F._sample_logits = _margin_sample

# The active uniform recording target (set by controlled_uniforms).
UNIFORM_CALLS: list[list[float]] = []

from mlx_audio.tts.models.fish_qwen3_omni.config import (  # noqa: E402
    FishAudioDecoderConfig,
    FishTextConfig,
    ModelConfig,
)

OUT = Path("../turbospark/crates/audio/testdata/tts_fish_qwen3_omni")
DAC_TESTDATA = Path("../turbospark/crates/audio/testdata/fish_s1_dac")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

# Stub tokenizer: one id per string, stable across runs.
IM_END_ID = 299
SEMANTIC_BEGIN_ID = 100


def stub_encode(text: str) -> list[int]:
    value = sum(ord(ch) for ch in text)
    return [3 + value % 90]


class StubTokenizer:
    def __init__(self):
        self.semantic_begin_id = SEMANTIC_BEGIN_ID
        self.semantic_end_id = SEMANTIC_BEGIN_ID + 15
        self._vocab_size = 300

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return stub_encode(text)

    def get_token_id(self, token: str) -> int:
        if token == "<|im_end|>":
            return IM_END_ID
        return stub_encode(token)[0]


def tiny_config() -> ModelConfig:
    text = FishTextConfig(
        model_type="fish_qwen3",
        vocab_size=300,
        n_layer=2,
        n_head=4,
        dim=32,
        intermediate_size=48,
        n_local_heads=2,
        head_dim=8,
        rope_base=10000.0,
        norm_eps=1e-5,
        max_seq_len=64,
        attention_qkv_bias=False,
        attention_o_bias=False,
        attention_qk_norm=True,
    )
    audio = FishAudioDecoderConfig(
        model_type="fish_qwen3_audio_decoder",
        vocab_size=12,
        n_layer=2,
        n_head=4,
        dim=24,
        intermediate_size=32,
        n_local_heads=2,
        head_dim=6,
        rope_base=10000.0,
        norm_eps=1e-5,
        max_seq_len=4,
        attention_qkv_bias=False,
        attention_o_bias=False,
        attention_qk_norm=False,
        text_dim=24,
        num_codebooks=4,
    )
    return ModelConfig(
        model_type="fish_speech",
        pad_token_id=151669,
        eos_token_id=151645,
        audio_pad_token_id=151677,
        semantic_start_token_id=SEMANTIC_BEGIN_ID,
        semantic_end_token_id=SEMANTIC_BEGIN_ID + 15,
        sample_rate=44100,
        text_config=text,
        audio_decoder_config=audio,
    )


def tiny_dac():
    """The tiny Fish S1 DAC from the codec fixtures, loaded from the
    saved seeded weights so the Rust codec instance matches exactly."""
    spec = importlib.util.spec_from_file_location(
        "gen_fish_s1_dac_fixtures",
        "../turbospark/crates/audio/tools/gen_fish_s1_dac_fixtures.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dac = module.make_tiny_dac()
    saved = mx.load(str(DAC_TESTDATA / "tiny_weights.safetensors"))
    dac.load_weights(list(saved.items()), strict=False)
    mx.eval(dac)
    return dac


def build_model(seed: int, amplify_semantic: float, amplify_im_end: float):
    config = tiny_config()
    mx.random.seed(seed)
    model = F.Model(config)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith("norm.weight"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    embed = np.array(flat["model.embeddings.weight"], dtype=np.float32)
    embed[SEMANTIC_BEGIN_ID + 3] *= amplify_semantic
    embed[IM_END_ID] *= amplify_im_end
    flat["model.embeddings.weight"] = mx.array(embed)
    model.load_weights(list(flat.items()), strict=True)
    mx.eval(model)
    return model, flat


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


def ints2d(arr) -> list:
    return [ints(row) for row in np.asarray(arr)]


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def controlled_uniforms(seed: int):
    """Patch mx.random.uniform to draw from a fresh numpy stream: the
    RAS high-temperature resample consumes the global MLX random state,
    so scenario runs would otherwise not be reproducible across calls.
    Every scenario starts its own stream; the drawn arrays are recorded
    in call order for the Rust replay (one call = one full-vocabulary
    array, the reference draw shape)."""
    calls: list[list[float]] = []
    rng = np.random.default_rng(seed)

    def wrapper(*args, **kwargs):
        shape = kwargs.get("shape") or (1,)
        width = 1
        for dim in shape:
            width *= int(dim)
        draw = rng.uniform(1e-6, 1.0, size=width).astype(np.float32)
        calls.append(floats(draw))
        UNIFORM_CALLS.append(calls[-1])
        return mx.array(draw.reshape(shape))

    return calls, mock.patch.object(mx_random, "uniform", wrapper)


def encode_matrix(tokenizer, messages) -> dict:
    matrix = F.Conversation.encode_for_inference(
        F.Conversation(list(messages)), tokenizer, num_codebooks=4
    )
    return {
        "tokens": ints(matrix[0]),
        "codebooks": [ints(matrix[1 + book]) for book in range(4)],
    }


def headers() -> dict:
    """The stub-encoded header/id sets the Rust caller supplies."""
    return {
        "system_header_ids": stub_encode("<|im_start|>system\n"),
        "user_header_ids": stub_encode("<|im_start|>user\n"),
        "assistant_voice_header_ids": stub_encode(
            "<|im_start|>assistant\n<|voice|>"
        ),
        "im_end_ids": stub_encode("<|im_end|>\n"),
        "im_end_token_id": IM_END_ID,
    }


def attach(model, dac):
    model.tokenizer = StubTokenizer()
    model.codec = dac
    vocab_size = max(model.tokenizer.vocab_size, model.config.text_config.vocab_size)
    bias = mx.full((1, vocab_size), -1e9, dtype=mx.float32)
    bias[:, SEMANTIC_BEGIN_ID : SEMANTIC_BEGIN_ID + 16] = 0.0
    bias[:, IM_END_ID] = 0.0
    model.semantic_logit_bias = bias


# Fixed uniform-stream seeds per scenario (paired with the chosen model
# seed below; both are pinned after the search).
MODEL_SEED = None
EOS_STREAM = 901
SAMPLED_STREAM = 902
BUDGET_STREAM = 903


def build_forward_trace(model, with_header, codes_mx) -> dict:
    """Prefill and one decode step through the slow stack: the last-row
    logits and the projected hidden states the fast stack consumes."""
    prompt_input = F.Conversation.encode_for_inference(
        with_header, model.tokenizer, num_codebooks=4
    )
    # The model input is book-major [B, num_codebooks + 1, T].
    inp = np.asarray(prompt_input)[None].astype(np.int32)
    cache = model.model.make_cache()
    result = model.model(mx.array(inp), cache=cache)
    mx.eval(result.logits, result.hidden_states)
    trace = {
        "input": ints(inp[0]),
        "prompt_logits": floats(result.logits[0, -1]),
        "prompt_hidden": floats(result.hidden_states[0, -1]),
    }
    # One decode step on the first eos row, book-major
    # [B, num_codebooks + 1, 1]: [semantic, books...].
    semantic = int(codes_mx[0][0]) + SEMANTIC_BEGIN_ID
    planes = [[semantic]] + [[int(codes_mx[book][0])] for book in range(4)]
    step = np.array([planes], dtype=np.int32)
    step_result = model.model(mx.array(step), cache=cache)
    mx.eval(step_result.logits, step_result.hidden_states)
    trace["step_input"] = ints(step[0])
    trace["step_logits"] = floats(step_result.logits[0, -1])
    trace["step_hidden"] = floats(step_result.hidden_states[0, -1])
    # The fast stack over the PREFILL hidden state (the first generate
    # step's actual input), then over the step hidden state: residual
    # fills feed the fixture codes so the Rust side can compare every
    # fast logits row.
    hidden_prefill = result.hidden_states[:, -1]
    fast_cache = model.model.make_fast_cache()
    fast_prefill = model.model.fast_forward_cached(hidden_prefill, fast_cache)
    mx.eval(fast_prefill)
    trace["prefill_fast_logits"] = floats(fast_prefill[0])
    code0 = int(codes_mx[0][0])
    fh = model.model.fast_embeddings(mx.array([code0]))
    rl = model.model.fast_forward_cached(fh, fast_cache)
    mx.eval(rl)
    trace["prefill_residual0_logits"] = floats(rl[0])
    trace["prefill_semantic_code"] = code0
    hidden_step = step_result.hidden_states[:, -1]
    fast_cache = model.model.make_fast_cache()
    fast_prefill = model.model.fast_forward_cached(hidden_step, fast_cache)
    mx.eval(fast_prefill)
    trace["step_fast_prefill_logits"] = floats(fast_prefill[0])
    residuals = []
    code = int(codes_mx[0][0])
    fast_hidden = model.model.fast_embeddings(mx.array([code]))
    for book in range(1, 4):
        rl = model.model.fast_forward_cached(fast_hidden, fast_cache)
        mx.eval(rl)
        residuals.append(floats(rl[0]))
        code = int(codes_mx[book][0])
        fast_hidden = model.model.fast_embeddings(mx.array([code]))
    trace["residual_logits"] = residuals
    trace["semantic_code"] = int(codes_mx[0][0])
    return trace


def build_generate_case(dac):
    texts = ["hello friend", "one two three"]
    stub = StubTokenizer()
    token_counts = [len(stub.encode(text)) for text in texts]

    # eos scenario search: the greedy draw repeats the amplified
    # semantic token, so the RAS high-temperature resample fires every
    # step until the amplified im_end token falls out of a draw. The
    # accepted run must stop on im_end strictly before the 12-step
    # budget. Both the model seed and the uniform-stream seed are
    # searched: every greedy argmax needs a comfortable logit gap and
    # every sampled draw a comfortable Gumbel-score ratio so the Rust
    # f32 port cannot flip a decision.
    chosen = None
    for seed in range(40):
        model, flat = build_model(seed, amplify_semantic=60.0, amplify_im_end=10.0)
        attach(model, dac)
        base = model._build_conversation([], [], instruct=None)
        conversation = F.Conversation(list(base.messages))
        conversation.append(
            F.Message(
                role="user",
                parts=[F.TextPart(texts[0])],
                add_im_start=True,
                add_im_end=True,
            )
        )
        with_header = F.Conversation(list(conversation.messages))
        with_header.append(
            F.Message(
                role="assistant",
                parts=[],
                modality="voice",
                add_im_start=True,
                add_im_end=False,
            )
        )
        for stream_offset in range(40):
            draws, patch = controlled_uniforms(EOS_STREAM + stream_offset)
            MARGINS.clear()
            SAMPLED_MARGINS.clear()
            codes = None
            try:
                with patch:
                    codes = model._generate_codes_for_batch(
                        conversation=conversation,
                        batch_text=texts[0],
                        max_new_tokens=12,
                        top_p=0.7,
                        top_k=30,
                        temperature=0.0,
                    )
                mx.eval(codes)
                steps = int(codes.shape[1])
                wave = floats(model._decode_codes(codes))
            except RuntimeError:
                continue
            if codes is None:
                continue
            mx.eval(mx.array(wave))
            if steps == 0 or len(wave) != steps * 32:
                continue
            greedy_ok = min(MARGINS) > 3e-2 if MARGINS else True
            sampled_ok = min(SAMPLED_MARGINS) > 5.0 if SAMPLED_MARGINS else True
            # The stop must be im_end (strictly before budget) with at
            # least one generated step, and every decision must hold a
            # comfortable margin against cross-engine accumulation noise.
            if 2 <= steps <= 10 and greedy_ok and sampled_ok:
                # Re-run the accepted scenario on a fresh stream with the
                # same seed and require identical codes: the recorded
                # fixture must be one self-consistent run.
                verify_draws, verify_patch = controlled_uniforms(
                    EOS_STREAM + stream_offset
                )
                try:
                    with verify_patch:
                        verify_codes = model._generate_codes_for_batch(
                            conversation=conversation,
                            batch_text=texts[0],
                            max_new_tokens=12,
                            top_p=0.7,
                            top_k=30,
                            temperature=0.0,
                        )
                    mx.eval(verify_codes)
                except RuntimeError:
                    continue
                if ints2d(verify_codes) != ints2d(codes):
                    print(f"stream {EOS_STREAM + stream_offset}: rerun disagrees, skip")
                    continue
                verify_wave = floats(model._decode_codes(verify_codes))
                mx.eval(mx.array(verify_wave))
                chosen = (
                    seed,
                    EOS_STREAM + stream_offset,
                    model,
                    flat,
                    with_header,
                    verify_codes,
                    verify_wave,
                    verify_draws,
                    steps,
                )
                print(
                    f"generate model seed {seed} stream {EOS_STREAM + stream_offset}: "
                    f"eos after {steps} steps"
                )
                break
        if chosen is not None:
            break
    if chosen is None:
        raise SystemExit("no seed produced an acceptable eos scenario")
    (
        seed,
        eos_stream,
        model,
        flat,
        with_header,
        eos_codes_mx,
        eos_wave,
        eos_draws,
        eos_steps,
    ) = chosen

    global MODEL_SEED
    MODEL_SEED = seed

    eos_codes = ints2d(eos_codes_mx)

    # Prompt matrices, assembled exactly as the reference generate does.
    base = model._build_conversation([], [], instruct=None)
    conversation = F.Conversation(list(base.messages))
    conversation.append(
        F.Message(
            role="user",
            parts=[F.TextPart(texts[0])],
            add_im_start=True,
            add_im_end=True,
        )
    )
    first_prompt_matrix = encode_matrix(model.tokenizer, conversation.messages)
    assistant_prompt_matrix = encode_matrix(model.tokenizer, with_header.messages)

    # Sampled scenario: temperature 0.8 over 6 steps; its stream seed is
    # searched for comfortable draw margins like the eos scenario.
    MARGINS.clear()
    SAMPLED_MARGINS.clear()
    sampled_draws = None
    sampled_stream = None
    for offset in range(200):
        draws, patch = controlled_uniforms(SAMPLED_STREAM + offset)
        MARGINS.clear()
        SAMPLED_MARGINS.clear()
        try:
            with patch:
                sampled_mx = model._generate_codes_for_batch(
                    conversation=conversation,
                    batch_text=texts[0],
                    max_new_tokens=6,
                    top_p=0.7,
                    top_k=30,
                    temperature=0.8,
                )
            mx.eval(sampled_mx)
            sampled_wave = floats(model._decode_codes(sampled_mx))
        except RuntimeError:
            continue
        mx.eval(mx.array(sampled_wave))
        greedy_ok = not MARGINS or min(MARGINS) > 3e-2
        sampled_ok = not SAMPLED_MARGINS or min(SAMPLED_MARGINS) > 3.0
        if greedy_ok and sampled_ok:
            sampled_draws = draws
            sampled_stream = SAMPLED_STREAM + offset
            break
    assert sampled_draws is not None, "no sampled stream produced robust draws"
    sampled_codes = ints2d(sampled_mx)
    sampled_steps = int(sampled_mx.shape[1])
    assert 1 <= sampled_steps <= 6

    # Budget scenario: the min(max_tokens, ...) budget ends the loop;
    # its stream seed is searched in case RAS resamples fire.
    budget_draws = None
    budget_stream = None
    for offset in range(200):
        MARGINS.clear()
        SAMPLED_MARGINS.clear()
        draws, patch = controlled_uniforms(BUDGET_STREAM + offset)
        try:
            with patch:
                budget_mx = model._generate_codes_for_batch(
                    conversation=conversation,
                    batch_text=texts[0],
                    max_new_tokens=2,
                    top_p=0.7,
                    top_k=30,
                    temperature=0.0,
                )
            mx.eval(budget_mx)
            budget_wave = floats(model._decode_codes(budget_mx))
        except RuntimeError:
            continue
        mx.eval(mx.array(budget_wave))
        greedy_ok = min(MARGINS) > 3e-2 if MARGINS else True
        sampled_ok = not SAMPLED_MARGINS or min(SAMPLED_MARGINS) > 3.0
        if greedy_ok and sampled_ok:
            budget_draws = draws
            budget_stream = BUDGET_STREAM + offset
            break
    assert budget_draws is not None, "no budget stream produced a robust run"
    budget_codes = ints2d(budget_mx)
    assert budget_codes and len(budget_codes[0]) == 2

    # The second prompt carries the first assistant VQ reply.
    conversation_after = F.Conversation(list(base.messages))
    conversation_after.append(
        F.Message(
            role="user",
            parts=[F.TextPart(texts[0])],
            add_im_start=True,
            add_im_end=True,
        )
    )
    conversation_after.append(
        F.Message(
            role="assistant",
            parts=[F.VQPart(eos_codes_mx)],
            modality="voice",
            add_im_start=True,
            add_im_end=True,
        )
    )
    conversation_after.append(
        F.Message(
            role="user",
            parts=[F.TextPart(texts[1])],
            add_im_start=True,
            add_im_end=True,
        )
    )
    second_prompt_matrix = encode_matrix(model.tokenizer, conversation_after.messages)

    case_dir = OUT / "generate"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), flat)
    config = tiny_config()
    config_dict = {
        "model_type": config.model_type,
        "pad_token_id": config.pad_token_id,
        "eos_token_id": config.eos_token_id,
        "audio_pad_token_id": config.audio_pad_token_id,
        "semantic_start_token_id": config.semantic_start_token_id,
        "semantic_end_token_id": config.semantic_end_token_id,
        "sample_rate": config.sample_rate,
        "text_config": dict(vars(config.text_config)),
        "audio_decoder_config": dict(vars(config.audio_decoder_config)),
    }
    (case_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")

    traces = {
        "batch_texts": texts,
        "batch_text_ids": [stub_encode(text) for text in texts],
        "batch_token_counts": token_counts,
        "model_seed": seed,
        "uniform_stream_seeds": {
            "eos": eos_stream,
            "sampled": sampled_stream,
            "budget": budget_stream,
        },
        "forward": build_forward_trace(model, with_header, eos_codes_mx),
        "headers": headers(),
        "semantic_logit_bias": {
            "vocab_size": max(stub.vocab_size, 300),
            "semantic_start": SEMANTIC_BEGIN_ID,
            "semantic_end": SEMANTIC_BEGIN_ID + 15,
            "im_end_id": IM_END_ID,
        },
        "first_prompt_matrix": first_prompt_matrix,
        "assistant_prompt_matrix": assistant_prompt_matrix,
        "second_prompt_matrix": second_prompt_matrix,
        "scenarios": {
            "eos": {
                "max_tokens": 12,
                "temperature": 0.0,
                "steps": eos_steps,
                "codes": eos_codes,
                "uniform_calls": eos_draws,
                "waveform": eos_wave,
            },
            "sampled": {
                "max_tokens": 6,
                "temperature": 0.8,
                "steps": sampled_steps,
                "codes": sampled_codes,
                "uniform_calls": sampled_draws,
                "waveform": sampled_wave,
            },
            "budget": {
                "max_tokens": 2,
                "temperature": 0.0,
                "steps": 2,
                "codes": budget_codes,
                "uniform_calls": budget_draws,
                "waveform": budget_wave,
            },
        },
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))


def build_sampling_case():
    rng = np.random.default_rng(910)
    cases = []
    for index, (name, temp, top_p, top_k) in enumerate(
        (
            ("greedy", 0.0, 0.7, 30),
            ("sampled", 0.8, 0.7, 30),
            ("wide_top_p", 0.9, 0.95, 0),
            ("tight_top_k", 1.1, 1.0, 3),
        )
    ):
        logits = rng.standard_normal(48).astype(np.float32) * 2.0
        draws, patch = controlled_uniforms(950 + index)
        with patch:
            token = F._sample_logits(
                mx.array(logits)[None, :],
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
            )
            mx.eval(token)
        cases.append(
            {
                "name": name,
                "logits": floats(logits),
                "temperature": temp,
                "top_p": top_p,
                "top_k": top_k,
                "uniform_calls": draws,
                "token": int(token.item()),
            }
        )

    # RAS decisions on fixed logits and histories.
    vocab = 300
    logits = rng.standard_normal(vocab).astype(np.float32) * 2.0
    bias = np.full(vocab, -1e9, dtype=np.float32)
    bias[SEMANTIC_BEGIN_ID : SEMANTIC_BEGIN_ID + 16] = 0.0
    bias[IM_END_ID] = 0.0
    biased = logits + bias
    normal = F._sample_logits(
        mx.array(biased)[None, :], temperature=0.0, top_p=0.7, top_k=30
    )
    mx.eval(normal)
    argmax_token = int(normal[0].item())
    ras_cases = []
    # The repeat case pins the history to the in-range argmax token so
    # the RAS branch deterministically fires.
    for index, (name, previous) in enumerate(
        (
            ("repeat_triggers_high", [argmax_token]),
            ("no_repeat_stays_normal", []),
            ("outside_range_stays_normal", [5]),
        )
    ):
        should_use_high = (
            argmax_token in previous
            and SEMANTIC_BEGIN_ID <= argmax_token <= SEMANTIC_BEGIN_ID + 15
        )
        draws, patch = controlled_uniforms(970 + index)
        if should_use_high:
            with patch:
                high = F._sample_logits(
                    mx.array(biased)[None, :],
                    temperature=1.0,
                    top_p=0.9,
                    top_k=30,
                )
                mx.eval(high)
            final = int(high[0].item())
        else:
            final = argmax_token
        ras_cases.append(
            {
                "name": name,
                "logits": floats(logits),
                "previous": previous,
                "temperature": 0.0,
                "normal_token": argmax_token,
                "should_use_high": should_use_high,
                "final_token": final,
                "uniform_calls": draws,
            }
        )
    case_dir = OUT / "sampling"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(
        json.dumps({"cases": cases, "ras_cases": ras_cases})
    )


def build_prompt_case(tokenizer):
    """Conversation encoding with VQ parts."""
    codes = np.array([[3, 7, 2], [5, 1, 4], [9, 0, 6], [8, 2, 1]], dtype=np.int32)
    messages = [
        F.Message(
            role="system",
            parts=[
                F.TextPart("convert the provided text to speech"),
                F.TextPart("<|speaker:0|>hello"),
                F.TextPart("\n\nSpeech:\n"),
                F.VQPart(codes),
            ],
            add_im_start=True,
            add_im_end=True,
        ),
        F.Message(
            role="user",
            parts=[F.TextPart("hi there")],
            add_im_start=True,
            add_im_end=True,
        ),
        F.Message(
            role="assistant",
            parts=[],
            modality="voice",
            add_im_start=True,
            add_im_end=False,
        ),
    ]
    matrix = encode_matrix(tokenizer, messages)
    case_dir = OUT / "prompt"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(
        json.dumps(
            {
                "matrix": matrix,
                "semantic_begin_id": SEMANTIC_BEGIN_ID,
                "vq_codes": ints2d(codes),
            }
        )
    )


def build_speed_case():
    wave = (np.random.default_rng(77).standard_normal(100) * 0.2).astype(np.float32)
    cases = []
    for speed in (1.0, 0.8, 1.5):
        out = F._adjust_speed(mx.array(wave), speed)
        mx.eval(out)
        cases.append(
            {
                "speed": speed,
                "input": floats(wave),
                "output": floats(out),
            }
        )
    case_dir = OUT / "speed"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "traces.json").write_text(json.dumps({"cases": cases}))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dac = tiny_dac()
    stub = StubTokenizer()

    build_generate_case(dac)
    build_sampling_case()
    build_prompt_case(stub)
    build_speed_case()

    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_fish_qwen3_omni_fixtures.py",
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
