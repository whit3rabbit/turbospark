#!/usr/bin/env python3
"""Regenerate the bailingmm (Ming Omni TTS) family Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/tts/models/bailingmm, mlx_audio/tts/models/dense,
mlx_audio/lm/models/bailing_moe, mlx 0.32.3). The reference modules run
in full: the Ming bailing MoE (or dense Qwen2) LLM backbone read for
hidden states, the AudioVAE (Ming Qwen2 encoder over frame windows and
decoder with the streaming ISTFT head), the DiT flow-matching head with
the CFM solver (EPSS timesteps, sway sampling, CFG guide), the DiT
aggregator, the two-class stop head, and the reference
`Model.sample`/`Model.generate` loops, with:

- a deterministic offline mock tokenizer (`encode(text,
  add_special_tokens=False)` word ids plus fixed special-token ids)
  standing in for the checkpoint tokenizer, injected as
  `model.tokenizer`;
- seeded tiny stacks: a 3-layer bailing MoE (4 experts top-2, shared
  expert, q/k norms, untied head) or a 2-layer dense Qwen2 (tied head)
  as the LLM; 2-layer Ming Qwen2 audio encoder/decoder backbones
  (the dense case's decoder carries a sliding window on its second
  layer); a 2-block DiT for the flow head and the aggregator;
- every `mx.random.normal` draw recorded (in reference draw order:
  the flow noise per step, then the solver noise per integration step)
  and replay-verified across the two runs of each case.

`Model.sample` is driven directly for the per-step latents and stop
decisions, the reference generate decode loop body
(`audio.decode` with the cache and stream state) is replayed for the
per-step chunks, and `Model.generate` produces the final waveform the
concatenated chunks must equal.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_bailingmm_fixtures.py

Outputs into crates/audio/testdata/tts_bailingmm/:

- moe_generate/ (stop head fires at step 4), moe_length/ (never fires,
  max steps), dense_generate/ (dense variant, sliding decoder window),
  voice_clone/ (MoE with a reference-audio prompt latent)
    Each: config.json + model.safetensors + traces.json. Traces: the
    assembled prompt ids and input embeddings, the flow timestep grid,
    per-step latent patches and stop decisions, the per-step decoded
    chunks (empty first chunk included), the joined waveform, the noise
    draws, and for the voice-clone case the padded reference audio,
    encoder latent, and projected patch embeddings.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

sys.path.insert(0, str(Path.cwd()))

import mlx_audio.tts.models.bailingmm.bailingmm as bailingmm
import mlx_audio.tts.models.dense.dense as dense_mod
from mlx_audio.tts.models.bailingmm.bailingmm import get_epss_timesteps

OUT = Path("../turbospark/crates/audio/testdata/tts_bailingmm")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

MOCK_IDS = {
    "<|endoftext|>": 3,
    "<audio>": 1,
    "<audioPatch>": 2,
}

MOE_LLM = {
    "model_type": "bailing_moe",
    "hidden_size": 32,
    "intermediate_size": 64,
    "max_position_embeddings": 512,
    "moe_intermediate_size": 32,
    "num_experts": 4,
    "num_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 3,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "vocab_size": 96,
    "first_k_dense_replace": 1,
    "use_qk_norm": True,
    "tie_word_embeddings": False,
}

DENSE_LLM = {
    "model_type": "qwen2",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 96,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}

AUDIO_BACKBONE = {
    "model_type": "qwen2",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 8,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
}

DITAR = {
    "patch_size": 2,
    "history_patch_size": 2,
    "hidden_size": 32,
    "depth": 2,
    "num_heads": 4,
    "mlp_ratio": 2.0,
}

AGGREGATOR = {
    "hidden_size": 32,
    "depth": 2,
    "num_heads": 4,
    "mlp_ratio": 2.0,
}

# The reference audio: exactly one alignment unit
# ((1 / 12.5) * patch_size * sample_rate = 3840 samples), so the frames
# divide evenly into patches.
SAMPLE_RATE = 24000
REF_SAMPLES = (1 / 12.5) * DITAR["patch_size"] * SAMPLE_RATE

PROMPT_TEXT = "Please generate speech based on the following description.\n"
SPEECH_TEXT = "hello world"


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def ints(arr) -> list:
    return [int(v) for v in np.asarray(arr).ravel().tolist()]


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


class MockTokenizer:
    """Deterministic word-id tokenizer with fixed special tokens."""

    def __init__(self):
        self.vocab = {}

    def _id(self, word: str) -> int:
        if word in MOCK_IDS:
            return MOCK_IDS[word]
        if word not in self.vocab:
            self.vocab[word] = 4 + 2 * len(self.vocab)
        return self.vocab[word]

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        del add_special_tokens
        if text in MOCK_IDS:
            return [MOCK_IDS[text]]
        return [self._id(word) for word in text.split(" ")]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._id(token)


def make_config(llm: dict, model_type: str, sliding_decoder: bool) -> dict:
    dec_backbone = dict(AUDIO_BACKBONE)
    if sliding_decoder:
        dec_backbone.update(
            use_sliding_window=True, sliding_window=4, max_window_layers=1
        )
    return {
        "model_type": model_type,
        "llm_config": dict(llm),
        "audio_tokenizer_config": {
            "sample_rate": SAMPLE_RATE,
            "patch_size": DITAR["patch_size"],
            "enc_kwargs": {
                "backbone": dict(AUDIO_BACKBONE),
                "input_dim": 64,
                "hop_size": 64,
                "latent_dim": 8,
            },
            "dec_kwargs": {
                "backbone": dec_backbone,
                "output_dim": 16,
                "latent_dim": 8,
            },
        },
        "ditar_config": dict(DITAR),
        "aggregator_config": dict(AGGREGATOR),
    }


def seeded_model(model_cls, config: dict, seed: int):
    """Deterministic weights: small uniform everywhere, norms biased up,
    the stop head zeroed and re-biased per case by the caller."""
    mx.random.seed(seed)
    model = model_cls(dict(config))

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith(("norm.weight", "layernorm.weight", "0.weight", "norm_final.weight")):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    # The reference cls_embed is a plain random array that never reaches
    # parameters() (so a checkpoint save drops it); pin it as a fixed
    # parameter so the fixture carries exactly the value the traces used.
    flat["audio.encoder.cls_embed"] = mx.zeros(
        (1, config["audio_tokenizer_config"]["enc_kwargs"]["backbone"]["hidden_size"])
    )
    model.update(tree_unflatten(flat))
    model.tokenizer = MockTokenizer()
    return model


def rebias_stop_head(model, stop_bias) -> None:
    flat = dict(tree_flatten(model.parameters()))
    flat["stop_head.weight"] = mx.zeros(flat["stop_head.weight"].shape)
    flat["stop_head.bias"] = mx.array(stop_bias, dtype=mx.float32)
    model.update(tree_unflatten(flat))


def reference_audio() -> np.ndarray:
    rng = np.random.default_rng(97)
    t = np.arange(int(REF_SAMPLES), dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * rng.standard_normal(len(t))).astype(
        np.float32
    )


def save_case(name: str, model, config: dict) -> None:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(case_dir / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def run_case(name: str, model, config: dict, ref_audio=None, ref_text=None) -> None:
    save_case(name, model, config)
    generate_kwargs = dict(
        prompt=PROMPT_TEXT,
        text=SPEECH_TEXT,
        cfg_scale=2.0,
        sigma=0.25,
        temperature=0.0,
        ddpm_steps=5,
        max_tokens=12,
    )
    if ref_audio is not None:
        generate_kwargs.update(ref_audio=mx.array(ref_audio), ref_text=ref_text)

    # Run 1: the reference sample() generator for the per-step latents
    # and stop decisions, with the flow stack instrumented to record the
    # per-step (z, latent_history, latent) triples and the first raw DiT
    # guide calls for stage-level Rust gates.
    flow_steps_trace = []
    dit_calls_trace = []
    encoder_stats = []
    original_encoder_call = bailingmm.Encoder.__call__
    original_flow_sample = model.flowloss.sample
    original_fwd_with_cfg = model.flowloss.cfm.model.forward_with_cfg

    def recorded_encoder_call(self, waveform):
        out = original_encoder_call(self, waveform)
        encoder_stats.append(floats(out))
        return out

    bailingmm.Encoder.__call__ = recorded_encoder_call

    def recorded_flow_sample(z, latent_history, **kwargs):
        sampled, trajectory = original_flow_sample(z, latent_history, **kwargs)
        flow_steps_trace.append(
            {
                "z": floats(z),
                "latent_history": floats(latent_history),
                "latent": floats(sampled),
            }
        )
        return sampled, trajectory

    block_records = []
    full_records = []
    original_block_call = bailingmm.DiTBlock.__call__
    original_dit_call = bailingmm.DiT.__call__

    def recorded_block_call(self, x, mask=None, rope=None):
        out = original_block_call(self, x, mask, rope)
        block_records.append({"input": floats(x), "out": floats(out)})
        return out

    def recorded_dit_call(self, x, t, c, latent_history, mask=None):
        out = original_dit_call(self, x, t, c, latent_history, mask)
        full_records.append(floats(out))
        return out

    bailingmm.DiTBlock.__call__ = recorded_block_call
    bailingmm.DiT.__call__ = recorded_dit_call

    def recorded_fwd_with_cfg(x, t, c, cfg_scale, latent_history, patch_size):
        block_records.clear()
        full_records.clear()
        out = original_fwd_with_cfg(x, t, c, cfg_scale, latent_history, patch_size)
        if len(dit_calls_trace) < 2:
            # The reference forward_with_cfg makes exactly one full DiT
            # call (batch doubled) with the same block count.
            assert len(full_records) == 1 and len(block_records) == 2, (
                len(full_records), len(block_records)
            )
            dit_calls_trace.append(
                {
                    "x": floats(x),
                    "t": float(t),
                    "cfg": float(cfg_scale),
                    "latent_history": floats(latent_history),
                    "inner_out": full_records[0],
                    "blocks": list(block_records),
                    "out": floats(out),
                }
            )
        return out

    model.flowloss.sample = recorded_flow_sample
    model.flowloss.cfm.model.forward_with_cfg = recorded_fwd_with_cfg
    with DrawRecorder() as recorder:
        mx.random.seed(101)
        steps = list(
            model.sample(
                prompt=PROMPT_TEXT,
                text=SPEECH_TEXT,
                prompt_waveform=None if ref_audio is None else mx.array(ref_audio),
                prompt_text=ref_text,
                max_decode_steps=12,
                cfg=2.0,
                sigma=0.25,
                temperature=0.0,
                flow_steps=5,
            )
        )
        noise_draws = recorder.flush()
    model.flowloss.sample = original_flow_sample
    model.flowloss.cfm.model.forward_with_cfg = original_fwd_with_cfg
    bailingmm.Encoder.__call__ = original_encoder_call

    # Run 2: the reference decode loop body verbatim for the chunks,
    # with the ISTFT head instrumented (input hidden rows, streaming
    # flags, output samples).
    head_calls = []
    original_head_call = bailingmm.ISTFTHead.__call__

    def recorded_head_call(self, x, audio_buffer=None, window_buffer=None, streaming=False, last_chunk=False):
        out = original_head_call(
            self, x, audio_buffer=audio_buffer, window_buffer=window_buffer,
            streaming=streaming, last_chunk=last_chunk,
        )
        if streaming:
            head_calls.append(
                {
                    "input": floats(x),
                    "projected": floats(head_projection_sink[-1]) if head_projection_sink else None,
                    "streaming": True,
                    "last_chunk": bool(last_chunk),
                    "out": floats(out[0]),
                }
            )
        return out

    class RecordingLinear:
        def __init__(self, inner, sink):
            self.inner = inner
            self.sink = sink

        def __call__(self, x):
            out = self.inner(x)
            self.sink.append(np.array(out, dtype=np.float32))
            return out

    head_projection_sink = []
    model.audio.decoder.head.out = RecordingLinear(
        model.audio.decoder.head.out, head_projection_sink
    )

    bailingmm.ISTFTHead.__call__ = recorded_head_call
    with DrawRecorder() as recorder:
        mx.random.seed(101)
        speech_chunks = []
        stream_state = (None, None, None)
        past_key_values = None
        for sampled_tokens, last_chunk in steps:
            speech_tmp, stream_state, past_key_values = model.audio.decode(
                sampled_tokens,
                past_key_values=past_key_values,
                use_cache=True,
                stream_state=stream_state,
                last_chunk=last_chunk,
            )
            speech_chunks.append(np.array(speech_tmp, dtype=np.float32))
            if last_chunk:
                break
        decode_draws = recorder.flush()
    bailingmm.ISTFTHead.__call__ = original_head_call
    # The decode consumes no randomness at all.
    assert decode_draws == [], "the decode loop drew random numbers"
    joined = np.concatenate(speech_chunks, axis=1)[0]

    # Run 3: the full reference generate() (final waveform gate).
    with DrawRecorder() as recorder:
        mx.random.seed(101)
        result = list(model.generate(**generate_kwargs))[0]
        generate_draws = recorder.flush()
    assert generate_draws == noise_draws, "generate draw replay diverged"
    assert np.allclose(joined, np.array(result.audio), atol=0.0, rtol=0.0), (
        "the decode loop and generate disagree"
    )

    # The assembled prompt (ids + embeddings) from the reference
    # _prepare_input_embed (deterministic; no draws involved).
    prompt_latent = None
    if ref_audio is not None:
        padded = model._pad_prompt_waveform(mx.array(ref_audio))
        # Re-seed to the run-1 state so this encode's draw is exactly the
        # recorded first draw (the per-step z values used the same latent).
        mx.random.seed(101)
        prompt_latent, _ = model.audio.encode_latent(
            padded, mx.array([int(padded.shape[1])], dtype=mx.int32)
        )
        # The per-step z values in run 1 used exactly this latent (its
        # draw is the recorded first draw, asserted above by the replay).
        assert noise_draws[0][0] == [
            1,
            int(model.latent_dim),
            int(prompt_latent.shape[1]),
        ], noise_draws[0][0]
        input_ids, inputs_embeds = model._prepare_input_embed(
            prompt=PROMPT_TEXT,
            text=SPEECH_TEXT,
            prompt_latent=prompt_latent,
            prompt_text=ref_text,
        )
    else:
        input_ids, inputs_embeds = model._prepare_input_embed(
            prompt=PROMPT_TEXT, text=SPEECH_TEXT
        )

    flow_grid = get_epss_timesteps(5, mx.float32)
    sway = flow_grid + (-1.0) * (mx.cos(np.pi / 2 * flow_grid) - 1 + flow_grid)

    tok = model.tokenizer
    traces = {
        "prompt": {
            "prompt_string": PROMPT_TEXT,
            "text": SPEECH_TEXT,
            "input_ids": ints(input_ids),
            "input_embeds": floats(inputs_embeds),
            # The patch-TOKEN count (the latent rows divided by the
            # ditar patch size), which is what the prompt carries.
            "num_patches": 0
            if prompt_latent is None
            else int(prompt_latent.shape[1]) // int(DITAR["patch_size"]),
            "segments": {
                "role_human": ints(tok.encode("<role>HUMAN</role>")),
                "text_input_header": ints(tok.encode(" Text input:\n")),
                "role_assistant": ints(tok.encode("<role>ASSISTANT</role>")),
                "prompt": ints(tok.encode(PROMPT_TEXT)),
                "text": ints(tok.encode(SPEECH_TEXT)),
                "audio_token": int(tok.convert_tokens_to_ids("<audio>")),
                "patch_token": int(tok.convert_tokens_to_ids("<audioPatch>")),
                "end_of_text": int(tok.convert_tokens_to_ids("<|endoftext|>")),
            },
        },
        "flow": {
            "steps": 5,
            "cfg": 2.0,
            "sigma": 0.25,
            "temperature": 0.0,
            "t_grid": floats(flow_grid),
            "sway_grid": floats(sway),
        },
        "noise_draws": [
            {"shape": shape, "values": values} for shape, values in noise_draws
        ],
        "steps": [
            {
                "index": index,
                "latent": floats(sampled_tokens),
                "is_last": bool(last_chunk),
            }
            for index, (sampled_tokens, last_chunk) in enumerate(steps)
        ],
        "flow_steps": flow_steps_trace,
        "dit_calls": dit_calls_trace,
        "chunks": [floats(chunk) for chunk in speech_chunks],
        "head_calls": head_calls,
        "waveform": floats(joined),
        "sample_rate": int(model.sample_rate),
    }
    if ref_audio is not None:
        traces["voice_clone"] = {
            "ref_audio": floats(ref_audio),
            "prompt_latent": floats(prompt_latent),
            # The encoder's [frames, 2 * latent] mean/logvar rows (the
            # pre-noise statistics; one call).
            "encoder_stats": encoder_stats[0],
        }
    (OUT / name / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(
        f"{name}: {len(steps)} steps, stopped "
        f"{'stop_head' if steps[-1][1] else 'max_steps'}, "
        f"{len(joined)} samples"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # MoE, stop head fires at step 4 (needs step > 3 first).
    model = seeded_model(bailingmm.Model, make_config(MOE_LLM, "ming_omni_tts", False), 7)
    rebias_stop_head(model, [0.0, 10.0])
    run_case("moe_generate", model, make_config(MOE_LLM, "ming_omni_tts", False))

    # MoE, stop head silenced, stops at max_decode_steps.
    model = seeded_model(bailingmm.Model, make_config(MOE_LLM, "ming_omni_tts", False), 7)
    rebias_stop_head(model, [10.0, 0.0])
    run_case("moe_length", model, make_config(MOE_LLM, "ming_omni_tts", False))

    # Dense variant (the dense/ directory): Qwen2 backbone, tied head,
    # sliding window on the decoder's second layer.
    model = seeded_model(dense_mod.Model, make_config(DENSE_LLM, "dense", True), 7)
    rebias_stop_head(model, [0.0, 10.0])
    run_case("dense_generate", model, make_config(DENSE_LLM, "dense", True))

    # Voice clone: MoE with a reference-audio prompt latent.
    model = seeded_model(bailingmm.Model, make_config(MOE_LLM, "ming_omni_tts", False), 7)
    rebias_stop_head(model, [0.0, 10.0])
    run_case(
        "voice_clone",
        model,
        make_config(MOE_LLM, "ming_omni_tts", False),
        ref_audio=reference_audio(),
        ref_text="hello world",
    )

    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "generator": "tools/gen_bailingmm_fixtures.py",
        "files": {
            str(path.relative_to(OUT)): "sha256:"
            + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures under {OUT}")


if __name__ == "__main__":
    main()
