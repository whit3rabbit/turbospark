#!/usr/bin/env python3
"""Regenerate the shared LM backbone Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/lm, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_lm_fixtures.py

Outputs into crates/audio/testdata/lm/:

- <case>.safetensors
    Seeded tiny backbone weights in the full-model (model.) layout, plus
    untied lm_head cases.
- <case>.json
    Input tokens, the prefill logits, the per-step decode logits with a
    prompt cache, and the final hidden state after the backbone norm.
- manifest.json
    Case list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. Every rope
construction initialize_rope supports is exercised through a llama case
except mrope, which the reference reduces to the plain path (pinned by
comparing the llama3 fixture input against the plain construction).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_audio.lm.models.llama import Model as LlamaModel, ModelArgs as LlamaArgs
from mlx_audio.lm.models.qwen2 import Model as Qwen2Model, ModelArgs as Qwen2Args
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen3 import Model as Qwen3Model, ModelArgs as Qwen3Args
from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.gpt2 import Model as Gpt2Model, ModelArgs as Gpt2Args
from mlx_audio.lm.models.cohere2 import Model as Cohere2Model, ModelArgs as Cohere2Args
from mlx_audio.lm.models.bailing_moe import (
    Model as BailingMoeModel,
    ModelArgs as BailingMoeArgs,
)

OUT = Path("../turbospark/crates/audio/testdata/lm")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

TOKENS = [3, 17, 5, 42, 88, 1, 60]
DECODE_TOKENS = [11, 27, 4]


def tiny_llama_args(**overrides):
    fields = dict(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=96,
        head_dim=None,
        max_position_embeddings=64,
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
    fields.update(overrides)
    return LlamaArgs(**fields)


def seeded_parameters(model):
    """Deterministic weights: small uniform everywhere, norms biased up so
    activations stay in a healthy f32 range."""
    mx.random.seed(7)

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        # The q_norm/k_norm suffixes cover qwen3; query/key_layernorm cover
        # bailing_moe; the ln_* names cover gpt2. None of these collide
        # with the earlier cases' names.
        if name.endswith(
            (
                "norm.weight",
                "q_norm.weight",
                "k_norm.weight",
                "query_layernorm.weight",
                "key_layernorm.weight",
                "ln_1.weight",
                "ln_2.weight",
                "ln_f.weight",
            )
        ):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    model.update(tree_unflatten(flat))
    return model


def run_case(name, model, tie_head, num_layers, args, make_cache=None, chunk_tokens=None):
    """Runs prefill + 3 cached decode steps and records every output.

    `chunk_tokens` optionally adds a second multi-position call between
    prefill and the decode steps (the cohere2 streaming shape: a chunk
    arriving after the sliding window is already full)."""
    weights = dict(tree_flatten(model.parameters()))
    if tie_head:
        weights.pop("lm_head.weight", None)

    tokens = mx.array(TOKENS)[None]
    cache = (
        [KVCache() for _ in range(num_layers)]
        if make_cache is None
        else make_cache()
    )
    logits = model(tokens, cache=cache)
    prefill = np.array(logits[0]).astype(np.float32)

    chunk_logits = None
    if chunk_tokens is not None:
        chunk_logits = np.array(
            model(mx.array(chunk_tokens)[None], cache=cache)[0]
        ).astype(np.float32)

    steps = []
    for token in DECODE_TOKENS:
        step_logits = model(mx.array([token])[None], cache=cache)
        steps.append(np.array(step_logits[0, -1]).astype(np.float32))

    # A cacheless pass pins the no-cache attention path; the hidden state
    # comes from the same pass and pins the post-backbone norm.
    cacheless_hidden = model.model(mx.array(TOKENS)[None])
    cacheless = model(mx.array(TOKENS)[None], cache=None)

    return {
        "name": name,
        "tie_head": tie_head,
        "tokens": TOKENS,
        "decode_tokens": DECODE_TOKENS,
        "prefill_logits": prefill.tolist(),
        "decode_logits": [step.tolist() for step in steps],
        "chunk_tokens": chunk_tokens,
        "chunk_logits": None if chunk_logits is None else chunk_logits.tolist(),
        "cacheless_prefill_logits": np.array(cacheless[0]).astype(np.float32).tolist(),
        "hidden_state": np.array(cacheless_hidden[0]).astype(np.float32).tolist(),
        "weights": weights,
        "config": asdict(args),
    }


def indextts_gpt2_forward(gpt, embeddings, cache=None):
    """The `mlx_audio/tts/models/indextts` GPT2Model loop: `wte` is the
    identity (embeddings in), no position add (`wpe` bypassed), causal
    mask from the first cache, blocks, `ln_f`."""
    hidden = embeddings
    if cache is None:
        cache = [None] * len(gpt.h)
    mask = create_attention_mask(hidden, cache[0])
    for layer, c in zip(gpt.h, cache):
        hidden = layer(hidden, mask, cache=c)
    return gpt.ln_f(hidden)


def run_embeddings_case(name, model, num_layers, args):
    """The indextts consumption shape: `wte`/`wpe` bypassed (the family
    patches both out and feeds precomputed embeddings), the backbone driven
    by float embeddings, and the tied head applied from the captured `wte`
    table. Mirrors the Rust `Input::Embeddings` path with the position
    table dropped."""
    dim = args.n_embd
    weights = dict(tree_flatten(model.parameters()))
    wte_weight = weights["model.wte.weight"]

    mx.random.seed(11)
    embeddings = np.array(
        mx.random.uniform(low=-0.5, high=0.5, shape=(len(TOKENS), dim))
    ).astype(np.float32)
    decode_embeddings = np.array(
        mx.random.uniform(low=-0.5, high=0.5, shape=(len(DECODE_TOKENS), dim))
    ).astype(np.float32)

    inputs = mx.array(embeddings)[None]
    cache = [KVCache() for _ in range(num_layers)]
    hidden = indextts_gpt2_forward(model.model, inputs, cache)
    prefill = np.array((hidden @ wte_weight.T)[0]).astype(np.float32)

    steps = []
    for row in decode_embeddings:
        step_hidden = indextts_gpt2_forward(
            model.model, mx.array(row)[None, None], cache
        )
        steps.append(np.array((step_hidden @ wte_weight.T)[0, -1]).astype(np.float32))

    cacheless_hidden = indextts_gpt2_forward(model.model, inputs)
    cacheless = np.array((cacheless_hidden @ wte_weight.T)[0]).astype(np.float32)

    return {
        "name": name,
        "tie_head": True,
        "tokens": TOKENS,
        "decode_tokens": DECODE_TOKENS,
        "embeddings": embeddings.tolist(),
        "decode_embeddings": decode_embeddings.tolist(),
        "prefill_logits": prefill.tolist(),
        "decode_logits": [step.tolist() for step in steps],
        "cacheless_prefill_logits": cacheless.tolist(),
        "hidden_state": np.array(cacheless_hidden[0]).astype(np.float32).tolist(),
        "weights": weights,
        "config": asdict(args),
    }


def save_case(case):
    case_dir = OUT / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    weights_path = case_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), case["weights"])
    (case_dir / "config.json").write_text(json.dumps(case["config"], indent=2) + "\n")
    record = {
        k: v
        for k, v in case.items()
        if k not in ("weights", "config")
    }
    (case_dir / "traces.json").write_text(json.dumps(record))
    return weights_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []

    # Plain llama, tied embeddings, no cache-returning hidden path.
    model = seeded_parameters(LlamaModel(tiny_llama_args()))
    cases.append(run_case("llama_default", model, tie_head=True, num_layers=2, args=tiny_llama_args()))

    # llama3 rope scaling.
    model = seeded_parameters(
        LlamaModel(
            tiny_llama_args(
                rope_scaling={
                    "type": "llama3",
                    "factor": 4.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8,
                }
            )
        )
    )
    cases.append(run_case(
        "llama_llama3_rope",
        model,
        tie_head=True,
        num_layers=2,
        args=tiny_llama_args(
            rope_scaling={
                "type": "llama3",
                "factor": 4.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8,
            }
        ),
    ))

    # mrope reduces to the plain construction in the reference.
    model = seeded_parameters(
        LlamaModel(
            tiny_llama_args(
                rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]}
            )
        )
    )
    cases.append(run_case(
        "llama_mrope_rope",
        model,
        tie_head=True,
        num_layers=2,
        args=tiny_llama_args(rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]}),
    ))

    # linear rope scaling, untied head.
    model = seeded_parameters(
        LlamaModel(
            tiny_llama_args(
                rope_scaling={"type": "linear", "factor": 4.0},
                tie_word_embeddings=False,
            )
        )
    )
    cases.append(run_case(
        "llama_linear_rope_untied",
        model,
        tie_head=False,
        num_layers=2,
        args=tiny_llama_args(
            rope_scaling={"type": "linear", "factor": 4.0},
            tie_word_embeddings=False,
        ),
    ))

    # yarn rope scaling.
    model = seeded_parameters(
        LlamaModel(
            tiny_llama_args(
                rope_scaling={
                    "type": "yarn",
                    "factor": 2.0,
                    "original_max_position_embeddings": 16,
                    "beta_fast": 8,
                    "beta_slow": 1,
                    "mscale": 1.0,
                    "mscale_all_dim": 0.0,
                }
            )
        )
    )
    cases.append(run_case(
        "llama_yarn_rope",
        model,
        tie_head=True,
        num_layers=2,
        args=tiny_llama_args(
            rope_scaling={
                "type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 16,
                "beta_fast": 8,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 0.0,
            }
        ),
    ))

    # longrope (SuScaled) rope scaling.
    model = seeded_parameters(
        LlamaModel(
            tiny_llama_args(
                rope_scaling={
                    "type": "longrope",
                    "original_max_position_embeddings": 16,
                    "short_factor": [1.0] * 4,
                    "long_factor": [1.7, 1.3, 1.1, 1.05],
                }
            )
        )
    )
    cases.append(run_case(
        "llama_longrope_rope",
        model,
        tie_head=True,
        num_layers=2,
        args=tiny_llama_args(
            rope_scaling={
                "type": "longrope",
                "original_max_position_embeddings": 16,
                "short_factor": [1.0] * 4,
                "long_factor": [1.7, 1.3, 1.1, 1.05],
            }
        ),
    ))

    # proportional (partial rotary) rope scaling.
    model = seeded_parameters(
        LlamaModel(
            tiny_llama_args(
                rope_scaling={"type": "proportional", "partial_rotary_factor": 0.5}
            )
        )
    )
    cases.append(run_case(
        "llama_proportional_rope",
        model,
        tie_head=True,
        num_layers=2,
        args=tiny_llama_args(
            rope_scaling={"type": "proportional", "partial_rotary_factor": 0.5}
        ),
    ))

    # qwen2: attention q/k/v biases, head_dim derived.
    qwen2_args = Qwen2Args(
                model_type="qwen2",
                hidden_size=32,
                num_hidden_layers=2,
                intermediate_size=64,
                num_attention_heads=4,
                rms_norm_eps=1e-5,
                vocab_size=96,
                num_key_value_heads=2,
                max_position_embeddings=64,
                rope_theta=1000000.0,
    )
    model = seeded_parameters(Qwen2Model(qwen2_args))
    cases.append(run_case("qwen2_default", model, tie_head=True, num_layers=2, args=qwen2_args))

    # qwen3: q/k norms, explicit head_dim.
    qwen3_args = Qwen3Args(
                model_type="qwen3",
                hidden_size=32,
                num_hidden_layers=2,
                intermediate_size=64,
                num_attention_heads=4,
                rms_norm_eps=1e-5,
                vocab_size=96,
                num_key_value_heads=2,
                max_position_embeddings=64,
                rope_theta=1000000.0,
                head_dim=8,
                tie_word_embeddings=True,
    )
    model = seeded_parameters(Qwen3Model(qwen3_args))
    cases.append(run_case("qwen3_default", model, tie_head=True, num_layers=2, args=qwen3_args))

    # gpt2: fused biased QKV, gelu MLP, learned token/position embeddings,
    # LayerNorm pre-norm, tied head. num_key_value_heads is left unset so
    # the reference __post_init__ default (kv heads = query heads) is what
    # the model runs (it never reads the field anyway).
    gpt2_args = Gpt2Args(
        model_type="gpt2",
        n_ctx=64,
        n_embd=32,
        n_head=4,
        n_layer=2,
        n_positions=64,
        layer_norm_epsilon=1e-5,
        vocab_size=96,
    )
    model = seeded_parameters(Gpt2Model(gpt2_args))
    cases.append(run_case("gpt2_default", model, tie_head=True, num_layers=2, args=gpt2_args))

    # gpt2 driven the way indextts consumes it: ModelArgs as the family
    # builds them (n_ctx 1, kv heads defaulting to query heads) with the
    # degenerate vocab/n_positions widened so the shared model can run,
    # wte/wpe bypassed, and float embeddings as the input. The tied-head
    # logits come from the captured wte table, matching the Rust
    # Input::Embeddings path.
    indextts_args = Gpt2Args(
        "gpt2",
        1,      # n_ctx, as indextts passes
        32,     # n_embd
        4,      # n_head
        2,      # n_layer
        64,     # n_positions (indextts passes 1 with wpe bypassed)
        1e-5,   # layer_norm_epsilon
        96,     # vocab_size (indextts passes 1 with wte bypassed)
    )
    model = seeded_parameters(Gpt2Model(indextts_args))
    cases.append(run_embeddings_case("gpt2_indextts_embeddings", model, 2, indextts_args))

    # cohere2: sliding-window attention with the rotating ring. Layers
    # 0..2 are sliding (window 4, traditional rope), layer 3 is global
    # (no rope, dense cache); the 7-token prompt is longer than the window
    # so both the prefill window mask and the decode rotation are
    # exercised.
    cohere2_args = Cohere2Args(
        model_type="cohere2",
        hidden_size=32,
        head_dim=8,
        num_hidden_layers=4,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_theta=50000.0,
        vocab_size=96,
        layer_norm_eps=1e-5,
        logit_scale=0.0625,
        attention_bias=False,
        layer_norm_bias=False,
        sliding_window=4,
        sliding_window_pattern=4,
    )
    model = seeded_parameters(Cohere2Model(cohere2_args))
    cases.append(run_case(
        "cohere2_default",
        model,
        tie_head=True,
        num_layers=4,
        args=cohere2_args,
        make_cache=model.make_cache,
    ))

    # The streaming shape: a second two-token chunk arrives after the
    # window is full, exercising the `_update_concat` trim path and the
    # clamped-offset window mask, then decode steps continue on the
    # rotated ring.
    model = seeded_parameters(Cohere2Model(cohere2_args))
    cases.append(run_case(
        "cohere2_streaming",
        model,
        tie_head=True,
        num_layers=4,
        args=cohere2_args,
        make_cache=model.make_cache,
        chunk_tokens=[11, 27],
    ))

    # bailing_moe: one dense layer then two MoE layers (first_k_dense_replace
    # = 1), 4 routed experts with top-2 routing, one shared expert, q/k
    # norms, untied head. The router exercises softmax scores, the expert
    # bias (seeded nonzero), norm_topk_prob, and routed_scaling_factor 2.0
    # (the reference multiplies it unconditionally).
    bailing_args = BailingMoeArgs(
        model_type="bailing_moe",
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=64,
        moe_intermediate_size=32,
        num_experts=4,
        num_shared_experts=1,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_experts_per_tok=2,
        num_hidden_layers=3,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        vocab_size=96,
        first_k_dense_replace=1,
        use_bias=False,
        use_qkv_bias=False,
        norm_head=False,
        use_qk_norm=True,
        tie_word_embeddings=False,
        moe_router_enable_expert_bias=True,
        routed_scaling_factor=2.0,
        score_function="softmax",
        n_group=1,
        topk_group=4,
    )
    model = seeded_parameters(BailingMoeModel(bailing_args))
    cases.append(run_case("bailing_moe_default", model, tie_head=False, num_layers=3, args=bailing_args))

    for case in cases:
        save_case(case)

    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_lm_fixtures.py",
        "cases": {},
    }
    for path in sorted(OUT.glob("*/*")):
        manifest["cases"][f"{path.parent.name}/{path.name}"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
