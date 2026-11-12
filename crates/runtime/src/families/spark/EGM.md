# Spark 2.5 Decode Flow

## Overview
The `spark2_5` (Spark-X2.5-4B) decode flow for `RealForwardRunner`: a dense 36-layer GQA stack with a three-sliding/one-full window pattern, a fused QKV projection, per-layer-class rope, a headwise sigmoid attention gate, exact-erf GELU FFN, and a tied head. There are no experts, no router, and no mid-layer host round trip, so a whole token (or a whole prefill micro-batch) encodes into one command buffer. State is allocated once at install open and refuses installs that carry any behavioural extension this family lacks.

## Interface
- `RealForwardRunner::produce_real_spark` [mod.rs:produce_real_spark]: single-token decode entry; times the call into `phases`
- `produce_real_spark_inner` [mod.rs:produce_real_spark_inner]: embed, per-layer norm/attention/residual/MLP/steering/capture, final norm and tied raw-logit head
- `layer_tensor` [mod.rs:layer_tensor]: builds `language_model.model.layers.{n}.{suffix}` names
- `RealSparkState` [state.rs:RealSparkState]: `full_rotary_dim`, fused `qkv` staging buffer, per-head `attn_gate` buffer; `build` validates the install
- `RMS_EPS` [state.rs:RMS_EPS]: the single 1e-6 epsilon used by every norm
- `encode_attention_block` [attn.rs:encode_attention_block]: fused q_k_v GEMV, g_proj GEMV, split into q and cache slots, per-class rope, windowed attention, headwise gate, o_proj
- `rope_params_for` [attn.rs:rope_params_for]: selects (rotary dim, theta) by layer class
- `encode_mlp_block` [mlp.rs:encode_mlp_block]: post-attention norm, gate/up GEMVs, exact-erf GELU multiply, down GEMV, raw residual add
- `RealForwardRunner::prefill_chunk_real_spark` [prefill.rs:prefill_chunk_real_spark]: chunked prefill in `MAX_PREFILL_BATCH` micro-batches
- `prefill_micro_batch_spark_inner` [prefill.rs:prefill_micro_batch_spark_inner]: one command buffer for all layers of all tokens in the micro-batch; head only on the last chunk

## Invariants
- Positions must be sequential against `kv.position()` and token ids below vocab, in both decode and prefill [mod.rs:produce_real_spark_inner, prefill.rs:prefill_micro_batch_spark_inner]
- Build refuses any other family, any of ffnSandwichNorms / routerScaled / embeddingScaledBySqrtHidden / attnOutputGate / sharedExpertGated / attentionKEqV, a nonzero softcap, and untied embeddings [state.rs:RealSparkState::build]
- `hiddenActivation` must be exactly `"gelu"` (exact erf, never the tanh approximation) [state.rs:RealSparkState::build, mlp.rs:encode_mlp_block]
- Dense only; mask values are 0 or 1 with at least one sliding layer and a positive sliding window [state.rs:RealSparkState::build]
- Both rope thetas must be positive, `partialRotaryFactor` in (0, 1], head dim and kv-head count equal across layer kinds, and the full-layer rotary dim even [state.rs:RealSparkState::build]
- Rope is per layer class: full layers rotate the leading `full_rotary_dim` at full theta, sliding layers rotate the whole head at sliding theta [attn.rs:rope_params_for]
- The headwise gate is applied to the attention output before `o_proj`, and g_proj reads the same normed input as QKV [attn.rs:encode_attention_block]
- All per-layer tensors exist at open, and `lm_head.weight` must not exist (tied head) [state.rs:RealSparkState::build]
- The `qkv` buffer is sized `q + 2*kv` elements in bytes of f16; undersizing corrupts neighbours [state.rs:RealSparkState::build]
- `TURBOSPARK_BATCHED_GEMV` prefill is refused by name, never ignored [prefill.rs:prefill_chunk_real_spark]
- Producer returns raw logits and the caller's slice length must equal vocab [mod.rs:produce_real_spark_inner]
- Steering is applied before residual capture at the layer's FFN residual boundary [mod.rs:produce_real_spark_inner]

## Side Effects
- Metal: one labeled pass per token or micro-batch, committed and waited at the end [mod.rs:produce_real_spark_inner, prefill.rs:prefill_micro_batch_spark_inner]
- Allocates two Metal buffers at build [state.rs:RealSparkState::build]
- Advances the KV position (`advance` or `advance_by`) [mod.rs:produce_real_spark_inner, prefill.rs:prefill_micro_batch_spark_inner]
- Records residual capture and FFN activation histogram passes; under the histogram census the activation writes to the capture buffer [mod.rs:produce_real_spark_inner, mlp.rs:encode_mlp_block]
- Quantizes KV staging rows in place on TurboQuant layers [attn.rs:encode_attention_block]

## Verification
- Run: `cargo test -p turbospark-runtime`
