# Llama Family Flow

## Overview
The plain grouped-query-attention decode and chunked-prefill flow for `RealForwardRunner`. One flow serves the `llama` architecture string (Mixtral MoE, Mistral and Llama 2/3.x dense) and, through family-selected state, Qwen2 dense, Qwen3 dense/MoE/VL trunk and MiniMax-M2. `RealLlamaState.dense` (from `num_experts == 0`) picks the FFN half; attention, norms, residuals and the head are shared. Dense layers stay in one command buffer; MoE layers commit mid-layer to read the router top-k back on the host.

## Interface
- `RealLlamaState` [state.rs:RealLlamaState]: per-install state (rotated pairs, QkNorm, qkv bias, rms_eps, router buffers, `dense`, optional batch-attention buffers); built once by `RealLlamaState::build`
- `RealLlamaState::build` [state.rs:build]: validates the install and tensors at open, allocates GPU buffers
- `BatchAttentionBuffers` [state.rs:BatchAttentionBuffers]: capacity-sized Q/output/split-KV scratch for dense Llama batched attention
- `layer_tensor` [mod.rs:layer_tensor]: builds `language_model.model.layers.{n}.{suffix}` names
- `RealForwardRunner::produce_real_llama` [mod.rs:produce_real_llama]: sequential one-token decode (embed, layers, optional head), wraps `produce_real_llama_inner` with phase timing
- `encode_attention_block` [attn.rs:encode_attention_block]: q/k/v projection, optional bias, Q/K norm, RoPE (plain, neox-subdim, mRoPE), KV commit, attention, o_proj
- `encode_attention_inputs_batch` / `PreparedAttentionRows` [attn.rs:encode_attention_inputs_batch]: stages query rows and K/V cache writes for batched attention
- `RealForwardRunner::encode_llama_layer_attn_and_router` [attn.rs:encode_llama_layer_attn_and_router]: MoE chunk cb1 per-token attention, residual, post-norm, router GEMV
- `encode_llama_layer_dense` [dense.rs:encode_llama_layer_dense]: gate/up GEMV, SiLU or GELU mul, down GEMV, raw residual add at `x_off`
- `encode_llama_layer_moe` [moe.rs:encode_llama_layer_moe]: host top-k, expert streaming or mapped binding, phase1/phase2, residual add; returns bound cache slots
- `RealForwardRunner::encode_llama_layer_routed_moe_pipelined` [moe.rs:encode_llama_layer_routed_moe_pipelined]: depth-1 pipelined routed loop across a micro-batch
- `router::encode`, `router::sigmoid_topk`, `router::fp32_tensor` [router.rs:encode]: Gemma-style INT8 router GEMV or MiniMax FP32 router; biased-sigmoid selection with unbiased weights
- `RealForwardRunner::prefill_chunk_real_llama_dense` [prefill.rs:prefill_chunk_real_llama_dense]: dense chunked prefill in one command buffer per micro-batch, with optional batched attention and vision injection
- `batch_attention_route_eligible` / `stage_attention_inputs_batch` [prefill.rs:batch_attention_route_eligible]: gate and stage the dense batched-attention route
- `RealForwardRunner::prefill_chunk_real_llama_moe` [moe_prefill.rs:prefill_chunk_real_llama_moe]: MoE chunked prefill, per-layer cb1 plus pipelined routed half
- Subdirectories: None

## Invariants
- Build refuses Gemma extensions (sandwich norms, scaled router, embed scaling, attn output gate, gated shared expert, k_eq_v, non-MiniMax neox subdim), any logit softcap, and any non-full-attention layer [state.rs:build]
- `num_experts` and `top_k_experts` must both be zero or both positive, non-negative, and `top_k <= gpu::MAX_STREAMED_EXPERTS` [state.rs:build]
- Q/K norm kind, rms_eps and qkv bias are read from `ArchConfig.family`, never sniffed from tensor presence; all required tensors are checked at open [state.rs:build]
- The dense FFN width is `intermediate_size`, never `moe_intermediate_size`; build refuses a dense install with `intermediate_size <= 0` [dense.rs:encode_llama_layer_dense, state.rs:build]
- Q/K norm runs before RoPE; qkv bias is added before RoPE [attn.rs:encode_attention_input_row]
- Attention and FFN outputs are added raw to the stream (no sandwich or post norm) [mod.rs:produce_real_llama_inner]
- Decode and prefill require sequential positions equal to the KV position and token ids below vocab [mod.rs:produce_real_llama_inner] [prefill.rs:prefill_micro_batch_llama_dense_inner]
- Producers return raw logits (no softmax, no softcap); vocab must equal `logits.len()` [mod.rs:produce_real_llama_inner]
- Routed slots dispatch in router ranking order (slot order is FP reduction order); phase 2 seeds from `zero_hidden` so the residual is added exactly once [moe.rs:encode_llama_layer_moe]
- MiniMax requires FP32 `mlp.gate.weight`/`e_score_correction_bias` (dtype 3), finite bias, a MoE install with sigmoid scoring and `rope_neox_subdim`, an untied head; the correction biases select but never weight [state.rs:build] [router.rs:sigmoid_topk]
- Batched attention is eligible only for dense `ModelFamily::Llama`, all-full-attention layers, unquantized KV, more than one live row, and GPU capacity equal to `MAX_PREFILL_BATCH` [state.rs:batch_attention_eligible] [prefill.rs:batch_attention_route_eligible]
- The MoE chunked driver refuses `TURBOSPARK_ROUTED_BATCH`, `TURBOSPARK_BATCHED_GEMV` (dense refuses the latter) and a live image map by name instead of running the unbatched path [moe_prefill.rs:prefill_chunk_real_llama_moe] [prefill.rs:prefill_chunk_real_llama_dense]

## Side Effects
- Allocates Metal buffers (router logits, `moe_x`, `h2`, ones, batch attention) at install open [state.rs:build]
- Reads MiniMax correction bias back from the resident weight buffer at open [state.rs:build]
- Encodes and commits Metal command buffers; MoE decode and prefill commit and wait mid-layer, dense commits once per micro-batch (or per layer on the batched-attention route) [mod.rs:produce_real_llama_inner] [prefill.rs:prefill_micro_batch_llama_dense_inner]
- Mutates the KV cache (advance, KV writes), phase counters, router histogram, steering and resid-capture state [mod.rs:produce_real_llama_inner] [moe.rs:encode_llama_layer_moe]
- Expert I/O: `pread` streaming or mapped residency, writes routing weights into `scratch.routing_w` [moe.rs:encode_llama_layer_moe]
- Reads env-driven runner flags `routed_batch_prefill`, `batched_gemv_prefill`, `routed_pipeline`; `TURBOSPARK_STANDIN` via `lab_hooks::standin` [moe_prefill.rs:prefill_chunk_real_llama_moe] [mod.rs:produce_real_llama_inner]
- Writes image-pad embedding rows into `scratch.x` via host write [mod.rs:produce_real_llama_inner]

## Verification
- Run: `cargo test -p turbospark-runtime`
