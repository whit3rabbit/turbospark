# gpt-oss Decode Flow

## Overview
The fifth runtime decode flow, serving the `gpt-oss` MoE family (MXFP4 experts, biased attention, YaRN RoPE, attention sinks, alternating sliding window). It is implemented as `impl RealForwardRunner` methods plus a per-install state struct, with no sandwich norms, shared expert, output gate or logit softcap. Decode runs two command buffers per layer (attention plus router GEMV, then routed experts) because the router top-k is host-side. Chunked prefill pipelines the same per-token routed half, with an optional batched MXFP4 route-list arm.

## Interface
- `layer_tensor` [mod.rs:layer_tensor]: builds `language_model.model.layers.{n}.{suffix}` tensor names
- `RealGptOssState` [state.rs:RealGptOssState]: per-open state: YaRN frequency table, mscale, host router bias, router ones, router logits and `moe_x` scratch
- `RealGptOssState::build` [state.rs:build]: validates family and manifest flags, checks every bias and sink tensor exists, uploads the YaRN table
- `RealGptOssState::ensure_batched` [state.rs:ensure_batched]: lazily allocates `BatchedRoutedScratch`
- `BatchedRoutedScratch` [state.rs:BatchedRoutedScratch]: M-row buffers for the batched routed pair (acts, y, routing weights, routes, zeros, wide blobs)
- `RealForwardRunner::produce_real_gpt_oss` [mod.rs:produce_real_gpt_oss]: single-token decode producing raw f16 logits
- `encode_attention_block` [attn.rs:encode_attention_block]: GQA with projection biases, YaRN RoPE, sinks, alternating window, o_proj bias
- `encode_gpt_oss_layer_attn_and_router` [attn.rs:encode_gpt_oss_layer_attn_and_router]: cb1 for one token of a prefill micro-batch
- `encode_gpt_oss_layer_moe` [moe.rs:encode_gpt_oss_layer_moe]: host router bias add, top-k, expert plan and stream, phase 1 and 2, residual add
- `encode_gpt_oss_layer_routed_moe_pipelined` [moe.rs:encode_gpt_oss_layer_routed_moe_pipelined]: per-token routed loop for a micro-batch
- `encode_gpt_oss_layer_routed_moe_batched` [moe_batch.rs:encode_gpt_oss_layer_routed_moe_batched]: union-bounded sub-batch route-list MXFP4 kernels
- `prefill_chunk_real_gpt_oss` [prefill.rs:prefill_chunk_real_gpt_oss]: chunked prefill in micro-batches of `MAX_PREFILL_BATCH`

## Invariants
- The router bias is added to read-back logits before top-k, never after [moe.rs:encode_gpt_oss_layer_moe]
- Expert weights are a softmax over the selected k logits only (top-k on the biased logits, no full-set renormalization), via `router_topk_gemma4` with unit per-expert scale [moe.rs:encode_gpt_oss_layer_moe]
- Biases on q/k/v are applied before RoPE; V is not rotated [attn.rs:encode_attention_block]
- Sinks are one BF16 logit per query head (`num_heads * 2` bytes), checked at open [state.rs:build]
- `build` refuses any non-gpt-oss family, Gemma-style flags, softcap, tied embeddings, non-MoE, `top_k` above `MAX_STREAMED_EXPERTS`, mask values above 1, non-positive window and zero YaRN factor [state.rs:build]
- Slot order is the router ranking, since phase 2 reduces in slot order [moe.rs:encode_gpt_oss_layer_moe]
- Producer returns raw logits (no softmax, no softcap) and requires sequential positions and in-vocab tokens [mod.rs:produce_real_gpt_oss_inner]
- The batched routed half refuses non-MXFP4 layouts, mapped residency, too many slots and the pilot probe [moe_batch.rs:encode_gpt_oss_layer_routed_moe_batched]
- `TURBOSPARK_BATCHED_GEMV` is refused by name for this family [prefill.rs:prefill_chunk_real_gpt_oss]
- Router tensor must be INT8 (dtype 5) with `num_experts * hidden` bytes [mod.rs:produce_real_gpt_oss_inner]

## Side Effects
- Metal command buffers, GPU buffer allocation and writes (YaRN table, ones, zero buffers) [state.rs:build]
- Expert streaming `pread` through `PreadExpertStreamer` and expert cache state [moe.rs:encode_gpt_oss_layer_moe]
- Mutates KV cache position via `kv.advance()` [mod.rs:produce_real_gpt_oss_inner]
- Updates phase counters, router histogram and residual capture [moe.rs:encode_gpt_oss_layer_moe]
- Reads env-driven seams `TURBOSPARK_ROUTED_BATCH`, `TURBOSPARK_ROUTED_PIPELINE`, `TURBOSPARK_EXPERT_RESIDENCY` indirectly through runner flags [prefill.rs:prefill_micro_batch_gpt_oss_inner]

## Verification
- Run: `cargo test -p turbospark-runtime`
