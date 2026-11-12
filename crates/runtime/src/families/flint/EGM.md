# Flint Decode Flow

## Overview
The `flint` decode flow for `RealForwardRunner`: 49 layers of GQA attention (per-head QK norm, learned sinks, NoPE on full-attention layers, sliding window on the rest) followed by a dense SwiGLU lead (layers 0 and 1 for the published checkpoint; the code takes the count from `num_dense_leading_layers`) and then sigmoid-routed MoE layers with an ungated shared expert. It is a distinct flow rather than a variant of an existing family; the skeleton follows `families/deepseek2` and the attention follows `families/gptoss`. A sequential per-token `produce` path and a layer-batched chunked prefill path share the same per-token attention and routed-MoE encoders.

## Interface
- `RealFlintState` [state.rs:RealFlintState]: open-time state (scratch buffers, lead, widths, rms_eps); built and validated by `RealFlintState::build`
- `layer_tensor` [mod.rs:layer_tensor]: builds `language_model.model.layers.{layer}.{suffix}` tensor names
- `produce_real_flint` [mod.rs:produce_real_flint]: one-token decode (also sequential prefill) on `RealForwardRunner`, writes raw f16 logits
- `encode_attention_block` [attn.rs:encode_attention_block]: q/k/v GEMV, per-head QK norm, conditional RoPE, KV commit, sink attention, o_proj
- `sigmoid_topk_renorm` [moe.rs:sigmoid_topk_renorm]: host top-k ranked by logit, weights sigmoid/sum times scale
- `encode_swiglu` [moe.rs:encode_swiglu]: dense SwiGLU for the lead (width 6144) and the shared expert (width 1024)
- `encode_flint_layer_moe` [moe.rs:encode_flint_layer_moe]: shared expert, expert plan/stream, phase 1/2 seeded by shared output, residual add; returns used cache slots
- `prefill_chunk_real_flint` [prefill.rs:prefill_chunk_real_flint]: chunked prefill driver, micro-batches of `MAX_PREFILL_BATCH`
- `dump_router` / `debug_report` [mod.rs:dump_router]: env-gated diagnostics

## Invariants
- Flow is selected by family, and `build` refuses any family other than `ModelFamily::Flint`, sandwich norms, scaled router/embedding, output gate, gated shared expert, K=V, softcap, tied embeddings, non-sigmoid router [state.rs:RealFlintState::build]
- Per-head QK RMSNorm runs before RoPE; V is never normed or rotated [attn.rs:encode_attention_block]
- RoPE is skipped exactly when the layer mask is 1 (full attention is NoPE), and `build` requires `full_rope_theta == 0` [attn.rs:encode_attention_block, state.rs:RealFlintState::build]
- One sink per query head: `softmax_offset` must be `num_heads * 2` bytes, checked at open [state.rs:RealFlintState::build]
- Sliding window attends `sliding_window` keys including the current one (`kv_start = seq_len - window`) [attn.rs:encode_attention_block]
- Expert selection ranks by logit (stable, lower index wins ties), not by saturating sigmoid score; slot order is reduction order [moe.rs:sigmoid_topk_renorm]
- Shared expert runs before phase 1/2 so its output seeds phase 2's residual input; blob index is `layer - lead` [moe.rs:encode_flint_layer_moe]
- Router tensors must be INT8 (dtype 5) of `num_experts * hidden` bytes; lead must be in `1..num_layers` [state.rs:RealFlintState::build]
- Positions must be sequential with the KV cache, and token ids below vocab [mod.rs:produce_real_flint_inner, prefill.rs:prefill_micro_batch_flint_inner]
- Producer returns raw logits, no softcap; vocab must match `logits.len()` [mod.rs:produce_real_flint_inner]
- `TURBOSPARK_BATCHED_GEMV` and `TURBOSPARK_ROUTED_BATCH` are refused by name for flint [prefill.rs:prefill_chunk_real_flint]
- Per-token attention reuses single-row scratch inside one command buffer; moe_x and router_logits_f32 hold one row per micro-batch token [prefill.rs:encode_flint_chunk_token, state.rs:RealFlintState]
- Routed pipeline depth is 1 in flight (one pending pass), using two banks when the expert cache holds at least 2*top_k slots (`routed_pipeline_banks`); with one bank the previous pass is retired before planning, otherwise previous slots are protected [prefill.rs:encode_flint_layer_routed_moe_pipelined]

## Side Effects
- Metal command buffers committed and waited per MoE layer (decode) and per layer per micro-batch (prefill); allocates Metal buffers at open [mod.rs:produce_real_flint_inner, state.rs:RealFlintState::build]
- Expert `pread` streaming and expert-cache mutation via `PreadExpertStreamer` [moe.rs:encode_flint_layer_moe]
- Writes routing weights into `scratch.routing_w` and records router histogram and phase counters [moe.rs:encode_flint_layer_moe]
- Advances the KV position (`kv.advance` / `advance_by`) and records residual capture [mod.rs:produce_real_flint_inner, prefill.rs:prefill_micro_batch_flint_inner]
- Env `TURBOSPARK_FLINT_DEBUG` (per-layer readback; stderr only for non-finite values or magnitude above `DEBUG_REPORT_ABOVE`) [mod.rs:produce_real_flint_inner, mod.rs:debug_report]
- Env `TURBOSPARK_FLINT_ROUTER_DUMP` (appends binary records to a file) [mod.rs:dump_router]
- Env `TURBOSPARK_FLINT_CHUNKED_PREFILL` (default on, `0` disables; cached in a `OnceLock`) [prefill.rs:chunked_prefill_enabled]

## Verification
- Run: `cargo test -p turbospark-runtime flint`
