# LFM3 Decode Flow

## Overview
The `lfm3` decode flow for `RealForwardRunner` (LiquidAI LFM3-1.7B): a dense 24-layer hybrid of gated short-convolution layers (mask 6), full GQA layers (mask 1) and sliding-window GQA layers (mask 0). Every layer is sandwich-normed (operator norm and post-operator norm, FFN norm and post-FFN norm), the head is tied to the embedding, and the final norm is `embedding_norm`. A single-row decode path and an M-row chunked prefill driver for INT4-affine installs share the same per-class rope selection.

## Interface
- `RealLfm3State` [state.rs:RealLfm3State]: open-time state (conv tails, scratch, taps, fused-FFN flag, lazily allocated batched buffers); `build` validates the install and `reset` zeroes conv tails
- `Lfm3Batched` [state.rs:Lfm3Batched]: M-row buffers for the chunked driver, allocated at the first chunk
- `RMS_EPS` [state.rs:RMS_EPS]: 1e-5 on every norm including per-head q/k norms
- `layer_tensor` [mod.rs:layer_tensor]: builds `language_model.model.layers.{layer}.{suffix}` names
- `produce_real_lfm3` [mod.rs:produce_real_lfm3]: one-token decode (also sequential prefill), writes raw f16 logits
- `set_lfm3_fused_ffn` [mod.rs:set_lfm3_fused_ffn]: test hook toggling the fused FFN
- `short_conv_tail_abs_max` [mod.rs:short_conv_tail_abs_max]: test probe of a conv layer's tail
- `encode_conv_block` [conv.rs:encode_conv_block]: in_proj GEMV, short-conv decode kernel (tail update), out_proj GEMV
- `encode_attention_block` [attn.rs:encode_attention_block]: q/k/v/gate GEMVs, per-head QK norm, per-class rope, attention, headwise sigmoid gate, o_proj
- `rope_params_for` [attn.rs:rope_params_for]: `(rotated_pairs, theta)` per layer class
- `encode_mlp_block` [mlp.rs:encode_mlp_block]: pre-FFN norm, SwiGLU (fused or chained), post-FFN norm, residual add
- `lfm3_fused_ffn` [mlp.rs:lfm3_fused_ffn]: cached `TURBOSPARK_LFM3_FUSED_FFN` switch
- `prefill_chunk_real_lfm3` [prefill.rs:prefill_chunk_real_lfm3]: chunked prefill entry; `lfm3_chunked_refusal` is the shared support predicate
- `lfm3_chunked_prefill` / `LFM3_PREFILL_ROWS` [prefill.rs:lfm3_chunked_prefill]: switch and micro-batch width

## Invariants
- `build` refuses any family other than `ModelFamily::Lfm3`, sandwich norms off, router/embedding scaling, attn output gate, shared expert gate, K=V, `rope_neox_subdim`, softcap, untied head, non-silu activation, MoE, and layer masks other than 0/1/6 [state.rs:RealLfm3State::build]
- The install must not carry `language_model.lm_head.weight`; logits use `embed_tokens` [state.rs:RealLfm3State::build, mod.rs:produce_real_lfm3_inner]
- Both rope thetas must be positive, full-layer rotation must be a whole number of pairs, taps in 2..=16, conv weight is BF16 `[hidden, taps]` [state.rs:RealLfm3State::build]
- Rope is `rope_proportional_neox` on both classes: full layers rotate `full_rotated_pairs` at `full_rope_theta`, sliding layers rotate all pairs at `rope_theta` [attn.rs:rope_params_for]
- Per-head q/k norm runs before rope; V is not normed; the head gate reads `scratch.normed` before it is overwritten and is applied before o_proj [attn.rs:encode_attention_block]
- Short conv applies no activation, and the tail is the layer's whole recurrent state; `reset` must zero it [conv.rs:encode_conv_block, state.rs:RealLfm3State::reset]
- Embedding has no scale and no norm; the sandwich norms precede each residual add (post-FFN norm is not optional) [mod.rs:produce_real_lfm3_inner, mlp.rs:encode_mlp_block]
- Fused gate+up+silu is used only when both weights are INT4 and `int4_mma8_takes` is false, and must stay bit-exact against the chain [mlp.rs:encode_mlp_block]
- The chunked driver needs every projection INT4-affine and refuses quantized KV; `prefill_chunk_real_lfm3` itself refuses `TURBOSPARK_FFN_HIST` [prefill.rs:lfm3_chunked_refusal, prefill.rs:prefill_chunk_real_lfm3]
- Micro-batch width is capped to `ring - window` so a sliding ring never evicts a position a row still reads; K/V go through staging rows and contiguous-run copies [prefill.rs:lfm3_prefill_width, prefill.rs:commit_kv_rows]
- Multi-row GQA attention serves full layers only (m > 1); sliding layers keep one-row attention per row with per-row window start [prefill.rs:encode_attention_rows]
- Positions must be sequential with the KV cache and token ids below vocab; logits are raw with a vocab length check [mod.rs:produce_real_lfm3_inner, prefill.rs:prefill_micro_batch_lfm3]

## Side Effects
- Metal command buffers committed and waited once per token (decode) or once per micro-batch (prefill); Metal buffers allocated at open and lazily for `Lfm3Batched` [mod.rs:produce_real_lfm3_inner, prefill.rs:prefill_chunk_real_lfm3]
- Mutates conv tail state and advances the KV position (`kv.advance` / `advance_by`) [conv.rs:encode_conv_block, prefill.rs:prefill_micro_batch_lfm3]
- Host writes of vision rows into `scratch.x` for image-pad positions before commit [mod.rs:produce_real_lfm3_inner, prefill.rs:prefill_micro_batch_lfm3]
- Records FFN activation histograms when `ffn_hist` is present [mlp.rs:encode_mlp_block, mod.rs:produce_real_lfm3_inner]
- Env `TURBOSPARK_LFM3_FUSED_FFN` (default on, `OnceLock`) [mlp.rs:lfm3_fused_ffn]
- Env `TURBOSPARK_LFM3_CHUNKED_PREFILL` (default on, `0` is the per-token loop, `OnceLock`) [prefill.rs:lfm3_chunked_prefill]
- Related switches read elsewhere: `TURBOSPARK_PREFILL_STEEL`, `TURBOSPARK_PREFILL_ROWS_ATTN` via runner fields [prefill.rs:prefill_micro_batch_lfm3]

## Verification
- Run: `cargo test -p turbospark-runtime lfm3`
