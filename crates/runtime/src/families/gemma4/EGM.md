# Gemma 4 Family Flow

## Overview
The real-checkpoint Gemma 4 decode and prefill flow, implemented as `impl RealForwardRunner` blocks plus the per-layer state built at open. It covers the dense trunk and the routed-MoE trunk (shared dense expert plus top-k routed experts in a sandwich-norm layer), single-token decode, chunked and micro-batched prefill, and image-span prefill with the bidirectional sliding mask. Decode flow is chosen by the arch config (`num_experts == 0` selects dense), never by tensor names. The module is crate-private and is driven by `RealForwardRunner` in `real_forward`.

## Interface
- `RealGemmaState` [state.rs:RealGemmaState]: per-layer effective router scale, per-expert scale, layer scalar, router/scratch buffers, lazily allocated batched scratch
- `RealGemmaState::build` [state.rs:build]: open-time construction; pre-folds 1/sqrt(hidden) into `router.scale`, decodes scales
- `RealGemmaState::ensure_batched` / `batched` [state.rs:ensure_batched]: first-use allocation of `BatchedPrefillScratch` and the accessor
- `produce_real_gemma4` [mod.rs:produce_real_gemma4]: single-token decode, timed wrapper over `produce_real_gemma4_inner`
- `prefill_chunk_real_gemma4` [prefill.rs:prefill_chunk_real_gemma4]: whole-chunk prefill, walks micro-batches of at most `MAX_PREFILL_BATCH`
- `prefill_micro_batch_gemma4_dense` [prefill.rs:prefill_micro_batch_gemma4_dense]: dense trunk micro-batch with no host readback between halves
- `prefill_micro_batch_gemma4` [prefill.rs:prefill_micro_batch_gemma4]: MoE micro-batch (attention pass, router readback, routed half)
- `encode_gemma4_routed_half_per_token` [prefill.rs:encode_gemma4_routed_half_per_token]: per-token routed half used by prefill
- `encode_gemma4_layer_attn_and_router`, `encode_gemma4_layer_qkv`, `encode_gemma4_layer_attend_and_router` [attn.rs:encode_gemma4_layer_attn_and_router]: decode attention and router encoding
- `router_offsets_gemma4` [attn.rs:router_offsets_gemma4]: INT8 router tensor offsets with dtype and size checks
- `encode_gemma4_layer_tail` / `encode_gemma4_layer_tail_dense` [attn.rs]: sandwich-norm tail of a layer (MoE and dense arms)
- `encode_gemma4_layer_attn_and_router_batched` [attn_batch.rs]: batched-GEMM attention half for `TURBOSPARK_BATCHED_GEMV`
- `encode_gemma4_layer_routed_moe` [moe.rs:encode_gemma4_layer_routed_moe]: per-token routed expert pass; re-exports `RoutedSlot`
- `encode_gemma4_layer_routed_moe_batched` [moe_batch.rs:encode_gemma4_layer_routed_moe_batched]: sub-batched routed pass for `TURBOSPARK_ROUTED_BATCH`
- `encode_gemma4_dense_ffn` / `_batched`, `encode_shared_expert_branch` / `_batched` [shared_expert.rs]: dense FFN arm and shared-expert branch
- `prefill_image_span_gemma4`, `prefill_chunk_gemma4_with_images`, `gemma4_span_refusal` [vision_prefill.rs]: image-span prefill and its refusal rules
- `span_attention_range` [vision_prefill.rs:span_attention_range]: sliding/full key range for a row inside an image span

## Invariants
- Acceptance is flag-based, not family-keyed: any install with `ffn_sandwich_norms` and `router_scaled` both set passes, and `ModelFamily` is not checked here [state.rs:build]
- Dense vs MoE is keyed on `arch.num_experts == 0`; MoE needs positive `top_k_experts` not above `gpu::MAX_STREAMED_EXPERTS` [state.rs:build]
- Decode positions must be sequential with the KV cache, and token ids must be below vocab [mod.rs:produce_real_gemma4_inner]
- Image placeholder positions are refused in single-token decode; they must go through `prefill_chunk` [mod.rs:produce_real_gemma4_inner]
- Router weights must be INT8 (dtype 5) with size exactly num_experts*hidden [attn.rs:router_offsets_gemma4]
- Empty prefill chunks are refused; scratch is ensured before any layer runs, so `batched()` may `expect` [prefill.rs:prefill_chunk_real_gemma4, state.rs:batched]
- Batched scratch is sized by the widest q_dim (`max(head_dim, full_head_dim)`) because full layers use a wider head [state.rs:ensure_batched]
- Routed pipeline depth is 1 [mod.rs:PendingRouted, prefill.rs:encode_gemma4_routed_half_per_token]
- Batched routed prefill refuses non-affine INT4 blobs, mapped expert residency, slot counts above `MAX_PREFILL_EXPERT_BINDINGS`, and the pilot probe, rather than silently falling back [moe_batch.rs:encode_gemma4_layer_routed_moe_batched]
- Batched attention refuses quantized KV layers (`--kv-bits`) [attn_batch.rs:encode_gemma4_layer_attn_and_router_batched]
- Image spans run whole across all layers; refused under `TURBOSPARK_BATCHED_GEMV`, `TURBOSPARK_ROUTED_BATCH`, quantized KV, or a sliding ring smaller than window + span length - 1 [vision_prefill.rs:gemma4_span_refusal]
- Image rows are written without the sqrt(hidden) embed scale; full layers stay causal inside a span [vision_prefill.rs:span_attention_range]

## Side Effects
- Metal buffer allocation at open (router scales, scratch) and lazily on first prefill (batched scratch) [state.rs:build, ensure_batched]
- Metal command encoding and commits per layer; routed-expert `pread` and expert-cache planning in the batched routed path [moe_batch.rs:encode_gemma4_layer_routed_moe_batched]
- Router readbacks to the host after each layer's commit [prefill.rs:prefill_micro_batch_gemma4]
- Mutates runner phase timers (`phases.calls`, `total_nanos`) and KV position [mod.rs:produce_real_gemma4]
- Optional residual capture and steering dispatches [mod.rs, prefill.rs: encode_resid_capture, encode_steering]
- Env-driven seams: `TURBOSPARK_PILOT_PROBE` router probe [attn.rs:encode_gemma4_pilot_probe]; `TURBOSPARK_BATCHED_GEMV`, `TURBOSPARK_ROUTED_BATCH` consulted via runner flags [vision_prefill.rs:gemma4_span_refusal]

## Verification
- Run: `cargo test -p turbospark-runtime`
