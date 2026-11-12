# DeepSeek2 Decode Flow

## Overview
The `deepseek2` family decode flow for `RealForwardRunner`: MLA attention over a compressed one-row KV cache (or plain MHA for the DeepSeek-VL2 trunk), a dense lead layer, and fine-grained MoE with a fused shared expert. The MoE half reuses the llama flow's routed phase 1/2, pipelined slot cache and mapped residency; the attention half, the softmax top-k without renormalization, and the shared-expert residual seed are specific to this family. State is allocated once at install open and validated against the manifest.

## Interface
- `RealForwardRunner::produce_real_deepseek2` [mod.rs:produce_real_deepseek2]: single-token decode entry; times the call into `phases`, then runs the inner flow
- `produce_real_deepseek2_inner` [mod.rs:produce_real_deepseek2_inner]: per-layer encode (input norm, attention, residual, post-attn norm, dense lead SwiGLU or router + commit + routed MoE), per-layer steering dispatch (`encode_steering`, dense-lead and MoE layers), final norm and raw-logit head (skipped when `skip_head` is set, as in prefill: KV still advances and residual capture records the skip)
- `layer_tensor` [mod.rs:layer_tensor]: builds `language_model.model.layers.{n}.{suffix}` tensor names
- `RealDeepseek2State` [state.rs:RealDeepseek2State]: MLA geometry, yarn rope table, shared-expert and router scratch, lead/dense/shared widths; `build` and `build_mha` constructors
- `encode_mla_attention_block` [attn.rs:encode_mla_attention_block]: q proj, in-place q_pe rope, kv_a written straight into the cache row, in-place latent norm and k_pe rope, absorb, MQA attention, v-combine, o_proj
- `encode_mha_attention_block` [attn.rs:encode_mha_attention_block]: plain multi-head attention for the VL2 trunk (NeoX half-split rope, standard KV layout)
- `Q_PROJ`, `KV_A_PROJ`, `KV_A_NORM`, `KV_B_PROJ`, `O_PROJ` [attn.rs:Q_PROJ]: attention tensor suffixes
- `softmax_topk_no_renorm` [moe.rs:softmax_topk_no_renorm]: softmax over all experts, top-k, weights unrenormalized
- `encode_shared_expert` [moe.rs:encode_shared_expert]: dense SwiGLU at the fused width into `shared_out`
- `encode_deepseek2_layer_moe` [moe.rs:encode_deepseek2_layer_moe]: shared expert, host top-k, expert streaming or mapped lookup, routed phase 1/2, residual add; returns the used cache slots

## Invariants
- Positions must be sequential (`position == kv.position()`) and the token id must be below the vocab, else `Unsupported` [mod.rs:produce_real_deepseek2_inner]
- State build refuses any other family and any of ffnSandwichNorms, routerScaled, embeddingScaledBySqrtHidden, attnOutputGate, sharedExpertGated, attentionKEqV, and a nonzero final logit softcap [state.rs:RealDeepseek2State::build]
- The MLA shape requires layer mask 5 on every layer, `q_lora_rank == 0`, `kv_lora == 512`, `nope % 32`, `kv_lora % 32`, `v_dim % 8`, `rope_dim % 2` [state.rs:RealDeepseek2State::build]
- The non-MLA shape requires mask 1 on every layer, no rope scaling [state.rs:build_mha]
- `top_k` must not exceed `gpu::PHASE2_FIXED_SLOTS`, experts must be positive, and the dense lead must leave at least one MoE layer [state.rs:RealDeepseek2State::build]
- Router weights are full-softmax values with no renormalization and no scale factor; slot order is router ranking order [moe.rs:softmax_topk_no_renorm, moe.rs:encode_deepseek2_layer_moe]
- The shared expert runs before the routed phases because phase 2's residual seed must hold its output [moe.rs:encode_deepseek2_layer_moe]
- Routed blob index is `layer - state.lead` [moe.rs:encode_deepseek2_layer_moe]
- MLA kv_b must be Q8_0 with exact shape `(heads*(nope+v_dim), kv_lora)` and exact packed size, with overflow-checked arithmetic [attn.rs:validate_q8_0_matrix]
- The YaRN magnitude (`yarn_mscale()`) is computed but deliberately not applied: `rope_mscale` is pinned to 1.0 for cross-engine parity with llama.cpp, whose factors cancel for this checkpoint [state.rs:RealDeepseek2State::build]
- Producer returns raw logits (no softmax, no softcap); caller logits length must equal vocab [mod.rs:produce_real_deepseek2_inner]
- Missing tensors fail at build with `MissingTensor` [state.rs:RealDeepseek2State::build]

## Side Effects
- Metal: begins a labeled pass, commits and waits mid-layer after the router so the host can rank experts, final commit and wait [mod.rs:produce_real_deepseek2_inner]
- Advances the KV position and records residual capture [mod.rs:produce_real_deepseek2_inner]
- Encodes steering vectors into the residual after each layer when steering is set [mod.rs:produce_real_deepseek2_inner]
- Writes the vision row into `scratch.x` for flat-injected image positions [mod.rs:produce_real_deepseek2_inner]
- Reads router logits back to the host; writes routing weights buffer [moe.rs:encode_deepseek2_layer_moe]
- Expert streaming I/O via `PreadExpertStreamer`, updating cache and I/O phase counters; records router histogram [moe.rs:encode_deepseek2_layer_moe]
- Allocates Metal buffers once at build [state.rs:RealDeepseek2State::build]

## Verification
- Run: `cargo test -p turbospark-runtime`
