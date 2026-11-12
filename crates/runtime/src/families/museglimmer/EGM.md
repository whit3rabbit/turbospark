# Muse Glimmer Decode Flow

## Overview
The `muse_glimmer` decode flow for `RealForwardRunner`: a dense 52-layer GQA stack with a three-sliding/one-full window pattern, sandwich norms, no-scale per-head q/k norms with a second Q scale, NoPE on full-attention layers, a separate output gate projection, an embedding norm, and an untied softcapped head. There are no experts or router, so a token or a prefill micro-batch encodes into one command buffer. State is allocated once at install open and refuses installs that disagree with the family's shape.

## Interface
- `RealForwardRunner::produce_real_muse` [mod.rs:produce_real_muse]: single-token decode entry; times the call into `phases`
- `produce_real_muse_inner` [mod.rs:produce_real_muse_inner]: embed (or image row), embed norm, per-layer centered norms, attention, post-attn sandwich norm, MLP, steering, capture, plain final norm, lm_head, output multiplier, softcap
- `layer_tensor` [mod.rs:layer_tensor]: builds `language_model.model.layers.{n}.{suffix}` names
- `RealMuseState` [state.rs:RealMuseState]: `rotated_pairs` and the full-width `attn_gate` buffer; `build` validates the install
- `RMS_EPS`, `POST_NORM_EPS`, `QK_SCALE_FACTOR`, `OUTPUT_MULTIPLIER` [state.rs:RMS_EPS]: the four published scalars (1e-5, 1e-8, 3.87, 26^-0.5)
- `encode_attention_block` [attn.rs:encode_attention_block]: q/k/v and gate GEMVs, no-scale per-head q/k norm, Q scale, rope on sliding layers only, windowed attention, sigmoid gate, o_proj
- `encode_mlp_block` [mlp.rs:encode_mlp_block]: pre-FFN centered norm, gate/up, SiLU multiply, down, post-FFN centered norm, residual add
- `RealForwardRunner::prefill_chunk_real_muse` [prefill.rs:prefill_chunk_real_muse]: chunked prefill in `MAX_PREFILL_BATCH` micro-batches
- `prefill_micro_batch_muse_inner` [prefill.rs:prefill_micro_batch_muse_inner]: one command buffer for all layers of the micro-batch; head only on the last chunk

## Invariants
- Positions must be sequential against `kv.position()` and token ids below vocab [mod.rs:produce_real_muse_inner, prefill.rs:prefill_micro_batch_muse_inner]
- The four per-layer norms are centered `(1 + w)` and the final norm is plain `w`, selected per tensor [mod.rs:produce_real_muse_inner]
- Epsilon is 1e-5 everywhere except the two post norms, which use 1e-8 [state.rs:POST_NORM_EPS, mod.rs:produce_real_muse_inner]
- The output multiplier is applied before the softcap; the order is the function [mod.rs:produce_real_muse_inner]
- Image positions take the tower row instead of the embedding lookup and skip `embed_norm` [mod.rs:produce_real_muse_inner]
- Build requires family MuseGlimmer, `ffnSandwichNorms` true, a positive softcap, untied embeddings, and none of routerScaled / embeddingScaledBySqrtHidden / attnOutputGate / sharedExpertGated / ropeNeoxSubdim / attentionKEqV [state.rs:RealMuseState::build]
- `head_dim` must equal `full_head_dim` and the rotated pair count (`partial_rotary_factor`) must be positive and at most half the head [state.rs:RealMuseState::build]
- Dense only; mask values 0 or 1 with at least one sliding layer and a positive window [state.rs:RealMuseState::build]
- Full layers are NoPE: `fullRopeTheta` must be 0 and `ropeTheta` positive; rope runs only on sliding layers [state.rs:RealMuseState::build, attn.rs:encode_attention_block]
- q/k norm weights must not exist in the install (no-scale norms); all other per-layer tensors, lm_head, norm and embedding must exist at open [state.rs:RealMuseState::build]
- `QK_SCALE_FACTOR` is kept separate from `attention_scale`, not folded in [state.rs:QK_SCALE_FACTOR]
- `TURBOSPARK_BATCHED_GEMV` prefill is refused by name [prefill.rs:prefill_chunk_real_muse]
- Producer returns softcapped raw logits and the caller's slice length must equal vocab [mod.rs:produce_real_muse_inner]

## Side Effects
- Metal: one labeled pass per token or micro-batch, committed and waited at the end [mod.rs:produce_real_muse_inner, prefill.rs:prefill_micro_batch_muse_inner]
- Allocates one Metal buffer at build [state.rs:RealMuseState::build]
- Host write of the vision row into `scratch.x` before commit [mod.rs:produce_real_muse_inner]
- Advances the KV position and records residual capture and FFN histogram passes [mod.rs:produce_real_muse_inner, prefill.rs:prefill_micro_batch_muse_inner]
- Quantizes KV staging rows in place on TurboQuant layers [attn.rs:encode_attention_block]

## Verification
- Run: `cargo test -p turbospark-runtime`
