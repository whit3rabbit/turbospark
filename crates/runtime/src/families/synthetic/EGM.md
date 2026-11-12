# Synthetic Short-Name Forward Flow

## Overview
The fallback decode path of `RealForwardRunner` for checkpoints with short tensor names (`q_proj{layer}`, `embed_lm_head`) that match no real family. `produce_inner` is the single funnel for every per-token advance: it dispatches to a family flow when one is loaded, and otherwise runs this INT4 GPU flow. `layer.rs` encodes one layer, with dense FFN, GPU streamed-expert MoE, or a host-bridged MoE.

## Interface
- `RealForwardRunner::produce_inner` [mod.rs:produce_inner]: per-token entry; dispatches to the `produce_real_*` family flows, else runs the synthetic flow
- `RealForwardRunner::moe_ffn_host` [mod.rs:moe_ffn_host]: GPU router GEMV, host top-k softmax, per-expert `compute::run_ffn`, weighted sum
- `RealForwardRunner::encode_synthetic_layer` [layer.rs:encode_synthetic_layer]: one layer's attention plus FFN into a `PassEncoder`
- Subdirectories: None

## Invariants
- `note_eager_advance` runs before any dispatch, so callers bypassing the trait `produce` still invalidate lazy checkpoints [mod.rs:produce_inner]
- `--kv-bits` is refused here because K and V alias one buffer and TurboQuant uses different K/V codebooks [mod.rs:produce_inner]
- Position must equal `kv.position()`, `attention_k_eq_v` must be set, token must be < vocab [mod.rs:produce_inner]
- Logit length must equal vocab; logits are raw, with softcap applied only when positive [mod.rs:produce_inner]
- Sliding layers derive `kv_start` and `active_ring` from the layer mask and ring capacity [layer.rs:encode_synthetic_layer]
- Expert slots keep router ranking order; routing weights are zero-padded to `MAX_STREAMED_EXPERTS` [layer.rs:encode_synthetic_layer]
- MoE branches commit the pass to read router logits on the host, then open a new pass [layer.rs:encode_synthetic_layer]
- `skip_head` skips norm and LM head and returns without writing logits [mod.rs:produce_inner]

## Side Effects
- Metal encoding, `commit_and_wait`, and buffer writes/reads via the `gpu` crate [mod.rs:produce_inner, layer.rs:encode_synthetic_layer]
- Advances the KV cache with `kv.advance()` [mod.rs:produce_inner]
- Pread expert streaming into slot buffers via `PreadExpertStreamer` [layer.rs:encode_synthetic_layer]
- No env vars read here

## Verification
- No tests live in this directory. The flow is exercised by `the_synthetic_flows_arithmetic_is_frozen` in `crates/runtime/tests/real_forward_spark.rs`, `real_forward_lfm3.rs`, `real_forward_muse.rs`, and `real_forward_qwen4.rs` (for example `cargo test -p turbospark-runtime --test real_forward_spark`)
