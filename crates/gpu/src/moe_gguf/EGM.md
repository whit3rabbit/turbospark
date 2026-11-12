# gpu moe_gguf

## Overview
Host-side dispatch for the streamed routed-expert decode path over GGUF block-quantized expert blobs: phase 1 (gate/up/activation) and phase 2 (down/reduce) per quant type. Kernels are in `shaders/moe_gguf.metal`.

## Interface
- Core: [mod.rs:moe_gguf_source], [mod.rs:encode_phase1], [mod.rs:encode_phase2], [mod.rs:PHASE2_FIXED_SLOTS].
- K-quants and Q8_0: [kquants.rs:encode_moe_phase1_q4_k], [kquants.rs:encode_moe_phase2_q4_k], [kquants.rs:encode_moe_phase1_q8_0], [kquants.rs:encode_moe_phase2_q8_0], [kquants.rs:encode_moe_phase2_q6_k].
- IQ codebook types: [iq.rs:encode_moe_phase1_iq2_s], [iq.rs:encode_moe_phase1_iq2_xxs], [iq.rs:encode_moe_phase1_iq1_m], [iq.rs:encode_moe_phase1_iq3_xxs], [iq.rs:encode_moe_phase1_iq4_xs], [iq.rs:encode_moe_phase2_iq4_nl].
- MXFP4: [mxfp4.rs:MXFP4_BLOCK_ELEMS], [mxfp4.rs:MXFP4_BLOCK_BYTES], [mxfp4.rs:mxfp4_row_bytes], [mxfp4.rs:Mxfp4Activation] (`PLAIN`, `GPT_OSS`), [mxfp4.rs:encode_moe_phase1_mxfp4], [mxfp4.rs:encode_moe_phase2_mxfp4].
- Q2_0 (Qwen4Exp top-10 down): [q2_0.rs:encode_moe_phase1_q2_0], [q2_0.rs:encode_moe_phase2_q2_0_top10].

## Invariants
- Phase 2 handles at most `PHASE2_FIXED_SLOTS` (8) routed slots; `top_k` above that is rejected by assert [mod.rs:encode_phase2]. The Q2_0 path is separate and requires exactly `top_k=10` [q2_0.rs:encode_moe_phase2_q2_0_top10].
- Routed expert slots are dispatched in router ranking order; slot order is reduction order (workspace rule).

## Side Effects
- Encodes compute dispatches onto a caller-supplied `PassEncoder`; no commit here [mod.rs:encode_phase1].

## Verification
- Run: `cargo test -p turbospark-gpu`
