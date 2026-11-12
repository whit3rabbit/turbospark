# turbospark-compute (src)

## Overview
Portable FP32 CPU reference kernels that serve as the numerical ground truth for the Metal kernels in `turbospark-gpu`: attention, RoPE, RMSNorm, GDN, MLA, MoE FFN, PLE, QSA, quantization codecs (int1/2/4/8, GGUF K-quants, IQ, MXFP4), TurboQuant KV codec, vision and Whisper/encoder towers, steering, and tolerance helpers. Pure safe Rust.

## Interface
- Marker type: [lib.rs:ComputeStrategy], re-export [lib.rs:TokenId].
- Attention and positions: [attention.rs:causal_attention], [attention.rs:causal_attention_with_sinks], [attention.rs:indexed_attention], [rope.rs:rope_paired], [rope.rs:rope_neox], [rope.rs:rope_mrope_interleaved], [rope.rs:yarn_frequencies], [rope.rs:YarnSpec], [mla.rs:mla_attention_decode], [mla.rs:mla_absorb_q], [kv_quant_attention.rs:causal_attention_tq], [dash.rs:DashParams].
- Norm, gating, sampling: [rms_norm.rs:rms_norm], [rms_norm.rs:rms_norm_grouped], [rms_norm.rs:rms_norm_centered] (x * (1 + w)), [wht.rs:wht], [gating.rs:sigmoid_gate_mul], [gating.rs:sigmoid_head_gate_mul], [gating.rs:split_qkv], [sampling.rs:logit_softcap_softmax].
- Blocks: [gdn.rs:GdnReference], [gdn.rs:GdnDims], [moe.rs:run_ffn], [moe.rs:apply_streamed_routed], [ple.rs:ple_gate], [qsa_indexer.rs:select_blocks], [hyper_connection.rs:hc_mix], [short_conv.rs:short_conv_reference].
- Quantization: [quant.rs:quantize_int4_affine], [quant.rs:dequant_int4_gemv], [quant.rs:GROUP_SIZE], [quant_1bit.rs:dequant_int1_gemv], [quant_2bit.rs:dequant_int2_gemv], [quant_gguf_iq.rs] (`dequantize_iq*`, `dequant_iq*_gemv`), [quant_gguf_mxfp4.rs:dequantize_mxfp4], generated codebooks [quant_gguf_iq_tables.rs:IQ4NL_VALUES], [quant_gguf_iq_lowbit_tables.rs:IQ2XXS_GRID], K-quant and Q-formats in [quant_gguf](./quant_gguf/EGM.md).
- KV codec: [kv_quant.rs:quantize_row], [kv_quant.rs:rht_forward], [kv_quant.rs:rht_inverse], [kv_quant.rs:KEY_SEED], [kv_quant.rs:VALUE_SEED], [kv_quant.rs:QuantizedRow].
- Towers and audio: [encoder.rs:encoder_block_forward], [encoder.rs:EncoderReferenceConfig], [encoder.rs:cls_pool_and_normalize], [encoder.rs:cosine_similarity], [vision.rs:bidirectional_attention], [vision_gemma4.rs:gemma4_vision_forward], [uocr2_tower.rs:conv2d], [uocr2_tower.rs:relpos_attention], [whisper.rs:whisper_encoder_layer], [whisper.rs:whisper_decoder_layer_step].
- Steering: [steering.rs:steer_in_place], [steering_extract.rs:difference_of_means], [steering_extract.rs:CaptureSet], [steering_extract.rs:layer_stats], [steering_extract.rs:derived_alpha_cap].
- Tolerance: [tolerance.rs:rel_error], [tolerance.rs:max_abs_diff], [tolerance.rs:Tolerance] (`IDENTITY` 1e-5, `FP16_REDUCTION` 5e-3, `FP16_CHAINED_REDUCTION` 1e-2, `QUANT_INT4` 1.5e-3, `QUANT_INT8` 1e-3).
- Child: [quant_gguf](./quant_gguf/EGM.md).

## Invariants
- `#![forbid(unsafe_code)]` [lib.rs].
- Reference kernels are FP32 ground truth for GPU parity; GPU kernels are checked against them, not the other way round [lib.rs] (module doc).
- INT4/INT8 affine quantization uses 64-element groups with scales/biases carried as raw bit patterns; `n` must equal `packed.len() * 2` for int4 [quant.rs:GROUP_SIZE], [quant.rs:dequantize_int4_affine]. The 1-bit and 2-bit formats use 128-element groups [quant_1bit.rs:BONSAI_GROUP_SIZE], [quant_2bit.rs:TERNARY_GROUP_SIZE].
- IQ codebook tables are generated from the repo-root `scripts/ggml_tables.c` (not under crates/compute); do not hand-edit [quant_gguf_iq_lowbit_tables.rs], [quant_gguf_iq_tables.rs].
- `quant_2bit` shares `quant_1bit`'s f16 conversion pair instead of re-exporting a second copy [lib.rs] (comment above `quant_2bit` re-export).
- `turbospark-core` is imported under the alias `foundation` [lib.rs] (module doc).

## Side Effects
- None (pure computation, no I/O or globals).

## Verification
- Run: `cargo test -p turbospark-compute`
