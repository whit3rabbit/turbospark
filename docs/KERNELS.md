# Metal compute kernels and quantization reference

This page inventories the Metal kernels, supported quantization formats, and
kernel-level optimizations in `turbospark`. Use it with
[`TESTING.md`](TESTING.md) when a kernel change needs a parity gate.

---

## 1. Overview

- **Total Metal Compute Kernels**: 98 kernels across 17 `.metal` shader files in `crates/gpu/src/shaders/`.
- **Host Dispatch Layer**: `crates/gpu/src/` (macOS Metal runtime with zero-copy buffer bindings and address-keyed pipeline caching).
- **CPU Reference Implementations**: `crates/compute/src/` (bit-accurate and parity-checked CPU models for testing and verification).
- **Supported Quantization Formats**: 12 formats spanning MLX affine formats, sub-byte representations, GGUF K-quants, and GGUF IQ codebooks.

---

## 2. Supported Quantizations Matrix

Quantization support is specialized per operation role (dense GEMV, embedding lookup, MoE phase 1 gate/up, MoE phase 2 down-reduce).

| Format / Type | Container | Nominal Bits | Block / Group Size | Companion Dtype | Matrix GEMV | Embedding Lookup | MoE Phase 1 (Gate/Up) | MoE Phase 2 (Down) | Shader Source | Host Dispatch |
|---|---|---:|---:|---|:---:|:---:|:---:|:---:|---|---|
| **INT1 (Affine)** | MLX / Custom | 1.0 | 128 | FP16 scale + bias | Yes | Yes | No | No | [`dequant_1bit.metal`](../crates/gpu/src/shaders/dequant_1bit.metal) | [`dequant_1bit_gemv.rs`](../crates/gpu/src/dequant_1bit_gemv.rs) |
| **INT2 (Ternary)** | MLX / Custom | 2.0 | 128 | FP16 scale + bias | Yes | Yes | Yes | Yes | [`dequant_2bit.metal`](../crates/gpu/src/shaders/dequant_2bit.metal) | [`dequant_2bit_gemv.rs`](../crates/gpu/src/dequant_2bit_gemv.rs) |
| **INT4 (Affine)** | MLX / `.gturbo` | 4.0 | 64 | BF16 scale + bias | Yes | Yes | Yes | Yes | [`dequant_int4.metal`](../crates/gpu/src/shaders/dequant_int4.metal) | [`dequant_int4_gemv.rs`](../crates/gpu/src/dequant_int4_gemv.rs) |
| **INT8 (Affine)** | MLX / `.gturbo` | 8.0 | 64 | BF16 scale + bias | Yes | No | Yes (Shared) | No | [`dequant_int8.metal`](../crates/gpu/src/shaders/dequant_int8.metal) | [`dequant_int8_gemv.rs`](../crates/gpu/src/dequant_int8_gemv.rs) |
| **Q4_K** | GGUF | 4.5 | 256 (32 sub) | FP16 / 6-bit scales | Yes | Yes | Yes | Yes | [`dequant_q4_k.metal`](../crates/gpu/src/shaders/dequant_q4_k.metal) | [`dequant_q4_k_gemv.rs`](../crates/gpu/src/dequant_q4_k_gemv.rs) |
| **Q5_K** | GGUF | 5.5 | 256 (32 sub) | FP16 / 6-bit scales | Yes | No | No | No | [`dequant_q5_k.metal`](../crates/gpu/src/shaders/dequant_q5_k.metal) | [`dequant_q5_k_gemv.rs`](../crates/gpu/src/dequant_q5_k_gemv.rs) |
| **Q6_K** | GGUF | 6.56 | 256 (16 sub) | FP16 scales | Yes | Yes | No | Yes | [`dequant_q6_k.metal`](../crates/gpu/src/shaders/dequant_q6_k.metal) | [`dequant_q6_k_gemv.rs`](../crates/gpu/src/dequant_q6_k_gemv.rs) |
| **Q8_0** | GGUF | 8.5 | 32 | FP16 scale | Yes | Yes | Yes | Yes | [`dequant_q8_0.metal`](../crates/gpu/src/shaders/dequant_q8_0.metal) | [`dequant_q8_0_gemv.rs`](../crates/gpu/src/dequant_q8_0_gemv.rs) |
| **Q2_K** | GGUF | 2.63 | 256 (16 sub) | FP16 scale/min | Yes | No | No | No | [`dequant_q2_k.metal`](../crates/gpu/src/shaders/dequant_q2_k.metal) | [`dequant_q2_k_gemv.rs`](../crates/gpu/src/dequant_q2_k_gemv.rs) |
| **IQ2_XXS / IQ2_XS / IQ1_S / IQ3_S / IQ2_S** | GGUF | mixed | 256 | Codebook + FP16 scale | Yes | No | No | No | [`dequant_iq.metal`](../crates/gpu/src/shaders/dequant_iq.metal) | [`dequant_iq_gemv.rs`](../crates/gpu/src/dequant_iq_gemv.rs) |
| **IQ1_M** | GGUF | mixed | 256 | Codebook + packed scale/delta | Yes | Yes | No | No | [`dequant_iq.metal`](../crates/gpu/src/shaders/dequant_iq.metal) | [`dequant_iq_gemv.rs`](../crates/gpu/src/dequant_iq_gemv.rs) |
| **IQ3_XXS** | GGUF | 3.06 | 256 | Codebook + FP16 scale | Yes | No | Yes | No | [`dequant_iq.metal`](../crates/gpu/src/shaders/dequant_iq.metal) | [`dequant_iq_gemv.rs`](../crates/gpu/src/dequant_iq_gemv.rs) |
| **IQ4_XS** | GGUF | 4.25 | 256 | Codebook + FP16 scale | Yes | No | Yes | No | [`dequant_iq.metal`](../crates/gpu/src/shaders/dequant_iq.metal) | [`dequant_iq_gemv.rs`](../crates/gpu/src/dequant_iq_gemv.rs) |
| **IQ4_NL** | GGUF | 4.5 | 32 | Non-linear table + FP16 | Yes | No | No | Yes | [`dequant_iq.metal`](../crates/gpu/src/shaders/dequant_iq.metal) | [`dequant_iq_gemv.rs`](../crates/gpu/src/dequant_iq_gemv.rs) |
| **MXFP4** | GGUF / OCP | 4.0 | 32 | E2M1 FP8 scale | No | No | Yes | Yes | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | [`moe_gguf/mxfp4.rs`](../crates/gpu/src/moe_gguf/mxfp4.rs) |

---

## 3. Complete Kernel Catalog by Domain

### 3.1 Quantized Matrix-Vector Multiplication & Embedding

Dense projections (Q/K/V, output projection, FFN gate/up/down on dense models) and token embedding lookup.

| Kernel Name | Shader File | Line | Description & Optimizations | Host Dispatch |
|---|---|---:|---|---|
| `embed_lookup_int1` | [`dequant_1bit.metal`](../crates/gpu/src/shaders/dequant_1bit.metal) | 117 | 1-bit packed embedding row extraction with scale/bias unpack | [`dequant_1bit_gemv.rs`](../crates/gpu/src/dequant_1bit_gemv.rs) |
| `dequant_int1_gemv_simd` | [`dequant_1bit.metal`](../crates/gpu/src/shaders/dequant_1bit.metal) | 54 | 1-bit SIMD matrix-vector product with affine scale/bias | [`dequant_1bit_gemv.rs`](../crates/gpu/src/dequant_1bit_gemv.rs) |
| `dequant_int1_gemv_symmetric_simd` | [`dequant_1bit.metal`](../crates/gpu/src/shaders/dequant_1bit.metal) | 150 | Symmetric (-1, +1) 1-bit GEMV without additive zero-point | [`dequant_1bit_gemv.rs`](../crates/gpu/src/dequant_1bit_gemv.rs) |
| `embed_lookup_int2` | [`dequant_2bit.metal`](../crates/gpu/src/shaders/dequant_2bit.metal) | 128 | 2-bit (ternary) embedding lookup with 4-weights-per-byte unpacking | [`dequant_2bit_gemv.rs`](../crates/gpu/src/dequant_2bit_gemv.rs) |
| `dequant_int2_gemv_simd` | [`dequant_2bit.metal`](../crates/gpu/src/shaders/dequant_2bit.metal) | 65 | 2-bit SIMD GEMV for Ternary-Bonsai architectures | [`dequant_2bit_gemv.rs`](../crates/gpu/src/dequant_2bit_gemv.rs) |
| `embed_lookup_int4` | [`dequant_int4.metal`](../crates/gpu/src/shaders/dequant_int4.metal) | 63 | INT4 nibble unpacking and affine dequantization for embedding | [`dequant_int4_gemv.rs`](../crates/gpu/src/dequant_int4_gemv.rs) |
| `dequant_int4_gemv_simd` | [`dequant_int4.metal`](../crates/gpu/src/shaders/dequant_int4.metal) | 174 | High-throughput INT4 GEMV using SIMD shuffle reductions & function constants | [`dequant_int4_gemv.rs`](../crates/gpu/src/dequant_int4_gemv.rs) |
| `dequant_int4_qkv_gemv_simd` | [`dequant_int4.metal`](../crates/gpu/src/shaders/dequant_int4.metal) | 194 | Fused 3-way Q/K/V matrix-vector projection in a single kernel | [`dequant_int4_gemv.rs`](../crates/gpu/src/dequant_int4_gemv.rs) |
| `dequant_int4_gemm_simd` | [`dequant_int4_batch.metal`](../crates/gpu/src/shaders/dequant_int4_batch.metal) | 88 | Batched INT4 GEMM for multi-token verification, drafter and prefill passes; `FC_GEMM_R` picks rows-per-SIMD-group per batch width (`best_row_block`, measured in `docs/BATCHED_PREFILL.md`) | [`dequant_int4_batch.rs`](../crates/gpu/src/dequant_int4_batch.rs) |
| `dequant_int4_gemm_mma` | [`dequant_int4_mma.metal`](../crates/gpu/src/shaders/dequant_int4_mma.metal) | 88 | Experimental `simdgroup_matrix` MMA implementation for batched GEMM | [`dequant_int4_batch.rs`](../crates/gpu/src/dequant_int4_batch.rs) |
| `dequant_int8_gemv_simd` | [`dequant_int8.metal`](../crates/gpu/src/shaders/dequant_int8.metal) | 66 | INT8 matrix-vector multiplication with BF16 scales and biases | [`dequant_int8_gemv.rs`](../crates/gpu/src/dequant_int8_gemv.rs) |
| `shared_int8_gate_up_act_simd` | [`dequant_int8.metal`](../crates/gpu/src/shaders/dequant_int8.metal) | 109 | Fused gate + up projection with SiLU activation for INT8 shared experts | [`dequant_int8_gemv.rs`](../crates/gpu/src/dequant_int8_gemv.rs) |
| `embed_lookup_q4_k` | [`dequant_q4_k.metal`](../crates/gpu/src/shaders/dequant_q4_k.metal) | 142 | GGUF Q4_K block dequantization for embedding token lookup | [`dequant_q4_k_gemv.rs`](../crates/gpu/src/dequant_q4_k_gemv.rs) |
| `dequant_q4_k_gemv_simd` | [`dequant_q4_k.metal`](../crates/gpu/src/shaders/dequant_q4_k.metal) | 111 | GGUF Q4_K matrix-vector multiplication with 2-level hierarchical scales | [`dequant_q4_k_gemv.rs`](../crates/gpu/src/dequant_q4_k_gemv.rs) |
| `dequant_q5_k_gemv_simd` | [`dequant_q5_k.metal`](../crates/gpu/src/shaders/dequant_q5_k.metal) | 108 | GGUF Q5_K matrix-vector multiplication (5-bit packed quants) | [`dequant_q5_k_gemv.rs`](../crates/gpu/src/dequant_q5_k_gemv.rs) |
| `embed_lookup_q6_k` | [`dequant_q6_k.metal`](../crates/gpu/src/shaders/dequant_q6_k.metal) | 141 | GGUF Q6_K block dequantization for embedding token lookup | [`dequant_q6_k_gemv.rs`](../crates/gpu/src/dequant_q6_k_gemv.rs) |
| `dequant_q6_k_gemv_simd` | [`dequant_q6_k.metal`](../crates/gpu/src/shaders/dequant_q6_k.metal) | 97 | GGUF Q6_K matrix-vector multiplication (6-bit quants with 8-bit scales) | [`dequant_q6_k_gemv.rs`](../crates/gpu/src/dequant_q6_k_gemv.rs) |
| `embed_lookup_q8_0` | [`dequant_q8_0.metal`](../crates/gpu/src/shaders/dequant_q8_0.metal) | 74 | GGUF Q8_0 block dequantization for embedding token lookup | [`dequant_q8_0_gemv.rs`](../crates/gpu/src/dequant_q8_0_gemv.rs) |
| `dequant_q8_0_gemv_simd` | [`dequant_q8_0.metal`](../crates/gpu/src/shaders/dequant_q8_0.metal) | 35 | GGUF Q8_0 standard linear GEMV with SIMD reduction | [`dequant_q8_0_gemv.rs`](../crates/gpu/src/dequant_q8_0_gemv.rs) |
| `dequant_iq3_xxs_gemv_simd` | [`dequant_iq.metal`](../crates/gpu/src/shaders/dequant_iq.metal) | 321 | Sub-4-bit IQ3_XXS codebook lookup and matrix multiplication | [`dequant_iq_gemv.rs`](../crates/gpu/src/dequant_iq_gemv.rs) |
| `dequant_iq4_xs_gemv_simd` | [`dequant_iq.metal`](../crates/gpu/src/shaders/dequant_iq.metal) | 304 | GGUF IQ4_XS codebook lookup GEMV | [`dequant_iq_gemv.rs`](../crates/gpu/src/dequant_iq_gemv.rs) |
| `dequant_iq4_nl_gemv_simd` | [`dequant_iq.metal`](../crates/gpu/src/shaders/dequant_iq.metal) | 287 | Non-linear 4-bit table dequantization GEMV | [`dequant_iq_gemv.rs`](../crates/gpu/src/dequant_iq_gemv.rs) |

---

### 3.2 Mixture of Experts (MoE Routing, Decode & Prefill)

Kernels executing the MoE routing, top-k selection, and two-phase expert computation (Phase 1: Gate + Up with activation, Phase 2: Down projection with routing-weight reduction).

| Kernel Name | Shader File | Line | Description & Optimizations | Host Dispatch |
|---|---|---:|---|---|
| `router_gemv_gemma4_r4` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 142 | Router projection with 4 rows per threadgroup (INT4 weights) | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `router_gemv_bf16_r4` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 162 | BF16 router projection for unquantized routing heads | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `router_topk_select_k8` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 190 | Serial bitonic top-8 selection and softmax normalization | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `router_topk_select_k8_par` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 291 | Parallel reduction top-8 selector for large expert counts | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `router_topk_select_sqrtsoftplus_k6` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 620 | Top-6 selector with sqrt-softplus scoring activation | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `router_topk_select_sqrtsoftplus_k6_par` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 675 | Parallel top-6 selector with sqrt-softplus activation | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `router_hash_weights_k6` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 727 | Hash-based expert weight routing for DeepSeek/specialized MoEs | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `router_hash_weights_k6_par` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 751 | Parallel hash-based expert routing | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `moe_phase1_gate_up_act_u16load` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 567 | INT4 gate+up projection with GeLU/SiLU; loads expert blobs zero-copy | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `moe_phase1_gate_up_act_subset_u16load` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 585 | Subset gate+up projection for routed micro-batches | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `moe_phase2_down_reduce_k8` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 1072 | Down projection across top-8 experts with fused weighted accumulation | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `moe_phase1_gate_up_act_int2` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 974 | 2-bit (Ternary) MoE gate+up projection | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `moe_phase1_gate_up_act_subset_int2` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 993 | 2-bit subset MoE gate+up projection | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `moe_phase2_down_reduce_int2_k6` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 1033 | 2-bit MoE down projection with top-6 reduction | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `dsv4_prefill_moe_phase1_pairs_int2` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 1138 | DeepSeek-V4 paired expert gate+up INT2 prefill kernel | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `dsv4_prefill_moe_down_pairs_int2` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 1184 | DeepSeek-V4 paired expert down INT2 prefill kernel | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `dsv4_prefill_moe_reduce_pairs_k6` | [`moe.metal`](../crates/gpu/src/shaders/moe.metal) | 1227 | DeepSeek-V4 paired expert top-6 reduction kernel | [`moe_decode.rs`](../crates/gpu/src/moe_decode.rs) |
| `moe_phase1_gate_up_act_q8_0` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 97 | GGUF Q8_0 MoE Phase 1 gate+up with SiLU activation | [`moe_gguf/kquants.rs`](../crates/gpu/src/moe_gguf/kquants.rs) |
| `moe_phase2_down_reduce_k8_q8_0` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 129 | GGUF Q8_0 MoE Phase 2 down projection + weighted reduce | [`moe_gguf/kquants.rs`](../crates/gpu/src/moe_gguf/kquants.rs) |
| `moe_phase1_gate_up_act_q4_k` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 179 | GGUF Q4_K MoE Phase 1 gate+up with SiLU activation | [`moe_gguf/kquants.rs`](../crates/gpu/src/moe_gguf/kquants.rs) |
| `moe_phase2_down_reduce_k8_q4_k` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 213 | GGUF Q4_K MoE Phase 2 down projection + weighted reduce | [`moe_gguf/kquants.rs`](../crates/gpu/src/moe_gguf/kquants.rs) |
| `moe_phase2_down_reduce_k8_q6_k` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 396 | GGUF Q6_K MoE Phase 2 down projection + weighted reduce | [`moe_gguf/kquants.rs`](../crates/gpu/src/moe_gguf/kquants.rs) |
| `moe_phase1_gate_up_act_iq3_xxs` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 276 | GGUF IQ3_XXS sub-4-bit codebook MoE Phase 1 gate+up | [`moe_gguf/iq.rs`](../crates/gpu/src/moe_gguf/iq.rs) |
| `moe_phase1_gate_up_act_iq4_xs` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 309 | GGUF IQ4_XS sub-4-bit codebook MoE Phase 1 gate+up | [`moe_gguf/iq.rs`](../crates/gpu/src/moe_gguf/iq.rs) |
| `moe_phase2_down_reduce_k8_iq4_nl` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 343 | GGUF IQ4_NL non-linear table MoE Phase 2 down projection | [`moe_gguf/iq.rs`](../crates/gpu/src/moe_gguf/iq.rs) |
| `moe_phase1_gate_up_act_mxfp4` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 561 | GGUF / OCP MXFP4 (E2M1 FP8-scaled) MoE Phase 1 gate+up | [`moe_gguf/mxfp4.rs`](../crates/gpu/src/moe_gguf/mxfp4.rs) |
| `moe_phase2_down_reduce_k8_mxfp4` | [`moe_gguf.metal`](../crates/gpu/src/shaders/moe_gguf.metal) | 608 | GGUF / OCP MXFP4 MoE Phase 2 down projection + reduce | [`moe_gguf/mxfp4.rs`](../crates/gpu/src/moe_gguf/mxfp4.rs) |
| `moe_prefill_phase1_routes_int4` | [`moe_prefill_batch.metal`](../crates/gpu/src/shaders/moe_prefill_batch.metal) | 45 | Batched multi-token MoE Phase 1 gate+up for chunked prefill (PF-02) | [`moe_prefill_batch.rs`](../crates/gpu/src/moe_prefill_batch.rs) |
| `moe_prefill_phase2_fused_int4` | [`moe_prefill_batch.metal`](../crates/gpu/src/shaders/moe_prefill_batch.metal) | 91 | Batched multi-token MoE Phase 2 down-reduce for chunked prefill | [`moe_prefill_batch.rs`](../crates/gpu/src/moe_prefill_batch.rs) |

---

### 3.3 Attention & KV-Cache Management

Two-pass split-KV decode attention supporting Grouped Query Attention (GQA), Sliding Window Attention (SWA) ring buffers, and Attention Sinks.

| Kernel Name | Shader File | Line | Description & Optimizations | Host Dispatch |
|---|---|---:|---|---|
| `attention_decode_partial` | [`attention.metal`](../crates/gpu/src/shaders/attention.metal) | 142 | Split-KV stage 1 partial attention with online safe softmax (linear KV) | [`attention_decode.rs`](../crates/gpu/src/attention_decode.rs) |
| `attention_decode_gqa_swa_partial` | [`attention.metal`](../crates/gpu/src/shaders/attention.metal) | 234 | Split-KV stage 1 partial attention with SWA ring buffer wrap and attention sinks | [`attention_decode.rs`](../crates/gpu/src/attention_decode.rs) |
| `attention_decode_combine` | [`attention.metal`](../crates/gpu/src/shaders/attention.metal) | 350 | Split-KV stage 2 reduction combining partial attention sums across KV chunks | [`attention_decode.rs`](../crates/gpu/src/attention_decode.rs) |

---

### 3.4 Gated DeltaNet (GDN) Linear Attention

Complete linear attention pipeline for Qwen 3.6 hybrid layers, eliminating context-growing KV memory.

| Kernel Name | Shader File | Line | Description & Optimizations | Host Dispatch |
|---|---|---:|---|---|
| `gdn_in_proj_gemv_simd` | [`gdn.metal`](../crates/gpu/src/shaders/gdn.metal) | 86 | Fused 4-way projection (Q, K, V, and gate Z) with INT4 affine weights | [`gdn.rs`](../crates/gpu/src/gdn.rs) |
| `gdn_conv_mix_decode` | [`gdn.metal`](../crates/gpu/src/shaders/gdn.metal) | 153 | 1D causal depthwise convolution over recent token history during decode | [`gdn.rs`](../crates/gpu/src/gdn.rs) |
| `gdn_conv_mix_prefill` | [`gdn.metal`](../crates/gpu/src/shaders/gdn.metal) | 185 | 1D causal depthwise convolution across token sequence during prefill | [`gdn.rs`](../crates/gpu/src/gdn.rs) |
| `gdn_conv_tail_update` | [`gdn.metal`](../crates/gpu/src/shaders/gdn.metal) | 219 | Updates the trailing conv state ring buffer with newest input activations | [`gdn.rs`](../crates/gpu/src/gdn.rs) |
| `gdn_qk_norm` | [`gdn.metal`](../crates/gpu/src/shaders/gdn.metal) | 276 | Per-head RMS normalization for Q and K projection vectors | [`gdn.rs`](../crates/gpu/src/gdn.rs) |
| `gdn_delta_step_decode` | [`gdn.metal`](../crates/gpu/src/shaders/gdn.metal) | 328 | Single-step Gated DeltaNet recurrence: updates fixed-size FP32 recurrent state | [`gdn.rs`](../crates/gpu/src/gdn.rs) |
| `gdn_delta_step_prefill` | [`gdn.metal`](../crates/gpu/src/shaders/gdn.metal) | 389 | Sequential multi-token Gated DeltaNet state recurrence across prefill tokens | [`gdn.rs`](../crates/gpu/src/gdn.rs) |
| `gdn_gated_norm` | [`gdn.metal`](../crates/gpu/src/shaders/gdn.metal) | 468 | Gated output normalization with elementwise multiplication by gate `z` | [`gdn.rs`](../crates/gpu/src/gdn.rs) |

---

### 3.5 Normalization, RoPE & Elementwise Primitives

Core math building blocks supporting Gemma 4, Qwen 3.5/3.6/3.8, Mistral, Llama, and Muse Glimmer.

| Kernel Name | Shader File | Line | Description & Optimizations | Host Dispatch |
|---|---|---:|---|---|
| `rmsnorm_bf16w` | [`rmsnorm.metal`](../crates/gpu/src/shaders/rmsnorm.metal) | 69 | RMSNorm with BF16 learned scale weights | [`rms_norm.rs`](../crates/gpu/src/rms_norm.rs) |
| `rmsnorm_bf16w_centered` | [`rmsnorm.metal`](../crates/gpu/src/shaders/rmsnorm.metal) | 117 | Centered RMSNorm `x * (1 + w)` for Qwen 3.5 MTP heads | [`rms_norm.rs`](../crates/gpu/src/rms_norm.rs) |
| `rmsnorm_bf16w_perhead` | [`rmsnorm.metal`](../crates/gpu/src/shaders/rmsnorm.metal) | 148 | Per-head RMSNorm across all heads in one dispatch (1 threadgroup per head) | [`rms_norm.rs`](../crates/gpu/src/rms_norm.rs) |
| `rmsnorm_bf16w_perhead_centered` | [`rmsnorm.metal`](../crates/gpu/src/shaders/rmsnorm.metal) | 194 | Centered per-head RMSNorm `x * (1 + w)` for MTP heads | [`rms_norm.rs`](../crates/gpu/src/rms_norm.rs) |
| `rmsnorm_no_scale_perhead` | [`rmsnorm.metal`](../crates/gpu/src/shaders/rmsnorm.metal) | 221 | Unscaled per-head RMSNorm (used for Gemma 4 V normalization) | [`rms_norm.rs`](../crates/gpu/src/rms_norm.rs) |
| `rmsnorm_no_scale` | [`rmsnorm.metal`](../crates/gpu/src/shaders/rmsnorm.metal) | 247 | Unscaled single-vector RMSNorm | [`rms_norm.rs`](../crates/gpu/src/rms_norm.rs) |
| `rope_default_neox` | [`rope.metal`](../crates/gpu/src/shaders/rope.metal) | 52 | Standard NeoX rotary position embedding | [`rope.rs`](../crates/gpu/src/rope.rs) |
| `rope_neox_subdim` | [`rope.metal`](../crates/gpu/src/shaders/rope.metal) | 80 | Rotary position embedding with partial rotary dimension | [`rope.rs`](../crates/gpu/src/rope.rs) |
| `rope_proportional_neox` | [`rope.metal`](../crates/gpu/src/shaders/rope.metal) | 104 | Proportional NeoX RoPE with parameterized rotated pairs | [`rope.rs`](../crates/gpu/src/rope.rs) |
| `rope_neox_freqs` | [`rope.metal`](../crates/gpu/src/shaders/rope.metal) | 149 | RoPE with explicit precomputed frequency table | [`rope.rs`](../crates/gpu/src/rope.rs) |
| `gelu_mul_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 25 | In-place elementwise GeLU gated multiplication `x * gelu(gate)` | [`utility.rs`](../crates/gpu/src/utility.rs) |
| `silu_mul_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 40 | In-place elementwise SiLU gated multiplication `x * silu(gate)` | [`utility.rs`](../crates/gpu/src/utility.rs) |
| `sigmoid_gate_mul_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 55 | Elementwise Sigmoid gating multiplication `x * sigmoid(gate)` | [`utility.rs`](../crates/gpu/src/utility.rs) |
| `sigmoid_scalar_mul_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 68 | Scalar sigmoid activation | [`utility.rs`](../crates/gpu/src/utility.rs) |
| `split_q_gate_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 83 | Splits fused Q+Gate projection vector into separate Q and Gate tensors | [`utility.rs`](../crates/gpu/src/utility.rs) |
| `residual_add_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 102 | In-place residual stream addition `dst += src` (vectorized `half4`) | [`utility.rs`](../crates/gpu/src/utility.rs) |
| `scalar_mul_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 118 | In-place scalar multiplication across tensor elements | [`utility.rs`](../crates/gpu/src/utility.rs) |
| `logit_softcap_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 141 | In-place logit softcapping `cap * tanh(z / cap)` | [`utility.rs`](../crates/gpu/src/utility.rs) |
| `bias_add_bf16_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 165 | In-place BF16 bias vector addition onto FP16 activations | [`utility.rs`](../crates/gpu/src/utility.rs) |

---

### 3.6 Directional Weight Steering & Abliteration

Real-time steering vectors applied to the residual stream (`docs/OBLITERATION.md`).

| Kernel Name | Shader File | Line | Description & Optimizations | Host Dispatch |
|---|---|---:|---|---|
| `steer_direction_fp16` | [`utility.metal`](../crates/gpu/src/shaders/utility.metal) | 249 | Fused projection and steering edit: supports `ablate`, `add`, `clamp`, and `renorm` modes with dot product projection | [`utility.rs`](../crates/gpu/src/utility.rs) |

---

### 3.7 Speculative Drafters (DFlash2 & MTP)

Speculative block draft generation and state propagation.

| Kernel Name | Shader File | Line | Description & Optimizations | Host Dispatch |
|---|---|---:|---|---|
| `dflash_grouped_conv_fp16` | [`dflash_conv.metal`](../crates/gpu/src/shaders/dflash_conv.metal) | 44 | Grouped 1D causal convolution for DFlash2 block drafter state | [`dflash_conv.rs`](../crates/gpu/src/dflash_conv.rs) |
| `copy_strided_rows_fp16` | [`dflash_conv.metal`](../crates/gpu/src/shaders/dflash_conv.metal) | 89 | Strided row buffer copy for drafter history updates | [`dflash_conv.rs`](../crates/gpu/src/dflash_conv.rs) |

---

### 3.8 Output Head & Sampling

Logit evaluation and greedy chunked reduction.

| Kernel Name | Shader File | Line | Description & Optimizations | Host Dispatch |
|---|---|---:|---|---|
| `logit_softcap_softmax` | [`logit.metal`](../crates/gpu/src/shaders/logit.metal) | 55 | Online safe softmax with inline logit softcapping over full vocabulary | [`logit_softmax.rs`](../crates/gpu/src/logit_softmax.rs) |
| `sample` | [`logit.metal`](../crates/gpu/src/shaders/logit.metal) | 186 | GPU sampling kernel (vendored reference) | [`crates/selection`](../crates/selection) |
| `sample_topk64_stage1` | [`logit.metal`](../crates/gpu/src/shaders/logit.metal) | 492 | Stage 1 top-64 partial candidate extraction | [`crates/selection`](../crates/selection) |
| `sample_topk64_reduce` | [`logit.metal`](../crates/gpu/src/shaders/logit.metal) | 519 | Stage 2 top-64 reduction across threadgroups | [`crates/selection`](../crates/selection) |
| `sample_topk64_final` | [`logit.metal`](../crates/gpu/src/shaders/logit.metal) | 547 | Stage 3 final top-64 sorting and CDF sampling | [`crates/selection`](../crates/selection) |
| `lm_head_greedy_int4_rows_chunk_raw` | [`logit.metal`](../crates/gpu/src/shaders/logit.metal) | 690 | INT4 LM-head chunked evaluation for greedy argmax without materializing full logits | [`crates/selection`](../crates/selection) |
| `lm_head_greedy_int4_rows_reduce` | [`logit.metal`](../crates/gpu/src/shaders/logit.metal) | 743 | Final reduction of chunked argmax candidates into single token ID | [`crates/selection`](../crates/selection) |

---

### 3.9 Vision & Multimodal Processing

Pre-processing and vision transformer projection kernels for vision-language models (`crates/vision-io`).

| Kernel Name | Shader File | Line | Description & Optimizations | Host Dispatch |
|---|---|---:|---|---|
| `vision_layer_norm_fp16` | [`vision.metal`](../crates/gpu/src/shaders/vision.metal) | 45 | Layer normalization with mean and variance reduction for image patch embeddings | [`vision.rs`](../crates/gpu/src/vision.rs) |
| `vision_gelu_tanh_fp16` | [`vision.metal`](../crates/gpu/src/shaders/vision.metal) | 112 | Fast tanh-approximation GeLU for vision FFN blocks | [`vision.rs`](../crates/gpu/src/vision.rs) |
| `vision_gelu_erf_fp16` | [`vision.metal`](../crates/gpu/src/shaders/vision.metal) | 150 | Exact erf-based GeLU activation | [`vision.rs`](../crates/gpu/src/vision.rs) |
| `vision_rope_2d_fp16` | [`vision.metal`](../crates/gpu/src/shaders/vision.metal) | 174 | 2D Spatial Rotary Position Embedding for visual grid patch coordinates | [`vision.rs`](../crates/gpu/src/vision.rs) |
| `vision_attention_bidir_fp16` | [`vision.metal`](../crates/gpu/src/shaders/vision.metal) | 233 | Full bidirectional self-attention across image patch tokens | [`vision.rs`](../crates/gpu/src/vision.rs) |
| `vision_matmul_fp16` | [`vision.metal`](../crates/gpu/src/shaders/vision.metal) | 358 | FP16 matrix multiplication for visual projection heads | [`vision.rs`](../crates/gpu/src/vision.rs) |
| `vision_residual_add_fp16` | [`vision.metal`](../crates/gpu/src/shaders/vision.metal) | 414 | In-place residual connection addition for vision transformer blocks | [`vision.rs`](../crates/gpu/src/vision.rs) |

---

## 4. Key Kernel Optimizations

1. **SIMD-Group Reductions**: Reductions use Metal `simd_sum`, `simd_max`, and `simd_shuffle` intrinsics to collapse 32-lane partial sums in register space with zero threadgroup memory traffic.
2. **Function Constant Specialization**: Critical shape dimensions (`D`, `N`, `GroupSize`, `TopK`) are bound at pipeline creation time via Metal function constants, enabling LLVM backend unrolling and dead-code elimination.
3. **Zero-Copy Host/Device Pointers**: Resident weights are mapped via `newBufferWithBytesNoCopy` directly over the file mapping, and SSD streamer slots are page-aligned (`posix_memalign`) to allow zero-copy wrapping without CPU-to-GPU staging copies.
4. **Two-Pass Split-KV Attention**: Long context attention splits KV into chunked parallel threadgroups (`attention_decode_partial`), combining intermediate online-softmax accumulators in a lightweight stage 2 pass (`attention_decode_combine`).
5. **Centered Norm Formulations**: Implements both standard `x * w` and centered `x * (1 + w)` norms, critical for accurate multi-token prediction (MTP) drafting.
6. **Sub-Byte Unrolling**: 1-bit and 2-bit kernels unpack up to 8 weights per byte using bit-shift arithmetic and bitmask LUTs, eliminating memory traffic bottlenecks.
