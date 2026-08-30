# turbospark-gpu

Metal device context (`MetalContext`), pipeline cache, command pass encoders (`PassEncoder`, `CommittedPass`), per-kernel dispatches, KV cache manager (`KvCacheManager`), and resident weight Metal buffer wrappers (`ResidentGpuWeights`).

## Platform Requirements

- **macOS only**: All source files are gated with `#[cfg(target_os = "macos")]`. On non-macOS platforms, this crate compiles to an empty module.
- Requires a Metal-capable device and Xcode's `metal` toolchain (`xcrun -sdk macosx metal`) to compile and execute tests.

## Directory & File Structure

```
crates/gpu/
+-- Cargo.toml                      # Crate manifest
+-- src/
|   +-- lib.rs                      # Library root
|   +-- context/                    # MetalContext, PassEncoder, CommittedPass, errors, buffer IO
|   +-- kv_cache.rs                 # KvCacheManager managing per-layer Metal KV buffers
|   +-- kv_cache_mem.rs             # Memory allocation and POSIX advice helpers for KV cache
|   +-- attention_decode.rs         # Split-KV decode attention dispatch
|   +-- attention_decode_tests.rs   # Unit tests for attention decode dispatch
|   +-- moe_decode.rs               # MoE router, phase 1 GEMV, phase 2 down-reduce dispatches
|   +-- moe_prefill_batch.rs        # Batched MoE prefill GEMV dispatch (INT4 affine)
|   +-- moe_prefill_batch_gguf.rs   # The same over MXFP4 expert blobs (step 5; gpt-oss)
|   +-- rms_norm.rs                 # RMSNorm dispatches (no-scale, BF16, per-head)
|   +-- rope.rs                     # RoPE positional embedding dispatch
|   +-- dequant_1bit_gemv.rs        # MLX 1-bit affine GEMV pair (port-local, ROADMAP's 1-bit entry)
|   +-- dequant_2bit_gemv.rs        # MLX 2-bit affine GEMV + lookup (port-local, ROADMAP's ternary entry)
|   +-- dequant_int4_gemv.rs        # INT4 SIMD GEMV dispatches (resident & streamed)
|   +-- dequant_int4_batch.rs       # The M-ROW batched INT4 GEMM (ROADMAP Phase D2); see its header for the two register-file dead ends
|   +-- dequant_iq_gemv.rs          # IQ3_XXS / IQ4_NL / IQ4_XS codebook GEMV (Phase S)
|   +-- dequant_int8_gemv.rs        # INT8 SIMD GEMV dispatches (resident & streamed)
|   +-- dequant_q4_k_gemv.rs        # GGUF Q4_K SIMD GEMV + embed lookup (port-local, Phase G)
|   +-- dequant_q5_k_gemv.rs        # GGUF Q5_K SIMD GEMV (port-local, Phase M2)
|   +-- dequant_q6_k_gemv.rs        # GGUF Q6_K SIMD GEMV dispatch (port-local, Phase G)
|   +-- moe_gguf/                   # GGUF routed-expert decode pairs (Q8_0, K-quants, IQ, MXFP4)
|   +-- dequant_q8_0_gemv.rs        # GGUF Q8_0 SIMD GEMV dispatch (port-local, Phase G)
|   +-- resident_metal.rs           # ResidentGpuWeights mmap zero-copy MTLBuffer wrapper
|   +-- dispatch_profile.rs         # Intra-command-buffer dispatch profiler
|   +-- device_memory.rs            # MTLDevice recommendedMaxWorkingSetSize & name
|   +-- power_state.rs              # NSProcessInfo thermal state & Low Power Mode probes
|   +-- utility.rs                  # Elementwise helpers (scalar mul, softcap, M5's bias add, the steering edit)
|   +-- logit_softmax.rs            # Softcap and logit softmax helpers
|   +-- bytes.rs                    # Metal buffer byte alignment utilities
|   +-- gdn.rs                      # Gated-DeltaNet kernel dispatches (8 kernels)
|   +-- gdn_shape.rs                # GdnShape layout and validation
|   +-- gdn_state.rs                # GDN recurrent state buffers (Qwen flow's)
|   +-- dflash_conv.rs              # DFlash2 dynamic depthwise conv and state capture
|   +-- dsv4_state.rs               # DSV4 Metal buffer allocation (unwired)
|   +-- prefill_scratch.rs          # Chunked prefill scratch buffer layout (undispatched)
|   +-- vision.rs                   # qwen3_5 vision tower dispatches (port-local, ROADMAP M-V2)
|   \-- shaders/                    # MSL source, vendored from Swift except where marked port-local
|       +-- attention.metal         # Decode attention Metal shader source
|       +-- dequant_1bit.metal      # MLX 1-bit affine GEMV + the `+/-1` form
|       +-- dequant_2bit.metal      # MLX 2-bit affine GEMV + embedding lookup
|       +-- dequant_int4.metal      # INT4 dequantization GEMV shader source
|       +-- dequant_int4_batch.metal# Batched M-row INT4 GEMM shader source
|       +-- dequant_int4_mma.metal  # INT4 simdgroup_matrix MMA shader source (dead end)
|       +-- dequant_int8.metal      # INT8 dequantization GEMV shader source
|       +-- dequant_iq.metal        # GGUF IQ-codebook GEMV shader source
|       +-- dequant_q4_k.metal      # GGUF Q4_K dequantization GEMV + embedding lookup
|       +-- dequant_q5_k.metal      # GGUF Q5_K dequantization GEMV (after dequant_q4_k.metal)
|       +-- dequant_q6_k.metal      # GGUF Q6_K dequantization GEMV shader source
|       +-- moe_gguf.metal          # GGUF MoE decode pairs (after moe.metal + dequant_q4_k.metal)
|       +-- dequant_q8_0.metal      # GGUF Q8_0 dequantization GEMV shader source
|       +-- dflash_conv.metal       # DFlash2 grouped depthwise conv shader source
|       +-- gdn.metal               # Gated-DeltaNet (Qwen 3.6 linear attention) shader source
|       +-- logit.metal             # Logit softcap and softmax shader source
|       +-- moe.metal               # MoE router GEMV and phase 1/2 shader source
|       +-- moe_prefill_batch.metal # Batched MoE prefill GEMV shader source
|       +-- moe_prefill_batch_gguf.metal # The MXFP4 batched pair (after the moe_gguf chain)
|       +-- rmsnorm.metal           # RMSNorm shader source
|       +-- rope.metal              # RoPE shader source
|       +-- utility.metal           # Elementwise utility shader source + the steering edit
|       \-- vision.metal            # qwen3_5 vision tower shader source (port-local)
\-- tests/                          # Metal numerical parity & allocation unit tests
    +-- attention_chunk_bench.rs
    +-- attention_decode_parity.rs
    +-- attention_sinks.rs
    +-- attention_swa.rs
    +-- dequant_1bit_gemv_parity.rs
    +-- dequant_2bit_gemv_parity.rs
    +-- dequant_int4_gemm_parity.rs
    +-- dequant_int4_gemv_parity.rs
    +-- dequant_int4_mma_parity.rs
    +-- dequant_int8_gemv_parity.rs
    +-- dequant_iq_gemv_parity.rs
    +-- dequant_q4_k_gemv_parity.rs
    +-- dequant_q5_k_gemv_parity.rs
    +-- dequant_q6_k_gemv_parity.rs
    +-- dequant_q8_0_gemv_parity.rs
    +-- dflash_conv_parity.rs
    +-- dispatch_profile.rs
    +-- dsv4_state.rs
    +-- gdn_parity.rs
    +-- gdn_state.rs
    +-- gemma4_real_kernels_parity.rs
    +-- gemv_bandwidth_bench.rs
    +-- kv_cache.rs
    +-- logit_softmax_parity.rs
    +-- moe_decode.rs
    +-- moe_gguf_parity.rs
    +-- moe_prefill_batch_bench.rs
    +-- moe_prefill_batch_gguf_parity.rs
    +-- moe_prefill_batch_parity.rs
    +-- power_state.rs
    +-- prefill_scratch.rs
    +-- resident_metal.rs
    +-- rms_norm_parity.rs
    +-- rope_parity.rs
    +-- rope_yarn_parity.rs
    +-- scaled_norm_and_embed.rs
    +-- utility_and_pass.rs
    +-- vision_block_parity.rs
    \-- vision_parity.rs
```

## Key Modules

- `context/`: `MetalContext` managing `MTLDevice`, `MTLCommandQueue`, pipeline state caching, pass encoders (`PassEncoder`, `CommittedPass`), errors (`GpuError`), and buffer IO.
- `kv_cache.rs`: `KvCacheManager` managing persistent per-layer Metal KV buffers (linear or sliding-window ring).
- `attention_decode.rs`: Split-KV decode attention dispatch (`chunks_for`, up to 16 splits), optionally with ROADMAP M5's ATTENTION SINKS: one learned logit per q head joining the softmax DENOMINATOR and nothing else, folded into `attention_decode_combine`'s existing cross-chunk max and denominator. It rides a FUNCTION CONSTANT (`FC_ATTN_HAS_SINKS`) rather than a uniform, unlike the MXFP4 activation next door, because an unbound buffer is undefined behaviour and the argument therefore has to not exist for the families without one -- which means its byte belongs in `attention_constants_key` beside scale, ring capacity and chunk count (Gotcha 7's rule, fourth axis).
- `moe_decode.rs`: Dispatches MoE router GEMV, phase-1 GEMVs, host router wait, and phase-2 down reduction.
- `rms_norm.rs`: RMSNorm dispatches (`rmsnorm_no_scale`, `rms_norm_bf16w`, `rmsnorm_bf16w_centered`, per-head norm variants). The centered one is PORT-LOCAL (Swift reads no architecture with that convention) and its contract is `turbospark_compute::rms_norm_centered`: `out = x * inv * (1 + w)`, the form whose checkpoint weight is an OFFSET FROM UNITY. **It is a separate KERNEL and not a function constant on its plain sibling**, because `muse_glimmer` uses both conventions in ONE forward pass -- four centered per-layer norms and a plain final one -- so a specialization axis whose byte missed `constants_key` would silently make them the same function (Gotcha 1's trap, and Gotcha 10). The `1 +` is applied on the FP32 accumulator and never baked into the stored weight: these weights sit near ZERO and BF16's resolution near 1.0 is 2^-8, so a bake would destroy ~39% of a 0.01 weight's magnitude. `tests/scaled_norm_and_embed.rs` holds the parity case and, beside it, `the_two_norm_conventions_are_different_functions`, which asserts the fixture DISCRIMINATES before the parity case is believed -- at weights near zero the two forms converge and a tidy fixture would pass against either kernel. **`rmsnorm_bf16w_perhead_centered` is the PER-HEAD sibling of that kernel** (the `qwen3_5` MTP head's `q_norm`/`k_norm`), and it is the sharpest case for the per-tensor rule in the repo: the TRUNK of the same model carries tensors of those exact names, at that exact shape, resolved through the same `encode_full_attention_block`, and they are plain. `crates/runtime`'s `QkNormConvention` is what lets the two call sites disagree. Its parity case has the same discriminating guard, with the fixture at 0.78 rather than near zero because that is where the real weights sit; the whole-vector fixture's near-zero range would not have separated the two.
- `rope.rs`: Rotary positional embedding dispatches (`rope_proportional_neox`, `rope_neox_subdim`, and ROADMAP M5's `rope_neox_freqs`). The first three derive each pair's frequency from a SCALAR theta; `rope_neox_freqs` reads a precomputed per-pair table plus a magnitude scale, because YaRN interpolates per dimension along a ramp and no single number expresses that. The table is position-independent, so it is built once at open by `turbospark_compute::yarn_frequencies`. Incidentally the shape a LEARNED frequency table needs, which is what `rope_freqs.weight` is -- refused by name since M4, and nothing here wires it up.
- `gdn.rs`: The eight gated-DeltaNet dispatches plus `GdnShape` and its structural preconditions.
- `dflash_conv.rs`: DFlash2's grouped dynamic depthwise conv (`dflash_grouped_conv_fp16`, one kernel for all four call sites via `side` and `delta_col_base`) plus `copy_strided_rows_fp16`, which is the whole of the drafter's aux-state capture. Port-local; no reference backend kernel exists, because llama.cpp composes both from existing ggml ops. **Its `out_scale` is not cosmetic**: the `finish` call sites divide by `DFLASH_RESIDUAL_SCALE` so the drafter's residual stream, which peaks at 113,920, fits FP16's 65,504 ceiling (AGENTS.md Gotcha 60). A power of two shifts the exponent and leaves the mantissa alone, so `tests/dflash_conv_parity.rs` asserts the scaling EXACTLY rather than within a tolerance, and asserts that enough elements are large enough for that claim to mean something.
- `dequant_int4_gemv.rs` & `dequant_int8_gemv.rs`: INT4/INT8 GEMV SIMD dispatches. Since 2026-08-16 the INT4 GEMV bakes M/N as function constants per shape (`specialized_constants`), as does the fused GDN in_proj (`gdn.rs::in_proj_pipeline`, constants 90-94): +3-5% cold weight-read rate isolated, +4.5-8.4% end to end on the dense 27B, output byte-identical on both INT4 families (measured, not assumed -- fast-math unrolling could legally have reassociated the accumulator and did not). The pipeline-cache KEY carries the baked values; a shared key is Gotcha 1's silent-wrong-pipeline trap, and the bench that measured this (`gemv_bandwidth_bench.rs::baking_m_n_...`) fell into it on its own first run (+322% from a wrong-N pipeline reading a third of each row). `embed_lookup_int4` stays unspecialized: a lookup has no trip count to bake.
- `dequant_1bit_gemv.rs`: the MLX `affine` GEMV pair at ONE bit (`prism-ml/Bonsai-27B-mlx-1bit`, ROADMAP's 1-bit entry step 2). PORT-LOCAL for the GGUF set's reason -- the Swift engine reads no 1-bit checkpoint -- and its contract is `turbospark_compute::quant_1bit`. Same CONTAINER as the INT4 sibling next door and NOT a narrower version of it: the companions bind as `device const half*` rather than `bfloat*` (same width, so a misread passes every length check and reads the checkpoint's 0.0271 scales as 1.7e-16), the group size is the checkpoint's 128 rather than the sibling's compile-time 64, and a byte holds 8 elements rather than 2, which makes the LSB-first bit order inside it load-bearing. **The group size is a runtime UNIFORM, deliberately not a function constant**: it is a property of the checkpoint rather than of the container, and a specialization axis that missed `pipeline`'s constants key would silently reuse the wrong pipeline (Gotcha 1). **`dequant_int1_gemv_symmetric_simd` is a SECOND KERNEL, not a fast path inside the first**, for two reasons that both matter: factoring the group scale out of the sum reassociates it (AGENTS.md Gotcha 27), and it binds no bias plane at all, so `Int1SymmetricRowGpu` has no field through which an unchecked bias could reach it -- the `bias == -scale/2` check is the caller's, and it is measured (`turbospark_compute::is_symmetric`) rather than assumed. The symmetric form has no resident variant on purpose, and step 4 turned that from "not yet" into a decision: the repack step writes a bias plane for every 1-bit tensor whether or not it is symmetric, so the fast path would have to be SELECTED per tensor at dispatch time, which would make the generated bytes a function of which tensors happened to quantize symmetrically. Reaching it needs its own dtype tag written at repack time, not a check at the dispatch. **`embed_lookup_int1` is the third kernel and it was not in the plan**: the checkpoint's own safetensors header says `embed_tokens` and `lm_head` are BOTH quantized at one bit (`U32 [248320, 160]`, F16 companions), and while the head is just another matrix through the GEMV, the table needs a lookup. So this type's footing is a GEMV plus a lookup and NO routed-expert pair -- Gotcha 29's per-type rule as the one real file decides it, the model being dense. Since step 4 the `qwen3_5` decode flow dispatches the general GEMV's resident form and the lookup, deriving the group size from each tensor's own companion planes (`crates/runtime` Gotcha 12); the symmetric one is reachable from no flow. `tests/dequant_1bit_gemv_parity.rs` is still where the NUMBERS are held, and its two trap cases (bit order, companion dtype) construct data that discriminates and then ASSERT that it does, because at one bit a wrong reading leaves every magnitude, every group scale and the total popcount untouched.
- `dequant_2bit_gemv.rs`: the MLX `affine` GEMV plus embedding lookup at TWO bits (`prism-ml/Ternary-Bonsai-27B-mlx-2bit`, ROADMAP's ternary entry step 2). PORT-LOCAL for the 1-bit sibling's reason, and its contract is `turbospark_compute::quant_2bit`. It inherits that sibling's three departures from the INT4 GEMV -- `half` companions rather than `bfloat` (same width, so a misread reads this checkpoint's 0.0137 scales as 7e-18), the checkpoint's group size as a runtime UNIFORM rather than a compile-time 64, and a byte holding several elements so the LSB-first field order is load-bearing -- and differs from it in one: **TWO kernels, not three.** The checkpoint is TERNARY (`bias == -scale` in every probed group, level 3 never used), so the analogue of the 1-bit `+/-1` fast path exists mathematically and is deliberately not built: it would reassociate the sum the same way (AGENTS.md Gotcha 27), and the 1-bit one is already reachable from no decode flow, so a second dead kernel buys nothing. **The GEMV is the GENERAL affine form and decodes level 3 correctly**, which `the_fourth_level_is_decoded_not_clamped` pins -- a kernel written from the ternary description could mask the top bit and pass every other case. `tests/dequant_2bit_gemv_parity.rs` holds the numbers, with the two trap cases (field order, companion dtype) constructing discriminating data and then ASSERTING that it discriminates, because at two bits a wrong field order permutes within a run of four and leaves every magnitude, every group scale and the whole level histogram untouched.
- `dequant_int4_batch.rs`: `dequant_int4_gemm_simd`, the M-row batched form of the above (ROADMAP Phase D2's verify kernel). Not on any decode path -- decode is M=1 -- and kept because a tile-kernel phase would otherwise rebuild it. **Read its header before optimizing it**: threadgroup staging of `x` and register blocking over rows are both measured LOSSES on every shape. Since 2026-08-17 it bakes M, N and **B** as function constants (100-103) and bounds its inner unroll at 4, which together roughly HALVED `c(M)` -- 0.85 to 0.46 at M=8 on the dense 27B shapes (`docs/MTP_SPECULATIVE.md`). B is the one that matters and the unroll bound is not optional: baking B lets the compiler unroll fully, and a full unroll at B=16 holds ~128 activation floats live beside `acc[16]` and SPILLS, reading 1.14 where the bounded form reads 0.44. The factor was swept (2/4/8/full), not chosen. Parity against the GEMV stays EXACT; `tests/dequant_int4_gemm_parity.rs` pins that. **IT DOES NOT FOLLOW THAT A BATCHED VERIFY IS BIT-IDENTICAL TO A SEQUENTIAL DECODE, and this line used to say it did.** Measured 2026-08-20 on the real `qwen3_5` DFlash2 install: a greedy speculative stream tracks a greedy sequential one for ~154 tokens and then diverges. Block sizes 2, 4, 7 and 8 all produce IDENTICAL text to each other and part from sequential at the same token, which says the batch WIDTH is not the variable. **THIS KERNEL WAS BLAMED FOR IT AND HAS BEEN EXONERATED, measured 2026-08-21.** The attribution was an inference from reading two files, and it is wrong twice over. Reading cannot settle it at all -- Metal's fast-math reassociates freely, so rewriting a sum in the opposite order compiles to identical code (verified: that mutation survives every case in the parity file) and SOURCE order does not determine COMPILED order in either kernel. And the measurement goes the other way: `the_gemm_and_the_gemv_agree_on_data_that_can_see_reassociation` runs both on ragged BF16 companions and full-mantissa FP16 activations spanning ~1e-3 to ~1e2 with alternating signs, and they agree BIT-FOR-BIT on every output at B=1, 2, 8 and 16. Its positive control is what makes that green mean something: `dequant_int4_gemm_mma`, which reduces in an order Apple does not document, differs from the GEMV on ~39% of outputs on the SAME fixture. So a fixture that can see a reassociation sees none here. The `gdn.metal` multi-row kernels were the next suspect and are also exact (0 of 896 outputs differ bitwise in `gdn_parity.rs`'s prefill-vs-decode case, so its 5e-2 state tolerance is conservative rather than descriptive). **AND THE DIVERGENCE IS THIS PORT'S SHAPE FLOOR RATHER THAN A DEFECT**, measured the same day: `produce_batched` and `produce` differ by 6.2e-8 to 1.5e-5 NATS with the argmax agreeing on every row, against a dense batched-vs-cached shape floor of 7.4e-6 measured on MLX for this architecture and 1.57e-5 for this repo's own cross-engine result on the family, published as no detectable gap. The differing-LOGIT COUNT is the wrong unit and reading it as the magnitude is what made this look worth bisecting -- 88% of a vocabulary moving at 1e-5 nats is a tiny distributional difference, and the argmax never moved. Its cause is bounded to the attention reduction: present at M=1 (so not the batch width), absent at one and two keys and present from three (and a two-key softmax exposes the score in full, so q, k and V are exact there), and not the split-KV combine, which does not run until span 32. `crates/bench/tests/batched_forward_probe.rs` is the instrument. There is nothing to make bit-identical here that every engine does not also have. The same test also pins, in `a_second_shape_in_one_process_does_not_reuse_the_first_shapes_pipeline`, that the cache key carries all three baked values (mutation-checked on each). Beside it, `dequant_int4_mma.metal`'s `dequant_int4_gemm_mma` is the `simdgroup_matrix` form and is a MEASURED DEAD END kept so it is not re-proposed: it loses at every batch width (6.6x at M=2, 1.33x at M=16) and PLATEAUS at ~0.5 past M=16, worse than the exact kernel, because a packed INT4 run cannot be `simdgroup_load`ed and the dequant-into-threadgroup work it therefore needs is independent of B while the MACs matrix hardware accelerates scale with B. Nothing dispatches it. `tests/dequant_int4_mma_parity.rs` is the repo's only NON-exact batched-kernel test and says why in its header; its fixture deliberately uses non-power-of-two companions, and asserts that it discriminates, because the exact test's fixture sums exactly in FP32 and cannot see reassociation at all (the first run read 4096/4096 bit-identical and bounded nothing).

    **EVERY MEASUREMENT OF `dequant_int4_gemm_simd` WAS TAKEN AT M<=16,
    WHICH IS THE VERIFY REGIME AND NOT THE PREFILL ONE.** That is true BY
    CONSTRUCTION rather than by oversight -- the host asserts
    `1..=MAX_BATCH_ROWS` -- so this kernel's "losses on every shape" means
    every shape it can currently be dispatched at, and prefill wants a wider
    one. Measured 2026-08-29 on the real `qwen38-27b` install
    (`docs/BENCHMARKS.md`, the Qwen3.8-27B external reference points): its
    per-row cost PLATEAUS at `c ~ 0.5` and stays there, so 16 rows cost 8.2
    single-row passes and the batched prefill arm wins only 1.94x where the
    width promises 16x. Against mlx-lm on the SAME machine and checkpoint it
    is 8.2x off its own roofline where mlx-lm is 3.8x off its own, and that
    plus the width accounts for the whole 4.94x prefill gap in two nearly
    equal factors of 2.2x and 2.3x.

    **THE SCOPING ABOVE DOES NOT EXTEND TO THE `simdgroup_matrix` SIBLING,
    AND A COMMIT MESSAGE SAID IT DID.** `9b0aad6` wrote "every measurement in
    this bullet was taken at M<=16 ... nobody tried the shape prefill wants"
    and sent the next reader off to re-measure `dequant_int4_gemm_mma` at
    B=64 before believing its dead-end label. That kernel was ALREADY measured
    there: `MMA_MAX_BATCH_ROWS` is 64, `gemv_bandwidth_bench.rs`'s
    `c_of_m_matrix_against_exact_at_qwen38_shapes` sweeps
    `[2, 4, 8, 16, 32, 64]`, and `dequant_int4_mma.metal`'s header records
    **0.52 at M=32 and 0.58 at M=64** -- a plateau ABOVE the SIMD kernel's
    0.44 at M=16, with "running past the SIMD kernel's register cap to M=32
    and 64 changed nothing" stated in the same table. The prediction that
    B=64 would amortize what B=8 could not is therefore already refuted, by
    the mechanism this bullet names two sentences earlier: the dequant into
    threadgroup memory is proportional to `rows x K` and independent of B, so
    a bigger B grows only the term matrix hardware was already accelerating.
    AGENTS.md Gotcha 62, arriving in the note rather than in the artifact --
    the numbers were one file away from the sentence that denied them.

    **AND THE M=64 WIDTH TARGET IS DOWNSTREAM OF THE KERNEL RATHER THAN
    INDEPENDENT OF IT.** `docs/BENCHMARKS.md` derives it from mlx-lm's 1.32 ms
    compute floor, which is ~41 TFLOP/s on this model and needs FP16 matrix
    hardware; this kernel is scalar FP32 `fma`, so its own compute floor is
    several times higher and is crossed at a much smaller M. The shipped
    `count(4)` row in `dequant_int4_batch.metal` reads
    `M=2 0.50, M=4 0.55, M=8 0.46, M=16 0.44`, which is FLAT -- time scaling
    linearly in M is what a COMPUTE-bound kernel looks like, and it says
    widening M amortizes weight bytes that were not the cost. So raise the
    kernel's arithmetic intensity first and let the measured `c(R, B)` decide
    the width; a cap raise taken on its own buys footprint and no throughput.
    Unconfirmed on AC as of 2026-08-29 (battery, load 4.6, swap 3.0/4.1 GB),
    which is why it is written as a reading of an existing table and not as a
    measurement.

    **`FC_GEMM_R` (constant 104) IS THAT ARITHMETIC INTENSITY, AND IT IS
    MEASURED: 1.30x AT THE WIDTH THE PREFILL DRIVER ALREADY USES.** One SIMD
    group owns `row_block` CONTIGUOUS output rows, which divides the
    per-block activation loads by R and hoists `sum = e0 + ... + e7` out of
    the row loop -- it depends on `bi` alone, and the one-row kernel
    recomputes it once per row. The irreducible term is the 8 `fma` per
    `(r, bi)`. It moves no bits at any width, asserted on the hostile fixture
    with the MMA positive control
    (`row_blocking_does_not_move_a_single_bit`), so its gate is the parity
    test and it owes no quality gate.

    `c(R, M)` on AC, 2026-08-29, mean of four runs, gate/up 17408x5120 (the
    other five shapes agree to 0.03 and the spread within a cell is 0.00 to
    0.05):

    |  M  |  R=1  |  R=2  |  R=4  | best gain |
    |---|---|---|---|---|
    |  2  | 0.510 | 0.505 | 0.617 |  1.01x    |
    |  4  | 0.607 | **0.307** | 0.365 |  1.98x    |
    |  8  | 0.502 | 0.448 | 0.427 |  1.18x    |
    | 16  | 0.487 | 0.425 | **0.375** |  1.30x    |

    **R=4 IS THE ANSWER AT M=16 AND R=2 AT M=4; R=4 AT M=2 IS A LOSS**
    (0.51 to 0.62) and at M=1 a bigger one (1.00 to 1.22). So the width is
    chosen per batch by `best_row_block` -- a TABLE, not the global constant
    the M=16 row alone would suggest, because the widths R=4 regresses are
    exactly the ones the MTP and DFlash2 verify run at through the same
    entry point. That selection is legitimate here and forbidden for the
    matrix kernel next door, for a reason spelled out on the function.

    Wired 2026-08-29 and measured end to end on the real install: the
    frozen `long-synthesis` prompt prefills in 69.75 s against 87.80 s,
    three interleaved pairs, **1.26x** against the 1.24x the GEMM's 85.4%
    share predicted. Byte-identical output, verified THROUGH the batched arm
    -- plain decode is M=1 GEMV and never enters this kernel, so a smoke run
    without `MFERENCE_BATCHED_GEMV=1` cannot see a change to it.

    **THE M=4 CELL IS THE BEST NUMBER IN THE TABLE AND IS NOT A LICENCE TO
    NARROW THE MICRO-BATCH.** 0.307 against M=16/R=4's 0.375 is a further
    1.23x on the GEMM alone, and `c` is comparable across M here (verified:
    re-running with `groups` FIXED at 64, so every M moves identical weight
    bytes, reproduces the table cell for cell). But prefill's per-micro-batch
    costs -- command-buffer commits, host encode -- scale with the NUMBER of
    micro-batches, and going 16 to 4 quadruples them. This bench prices the
    GEMM and cannot see that, so the M=4 route needs an end-to-end run before
    anyone believes it. Note also WHY M=4 is special: `unroll_count(4)` makes
    `b_dim == 4` exactly one full unroll, and the R=1 column's own bump at
    M=4 (0.607 against 0.50 either side) is that interaction going badly,
    which R=2 removes.

    **THE FROZEN `count(4)` ROW IS NOT COMPARABLE ACROSS SESSIONS, measured
    the same day.** That table reads 0.50 / 0.55 / 0.46 / 0.44 and the
    IDENTICAL code read 0.51 / 0.61 / 0.50 / 0.49 here -- up to 11% apart on
    the same machine and the same shapes. It is not a regression from
    `FC_GEMM_R`, and the check that says so is an interleaved A/B against
    `aa094b4^`: three pairs, baseline and R=1 agreeing to 0.01 on every cell.
    So the 2-D `acc[kMaxRowBlock][kMaxBatchRows]` declaration DOES collapse
    at R=1 (the thing no static instrument here could confirm), and Gotcha
    22's rule holds -- read a `c(M)` number against arms measured beside it,
    never against a frozen row from another day.

    **PIPELINE REFLECTION CANNOT PRICE IT. MEASURED NEGATIVE, 2026-08-29.**
    The register file is this kernel's binding constraint and every statement
    about it in this bullet was read off a TIMING, so
    `maxTotalThreadsPerThreadgroup` looked like a way to answer "does this
    width spill" statically -- a Metal device, no model, no clock. It reads
    **1024 on every `(R, B)` shape**, and the discrimination check is what
    settles it rather than the table: raising `kMaxRowBlock` to 64 and
    probing R=64 at B=16 declares `acc[64][16]`, a thousand floats per lane
    that fit no register file on any GPU, and it still reads 1024. An
    instrument at its ceiling on an impossible configuration is at its rail
    (Gotcha 59's shape), and Metal exposes no register or occupancy count
    publicly, so there is no second instrument. What survives is narrower and
    is asserted: `static_threadgroup_memory_length` must stay 0, which is the
    one cheap guard against note 1's threadgroup staging being re-added
    unmeasured. `c(R, B)` on AC remains the only instrument that can price a
    row block, R=1 included.

    **AND THE DISPATCH ARITHMETIC NEEDED ITS OWN TEST, WHICH IS THE PART
    WORTH CARRYING.** Dropping `* row_block` from the threadgroup count
    survived every parity case, because it OVER-dispatches: the surplus
    threadgroups compute a `row0` past `M` and return at the kernel's first
    branch, so the output stays bit-correct and only the COST moves, by a
    factor of R. That is the worst shape a bug can have on an axis whose
    whole purpose is to be timed -- the variant would read as "row blocking
    does not help" and the axis would close on a measurement of the mistake.
    `gemm_threadgroups` is a named function asserted as arithmetic instead,
    on the precedent `steering.rs` set for its row partition, and the
    assertion that catches it is the no-idle-threadgroup half rather than the
    coverage half.
- `dequant_iq_gemv.rs`: the IQ3_XXS / IQ4_NL / IQ4_XS codebook GEMV dispatches (ROADMAP Phase S), whose tables are GENERATED from libggml by `scripts/ggml_tables.c` rather than transcribed, so the CPU and MSL copies cannot drift. `tests/dequant_iq_gemv_parity.rs`.
- `dequant_q8_0_gemv.rs`: the GGUF Q8_0 GEMV dispatch (ROADMAP Phase G Stage 2). PORT-LOCAL, not vendored -- the Swift engine has no GGUF intake, so its only contract is `turbospark_compute::dequant_q8_0_gemv`. Shorter than the INT8 sibling because a Q8_0 row is ONE byte run: the scale lives inside each 34-byte block, so there are no scale or bias planes to bind and no group size to agree on. 32 lanes over a 32-element block, one weight per lane.
- `dequant_q4_k_gemv.rs`: the GGUF Q4_K GEMV dispatch, port-local for the same reason. Same 32-lane shape for a DIFFERENT reason: a lane owns one of the 32 nibble BYTES in a group, hence two elements 32 apart that belong to two different sub-blocks. Q4_K is two-level (f16 `d`/`dmin` per 256-element superblock, 6-bit scale and 6-bit min per 32-element sub-block, packed 12 bytes and split across bytes for sub-blocks 4..8) and asymmetric (`w = d*sc*q - dmin*m`, unsigned quants), so none of the Q8_0 habits carry over.
- `dequant_q5_k_gemv.rs`: the GGUF Q5_K GEMV dispatch, port-local (ROADMAP Phase M2). Exists for one tensor shape: Mixtral 8x7B's Q4_K_M carries `attn_output` in Q5_K while its experts are Q4_K, so there is no embedding-lookup and no MoE sibling. Q5_K is Q4_K plus a fifth bit in its own 32-byte `qh` run, indexed by the element's position WITHIN a sub-block at a bit that advances with the 64-element group, so one `qh` byte is read eight times. Its shader source is a CONCATENATION with `dequant_q4_k.metal` (Gotcha 4): the 6-bit scale/min packing is Q4_K's byte for byte and it calls `q4_k_scale_min` rather than carrying a second copy.
- `dequant_q6_k_gemv.rs`: the GGUF Q6_K GEMV dispatch, port-local. Exists for ONE tensor: Qwen 3.6's Q4_K_M carries a single Q6_K weight, `output.weight`, and nothing else in either real file uses the type, so there is deliberately no embedding-lookup or MoE sibling. Same 32-lane shape for a third reason: a lane owns one `qh` byte and hence the four elements it supplies high bits for, 32 apart inside a 128-element half. Q6_K splits an element's six bits across `ql` and `qh`, biases the quant by a fixed 32 rather than carrying a per-sub-block min, and stores sixteen SIGNED int8 sub-block scales as plain bytes.
- `moe_gguf/`: the routed-expert decode pairs for GGUF expert blobs, port-local: `moe_phase1_gate_up_act_{q8_0,q4_k,mxfp4}` + `moe_phase2_down_reduce_k8_{q8_0,q4_k,mxfp4}`, plus phase-1-only IQ3_XXS/IQ4_XS, phase-2-only IQ4_NL, and (ROADMAP Phase M2) a phase-2-only Q6_K for the `ffn_down_exps` of 16 of Mixtral's 32 layers -- the first mixture whose two types differ across LAYERS rather than across the phases of one expert. The Q4_K pair does not re-implement the unpack; it calls `dequant_q4_k_row_simd` out of `dequant_q4_k.metal`, which joins the concatenation ahead of `moe_gguf.metal`. Both pairs share one host-side dispatch body, so only the kernel name and the block-size precondition differ (32 elements per row for Q8_0, 256 for Q4_K). Separate kernels rather than a function-constant variant of the vendored pair because the two blob layouts share no addressing: a GGUF blob has no scale or bias planes at all, so six of the nine `MoeExpertOffsets` fields are zero and only the three weight offsets locate anything. Shares the vendored pair's `RoutedBlobs` argument buffer, which is one array of eight pointers whatever the blobs contain. **MXFP4 (ROADMAP M5) is the one pair that does not fit that description, in two ways.** It has BOTH phases and no resident GEMV, the mirror of Q5_K's and Q6_K's footing, because `gpt-oss` puts MXFP4 in `ffn_*_exps` and nowhere else -- so its row unpack is written in `moe_gguf.metal` rather than borrowed, there being no `dequant_mxfp4.metal` to borrow from. And it carries the FAMILY's expert math as well as the block layout: the clamped SwiGLU (`min(gate, limit)`, a swish with `alpha`, times `(clamp(up) + 1)`, all from ggml's `ggml_compute_forward_swiglu_oai_f32`) and the per-expert biases the blob holds at the three `MoeExpertOffsets` bias fields every earlier GGUF install left at zero. Both ride as UNIFORMS via `Mxfp4Activation`, never function constants: `constants_key` is one byte wide and a specialization axis that misses it silently reuses the wrong pipeline. `Mxfp4Activation::PLAIN` keeps the block-type parity cases and the family cases in one kernel so neither hides the other. A 17-byte block also makes MXFP4 rows odd-length, which is why every read in it is `uint8_t`: an `as_type<half>` copied in from the Q8_0 helper next door would be misaligned on half the rows.
- `utility.rs`'s `encode_steer_direction` (+ `SteerParams`): the DIRECTIONAL
  STEERING edit on a residual stream row (ROADMAP item 9, in its RUNTIME form
  rather than the repack-time one that entry scopes). Port-local; its contract
  is `turbospark_compute::steering`. Four modes off ONE reduction pass --
  `ablate` (`x -= alpha * c_hat * d_hat`, which at `alpha = 1` is exactly the
  operation abliteration's weight edit precomputes), `add` (llama.cpp control
  vectors), `clamp` (feature clamping), and `renorm` (`ablate`, then rescale
  the row back to its original `||x||`) -- plus a per-row coefficient
  readback that IS the measurement: `c_hat` says how much of the direction the
  stream carried, which is the steered-vs-unsteered signal without a second
  model to compare against. Four things about it are decisions rather than
  detail. **The mode is a UNIFORM and never a function constant**: a
  specialization axis whose byte misses `pipeline`'s `constants_key` silently
  reuses whichever pipeline compiled first (Gotcha 1), and here that would
  make two different edits one function on the second dispatch. **One dispatch
  and not two**, because the cost of this kernel is encode rather than
  bandwidth -- it adds 64-128 dispatches per token on a 64-layer model against
  a decode step's ~810, while reading one row against a projection's whole
  weight matrix. **The block reduce is `rmsnorm.metal`'s `rms_block_inv`
  shape**, which reads `simdgroups` at runtime instead of hardcoding a loop
  bound, and that is what keeps it correct where `gdn.metal`'s two norms are
  correct at EXACTLY 128 threads and silently wrong otherwise (Gotcha 5);
  `partial` still sizes for 256 threads, so the dispatch must not widen the
  threadgroup. **The gate is evaluated INSIDE the kernel**, never by a host
  reading the coefficient back, which would cost a synchronization per layer
  per token. `row_stride` is in ELEMENTS rather than bytes on purpose: every
  residual call site computes its row offset in bytes, so a byte-taking stride
  would read correctly at `rows == 1` and edit the wrong rows at `rows > 1`.
  Note `ablate` can only REDUCE `|x|` and so cannot overflow, while `add` and
  `clamp` can push an FP16 stream past 65,504 at a large enough alpha --
  arriving as `inf` and then NaN, which reads as a perfect score on any rank
  instrument (AGENTS.md Gotchas 59 and 60). **`renorm` cannot overflow
  either**, for a different reason: it restores a magnitude the row already
  carried, so no element can exceed a value that was already representable.
  **It adds a SECOND reduction and neither of the costs that suggests.**
  `||x||^2` accumulates in the loop that already computes `c` -- the row
  element is in register for the dot product, so it is one more fma and no
  memory traffic -- and the POST-edit norm is ANALYTIC rather than a second
  pass over the written row, because the projection is orthogonal:
  `||x'||^2 = ||x||^2 - alpha*(2 - alpha)*c_hat^2`, exactly, which at
  `alpha = 1` is Pythagoras. So the kernel stays at one reduction and one
  write, which is what matters on a dispatch-bound path. Its rescale factor is
  exactly `1.0f` in the other three modes and `1.0f * v == v` in IEEE-754, so
  ONE write loop serves all four and the older three stay bit-identical --
  measured, not asserted: `alpha_zero_leaves_the_row_bit_identical` and
  `full_ablation_and_a_zero_clamp_are_the_same_edit` both compare `to_bits`
  and both stayed green through the addition. `partial_c` and `partial_xx`
  size for 256 threads apiece.
  `tests/utility_and_pass.rs` holds
  the numbers, with `the_original_three_modes_are_different_functions`
  asserting the
  fixture DISCRIMINATES before the parity cases are believed, and the ablate
  case deliberately at a FRACTIONAL alpha: `ablate` at 1.0 and `clamp` at
  target 0 are the same function, so a mutation swapping the shader's mode
  codes passed that case until the parameters were moved off the degenerate
  point. **`renorm` has its own discrimination case and its own fixture**, and
  the reason is the same trap one turn further on: its effect scales with how
  much of the direction the row carries, and on the DEFAULT fixture the
  rescale factor is 1.006 -- so it would be compared against `ablate` where
  the two genuinely almost agree. `steer_row_carrying` is the fixture that can
  see it. Its parity case also runs at a fractional alpha for a SECOND
  degenerate point: `alpha * (2 - alpha)` equals `alpha` at exactly 1.0, so a
  case at full strength alone cannot tell the correct rescale from a plain
  `alpha` -- and full strength is the natural value to reach for, since making
  it usable is the mode's whole purpose. Both mutations were checked and both
  redden only their own cases.
- `vision.rs` + `shaders/vision.metal`: the `qwen3_5` vision tower's six kernels (ROADMAP M-V2) -- `vision_layer_norm_fp16`, `vision_gelu_tanh_fp16` / `vision_gelu_erf_fp16`, `vision_rope_2d_fp16`, `vision_attention_bidir_fp16`, `vision_matmul_fp16`, `vision_residual_add_fp16`. PORT-LOCAL; their contract is `turbospark_compute::vision`. **They have a PRODUCTION CALLER since M-V4** -- `crates/runtime/src/vision/` dispatches all six per image, and the whole tower now reads cosine 0.99999334 against mlx-vlm at the merger, which is that reference's own FP16-vs-FP32 floor (`docs/VISION.md`). So these are live rather than scaffolding, and a change to any of them owes that gate as well as the per-kernel parity cases. **FIVE THINGS TO READ BEFORE TOUCHING THEM.**
  **Every weight is `half`, not `bfloat`**, alone in this crate. The tower is signed off at FP16 end to end (`docs/VISION_PHASE0.md`): its checkpoints ship F16, the extreme-page probe puts peak activations at 13.8% of FP16's ceiling with a factor of 7.3 in hand, and INT4 was measured and REJECTED on OCR quality. The two types are the same WIDTH, so binding a BF16 tensor here passes every length check and reads the bytes as a different number.
  **LayerNorm reduces TWICE** (mean, then variance about that mean) rather than using the one-pass `E[x^2] - E[x]^2` identity. That identity cancels catastrophically when the mean is large relative to the spread, which is this tower's NORMAL condition rather than an edge case: block 26 runs at absmax 9,024 with rms 145.
  **The two GELUs are separate KERNELS, never one with a mode uniform** -- Gotcha 1's trap, and here it would make the tower's two activations one function. **Metal ships no `erf`**, which reads like it should (`metal_math` has every other libm name); the series is written out as the same Abramowitz-Stegun 7.1.26 the CPU reference uses, deliberately the same approximation so the parity bound measures FP32-vs-FP64 and FP16 storage rather than a gap between rival expansions.
  **The attention is ONE PASS over the keys, online softmax**, with the 8 SIMD groups splitting keys and the 32 lanes splitting the head dimension into a register tile. The straightforward three-pass shape (max, denominator, weighted sum with threads over the head dim) is CORRECT and recomputes every dot product once per output element -- `head_dim` times too much work, a factor of 72 here. The first draft did exactly that. `MAX_ATTENTION_HEAD_DIM` is 128 from that register tile and is REFUSED rather than clamped, since exceeding it truncates a head silently.
  **`vision_matmul_fp16` takes all three position attributes as `uint2`**, which is a Metal rule and not a style: a kernel's position inputs must be all scalar or all vectors of the same width, so a `uint2` threadgroup position beside a scalar `thread_position_in_threadgroup` does not compile. The simdgroup attributes are a different family and stay scalar.
  `tests/vision_parity.rs` holds the per-kernel numbers, each parity case paired with a DISCRIMINATION case (the tower has three places where two functions nearly coincide). `tests/vision_block_parity.rs` is the composition gate: every per-kernel case passes while the block is wired wrongly, so it runs a whole block at the real widths and checks q/k/v slicing, rope reaching q and k but NOT v, both residuals, and that every one of the twelve weights reaches the output.
- `moe_prefill_batch_gguf.rs` + `shaders/moe_prefill_batch_gguf.metal`: the batched routed pair over MXFP4 expert blobs (`docs/BATCHED_PREFILL.md` step 5), which is `gpt-oss`'s only layout. PORT-LOCAL; the contract is BIT-IDENTITY with the MXFP4 DECODE pair run M times, held by `tests/moe_prefill_batch_gguf_parity.rs`. **FOUR THINGS TO READ BEFORE TOUCHING IT.**
  **It is a SUBSTITUTION of `moe_prefill_batch.metal`, not a generalization of it**, and deliberately so: the two blobs share no row addressing (an affine blob has scale and bias PLANES the helper strides through, an MXFP4 block carries its own E8M0 exponent inline in 17 bytes), so a function-constant variant selecting between them would be two kernels wearing one name, with a specialization byte that has to reach `constants_key` or the cache hands back whichever compiled first (Gotcha 1).
  **The activation parameters and `has_bias` ride as UNIFORMS**, inherited from the decode pair's decision for the same reason. `a_plain_activation_is_a_different_function` is what says they reach the kernel at all: the fixture writes bias planes on BOTH arms, so if the uniforms were ignored the `GPT_OSS` and `PLAIN` runs would agree and both parity cases would pass against a kernel that read neither.
  **The bias placement is the family's math and not the block type's**, in two places that a reader will not find by looking at the flow: gate/up biases are added BEFORE the activation, where the clamp reads them (`min(gate + b, limit)`), and the down bias is added PER SLOT INSIDE the routing weight, because it is that expert's own bias on that expert's own output. Both were mutation-checked; each wrong order reddens exactly the three `GPT_OSS` cases and neither `PLAIN` one.
  **This arm was built before the Q4_K/Q6_K one on a measurement, not on convenience** (`docs/BATCHED_PREFILL.md`, "Step 5's two arms"): `gpt-oss`'s routed pair is 61.4% of its prefill GPU device time against Gemma's 38.2%, its un-batchable `pread` is 8.2% against 25-37%, and it is the only family whose union stays under the slot count at every M -- so `the_batched_mxfp4_pair_runs_at_the_full_batch_width` is the repo's ONLY batched-routed case at M=16, both 128-expert families capping at M=8.
  `RoutedBlobsWideBuffer` is shared with the affine pair but is created and bound through `new_for` / `bind_for`, taking the source and function that will READ it. The two libraries declare `RoutedBlobsWide` identically, so one encoder would work today; taking it from the reading function instead means a layout that ever diverged is a compile-time mismatch rather than a silently misread pointer array.
- `resident_metal.rs`: `ResidentGpuWeights` zero-copy `MTLBuffer` wrapping around `mmap` slices.
- `device_memory.rs`: `recommended_max_working_set`, the Metal device's own
  ceiling plus its name, here for the same reason `power_state.rs` is. Adapted
  from shoehorn's `src/vram.rs` (see `NOTICE`), Metal arm only -- its NVML and
  `rocm-smi` arms were dropped rather than adapted, since this crate compiles
  to nothing off macOS and they would be permanently unreachable code reading
  as multi-vendor support this port does not have. **ADVISORY, and nothing
  budgets from it**: both sizing policies budget from `physical_memory()`, and
  a recommendation that disagreed with the `open()` it is recommending would
  be worse than none. On a unified-memory Mac it lands near 75% of installed
  RAM, so it is below `physical - reserve` on essentially every machine and is
  worth reporting per CANDIDATE (does this one's allocation cross it) rather
  than per machine.
- `power_state.rs`: `thermal_state_raw` and `low_power_mode_enabled`, two `NSProcessInfo` message sends backing ROADMAP Phase P2's rate control. They live here rather than in `crates/runtime` because that crate is `#![forbid(unsafe_code)]` and this one already reaches Objective-C through `metal::objc`. Nothing GPU about them beyond that; they take no device and encode nothing. The `#[link(name = "Foundation", kind = "framework")]` block is not decoration: `class!` PANICS on a class it cannot find, and Metal alone does not guarantee Foundation is on the link line.

## Development & Test Commands

```sh
# Run tests for turbospark-gpu (macOS only)
cargo test -p turbospark-gpu
```

## Crate Gotchas

1. **Pipeline Cache Keying on Address**: `MetalContext::pipeline` keys its function and pipeline caches on shader string memory ADDRESS (`&'static str`), NOT string contents. Callers MUST pass identical `include_str!` static constants.
2. **Autorelease Pool Wrapping**: Metal command buffer and compute encoder creations return autoreleased objects. Repeated encode loops MUST be wrapped in `gpu::autorelease_pool`.
3. **MoE Phase 2 Down Reduction**: `moe_phase2_down_reduce_k8` reduces all 8 slots unconditionally. Unused slots must have a 0.0 routing weight, a valid blob pointer, and a finite activation row.
4. **Two shader sources are CONCATENATIONS, and both for the same reason.** `gdn.rs` passes `dequant_int4.metal` + `gdn.metal` (the fused input projection calls `dequant_int4_gemv_simd_body`, a `static inline` in the former); `moe_gguf/` dispatches pass `moe.metal` + `dequant_q4_k.metal` + `dequant_q6_k.metal` + `dequant_iq.metal` + `moe_gguf.metal` (the last uses the first's `RoutedBlobs`, `ExpertOffsets` and `moe_hidden_activation`, and the second's `dequant_q4_k_row_simd`). Both resolve because the Swift build concatenates every module into one library, which is how these files were always compiled. `concat!` of two `include_str!`s is one `&'static str` with one stable address, which is what Gotcha 1's address-keyed cache needs. Never pass a component file's own constant to a dispatch that wants the concatenation (it would miss the cache, not misbehave), and expect the shared file's kernels to be compiled twice in the process.
5. **`gdn_qk_norm` / `gdn_gated_norm` need EXACTLY 128 threads per threadgroup** -- both reduce four SIMD partials with a hardcoded loop. Fewer sums uninitialized slots, more drops work. `NORM_THREADS` pins it. The delta kernels are likewise fixed at `(32, 4)` threads, which is where `GdnShape::validate`'s `Dk % 32 == 0` and `Dk / 32 <= 8` come from.
6. **`PassEncoder` ends encoding on drop, and that is load-bearing rather than tidy.** Every `?` between `begin_pass` and `commit` used to drop an encoder that had never been sent `endEncoding`, and Metal aborts the process from `-[_MTLCommandEncoder dealloc]` when that happens -- while the real error is still travelling up the stack, so the assertion is all anyone sees. `commit` therefore takes its profile with `Option::take` and clones the command buffer instead of moving fields out (a struct with a `Drop` impl cannot be destructured). A new pass-like wrapper needs the same or it reintroduces the blindfold.
7. **KV Cache Ring Specialization**: `KvCacheManager`'s `fp16_ring_enabled` mode specializes `FC_ATTN_RING_CAP` into Metal pipelines. Ring capacity values MUST be included in the pipeline cache constants key.
8. **Command buffers on one queue execute in COMMIT ORDER, and that is what licenses reusing single-row scratch across them.** A GPU-only intermediate written by buffer N and read by buffer N+1 needs no fence and no second copy; the shared-expert-into-routed chain has always relied on it, and `crates/runtime`'s chunked prefill driver relies on it to run several tokens through one set of projection and FFN scratch. What it does NOT cover is the HOST: a buffer the CPU writes (routing weights, an argument buffer) or reads back can race a buffer still in flight, because those writes never enter the queue. When deciding whether something needs a second copy or a per-token row, ask who WRITES it, not who reads it.
9. **The whole-block vision parity test CANNOT see which GELU the block selects, and that is recorded rather than fixed.** The two forms agree to ~3e-4 while a block's output carries the accumulated FP16 error of two norms, five GEMMs of up to 4,304 terms and an attention, so the difference is two orders of magnitude under the bound the parity assertion has to allow. No tightening fixes it. `the_blocks_gelu_choice_is_invisible_at_the_parity_bound` states the limitation as an assertion (the two kinds differ, and by less than the bound) so a reader cannot assume the composition test covers it; the choice is pinned instead by the two per-kernel cases plus the one-line call site. Found by mutation, not by review.

10. **A NORMALIZATION CONVENTION IS A PROPERTY OF THE TENSOR, NOT OF THE
    FAMILY, AND ONE MODEL CAN USE TWO.** `muse_glimmer`'s four per-layer
    norms are centered (the stored weight is an OFFSET FROM UNITY, so the
    effective scale is `1 + w`) while its final `model.norm.weight` is a
    plain RMS norm at `w`. Both live in one forward pass, so "which norm
    does this family use" has no answer; only "which norm does this TENSOR
    use" does. `rmsnorm_bf16w_centered` and `rmsnorm_bf16w` are therefore
    separate kernels selected per dispatch site.

    **The `+1` is a SEPARATE KERNEL and not a function constant.** A
    specialization axis whose byte missed `constants_key` would silently
    reuse whichever pipeline compiled first (Gotcha 1), making the two norms
    of one model the same function: fluent, wrong, invisible. A distinct
    kernel name is a distinct pipeline by construction.

    **The `1 +` is applied on the FP32 accumulator, never baked into the
    stored weight at repack time.** Baking is the obvious cheaper option and
    is lossy where it matters: a centred weight sits near ZERO, and BF16's
    absolute resolution near 1.0 is 2^-8, so a `w` of 0.01 loses ~39% of its
    own magnitude. But re-argue that per tensor rather than quoting it. The
    `qwen3_5` MTP head's centered pair reads 0.780 and 0.797, where baking
    would cost a factor of two in resolution rather than 39% of the
    magnitude; a separate kernel was still taken, on exactness and the
    smaller API.

    **Gemma's `+1` was a real candidate and was REFUTED by measurement**
    (`gguf_norm_convention_probe`: its GGUF and MLX installs' resident cores
    are bit-identical, and llama.cpp's converter does add 1 to Gemma norms),
    so `rms_norm` stays the plain form. Do not "unify" them.

    **A parity fixture whose norm weights sit near zero CANNOT TELL THE TWO
    CONVENTIONS APART**, because `x * w` and `x * (1 + w)` converge as
    `w -> 0`. `the_two_norm_conventions_are_different_functions` asserts the
    fixture discriminates before the parity case is believed.

    The sharper form of the rule is one family over: the `qwen3_5` trunk and
    its MTP head both carry `self_attn.q_norm.weight` at the same shape
    through the same `encode_full_attention_block`, and the head's are
    centered while the trunk's are plain, because mlx-vlm's converter bakes
    the `+1` into the published trunk weights and leaves the head's alone.
    The convention is not a property of the NAME either; it is a property of
    the tensor at a call site, which is why `QkNormConvention` is a
    parameter on that shared function.

11. **`rope_mrope_interleaved` TAKES THREE SCALAR POSITIONS AND RUNS THE
    SELECTOR IN-SHADER, WHICH IS WHAT BUYS ITS EXACTNESS.** M-V2's kernel
    table sketched a host-precomputed cos/sin table for the trunk's mRoPE
    (ROADMAP M-V5, `docs/VISION.md`). The kernel takes `(t, h, w)` plus two
    `mrope_section` entries instead and calls the SAME `apply_neox_pair`
    `rope_neox_subdim` calls, so at `t == h == w` it executes the identical
    float sequence -- BIT-identical, not merely close, which
    `at_t_equals_h_equals_w_it_is_bit_identical_to_rope_neox_subdim` asserts
    on `to_bits`. That is what lets `crates/runtime` keep a mixed prompt's
    TEXT tokens byte-exact against the pre-vision engine. A rewrite that
    precomputes on the host turns a structural guarantee into a
    floating-point coincidence.

    **IMPLEMENT THE TWO `min()` CLAMPS AND NOT THE `i % 3` COLLAPSE.** The
    collapse holds only because this family's `[11, 11, 10]` tiles `freq_dim`
    32 exactly; a config whose sections sum to less leaves a tail of pairs on
    `t` that the collapse would hand to `h` and `w`.

    **AND A CLAMP IS ONLY OBSERVABLE AT A LOW PAIR INDEX**, which is the part
    a reader will not guess. At section `[4, 4, 4]` the clamp first binds at
    pair 13, whose frequency is `theta^(-26/64)` = 0.0014, so deleting it
    moves the output ~3e-3 relative -- UNDER the FP16 bar the parity file has
    to allow, and the mutation survived all five cases. At `[1, 1, 1]` the
    first wrongly-claimed pair is 4 at frequency 0.133 and the same mutation
    reddens. The rule generalises past this kernel: a rope fixture can only
    see a change to a pair whose FREQUENCY is large, so a case about which
    position drives which pair has to place the disagreement near pair 0. A
    tolerance-free selector assertion sits beside it for the same reason.

    Its parity file also carries Gotcha 9's shape a second time: the first
    draft compared per element against an absolute `2e-3` and failed a
    CORRECT kernel at `cpu 16.882074, gpu 16.875`, a gap under one FP16 ULP
    at that magnitude. It uses `rel_error` against `FP16_REDUCTION` now.
