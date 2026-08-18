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
|   +-- attention_decode.rs         # Split-KV decode attention dispatch
|   +-- moe_decode.rs               # MoE router, phase 1 GEMV, phase 2 down-reduce dispatches
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
|   +-- utility.rs                  # Elementwise helpers (scalar mul, softcap, M5's bias add)
|   +-- logit_softmax.rs            # Softcap and logit softmax helpers
|   +-- bytes.rs                    # Metal buffer byte alignment utilities
|   +-- gdn.rs                      # Gated-DeltaNet kernel dispatches (8 kernels)
|   +-- gdn_state.rs                # GDN recurrent state buffers (Qwen flow's)
|   +-- dsv4_state.rs               # DSV4 Metal buffer allocation (unwired)
|   +-- prefill_scratch.rs          # Chunked prefill scratch buffer layout (undispatched)
|   \-- shaders/                    # MSL source, vendored from Swift except where marked port-local
|       +-- attention.metal         # Decode attention Metal shader source
|       +-- dequant_1bit.metal      # MLX 1-bit affine GEMV + the `+/-1` form
|       +-- dequant_2bit.metal      # MLX 2-bit affine GEMV + embedding lookup
|       +-- dequant_int4.metal      # INT4 dequantization GEMV shader source
|       +-- dequant_int8.metal      # INT8 dequantization GEMV shader source
|       +-- dequant_q4_k.metal      # GGUF Q4_K dequantization GEMV + embedding lookup
|       +-- dequant_q5_k.metal      # GGUF Q5_K dequantization GEMV (after dequant_q4_k.metal)
|       +-- dequant_q6_k.metal      # GGUF Q6_K dequantization GEMV shader source
|       +-- moe_gguf.metal          # GGUF MoE decode pairs (after moe.metal + dequant_q4_k.metal)
|       +-- dequant_q8_0.metal      # GGUF Q8_0 dequantization GEMV shader source
|       +-- gdn.metal               # Gated-DeltaNet (Qwen 3.6 linear attention) shader source
|       +-- logit.metal             # Logit softcap and softmax shader source
|       +-- moe.metal               # MoE router GEMV and phase 1/2 shader source
|       +-- rmsnorm.metal           # RMSNorm shader source
|       +-- rope.metal              # RoPE shader source
|       \-- utility.metal           # Elementwise utility shader source
\-- tests/                          # Metal numerical parity & allocation unit tests
    +-- attention_chunk_bench.rs
    +-- attention_decode_parity.rs
    +-- attention_swa.rs
    +-- dequant_1bit_gemv_parity.rs
    +-- dequant_2bit_gemv_parity.rs
    +-- dequant_int4_gemv_parity.rs
    +-- dequant_int8_gemv_parity.rs
    +-- dequant_q4_k_gemv_parity.rs
    +-- dequant_q5_k_gemv_parity.rs
    +-- dequant_q6_k_gemv_parity.rs
    +-- attention_sinks.rs
    +-- moe_gguf_parity.rs
    +-- dequant_q8_0_gemv_parity.rs
    +-- dispatch_profile.rs
    +-- dsv4_state.rs
    +-- gdn_parity.rs
    +-- gdn_state.rs
    +-- gemma4_real_kernels_parity.rs
    +-- kv_cache.rs
    +-- logit_softmax_parity.rs
    +-- moe_decode.rs
    +-- prefill_scratch.rs
    +-- resident_metal.rs
    +-- rms_norm_parity.rs
    +-- rope_yarn_parity.rs
    +-- rope_parity.rs
    +-- scaled_norm_and_embed.rs
    \-- utility_and_pass.rs
```

## Key Modules

- `context/`: `MetalContext` managing `MTLDevice`, `MTLCommandQueue`, pipeline state caching, pass encoders (`PassEncoder`, `CommittedPass`), errors (`GpuError`), and buffer IO.
- `kv_cache.rs`: `KvCacheManager` managing persistent per-layer Metal KV buffers (linear or sliding-window ring).
- `attention_decode.rs`: Split-KV decode attention dispatch (`chunks_for`, up to 16 splits), optionally with ROADMAP M5's ATTENTION SINKS: one learned logit per q head joining the softmax DENOMINATOR and nothing else, folded into `attention_decode_combine`'s existing cross-chunk max and denominator. It rides a FUNCTION CONSTANT (`FC_ATTN_HAS_SINKS`) rather than a uniform, unlike the MXFP4 activation next door, because an unbound buffer is undefined behaviour and the argument therefore has to not exist for the families without one -- which means its byte belongs in `attention_constants_key` beside scale, ring capacity and chunk count (Gotcha 7's rule, fourth axis).
- `moe_decode.rs`: Dispatches MoE router GEMV, phase-1 GEMVs, host router wait, and phase-2 down reduction.
- `rms_norm.rs`: RMSNorm dispatches (`rmsnorm_no_scale`, `rms_norm_bf16w`, `rmsnorm_bf16w_centered`, per-head norm variants). The centered one is PORT-LOCAL (Swift reads no architecture with that convention) and its contract is `turbospark_compute::rms_norm_centered`: `out = x * inv * (1 + w)`, the form whose checkpoint weight is an OFFSET FROM UNITY. **It is a separate KERNEL and not a function constant on its plain sibling**, because `muse_glimmer` uses both conventions in ONE forward pass -- four centered per-layer norms and a plain final one -- so a specialization axis whose byte missed `constants_key` would silently make them the same function (Gotcha 1's trap, and AGENTS.md Gotcha 50). The `1 +` is applied on the FP32 accumulator and never baked into the stored weight: these weights sit near ZERO and BF16's resolution near 1.0 is 2^-8, so a bake would destroy ~39% of a 0.01 weight's magnitude. `tests/scaled_norm_and_embed.rs` holds the parity case and, beside it, `the_two_norm_conventions_are_different_functions`, which asserts the fixture DISCRIMINATES before the parity case is believed -- at weights near zero the two forms converge and a tidy fixture would pass against either kernel.
- `rope.rs`: Rotary positional embedding dispatches (`rope_proportional_neox`, `rope_neox_subdim`, and ROADMAP M5's `rope_neox_freqs`). The first three derive each pair's frequency from a SCALAR theta; `rope_neox_freqs` reads a precomputed per-pair table plus a magnitude scale, because YaRN interpolates per dimension along a ramp and no single number expresses that. The table is position-independent, so it is built once at open by `turbospark_compute::yarn_frequencies`. Incidentally the shape a LEARNED frequency table needs, which is what `rope_freqs.weight` is -- refused by name since M4, and nothing here wires it up.
- `gdn.rs`: The eight gated-DeltaNet dispatches plus `GdnShape` and its structural preconditions.
- `dequant_int4_gemv.rs` & `dequant_int8_gemv.rs`: INT4/INT8 GEMV SIMD dispatches. Since 2026-08-16 the INT4 GEMV bakes M/N as function constants per shape (`specialized_constants`), as does the fused GDN in_proj (`gdn.rs::in_proj_pipeline`, constants 90-94): +3-5% cold weight-read rate isolated, +4.5-8.4% end to end on the dense 27B, output byte-identical on both INT4 families (measured, not assumed -- fast-math unrolling could legally have reassociated the accumulator and did not). The pipeline-cache KEY carries the baked values; a shared key is Gotcha 1's silent-wrong-pipeline trap, and the bench that measured this (`gemv_bandwidth_bench.rs::baking_m_n_...`) fell into it on its own first run (+322% from a wrong-N pipeline reading a third of each row). `embed_lookup_int4` stays unspecialized: a lookup has no trip count to bake.
- `dequant_1bit_gemv.rs`: the MLX `affine` GEMV pair at ONE bit (`prism-ml/Bonsai-27B-mlx-1bit`, ROADMAP's 1-bit entry step 2). PORT-LOCAL for the GGUF set's reason -- the Swift engine reads no 1-bit checkpoint -- and its contract is `turbospark_compute::quant_1bit`. Same CONTAINER as the INT4 sibling next door and NOT a narrower version of it: the companions bind as `device const half*` rather than `bfloat*` (same width, so a misread passes every length check and reads the checkpoint's 0.0271 scales as 1.7e-16), the group size is the checkpoint's 128 rather than the sibling's compile-time 64, and a byte holds 8 elements rather than 2, which makes the LSB-first bit order inside it load-bearing. **The group size is a runtime UNIFORM, deliberately not a function constant**: it is a property of the checkpoint rather than of the container, and a specialization axis that missed `pipeline`'s constants key would silently reuse the wrong pipeline (Gotcha 1). **`dequant_int1_gemv_symmetric_simd` is a SECOND KERNEL, not a fast path inside the first**, for two reasons that both matter: factoring the group scale out of the sum reassociates it (AGENTS.md Gotcha 27), and it binds no bias plane at all, so `Int1SymmetricRowGpu` has no field through which an unchecked bias could reach it -- the `bias == -scale/2` check is the caller's, and it is measured (`turbospark_compute::is_symmetric`) rather than assumed. The symmetric form has no resident variant on purpose, and step 4 turned that from "not yet" into a decision: the repack step writes a bias plane for every 1-bit tensor whether or not it is symmetric, so the fast path would have to be SELECTED per tensor at dispatch time, which would make the generated bytes a function of which tensors happened to quantize symmetrically. Reaching it needs its own dtype tag written at repack time, not a check at the dispatch. **`embed_lookup_int1` is the third kernel and it was not in the plan**: the checkpoint's own safetensors header says `embed_tokens` and `lm_head` are BOTH quantized at one bit (`U32 [248320, 160]`, F16 companions), and while the head is just another matrix through the GEMV, the table needs a lookup. So this type's footing is a GEMV plus a lookup and NO routed-expert pair -- Gotcha 29's per-type rule as the one real file decides it, the model being dense. Since step 4 the `qwen3_5` decode flow dispatches the general GEMV's resident form and the lookup, deriving the group size from each tensor's own companion planes (`crates/runtime` Gotcha 12); the symmetric one is reachable from no flow. `tests/dequant_1bit_gemv_parity.rs` is still where the NUMBERS are held, and its two trap cases (bit order, companion dtype) construct data that discriminates and then ASSERT that it does, because at one bit a wrong reading leaves every magnitude, every group scale and the total popcount untouched.
- `dequant_2bit_gemv.rs`: the MLX `affine` GEMV plus embedding lookup at TWO bits (`prism-ml/Ternary-Bonsai-27B-mlx-2bit`, ROADMAP's ternary entry step 2). PORT-LOCAL for the 1-bit sibling's reason, and its contract is `turbospark_compute::quant_2bit`. It inherits that sibling's three departures from the INT4 GEMV -- `half` companions rather than `bfloat` (same width, so a misread reads this checkpoint's 0.0137 scales as 7e-18), the checkpoint's group size as a runtime UNIFORM rather than a compile-time 64, and a byte holding several elements so the LSB-first field order is load-bearing -- and differs from it in one: **TWO kernels, not three.** The checkpoint is TERNARY (`bias == -scale` in every probed group, level 3 never used), so the analogue of the 1-bit `+/-1` fast path exists mathematically and is deliberately not built: it would reassociate the sum the same way (AGENTS.md Gotcha 27), and the 1-bit one is already reachable from no decode flow, so a second dead kernel buys nothing. **The GEMV is the GENERAL affine form and decodes level 3 correctly**, which `the_fourth_level_is_decoded_not_clamped` pins -- a kernel written from the ternary description could mask the top bit and pass every other case. `tests/dequant_2bit_gemv_parity.rs` holds the numbers, with the two trap cases (field order, companion dtype) constructing discriminating data and then ASSERTING that it discriminates, because at two bits a wrong field order permutes within a run of four and leaves every magnitude, every group scale and the whole level histogram untouched.
- `dequant_int4_batch.rs`: `dequant_int4_gemm_simd`, the M-row batched form of the above (ROADMAP Phase D2's verify kernel). Not on any decode path -- decode is M=1 -- and kept because a tile-kernel phase would otherwise rebuild it. **Read its header before optimizing it**: threadgroup staging of `x` and register blocking over rows are both measured LOSSES on every shape. Since 2026-08-17 it bakes M, N and **B** as function constants (100-103) and bounds its inner unroll at 4, which together roughly HALVED `c(M)` -- 0.85 to 0.46 at M=8 on the dense 27B shapes (`docs/MTP_SPECULATIVE.md`). B is the one that matters and the unroll bound is not optional: baking B lets the compiler unroll fully, and a full unroll at B=16 holds ~128 activation floats live beside `acc[16]` and SPILLS, reading 1.14 where the bounded form reads 0.44. The factor was swept (2/4/8/full), not chosen. Parity stays EXACT, which is what keeps a batched verify bit-identical to a sequential decode; `tests/dequant_int4_gemm_parity.rs` pins that and, in `a_second_shape_in_one_process_does_not_reuse_the_first_shapes_pipeline`, pins that the cache key carries all three baked values (mutation-checked on each). Beside it, `dequant_int4_mma.metal`'s `dequant_int4_gemm_mma` is the `simdgroup_matrix` form and is a MEASURED DEAD END kept so it is not re-proposed: it loses at every batch width (6.6x at M=2, 1.33x at M=16) and PLATEAUS at ~0.5 past M=16, worse than the exact kernel, because a packed INT4 run cannot be `simdgroup_load`ed and the dequant-into-threadgroup work it therefore needs is independent of B while the MACs matrix hardware accelerates scale with B. Nothing dispatches it. `tests/dequant_int4_mma_parity.rs` is the repo's only NON-exact batched-kernel test and says why in its header; its fixture deliberately uses non-power-of-two companions, and asserts that it discriminates, because the exact test's fixture sums exactly in FP32 and cannot see reassociation at all (the first run read 4096/4096 bit-identical and bounded nothing).
- `dequant_iq_gemv.rs`: the IQ3_XXS / IQ4_NL / IQ4_XS codebook GEMV dispatches (ROADMAP Phase S), whose tables are GENERATED from libggml by `scripts/ggml_tables.c` rather than transcribed, so the CPU and MSL copies cannot drift. `tests/dequant_iq_gemv_parity.rs`.
- `dequant_q8_0_gemv.rs`: the GGUF Q8_0 GEMV dispatch (ROADMAP Phase G Stage 2). PORT-LOCAL, not vendored -- the Swift engine has no GGUF intake, so its only contract is `turbospark_compute::dequant_q8_0_gemv`. Shorter than the INT8 sibling because a Q8_0 row is ONE byte run: the scale lives inside each 34-byte block, so there are no scale or bias planes to bind and no group size to agree on. 32 lanes over a 32-element block, one weight per lane.
- `dequant_q4_k_gemv.rs`: the GGUF Q4_K GEMV dispatch, port-local for the same reason. Same 32-lane shape for a DIFFERENT reason: a lane owns one of the 32 nibble BYTES in a group, hence two elements 32 apart that belong to two different sub-blocks. Q4_K is two-level (f16 `d`/`dmin` per 256-element superblock, 6-bit scale and 6-bit min per 32-element sub-block, packed 12 bytes and split across bytes for sub-blocks 4..8) and asymmetric (`w = d*sc*q - dmin*m`, unsigned quants), so none of the Q8_0 habits carry over.
- `dequant_q5_k_gemv.rs`: the GGUF Q5_K GEMV dispatch, port-local (ROADMAP Phase M2). Exists for one tensor shape: Mixtral 8x7B's Q4_K_M carries `attn_output` in Q5_K while its experts are Q4_K, so there is no embedding-lookup and no MoE sibling. Q5_K is Q4_K plus a fifth bit in its own 32-byte `qh` run, indexed by the element's position WITHIN a sub-block at a bit that advances with the 64-element group, so one `qh` byte is read eight times. Its shader source is a CONCATENATION with `dequant_q4_k.metal` (Gotcha 4): the 6-bit scale/min packing is Q4_K's byte for byte and it calls `q4_k_scale_min` rather than carrying a second copy.
- `dequant_q6_k_gemv.rs`: the GGUF Q6_K GEMV dispatch, port-local. Exists for ONE tensor: Qwen 3.6's Q4_K_M carries a single Q6_K weight, `output.weight`, and nothing else in either real file uses the type, so there is deliberately no embedding-lookup or MoE sibling. Same 32-lane shape for a third reason: a lane owns one `qh` byte and hence the four elements it supplies high bits for, 32 apart inside a 128-element half. Q6_K splits an element's six bits across `ql` and `qh`, biases the quant by a fixed 32 rather than carrying a per-sub-block min, and stores sixteen SIGNED int8 sub-block scales as plain bytes.
- `moe_gguf/`: the routed-expert decode pairs for GGUF expert blobs, port-local: `moe_phase1_gate_up_act_{q8_0,q4_k,mxfp4}` + `moe_phase2_down_reduce_k8_{q8_0,q4_k,mxfp4}`, plus phase-1-only IQ3_XXS/IQ4_XS, phase-2-only IQ4_NL, and (ROADMAP Phase M2) a phase-2-only Q6_K for the `ffn_down_exps` of 16 of Mixtral's 32 layers -- the first mixture whose two types differ across LAYERS rather than across the phases of one expert. The Q4_K pair does not re-implement the unpack; it calls `dequant_q4_k_row_simd` out of `dequant_q4_k.metal`, which joins the concatenation ahead of `moe_gguf.metal`. Both pairs share one host-side dispatch body, so only the kernel name and the block-size precondition differ (32 elements per row for Q8_0, 256 for Q4_K). Separate kernels rather than a function-constant variant of the vendored pair because the two blob layouts share no addressing: a GGUF blob has no scale or bias planes at all, so six of the nine `MoeExpertOffsets` fields are zero and only the three weight offsets locate anything. Shares the vendored pair's `RoutedBlobs` argument buffer, which is one array of eight pointers whatever the blobs contain. **MXFP4 (ROADMAP M5) is the one pair that does not fit that description, in two ways.** It has BOTH phases and no resident GEMV, the mirror of Q5_K's and Q6_K's footing, because `gpt-oss` puts MXFP4 in `ffn_*_exps` and nowhere else -- so its row unpack is written in `moe_gguf.metal` rather than borrowed, there being no `dequant_mxfp4.metal` to borrow from. And it carries the FAMILY's expert math as well as the block layout: the clamped SwiGLU (`min(gate, limit)`, a swish with `alpha`, times `(clamp(up) + 1)`, all from ggml's `ggml_compute_forward_swiglu_oai_f32`) and the per-expert biases the blob holds at the three `MoeExpertOffsets` bias fields every earlier GGUF install left at zero. Both ride as UNIFORMS via `Mxfp4Activation`, never function constants: `constants_key` is one byte wide and a specialization axis that misses it silently reuses the wrong pipeline. `Mxfp4Activation::PLAIN` keeps the block-type parity cases and the family cases in one kernel so neither hides the other. A 17-byte block also makes MXFP4 rows odd-length, which is why every read in it is `uint8_t`: an `as_type<half>` copied in from the Q8_0 helper next door would be misaligned on half the rows.
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
6. **`PassEncoder` ends encoding on drop, and that is load-bearing rather than tidy.** Every `?` between `begin_pass` and `commit` used to drop an encoder that had never been sent `endEncoding`, and Metal aborts the process from `-[_MTLCommandEncoder dealloc]` when that happens -- while the real error is still travelling up the stack, so the assertion is all anyone sees. `commit` therefore takes its profile with `Option::take` and clones the command buffer instead of moving fields out (a struct with a `Drop` impl cannot be destructured). A new pass-like wrapper needs the same or it reintroduces the blindfold. See AGENTS.md Gotcha 32.
7. **KV Cache Ring Specialization**: `KvCacheManager`'s `fp16_ring_enabled` mode specializes `FC_ATTN_RING_CAP` into Metal pipelines. Ring capacity values MUST be included in the pipeline cache constants key.
8. **Command buffers on one queue execute in COMMIT ORDER, and that is what licenses reusing single-row scratch across them.** A GPU-only intermediate written by buffer N and read by buffer N+1 needs no fence and no second copy; the shared-expert-into-routed chain has always relied on it, and `crates/runtime`'s chunked prefill driver relies on it to run several tokens through one set of projection and FFN scratch. What it does NOT cover is the HOST: a buffer the CPU writes (routing weights, an argument buffer) or reads back can race a buffer still in flight, because those writes never enter the queue. When deciding whether something needs a second copy or a per-token row, ask who WRITES it, not who reads it.
