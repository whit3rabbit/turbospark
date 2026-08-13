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
|   +-- dequant_int4_gemv.rs        # INT4 SIMD GEMV dispatches (resident & streamed)
|   +-- dequant_int4_batch.rs       # The M-ROW batched INT4 GEMM (ROADMAP Phase D2); see its header for the two register-file dead ends
|   +-- dequant_iq_gemv.rs          # IQ3_XXS / IQ4_NL / IQ4_XS codebook GEMV (Phase S)
|   +-- dequant_int8_gemv.rs        # INT8 SIMD GEMV dispatches (resident & streamed)
|   +-- dequant_q4_k_gemv.rs        # GGUF Q4_K SIMD GEMV + embed lookup (port-local, Phase G)
|   +-- dequant_q5_k_gemv.rs        # GGUF Q5_K SIMD GEMV (port-local, Phase M2)
|   +-- dequant_q6_k_gemv.rs        # GGUF Q6_K SIMD GEMV dispatch (port-local, Phase G)
|   +-- moe_gguf.rs                 # GGUF Q8_0 and Q4_K routed-expert decode pairs (port-local, Phase G)
|   +-- dequant_q8_0_gemv.rs        # GGUF Q8_0 SIMD GEMV dispatch (port-local, Phase G)
|   +-- resident_metal.rs           # ResidentGpuWeights mmap zero-copy MTLBuffer wrapper
|   +-- dispatch_profile.rs         # Intra-command-buffer dispatch profiler
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
- `rms_norm.rs`: RMSNorm dispatches (`rmsnorm_no_scale`, `rms_norm_bf16w`, per-head norm variants).
- `rope.rs`: Rotary positional embedding dispatches (`rope_proportional_neox`, `rope_neox_subdim`, and ROADMAP M5's `rope_neox_freqs`). The first three derive each pair's frequency from a SCALAR theta; `rope_neox_freqs` reads a precomputed per-pair table plus a magnitude scale, because YaRN interpolates per dimension along a ramp and no single number expresses that. The table is position-independent, so it is built once at open by `turbospark_compute::yarn_frequencies`. Incidentally the shape a LEARNED frequency table needs, which is what `rope_freqs.weight` is -- refused by name since M4, and nothing here wires it up.
- `gdn.rs`: The eight gated-DeltaNet dispatches plus `GdnShape` and its structural preconditions.
- `dequant_int4_gemv.rs` & `dequant_int8_gemv.rs`: INT4/INT8 GEMV SIMD dispatches.
- `dequant_1bit_gemv.rs`: the MLX `affine` GEMV pair at ONE bit (`prism-ml/Bonsai-27B-mlx-1bit`, ROADMAP's 1-bit entry step 2). PORT-LOCAL for the GGUF set's reason -- the Swift engine reads no 1-bit checkpoint -- and its contract is `turbospark_compute::quant_1bit`. Same CONTAINER as the INT4 sibling next door and NOT a narrower version of it: the companions bind as `device const half*` rather than `bfloat*` (same width, so a misread passes every length check and reads the checkpoint's 0.0271 scales as 1.7e-16), the group size is the checkpoint's 128 rather than the sibling's compile-time 64, and a byte holds 8 elements rather than 2, which makes the LSB-first bit order inside it load-bearing. **The group size is a runtime UNIFORM, deliberately not a function constant**: it is a property of the checkpoint rather than of the container, and a specialization axis that missed `pipeline`'s constants key would silently reuse the wrong pipeline (Gotcha 1). **`dequant_int1_gemv_symmetric_simd` is a SECOND KERNEL, not a fast path inside the first**, for two reasons that both matter: factoring the group scale out of the sum reassociates it (AGENTS.md Gotcha 27), and it binds no bias plane at all, so `Int1SymmetricRowGpu` has no field through which an unchecked bias could reach it -- the `bias == -scale/2` check is the caller's, and it is measured (`turbospark_compute::is_symmetric`) rather than assumed. The symmetric form has no resident variant yet on purpose: which offsets it would bind depends on whether the repack step writes a bias plane for a symmetric tensor at all. Nothing dispatches either kernel yet; `tests/dequant_1bit_gemv_parity.rs` is the whole of its current use, and its two trap cases (bit order, companion dtype) construct data that discriminates and then ASSERT that it does, because at one bit a wrong reading leaves every magnitude, every group scale and the total popcount untouched.
- `dequant_int4_batch.rs`: `dequant_int4_gemm_simd`, the M-row batched form of the above (ROADMAP Phase D2's verify kernel). Not on any decode path -- decode is M=1 -- and kept because a tile-kernel phase would otherwise rebuild it. **Read its header before optimizing it**: threadgroup staging of `x` and register blocking over rows are both measured LOSSES on every shape, and the reason (the register file cannot hold an M-wide activation tile, and threadgroup barriers cost more than they save) is the argument for `simdgroup_matrix` being the only remaining lever. `tests/dequant_int4_gemm_parity.rs` pins it against the GEMV on identical bytes.
- `dequant_iq_gemv.rs`: the IQ3_XXS / IQ4_NL / IQ4_XS codebook GEMV dispatches (ROADMAP Phase S), whose tables are GENERATED from libggml by `scripts/ggml_tables.c` rather than transcribed, so the CPU and MSL copies cannot drift. `tests/dequant_iq_gemv_parity.rs`.
- `dequant_q8_0_gemv.rs`: the GGUF Q8_0 GEMV dispatch (ROADMAP Phase G Stage 2). PORT-LOCAL, not vendored -- the Swift engine has no GGUF intake, so its only contract is `turbospark_compute::dequant_q8_0_gemv`. Shorter than the INT8 sibling because a Q8_0 row is ONE byte run: the scale lives inside each 34-byte block, so there are no scale or bias planes to bind and no group size to agree on. 32 lanes over a 32-element block, one weight per lane.
- `dequant_q4_k_gemv.rs`: the GGUF Q4_K GEMV dispatch, port-local for the same reason. Same 32-lane shape for a DIFFERENT reason: a lane owns one of the 32 nibble BYTES in a group, hence two elements 32 apart that belong to two different sub-blocks. Q4_K is two-level (f16 `d`/`dmin` per 256-element superblock, 6-bit scale and 6-bit min per 32-element sub-block, packed 12 bytes and split across bytes for sub-blocks 4..8) and asymmetric (`w = d*sc*q - dmin*m`, unsigned quants), so none of the Q8_0 habits carry over.
- `dequant_q5_k_gemv.rs`: the GGUF Q5_K GEMV dispatch, port-local (ROADMAP Phase M2). Exists for one tensor shape: Mixtral 8x7B's Q4_K_M carries `attn_output` in Q5_K while its experts are Q4_K, so there is no embedding-lookup and no MoE sibling. Q5_K is Q4_K plus a fifth bit in its own 32-byte `qh` run, indexed by the element's position WITHIN a sub-block at a bit that advances with the 64-element group, so one `qh` byte is read eight times. Its shader source is a CONCATENATION with `dequant_q4_k.metal` (Gotcha 4): the 6-bit scale/min packing is Q4_K's byte for byte and it calls `q4_k_scale_min` rather than carrying a second copy.
- `dequant_q6_k_gemv.rs`: the GGUF Q6_K GEMV dispatch, port-local. Exists for ONE tensor: Qwen 3.6's Q4_K_M carries a single Q6_K weight, `output.weight`, and nothing else in either real file uses the type, so there is deliberately no embedding-lookup or MoE sibling. Same 32-lane shape for a third reason: a lane owns one `qh` byte and hence the four elements it supplies high bits for, 32 apart inside a 128-element half. Q6_K splits an element's six bits across `ql` and `qh`, biases the quant by a fixed 32 rather than carrying a per-sub-block min, and stores sixteen SIGNED int8 sub-block scales as plain bytes.
- `moe_gguf.rs`: the routed-expert decode pairs for GGUF expert blobs, port-local: `moe_phase1_gate_up_act_{q8_0,q4_k,mxfp4}` + `moe_phase2_down_reduce_k8_{q8_0,q4_k,mxfp4}`, plus phase-1-only IQ3_XXS/IQ4_XS, phase-2-only IQ4_NL, and (ROADMAP Phase M2) a phase-2-only Q6_K for the `ffn_down_exps` of 16 of Mixtral's 32 layers -- the first mixture whose two types differ across LAYERS rather than across the phases of one expert. The Q4_K pair does not re-implement the unpack; it calls `dequant_q4_k_row_simd` out of `dequant_q4_k.metal`, which joins the concatenation ahead of `moe_gguf.metal`. Both pairs share one host-side dispatch body, so only the kernel name and the block-size precondition differ (32 elements per row for Q8_0, 256 for Q4_K). Separate kernels rather than a function-constant variant of the vendored pair because the two blob layouts share no addressing: a GGUF blob has no scale or bias planes at all, so six of the nine `MoeExpertOffsets` fields are zero and only the three weight offsets locate anything. Shares the vendored pair's `RoutedBlobs` argument buffer, which is one array of eight pointers whatever the blobs contain. **MXFP4 (ROADMAP M5) is the one pair that does not fit that description, in two ways.** It has BOTH phases and no resident GEMV, the mirror of Q5_K's and Q6_K's footing, because `gpt-oss` puts MXFP4 in `ffn_*_exps` and nowhere else -- so its row unpack is written in `moe_gguf.metal` rather than borrowed, there being no `dequant_mxfp4.metal` to borrow from. And it carries the FAMILY's expert math as well as the block layout: the clamped SwiGLU (`min(gate, limit)`, a swish with `alpha`, times `(clamp(up) + 1)`, all from ggml's `ggml_compute_forward_swiglu_oai_f32`) and the per-expert biases the blob holds at the three `MoeExpertOffsets` bias fields every earlier GGUF install left at zero. Both ride as UNIFORMS via `Mxfp4Activation`, never function constants: `constants_key` is one byte wide and a specialization axis that misses it silently reuses the wrong pipeline. `Mxfp4Activation::PLAIN` keeps the block-type parity cases and the family cases in one kernel so neither hides the other. A 17-byte block also makes MXFP4 rows odd-length, which is why every read in it is `uint8_t`: an `as_type<half>` copied in from the Q8_0 helper next door would be misaligned on half the rows.
- `resident_metal.rs`: `ResidentGpuWeights` zero-copy `MTLBuffer` wrapping around `mmap` slices.
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
4. **Two shader sources are CONCATENATIONS, and both for the same reason.** `gdn.rs` passes `dequant_int4.metal` + `gdn.metal` (the fused input projection calls `dequant_int4_gemv_simd_body`, a `static inline` in the former); `moe_gguf.rs` passes `moe.metal` + `dequant_q4_k.metal` + `dequant_q6_k.metal` + `dequant_iq.metal` + `moe_gguf.metal` (the last uses the first's `RoutedBlobs`, `ExpertOffsets` and `moe_hidden_activation`, and the second's `dequant_q4_k_row_simd`). Both resolve because the Swift build concatenates every module into one library, which is how these files were always compiled. `concat!` of two `include_str!`s is one `&'static str` with one stable address, which is what Gotcha 1's address-keyed cache needs. Never pass a component file's own constant to a dispatch that wants the concatenation (it would miss the cache, not misbehave), and expect the shared file's kernels to be compiled twice in the process.
5. **`gdn_qk_norm` / `gdn_gated_norm` need EXACTLY 128 threads per threadgroup** -- both reduce four SIMD partials with a hardcoded loop. Fewer sums uninitialized slots, more drops work. `NORM_THREADS` pins it. The delta kernels are likewise fixed at `(32, 4)` threads, which is where `GdnShape::validate`'s `Dk % 32 == 0` and `Dk / 32 <= 8` come from.
6. **`PassEncoder` ends encoding on drop, and that is load-bearing rather than tidy.** Every `?` between `begin_pass` and `commit` used to drop an encoder that had never been sent `endEncoding`, and Metal aborts the process from `-[_MTLCommandEncoder dealloc]` when that happens -- while the real error is still travelling up the stack, so the assertion is all anyone sees. `commit` therefore takes its profile with `Option::take` and clones the command buffer instead of moving fields out (a struct with a `Drop` impl cannot be destructured). A new pass-like wrapper needs the same or it reintroduces the blindfold. See AGENTS.md Gotcha 32.
7. **KV Cache Ring Specialization**: `KvCacheManager`'s `fp16_ring_enabled` mode specializes `FC_ATTN_RING_CAP` into Metal pipelines. Ring capacity values MUST be included in the pipeline cache constants key.
