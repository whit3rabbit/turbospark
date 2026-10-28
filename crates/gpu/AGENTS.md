# turbospark-gpu

This is the Metal boundary. Small dispatch mistakes can produce plausible text,
so the real-device checks below matter.

Keep it observable. A kernel can compile and still use the wrong stride, cache,
threadgroup shape, or normalization convention on the device.

## Read first

- [Detailed module guide](../../.claude/docs/modules/gpu.md)
- [Vision pipeline](../../docs/VISION.md)
- [Speech to text](../../docs/SPEECH_TO_TEXT.md)
- [Audio music generation](../../docs/audio/music.md)
- [Decode and memory guardrails](../../.claude/docs/engineering-gotchas.md)
- [Kernel modes](../../docs/KERNEL_MODES.md): read before adding an
  alternative kernel path behind a switch.

## Rules

- This crate is macOS-only and requires a Metal-capable device plus Xcode's
  Metal toolchain. CPU compilation is not hardware evidence.
- Gate every repeated Metal encode loop with `gpu::autorelease_pool`.
- Include dispatch-shape and specialization constants in the pipeline cache key.
  A ring capacity or layer shape silently reusing the wrong pipeline is a
  correctness bug.
- Gated GDN normalization uses exactly 128 threads per threadgroup. Preserve
  the kernel's layout contract.
- Music 3 kernels (`music3.metal`, `music3.rs`) carry an MLX bit-exactness
  contract. Before and after any change there, run `scripts/verify-music3.sh`;
  add `--mlx MODEL_DIR` when a dense or quantized linear kernel, a row-count
  threshold, or the depth/flow call shapes changed. A passing hash oracle does
  not cover dense kernels, and a raised tolerance is a finding, not a fix. Read
  [docs/audio/minimax_music3.md](../../docs/audio/minimax_music3.md) first.
- Split-KV decode attention: the GQA-shared partial (`attention_decode_gqa_partial`)
  serves all Q heads in a group from one K/V load. Its function constants
  (group, simdgroups, head_dim, unroll) MUST be appended to the pipeline cache
  key to prevent silent pipeline reuse across shapes.
- GQA decode attention supports head_dim 128 or 256 and groups up to 8
  (`MAX_GQA_GROUP`). Its chunk count is capped at `MAX_CHUNKS`, so it shares
  the barrier kernel's `AttentionScratch` size (more chunks measured slower).
  A family opts in on its scratch (`with_decode_kernel`), and
  `TURBOSPARK_ATTN_DECODE_KERNEL=barrier|gqa` overrides it process-wide.
- The multi-row GQA partial (`attention_decode_gqa_rows_partial`) bakes
  group, simdgroups, head_dim, unroll, tile and heads per simdgroup into
  function constants, all in its cache key. Its chunk boundaries are shared
  by every row and its summation order differs from the one-row kernel, so
  it is not bit-exact against it: it serves the chunked-prefill scratch only
  (`BatchedScratch::rows_attention`), never decode or speculative verify.
- A kernel pair that decode and speculative verify both reach for the same
  op (one-row and multi-row forms) must sum in the SAME order, term by term.
  `gdn_conv_mix_decode` and `gdn_conv_mix_prefill` once associated their
  four taps differently, and that one-ulp f32 difference survived half
  rounding about once in ten thousand elements: it was what made greedy
  speculation leave sequential decode on the real 27B. Random fixture data
  rarely reaches a rounding boundary, so pin such a pair at the real width
  with a full, nonzero state (`tests/gdn_conv_order_parity.rs`).
- `shaders/dequant_int4_steel.metal` is generated from pinned MLX sources by
  `scripts/vendor_mlx_steel_qmm.py`. Change the script and regenerate, never
  the shader. Steel is not bit-exact against the scalar INT4 kernels, so it
  stays prefill-only (`BatchedScratch::steel_gemm`), never decode or verify.
- GDN snapshots are GPU blits into pooled private buffers. `snapshot` does
  not wait, so no host write to the live state may land before a later
  command buffer on the same queue has completed.
- `ane_split.rs` (`GpuAneSplit`) bridges shared ANE output planes to Metal
  resident buffers. A non-finite value or NaN/Inf conversion flag signals a
  fallback to the Steel path for that micro-batch.
- Multi-row RMSNorm (`rmsnorm_bf16w_perhead`) and multi-row RoPE encode
  contiguous rows as head groups or unified dispatches; each row must compute
  the identical bits to the sequential single-row execution.
- Keep tensor normalization conventions attached to the tensor path, not
  inferred from the model family.
- Validate numerical kernel changes with a real model, sampled output, and an
  independent parity or quality check where available.
- The MLX INT3 routed pair (`moe_phase1_gate_up_act_int3`,
  `moe_phase2_down_reduce_int3` in `moe.metal`) reads weights a byte at a
  time (3-bit rows align to nothing wider) and repeats the INT4 pair's
  accumulation term by term, so codes that fit 3 bits decode bit-identically
  at either width; the runtime's Flint test relies on that. It binds through
  the INT4 phase-1 kernel's `RoutedBlobs` reflection
  (`tests/moe_decode_int3.rs` pins that the two agree).
- Fused INT4 FFN kernels (`dequant_int4_fused.metal`):
  `dequant_int4_gemv_fused_gate_up_act` and `dequant_int4_gemv_fused_down_epilogue`
  must remain bit-exact to separate GEMV and activation dispatches.
- The small-M INT4 matrix-unit GEMM (`dequant_int4_mma8.rs`) serves 1..=8
  token rows and is opt-in end to end (`TURBOSPARK_INT4_MMA8`, read by the
  runtime; default off). It is not bit-identical to the INT4 GEMV (another
  summation order), but a row's bits depend only on its own inputs, never on
  how many rows share the tile. That holds only while the K split stays a
  function of the shape alone and the code mode stays subnormal-only, both
  baked per shape with the N/K/KS function constants always in the pipeline
  cache key. Read `docs/INT4_SMALL_M_MMA.md` before touching it.
- GDN lazy tape fold (`gdn_state_fold`): performs register-fused accumulation of
  rank-1 updates and writes state once at round end. Commit command buffers are
  dispatched asynchronously on the GPU without blocking CPU drafting.
- GDN delta kernels (decode, eager and lazy prefill, fold) bake the key head
  dim through `FC_GDN_DK` (constant 98) when it is 128, so the per-lane state
  loop unrolls. Build them through `gdn::delta_pipeline`, whose key carries
  (RECORD, WRITE_STATE, Dk), never the plain `gdn::pipeline` helper. The baked
  and runtime builds must stay bit-identical (`src/gdn_tests.rs`, including
  its distinct-pipeline guard: a key without Dk passes the bit test
  vacuously). `TURBOSPARK_GDN_DK128=0` restores the runtime build.

## Checks

```sh
cargo test -p turbospark-gpu
cargo fmt --check
```

Run the real Metal gates from the linked verification and model-gate pages.

## Open issues

- `context::device::cache_tests::pipeline_cache_concurrent_writers_publish_complete_entries`
  failed once on the `verify (macos-latest)` CI job (commit `5d64049a`) and
  passed on the next runs of unrelated commits, so it looks timing-dependent.
  It was not reproduced locally. If it fails again, capture the assertion
  output before rerunning, because a rerun hides the evidence. A pipeline cache
  that publishes a partial entry under concurrent writers would be a real
  correctness bug, so do not just mark it ignored.

