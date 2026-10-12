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

## Rules

- This crate is macOS-only and requires a Metal-capable device plus Xcode's
  Metal toolchain. CPU compilation is not hardware evidence.
- Gate every repeated Metal encode loop with `gpu::autorelease_pool`.
- Include dispatch-shape and specialization constants in the pipeline cache key.
  A ring capacity or layer shape silently reusing the wrong pipeline is a
  correctness bug.
- Gated GDN normalization uses exactly 128 threads per threadgroup. Preserve
  the kernel's layout contract.
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
- `shaders/dequant_int4_steel.metal` is generated from pinned MLX sources by
  `scripts/vendor_mlx_steel_qmm.py`. Change the script and regenerate, never
  the shader. Steel is not bit-exact against the scalar INT4 kernels, so it
  stays prefill-only (`BatchedScratch::steel_gemm`), never decode or verify.
- GDN snapshots are GPU blits into pooled private buffers. `snapshot` does
  not wait, so no host write to the live state may land before a later
  command buffer on the same queue has completed.
- Keep tensor normalization conventions attached to the tensor path, not
  inferred from the model family.
- Validate numerical kernel changes with a real model, sampled output, and an
  independent parity or quality check where available.

## Checks

```sh
cargo test -p turbospark-gpu
cargo fmt --check
```

Run the real Metal gates from the linked verification and model-gate pages.
