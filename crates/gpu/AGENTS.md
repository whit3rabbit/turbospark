# turbospark-gpu

This is the Metal boundary. Small dispatch mistakes can produce plausible text,
so the real-device checks below matter.

Keep it observable. A kernel can compile and still use the wrong stride, cache,
threadgroup shape, or normalization convention on the device.

## Read first

- [Detailed module guide](../../.claude/docs/modules/gpu.md)
- [Vision pipeline](../../docs/VISION.md)
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
