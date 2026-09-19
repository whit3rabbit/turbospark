# turbospark-compute

Portable numerical primitives and quantized tensor helpers.

## Read first

- [Detailed module guide](../../.claude/docs/modules/compute.md)
- [Benchmark and quality limits](../../docs/BENCHMARKS.md)
- [Testing rules](../../docs/TESTING.md)

## Rules

- Keep `#![forbid(unsafe_code)]`.
- FP16 uses the `half` crate. BF16 conversion follows the existing
  bit-shift implementation. Do not introduce a second representation.
- Preserve quantization block layout and scale semantics. A self-consistency
  test is not an independent numerical oracle.
- Prefer exact fixture assertions. Mutation-check tests that guard a subtle
  packing or dequantization rule.
- Keep this crate portable. Cross-target compilation is part of the contract.

## Checks

```sh
cargo test -p turbospark-compute
cargo check --target x86_64-unknown-linux-gnu -p turbospark-compute
```
