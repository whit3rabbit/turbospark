# turbospark-bench

Measurement harnesses for throughput, memory, quality, cross-engine parity,
power, steering, and speculative decoding.

## Read first

- [Benchmarking protocol](../../docs/BENCHMARKING.md)
- [Frozen benchmark rows](../../docs/BENCHMARKS.md)
- [Detailed benchmark commands and gate notes](../../.claude/docs/benchmarks.md)
- [Per-family memory and quality gates](../../.claude/docs/model-gates.md)
- [Cross-engine checks](../../.claude/docs/cross-engine-kl.md)
- [Power measurements](../../.claude/docs/power-measurement.md)

## Directory map

- `src/bin/`: benchmark and probe binaries.
- `src/real_model.rs`: shared real-install protocol and model setup.
- `src/real_model_params.rs`: family-specific context and budget parameters.
- `tests/`: offline fixtures plus ignored real-install gates.
- `scripts/`: measurement and analysis helpers.

## Rules

- Never quote a number without its model family, artifact, context window,
  generation budget, chip, power source, and warmup conditions.
- Run formal measurements on AC power with a quiet host. Interleave A/B arms
  instead of running all of one arm before the other.
- A peak footprint is not an on-disk model size. MoE expert tables stream, while
  resident weights, slots, and KV cache contribute to the process footprint.
- CPU, synthetic, projected, or host-contaminated results do not establish a
  real Metal, quality, or memory claim.
- Preserve frozen vectors, fixtures, and baselines. Re-extraction or a changed
  protocol is a new experiment.
- Keep benchmark procedures here and in the canonical docs, never in the root
  AGENTS.md or unrelated crate memory files.

## Verification

```sh
cargo test -p turbospark-bench
```

The ignored real-install targets are expensive and require the model-specific
environment described in the linked benchmark and gate pages. Run only the
target for the behavior being changed, then record durable rows in the
canonical docs and catalog, not in this file.
