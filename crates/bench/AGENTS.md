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

- `src/main.rs`, `src/startup_probe_main.rs`, `src/meeting_eval_main.rs`: the
  `turbospark-bench`, `turbospark-startup-probe`, and `meeting_eval` binaries.
- `src/real_model.rs`: shared real-install protocol and model setup.
- `src/real_model_params.rs`: family-specific context and budget parameters.
- `src/lib.rs`: measurement math shared with `crates/lab` (`tslab`):
  `logit_compare` (KL and top-k next-token comparison), `quality_corpus`
  (prefill arm probe corpus slices), `tree_replay` and `tree_replay_record`
  (host-only tree versus chain speculative replay), and `meeting` (Frozen E
  cpWER scoring behind `meeting_eval`).
- `tests/`: offline fixtures plus ignored real-install gates and probes.
- `prompts/`: frozen protocol and quality reference prompt fixtures.
- `testdata/frozen_e/` and `scripts/`: the committed Frozen E references and
  frozen counts, the fixture rebuild helpers, and the reference Python scorer.
- Repository-root `scripts/` also holds measurement helpers this crate does
  not own (`parity.sh`, `power.sh`, and `bench_api.py` with
  `bench_api_drive.sh` for decode tok/s over an OpenAI-compatible endpoint).

## Rules

- Never quote a number without its model family, artifact, context window,
  generation budget, chip, power source, and warmup conditions.
- Run formal measurements on AC power with a quiet host. Interleave A/B arms
  instead of running all of one arm before the other.
- A peak footprint is not an on-disk model size. MoE expert tables stream, while
  resident weights, slots, and KV cache contribute to the process footprint.
- CPU, synthetic, projected, or host-contaminated results do not establish a
  real Metal, quality, or memory claim.
- The prefill arm probes and `tslab` measure their bounds through the shared
  math in `src/lib.rs` (`logit_compare`, `quality_corpus`). Do not fork private
  copies: a bound measured by one must be the bound measured by the other.
- Probes report, gates assert. An ignored probe needs a real install and
  prints its table for a reader; only a gate holds a frozen count, digest, or
  ceiling, and a gate that can fail on machine state is worse than none.
- The `meeting` scorer must reproduce the frozen Frozen E counts exactly, and
  agree count for count with `scripts/frozen_e_score.py`, before any native
  result is compared against them.
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
