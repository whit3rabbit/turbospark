# turbospark-lab (tslab)

The evaluator for kernel and method experiments: phase profiles, gated
interleaved A/B with a fixed verdict rule, shape-derived kernel benches,
Core ML placement, and the autoresearch loop's `run-experiment`.

## Read first

- [program.md](program.md): the loop's instructions and the spec format.
- [Benchmarking protocol](../../docs/BENCHMARKING.md) and the bench crate's
  [AGENTS.md](../bench/AGENTS.md): the measurement rules this crate encodes.

## Directory map

- `src/bin/tslab.rs`, `src/cli.rs`: the orchestrator's commands.
- `src/bin/tslab_worker.rs`, `src/worker.rs`: one arm, one run, one record.
- `src/orchestrate.rs`: fresh-process runs, rotation, the paired A/B.
- `src/stats.rs`, `src/gates.rs`: the verdict rule and the quality and
  memory bounds. Both are fixed in code on purpose.
- `src/host.rs`, `src/lock.rs`: quiet-host rules and the machine lock.
- `src/knobs.rs`: the runtime switch table (`tests/knob_registry.rs`).
- `src/experiment.rs`, `src/integrity.rs`, `src/spec.rs`: gate,
  run-experiment, confirm, and what an experiment may not touch.
- `src/kernel/`: microbenches. `src/ane.rs`: compute-plan placement.
- `src/tree_replay.rs`: the tree-verify acceptance replay. Its lattice
  recorders and replay math live in `crates/bench` (`tree_replay*.rs`), shared
  with `kflash_tree_replay_probe.rs`.

## Rules

- The orchestrator decides and the worker only measures. Keep every judgment
  (eligibility, verdict, gates) out of `worker.rs`, which links runtime code
  an experiment may edit.
- Every run is a fresh worker process with a cleared environment plus the
  arm's variables. Do not add an in-process arm switch: half the runtime
  knobs are cached in `OnceLock`s.
- The verdict rule, gate bounds and integrity lists only get stricter. A
  change that loosens one needs its own evidence and a human decision, and
  `run-experiment` refuses to grade a tree that changes this crate.
- Dirty runs are ledgered, never dropped. The ledger is append-only.
- No benchmark numbers in this file or any AGENTS.md: results live in the
  ledger, readings in the topic docs, frozen rows in `docs/BENCHMARKS.md`.
- New bench kinds take their shapes from `ArchConfig`, not constants, and
  keep the distinct-buffer, alternated-arm, discarded-first-rep discipline
  in `src/kernel/support.rs`.

## Verification

```sh
cargo test -p turbospark-lab
cargo clippy -p turbospark-lab --tests
```

`tests/knob_registry.rs` reddens when a runtime switch is renamed, and
`tests/phases_format.rs` when the CLI's phase table drifts from
`src/phases.rs`.
