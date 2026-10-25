# turbospark-lab (tslab)

The evaluator for kernel and method experiments: phase profiles, gated
interleaved A/B with a fixed verdict rule, shape-derived kernel benches,
traces, Core ML placement, resumable workflows, and the autoresearch
loop's `run-experiment`. Text and audio installs of every family.

## Read first

- [program.md](program.md): the loop's instructions and the spec format.
- [skill/SKILL.md](skill/SKILL.md): the `turbospark-lab` skill, how an
  agent drives tslab (the `--json` contract, workflows, what to quote), with
  its references in `skill/references/`. `scripts/install-tslab-skill.sh
  --repo` links it into `.claude/skills/`. `skill/portable/` is the
  binary-only `tslab` skill that `scripts/package-tslab-skill.sh` zips with
  release binaries and workloads. A change to a command, flag or record
  updates the matching reference.
- [Benchmarking protocol](../../docs/BENCHMARKING.md) and the bench crate's
  [AGENTS.md](../bench/AGENTS.md): the measurement rules this crate encodes.

## Directory map

- `src/bin/tslab.rs`, `src/cli.rs`: the orchestrator's commands and
  `dispatch`, the one entry every command and workflow step goes through.
- `src/schema.rs`, `src/out.rs`, `schema/tslab-1.schema.json`: the output
  contract (`--json` envelopes, the `schema` stamp, every ledger kind).
- `src/bench.rs`: `BenchResult`, the one shape every kernel bench emits.
- `src/series.rs`, `src/viz.rs`, `src/export.rs`: ledgers as plottable
  series, the offline HTML report, `tslab export` and `tslab report`.
- `src/families.rs`, `src/inventory.rs`, `src/audio_kind.rs`: per-install
  coverage (`tslab capabilities`), the resident-index weight inventory, and
  which audio pipeline an install is.
- `src/workflow.rs`, `workflows/*.json`: `tslab workflow` and the built-in
  plans.
- `src/trace.rs`, `tools/trace_tables.py`: `tslab trace` (L5).
- `src/round_cost.rs`: measured speculative round costs (L7).
- `src/bin/tslab_worker.rs`, `src/worker.rs`, `src/worker_audio.rs`: one
  arm, one run, one record (text or audio).
- `src/orchestrate.rs`: fresh-process runs, rotation, the paired A/B.
- `src/stats.rs`, `src/gates.rs`: the verdict rule and the quality and
  memory bounds. Both are fixed in code on purpose.
- `src/host.rs`, `src/lock.rs`: quiet-host rules and the machine lock.
- `src/knobs.rs`: the runtime switch table (`tests/knob_registry.rs`).
- `src/experiment.rs`, `src/integrity.rs`, `src/spec.rs`: gate,
  run-experiment, confirm, and what an experiment may not touch.
- `src/kernel/`: microbenches behind `KernelBench` (`kernel/mod.rs` holds
  the registry). `src/ane.rs`: compute-plan placement.
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
- A new bench implements `KernelBench`, returns a `BenchResult`, and is
  added to `kernel::registry()`. It takes its shapes from the install (the
  resident index via `matmul_shapes`, `ArchConfig`, or an audio
  `config.json`), never constants; a fixed corpus says so in its doc. It
  keeps the distinct-buffer, alternated-arm, discarded-first-rep discipline
  in `src/kernel/support.rs`, records `out_fnv` under `--hash`, and its
  `cases` returns `Err(reason)` for an install it cannot serve, which
  `tslab capabilities` reports.
- Commands print through `say!`, never `println!`: under `--json` stdout is
  exactly one envelope (`tests/schema_contract.rs`). A new ledger kind goes
  in `schema.rs` `KINDS` and the schema document in the same change.
- Changing a field's meaning in the ledger or the envelope is a new schema
  version, not an edit: dashboards and old ledgers outlive any build.
  `series.rs` keeps reading version 0 records.
- A runtime switch that outlives one experiment gets a knob row, and an
  `engagement` entry when its lane prints `foundation::engaged!`. A run
  whose stderr contradicts its arm's switches is ineligible.
- A workflow step runs a command through `cli::dispatch` and nothing
  else: no logic in plans that a command typed alone would not do.

## Verification

```sh
cargo test -p turbospark-lab
cargo clippy -p turbospark-lab --tests
```

`tests/knob_registry.rs` reddens when a runtime switch is renamed or an
engagement line stops being printed, `tests/phases_format.rs` when the CLI's
phase table drifts from `src/phases.rs`, `tests/schema_contract.rs` when a
ledger kind is missing from the contract or `--json` prints more than one
line, and `tests/family_coverage.rs` when a family loses its coverage.
