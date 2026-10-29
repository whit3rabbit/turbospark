---
description: "turbospark-lab (tslab): the evaluator for kernel and method experiments, with phase profiles, gated interleaved A/B verdicts, shape-derived kernel benches, sweeps, pricing, replays, resumable workflows, and the read-only results dashboard."
icon: flask
---

# lab

`crates/lab` (package `turbospark-lab`) is the evaluator for kernel and method
experiments: phase profiles, gated interleaved A/B with a fixed verdict rule,
shape-derived kernel benches, sweeps, kernel-mode pricing, speculation
replays, traces, Core ML placement, resumable workflows, and the
autoresearch loop's `run-experiment`. It ships two binaries, `tslab` (the
orchestrator) and `tslab-worker` (one arm, one run, one record), and is a dev
harness (`publish = false`) that measures local `.gturbo` installs on macOS,
like [bench](bench.md). Every result lands in an append-only ledger under the
lab dir; frozen results never live in this page. The user-facing guide is
[TurboSpark Lab (tslab)](../advanced/tslab.md), and the working rules live in
`crates/lab/AGENTS.md` and `crates/lab/program.md`.

## What it does

| Responsibility | Where |
| --- | --- |
| Binary entry points and `dispatch`, the one entry every command and workflow step goes through | `src/bin/tslab.rs`, `src/cli.rs` |
| Output contract: `--json` envelopes, the `tslab/1` schema stamp, every ledger kind | `src/schema.rs`, `src/out.rs`, `schema/tslab-1.schema.json` |
| Fresh-process runs, arm rotation, the paired A/B, the sweep judging | `src/orchestrate.rs`, `src/sweep.rs` |
| The verdict rule and the quality and memory bounds (fixed in code on purpose) | `src/stats.rs`, `src/gates.rs` |
| One arm per worker process, text and audio | `src/bin/tslab_worker.rs`, `src/worker.rs`, `src/worker_audio.rs` |
| Prompts, arms, metrics, slots, residency, and speculation vocabulary shared by CLI, spec, and worker | `src/workload.rs` |
| Microbenches behind `KernelBench`; ANE compute-plan placement | `src/kernel/`, `src/ane.rs` |
| Kernel-mode pricing (`project`) and its calibration against the verdict that followed (`calibrate`) | `src/projection.rs` |
| Tree-verify acceptance replay and adaptive speculation depth tuning | `src/tree_replay.rs`, `src/depth_replay.rs` |
| Metal System Trace of an arm's measured work | `src/trace.rs`, `tools/trace_tables.py` |
| Per-install coverage, the resident-index weight inventory, and the audio kind | `src/families.rs`, `src/inventory.rs`, `src/audio_kind.rs` |
| Per-family profiles: settings that pay, sweeps to run first, findings with provenance | `src/family_profile.rs`, `families/` |
| Pinned kernel settings from the compared engines, validated and embedded | `src/pinned.rs`, `pinned/` |
| The runtime switch table (`knobs`) and the kernel-mode lanes | `src/knobs.rs` |
| Gate, `run-experiment`, `confirm`, and the integrity lists an experiment may not touch | `src/experiment.rs`, `src/integrity.rs`, `src/spec.rs` |
| Resumable workflows and the built-in plans | `src/workflow.rs`, `workflows/` |
| Quiet-host rules and the one machine lock | `src/host.rs`, `src/lock.rs` |
| The append-only ledger and the lab dir | `src/ledger.rs`, `src/paths.rs` |
| Ledgers as plottable series, the offline HTML report, `tslab export` | `src/series.rs`, `src/viz.rs`, `src/export.rs` |
| `tslab serve`, the read-only dashboard over every ledger | `src/dash/` (`index.rs`, `pages.rs`, `api.rs`) |

### The `--json` output contract

Any command accepts `--json`, anywhere on the line. Stdout is then exactly
one `tslab/1` envelope `{schema, command, ok, exit_code, decision?, records,
result, warnings, error?}`; tables and progress go to stderr. `decision` is
present only for `run-experiment` and `confirm` (keep, discard,
inconclusive), and `records` names the ledger lines the invocation appended,
so an agent can cite results by ledger id without parsing tables. The JSON
Schema of everything tslab writes is embedded from
`schema/tslab-1.schema.json` and printed by `tslab schema`. Changing a
field's meaning is a new schema version, not an edit, because dashboards and
old ledgers outlive any build.

### The ledger

Each experiment owns a directory under the lab dir
(`$TURBOSPARK_LAB_DIR`, else `$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab`)
holding an append-only JSON-lines ledger (`ledger.jsonl`) and the artifacts
of its runs (`runs/<run_id>.json` records, request files, worker stderr, and
optional logits dumps). Every record gets a time-ordered id of the form
`<unix_ms>-<pid>-<n>`, and every kind (`run`, `verdict`, `sweep`,
`kernel_curve`, `projection`, `calibration`, and the rest) is listed in
`KINDS` in `src/schema.rs` alongside the schema document. Lines are never
rewritten: a correction is a new line that names the one it supersedes, and
dirty runs are ledgered, never dropped, so a noisy host stays visible. Each
line is appended under an exclusive `flock` and fsynced, so an orchestrator
and a `tslab report` in another shell cannot interleave bytes. The machine
lock is a separate `flock` on `<lab>/.machine.lock`; `tslab serve` never
takes it.

### Where it sits

The portable half (stats, ledger, spec, knob table, schema) builds on every
platform so its tests run anywhere; everything that opens a model or a Metal
device is macOS-only, like `crates/bench`. macOS dependencies are the runtime
and its support crates (`turbospark-runtime`, `turbospark-model-io`,
`turbospark-repack`, `turbospark-selection`, `turbospark-tokenizer`,
`turbospark-catalog`, `turbospark-gpu`, `turbospark-coreml`,
`turbospark-bench`, `turbospark-audio`) behind `turbospark-core` (aliased
`foundation`). The dashboard reuses the same `axum` and `tokio` versions the
server crate pins, so no new package enters the lockfile. The `ane-prefill`
cargo feature exists so the ANE+Steel prefill split can be an A/B arm.

### Tests

```sh
cargo test -p turbospark-lab
cargo clippy -p turbospark-lab --tests
```

The contract tests keep the crate honest against the tree:
`tests/knob_registry.rs` fails when a runtime switch is renamed or an
engagement line stops printing, `tests/family_profiles.rs` when a family or
audio kind has no profile or a finding loses its provenance,
`tests/schema_contract.rs` when a ledger kind is missing from the contract or
`--json` prints more than one line, `tests/dash_pages.rs` when a dashboard
page stops being self-contained, and `tests/phases_format.rs` when the CLI's
phase table drifts from `src/phases.rs`.

## Related pages

- [TurboSpark Lab (tslab)](../advanced/tslab.md)
- [bench](bench.md)
- [runtime](runtime.md)
