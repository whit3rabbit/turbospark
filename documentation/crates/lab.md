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
| `tslab serve`, the read-only dashboard over every ledger ([guide](../advanced/tslab-serve.md)) | `src/dash/` (`index.rs`, `pages.rs`, `api.rs`) |
| The agent skill, its doctor and preflight scripts, and evals ([guide](../advanced/tslab-skill.md)) | `skill/` |

### What it measures

The crate measures at three levels. The user-facing tables (every metric,
every bench, the verdict rule) are on
[TurboSpark Lab](../advanced/tslab.md#what-tslab-measures). This section is
where each lives in code.

| Level | What gets measured | Recorded as | Code |
| --- | --- | --- | --- |
| Whole-model run | Prefill and decode tok/s, prefill seconds, peak `phys_footprint`, follow-up TTFT, audio wall time and RTF, modeled prefill and decode joules, phase buckets, engaged lanes, host samples | `run` (plus `runs/<run_id>.json`), `profile_summary` | `src/worker.rs`, `src/worker_audio.rs`, `src/phases.rs`, `src/energy.rs`, `src/host.rs` |
| Paired comparison | Median effect, bootstrap 95% interval, clean pair count, null floor | `null_calibration`, `verdict`, `sweep` | `src/orchestrate.rs`, `src/sweep.rs`, `src/stats.rs` |
| Kernel | GPU time per case over M rows or context, optional output-bit hashes, modeled energy per case | `kernel_curve` | `src/kernel/` |
| Offline | Projected effect of a kernel mode, its miss against the verdict, speculation depth and tree acceptance from captures | `projection`, `calibration`, `depth_replay`, `tree_replay`, `hybrid_replay` | `src/projection.rs`, `src/depth_replay.rs`, `src/tree_replay.rs`, `src/hybrid_replay.rs` |

`Metric` in `src/workload.rs` is the one list of judgeable metrics, with each
metric's direction and the record field it reads. A new metric goes there.

**Kernel benches** implement `KernelBench` (`src/kernel/mod.rs`): `name`, a
one-line `describe` for `tslab kernel --list`, `cases` derived from the
install's own shapes (an `Err` is a coverage answer such as "no MLA kernel",
not a failure), and `run`. `registry()` is the order `--bench all` runs them.
`ane_matmul` and `ane_handoff` compile only with the `ane-prefill` feature.

**Knobs** (`src/knobs.rs`) classify every runtime switch an arm can set, and
`tslab knobs` prints the table. Two columns decide how an A/B on it is read:

| Field | Values | Meaning |
| --- | --- | --- |
| `read_at` | `open` | Captured into the runner when the model opens |
| | `process_cached` | A process-wide `OnceLock`. Changing it needs a new process, which is why every arm is a fresh worker |
| | `per_call` | Read on every call (diagnostics) |
| `exactness` | `bit_exact` | Output bits identical across values. The quality gate checks digests |
| | `kl_bounded` | Output moves within a distribution bound. The gate runs the KL probe |
| | `not_comparable` | Changes what is computed or measured. Compare within one value only |

A knob marked unquotable (profilers, probes, stand-ins, dumps) makes every run
that sets it ineligible for a verdict. Each knob also names the
`[engaged] NAME` lines that prove its lane ran, and the worker refuses a run
whose stderr contradicts its arm.

### Environment

| Variable | Read by | Purpose |
| --- | --- | --- |
| `TURBOSPARK_LAB_DIR` | `src/paths.rs` | Results root. Wins over `TURBOSPARK_HOME` |
| `TURBOSPARK_HOME` | `src/paths.rs` | Fallback root (`$TURBOSPARK_HOME/lab`), and the model store for aliases |
| `CARGO_TARGET_DIR` | `src/experiment.rs`, `skill/scripts/doctor.sh` | Where `run-experiment` and the doctor find the built `tslab-worker` |
| `TURBOSPARK_ENGAGEMENT` | set by `src/orchestrate.rs` | Always `1` on a worker, so engagement lines print |
| `TSLAB_ANE_MODEL` | `src/kernel/ane_handoff.rs` (ignored test) | The dense Qwen install the real-program `ane_handoff` test runs on |
| any `TURBOSPARK_*` | `src/provenance.rs` | Copied into every record's provenance when set on the tslab process |

A worker starts from a cleared environment plus `PATH`, `HOME`, `TMPDIR`,
`USER`, `LANG`, `LC_ALL`, `TURBOSPARK_HOME`, `TURBOSPARK_LAB_DIR`,
`TURBOSPARK_ENGAGEMENT=1`, and the arm's own variables (`PASSTHROUGH`,
`PASSTHROUGH_TURBOSPARK`, and `ORCHESTRATOR_ENV` in `src/orchestrate.rs`).
A knob exported in your shell does not leak into an arm.

### Examples

Build, and confirm the checkout is usable:

```sh
cargo build --release -p turbospark-lab
crates/lab/skill/scripts/doctor.sh
```

Build with the ANE benches and the ANE+Steel prefill split available as arms:

```sh
cargo build --release -p turbospark-lab --features ane-prefill
target/release/tslab kernel --model qwen38-27b --bench ane_handoff --contend 4 --experiment ane-handoff
```

Run the real-program `ane_handoff` test (ignored by default):

```sh
TSLAB_ANE_MODEL=~/models/qwen38-27b.gturbo \
  cargo test -p turbospark-lab --features ane-prefill --release -- --ignored runs_both_arms_on_the_real_programs
```

Keep an experiment's results apart from your main lab dir:

```sh
TURBOSPARK_LAB_DIR=~/lab-scratch target/release/tslab profile --model lfm3-1.7b --decode 64 --experiment scratch
target/release/tslab serve --lab ~/lab-scratch
```

Read one experiment's verdicts as JSON lines:

```sh
target/release/tslab ledger --experiment lfm3-ffn --kind verdict
```

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
- [The tslab dashboard](../advanced/tslab-serve.md)
- [The tslab agent skill](../advanced/tslab-skill.md)
- [bench](bench.md)
- [runtime](runtime.md)
