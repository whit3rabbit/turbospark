# turbospark-lab

`tslab`, the evaluator for kernel and method experiments: phase profiles,
gated interleaved A/B with a fixed verdict rule, shape-derived kernel
benches, sweeps, GPU traces, Core ML (ANE) placement planning, resumable
workflows, and the autoresearch loop's `run-experiment`. It measures text
and audio installs of every model family. A macOS Apple Silicon tool: the
`tslab` orchestrator decides, and the `tslab-worker` process it spawns for
every measured run only measures. `publish = false`: a dev harness over
local installs on this machine.

## Build and quick start

```sh
cargo build --release -p turbospark-lab
```

Both binaries land in `target/release/`; keep `tslab-worker` beside `tslab`.
`profile`, `ab`, `kernel`, and the rest of the measured commands find the
worker next to the orchestrator binary and stop without it. `M` below is an
install directory (`*.gturbo`) or a catalog alias.

```sh
# What tslab can measure on this install, with coverage tags and what to set first
target/release/tslab capabilities --model ~/models/qwen36.gturbo

# One knob, baseline against candidate, interleaved pairs, a verdict record
target/release/tslab ab --model ~/models/qwen36.gturbo \
  --prompt protocol:medium-review --decode 128 \
  --baseline off --candidate mma8:int4_mma8=1 --experiment qwen36-mma8

# The read-only dashboard over every ledger in the lab dir
target/release/tslab serve --open
```

## Commands

`tslab --help` prints the full flag reference. The subcommands:

| Command | What it does |
|---------|--------------|
| `profile` | Per-phase profile of an install; slots/residency arms, the dispatch probe and stand-ins for dense flows. |
| `ab` | One baseline against one candidate, interleaved pairs, judged by the fixed verdict rule. |
| `sweep` | Several candidates against one baseline in one interleaved session, each a paired A/B; profile sweeps come from `tslab family`. |
| `gate` | The quality and memory gates for a spec (loads the model; takes minutes). |
| `run-experiment` | The autoresearch loop entry: diff check, worker build, required tests, gates, paired A/B, `DECISION:`. |
| `confirm` | The held-out `confirm_workload` pass after a `keep_pending_confirm` verdict. |
| `kernel` | Kernel benches derived from the install's own shapes; `--list` shows the registered benches. |
| `hash-diff` | Exit 3 when any hashed bench output changed between two experiments. |
| `project` | Price a kernel mode from bench curves before any model run (no GPU work). |
| `calibrate` | Join a projection to the verdict that followed; `--all` lists every miss and the bias. |
| `tree-replay` | Tree-verify acceptance replay for the MTP and DFlash drafters. |
| `ane-plan` | Core ML compute-plan placement from an `.mlpackage` or manifest directory. |
| `expert-census` | MoE router capture: how experts repeat within batches and the union per batch against slot counts. |
| `chain-capture` | Record an MTP chain's per-position match length once. |
| `depth-replay` | Offline adaptive speculation depth tuning over a capture and a round-cost table. |
| `trace` | Metal System Trace of one arm's measured work; never quotable. |
| `ledger` | Read one experiment's ledger records by kind. |
| `workflow` | Run, list, print, or resume a multi-step plan; `workflows/*.json` holds the built-ins. |
| `capabilities` | What tslab can measure on this install: tags, refusals, and what to set first. |
| `family` | The family profile: recommended settings, sweeps, findings with provenance, open questions. |
| `inventory` | The install's weight shapes by dtype. |
| `pinned` | The kernel settings pinned from the compared engines. |
| `knobs` | The runtime switch table: every knob's values, benches, arms, and parity tests. |
| `export` | Ledgers as `tslab/1` series (`jsonl`, `json`, `csv`). |
| `report` | A self-contained offline HTML report per experiment. |
| `schema` | The JSON Schema of everything tslab writes. |
| `serve` | The read-only dashboard over every ledger. |

## Output contract

- `--json` is global (anywhere on the line): stdout is exactly one
  `tslab/1` envelope `{schema, command, ok, exit_code, decision?, records,
  result, warnings, error?}`. Tables and progress go to stderr.
- Exit codes: 0 ok / keep, 1 error, 2 usage, 3 discard, 4 inconclusive.
  3 and 4 are the `run-experiment` and `confirm` verdicts; 4 means do not
  revert.
- Every measured run appends records to the experiment's append-only
  JSON-lines ledger at `<lab>/<experiment>/ledger.jsonl`. Records are never
  dropped or rewritten; dirty runs land with their reasons. A new ledger
  kind goes in `src/schema.rs` `KINDS` and the schema document
  (`schema/tslab-1.schema.json`) in the same change.

## Where results live

`$TURBOSPARK_LAB_DIR`, else `$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab`
(`src/paths.rs`). Lab-owned environment:

- `TURBOSPARK_LAB_DIR`: the results root.
- `TURBOSPARK_HOME`: fallback root, the `lab` subdirectory of it.
- `CARGO_TARGET_DIR`: where `run-experiment` finds the freshly built
  `tslab-worker` (`<target>/release/tslab-worker`).
- Worker sandbox: every measured run is a fresh worker process with a
  cleared environment plus `PATH`, `HOME`, `TMPDIR`, `USER`, `LANG`,
  `LC_ALL`, `TURBOSPARK_HOME`, and `TURBOSPARK_LAB_DIR`. Each arm sets its
  own `TURBOSPARK_*` variables, and the orchestrator adds
  `TURBOSPARK_ENGAGEMENT=1` so engagement lines print. There is no
  in-process arm switch: half the runtime knobs are cached in `OnceLock`s.
- Every `TURBOSPARK_*` variable set on the tslab process is copied into
  each record's provenance.
- Runtime knob names (`TURBOSPARK_INT4_MMA8`, `TURBOSPARK_SPEC_STATS`, and
  so on) are arm variables; `tslab knobs` is their directory.

## Dashboard

`tslab serve [--port 7878] [--bind 127.0.0.1] [--lab DIR] [--open]` is
read-only over every ledger in the lab dir: no POST route, nothing written
under the lab dir, and the machine lock is never taken, so it can stay open
beside a measurement. The index re-reads a ledger when its size or mtime
moves. A loopback bind refuses a Host header that does not name this
machine. Server-rendered pages:

`/`, `/experiment/:name`, `/compare`, `/verdicts`, `/sweeps`, `/families`,
`/family/:family`, `/record/:name/:id`

and a JSON API of GET-only routes returning `tslab/1` envelopes, filters as
query parameters (`src/dash/api.rs`):

`/api/experiments`, `/api/experiment/:name`, `/api/ledger/:name`,
`/api/series?exp=a,b`, `/api/verdicts`, `/api/sweeps`,
`/api/family/:family`, `/api/record/:name/:id`

## Pointers

- [AGENTS.md](AGENTS.md): the crate's architecture map and rules (worker
  purity, ledger discipline, schema policy, bench registry).
- [program.md](program.md): the autoresearch loop, the spec format, and the
  checks for a new model family.
- [skill/SKILL.md](skill/SKILL.md): the agent-facing skill (the `--json`
  contract, which command answers what), with `skill/scripts/doctor.sh` and
  `preflight.sh`; repo-level `scripts/install-tslab-skill.sh` and
  `scripts/package-tslab-skill.sh` install and package it.
- [families/README.md](families/README.md): field rules for the per-family
  profiles under `families/`.
- [pinned/README.md](pinned/README.md): field rules for the pinned kernel
  references under `pinned/`.
- [Benchmarking protocol](../../docs/BENCHMARKING.md): the measurement
  rules this crate encodes.
- [Kernel modes](../../docs/KERNEL_MODES.md): the contract for a switch
  that selects an alternative kernel path, and what `tslab project` and
  `tslab calibrate` check about it.

## Verification

```sh
cargo test -p turbospark-lab
cargo clippy -p turbospark-lab --tests
```
