# turbospark-lab (tslab)

The evaluator for kernel and method experiments: phase profiles, gated
interleaved A/B with a fixed verdict rule, shape-derived kernel benches,
traces, Core ML placement, resumable workflows, and the autoresearch
loop's `run-experiment`. Text and audio installs of every family.

## Read first

- [program.md](program.md): the loop's instructions and the spec format.
- [skill/SKILL.md](skill/SKILL.md): the `tslab` skill, how an agent drives
  tslab (the `--json` contract, workflows, what to quote), with its
  references in `skill/references/`. One skill serves a checkout
  (`scripts/install-tslab-skill.sh --repo` links it into `.claude/skills/`
  and `.agents/skills/`) and the zip `scripts/package-tslab-skill.sh` builds
  with release binaries and workloads; `skill/scripts/doctor.sh` finds
  whichever binary applies.
  A change to a command, flag or record updates the matching reference.
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
- `src/family_profile.rs`, `families/`: one profile per text family and
  audio kind (`tslab family`): the settings known to pay, the sweeps to run
  first, the findings with their ledger id or doc heading, open questions,
  traps. Embedded and validated like the pinned set; `families/README.md`
  holds the field rules, and `tests/family_profiles.rs` holds every
  `ModelFamily` and `AudioKind` to a file.
- `src/dash/`: `tslab serve`, the read-only dashboard (`index.rs` reads every
  ledger and keeps it by `(mtime, len)`; `pages.rs` renders the experiment
  table, per-experiment, compare, verdict, sweep, family and record pages
  from the viz cards; `api.rs` the JSON routes). It never takes the machine
  lock and writes nothing. A loopback bind answers only loopback Host
  headers and every response is `nosniff` plus a `default-src 'none'` CSP;
  ledger-sourced strings are escaped or validated before they reach a page
  or a path, and the tests hold all four guards.
- `src/sweep.rs`: `tslab sweep`, several candidates against one baseline in
  one interleaved session, each judged by `orchestrate::run_sweep` under the
  same pair rule as `run_ab`; one `verdict` record per candidate and one
  `sweep` record.
- `src/workflow.rs`, `workflows/*.json`: `tslab workflow` and the built-in
  plans.
- `src/trace.rs`, `tools/trace_tables.py`: `tslab trace` (L5).
- `src/round_cost.rs`: measured speculative round costs (L7).
- `src/projection.rs`: `tslab project` and `tslab calibrate` (L8): a kernel
  mode priced from curves, counts and a measured base, then joined to the
  verdict that followed. Contract in
  [KERNEL_MODES.md](../../docs/KERNEL_MODES.md).
- `src/bin/tslab_worker.rs`, `src/worker.rs`, `src/worker_audio.rs`: one
  arm, one run, one record (text or audio).
- `src/orchestrate.rs`: fresh-process runs, rotation, the paired A/B.
- `src/stats.rs`, `src/gates.rs`: the verdict rule and the quality and
  memory bounds. Both are fixed in code on purpose.
- `src/ledger.rs`, `src/paths.rs`: the append-only JSON-lines ledger every
  record lands in, and the lab dir (`$TURBOSPARK_LAB_DIR`, else
  `$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab`).
- `src/host.rs`, `src/lock.rs`: quiet-host rules and the machine lock.
- `src/knobs.rs`: the runtime switch table and `LANES`, the knobs that are
  kernel modes with the bench, arms and parity tests that hold them
  (`tests/knob_registry.rs`).
- `src/experiment.rs`, `src/integrity.rs`, `src/spec.rs`: gate,
  run-experiment, confirm, and what an experiment may not touch.
- `src/workload.rs`: prompts, arms, metrics, slots, residency, and
  speculation: the vocabulary the CLI, spec, and worker share.
- `src/kernel/`: microbenches behind `KernelBench` (`kernel/mod.rs` holds
  the registry). `src/ane.rs`: compute-plan placement. `kernel/ane_matmul.rs`
  (feature `ane-prefill`): Steel against the Neural Engine per shape, with
  the row split, placement and joules, over programs from
  `crates/coreml/tools/gen_ane_sweep_programs.py` (random baked weights) or
  `bake_ane_prefill_programs.py` (an install's own weights, `real:` cases).
  Baked-INT4 programs did not beat the F16 split end to end; the findings
  and the reasons are in `docs/SPEEDUP_PLAYBOOK.md`. `kernel/ane_handoff.rs`
  (feature `ane-prefill`): one split point's host handoff, `wait` against
  `poll`, quiet and under `--contend N`.
- `src/energy.rs`: the IOReport energy meter (modeled, whole machine). The
  worker, the kernel command and the ANE bench read it; `report.rs` prints
  it; it is never a wall-power claim.
- `src/census.rs`: `tslab expert-census`, one router capture run reduced to
  expert reuse within batches, union against slot counts, and the routed
  sub-batch replay. `src/phases.rs` also classifies a MoE stretch
  (`moe_bound`): I/O, host loop, or compute bound.
- `src/pinned.rs`, `pinned/`: the pinned kernel references, one JSON per
  kernel from `nace-ai/kapitsa`, `jundot/omlx`, and `ddalcu/mlx-serve`,
  each with the repo revision it was read at, the models it serves, and its
  hardcoded settings traceable to `path:line`. The reader validates every
  file against `pinned/README.md`, embeds the set into the binary, and
  `tslab pinned` lists or inspects it; a sweep loads a tweaked copy with
  `--dir`. `tests` in `src/pinned.rs` fail when a file drifts from the
  embedded copy or is added without registering it.
- `src/tree_replay.rs`: the tree-verify acceptance replay. Its lattice
  recorders and replay math live in `crates/bench` (`tree_replay*.rs`), shared
  with `kflash_tree_replay_probe.rs`.
- `src/depth_replay.rs`: offline tuning for adaptive speculation depth.
  `chain-capture` records an MTP chain's per-position match length once;
  `depth-replay` walks the runtime's own controller over that capture and a
  round-cost table.

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
  ledger, findings with provenance in `families/<family>.json`, readings in
  the topic docs, frozen rows in `docs/BENCHMARKS.md`. A profile finding
  from a doc table is a `reading`, never a verdict.
- A new bench implements `KernelBench`, returns a `BenchResult`, and is
  added to `kernel::registry()`. It takes its shapes from the install (the
  resident index via `matmul_shapes`, `ArchConfig`, or an audio
  `config.json`), never constants; a fixed corpus says so in its doc. It
  keeps the distinct-buffer, alternated-arm, discarded-first-rep discipline
  in `src/kernel/support.rs`, records `out_fnv` under `--hash`, and its
  `cases` returns `Err(reason)` for an install it cannot serve, which
  `tslab capabilities` reports.
- Commands print through `say!`, never `println!`: under `--json` stdout is
  exactly one envelope (`tests/schema_contract.rs`).
- The dashboard stays read-only and lock-free: no POST route, no write under
  the lab dir, no `MachineLock`. A page fetches nothing and its one script
  is inline and optional (`DASH_JS` in `dash/pages.rs` adds hover tooltips,
  legend toggles and the compare tray, and `tests/dash_pages.rs` asserts no
  `src` and no URL), so a saved page reads the same as a live one. A new
  ledger kind goes
  in `schema.rs` `KINDS` and the schema document in the same change.
- Changing a field's meaning in the ledger or the envelope is a new schema
  version, not an edit: dashboards and old ledgers outlive any build.
  `series.rs` keeps reading version 0 records.
- A runtime switch that outlives one experiment gets a knob row, and an
  `engagement` entry when its lane prints `foundation::engaged!`. A run
  whose stderr contradicts its arm's switches is ineligible.
- A lane's bench arm calls the same function the runtime calls (the gpu
  crate's routing, e.g. `gpu::encode_dequant_int4_gemm_small_n`), never a
  copy of its thresholds, so the bench prices what ships. A kernel bench's
  `exactness` hint is read off the bench's synthetic weights: it can say
  bit-equal where the real install differs, so the lane's class comes from
  the install's KL or digest gate, not from the hint.
- Price a runtime route twice before quoting it: the kernel-priced part
  (per-projection bench times by count) and the observed model delta. When
  the observed delta is larger than the priced part, say so and attribute the
  rest with a per-op-class profile before it goes in a doc or a profile
  finding. A prefill A/B runs both arms in one binary through the knob's
  kill switch, alternates direction, discards a warmup round, and counts only
  runs the overlap watcher left clean: a quiet host at the start does not
  keep another session from starting a build mid-run.
- A workflow step runs a command through `cli::dispatch` and nothing
  else: no logic in plans that a command typed alone would not do.
- Joules in a record are IOReport's model of the whole machine
  (`energy_model: ioreport_whole_machine`), comparable within one
  experiment only. They never go beside `docs/POWER_BASELINE.md`
  (powermetrics) and never into `docs/BENCHMARKS.md`. A `~hog:N` run and a
  router-capture run are unquotable by construction.

## Verification

```sh
cargo test -p turbospark-lab
cargo clippy -p turbospark-lab --tests
```

`tests/knob_registry.rs` reddens when a runtime switch is renamed or an
engagement line stops being printed, `tests/family_profiles.rs` when a
family or audio kind has no profile, a knob row names a family the runtime
does not write, or a finding's doc heading is gone, `tests/phases_format.rs` when the CLI's
phase table drifts from `src/phases.rs`, `tests/schema_contract.rs` when a
ledger kind is missing from the contract or `--json` prints more than one
line, `tests/dash_pages.rs` when a dashboard page stops being
self-contained or the index miscounts a ledger, and `tests/family_coverage.rs` when a family loses its coverage.
