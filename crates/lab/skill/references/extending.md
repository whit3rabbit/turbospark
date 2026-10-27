# Extending tslab (repository only)

Read `crates/lab/AGENTS.md` first: it maps the modules and holds the rules.
The evaluator only gets stricter, and `run-experiment` refuses to grade a tree
that changes `crates/lab`, so lab changes are their own PRs, never part of an
experiment.

## A new kernel bench

1. Add `crates/lab/src/kernel/<name>.rs` with a type implementing
   `KernelBench` (`kernel/mod.rs`):
   - `cases(subject, opts)` returns the cases for this install, or
     `Err(reason)` when it cannot serve it. That reason is what `tslab
     capabilities` shows.
   - `run(...)` returns a `BenchResult` (`src/bench.rs`).
2. Take shapes from the install: `kernel::matmul_shapes(subject, dtype,
   fallback)` for weights in the resident index, `ArchConfig` for the rest,
   or `subject.audio_config` for audio. Never constants. A fixed corpus says
   so in its module doc.
3. Keep the discipline in `kernel/support.rs`: distinct weight copies per
   sample, arms alternated within each repetition, the first repetition
   discarded, GPU time from the command buffer. Under `opts.hash`, set each
   point's `out_fnv` from the output bytes.
4. Register it in `kernel::registry()`. The ledger, `--json`, `tslab export`
   and `tslab report` then show it with no other code.
5. Unit-test its case generation against a synthetic `ArchConfig` or config.
   Add a one-rep Metal run if it is cheap.

## A runtime switch the lab should know

1. Read the variable once through a `OnceLock` at its use site, default off.
2. Where the new lane actually runs, add `foundation::engaged!("<name>")`.
3. Add a row to `KNOBS` in `src/knobs.rs`: env, values (unset first), when it
   is read, exactness, families, quotable, the doc that describes it, and
   `engagement: &[on("*", "<name>")]` (or `off("0", "<name>")` for a kill
   switch of a default-on lane).
4. Mention the variable in that doc (`docs/ENV.md` for most).

`tests/knob_registry.rs` fails if the env name or the engagement call goes
missing from `crates/*/src`, or the doc stops mentioning it.

## A knob that is a kernel mode

A switch that selects an alternative kernel lane (`docs/KERNEL_MODES.md`) gets
a `Lane` in `knobs::LANES` as well as its `KNOBS` row, so `tslab project
--knob NAME` can price it and `tests/knob_registry.rs` can hold it to the tree:

1. The bench that times the lane is registered, and its arms have a fixed
   label list in `kernel::arm_labels`. A bench that builds arms dynamically
   gets a `pub const ARMS` and uses it where it labels them.
2. `compare` is `Arms { baseline, lane }` when both arms run in one
   experiment, or `Process { arm, baseline_env }` when the gpu crate reads the
   switch once per process and the bench calls the kernels directly: time the
   same arm in two `tslab kernel` processes and give `project`
   `--default-from` and `--mode-from`.
3. `parity` lists the test files that hold the lane to the default's bits or
   to its stated bound. A lane with none cannot claim an exactness class.
4. The bench fills `Point.row_fnv` (the first output row alone) under
   `--hash`, so the exactness hint can say whether a row depends on its batch.

## A family finding, recommendation or sweep

Profiles live in `crates/lab/families/<family>.json` (`README.md` there has
the field rules) and are embedded by `src/family_profile.rs`:

1. A finding: copy the verdict record's experiment and id (or the doc path
   and heading for a reading), the install, machine, power, revision,
   workload and conditions, and the numbers. Its id is
   `<family>.<slug>-<YYYY-MM-DD>`.
2. A recommendation: one of `knob` plus `value` (a `KNOBS` row), a
   `speculation`, `slots`/`residency`, or `chunk`, with `why` and the
   finding ids that back it. `when` holds coverage tags.
3. A sweep: a baseline arm and candidate arms in `tslab` arm syntax, the
   metric and the exactness class, and a workload when the default is wrong.

`cargo test -p turbospark-lab` validates every file (`family_profile.rs`
unit tests, `tests/family_profiles.rs`): a knob the table lacks, a tag
coverage never produces, a finding without conditions, a doc heading that
does not exist, or a verdict on a doc reading all fail there. A new
`ModelFamily` variant or `AudioKind` needs a new file.

## A new ledger kind

Append it through `Ledger::append`, add it to `KINDS` in `src/schema.rs` and
to the `kind` enum in `schema/tslab-1.schema.json` in the same change. If it
should plot, add a series adapter in `src/series.rs`.
`tests/schema_contract.rs` fails on a kind missing from either list.

## A new command

Add it to `cli::dispatch` and `cli::COMMANDS`, print through `say!` (never
`println!`), and return a `CmdOutput` whose `result` is useful to an agent.
`tests/schema_contract.rs` checks that `--json` prints exactly one envelope.

## A new built-in workflow

Add `crates/lab/workflows/<name>.json` and list it in `workflow::BUILTIN`.
Steps may only call commands in `cli::COMMANDS`, and the workflow unit tests
parse and validate every built-in.

## Checks before a lab PR

```sh
cargo test -p turbospark-lab
cargo clippy -p turbospark-lab --tests
cargo fmt --all --check
```

`cargo fmt --all --check` writes nothing and covers the runtime too: CI's
macOS job runs it first and stops there on any diff, skipping its build and
tests. Format only the files you changed (`rustfmt --edition 2021 <file>`),
since other sessions share this tree.
