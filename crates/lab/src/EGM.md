# turbospark-lab (src)

## Overview
`tslab`, the evaluator behind kernel and method experiments. The orchestrator (`tslab`) decides; `tslab-worker` only measures, one fresh process per run [src/lib.rs:worker]. Results go to an append-only ledger. The portable half (stats, ledger, spec, knobs, gates, schema) builds everywhere; everything touching a model, Metal or `pmset` is `cfg(target_os = "macos")` [src/lib.rs:cli].

## Interface
- Commands: `profile`, `ab`, `sweep`, `gate`, `run-experiment`, `confirm`, `kernel`, `project`, `calibrate`, `tree-replay`, `ane-plan`, `expert-census`, `chain-capture`, `depth-replay`, `hybrid-replay`, `ledger`, `trace`, `workflow`, `capabilities`, `family`, `inventory`, `pinned`, `knobs`, `export`, `report`, `schema`, `serve`, `hash-diff` [src/cli.rs:dispatch]. `cli::main`, `finish`, `dispatch` [src/cli.rs:main].
- Orchestration: `Session`, `Outcome`, `run_ab`, `PASSTHROUGH`, `ORCHESTRATOR_ENV` [src/orchestrate.rs:Session] [src/orchestrate.rs:PASSTHROUGH].
- Worker: `main_entry`, `WorkerRequest`, `RunRecord`, `EXIT_REFUSED` [src/worker.rs:main_entry] [src/worker.rs:EXIT_REFUSED]; audio arm `run` [src/worker_audio.rs:run].
- Stats and gates: `Verdict`, `VerdictReport`, `median`, `mad` [src/stats.rs:VerdictReport]; `GateClass`, `QualityBounds`, `effective_bounds`, `judge_quality` [src/gates.rs:judge_quality]; `fit_affine`, `break_even`, `c_of_m` [src/fit.rs:fit_affine].
- Experiment loop: `ExperimentSpec` [src/spec.rs:ExperimentSpec]; `quality_gate`, `gate_cmd` [src/experiment.rs:gate_cmd]; `violations`, `FORBIDDEN_PREFIXES`, `trusted_exe` [src/integrity.rs:violations].
- Storage and output: `Ledger` [src/ledger.rs:Ledger]; `SCHEMA = "tslab/1"`, `CmdOutput`, `Envelope` [src/schema.rs:SCHEMA]; `write_new` [src/atomic.rs:write_new]; `lab_dir`, `experiment_dir` [src/paths.rs:lab_dir]; `Series` [src/series.rs:Series]; `export_cmd`, `report_cmd` [src/export.rs:export_cmd]; `render` HTML [src/viz.rs:render].
- Host and provenance: `HostMonitor`, `HostSample`, `on_ac_power` [src/host.rs:HostMonitor]; `MachineLock` [src/lock.rs:MachineLock]; `Provenance`, `git_dirty`, `sha256_file` [src/provenance.rs:Provenance]; `EnergyMeter` [src/energy.rs:EnergyMeter].
- Data tables: `Knob`, `Lane` [src/knobs.rs:Knob]; `FamilyFacts`, `text_coverage`, `audio_coverage` [src/families.rs:facts]; `FamilyProfile` [src/family_profile.rs:FamilyProfile]; pinned settings `Setting` [src/pinned.rs:Setting]; `Shape`, `load` [src/inventory.rs:load]; `AudioKind` [src/audio_kind.rs:AudioKind].
- Analysis: `project`, `Projection` [src/projection.rs:project]; `sweep_cmd` [src/sweep.rs:sweep_cmd]; `trace_cmd` [src/trace.rs:trace_cmd]; `workflow_cmd` `Plan`, `Step` [src/workflow.rs:Plan]; replays [src/depth_replay.rs:Policy] [src/hybrid_replay.rs:simulate_hybrid] [src/tree_replay.rs:tree_replay_cmd]; `census_cmd` [src/census.rs:summarize]; `ane_plan_cmd` [src/ane.rs:ane_plan_cmd]; `round_cost` [src/round_cost.rs:from_run].
- Subdirectories: [./bin/EGM.md](./bin/EGM.md), [./dash/EGM.md](./dash/EGM.md), [./kernel/EGM.md](./kernel/EGM.md).

## Invariants
- Each run is a fresh worker with a CLEARED environment plus `PASSTHROUGH`, `PASSTHROUGH_TURBOSPARK` (`TURBOSPARK_HOME`, `TURBOSPARK_LAB_DIR`), `ORCHESTRATOR_ENV` (`TURBOSPARK_ENGAGEMENT=1`) and the arm's own variables, so `OnceLock` knobs match open-time knobs [src/orchestrate.rs:PASSTHROUGH] [src/orchestrate.rs:PASSTHROUGH_TURBOSPARK].
- Ledger lines are never rewritten; host-rejected runs are kept with `eligible: false` [src/ledger.rs:Ledger].
- Result files are write-once: hard-link publish refuses an existing target [src/atomic.rs:write_new].
- Gate bounds live in this crate and a spec may only tighten them [src/gates.rs:effective_bounds].
- An experiment diff touching `crates/lab/`, frozen prompts, `docs/BENCHMARKS.md`, `Cargo.lock`, `.github/`, or any `AGENTS.md`/`CLAUDE.md`/`Cargo.toml` is refused before measuring, as is any change to an existing file under `crates/*/tests/` (new test files are allowed) [src/integrity.rs:FORBIDDEN_PREFIXES] [src/integrity.rs:FORBIDDEN_NAMES] [src/integrity.rs:is_test_file].
- One measuring session per machine via `.machine.lock` flock, nestable under a workflow hold [src/lock.rs:MachineLock].
- Experiment names are validated, not sanitized [src/paths.rs:validate_name].
- `run-experiment` and `confirm` exit 0 keep, 3 discard, 4 inconclusive [src/experiment.rs:exit_code_for]; any returned error exits 1 [src/cli.rs:finish]. Exit 2 happens only for an empty argv [src/cli.rs:main]. Under `--json` stdout carries exactly one envelope [src/out.rs:set_json].
- Worker refusals exit 3 [src/worker.rs:EXIT_REFUSED].

## Side Effects
- Spawns `tslab-worker` subprocesses and samples host load (`ps`, power state) [src/orchestrate.rs:sibling_worker] [src/host.rs:HostMonitor].
- Appends to ledgers and writes files under `$TURBOSPARK_LAB_DIR`, else `$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab` [src/paths.rs:lab_dir].
- Runs `git` (provenance, integrity diff) [src/provenance.rs:git_rev] [src/integrity.rs:changes_since]; runs `xcrun xctrace` for `trace` [src/trace.rs:trace_cmd]; `gate` can run required cargo tests [src/experiment.rs:required_tests].
- Opens Metal devices and models; reads IOReport energy counters [src/energy.rs:EnergyMeter].

## Verification
- Run: `cargo test -p turbospark-lab`
