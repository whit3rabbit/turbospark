# tslab skill scripts

## Overview
Two read-only POSIX shell checks that the tslab skill runs before measuring. They report, they fix and kill nothing.

## Interface
- `doctor.sh`: checks macOS on Apple Silicon, `tslab` and `tslab-worker` binaries (bundled `bin/` or checkout `target/release`), Gatekeeper quarantine, `tslab knobs --json`, the workloads dir and the results dir [skill/scripts/doctor.sh:SKILL, MODE, BIN]. Exit 0 when all required checks pass, 1 otherwise.
- `preflight.sh`: checks AC power, foreign model or build processes (exact first-word basenames, a shorter list than the guard's name-fragment `MODEL_MARKERS`, so test binaries such as `*_gate-<hash>` are not flagged) and 10 GiB free disk, which copies the guard's `default_min_free_gb` [crates/lab/src/host.rs:default_min_free_gb] [crates/lab/src/host.rs:MODEL_MARKERS]. It has no CPU-load check, so a clean preflight does not rule out a run the guard rejects for sustained load. Exit 1 when a run would likely be dirty [skill/scripts/preflight.sh:fail].

## Invariants
- Both scripts run under `set -u` and only report [skill/scripts/doctor.sh:set -u] [skill/scripts/preflight.sh:set -u].
- A quarantined binary is reported with the `xattr` command but never cleared by the script [skill/scripts/doctor.sh:com.apple.quarantine].
- The results dir resolves as `TURBOSPARK_LAB_DIR`, else `TURBOSPARK_HOME/lab`, else `$HOME/.turbospark/lab`, matching `paths::lab_dir` [skill/scripts/doctor.sh:LAB].

## Side Effects
- `doctor.sh` runs `mkdir -p` on the results directory and runs `tslab knobs --json` [skill/scripts/doctor.sh:LAB].
- `preflight.sh`: None beyond read-only queries (`pmset`, `ps`, `df`, `sysctl`, `lsof`).

## Verification
- Run: `cargo test -p turbospark-lab` (the scripts have no tests of their own; run `sh skill/scripts/doctor.sh` manually).
