# tslab binaries

## Overview
Two thin entry points. Both exit 2 with a message on non-macOS.

## Interface
- `tslab`: calls `turbospark_lab::cli::main` [src/bin/tslab.rs:main].
- `tslab-worker`: calls `turbospark_lab::worker::main_entry`; measures one arm once and writes one record [src/bin/tslab_worker.rs:main].

## Invariants
- Both are gated `cfg(target_os = "macos")` with a stub `main` elsewhere [src/bin/tslab.rs:main] [src/bin/tslab_worker.rs:main].

## Side Effects
- None here; see `../cli.rs` and `../worker.rs` behavior in [../EGM.md](../EGM.md).

## Verification
- Run: `cargo test -p turbospark-lab`
