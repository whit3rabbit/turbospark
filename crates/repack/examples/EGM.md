# Repack Examples

## Overview
Four standalone cargo examples that drive the `turbospark_repack` public API for local bring-up: streaming intake of a checkpoint into an install directory, a Flint SwiGLU migration, and a MiniCPM-V tower fixture. Each is a `main` binary that reads CLI args, writes to a new output directory, and is not part of the shipped install path (the catalog pull is).

## Interface
- `main` [fish_drama3_intake.rs:main]: streams Fish Drama 3 shards over `HttpRangeSource` into a Drama install via `write_fish_drama3_install_streamed`
- `main`/`run` [flint_local_intake.rs:run]: builds a Flint install from a local checkpoint dir through `write_flint_install_streamed`, copies four tokenizer sidecars, records trained context
- `FileRangeSource` [flint_local_intake.rs:FileRangeSource]: `RangeSource` over a local file via positional `read_at`
- `shard_names` [flint_local_intake.rs:shard_names]: shard file list from `model.safetensors.index.json`
- `main` [flint_rebalance.rs:main]: migrates an existing Flint install in place via `rebalance_install_in_place`; `--undo` reverses it
- `main`/`run` [minicpm_tower_local.rs:run]: builds a tower-only MiniCPM-V 4.7 directory via `read_vision_entries_quantized` and `write_vision_sidecar`
- `FileRangeSource` [minicpm_tower_local.rs:FileRangeSource]: duplicate local-file `RangeSource`
- Subdirectories: none

## Invariants
- The shard URL prefix must end with `/` or the run errors [fish_drama3_intake.rs:main]
- Shard bytes are not hash-checked here; range reads never see a whole file [fish_drama3_intake.rs:main]
- An existing destination directory is refused rather than overwritten [flint_local_intake.rs:run, minicpm_tower_local.rs:run]
- A short read returns `DownloadError::ShortRead` instead of a partial buffer [flint_local_intake.rs:FileRangeSource, minicpm_tower_local.rs:FileRangeSource]
- Rebalance refuses an install whose marker already records a shift, and `--undo` refuses one with no shift (exit 2) [flint_rebalance.rs:main]
- The MiniCPM output is a test fixture, not an install; the example header says the catalog and runtime refuse to attach it (enforced in other crates, not by `run`) [minicpm_tower_local.rs:run]
- Tower run errors when config declares no active vision tower [minicpm_tower_local.rs:run]

## Side Effects
- Network: HTTP range reads of remote safetensors shards [fish_drama3_intake.rs:main]
- Filesystem: creates the output directory and writes install files; stderr progress lines [fish_drama3_intake.rs:main, flint_local_intake.rs:run]
- Filesystem: copies tokenizer sidecars and writes trained-context metadata [flint_local_intake.rs:run]
- Filesystem: mutates an existing install in place [flint_rebalance.rs:main]
- Filesystem: copies `preprocessor_config.json` into the tower dir [minicpm_tower_local.rs:run]
- Process: exits non-zero on error; `flint_rebalance` panics via `expect` on bad usage or IO [flint_rebalance.rs:main]

## Verification
- Run: `cargo build -p turbospark-repack --examples`
