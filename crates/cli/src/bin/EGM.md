# bin

## Overview
Binary entry points of `turbospark-cli`: the `turbospark` front door and the peer tools `turbospark-model`, `turbospark-image`, `turbospark-music`, `turbospark-clone`, `turbospark-speak`, `turbospark-steer` and `turbospark-ocr`. `turbospark-check` is `src/main.rs`. Each is a thin shell: parse argv, call library crates, print, pick an exit code.

## Interface
- [turbospark.rs] front door; includes `../agent.rs`, `../daemon.rs`, `../ui/mod.rs`, `../front/models.rs`, `../front/server.rs` by `#[path]`.
- [model.rs] catalog, probe, pull; delegates to `model_cmd`.
- [steer.rs] trait steering; exit 0 success, 1 refused or failed, 2 invalid command line.
- [image.rs], [music.rs], [clone.rs], [speak.rs], [ocr.rs]: per-modality commands.
- Children: [model_cmd](./model_cmd/EGM.md), [steer_cmd](./steer_cmd/EGM.md). `steer_data/` holds embedded pack data (no EGM.md).

## Invariants
- Model and steer commands are separate binaries so `crates/invocation` keeps its flat `--model` plus one mode contract [model.rs module doc], [steer.rs module doc].

## Side Effects
- Process exit codes, stdout/stderr output, and child process spawning via the front door [turbospark.rs].

## Verification
- Run: `cargo test -p turbospark-cli`
