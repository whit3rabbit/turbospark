# front

## Overview
Modules for the `turbospark` front-door binary (included via `#[path]` from `bin/turbospark.rs`). They translate user model spellings to catalog aliases and delegate work to peer binaries, so install and delete logic exists once in `turbospark-model`.

## Interface
- [models.rs:Resolved], [models.rs:resolve], [models.rs:alias_for], [models.rs:looks_like_path], [models.rs:open_catalog]: model name resolution.
- [models.rs:rewrite_model_flags], [models.rs:plan_run], [models.rs:RunPlan], [models.rs:check_args], [models.rs:run]: `run` planning and launch of `turbospark-check`.
- [models.rs:ensure_installed], [models.rs:pull], [models.rs:rm], [models.rs:pull_args], [models.rs:unknown_model]: install and removal front ends.
- [models.rs:check_command], [models.rs:run_check], [models.rs:HIDE_RESOLVED_REQUEST_ENV]: peer invocation.
- [server.rs:status], [server.rs:ps]: managed daemon status via pid file and `/health`.

## Invariants
- Paths are the caller's: a path-looking name is passed through verbatim [models.rs:looks_like_path].
- Check invocations always set `TURBOSPARK_HIDE_RESOLVED_REQUEST=1` [models.rs:check_command].
- Only the managed daemon is visible to status; foreground `serve` writes no pid file [server.rs:snapshot].

## Side Effects
- Spawns `turbospark-check` and `turbospark-model` child processes [models.rs:run], [models.rs:pull].
- Prompts for confirmation only on an interactive terminal [models.rs:confirm]; with `--yes` or a non-terminal stdin or stderr, `run` downloads a missing curated model without asking [models.rs:ensure_installed], [models.rs:interactive].
- HTTP health probe on loopback [server.rs:snapshot].

## Verification
- Run: `cargo test -p turbospark-cli`
