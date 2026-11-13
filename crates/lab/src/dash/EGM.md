# tslab dashboard (`tslab serve`)

## Overview
A read-only, server-rendered dashboard over the lab dir: experiments table, charts, compare view, verdict board, sweeps, per-family profiles, and a JSON API. No POST route exists and the machine lock is never taken, so it can stay open beside a measurement [src/dash/mod.rs:router].

## Interface
- `serve_cmd(args)`: `tslab serve [--port 7878] [--bind 127.0.0.1] [--lab DIR] [--open]` [src/dash/mod.rs:serve_cmd].
- `router(app)` (bare, for tests) and `serve_router(app, loopback)` (adds hardening) [src/dash/mod.rs:router] [src/dash/mod.rs:serve_router].
- `App::new(lab)`, `App::with_index` [src/dash/mod.rs:App].
- HTML routes: `/`, `/experiment/:name`, `/compare`, `/verdicts`, `/sweeps`, `/families`, `/family/:family`, `/record/:name/:id`; JSON routes under `/api/` (`experiments`, `experiment`, `ledger`, `series`, `verdicts`, `sweeps`, `family`, `record`) [src/dash/mod.rs:router] [src/dash/api.rs:experiments].
- `LabIndex::build` / `refresh` / `record`, `ExperimentSummary`, `VerdictRow`, `SweepRow`, `ProfileRow` [src/dash/index.rs:LabIndex].
- Page renderers and `Params`, `shell`, `not_found` [src/dash/pages.rs:Params] [src/dash/pages.rs:shell].

## Invariants
- GET-only: only `get(...)` routes are registered [src/dash/mod.rs:router].
- On a loopback bind, a non-loopback Host header gets 403 (DNS rebinding defense) [src/dash/mod.rs:hardening] [src/dash/mod.rs:host_is_loopback].
- Every response, including the 403 Host refusal, carries `nosniff` and a CSP with `default-src 'none'` [src/dash/mod.rs:hardening].
- A non-loopback `--bind` turns the Host check off and the dashboard has no authentication; `serve_cmd` prints a warning [src/dash/mod.rs:serve_cmd].
- The index is rebuilt only when a ledger's size or mtime moves [src/dash/index.rs:LabIndex].

## Side Effects
- Binds a TCP listener and runs a tokio runtime [src/dash/mod.rs:serve_cmd].
- With `--open`, spawns macOS `open` on the URL [src/dash/mod.rs:serve_cmd].
- Reads ledgers under the lab dir; writes nothing there.

## Verification
- Run: `cargo test -p turbospark-lab`
