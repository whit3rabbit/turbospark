---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e04"
title: "Where config lives"
summary: "One version in root Cargo.toml, no [profile.release] section (measured, not an oversight), TURBOSPARK_* env vars for everything runtime-tunable"
tags: ["config", "day-one"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e05"]
source: "Cargo.toml, docs/ENV.md, docs/MODELS.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## Where does config live?

There's no separate config file format in this repo. Configuration is one
of three things: `Cargo.toml` (build/workspace), `TURBOSPARK_*` environment
variables (runtime), or CLI flags (per-invocation, see `docs/CLI.md`).

**Build config.** The root `Cargo.toml` holds `[workspace]` (the 17 crate
members) and `[workspace.package]` (one shared `version`, `edition = 2021`,
`rust-version = "1.82"`, `license = "MIT"`). Every crate inherits these via
`version.workspace = true` etc. `rust-toolchain.toml` pins the toolchain
channel (`stable`) and components (`rustfmt`, `clippy`), not a version
number.

**No `[profile.release]` section, on purpose.** Cargo's stock opt-level 3,
no LTO, 16 codegen units. This was measured, not skipped: `lto = "thin"` +
`codegen-units = 1` was worth +0.27% decode throughput on the real Gemma 4
install (44.545 -> 44.667 tok/s) but took the CLI's release build from 6.9s
to 30.6s, a 4.5x hit to the edit-measure loop for a gain inside the
run-to-run measurement spread. See the comment block above `[workspace]` in
`Cargo.toml` for the full writeup.

**Runtime config.** Everything that changes behavior at runtime is a
`TURBOSPARK_*` env var, catalogued in full in [`docs/ENV.md`](../docs/ENV.md).
The load-bearing ones for day-to-day work:

- `TURBOSPARK_HOME` (default `~/.turbospark`): the model store root.
  Installs live at `$TURBOSPARK_HOME/models/<alias>.gturbo`, and
  `$TURBOSPARK_HOME/installed.json` is the install record.
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`: gated Hugging Face repo access for
  `turbospark-model probe`/`pull`.
- `TURBOSPARK_API_KEY`: optional bearer-token auth for `turbospark-server`.
- `TURBOSPARK_PHASES=1`, `TURBOSPARK_DISPATCH_PROFILE=1`: decode-path
  profiling seams on `turbospark-check`.

Most of the rest of `docs/ENV.md`'s ~50 variables are test-only (pointing a
gated test at a real `.gturbo` install) or experimental A/B seams
(`TURBOSPARK_SHARED_CB`, `TURBOSPARK_ROUTED_PIPELINE`, etc.) rather than
something a user sets.

## Don't

- Don't add a new config file format. This project has stayed on
  Cargo.toml + env vars + CLI flags deliberately. A new flag touches five
  places in `crates/invocation` (see that crate's own `CLAUDE.md`).
- Don't assume `--model <name>` needs a full path. It resolves an alias
  against the catalog store too, through one shared `resolve_model_arg` used
  by both `turbospark-check` and `turbospark-server`. An existing directory
  always wins over an alias of the same name.
- Don't add a `[profile.release]` section without re-measuring on real
  hardware first, and reading the existing comment block in `Cargo.toml`.
  The prior attempt is documented there with numbers.
