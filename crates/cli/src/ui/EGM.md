# ui

## Overview
Terminal presentation shared by the `turbospark` and `turbospark-model` binaries: ASCII-only styled text, tables, and section headers. It computes no domain data.

## Interface
- [mod.rs:Stream], [mod.rs:color_enabled]: per-stream color decision.
- [mod.rs:Paint] with `new`, `out`, `err`, `plain`, `bold`, `dim`, `accent`, `link`, `warn`, `error`.
- [mod.rs:ok], [mod.rs:note], [mod.rs:warn], [mod.rs:error]: status lines.
- [mod.rs:columns], [mod.rs:clip], [mod.rs:section].
- [mod.rs:Table], [mod.rs:Cell], [mod.rs:CellStyle], [mod.rs:Align]: table rendering.

## Invariants
- Color is off under `NO_COLOR` and `TERM=dumb`, but `CLICOLOR_FORCE` (any value but `0`) is checked first and overrides both [mod.rs:color_enabled].
- Output is ASCII only; helpers emit `+`, `>`, `!` [mod.rs:ok], [mod.rs:note], [mod.rs:warn]; no helper emits `*` though the module doc lists it.
- `--json` views bypass this module [mod.rs module doc].

## Side Effects
- `ok`, `note`, `warn`, `error` write to stderr; stdout only via callers printing `Paint` text [mod.rs:ok], [mod.rs:error]; reads `COLUMNS` and `TERM` [mod.rs:columns].

## Verification
- Run: `cargo test -p turbospark-cli`
