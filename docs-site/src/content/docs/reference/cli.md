---
title: CLI Reference
description: The four turbospark command-line entry points, captured live from each binary's --help output.
---

<!-- generated: cli-help lane, signal: crates/cli/src/main.rs (plus crates/cli/src/bin/model.rs, crates/server/src/main.rs, crates/bench/src/main.rs) -->

# CLI Reference

This workspace ships four command-line entry points. Everything on these pages
was captured by running the built release binaries with `--help` (2026-09-05);
the binaries were newer than every source file in scope at capture time, so the
text reflects the current tree.

| Command | One-line summary | Reference page |
|---|---|---|
| `turbospark-check` | Run generation once against an install: a raw prompt, a rendered chat conversation, or an interactive REPL. | [turbospark-check](./cli-turbospark-check) |
| `turbospark-model` | "find, inspect and install models" (its own header line): list, inspect, probe, recommend, pull, path, rm. | [turbospark-model](./cli-turbospark-model) |
| `turbospark-server` | Serve an HTTP API from a `.gturbo` install or catalog alias, with one flag surface for ports, guardrails, prefix reuse, speculation and steering. | [turbospark-server](./cli-turbospark-server) |
| `turbospark-bench` | Throughput benchmark harness: `usage: turbospark-bench <tokenizer-dir> [--real] \| --model <install-dir> [--case <id>]` (its own usage line, verbatim). | [turbospark-bench](./cli-turbospark-bench) |

The hand-written narrative for these same binaries is
`docs/CLI.md` at the repository root. The pages under `reference/` are the
live cross-check against it.

## Common behavior

Verified against the release binaries:

- `--help` short-circuits and exits 0 on `turbospark-check`, `turbospark-model`
  and `turbospark-server` without any model or tokenizer argument.
- `--version` prints `turbospark 0.1.0` on the same three binaries.
- `turbospark-bench` has neither flag: it treats the first argument as a
  tokenizer directory, so `--version` fails with
  `failed to load tokenizer: invalid chat messages: No such file or directory (os error 2)`
  (exit 1), and a bare invocation prints its usage line to stderr with exit 2.
- `turbospark-model` accepts `--help` after any sub-command (`list`, `info`,
  `probe`, `recommend`, `pull`, `path`, `rm`) and prints the same top-level
  help text each time; there is no per-sub-command help page (verified for all
  seven, each exit 0).

## Drift found against docs/CLI.md

`docs/CLI.md` is the hand-written reference for these binaries. A sweep of all
58 distinct flags in the captured help output against that page found two
flags documented elsewhere in the repo but absent from it, and one present
only in passing:

- `--session-slots` (`turbospark-server`): in the live `--help`, documented in
  `CHANGELOG.md`, no match in `docs/CLI.md` under any spelling.
- `--reuse-trunk-from` (`turbospark-model`): in the live `--help`, documented
  in `docs/MTP.md` and `docs/MTP_SPECULATIVE.md`, no match in `docs/CLI.md`.
- `--prefix-reuse` (`turbospark-server`): in the live `--help`; `docs/CLI.md`
  mentions the `[prefix-reuse] N/M` counter and the `TURBOSPARK_PREFIX_REUSE`
  environment variable but has no flag row for the flag itself.

Every other captured flag appears in `docs/CLI.md`.
