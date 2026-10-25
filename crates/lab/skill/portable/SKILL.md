---
name: tslab
description: Measure TurboSpark model installs on an Apple Silicon Mac with the bundled tslab binaries, no source checkout needed. Use when asked to profile a TurboSpark install, A/B a TurboSpark runtime switch (TURBOSPARK_* variables), bench its Metal kernels at an install's shapes, check what tslab supports for a model family, trace GPU work, measure a TurboSpark audio pipeline, or read, export or plot tslab ledgers (~/.turbospark/lab).
---

# tslab (bundled binaries)

This skill ships `bin/tslab` and `bin/tslab-worker` for macOS on Apple
Silicon. `tslab` measures and decides. You pick the command, run it, read its
envelope, and report what the ledger says with its conditions.

## Quick start

```sh
SKILL=<this skill's directory>
"$SKILL/scripts/doctor.sh"                     # binaries, quarantine, tools, lab dir
TSLAB="$SKILL/bin/tslab"
$TSLAB capabilities --model <install> --json   # what this install supports, and tags
$TSLAB workflow new-family --model <install> --experiment <name> --dry-run
```

`--model` is a TurboSpark install directory (`*.gturbo`) or an alias from
`~/.turbospark/models`. Results go to `$TURBOSPARK_LAB_DIR`, else
`$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab`. Keep `tslab-worker` beside
`tslab`: every measured run is a fresh worker process.

## Before you measure

- AC power, no other model process (`ps`), no build or heavy app running,
  and `sysctl vm.swapusage` low. Busy hosts give dirty runs, not errors.
- One measuring session per machine: tslab holds a lock. If it is held, wait.
- A model's first load reads it from disk. The warmup round absorbs that.

## Which reference to read

| Task | Read |
|---|---|
| Pick a command, its flags and traps | [references/commands.md](references/commands.md) |
| Parse `--json`, the ledger, `BenchResult` or exported series | [references/output.md](references/output.md) |
| Run or write a workflow | [references/workflows.md](references/workflows.md) |
| Interpret a verdict, a dirty run, or what may be quoted | [references/reading-results.md](references/reading-results.md) |

## Common tasks

```sh
$TSLAB profile --model M --prompt protocol:long-synthesis --decode 64 --experiment E
$TSLAB ab --model M --null --experiment E                         # once per workload
$TSLAB ab --model M --baseline off --candidate on:KNOB=V --experiment E --json
$TSLAB kernel --model M --bench all --hash --experiment E
$TSLAB trace --model M --prompt protocol:short-explanation --decode 64 --experiment E
$TSLAB profile --model A --audio-job "$SKILL/workloads/audio/music-short.json" --experiment E
$TSLAB report --experiment E --html E.html
```

`tslab knobs` lists the switches an arm can set (`KNOB=V`). Bundled jobs:
`$SKILL/workloads/` (message and audio jobs) and `$SKILL/chats/` (a long
document chat, `--prompt chat:PATH`).

## What needs the source tree

Without a TurboSpark checkout these refuse or degrade, so do not plan on them:
`gate`, `run-experiment` and `confirm` (they diff and build the source),
`corpus:N` prompts (they read the repository's docs, so use `protocol:` prompts),
and the git revision in run provenance (each run still records the binary's
SHA-256, which identifies the build).

## Rules

- Quote a number only from clean, quotable runs, with its ledger id, model,
  workload, chip, power source and page-cache class.
- NOISE is not a reason to rerun until it wins. INCONCLUSIVE means a busy
  host: fix it and rerun.
- These binaries measure the TurboSpark build they came from. A result about
  a newer build needs binaries built from it.
