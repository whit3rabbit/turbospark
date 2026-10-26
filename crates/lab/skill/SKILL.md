---
name: tslab
description: Measure TurboSpark model installs on an Apple Silicon Mac with tslab, the kernel and method evaluator from crates/lab, from a TurboSpark checkout or from the bundled binaries in this skill. Use when asked to profile a TurboSpark install, bench or A/B a Metal kernel or runtime switch (TURBOSPARK_* variables), survey a new model family, trace GPU work, price speculative decoding, measure an audio pipeline, run the program.md autoresearch loop, read, export or plot tslab ledgers (~/.turbospark/lab), or add a bench, knob, ledger kind or workflow to tslab.
---

# tslab

`tslab` measures and decides. You pick the command, run it, read its
envelope, and report what the ledger says with its conditions. You never
decide a result yourself.

## Quick start

```sh
SKILL=<this skill's directory>
"$SKILL/scripts/doctor.sh"                     # finds tslab, checks it can run
```

Use the `tslab` doctor reports:

- From the zip: `TSLAB="$SKILL/bin/tslab"`. Jobs are in `$SKILL/workloads/`
  and `$SKILL/chats/`.
- In a TurboSpark checkout: `cargo build --release -p turbospark-lab`, then
  `TSLAB=target/release/tslab`. Jobs are in `crates/lab/workloads/` and
  `crates/lab/chats/`.

```sh
$TSLAB capabilities --model <install> --json   # what this install supports, and tags
$TSLAB workflow new-family --model <install> --experiment <name> --dry-run
```

`--model` is an install directory (`*.gturbo`) or a catalog alias from
`~/.turbospark/models`. Results go to `$TURBOSPARK_LAB_DIR`, else
`$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab`. Keep `tslab-worker` beside
`tslab`: every measured run is a fresh worker process.

## Before you measure

- Finish building first. A `cargo build`, a test run or an editor's
  background check makes runs dirty.
- Check AC power, `sysctl vm.swapusage`, and `ps` for another model process
  or `turbospark-bench`. Do not time beside another session's measurement.
- One measuring session per machine: tslab holds a lock. If it is held, wait.

## Which reference to read

| Task | Read |
|---|---|
| Pick a command, its flags and traps | [references/commands.md](references/commands.md) |
| Parse `--json`, the ledger, `BenchResult` or exported series | [references/output.md](references/output.md) |
| Run or write a workflow | [references/workflows.md](references/workflows.md) |
| Interpret a verdict, a dirty run, or what may be quoted | [references/reading-results.md](references/reading-results.md) |
| Add a bench, switch, ledger kind, command or workflow (checkout only) | [references/extending.md](references/extending.md) |
| Iterate on a kernel change against the fixed evaluator (checkout only) | `crates/lab/program.md` |

## Common tasks

```sh
$TSLAB profile --model M --prompt protocol:long-synthesis --decode 64 --experiment E
$TSLAB ab --model M --null --experiment E                         # once per workload
$TSLAB ab --model M --baseline off --candidate on:KNOB=V --experiment E --json
$TSLAB kernel --model M --bench int4_matmul,attn_decode --hash --experiment E
$TSLAB profile --model M --decode 64 --rounds 6 --dispatch-probe 0,4,8,16 --experiment E
$TSLAB trace --model M --prompt protocol:short-explanation --decode 64 --experiment E
$TSLAB profile --model A --audio-job <workloads>/audio/music-short.json --experiment E
$TSLAB report --experiment E --html E.html
$TSLAB project --knob int4_mma8 --curves E --x 1 --base-from E:base --floor-from E   # price a mode first
$TSLAB calibrate --projection E --verdict E                                          # then check the price
```

`tslab knobs` lists the switches an arm can set (`KNOB=V`) and, below them,
the ones that are kernel modes (`docs/KERNEL_MODES.md`) with the bench that
prices each.

## What needs a checkout

From the zip these refuse or degrade, so do not plan on them: `gate`,
`run-experiment` and `confirm` (they diff and build the source), `corpus:N`
prompts (they read the repository's docs, so use `protocol:` prompts), and
the git revision in run provenance (each run still records the binary's
SHA-256, which identifies the build). The zip's binaries measure the build
they came from (`BUILD.txt`). A result about a newer build needs binaries
built from it.

## Rules

- Quote a number only from clean, quotable runs, with its ledger id, model,
  workload, chip, power source and page-cache class.
- NOISE is not a reason to rerun until it wins. INCONCLUSIVE means a busy
  host: fix it and rerun, never revert.
- Do not loosen a bound, flip a default or freeze a `docs/BENCHMARKS.md` row.
  Those are human decisions.
- Do not edit `crates/lab` while it grades you (`run-experiment` refuses).
- Readings belong in the topic docs (`docs/MLX_SERVE_KERNELS.md`,
  `docs/KAPITSA_KERNELS.md`, the playbook) with their conditions, never in an
  AGENTS.md.
