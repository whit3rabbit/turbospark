---
name: turbospark-lab
description: Measure and extend TurboSpark inside its repository with tslab, the kernel and method evaluator in crates/lab. Use when working in the turbospark repo and the task is to profile a model install, bench or A/B a Metal kernel or runtime switch, survey a new model family, trace GPU work, price speculative decoding, measure an audio pipeline, run the program.md autoresearch loop, read or plot tslab ledgers, or add a bench, knob, ledger kind or workflow to tslab.
---

# tslab in the TurboSpark repository

`tslab` measures and decides. You pick the command, run it, read its
envelope, and report what the ledger says with its conditions. You never
decide a result yourself.

## Quick start

```sh
cargo build --release -p turbospark-lab        # tslab and tslab-worker, side by side
TSLAB=target/release/tslab
$TSLAB capabilities --model <install> --json   # what this install supports, and tags
$TSLAB workflow new-family --model <install> --experiment <name> --dry-run
```

`--model` is an install directory or a catalog alias. Results go to
`$TURBOSPARK_LAB_DIR`, else `$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab`.

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
| Add a bench, switch, ledger kind, command or workflow | [references/extending.md](references/extending.md) |
| Iterate on a kernel change against the fixed evaluator | `crates/lab/program.md` |

## Common tasks

```sh
$TSLAB profile --model M --prompt protocol:long-synthesis --decode 64 --experiment E
$TSLAB ab --model M --null --experiment E                         # once per workload
$TSLAB ab --model M --baseline off --candidate on:KNOB=V --experiment E --json
$TSLAB kernel --model M --bench int4_matmul,attn_decode --hash --experiment E
$TSLAB profile --model M --decode 64 --rounds 6 --dispatch-probe 0,4,8,16 --experiment E
$TSLAB trace --model M --prompt protocol:short-explanation --decode 64 --experiment E
$TSLAB report --experiment E --html E.html
```

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

The binary-only version of this skill, for machines without the source, is
built by `scripts/package-tslab-skill.sh` from `portable/SKILL.md` and these
references.
