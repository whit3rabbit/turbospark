---
name: tslab
description: Drive tslab, TurboSpark's kernel and method evaluator, as an agent. Use whenever the task is to profile a model install, bench a Metal kernel, A/B a runtime switch, survey a new model family, trace GPU activity, price speculative decoding, measure an audio pipeline, or read and visualize tslab ledgers in this repository. Covers the --json contract, which command answers which question, workflows, and what may be quoted.
---

# tslab for agents

`tslab` (crate `crates/lab`) measures, and it never needs you to decide a
result. Your job is to pick the command, run it, read its envelope, and
report what the ledger says with its conditions. The loop for proposing
kernel changes is `crates/lab/program.md`. Read it before editing runtime
code against tslab.

## Before the first command

1. Build both binaries from a clean tree, release:
   `cargo build --release -p turbospark-lab`. `profile`, `ab`, `trace` and
   workflows need `tslab-worker` beside `tslab`.
2. Check the host: AC power, no other model process (`ps`), no build
   running. tslab records dirty runs instead of refusing them, so a busy
   host gives you ineligible runs, not errors.
3. Ask the install what it supports. Every later choice keys off this:

   ```sh
   tslab capabilities --model <install> --json
   ```

   `result.tags` (for example `moe`, `drafter`, `mtp`, `probe`,
   `chunked_prefill`, `bench:verify_ladder`, `audio:music`) and
   `result.benches` (cases, or why not) are what workflows branch on.

## The output contract

Add `--json` to any command. Stdout is then exactly one object:

```json
{"schema": "tslab/1", "command": "ab", "ok": true, "exit_code": 0,
 "decision": "keep_pending_confirm", "records": [{"id": "...", "kind": "verdict", "ledger": "..."}],
 "result": {...}, "warnings": [], "error": "only when ok is false"}
```

Tables and progress go to stderr. `tslab schema --json` returns the JSON
Schema of the envelope, every ledger kind, `BenchResult` and the exported
series. Exit codes: 0 ok or keep, 1 error, 2 usage, 3 discard (also
`hash-diff` with a changed output), 4 inconclusive (do not revert, fix the
host and rerun).

## Which command answers what

| Question | Command |
|---|---|
| What can tslab measure on this install? | `capabilities --model M` |
| Which weight shapes does it bind? | `inventory --model M` |
| Where does a pass spend its time? | `profile --model M --prompt protocol:long-synthesis --decode 64` |
| What does one block cost in place? (dense qwen, llama) | `profile --model M --decode 64 --standin attn,mlp,gdn` |
| What does one chained dispatch cost? | `profile --model M --decode 64 --dispatch-probe 0,4,8` |
| Is the GPU launch-bound? Which kernels dominate? | `trace --model M --prompt protocol:short-explanation --decode 64` |
| A kernel's cost curve at the install's shapes | `kernel --model M --bench NAME[,NAME] --hash` (`kernel --list`) |
| Did a change alter any kernel's output bits? | `kernel ... --hash` on both builds, then `hash-diff --experiment A --against B` |
| Whole-model GEMM cost per verify width | `kernel --model M --bench verify_ladder --rows 1,2,3,4,5,8,9` |
| Is this host's noise floor known? | `ab --model M --null` (once per workload) |
| Does switch X win? | `ab --model M --baseline off --candidate on:KNOB=V` |
| Keep or discard a committed change? | `run-experiment --spec S.json`, then `confirm` |
| Which speculative depth? | `workflow spec-depth --model M` |
| Would tree verify pay? | `tree-replay --model M --drafter mtp|dflash` |
| An audio pipeline's stage times | `profile --model A --audio-job crates/lab/workloads/audio/music-short.json` |

Knob names for `KEY=V` are in `tslab knobs`. A registered knob that prints
an engagement line is checked: an arm whose run shows it did not take is
marked `not engaged` and is ineligible. Read the `reasons` of dirty runs
before rerunning.

## Workflows

```sh
tslab workflow --list
tslab workflow new-family --model M --experiment <name> --dry-run
tslab workflow new-family --model M --experiment <name> --json
tslab workflow new-family --model M --experiment <name> --resume <run>
```

Built-ins: `new-family`, `kernel-candidate` (`--var against=EXP --var
candidate=KNOB=V`), `spec-depth`, `audio-survey`, `report`. Steps whose
`when` does not hold for the install are skipped with the reason. One step
runs alone with `--step ID`. A custom plan is `--plan PLAN.json` in the same
format (`tslab workflow --print new-family` shows one).

## Visualizing

```sh
tslab report --experiment E --html E.html            # offline page, inline SVG
tslab export --experiment E --format json --redact-paths --out E.series.json
```

Series are `tslab/1` objects with axes, points, and the model, family,
chip, rev, eligibility and quotability to label them. A new kernel bench
appears in both with no extra code once it returns a `BenchResult`.

## Rules

- Never quote a number without its ledger id, model, workload, chip, power
  source and page-cache class. Never quote a dirty, traced, probed or
  stood-in run.
- Do not edit `crates/lab` while it is grading you, and do not loosen a
  bound, flip a default or freeze a `docs/BENCHMARKS.md` row: those are
  human decisions.
- A verdict of NOISE is not a reason to rerun until it wins. INCONCLUSIVE
  means the host was busy: fix that, then rerun the same spec.
- One measuring session per machine: tslab holds a lock, and a workflow
  holds it for all its steps. If it is taken, wait.
