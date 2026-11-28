---
name: tslab
description: Measures TurboSpark model installs on an Apple Silicon Mac with tslab, the kernel and method evaluator from crates/lab, using a TurboSpark checkout or the bundled binaries. Use when asked to profile an install, bench or A/B a Metal kernel or a knob (a TURBOSPARK_* switch), survey a new model family, trace GPU work, price speculative decoding, measure an audio pipeline, check output quality (golden answers, digests, KL against llama.cpp), load-test the HTTP server, look up pinned kernel settings, run the program.md autoresearch loop, read, export or plot ledgers (~/.turbospark/lab), or add a bench, knob, ledger kind or workflow to tslab. Not for freezing rows in docs/BENCHMARKS.md.
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

Below, `M` is the model (an install directory `*.gturbo`, or a catalog alias
from `~/.turbospark/models`) and `E` is an experiment name you choose.

```sh
$TSLAB capabilities --model M --json           # what this install supports, tags, what to set first
$TSLAB family --model M                        # the family profile: known-best settings, sweeps, findings
$TSLAB workflow new-family --model M --experiment E --dry-run
```

Results go to `$TURBOSPARK_LAB_DIR`, else `$TURBOSPARK_HOME/lab`, else
`~/.turbospark/lab`. Keep `tslab-worker` beside `tslab`: every measured run
is a fresh worker process.

## Before you measure

Run `"$SKILL/scripts/preflight.sh"`. It reports AC power, swap, and other
model processes, builds or editors that would make runs dirty. Finish builds
first. tslab holds one machine lock, so a second measuring session waits.

## Standard A/B loop

Copy this checklist and tick it off. Do not skip step 2: without a null floor
the effect floor is only the 2% default.

```
- [ ] 1. preflight.sh is clean
- [ ] 2. $TSLAB ab --model M --null W --metric X --experiment E      (once per workload and metric)
- [ ] 3. $TSLAB ab --model M --baseline off --candidate on:KNOB=V W --metric X --experiment E --json
- [ ] 4. Read decision and exit code (below), then the verdict in reading-results.md
- [ ] 5. Check the arms ran what they claim: each run record's `engaged` list
- [ ] 6. $TSLAB report --experiment E --html E.html
```

`W` is the workload flags, identical in steps 2 and 3. Always pass
`--metric`: `ab` defaults to `prefill_tok_s`, so a decode switch measured
without `--metric decode_tok_s` yields a prefill number. The header line
names the metric it used (`metric DecodeTokS`); read it before quoting. In
zsh, hold `W` in an array (`W=(--prompt protocol:long-synthesis --decode
128)`, then `$W`): a plain string reaches tslab as one argument.

Exit codes: 0 ok or keep, 1 error (including an unknown command or bad
flag), 2 bare `tslab` with no arguments, 3 discard, 4 inconclusive. On 4
the host was busy: fix it and rerun the same command. Never revert on 4, and
never rerun a NOISE verdict until it wins.

Some commands reuse the codes, with `ok: true` in the envelope: `sanity`,
`quant-ab` and `release-gate` exit 1 when a check failed (a finding, not a
tool error), and `hash-diff`, `anchor`, `image-stage`, `serve-bench` and
`soak` exit 3 when an output changed or a request failed. Read `ok` and the
record before reading the code.

On a family you have not measured before, run `$TSLAB sanity --model M`
before the first sweep. A forward pass that decodes garbage can still run
fast and win (reading-results.md, "Check the output before the timings").

## Which reference to read

| Task | Read |
|---|---|
| Pick a command, its flags and traps | [references/commands.md](references/commands.md) |
| Browse or compare results across experiments | `tslab serve`, then [references/reading-results.md](references/reading-results.md) |
| Parse `--json`, the ledger, `BenchResult` or exported series | [references/output.md](references/output.md) |
| Run or write a workflow | [references/workflows.md](references/workflows.md) |
| Interpret a verdict, a dirty run, or what may be quoted | [references/reading-results.md](references/reading-results.md) |
| What a family is known to want, which sweep first, where its readings came from | [references/families.md](references/families.md) |
| Add a bench, knob, ledger kind, command or workflow (checkout only) | [references/extending.md](references/extending.md) |
| Iterate on a kernel change against the fixed evaluator (checkout only) | `crates/lab/program.md` |

## Other common tasks

```sh
$TSLAB profile --model M --prompt protocol:long-synthesis --decode 64 --experiment E
$TSLAB kernel --model M --bench int4_matmul,attn_decode --hash --experiment E
$TSLAB profile --model M --decode 64 --rounds 6 --dispatch-probe 0,4,8,16 --experiment E
$TSLAB trace --model M --prompt protocol:short-explanation --decode 64 --experiment E
$TSLAB profile --model A --audio-job <workloads>/audio/music-short.json --experiment E
$TSLAB report --experiment E --html E.html
$TSLAB project --knob int4_mma8 --curves E --x 1 --base-from E:base --floor-from E   # price a mode first
$TSLAB calibrate --projection E --verdict E                                          # then check the price
$TSLAB pinned --repo kapitsa --class gemv        # verified settings of compared engines
$TSLAB sweep --model M --sweep all --experiment E --json   # every profile sweep that applies, one verdict per candidate
$TSLAB sweep --model M --baseline plain --candidate mtp2~mtp:2 --candidate mtp4~mtp:4 --metric decode_tok_s --experiment E
$TSLAB family --family qwen35 --findings         # the recorded findings with their provenance
$TSLAB serve --open                              # the read-only dashboard over every ledger (never takes the lock)
$TSLAB sanity --model M --experiment E           # golden answers through the chat template; exit 1 on a miss
$TSLAB idle-gap --model M --experiment E         # first decode step after idle gaps; digests must match the none run
$TSLAB kernel --model M --bench gdn_trajectory --experiment E   # GDN route equivalence over many steps, untimed
$TSLAB serve-bench --server http://127.0.0.1:PORT --experiment E   # the HTTP path; a server claim needs this, not ab
```

`tslab knobs` lists the switches an arm can set (`KNOB=V`) and, below them,
the ones that are kernel modes (`docs/KERNEL_MODES.md`) with the bench that
prices each.

## What needs a checkout

From the zip these refuse or degrade, so do not plan on them: `gate`,
`run-experiment` and `confirm` (they diff and build the source),
`xengine-kl` (it runs `cargo test` and a script from the tree, and needs
`uv`), `image-stage` fixtures (the repository's `scripts/z_image_capture.py`), `corpus:N`
prompts and the `fill-ladder` workflow that uses them (they read the
repository's docs, so use `protocol:` prompts), and
the git revision in run provenance (each run still records the binary's
SHA-256, which identifies the build). The zip's binaries measure the build
they came from (`BUILD.txt`). A result about a newer build needs binaries
built from it.

## Rules

- Quote a number only from clean, quotable runs, with its ledger id, model,
  workload, chip, power source and page-cache class.
- Do not loosen a bound, flip a default or freeze a `docs/BENCHMARKS.md` row.
  Those are human decisions.
- Do not edit `crates/lab` while it grades you (`run-experiment` refuses).
- Readings belong in the family profile (`crates/lab/families/<family>.json`,
  with the ledger id or the doc heading they came from) and the topic docs
  (`docs/MLX_SERVE_KERNELS.md`, `docs/KAPITSA_KERNELS.md`, the playbook) with
  their conditions, never in an AGENTS.md. A profile's `reading` is a doc
  table, not a tslab verdict: do not quote it as one.
