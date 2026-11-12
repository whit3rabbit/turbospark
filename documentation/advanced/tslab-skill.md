---
description: "The tslab agent skill: what it teaches an agent (Claude Code, Codex, Antigravity) about driving TurboSpark Lab, how to install it from a checkout or a packaged zip, the doctor and preflight scripts, the evals, and example prompts."
icon: robot
---

# The tslab agent skill

`crates/lab/skill/` is an agent skill: a folder of instructions and scripts
that teaches a coding agent to run [tslab](tslab.md) the way the lab expects.
The entry point is `SKILL.md`. It works with Claude Code, Codex, and
Antigravity IDE, which all load skills from a `skills/<name>/SKILL.md` folder.

The skill does not measure anything on its own. It tells the agent which
command answers which question, how to read the `--json` envelope, and what
it may and may not quote. tslab still makes every decision.

## What it teaches

| Topic | Where in the skill |
| --- | --- |
| Finding the right binary (the zip's `bin/tslab` or a checkout's release build) | `SKILL.md`, `scripts/doctor.sh` |
| Checking the host is quiet before spending a run | `scripts/preflight.sh` |
| The standard A/B loop as a checklist: preflight, null floor, A/B, read the verdict, check engagement, report | `SKILL.md` |
| Every command, its flags, what it writes, and its traps | `references/commands.md` |
| The `--json` envelope, ledger records, `BenchResult`, exported series | `references/output.md` |
| Verdicts, dirty runs, sweeps, the compare view, and what may be quoted | `references/reading-results.md` |
| What each model family is known to want and which sweep to run first | `references/families.md` |
| Running and writing workflows | `references/workflows.md` |
| Adding a bench, knob, ledger kind, command or workflow (checkout only) | `references/extending.md` |

The rules it holds the agent to:

- Quote a number only from clean, quotable runs, with its ledger id, model,
  workload, chip, power source, and page-cache class.
- Always pass `--metric`. `ab` defaults to `prefill_tok_s`, so a decode switch
  measured without `--metric decode_tok_s` yields a prefill number.
- Exit 4 (inconclusive) means the host was busy. Fix it and rerun the same
  command. Never revert a change on exit 4, and never rerun a NOISE verdict
  until it wins.
- Prove each arm ran its lane from the run record's `engaged` list, not from
  the arm's name.
- Do not loosen a bound, flip a default, or freeze a `docs/BENCHMARKS.md`
  row. Those are human decisions.
- Do not edit `crates/lab` while it is grading the change (`run-experiment`
  refuses).

## Installing it

**Inside a TurboSpark checkout** (measures the tree you are editing):

```sh
cargo build --release -p turbospark-lab
scripts/install-tslab-skill.sh --repo
```

`--repo` symlinks `crates/lab/skill` into `.claude/skills/tslab` and
`.agents/skills/tslab`. Those directories are untracked, so each checkout or
worktree links them once.

**On a Mac without the source tree**, package the skill with its binaries on a
machine that has the checkout, then install the zip:

```sh
scripts/package-tslab-skill.sh                  # writes target/dist/tslab-skill-<rev>[-dirty]-macos-arm64.zip
scripts/install-tslab-skill.sh --zip target/dist/tslab-skill-<rev>-macos-arm64.zip
```

| Installer flag | Meaning |
| --- | --- |
| `--repo` | Link this checkout's skill |
| `--zip PATH` | Unpack a packaged zip into `DEST/tslab` |
| `--dest DIR` | Where `--zip` installs (default `~/.claude/skills`) |
| `--force` | Replace an existing install |

`package-tslab-skill.sh [--out DIR] [--no-build]` bundles `bin/tslab`,
`bin/tslab-worker`, the workloads and chats, the schema, and a `BUILD.txt`
with the source revision, the rustc version, and each binary's SHA-256. The
binaries are unsigned, so a downloaded zip arrives quarantined. `doctor.sh`
reports that and prints the `xattr` command. Clearing it is your call.

{% hint style="info" %}
The zip's binaries measure the build they came from. A result about a newer
build needs binaries built from that build. `BUILD.txt` says which one you
have.
{% endhint %}

## doctor.sh: can tslab run here?

```sh
crates/lab/skill/scripts/doctor.sh
```

It fixes nothing, it only reports. Required checks: macOS on Apple Silicon,
both binaries present and not quarantined, `tslab knobs` runs, the workloads
exist, and the results directory is writable. Optional checks: `xctrace` and
`python3`, needed only by `tslab trace`.

It honors `CARGO_TARGET_DIR`, `TURBOSPARK_LAB_DIR`, and `TURBOSPARK_HOME`.
Exit 0 when every required check passes, 1 otherwise. The last line is the
binary to use:

```
tslab skill at /Users/you/src/turbospark/crates/lab/skill
  ok    macOS on Apple Silicon
  ok    TurboSpark checkout at /Users/you/src/turbospark
  ok    tslab present
  ok    tslab-worker present
  ok    binaries not quarantined
  ok    tslab runs
  ok    workloads at /Users/you/src/turbospark/crates/lab/workloads
  ok    results directory /Users/you/.turbospark/lab
  ok    xctrace (tslab trace)
  ok    python3 (tslab trace)
use: TSLAB=/Users/you/src/turbospark/target/release/tslab
```

## preflight.sh: is the host quiet?

```sh
crates/lab/skill/scripts/preflight.sh
```

Run it before any measurement. It checks AC power, build and model processes
that would make runs dirty (`cargo`, `rustc`, `xctrace`, other engines such as
`ollama` or `mlx_lm`, another `tslab` or `tslab-worker`),
free disk against the 10 GiB guard default, swap, and who holds the machine
lock. `ready` means clean runs are possible. `not ready: runs would be dirty`
lists what to stop first:

```
tslab preflight
  ok    on AC power
  FAIL  busy processes: cargo x2 (finish or stop them)
  ok    free disk on /Users/you/.turbospark/lab: 81 GiB
  note  swap used: 6667.44M (nonzero swap means memory pressure, read timings with care)
not ready: runs would be dirty
```

A running `tslab serve` shows up as a busy `tslab` process. The dashboard is
read-only and never takes the lock, so check its command line before stopping
anything, and never stop another session's run.

## What degrades without a checkout

From the zip, `gate`, `run-experiment`, and `confirm` refuse, because they
diff and build the source. `corpus:N` prompts refuse, because they read the
repository's docs (use `protocol:` prompts). Run provenance records the
binary's SHA-256 instead of a git revision. Measurement, verdicts, kernel
benches, the dashboard, and workflows that only measure all work.

## Example prompts

Things you can ask an agent with the skill loaded, and what it should do:

| You ask | The agent runs |
| --- | --- |
| "Does `lfm3_fused_ffn` speed up decode on my LFM3 install?" | `preflight.sh`, then `ab --null --metric decode_tok_s`, then `ab --baseline default --candidate off:lfm3_fused_ffn=0 --metric decode_tok_s`, and reports the verdict with its ledger id |
| "Sweep expert cache slots 16, 32, 64 on Flint and tell me which wins" | `sweep --baseline ... --candidate s32@32/streamed --candidate s64@64/streamed`, then reads `best` (null if nothing won) |
| "What does mlx-serve pin for its verify matmul lanes?" | `pinned --repo mlx-serve --class ...`, citing the source line and revision |
| "My A/B exited 4. Revert the change?" | Refuses to revert, names the dirty reasons from the run records, and asks you to quiet the host and rerun |
| "I made GQA decode the default for my family. Is it still faster?" | Flips the A/B: baseline `default`, candidate `barrier:attn_decode_kernel=barrier`. A LOSS confirms the default |

## Evals

`crates/lab/skill/evals/evals.json` holds seven scenario evals, each a
`query`, the `files` it may read, and the `expected_behavior`. They cover the
traps above: a decode question without `--metric`, a request to revert on
exit 4, a gate request from the zip (which must refuse), pinned-setting
lookups, speculation depth, a slot sweep, and the flipped A/B after a default
changes. Use them when you change the skill's wording, to check an agent still
lands on the right command.

## Related pages

- [TurboSpark Lab (tslab)](tslab.md)
- [The tslab dashboard](tslab-serve.md)
- [lab (crate reference)](../crates/lab.md)
