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
| Exit codes, including the commands that use 1 or 3 for a failed check or a changed output | `SKILL.md`, `references/output.md` |
| Every command, its flags, what it writes, and its traps | `references/commands.md` |
| Output and correctness checks beside the A/B: `sanity`, `noise`, `floor`, `anchor`, `quant-ab`, `release-gate`, `idle-gap`, `spec-rounds`, `image-stage` | `references/commands.md` |
| Kernel benches, including `attn_decode` with TurboQuant arms, `attn_decode_gemma4`, `moe_routed --routes`, `verify_ladder` and the untimed `gdn_trajectory` route check | `references/commands.md` |
| The HTTP path: `serve-bench` and `soak` against a loopback server | `references/commands.md` |
| Cross-engine KL against llama.cpp on the same GGUF (`xengine-kl`, checkout only) | `references/commands.md` |
| The `--json` envelope, ledger records, `BenchResult`, exported series, methodology ids | `references/output.md` |
| Verdicts, dirty runs (including the opt-in memory and power rules), sweeps, the compare view, and what may be quoted | `references/reading-results.md` |
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
- Read `ok` in the envelope before the exit code. `sanity`, `quant-ab` and
  `release-gate` exit 1 with `ok: true` when a check failed, which is a
  finding, not a broken tool.
- On a family nobody has measured, run `sanity` before the first sweep. A
  forward pass that decodes garbage can still run fast, and on a MoE install
  degenerate routing can look like a win.
- A server claim comes from `serve-bench`, not `ab`, and its rows are not a
  paired verdict.
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
anything, and never stop another session's run. A `turbospark-server` also
shows up as busy: when it is the server a `serve-bench` targets, that is
expected.

`preflight.sh` has no CPU-load check. A clean preflight does not rule out runs
the guard later rejects for sustained foreign load, and builds that start
mid-run still make pairs dirty.

## What degrades without a checkout

From the zip, `gate`, `run-experiment`, and `confirm` refuse, because they
diff and build the source. `xengine-kl` needs the tree too (it runs
`cargo test` and `scripts/kld_llamacpp.py`) and `uv`. `image-stage` needs
fixtures from `scripts/z_image_capture.py`. `corpus:N` prompts refuse,
because they read the repository's docs (use `protocol:` prompts), and so
does the `fill-ladder` workflow, which uses them. Run provenance records the
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
| "New MoE family, run the slot sweep and tell me what to ship" | `sanity` first, then preflight, null and the sweep. Reads a near-99% expert hit rate as a possible routing symptom and reports `best` (or none) without flipping a default |
| "`sanity` exited 1. Is tslab broken?" | Reads `ok: true` as a failed check, names the failing cases from the `sanity` record, and does not rerun until it passes |
| "Is my server faster under concurrent load?" | `serve-bench --server http://127.0.0.1:PORT` against each server under the same flags, not `ab`. Reports exit 3 as a failed request to investigate |

## Evals

`crates/lab/skill/evals/evals.json` holds ten scenario evals, each a
`query`, the `files` it may read, and the `expected_behavior`. They cover the
traps above: a decode question without `--metric`, a request to revert on
exit 4, a gate request from the zip (which must refuse), pinned-setting
lookups, speculation depth, a slot sweep, the flipped A/B after a default
changes, a first sweep on an unmeasured family (sanity first), a `sanity`
exit 1 read as a finding rather than a tool error, and a server claim that
needs `serve-bench` rather than `ab`. Use them when you change the skill's
wording, to check an agent still lands on the right command.

## Related pages

- [TurboSpark Lab (tslab)](tslab.md)
- [The tslab dashboard](tslab-serve.md)
- [lab (crate reference)](../crates/lab.md)
