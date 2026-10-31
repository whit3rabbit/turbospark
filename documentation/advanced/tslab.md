---
description: "TurboSpark Lab (tslab): measure and A/B knobs, kernels, and speculation settings on an Apple Silicon Mac, sweep a model family, price a kernel mode, audit the result ledgers, and browse them in the tslab serve dashboard."
icon: stopwatch
---

# TurboSpark Lab (tslab)

`tslab` is TurboSpark's evaluator for kernel and method experiments. It
measures an installed model on an Apple Silicon Mac and decides, under a
fixed verdict rule, whether a change actually paid: phase profiles of an
install, gated interleaved A/B of two arms, sweeps of several candidates
against one baseline, shape-derived kernel benches, kernel-mode pricing,
speculation replays, Metal System Traces, and resumable workflows. It works
on text installs of every model family and on audio installs. You pick the
command and read the result; tslab decides. A number is quotable only from a
clean, quotable run, with its ledger id and conditions.

Reach for tslab when you want to:

- profile an install (slots, residency, dispatch behavior) or trace its GPU work,
- A/B a knob (a `TURBOSPARK_*` switch), a kernel change, or a speculation setting,
- sweep a whole family's candidate settings in one interleaved session,
- price a kernel mode from bench curves before spending a model run on it,
- audit what the lab already measured (ledgers, verdicts, family findings),
- or run the autoresearch loop in `crates/lab/program.md`.

The harness lives in `crates/lab` (see [lab](../crates/lab.md) for the crate
reference). Everything tslab writes goes to one append-only ledger under the
lab dir, which `tslab serve` renders read-only.

## Getting tslab

tslab runs on macOS on Apple Silicon and comes two ways. The bundled skill's
`doctor.sh` finds whichever applies and prints the `TSLAB=...` path to use.

**From a checkout** (measures the tree you are editing):

```sh
cargo build --release -p turbospark-lab
TSLAB=target/release/tslab
```

Keep `tslab-worker` beside `tslab`: every measured run is a fresh worker
process, so both binaries must exist and stay in step.

**From the packaged skill zip**, for an agent or a second Mac without the
source tree:

```sh
scripts/package-tslab-skill.sh          # builds release binaries and zips the skill
scripts/install-tslab-skill.sh --zip target/dist/tslab-skill-<rev>-macos-arm64.zip
```

The zip unpacks to one `tslab/` skill folder with `bin/tslab` and
`bin/tslab-worker`, the workloads, and a `BUILD.txt` naming the source
revision and the binaries' SHA-256 hashes. The zip's binaries measure the
build they came from; a result about a newer build needs binaries built from
it. See [the agent skill](#the-bundled-agent-skill) below.

Below, `M` is a model (an install directory `*.gturbo`, or a catalog alias
from the model store) and `E` is an experiment name you choose.

## What tslab measures

tslab answers one kind of question: did this change make this install faster
(or smaller, or cheaper), and is the difference bigger than the noise on this
Mac? It measures at three levels.

**Whole-model runs.** Every `profile`, `ab`, and `sweep` arm is one fresh
`tslab-worker` process that opens the install, runs the workload, and writes a
run record. The metrics an A/B or sweep can judge:

| Metric (`--metric`) | What it is | Better |
| --- | --- | --- |
| `prefill_tok_s` | Prompt tokens per prefill wall second (the default) | higher |
| `decode_tok_s` | Decode tokens per decode wall second | higher |
| `prefill_s` | Prefill wall seconds | lower |
| `peak_bytes` | Peak `phys_footprint` of the worker | lower |
| `followup_ttft_s` | Mean time to first token on every chat turn after the first (chat workloads) | lower |
| `audio_run_s` | Wall seconds of the audio job | lower |
| `rtf` | Audio job wall time over the audio's duration (real-time factor) | lower |
| `prefill_j`, `decode_j` | Modeled joules over the measured prefill or decode (IOReport, whole machine) | lower |

The energy pair is modeled from IOReport counters for the whole machine. It is
comparable within one experiment and never a wall-power claim.

A run record also carries the phase buckets (on routed MoE installs: layer GPU
wait, router readback, expert I/O, bind, retire, final wait), the `engaged`
lanes the runtime reported, host samples taken during the run (foreign CPU,
power source), and provenance (git revision or binary SHA-256, chip, cargo
features, every `TURBOSPARK_*` variable set on the tslab process).

**Kernel benches.** `tslab kernel` times Metal kernels in process at the shapes
the install binds, read from its own manifest, never constants. Each case is
GPU time over M rows or over context, not throughput. `tslab kernel --list`
prints the registered benches:

| Bench | Measures |
| --- | --- |
| `int4_matmul` | INT4 projections: per-row GEMV, scalar GEMM, Steel tiles, the 8-row MMA tile, and the small-N route, over M |
| `int8_matmul` | INT8 projections (routers, shared gates): per-row GEMV against the batched GEMM over M |
| `attn_decode` | Decode attention per variant (full, sliding, sinks): barrier against GQA-shared over context |
| `gdn_delta` | Gated-DeltaNet delta kernels: decode, eager and lazy prefill, state fold, over rows |
| `rows_attn` | Multi-row prefill attention against the per-row GQA loop, per context |
| `moe_routed` | Routed-expert pair: M sequential decode pairs against one route-list pair |
| `qkv_concurrent` | The q/k/v projection triple: serial pass against a concurrent encoder |
| `verify_ladder` | Whole-model GEMM time per speculative verify width |
| `music3_grid` | Music 3 linear and conv corpus: output hashes against goldens, per-config timing |
| `whisper_linear` | Whisper encoder GEMM, decoder GEMV, GEMM at one row |
| `ane_matmul` | Steel against the Neural Engine per shape and token count, the row split, baked INT4 programs (`ane_c4`, `split_c4`), and modeled energy. Needs `--features ane-prefill` |
| `ane_handoff` | One ANE split point's host handoff (wake and resume), `wait` against `poll`, quiet and under `--contend N`. Needs `--features ane-prefill` |

**Offline analysis.** `project`, `calibrate`, `depth-replay`, and
`hybrid-replay` do no GPU work. They price a change from curves and captures
already in the ledger, so you can tell whether a model run is worth spending.

### The verdict rule

`ab` and `sweep` run baseline and candidate in interleaved pairs, so thermal
drift hits both arms. Only clean pairs count. The effect floor is the largest
of 2%, `--min-effect`, and the null floor measured by `ab --null` for that
workload and metric. The outcomes:

| Verdict | Means |
| --- | --- |
| `WIN` | At least 3 clean pairs, median at or above the floor, bootstrap 95% interval above zero, and every pair agreeing (or a sign test at p <= 0.05 from 6 pairs) |
| `LOSS` | The same, the other way |
| `NOISE` | Enough clean pairs, no effect at the floor. This is a result |
| `INCONCLUSIVE` | Too few clean pairs, usually a busy host. Fix the host and rerun |

The rule and its bounds are fixed in code (`src/stats.rs`, `src/gates.rs`).

## Examples

What can this install do, and what is already known about its family:

```sh
tslab capabilities --model lfm3-1.7b
tslab family --model lfm3-1.7b --findings
```

Profile prefill and decode on a long prompt:

```sh
tslab profile --model qwen36 --prompt protocol:long-synthesis --decode 64 --experiment qwen36-base
```

A/B a decode switch. Measure the noise floor first, with the same workload and
metric, then the pair. LFM3's fused FFN is on by default, so the candidate is
the kill switch:

```sh
W=(--prompt protocol:medium-review --decode 128)
tslab ab --model lfm3-1.7b --null $W --metric decode_tok_s --experiment lfm3-ffn
tslab ab --model lfm3-1.7b $W --metric decode_tok_s --experiment lfm3-ffn \
  --baseline default --candidate chain:lfm3_fused_ffn=0 --json
```

Run every sweep the family profile recommends, one verdict per candidate:

```sh
tslab sweep --model lfm3-1.7b --sweep all --experiment lfm3-sweep
```

Time the decode attention kernels at this install's shapes, with output
hashes, then check the GQA kernel is still the faster one end to end:

```sh
tslab kernel --model flint --bench attn_decode --hash --experiment flint-attn
tslab ab --model flint --prompt protocol:long-synthesis --decode 128 --metric decode_tok_s \
  --baseline default --candidate barrier:attn_decode_kernel=barrier --experiment flint-attn
```

Price a kernel mode before spending a model run on it, then check the price
against the verdict that followed:

```sh
tslab kernel --model qwen36 --bench int4_matmul --hash --experiment mma8
tslab profile --model qwen36 --prompt protocol:long-synthesis --decode 64 --arm base --experiment mma8
tslab project --knob int4_mma8 --curves mma8 --x 1 --base-from mma8:base --floor-from mma8
tslab calibrate --projection mma8 --verdict mma8
```

Write a self-contained HTML report, or open the dashboard:

```sh
tslab report --experiment lfm3-ffn --html lfm3-ffn.html
tslab serve --open
```

## Before you measure

Run the skill's `preflight.sh` (or read the host rules yourself): tslab
expects AC power, a quiet host, and free disk. It holds one machine lock
(a `flock` on `<lab>/.machine.lock`), so a second measuring session waits.
The shared host guard flags are:

```sh
[--max-foreign-cpu 150] [--max-single-cpu 90] [--allow-battery]
[--timeout-min 60] [--quiet-wait-min 5] [--min-free-gb 10]
```

`--quiet-wait-min 0` measures on a busy host; `--min-free-gb 0` skips the
disk check; `--allow-battery` permits battery runs that are ledgered dirty.
Dirty runs are recorded, never dropped, so the ledger keeps showing how
noisy the host was.

## The command tour

The authoritative list is `tslab` with no arguments (or `--help`). The
commands group by purpose.

### Measure

| Command | What it does |
| --- | --- |
| `tslab profile --model M [workload] --experiment E` | Phase profile of the install: per-arm medians over clean rounds. `--slots auto,16,32` and `--residency streamed,mapped` name the axes; `--arm NAME[@SLOTS/RESIDENCY][:KEY=V,...]` adds custom arms; `--rounds 3` sets round count. `--dispatch-probe 0,4,8,16` or `--standin attn,mlp,gdn` runs the dispatch-price probes, which are their own arms and need `--decode`. |
| `tslab ab --model M [workload] --baseline NAME[:KEY=V,...] --candidate NAME[:KEY=V,...]` | The gated interleaved A/B. `--null` first runs an arm against itself to measure the noise floor (once per workload); `--metric` picks one of the metrics [above](#what-tslab-measures); `--min-effect 0.02` is the effect floor; `--pairs 3 --max-pairs 6` bound the paired rounds. |
| `tslab sweep --model M --sweep ID\|all --experiment E` | Several candidates against one baseline in one interleaved session, one verdict per candidate plus a sweep record. `--sweep all` runs every profile sweep that applies from `tslab family`; `--sweep-file F` loads a file; or name arms directly with `--baseline A --candidate B [--candidate C]...`. `--fail-on-loss` turns a LOSS into a failing exit. |

### Kernel evidence

| Command | What it does |
| --- | --- |
| `tslab kernel --model M --bench NAME[,NAME]\|all --experiment E` | Shape-derived kernel benches sized from the install (never constants; the list is [above](#what-tslab-measures)). `--list` prints the registered benches; `--programs DIR`, `--ane-dir DIR`, `--contend N` feed the two ANE benches; `--rows` and `--contexts` pick cases; `--hash` records output hashes; `--verify-from EXP` replays a prior experiment's cases; `--reps 5` sets repetitions. |
| `tslab hash-diff --experiment A --against B` | Compares the hashed outputs of two kernel experiments. Exits 3 when any hashed output changed, so a correctness regression is a verdict, not a reading. |
| `tslab pinned [--repo kapitsa\|omlx\|mlx-serve] [--class C] [--id ID]` | The verified kernel settings pinned from the compared engines, each traceable to its source revision. `--dir PATH` sweeps a tweaked copy. |
| `tslab ane-plan --path PROGRAM.mlpackage\|MANIFEST_DIR [--units ane\|all\|gpu\|cpu]` | Core ML compute-plan placement of one program. |

### Pricing

| Command | What it does |
| --- | --- |
| `tslab project --knob NAME\|--bench B --curves EXP --base-ms X ...` | Prices a kernel mode from bench curves, counts, and a measured base before any model run; no GPU work. Alternatives: `--default-from`/`--mode-from`, `--round-from` for speculative rounds, `--floor-from` and `--share-from` for floors and block shares. |
| `tslab calibrate --projection EXP[:ID] --verdict EXP[:ID]` | Joins a projection to the A/B verdict that followed: predicted, measured, miss. `--measured X --source TEXT` records an external measurement; `--all` lists every miss and the bias. |

### Replays

| Command | What it does |
| --- | --- |
| `tslab tree-replay --model M --drafter mtp\|dflash` | Replays tree-verify acceptance against chain acceptance at equal verify rows. `--depth`, `--blocks`, `--serve-block`, `--top-k`, `--slots` shape the run. |
| `tslab chain-capture --model M [--tokens 512] [--depth 8]` | Records an MTP chain's per-position match length once, for depth replay. |
| `tslab depth-replay [--costs flag_off,flag_on,ledger:EXP[@FLAGS][#KV]] [--top 10]` | Offline: walks the runtime's own depth controller over captured chains and a round-cost table to price controller settings. |
| `tslab expert-census --model M [workload]` | One router capture run on a MoE install, reduced: how experts repeat within batches, the union per batch against slot counts, and the routed sub-batch replay. |

### Introspection

| Command | What it does |
| --- | --- |
| `tslab capabilities --model M` | What tslab can measure on this install: tags, and what to set first. Start here. |
| `tslab family [--model M \| --family F]` | The family profile: recommended settings, the sweeps to run first, open questions, traps. `--findings` lists recorded findings with their ledger id or doc heading; `--sweeps` lists the sweeps `tslab sweep --sweep` consumes. |
| `tslab inventory --model M` | The install's weight shapes by dtype. |
| `tslab knobs` | The switches an arm can set (`KEY=V` short names plus their `TURBOSPARK_*` variables), and below them the kernel-mode lanes with the bench that prices each. |
| `tslab schema` | The JSON Schema of everything tslab writes (`tslab/1`). |

### Records

| Command | What it does |
| --- | --- |
| `tslab ledger --experiment NAME [--kind run\|verdict\|...]` | Prints one experiment's ledger lines, optionally one kind. |
| `tslab export --experiment A[,B]\|--all [--kind K[,K]] [--format jsonl\|json\|csv] [--redact-paths] [--out PATH [--force]]` | Ledgers as `tslab/1` series for plotting or handoff. |
| `tslab report --experiment A[,B]\|--all --html OUT.html` | The offline HTML report (charts per experiment). `--title`, `--kind`, `--redact-paths`, `--force` adjust it. |

### Automation

| Command | What it does |
| --- | --- |
| `tslab workflow NAME\|--plan PLAN.json [--var KEY=V]... [--dry-run]` | A plan whose steps are tslab commands, nothing more. `--list` and
`--print NAME` show the built-ins (`new-family`, `family-sweep`,
`kernel-candidate`, `mode-candidate`, `spec-depth`, `audio-survey`,
`ane-sweep`, `report`); `--step ID`, `--from ID`, and `--resume RUN` re-enter
a run. |
| `tslab gate --spec SPEC.json [--base REV]` | The quality gate for a spec. Loads the model, so it takes minutes. |
| `tslab run-experiment --spec SPEC.json [--base REV]` | The autoresearch loop step: runs the spec and grades the tree. Refuses to grade a tree that changes `crates/lab` itself. |
| `tslab confirm --spec SPEC.json [--base REV]` | Re-checks a kept result on the held-out workload before it is treated as settled. |

`--base REV` names the commit an experiment's diff is checked against
(default `main`). An experiment tree that merges in the lab itself needs
`--base <that merge commit>`.

### Observability

| Command | What it does |
| --- | --- |
| `tslab trace --model M [workload] [--arm NAME[:KEY=V,...]] [--record-s 10]` | A Metal System Trace of the arm's measured work (needs `xctrace` and `python3`). Trace output is never quotable: it explains, it does not decide. |

### Workloads

Measurement commands share a workload vocabulary:

```sh
[--prompt protocol:ID|messages:PATH|text:PATH|corpus:N|chat:PATH]...
[--decode N] [--chunk N|off] [--context N]
# or, for an audio install:
--audio-job JOB.json        # see crates/lab/workloads/audio/
```

`protocol:ID` prompts are the frozen text prompts; `corpus:N` prompts are
built from the repository's docs, so they need a checkout (the packaged zip
ships `protocol:` and `chat:` jobs in its `workloads/` and `chats/`).

## The `--json` envelope and exit codes

Any command accepts `--json`, anywhere on the line. Stdout is then exactly
one `tslab/1` envelope; tables and progress go to stderr:

`{schema, command, ok, exit_code, decision?, records, result, warnings, error?}`

`decision` appears only for `run-experiment` and `confirm` (keep, discard,
inconclusive). `records` names the ledger lines the invocation appended, so
a script can cite results by ledger id. `error` is present on failure.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | ok; for `run-experiment` and `confirm`: keep |
| `1` | error |
| `2` | usage error |
| `3` | discard (also `hash-diff` when a hashed output changed) |
| `4` | inconclusive: do not revert. Usually the host was busy; fix that and rerun the same command |

{% hint style="warning" %}
Exit 4 is not a negative result and never justifies reverting a change, and
a NOISE verdict is a result: do not rerun it until it wins. The verdict rule
and gate bounds are fixed in code on purpose; loosening one is a human
decision with its own evidence.
{% endhint %}

## Environment variables

**Where results live.** The lab dir resolves in this order:

1. `$TURBOSPARK_LAB_DIR` if set,
2. else `$TURBOSPARK_HOME/lab`,
3. else `~/.turbospark/lab`.

Both variables are passed through to every worker process, and the bundled
skill's `doctor.sh` and `preflight.sh` honor them too. They are listed in
the [Environment Variables Reference](environment-variables.md).

**How arms set knobs.** An arm is `NAME[:KEY=V,...]` on `--arm`,
`--baseline`, or `--candidate`, and `--env KEY=V` adds variables to every
arm. `KEY` is a `TURBOSPARK_*` name or a knob short name (`tslab knobs`
lists both). Every run is a fresh worker process with a cleared
environment, so a knob cached at process start still switches cleanly
between arms.

**What tslab sets itself.** The orchestrator clears the worker's environment
and then adds only `PATH`, `HOME`, `TMPDIR`, `USER`, `LANG`, `LC_ALL`, the
two results variables above, the arm's variables, and
`TURBOSPARK_ENGAGEMENT=1`. That last one is a probe: the runtime prints an
engagement line wherever an opt-in lane runs only when it is set, so a run
whose stderr contradicts its arm's switches is caught as ineligible rather
than measured as its opposite.

## The dashboard: `tslab serve`

```sh
tslab serve --open
```

A read-only local web server over every ledger in the lab dir: the experiments
table, one page per experiment with its charts, a compare view, the verdict
board, sweeps, family profiles, and a GET-only JSON API. It never takes the
machine lock, so it stays open beside a measurement. Pages, API routes,
examples, and the security guards are on
[The tslab dashboard](tslab-serve.md).

## The bundled agent skill

`crates/lab/skill/` teaches a coding agent (Claude Code, Codex, Antigravity) to
drive tslab: which command answers which question, the `--json` contract, the
standard A/B checklist, and what may be quoted. `skill/scripts/doctor.sh`
checks tslab can run here and `skill/scripts/preflight.sh` checks the host is
quiet. Install it with `scripts/install-tslab-skill.sh --repo` in a checkout,
or `--zip` from a packaged zip. Details, example prompts, and the evals are on
[The tslab agent skill](tslab-skill.md).

## Where results live

Everything lands under the lab dir (`$TURBOSPARK_LAB_DIR`, else
`$TURBOSPARK_HOME/lab`, else `~/.turbospark/lab`):

```
~/.turbospark/lab/
  .machine.lock                  the one-session machine lock (flock)
  <experiment>/                  one directory per experiment name
    ledger.jsonl                 the append-only JSON-lines ledger
    runs/<run_id>.json           one worker record per run
    runs/<run_id>.request.json   the worker's request
    runs/<run_id>.stderr         the worker's log
```

Experiment and arm names are restricted to 1 to 64 bytes of
`[A-Za-z0-9._-]` and cannot start with a dot, because the name becomes a
path component. Every ledger line carries a time-ordered record id
(`<unix_ms>-<pid>-<n>`), a `kind` (`run`, `verdict`, `sweep`, `kernel_curve`,
`projection`, `calibration`, and the rest, all listed by `tslab schema`), and
its payload. Lines are never rewritten: a correction is a new line that
names the one it supersedes, and dirty runs are ledgered too, so the ledger
always shows the conditions a number was measured under.

## Related pages

- [The tslab dashboard](tslab-serve.md)
- [The tslab agent skill](tslab-skill.md)
- [Environment Variables Reference](environment-variables.md)
- [lab (crate reference)](../crates/lab.md)
- [bench (crate reference)](../crates/bench.md)
- [Benchmarks and Performance](../benchmarks/performance.md)
