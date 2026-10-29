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
| `tslab ab --model M [workload] --baseline NAME[:KEY=V,...] --candidate NAME[:KEY=V,...]` | The gated interleaved A/B. `--null` first runs an arm against itself to measure the noise floor (once per workload); `--metric prefill_tok_s\|decode_tok_s\|prefill_s\|peak_bytes\|followup_ttft_s\|audio_run_s\|rtf` picks the metric; `--min-effect 0.02` is the effect floor; `--pairs 3 --max-pairs 6` bound the paired rounds. |
| `tslab sweep --model M --sweep ID\|all --experiment E` | Several candidates against one baseline in one interleaved session, one verdict per candidate plus a sweep record. `--sweep all` runs every profile sweep that applies from `tslab family`; `--sweep-file F` loads a file; or name arms directly with `--baseline A --candidate B [--candidate C]...`. `--fail-on-loss` turns a LOSS into a failing exit. |

### Kernel evidence

| Command | What it does |
| --- | --- |
| `tslab kernel --model M --bench NAME[,NAME]\|all --experiment E` | Shape-derived kernel benches sized from the install (never constants). `--list` prints the registered benches; `--rows` and `--contexts` pick cases; `--hash` records output hashes; `--verify-from EXP` replays a prior experiment's cases; `--reps 5` sets repetitions. |
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
tslab serve [--port 7878] [--bind 127.0.0.1] [--lab DIR] [--open]
```

A read-only dashboard over every ledger in the lab dir: the experiments
table, per-experiment pages with the report charts, a compare view that puts
several experiments' series on shared axes, the verdict board, the sweeps,
each family's profile over what this lab has measured for it, and a JSON
API. It never takes the machine lock and writes nothing, so it can stay open
beside a measurement, and the index re-reads automatically when a ledger's
size or modification time moves. Ctrl-C stops it; `--open` opens a browser.

**Pages** (server-rendered, nothing fetched from anywhere):

| Path | Page |
| --- | --- |
| `/` | Experiments table with filter and sort forms, plus the compare tray: tick experiments and compare their series |
| `/experiment/{name}` | One experiment: summary, notes, verdicts, sweeps, and the report charts |
| `/compare?exp=a,b` | Shared-axes series for the named experiments |
| `/verdicts` | The verdict board with `family`, `model`, `metric`, `exp` filters |
| `/sweeps` | Sweep results with `family` and `model` filters |
| `/families` | Every family profile beside this lab's live data for it |
| `/family/{family}` | One profile: settings, sweeps, findings, and its verdicts |
| `/record/{name}/{id}` | One ledger record rendered from its JSON |

**JSON API** (GET only; every response is a `tslab/1` envelope, and errors
come back as `{"schema": "tslab/1", "ok": false, "error": ...}`):

| Route | Returns |
| --- | --- |
| `GET /api/experiments` | The lab dir, every experiment's summary, and per-kind totals |
| `GET /api/experiment/{name}` | One experiment's summary, notes, verdicts, sweeps, and profiles |
| `GET /api/ledger/{name}` | Every record line of one experiment |
| `GET /api/series?exp=a,b[&kind=K]` | Plottable series; `exp` is required, `kind` filters |
| `GET /api/verdicts[?family=&model=&metric=&exp=a,b]` | Verdict rows under those filters |
| `GET /api/sweeps[?family=&model=]` | Sweep rows under those filters |
| `GET /api/family/{family}` | The embedded profile plus that family's verdicts, sweeps, and profiles |
| `GET /api/record/{name}/{id}` | One record; ids are `<unix_ms>-<pid>-<n>` |

{% hint style="info" %}
Every page fetches nothing and its single inline script only adds hover
tooltips, legend toggles, and the compare tray count, so a page saved to
disk reads the same as the live one. There is no authentication: on the
default loopback bind, a `Host` header that does not name this machine is
refused, and binding a non-loopback address prints a warning that the Host
check is off.
{% endhint %}

## The bundled agent skill

`crates/lab/skill/` is an agent skill (its `SKILL.md` is the entry point)
that teaches an agent to drive tslab: the `--json` contract, the standard
A/B loop, what may be quoted, and references for commands, output, reading
results, families, workflows, and extending the lab. Two scripts guard it:

- `skill/scripts/doctor.sh` checks that tslab can run here and says which
  binary applies: the zip's `bin/tslab` or a checkout's release build (it
  honors `CARGO_TARGET_DIR`). It verifies macOS on Apple Silicon, both
  binaries, Gatekeeper quarantine, that `tslab knobs` runs, the workloads, a
  writable results directory, and the optional `xctrace` and `python3` that
  only `tslab trace` needs. Exit 0 when every required check passes, 1
  otherwise.
- `skill/scripts/preflight.sh` checks that the host is quiet enough for
  clean runs before you spend one: AC power, build and model processes, free
  disk against the 10 GiB guard default, swap, and who holds the machine
  lock. Exit 0 means ready; 1 means runs would be ledgered dirty.

Install it with `scripts/install-tslab-skill.sh`:

- `--repo` links this checkout's skill into `.claude/skills/tslab` and
  `.agents/skills/tslab`. Those directories are untracked, so each checkout
  or worktree links them once. Inside a checkout prefer this: it measures
  the tree you are editing.
- `--zip PATH` unpacks a packaged zip into `DEST/tslab` (`--dest` changes
  the destination, default `~/.claude/skills`; `--force` replaces).

`scripts/package-tslab-skill.sh [--out DIR] [--no-build]` builds the release
binaries and zips the skill with them, the workloads, the schema, and a
`BUILD.txt` (source revision, rustc version, binary SHA-256 hashes) into
`target/dist/tslab-skill-<rev>[-dirty]-macos-arm64.zip`. The binaries are
unsigned, so a downloaded zip is quarantined; `doctor.sh` reports it and
says the `xattr` command, and clearing it is the person's call.

**What degrades without a checkout** (from the zip): `gate`, `run-experiment`,
and `confirm` refuse to run, because they diff and build the source;
`corpus:N` prompts refuse, because they read the repository's docs (use
`protocol:` prompts); and run provenance records the binary's SHA-256
instead of a git revision. Everything else, including measurement, verdicts,
the dashboard, and workflows that only measure, works.

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

- [Environment Variables Reference](environment-variables.md)
- [lab (crate reference)](../crates/lab.md)
- [bench (crate reference)](../crates/bench.md)
- [Benchmarks and Performance](../benchmarks/performance.md)
