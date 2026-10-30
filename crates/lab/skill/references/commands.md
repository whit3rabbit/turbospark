# tslab commands

`tslab --help` prints the exact flags of the build you have. This page says
what each command is for, what it writes, and the traps.

Every command takes `--json` anywhere on the line (see `output.md`).
Measuring commands take the machine lock, record every run (dirty ones
too) in the experiment's ledger, and give each arm a fresh `tslab-worker`
process, which must sit beside `tslab`.

## Orientation (no GPU work)

| Command | Answers | Writes |
|---|---|---|
| `capabilities --model M` | What tslab can measure on this install, and why not the rest. `result.tags` drive workflow `when` clauses; `result.recommended` and `result.sweeps` are the family profile's entries that apply | nothing |
| `family [--model M \| --family F] [--findings] [--sweeps] [--dir DIR]` | The family profile (`crates/lab/families/`): recommended settings with why, the sweeps, the findings with ledger or doc provenance, open questions, traps. With `--model`, filtered to what applies and findings marked `same_install`. No argument lists every profile. `--dir` overlays edited copies | nothing |
| `inventory --model M` | Every weight shape the install binds, by dtype, with per-forward counts | nothing |
| `knobs` | Runtime switches tslab knows: env name, values, exactness, quotable, engagement lines; then the kernel modes (lanes): the bench that prices each and how it is compared | nothing |
| `pinned [--repo kapitsa\|omlx\|mlx-serve] [--class C] [--id ID] [--dir PATH]` | The verified kernel settings pinned from the engines this repo compares against, each traced to a source line and revision. The baseline is embedded in the binary. `--dir` layers edited copies over it for a sweep (a colliding `id` replaces the baseline) | nothing |
| `kernel --list` | Registered kernel benches | nothing |
| `serve [--port 7878] [--bind 127.0.0.1] [--lab DIR] [--open]` | The read-only dashboard over every ledger in the lab dir: the experiment table with filters and a compare form, one page per experiment with its charts, runs and notes, `/compare?exp=a,b` with the experiments' series regrouped onto shared axes and a verdict board, `/verdicts`, `/sweeps`, `/families` and `/family/<f>` (the profile over what the lab measured for it), `/record/<exp>/<id>`, and a JSON API under `/api`. Never takes the machine lock, so it stays open beside a measurement. Every response is `nosniff` plus a `default-src 'none'` CSP, and a loopback bind answers only loopback Host headers (the DNS-rebind guard); an off-loopback `--bind` answers any Host and prints a warning. Blocks until Ctrl-C | nothing |
| `schema` | JSON Schema of everything tslab writes | nothing |
| `ledger --experiment E [--kind K]` | The raw records of one experiment | nothing |

`--model` takes an install directory or a catalog alias. Text installs have a
`manifest.json`, audio installs a `config.json`.

## Measuring a model

| Command | Use | Writes (`kind`) |
|---|---|---|
| `profile --model M --prompt protocol:long-synthesis --decode 64` | Where a pass spends time: prefill and decode tok/s, phase buckets, peak footprint | `run`, `profile_summary` |
| `profile ... --arm A --arm B:KNOB=V` | Several arms in rotated rounds, no verdict | same |
| `profile ... --dispatch-probe 0,4,8,16` | Price of one chained dispatch (dense qwen and llama trunks); the arms price decode, so give `--decode 8` or more | adds `dispatch_probe` |
| `profile ... --standin attn,mlp,gdn` | Each block's in-place cost (dense qwen and llama trunks); also needs `--decode 8` or more | adds `standin` |
| `ab --model M --null` | This host's noise floor for a workload and metric. Run once per workload before any `ab` | `null_calibration` |
| `ab --model M --baseline off --candidate on:KNOB=V` | Does a switch win? Paired, interleaved, with a fixed verdict rule | `run`, `verdict` |
| `sweep --model M --sweep ID\|all` | Every candidate of a profile sweep against its baseline, each by the same pair rule as `ab`, interleaved in one session; `--sweep-file F` or `--baseline A --candidate B --candidate C` for an ad hoc one. Workload flags override the sweep's workload. Exit 4 if any candidate is inconclusive, 3 only with `--fail-on-loss`. `--gate` runs the quality gate on each WIN at the sweep's declared exactness and `--enforce-memory` holds each WIN to the memory allowance; a WIN that fails either keeps its verdict but gets `rejected` and is never `best` | `run`, one `verdict` per candidate, `sweep` |
| `trace --model M --prompt ... --decode 64` | Metal System Trace of the measured work: GPU busy, idle gaps, per-kernel shares | `trace` |
| `tree-replay --model M --drafter mtp\|dflash` | Would tree verify accept more than the chain at equal rows | `tree_replay` |
| `chain-capture` then `depth-replay` | Best speculative depth policy, priced from modeled or ledger round costs | `chain_capture`, `depth_replay` |
| `chain-capture` then `hybrid-replay [--ngram 4] [--lookup-draft 8] [--tuned 3,1,1,2] [--cap 7]` | Would prompt lookup over the MTP chain commit more per round than the head alone, against fixed 2, the shipped controller and a tuned one. Needs a capture written with the token ids (this build's `chain-capture`); older captures are refused with that reason. Lookup rounds are priced at the MTP table's round of the same width, so the tok/s delta is conservative; the committed-per-round delta needs no cost table | `hybrid_replay` |
| `ane-plan --path PROGRAM.mlpackage\|DIR [--units ane\|all\|gpu\|cpu]` | Where Core ML places a program's ops (`MLComputePlan`), as cost shares on ANE, GPU and CPU plus the ops left off the ANE. No timing and no model install. Needs a macOS with compute plans | `ane_plan` |
| `expert-census --model M [workload]` | One router capture run of a MoE install, reduced: the share of selections in experts used 8/16/17/32+ times within an executed batch, the largest in-batch group, experts per layer covering 50/80/95% of selections, the expert union per full batch against 16/32/64/96 slots, and the routed sub-batch sizes per union cap. The capture run itself is unquotable | `run`, `expert_census` |

Workload flags: `--prompt protocol:ID|messages:PATH|text:PATH|corpus:N|chat:PATH`
(repeatable), `--decode N`, `--chunk N|off`, `--context N`. An audio install
takes `--audio-job JOB.json` instead (`workloads/audio/`), and its metrics are
`audio_run_s` and `rtf`.

Arm syntax: `NAME[@SLOTS/RESIDENCY][~DRAFTER:BLOCK[a]][~hog:N][:KEY=V,...]`,
for example `mtp2~mtp:2:spec_stats=1`. `KEY` is a `TURBOSPARK_*` name or a
knob short name. `~hog:N` runs N busy threads beside the worker for the
whole run (controlled contention, never quotable); its run records keep
`controlled_contention` and, with `ane_prefill_stats=1`, `ane_stats` with
the handoff share.

Metrics: `prefill_tok_s`, `decode_tok_s`, `prefill_s`, `peak_bytes`,
`followup_ttft_s`, `audio_run_s`, `rtf`, and the modeled-energy pair
`prefill_j`, `decode_j` (IOReport, whole machine, comparable within one
experiment; never a wall-power claim).

Host flags: `--max-foreign-cpu 150`, `--max-single-cpu 90`, `--allow-battery`,
`--quiet-wait-min 5`, `--min-free-gb 10`, `--timeout-min 60`. Loosening them
gets you numbers from a busy host, which is all they then are.

## Kernel benches (in process, no model weights)

`kernel --model M [--bench NAME[,NAME]|all] [--rows ...] [--contexts ...] [--case LABEL] [--reps 5] [--hash] [--programs DIR] [--ane-dir DIR] [--contend N]`

Every case records an `energy` delta over its run when the host has the
IOReport counters. `--bench ane_matmul` (a `tslab` built with
`--features ane-prefill`) needs `--programs DIR` from
`crates/coreml/tools/gen_ane_sweep_programs.py` and prints the exact
generator line for the install's shapes when they are missing: per shape
and token count it times Steel, the ANE program (conversions and handoff
included) and the row split at several shares, records where Core ML
planned each program, puts joules on each point, and fits the `ane gate`
thresholds in its notes. Extra arms: `ane_pred_x1` and `ane_pred_x2` (raw
predictions on one model and on two models driven by two threads: does a
second prediction in flight overlap), `tsplit` (the ANE takes the first half
of the tokens through every row), and, with `--const-int4` programs, `ane_c4`
and `split_c4@S` (weights baked into the program as uint4). `--case
real:LAYER.NAME` runs Steel, the F16-plane split and the baked split on one
tensor's own bytes from the install, with programs from
`crates/coreml/tools/bake_ane_prefill_programs.py`, and prints each ANE row
set's error against Steel's output.

| Bench | Measures |
|---|---|
| `int4_matmul` | Per-row GEMV, scalar GEMM, Steel tiles, the 8-row MMA tile and the small-N prefill route (`small_n`, only on shapes too narrow for a whole Steel tile, such as the 48-row GDN gates) over M, at every INT4 shape the install binds |
| `int8_matmul` | INT8 GEMV against the batched GEMM (routers, shared gates, INT8 installs) |
| `attn_decode` | Barrier against GQA-shared decode attention over context, per variant (full, sliding, sinks) |
| `rows_attn` | Multi-row prefill attention against the per-row loop, with layout arms, one case per context |
| `moe_routed` | Routed-expert pair: M decode pairs against one route-list pair (union modeled as uniform routing) |
| `verify_ladder` | Whole-model GEMM cost per verify width; `--verify-from EXP` adds the measured non-GEMM marginal |
| `qkv_concurrent` | The attention q/k/v triple in a serial pass against a concurrent compute encoder (shared input, no barrier); `c(M)` is each arm's time over the serial arm's |
| `gdn_delta` | Gated-DeltaNet delta kernels (decode, eager and lazy prefill, state fold) at the install's geometry. One build per process: put `TURBOSPARK_GDN_DK128=0` on the `tslab` process for the runtime-Dk build |
| `music3_grid` | Music 3 corpus: output hashes against goldens, per-config timing |
| `whisper_linear` | Whisper encoder GEMM, decoder GEMV, GEMM at one row |
| `ane_handoff` | Host handoff of one ANE prefill split point, `wait` against `poll`, quiet and under `--contend N` self-made CPU load. Needs a build with `--features ane-prefill`, a dense qwen install and its program set (`--ane-dir`, else the runtime's search list) |

`--hash` stores each point's output-bit hash and the hash of the first output
row alone, and prints an exactness hint per arm: `bit-equal` or `DIFFERS`
against the first arm, and whether the first row is `invariant` or `VARIES`
as the batch grows. A tile that DIFFERS but is row-invariant is the
shared-arithmetic signature (decode and verify can share it, frozen digests
re-freeze); one that DIFFERS and VARIES is a drift bound. These are hints
from one seed, never a gate: the parity tests are the gate. `hash-diff
--experiment A --against B` compares two hashed runs (two builds) and exits 3
if any output changed.

## Pricing a kernel mode (no GPU work)

| Command | Answers | Writes (`kind`) |
|---|---|---|
| `project --knob K --curves EXP --x N --base-from EXP:ARM` | What a mode should do to a decode token, prefill token or round, from kernel curves, per-forward counts and a measured base. Prints the per-shape table, `coverage` (the share of the base the benched kernels explain) and whether the effect clears the noise floor | `projection` |
| `project --knob K --default-from A --mode-from B ...` | The same for a switch the gpu crate reads once per process (`gdn_dk128`): one arm timed by two `tslab kernel` processes | `projection` |
| `calibrate --projection EXP --verdict EXP` | Predicted against measured, and whether it is a miss (wrong sign, or off by more than the floor) | `calibration` |
| `calibrate --measured X --source TEXT ...` | The same for a number read off a doc row, with where it came from | `calibration` |
| `calibrate --all` | Every calibration in the lab dir and the mean signed error per bench and arm pair | nothing |

Base: `--base-ms X`, `--base-from EXP[:ARM]` (a `profile_summary`, decode or
`--base-metric prefill_tok_s`), or `--round-from EXP --width W [--flags L]
[--kv N] [--phase verify]` (`round_cost` records). Add `--dispatches N
--probe-from EXP` for a mode that changes the dispatch count, `--floor-from
EXP` for the host's null floor, `--share-from EXP --block gdn|mlp|attn` to
read the benched kernels against a stand-in block. `--min-rows N` leaves
shapes under the mode's routing threshold on the default (the 8-row tile
serves 4096 rows and up), and `--shapes` prices only the named cases. `--counts LABEL=N,...` overrides a
case's recorded per-forward count (a routed expert shape that collides with a
shared expert's in the resident index records the shared one), and `--scale S`
multiplies every count (layers, or routes per row group). Read `coverage` first: a
projection that explains 40% of the base says nothing about the other 60%,
and the 8-row tile's lever 9 was exactly that.

## Results out

| Command | Output |
|---|---|
| `export --experiment E[,F]\|--all [--format jsonl\|json\|csv] [--redact-paths] [--out P]` | `tslab/1` series, one per plottable line or bar group |
| `report --experiment E --html OUT.html` | One self-contained HTML page with charts and tables |

## Running several commands at once

`workflow NAME|--plan PLAN.json [--model M] [--experiment E] [--var KEY=V]... [--step ID] [--from ID] [--resume RUN] [--dry-run]` runs several of these commands in order as one resumable, lock-holding run (`--list` and `--print NAME` need no model). It writes `workflow_step` and `workflow_run`. The built-in plans: [workflows.md](workflows.md).

## Experiments against a code change (repository only)

`gate`, `run-experiment` and `confirm` take a spec file, check the git diff
against `--base`, build the worker from the tree, run required tests, the
quality gate, the paired A/B and the memory check, and print a `DECISION`.
They need the TurboSpark source tree, and outside it they refuse. The procedure
is `crates/lab/program.md`.

## Traps

- `ab` and an ad hoc `sweep` default to `--metric prefill_tok_s` (a profile
  sweep carries its own metric). A decode claim needs
  `--metric decode_tok_s` on the null run and the A/B alike; the header line
  says which metric ran. Under `--chunk off` the prefill metric times the
  same per-token step as decode, so it moves in the same direction but by a
  different amount, and it is still not a decode number.
- After a switch becomes the default, the A/B flips: baseline `default`,
  candidate the old value (`barrier:attn_decode_kernel=barrier`). A LOSS then
  confirms the default and a KEEP means a regression. Flip the family
  profile's sweep the same way.
- Prove each arm ran its lane from the ledger, not the arm name: every run
  record carries `engaged` (the `[engaged]` lines). A default-on lane shows
  on the baseline with no env set, and a kill switch's arm must lack it.
- A rebase or merge can bring a new default route that engages on your arm
  (`prefill_small_n` reached the lfm3 chunk driver this way). Read the
  `engaged` list again and re-run the KL probe before carrying old numbers
  across the change.
- `preflight.sh` lists a `tslab serve` as busy. The dashboard is read-only
  and never takes the lock; check its command line before stopping anything,
  and never stop another session's process.
- Installs outside the catalog store (`~/models/...`) need the path for
  `--model`, not an alias.
- `ane-plan` says where ops run, not how fast. "Supported on the ANE" is not
  "placed there": read the cost share, and measure the ANE+Steel split as an
  ordinary `ab` arm (`ane_prefill` knob) on a worker built with `ane-prefill`.
- A forced-kernel variable such as `TURBOSPARK_ATTN_DECODE_KERNEL=barrier`
  changes more than the arm you meant (the trunk's verify too). The baseline
  is the variable unset.
- Probe, stand-in, trace and dispatch-profile runs are never quotable.
- One measuring session per machine. If the lock is held, wait.
- Under `int4_mma8` an adaptive arm (`~mtp:Na`) with N above 7 runs at cap
  7: the runtime holds the cap to the 8-row tile. The record still says
  block N, so compare caps above 7 with the tile off, or not at all.
- A cold page cache looks like a slow kernel. Let the warmup round absorb it,
  and check `sysctl vm.swapusage` before timing.
- A pass started on a cold device runs several times slower for its whole
  length, and only tens of milliseconds of work queued ahead of it bring the
  device up (1024 or 4096 tiny dispatches did not, 16384 did). `int4_matmul`
  commits a primer of that size right before every timed pass;
  `subject.primer.gpu_ms_median` on the record should read tens of ms. A
  bench of your own that times tiny passes with host work between them
  measures the device's state, not the kernel.
