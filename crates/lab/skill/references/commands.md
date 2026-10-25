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
| `capabilities --model M` | What tslab can measure on this install, and why not the rest. `result.tags` drive workflow `when` clauses | nothing |
| `inventory --model M` | Every weight shape the install binds, by dtype, with per-forward counts | nothing |
| `knobs` | Runtime switches tslab knows: env name, values, exactness, quotable, engagement lines; then the kernel modes (lanes): the bench that prices each and how it is compared | nothing |
| `kernel --list` | Registered kernel benches | nothing |
| `schema` | JSON Schema of everything tslab writes | nothing |
| `ledger --experiment E [--kind K]` | The raw records of one experiment | nothing |

`--model` takes an install directory or a catalog alias. Text installs have a
`manifest.json`, audio installs a `config.json`.

## Measuring a model

| Command | Use | Writes (`kind`) |
|---|---|---|
| `profile --model M --prompt protocol:long-synthesis --decode 64` | Where a pass spends time: prefill and decode tok/s, phase buckets, peak footprint | `run`, `profile_summary` |
| `profile ... --arm A --arm B:KNOB=V` | Several arms in rotated rounds, no verdict | same |
| `profile ... --dispatch-probe 0,4,8,16` | Price of one chained dispatch (dense qwen and llama trunks) | adds `dispatch_probe` |
| `profile ... --standin attn,mlp,gdn` | Each block's in-place cost (dense qwen and llama trunks) | adds `standin` |
| `ab --model M --null` | This host's noise floor for a workload and metric. Run once per workload before any `ab` | `null_calibration` |
| `ab --model M --baseline off --candidate on:KNOB=V` | Does a switch win? Paired, interleaved, with a fixed verdict rule | `run`, `verdict` |
| `trace --model M --prompt ... --decode 64` | Metal System Trace of the measured work: GPU busy, idle gaps, per-kernel shares | `trace` |
| `tree-replay --model M --drafter mtp\|dflash` | Would tree verify accept more than the chain at equal rows | `tree_replay` |
| `chain-capture` then `depth-replay` | Best speculative depth policy, priced from modeled or ledger round costs | `chain_capture`, `depth_replay` |

Workload flags: `--prompt protocol:ID|messages:PATH|text:PATH|corpus:N|chat:PATH`
(repeatable), `--decode N`, `--chunk N|off`, `--context N`. An audio install
takes `--audio-job JOB.json` instead (`workloads/audio/`), and its metrics are
`audio_run_s` and `rtf`.

Arm syntax: `NAME[@SLOTS/RESIDENCY][~DRAFTER:BLOCK[a]][:KEY=V,...]`, for
example `mtp2~mtp:2:spec_stats=1`. `KEY` is a `TURBOSPARK_*` name or a knob
short name.

Host flags: `--max-foreign-cpu 150`, `--max-single-cpu 90`, `--allow-battery`,
`--quiet-wait-min 5`, `--min-free-gb 10`, `--timeout-min 60`. Loosening them
gets you numbers from a busy host, which is all they then are.

## Kernel benches (in process, no model weights)

`kernel --model M [--bench NAME[,NAME]|all] [--rows ...] [--contexts ...] [--hash]`

| Bench | Measures |
|---|---|
| `int4_matmul` | Per-row GEMV, scalar GEMM, Steel tiles and the 8-row MMA tile over M, at every INT4 shape the install binds |
| `int8_matmul` | INT8 GEMV against the batched GEMM (routers, shared gates, INT8 installs) |
| `attn_decode` | Barrier against GQA-shared decode attention over context, per variant (full, sliding, sinks) |
| `rows_attn` | Multi-row prefill attention against the per-row loop, with layout arms, one case per context |
| `moe_routed` | Routed-expert pair: M decode pairs against one route-list pair (union modeled as uniform routing) |
| `verify_ladder` | Whole-model GEMM cost per verify width; `--verify-from EXP` adds the measured non-GEMM marginal |
| `qkv_concurrent` | The attention q/k/v triple in a serial pass against a concurrent compute encoder (shared input, no barrier); `c(M)` is each arm's time over the serial arm's |
| `gdn_delta` | Gated-DeltaNet delta kernels (decode, eager and lazy prefill, state fold) at the install's geometry. One build per process: put `TURBOSPARK_GDN_DK128=0` on the `tslab` process for the runtime-Dk build |
| `music3_grid` | Music 3 corpus: output hashes against goldens, per-config timing |
| `whisper_linear` | Whisper encoder GEMM, decoder GEMV, GEMM at one row |

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

## Experiments against a code change (repository only)

`gate`, `run-experiment` and `confirm` take a spec file, check the git diff
against `--base`, build the worker from the tree, run required tests, the
quality gate, the paired A/B and the memory check, and print a `DECISION`.
They need the TurboSpark source tree, and outside it they refuse. The procedure
is `crates/lab/program.md`.

## Traps

- A forced-kernel variable such as `TURBOSPARK_ATTN_DECODE_KERNEL=barrier`
  changes more than the arm you meant (the trunk's verify too). The baseline
  is the variable unset.
- Probe, stand-in, trace and dispatch-profile runs are never quotable.
- One measuring session per machine. If the lock is held, wait.
- A cold page cache looks like a slow kernel. Let the warmup round absorb it,
  and check `sysctl vm.swapusage` before timing.
