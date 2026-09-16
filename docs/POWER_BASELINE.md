# Power baseline

Watts and joules-per-token over the frozen community protocol, on AC and on
battery. ROADMAP Phase P1.

Seven installs have rows here now: Gemma 4 26B-A4B (INT4 and 3-bit),
Qwen 3.6 35B-A3B, Qwen3-30B-A3B, gpt-oss-20b, Muse Glimmer 30B and
Ornith-1.5 35B-A3B. This line read "both real installs" until 2026-08-22,
which was true of the Phase P1 capture it was written for and had been
wrong for five sections; **re-count it before quoting it**, since nothing
goes red when a count rots.

This is not a parity claim. Swift was never measured for power, here or
upstream, so there is no second engine in any table below; this is the
port measuring itself, like the quality gates. `docs/BENCHMARKS.md` holds
the Swift comparison and a summary of these numbers;
`docs/BENCHMARKING.md` documents the harness alongside the other bench
modes.

Reproduce with `scripts/power.sh`. It needs root, because `powermetrics`
does. `LABEL` must match the actual power source; the script refuses to
start if it does not.

```sh
LABEL=ac OUT=/tmp/power-gemma-ac MODEL=~/models/gemma4.gturbo scripts/power.sh 2
LABEL=ac OUT=/tmp/power-qwen-ac  MODEL=~/models/qwen36.gturbo scripts/power.sh 2
LABEL=ac MODEL=~/models/gemma4.gturbo CASES=short-explanation \
  ARMS=default,utility scripts/power.sh 3
```

`ARMS` is the comparison axis (`QOS` is its former name and still works).
It takes `default`/`utility`, which set `TURBOSPARK_READ_QOS`, or
`performance`/`balanced`/`efficiency`, which pass `--power-profile` to the
bench (ROADMAP Phase P2). The two kinds cannot be mixed in one run: the
arm name is a single column of `rows.tsv` and a single grouping key in the
summary, so mixing them would compare two different seams under one label.

## Accepted baselines below use sequential prefill

Stated up front because until 2026-09-05 it could not have been otherwise
and nothing said so. `scripts/power.sh` drives `turbospark-bench`, and that
binary's `--model` mode reached only `run_raw_completion` -- never
`run_raw_completion_chunked` -- so every capture on this page measured the
sequential prefill path no matter what `TURBOSPARK_PREFILL_CHUNK`,
`TURBOSPARK_ROUTED_BATCH` or `TURBOSPARK_BATCHED_GEMV` were set to, with
nothing in `rows.tsv` or the summary recording which path ran.

`turbospark-bench --prefill-chunk` and `scripts/power.sh`'s `seq|chunked`
arm pair close that. The flag DEFAULTS OFF, so every row here still
describes the invocation it was taken with, and the bench header now prints
`prefill=sequential` or `prefill=chunked` so a future row cannot silently be
the other one.

**A chunked row is a NEW row, not a re-freeze of an old one.** This is the
same rule the mapped-residency seam already carries in
`crates/bench/CLAUDE.md` Gotcha 1: an arm that changes which code path runs
produces a row that belongs beside its predecessor rather than replacing
it. Chunked prefill is the DEFAULT for the CLI and the server, so a chunked
row is arguably the more representative one for a user, and that is an
argument for measuring it, never for overwriting a sequential row with it.

### Sequential/chunked capture, 2026-09-10: not accepted as a baseline

The user ran `LABEL=ac ARMS=seq,chunked scripts/power.sh 2` on an M4 Max
with 36 GB, macOS 26.6.2, automatic cooling, and the Gemma install. Capture
started at 11:04:57 UTC, revision `7fda1bf` dirty. The benchmark header
confirms context 4096, max_new 1024, 16 expert slots, KV quantization off,
protocol sampling, and chunk size 128; routed_batch and batched_gemv were
both unset. This compares sequential with chunked prefill, not the separate
batched-GEMV switch. All measured arms stopped at endOfTurn.

Preserved evidence: [raw rows](verification/prefill-energy-2026-09-10.tsv)
and [machine provenance](verification/prefill-energy-2026-09-10-system.txt).
The TSV has the harness's 18 columns, in order: case, label, arm, run,
phase, seconds, tokens, joules, watts, J/token, CPU mW, GPU mW, E residency,
P residency, thermal pressure, samples, battery watts, cooling. Full logs
remain at `/tmp/roadmap-prefill-energy-20260910/`.

The timeline drift was -0.8 s over 1264.6 s, within the harness's 2 s
warning threshold. The minimum CPU power was 895 mW, below its steady-load
warning threshold. Neither establishes a clean capture: medium and long
arms reached Moderate or Heavy thermal pressure, and the long sequential
prefill's CPU power changed from 17.60 W to 4.89 W between repeats.
Intermittent background load is possible; the capture does not identify
its source. Short prefill windows integrated only 4-7 samples.

The repeated long-prefill readings show why the means are not baselines:

| Pair | Sequential s | Chunked s | Sequential J/token | Chunked J/token |
| --- | ---: | ---: | ---: | ---: |
| 1 | 76.54 | 58.82 | 0.9930 | 0.8109 |
| 2 | 63.70 | 50.70 | 0.4427 | 0.4108 |

Chunked prefill was faster in all six pairs, but the long-prefill energy
spread was 76.7% sequential and 65.5% chunked. Even the all-Nominal short
case had prefill energy spreads of 14.2% and 24.6%. Do not freeze the
averages or claim a reproducible energy saving from this run.

The forced-cooling retry below resolved thermal pressure, but did not
resolve energy repeatability. A successful forced-cooling row belongs
beside the automatic-cooling baselines and does not close the
shipping-cooling measurement by itself.

### Forced-cooling retry, 2026-09-10: thermally stable, energy unresolved

The user repeated the capture with `COOLING=max` and three pairs per case,
starting at 11:30:50 UTC. Machine, install hash, reported revision and
benchmark settings match the preceding capture. Fans were commanded to
5777 RPM and restored to the automatic curve at completion. All 42 phase
rows, including warmups, stayed Nominal; all measured arms stopped at
endOfTurn. Timeline drift was -0.6 s over 1494.9 s, and the CPU floor was
332 mW. These checks passed, but the energy-spread checks did not.

Evidence: [raw rows](verification/prefill-energy-max-2026-09-10.tsv), using
the same 18-column layout above, and
[machine provenance](verification/prefill-energy-max-2026-09-10-system.txt).
Full logs remain at `/tmp/roadmap-prefill-energy-max-20260910/`.

| Long prefill pair | Sequential s | Chunked s | Sequential J/token | Chunked J/token | Sequential CPU W |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 73.75 | 46.04 | 0.8261 | 0.4115 | 21.297 |
| 2 | 60.95 | 46.97 | 0.4622 | 0.4479 | 8.432 |
| 3 | 60.70 | 46.33 | 0.4689 | 0.4110 | 8.852 |

The first sequential long prefill remains anomalous despite Nominal
pressure: its CPU draw is over twice the later runs, while GPU draw is
lower (12.384 W versus 14.369/14.438 W). The resulting sequential energy
spread is 62.1%; chunked spread is 8.7%. Dropping the first pair would
produce a more attractive comparison without establishing why it differed,
so all three remain in the record and no aggregate energy saving is frozen.
The last two pairs alone imply 3.1% and 12.3% lower chunked prefill energy,
which are diagnostic observations, not an accepted baseline.

Short chunked prefill integrated four samples per run and its energy
spread was 30.2%. Medium prefill spread was 20.0% sequential and 13.8%
chunked. The short and medium paired energy differences each change sign
across repeats. Short and medium timing is much steadier: chunked speedups
range from 1.311x to 1.371x across their six pairs. This supports the
throughput direction, not a reproducible energy benefit.

Forced cooling removed the observed thermal-pressure problem. The CPU
variation is consistent with intermittent activity, but these logs have no
per-process CPU timeline and cannot attribute it to another process rather
than the benchmark itself. The low idle floor does not settle that question.
Before another full capture, correlate process CPU activity with the power
windows and investigate the first long sequential run. Keep short-window
sampling uncertainty separate from that long-window anomaly. The energy
baseline remains open; no existing baseline is replaced.

The harness now writes `processes.jsonl` throughout capture, including
warmups and the idle lead-in. Each snapshot records its start/end Unix
milliseconds and every visible process's PID, parent PID, command name,
cumulative CPU time and `ps` CPU percentage. It pauses one second between
snapshots. Join these timestamps to the stderr `power-window` markers;
the prefill boundary is the start marker plus the footer's prefill seconds.
Use cumulative CPU-time differences for interval activity: `ps` CPU
percentage is a decayed average, not an exact window measurement. Exited
short-lived processes may be missed, and PID reuse needs care. Process
activity can identify suspects but does not assign package joules to them.
The sampler adds overhead; retain it in both arms and report it when
comparing new captures with older ones. Errors go to `processes.stderr`;
a failed startup aborts capture and an early exit warns at completion.

### Process-traced long-prefill capture, 2026-09-10

The user ran three long-synthesis pairs at `COOLING=max`, starting at
21:52:42 UTC, revision `dc48cd5` dirty, with the process sampler enabled.
All phases stayed Nominal and measured arms stopped at endOfTurn. Drift
was -0.6 s over 1024.6 s, CPU floor 57 mW, and `processes.stderr` was empty.
Fans were restored. This capture has a different reported revision from
the morning attempts; it cannot establish the cause of their anomalies.

Evidence: [raw rows](verification/prefill-energy-cpu-trace-2026-09-10.tsv),
[provenance](verification/prefill-energy-cpu-trace-2026-09-10-system.txt),
and [process-window summary](verification/prefill-energy-cpu-trace-2026-09-10-process-summary.json).
The summary records window timestamps, coverage, the ten largest observed
CPU-time consumers per window, method limitations, and the original
process log's SHA-256. Full logs remain at
`/tmp/roadmap-prefill-energy-cpu-trace/`.

| Pair | Sequential s | Chunked s | Sequential J/token | Chunked J/token | Chunked energy change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 63.36 | 48.02 | 0.3904 | 0.3997 | +2.4% |
| 2 | 63.04 | 50.28 | 0.3851 | 0.3010 | -21.8% |
| 3 | 62.69 | 49.65 | 0.3776 | 0.3140 | -16.8% |

The process trace identifies real competing work in pair 1 chunked
prefill: `mediaanalysisd` (PID 75523) accumulated 69.55 CPU-seconds in
approximately 46.9 seconds of covered time, about 148% of one core. The
ChatGPT Sparkle `Autoupdate` process added 5.16 CPU-seconds. By comparison,
the benchmark accumulated 137.69 CPU-seconds in that window and about
125.3/125.1 in the later chunked windows. The media process had only
2.00 CPU-seconds in the following sequential window and was not among
the ten largest consumers in either later chunked window.

This is affirmative evidence of intermittent contamination, despite the
clean idle floor and Nominal pressure. It does not assign the additional
CPU/GPU/ANE watts to a particular process, nor prove all of the benchmark's
own CPU-time variation is caused by the competitor. All three pairs remain
in the record. Sequential prefill energy spread was 3.3%, chunked 29.2%;
the overall energy-saving mean is not accepted as a baseline.

The later pairs support a potential energy benefit and all three support
faster chunked prefill, but two less-contaminated pairs do not establish
a clean three-pair baseline. The next capture should wait until the
observed media-analysis and update activity has subsided, retain process
logging, and check each measured window rather than only the idle floor.
Do not disable system services or subtract estimated process energy from
these readings.

## Run provenance

| | |
| --- | --- |
| Date | 2026-08-07, two sessions: 23:58-00:40 UTC (battery) and 00:56-01:26 UTC (AC) |
| Chip | Apple M4 Max, 36 GB |
| macOS | 26.5.2 (25F84) |
| Installs | `~/models/gemma4.gturbo`, `~/models/qwen36.gturbo` |
| Expert-cache slots | 16 (protocol default) |
| Protocol | frozen `real-generation-v1`, seeds 20260721-23, temp 0.2, top-k 64, top-p 0.95, 1024 new-token budget, 4K context |

The last row is this table's own scope. The context window and the
generation budget became per-family on 2026-08-12, resolved by
`turbospark-bench --model` from the install's manifest, so the gpt-oss
section below is measured at 8,192/3,072 and its rows are not comparable
to a 4,096/1,024 row of another install without saying so.

| Sampler | `powermetrics -s cpu_power,gpu_power,thermal -i 200` |
| Runs per case | 1 discarded warmup process, then 2 measured (3 pairs for the QoS A/B) |

**One binary across both sessions.** `target/release/turbospark-bench` was
built once before the battery session and never rebuilt, so the AC/battery
comparison varies only the power source. The recorded git rev moves from
`bcaa62f` to `705bf37` because unrelated work committed to the branch
mid-session; it did not reach the binary.

`powermetrics` is not free: it wakes 5x/s at this interval and its own CPU
time lands in the counters it reads. That inflates absolute watts equally
across arms, so paired ratios are unaffected.

## What the numbers mean

**Watts and joules-per-token are CPU+GPU+ANE**, which is `powermetrics`'
`Combined Power`. DRAM, SSD, display and the rest of the SoC are excluded,
so this is compute energy, not wall energy.

Joules-per-token is integrated over the measured run only. The bench emits
`[power-window ...]` markers so that the 13 GB mmap, the Metal pipeline
compilation, and the discarded 1024-token warmup all fall outside the
total. Decode divides by generated tokens, prefill by prompt tokens.

## AC: the baseline

**Use these rows.** Every run of both installs stayed at Nominal thermal
pressure (50 of 50 sampled arms), so nothing here is filtered and every
row is n=2.

Decode:

| install | case | tok/s | watts | J/token | cpu W | gpu W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 26B-A4B | short-explanation | 40.70 | 16.66 | 0.3838 | 4.31 | 12.34 |
| Gemma 4 26B-A4B | medium-review | 38.40 | 17.83 | 0.4465 | 4.23 | 13.60 |
| Gemma 4 26B-A4B | long-synthesis | 34.75 | 17.77 | 0.4975 | 4.56 | 13.21 |
| Qwen 3.6 35B-A3B | short-explanation | 38.83 | 14.30 | 0.3513 | 3.95 | 10.35 |
| Qwen 3.6 35B-A3B | medium-review | 37.76 | 13.78 | 0.3517 | 3.50 | 10.27 |
| Qwen 3.6 35B-A3B | long-synthesis | 33.78 | 14.88 | 0.4337 | 3.44 | 11.45 |

Prefill:

| install | case | watts | J/prompt token |
| --- | --- | ---: | ---: |
| Gemma 4 26B-A4B | short-explanation | 17.24 | 0.3692 |
| Gemma 4 26B-A4B | medium-review | 21.30 | 0.4098 |
| Gemma 4 26B-A4B | long-synthesis | 20.57 | 0.4402 |
| Qwen 3.6 35B-A3B | short-explanation | 15.69 | 0.3306 |
| Qwen 3.6 35B-A3B | medium-review | 15.99 | 0.3512 |
| Qwen 3.6 35B-A3B | long-synthesis | 17.19 | 0.4227 |

Two shapes worth naming. **Energy per token grows with context** on both
families (Gemma 0.384 -> 0.447 -> 0.498 J/token across the three cases),
which tracks the throughput fall rather than any rise in power. And
**prefill draws more watts than decode while costing comparable energy per
token**, consistent with prefill being the per-token faster path: it packs
more work into the same second.

**Qwen is the more efficient engine on this hardware**, 0.35 J/token
against Gemma's 0.38-0.45 on the comparable cases, almost entirely from
GPU power (10.3 W against 12.3-13.6 W). That is the hybrid
linear-attention design showing up on the power axis the same way it
already does on the memory axis.

## The 3-bit install: the Phase S capture, and it is a loss

Measured 2026-08-09 on AC, `~/models/gemma4-iq3.gturbo`, same protocol,
same interval, 2 measured pairs per case after a discarded warmup, rev
`d9a9e49`, all runs `stop=endOfTurn`, no thermal-pressure exclusions.
Raw capture: `/tmp/power-iq3/`.

Decode:

| install | case | tok/s | watts | J/token | cpu W | gpu W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 IQ3_XXS/IQ4_NL | short-explanation | 28.37 | 27.20 | 0.9247 | 2.65 | 24.55 |
| Gemma 4 IQ3_XXS/IQ4_NL | medium-review | 27.26 | 27.71 | 0.9960 | 2.69 | 25.02 |
| Gemma 4 IQ3_XXS/IQ4_NL | long-synthesis | 25.30 | 25.38 | 0.9890 | 2.59 | 22.80 |

Prefill:

| install | case | watts | J/prompt token |
| --- | --- | ---: | ---: |
| Gemma 4 IQ3_XXS/IQ4_NL | short-explanation | 28.96 | 0.8265 |
| Gemma 4 IQ3_XXS/IQ4_NL | medium-review | 33.26 | 0.9667 |
| Gemma 4 IQ3_XXS/IQ4_NL | long-synthesis | 30.15 | 0.9449 |

**Against the incumbent INT4 install's AC rows above, decode
joules-per-token roughly doubles: 2.0x to 2.4x across the three cases**
(0.925-0.996 against 0.384-0.498). Prefill is the same story at
2.2-2.4x. The attribution is clean and entirely GPU-side: gpu W nearly
doubles (22.8-25.0 against 12.3-13.6) while cpu W falls (2.6 against
4.2-4.6), so the codebook dequant is burning the power, not the host.
The install draws more watts and runs 35% slower, and J/token compounds
the two.

This settles the phase's motivating axis, negatively. Fewer expert bytes
per miss was a joules claim, and the measured answer is that codebook
decode costs about twice the energy the smaller reads were supposed to
save. The 3-bit path is a memory and disk win only (-15% footprint, -20%
expert bytes); on every other axis it loses.

One caveat, and why it does not change the reading: this is a
cross-session comparison against the 2026-08-07 baseline, on a different
binary. Cross-session energy drift measured here is a few percent with no
consistent sign (Gotcha 22 in AGENTS.md); the effect is 100-140%. No
interleaved same-binary A/B could plausibly close that gap.

## Qwen3-30B-A3B (`qwen3moe`): the M3 capture

Measured 2026-08-12 on AC, `~/models/qwen3moe-gguf.gturbo`, same protocol,
same interval, 2 measured pairs per case after a discarded warmup, rev
`b964d0b`, all runs `stop=endOfTurn`, no thermal-pressure exclusions,
drift -0.7 s over a 1,364 s span. Raw capture: `/tmp/power-qwen3moe/`.
This closes the family's last Definition-of-Done item (ROADMAP M3).

Decode:

| install | case | tok/s | watts | J/token | cpu W | gpu W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B Q4_K_M | short-explanation | 25.10 | 20.78 | 0.7811 | 4.98 | 15.79 |
| Qwen3-30B-A3B Q4_K_M | medium-review | 21.20 | 19.79 | 0.9031 | 5.51 | 14.27 |
| Qwen3-30B-A3B Q4_K_M | long-synthesis | 15.57 | 21.99 | 1.4004 | 3.94 | 18.05 |

Prefill:

| install | case | watts | J/prompt token |
| --- | --- | ---: | ---: |
| Qwen3-30B-A3B Q4_K_M | short-explanation | 19.93 | 0.9431 |
| Qwen3-30B-A3B Q4_K_M | medium-review | 17.77 | 0.8397 |
| Qwen3-30B-A3B Q4_K_M | long-synthesis | 20.56 | 1.0493 |

**J/token is roughly double the two MLX-install families' (0.78-1.40
against Gemma's 0.38-0.50 and Qwen 3.6's 0.35-0.43), and the driver is
throughput, not watts.** Power sits at 20-22 W, between Gemma's 16.7-17.8
and the 3-bit install's 25-27, while decode runs 15.6-25.1 tok/s against
their 34-41. Same energy shape as every family: J/token grows with
context because tok/s falls, not because watts rise.

**Prefill is the energy story on the long case.** `long-synthesis`
prefills 2,842 tokens for ~146 s at ~20.6 W, so its prefill window costs
2,982 J against its decode window's 466 J -- 86% of the case's measured
energy is prompt processing. That is the descoped sequential-prefill gap
(one forward pass per prompt token) showing up on the joules axis; on the
faster-decoding families the same gap exists but the split is less
lopsided.

One spread observation, noted rather than excluded: `medium-review` p2's
prefill wall clock read 27.5 s against 16.1 s (p1) and 15.5 s (warmup) at
similar watts. This is a single-arm baseline rather than an A/B, so no
conclusion turns on it; the prefill row averages both pairs.

## gpt-oss-20b: all three cases, and what a competing load costs a row

Measured 2026-08-13 on AC, `~/models/gptoss-20b.gturbo`, rev `aa2c99f`,
2 measured pairs per case after a discarded warmup, every arm Nominal.
Raw captures: `long-synthesis` from `/tmp/power-gptoss3/`, the other two
from `/tmp/power-gptoss3b/`. **This supersedes the one-case capture of
2026-08-12** (`/tmp/power-gptoss/`), which the section below explains and
does not simply delete.

All three cases run now because `turbospark-bench --model` resolves the
protocol's context window and generation budget from the install's family
(8,192/3,072 here) instead of the shared 4,096/1,024, under which two of
the three stopped on `maxTokens` and a truncated run is not a protocol
row.

Decode (n=2 per case):

| install | case | tok/s | watts | J/token | cpu W | gpu W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gpt-oss-20b MXFP4 | short-explanation | 30.44 | 32.92 | 1.0569 | 1.48 | 31.44 |
| gpt-oss-20b MXFP4 | medium-review | 27.0 | 30.09 | 1.1109 | 1.71 | 28.36 |
| gpt-oss-20b MXFP4 | long-synthesis | 22.9 | 29.44 | 1.2813 | 1.73 | 27.70 |

Prefill (n=2 per case):

| install | case | watts | J/prompt token |
| --- | --- | ---: | ---: |
| gpt-oss-20b MXFP4 | short-explanation | 35.55 | 1.0257 |
| gpt-oss-20b MXFP4 | medium-review | 34.19 | 0.9576 |
| gpt-oss-20b MXFP4 | long-synthesis | 34.27 | 1.0793 |

**The three cases are internally consistent, which is part of why they
are believable**: watts fall and J/token rises monotonically as the case
lengthens (32.92 -> 30.09 -> 29.44 W against 1.0569 -> 1.1109 -> 1.2813
J/token) because decode slows with context (30.4 -> 27.0 -> 22.9 tok/s)
while instantaneous power barely moves. Three independently measured
cases landing on one trend is a stronger statement than any single row.

**This is still the highest sustained power of any install measured
here.** ~29-33 W combined and ~28-31 W GPU, against Gemma's 16.7-17.8,
Qwen 3.6's 13.8-14.9, Qwen3-30B-A3B's 19.8-22.0 and the 3-bit install's
25.4-27.7. Unlike `qwen3moe`, whose 2x J/token came from the tok/s
denominator, gpt-oss decodes at a healthy 30 tok/s and its ~1.1 J/token
comes from the watts numerator. The attribution is GPU-side (cpu W is
1.5-1.7), consistent with MXFP4 dequant running in every routed expert
the way the IQ install's codebooks did -- but no interleaved A/B isolates
that here, so read it as a shape, not a proof.

### Why the 2026-08-12 row moved, and it was not the engine

The superseded row read decode 36.67 W / 1.1506 J/token at 31.04 tok/s
against today's 32.92 W / 1.0569 at 30.44. Throughput is the same to 2%
and `gpu W` agrees to 2.9% (32.35 then 31.44); **the entire difference is
`cpu W`, 4.32 then 1.48**, and 2.84 of the 3.75 W gap is exactly that
term. The old row was measured while this machine's UI was busy, and
`powermetrics` Combined Power is SYSTEM-wide: a busy desktop app lands in
the same counter as the decode loop.

The first three-case attempt the same day caught it in the act, which is
how it was diagnosed. Every arm held Nominal, so the thermal exclusion
rule saw nothing, and yet `medium-review` read **1.7072 J/token on p1 and
1.0799 on p2 for byte-identical work** (2,597 tokens both times) -- a 37%
spread that the script's summary averaged into 1.3936, a number
describing neither run. `cpu W` fell monotonically through that capture
(4.76 / 4.07 / 3.34 / 3.03 early against 1.50 / 1.62 / 1.66 / 1.81 late)
as the UI went idle, with `gpu W` tracking it. The re-run on a quiet
machine reproduces to 0.4% on `short-explanation` (1.0590 / 1.0548) and
1.7% on `medium-review` (1.1204 / 1.1014).

`long-synthesis` is kept from that first capture rather than re-measured,
and the reason is the same discriminator: its arms were taken after the
machine went quiet (cpu W 1.66 / 1.81, in the re-run's 1.4-1.9 band) and
its three readings already agreed to 0.6% (1.2862 / 1.2781 / 1.2845).

The general form is now AGENTS.md Gotcha 43. A thermal-pressure check
cannot see a competing load; the tells are `cpu W` against the install's
own norm and DISPERSION between arms doing identical work, neither of
which `scripts/power.sh` puts in its summary.

### The AC throttle, in light of the above

The 2026-08-12 capture remains the only time this machine left Nominal on
AC: one decode window read Heavy, and that throttled arm read 1.1085
J/token against the clean 1.1506, i.e. 3.7% BETTER -- exactly the
direction AGENTS.md Gotcha 28 warns makes a throttled arm flatter a power
table. It is not retracted; it happened and its rows are in
`/tmp/power-gptoss/`.

What today qualifies is the WATTAGE at which it happened. Thirty arms
across the two captures here all held Nominal at a clean ~29-33 W, where
the throttled session was reading ~36-37 W with background load included.
So the install's own draw is a few watts lower than that session
suggested, and whatever pushed the machine over was partly not the
engine. Gotcha 28's conclusion -- that thermal saturation here is a
function of total draw rather than of the power source -- is unchanged;
the number attached to it is.

## Ornith-1.5 35B-A3B: the first row the contamination floor caught

Measured 2026-08-22 on AC, `~/models/ornith35b.gturbo` (MLX INT4, manifest
sha `e69caecb…`), `short-explanation` only, 16 slots, rev `e8deb6c`. Every
one of six rows Nominal, arms agreeing to 2.3%, 67 samples per decode arm.

| phase | tok/s | watts | J/token | cpu W | gpu W | samples/arm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 42.99 | 21.05 | 0.4731 | 7.96 | 13.09 | 67 |
| prefill | -- | 20.38 | 0.4301 | 8.08 | 12.30 | 6 |

**Take the decode row; the prefill row is thin.** Six samples at a 200 ms
interval over a 1.37 s window is close to the edge-error floor, and the
harness's own too-few-samples warning fires only under three. The two arms
agreeing to 2.3% is the reason it is printed at all.

**IT TOOK THREE CAPTURES, AND THE FIRST TWO PASSED EVERY TELL THIS PAGE HAD.**
That is what the row is really worth recording for.

| capture | CPU floor | decode cpu W | E% | tok/s | J/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,361 mW | 10.67 | 93 | 43.2 | 0.4927 |
| 2 | 2,480 mW | 3.21 | 74.8 | 42.0 | 0.3899 |
| 3 (this row) | **244 mW** | 7.96 | 94.7 | 43.0 | 0.4731 |

Capture 1 ran against a Finder stuck at 99% of a core with
`iconservicesagent` at 25%, and 269% of CPU summed across the machine. It
reproduced to 0.18% on `gpu_W` and 0.29% on tok/s and held Nominal on all
six rows, so BOTH of Gotcha 43's tells read clean: dispersion cannot see a
load that is CONSTANT, because it contaminates every arm equally, and the
`cpu_W`-against-norm tell needs a norm, which a first capture of a new
install does not have. Only the minimum CPU power anywhere in the log
separated them, which is why `scripts/power.sh` now prints it.

**CAPTURE 2 IS THE ONE WORTH STUDYING, because its `cpu_W` was the LOWEST
of the three and it is not the clean one.** Reading 3.21 W against capture
3's 7.96 invites the conclusion that capture 3 carries 4.7 W of background.
It does not. Capture 2's E-cluster residency fell to 74.8% against 93-95%
on either side of it, and its throughput to 42.0 -- the signature of
`read_pool` threads BLOCKED on real SSD reads rather than servicing page-
cache hits, with Brave, Parsec and WhatsApp crowding the cache that holds an
18 GB install's experts. Less CPU meant less work done, not less
contamination. Captures 1 and 3 agree on the working distribution (p25 7,948
and 7,124 mW) and differ almost exactly by capture 1's floor excess
(10.67 - 7.96 = 2.7 W against a floor difference of 3.1 W), which is what
identifies 3 as the clean one and ~8 W as the engine's own CPU draw.

The general form, and it is the inverse of the usual worry: **a LOWER `cpu_W`
can mean a WORSE capture.** Pair it with E-cluster residency and throughput
before reading it as cleanliness.

**Against its own architecture.** Ornith-1.5 35B-A3B is Qwen 3.6's
architecture retrained, so the AC table above is the natural comparison:
0.4731 J/token here against Qwen 3.6's 0.3513 on the same case, at 21.05 W
against 14.30, while decoding FASTER (43.0 against 38.8 tok/s). The gap is
mostly CPU (7.96 against 3.95) with GPU up 26%. **Do not read that as an
engine regression**: the Qwen 3.6 row is from 2026-08-07 on a different
binary, and cross-session absolutes are the thing that has repeatedly failed
to reproduce here (Gotcha 22). A controlled answer needs both installs in
one session, which is what the "Still owed" entry below asks for.

## Muse Glimmer 30B: two operating points, and the row is the unconstrained one

**This install cannot be characterised by the harness's normal protocol on
this machine.** It saturates thermally within about two minutes of decoding
regardless of starting temperature, so the measured pairs the summary is built
from are all governed rather than free-running. Two captures were needed to
establish that, and together they give a better answer than one clean run
would have.

`LABEL=ac MODEL=~/models/museglimmer-30b.gturbo scripts/power.sh 2`, rev
`c8037b0` dirty. Capture A 2026-08-16T13:31Z, all three cases, machine warm
from a preceding oracle and two quality gates. Capture B 16:01Z,
`CASES=short-explanation`, after a 30-minute idle. Every run stops
`endOfTurn`. Neither capture is contaminated: `cpu_W` reads 0.55-1.28 W on
capture A's measured rows against the ~4 W that signalled contamination in the
2026-08-13 gpt-oss capture.

### The unconstrained point, and it reproduces

The only Nominal windows in either capture are the warmups, which the summary
excludes by design. They agree across three sessions and three starting
temperatures (capture C is the profile A/B below, whose warmup runs
unconstrained like the others):

| short-explanation, Nominal | A | B | C | spread |
| --- | ---: | ---: | ---: | ---: |
| decode W | 38.24 | 37.69 | 38.17 | 1.4% |
| decode J/token | 2.0444 | 2.0135 | 2.0316 | 1.5% |
| prefill W | 39.42 | 38.59 | 39.56 | 2.5% |
| prefill J/token | 1.8445 | 1.8037 | 1.8507 | 2.6% |

**~37.7-38.2 W and ~2.01-2.04 J/token is this install's decode cost**, and it
is the highest draw in this document, above gpt-oss's 29-33 W, which was the
previous high. The 30-minute idle changed the warmup by 1.4%, which is what
says the reading is the workload's rather than the session's.

A warmup window is a legitimate measurement here and that is worth stating,
because elsewhere in this repo a first run is exactly what gets discarded
(AGENTS.md Gotcha 20, cold GPU at low DVFS clocks reading ~50% slow). It does
not apply: the model open, the mmap and the Metal pipeline compile all happen
before the `[power-window]` markers, and the warmup decodes at 18.53-18.59
tok/s against the measured arms' 18.35-18.40, marginally faster, not slower.

### The governed point, and it does not reproduce

| short-explanation decode, Heavy | capture A | capture B |
| --- | ---: | ---: |
| p1 J/token | 1.4901 | 1.6394 |
| p2 J/token | 1.4396 | 1.4833 |

p1 moves 10% between captures, and within capture B the two pairs differ by
10% on byte-identical work where capture A's agreed to 3.5%. That instability
is the tell: a thermally governed operating point is a control loop's answer,
not a property of the workload. **Do not publish a J/token row from these**,
and note the harness's summary is built entirely from them -- 1.4648 in
capture A, 1.5614 in capture B, for the same case.

### 28% less power for 1.3% less throughput

Same case, same 1,132 tokens, capture B:

| | watts | tok/s | J/token |
| --- | ---: | ---: | ---: |
| unconstrained (Nominal) | 37.69 | 18.593 | 2.0135 |
| governed (Heavy, p2) | 27.46 | 18.347 | 1.4833 |

Giving up **1.3%** of throughput bought **26%** less energy per token. That is
the sharpest instance in this document of the superlinear effect the thermal
section below describes -- the Gemma pair gave up 20% of throughput for its
31% saving, and this one gives up almost nothing.

**The size of that saving is not stable, and capture C says so.** Three more
governed decodes of the same case read 1.3267, 1.4128 and 1.6621 J/token, so
the governed point ranges from 35% to 18% below the unconstrained one rather
than sitting at 26%. The direction is solid across six governed decodes in
three sessions; the magnitude is a control loop's output and should be quoted
as a range or not at all.

The reading is that the unconstrained operating point is wasteful for this
workload: the GPU sits at a voltage and frequency far above what this decode
needs, and the governor's forced descent costs almost no work. **It is an
involuntary experiment, not a controlled one**, so it is an observation rather
than a result -- but it makes this install the strongest candidate in the repo
for Phase P2's deliberate version, `ARMS=performance,efficiency`, which is the
same descent chosen rather than imposed. That A/B is owed and would settle it.

### The `performance,efficiency` A/B: the cap works, the comparison does not

Capture C, 2026-08-16T16:23Z, `CASES=short-explanation
ARMS=performance,efficiency scripts/power.sh 3`. It was run because the
thermal governor had found 26% of energy for 1.3% of throughput
involuntarily, and Phase P2's rate cap is the deliberate version of that
descent.

**The prediction made before the run was wrong, in the informative
direction.** It said the efficiency arm's longer window gives it more time to
saturate. The opposite happened: the efficiency arm held Nominal on 6 rows of
6, and the performance arm went Heavy on 3 decodes of 3. Capping the rate
keeps the machine out of thermal governance entirely, which is a result about
the cap rather than about this model.

| decode | pressure | W | tok/s | J/token |
| --- | --- | ---: | ---: | ---: |
| unconstrained (warmup) | Nominal | 38.17 | 18.573 | 2.0316 |
| performance p1/p2/p3 | **Heavy** | 24.31 / 25.78 / 30.45 | ~18.2 | 1.3267 / 1.4128 / **1.6621** |
| efficiency p1/p2/p3 | Nominal | 16.35 / 16.12 / 15.67 | 10.00 | 1.6258 / 1.6075 / 1.5649 |

**The A/B is inconclusive on J/token and the reason is in the third column.**
The performance arm spans 25% across three pairs of byte-identical work
(1.3267 to 1.6621) and that range swallows the efficiency arm's (1.5649 to
1.6258): p1.performance beats every efficiency reading and p3.performance
loses to every one. A difference cannot be read off arms one of which is
being driven by a control loop. `scripts/power.sh`'s summary reports
efficiency 9% worse (1.5994 against 1.4672) and that number should not be
quoted: it compares a governed arm against a free one, which is not the
comparison the flag exists to make.

What is established, and each of these is stable:

- **The rate cap does exactly what it says.** 113.17 s and 10.002 / 10.003 /
  10.002 tok/s, three times. That is 0.01% reproducibility, tighter than
  anything else in this document.
- **The efficiency arm never throttles**, where the performance arm always
  does. On this install the cap is the difference between a measurable
  operating point and an unmeasurable one.
- **Efficiency is stable where performance is not**: 3.9% spread against 25%.
- **Against the only stable performance reading -- the unconstrained 2.0316 --
  efficiency saves ~21%** (to ~1.60). That is the comparison with two
  trustworthy sides, and it is the one worth carrying.

**A fair A/B needs hardware that can hold Nominal in performance mode**, which
this laptop cannot for a whole case ON ITS OWN FAN CURVE. **RESOLVED
2026-08-18 by supplying that condition rather than waiting for it**: with fans
pinned the performance arm holds Nominal 3 of 3 and the A/B completes -- see
"Forced cooling: the A/B, run" below, which supersedes both the "~21%" here
and the framing that the flag's only value is repeatability. The measured
answer is 14.9% for 51.5% of the throughput.

One ambiguity left open rather than resolved: efficiency prefill draws 30.3 W
against performance prefill's 37.0-38.4 W when the latter is Nominal, on
identical work at an identical 4.8 s. Prefill is not rate-capped (both arms
take the same time), so either the profile lowers more than the token rate,
or the efficiency prefill inherits a hot machine from the performance run it
is interleaved after. The interleaving makes those two indistinguishable
here, and separating them needs an arm order that is not paired.

### Forced cooling: the A/B, run

The paragraph above says forced cooling is the only way to find out here. It
was run 2026-08-18T19:42Z, and it works. `CASES=short-explanation
ARMS=performance,efficiency COOLING=max scripts/power.sh 3`, rev `14aa0ce`
dirty, fans pinned to 5,777 RPM through ThermalForge for the whole capture,
5,017 samples, drift -0.7 s, every arm `stop=endOfTurn`, `cpu_W` 0.22-0.71
throughout (so uncontaminated, Gotcha 43).

**Twelve of twelve measured rows held Nominal**, where the same case on the
same install goes Heavy on every performance decode uncooled. That is the
missing condition supplied, and the comparison it unblocks:

| decode, 3 pairs | pressure | W | tok/s | J/token | J/token spread |
| --- | --- | ---: | ---: | ---: | ---: |
| performance, uncooled (capture C) | **Heavy 3/3** | 24.3-30.5 | ~18.2 | 1.3267 / 1.4128 / 1.6621 | **25%** |
| performance, pinned | Nominal 3/3 | 31.73 | 19.425 | 1.6338 / 1.6175 / 1.6015 | **2.0%** |
| efficiency, pinned | Nominal 3/3 | 13.80 | 10.002 | 1.4182 / 1.3502 / 1.3596 | 5.0% |

**The rate cap saves 14.9% of energy per token for 51.5% of the throughput**
(1.3760 against 1.6176). Both sides are now trustworthy: the performance arm's
spread falls 25% -> 2.0% and its tok/s reproduces to 0.08% (19.423 / 19.434 /
19.419). Note this REPLACES the tentative "~21%" the section above offers
against the unconstrained warmup, and it is a worse trade than that number
made it look.

**Pinning fans LOWERED J/token, which is the opposite of the prediction made
before the run.** The prediction was that a cooler chip boosts to a higher V/f
point and so costs more per token. It reads 1.6176 against the unconstrained
warmup's 2.0316, at 31.73 W against 38.17 -- 20% less power and 3.7% MORE
throughput. The reasoning failed because it assumed headroom to boost into:
both readings are already unconstrained, so frequency was at its ceiling in
each, and what forced cooling actually removes is LEAKAGE, which climbs
steeply with die temperature. Same-day same-rev support, uncooled: that
session's one Nominal performance arm read 1.7876 against 1.60-1.63 pinned.

Read that as the direction plus an order of magnitude, not a coefficient. The
cooling effect is a CROSS-CAPTURE comparison -- cooling is a property of the
whole run, so the harness cannot interleave it the way it interleaves arms --
and cross-capture absolutes are what has repeatedly failed to reproduce here
(Gotcha 22). What IS interleaved, and therefore what this section actually
establishes, is the performance-vs-efficiency row.

### What is not established

These are still this laptop's numbers with its fans held at maximum, which is
an upper-headroom operating point and not a shipping one: no user runs this
way, and the fan power itself is invisible here because `powermetrics`
Combined Power is CPU+GPU+ANE only. A chassis that holds Nominal on its own
would land somewhere between these rows and the governed ones. The uncooled
rows above remain the ones that describe what a user sees.

## Ternary-Bonsai 27B: one governed capture, and the pairs that still reproduce (2026-09-15)

ROADMAP P4.3's ternary row. `LABEL=ac MODEL=~/models/ternary27b.gturbo
scripts/power.sh 2`, rev `28ecff57` dirty, install `~/models/ternary27b.gturbo`
(manifest sha `2374453a46604330c7801146cbd88d9709cd343ffaaa477dd4e83eb71138f39e`
-- the walk's byte-reproducibility hash, so the capture is pinned to exact
weights), 2 pairs, 200 ms interval, all three cases, arms `default`, cooling
`auto`, 23:41 UTC after a day of heavy Metal work. Contamination floor
**101 mW** over 15,853 samples: a clean machine by Gotcha 43's instrument,
second-cleanest of the captures calibrated there. Per-arm rows and the
capture's provenance block are archived at
`docs/verification/power-ternary27b-2026-09-15.tsv` and
`...-system.txt`; the raw powermetrics samples stayed in `/tmp` (16.6 MB).

**The caveat that governs every number below: every measured phase left
Nominal and reached Heavy** (medium's warmups held Moderate; nothing else
held anything). Muse Glimmer at least produced Nominal warmup windows to
publish as the unconstrained point; this capture arrived heat-soaked and
deepened, so there is no unconstrained window to contrast with. Everything
here is a governed operating point, read with Gotcha 28's direction trap in
mind: throttled arms read SLOWER and simultaneously BETTER J/token, so a
Heavy row never looks broken -- it looks efficient.

### The heat-soak progression is the capture's most instructive finding

Long-synthesis prefill, identical 2,940-token work three times in a row:

| run | secs | watts | J/prompt token |
| --- | ---: | ---: | ---: |
| warmup | 212.19 | 36.36 | 2.6253 |
| p1 | 708.99 | 42.73 | **10.3056** |
| p2 | 318.83 | 40.26 | 4.3685 |

p1 took 2.2x p2's wall time and drew 2.4x its energy per token on
byte-identical work, and the warmup -- before the soak -- was cheaper than
either. The harness's 80.9% spread warning on this row is it refusing to
average the three, correctly. **Do not quote a long-synthesis row from this
capture, in either direction**; what it documents is the soak, which is
exactly the condition `COOLING=max` exists to remove.

### What reproduces even under governance

Short-explanation decode, both pairs Heavy:

| | p1 | p2 | spread |
| --- | ---: | ---: | ---: |
| watts | 31.08 | 31.14 | 0.2% |
| J/token | 2.9513 | 2.9404 | 0.4% |
| tok/s | 10.398 | 10.443 | 0.4% |

Prefill pairs read 2.9611 against 2.8104 J/token (5.2%). So the publishable
sentence is narrow and honest: **short-explanation decode on the 2-bit dense
install reads ~31 W and ~2.95 J/token at ~10.4 tok/s, HEAVY-governed,
reproducing to 0.4% within the capture.** The medium rows spread 13.0%
(decode) and 15.7% (prefill) across pairs -- record as observations, not a
row. tok/s 10.4 against this family's 14.2 (2026-08-15, `docs/BENCHMARKS.md`)
is the governance discount, ~27%, consistent with the throttle reading.

### Comparison, and what it is not

Against the other sub-4-bit power point (Gemma 4 IQ3_XXS/IQ4_NL, 0.92-0.99
J/token decode at 25-28 tok/s, mostly Nominal): ternary reads ~3.1x the
J/token at ~40% of the throughput. The direction is the one Do-Not-Revisit 12
records for sub-4-bit weights, amplified twice over -- the 2-bit GEMV is
93.1% of this install's decode compute with no `+/-1` shortcut
(`docs/MTP_SPECULATIVE.md`), and every ternary arm was Heavy where the IQ3
capture was mostly not. It is NOT a clean ablation: different architecture,
different corpus position, different thermal state, and one governed against
one free-running. Read it as indicative and wait for the row that would be
clean -- `qwen38-27b` (4-bit, same architecture, install on disk since
2026-09-15) captured in the SAME session as a ternary COOLING=max rerun.

### Still owed on this install

A `COOLING=max` capture, for the unconstrained point this capture could not
produce (the museglimmer precedent says pinned fans read BETTER J/token, so
expect the number to move down, and per that section's rule the cooled row
publishes beside this one, never instead of it).

## Qwen3.8-27B 4-bit: the pairing row, and two refused means (2026-09-16)

ROADMAP P4.3's qwen38 row, and the same-architecture 4-bit leg of the
comparison the ternary section asked for. `LABEL=ac
MODEL=~/models/qwen38-27b.gturbo scripts/power.sh 2`, rev `28ecff57` dirty,
install re-streamed 2026-09-15 (manifest sha
`ec122390a327dffa8870923a965fadcbe663f2a70c8c183f5a5c58dc9d7be8e0`), 2 pairs,
200 ms, all three cases, cooling `auto`, 02:04 UTC -- about 2.3 hours and a
cool-down after the ternary capture on the same day. Contamination floor
**170 mW** over 8,374 samples: clean again. Per-arm rows and provenance
archived at `docs/verification/power-qwen38-27b-2026-09-16.tsv` and
`...-system.txt`.

### The row that reproduces

Long-synthesis, 637-token decode at 2,940-token context, p1 Heavy and p2
Moderate -- and it does not matter:

| | p1 | p2 | spread |
| --- | ---: | ---: | ---: |
| decode W | 26.18 | 26.19 | 0.04% |
| decode J/token | 1.5089 | 1.5057 | 0.2% |
| decode tok/s | 17.327 | 17.456 | 0.7% |
| prefill J/token | 1.2705 | 1.2524 | 1.4% |

**~26.2 W and ~1.507 J/token at ~17.4 tok/s is this install's decode cost,
prefill ~24.6 W and ~1.261 J/token**, reproducing across a pressure
transition -- the tightest multi-pair row on this page, and the cheapest
J/token of the three dense installs measured (IQ3 0.92-0.99 at 25-28 tok/s,
ternary 2-bit 2.95 governed). Against this family's 19.0 tok/s
(`docs/BENCHMARKS.md`, 2026-08-15) the discount is ~9%.

### Two refused means, stated rather than smoothed

**Medium-review p2 ran HEAVY and 33% FASTER than Moderate p1.** 21.04 tok/s
against 15.49, 24.92 W against 27.73, 1.1749 against 1.7744 J/token -- every
number moved the "wrong" way for governance, on byte-identical 795-token
work, and the summary's 40.7% spread warning is refusing to average them.
The pressure label and the throughput disagree about which run was
throttled, so one of the two instruments is not measuring what it names;
until that is resolved the pair is recorded, not used. What would settle it:
a `COOLING=max` rerun (which removes the label's ambiguity) or a third pair.

**Short-explanation decodes MORE expensively than long-synthesis** -- 1.79
J/token against 1.51, at the SHALLOWER context. The arithmetic is visible in
the rows: short carries `cpu_W` 4.3-7.0 against long's 1.6-2.5, and at 15-16
tok/s that 2-4 W delta is 0.2-0.4 J/token, which is the whole gap. Combined
Power includes the CPU, the 4-bit GPU kernel is cheap enough that the fixed
host-side per-token cost is a large fraction of a short-context token, and
why the CPU drew 2-4x more in the short case (it ran first, on the coolest
machine) is the unexplained residual. Pairs spread 12.1% (decode) and 23.7%
(prefill) besides. Observations, not a row.

### The pairing, and what would still make it clean

Same architecture, same corpus, one night apart: **ternary 2-bit reads
~2.95 J/token at ~10.4 tok/s (all-Heavy), qwen38 4-bit reads ~1.51 J/token
at ~17.4 tok/s (long, mostly Nominal-adjacent)** -- 2-bit costs about 2x the
energy per token and 40% of the throughput, which is the compute-bound
2-bit reading (`docs/MTP_SPECULATIVE.md`'s 93.1%-of-decode GEMV) made
quantitative on one architecture. The caveat from the ternary section
stands but shrinks: the thermal states differ (all-Heavy against mixed),
so the clean version is still the `COOLING=max` ternary rerun, ideally in
one session with a qwen38 arm for the paired reading.

## Battery, and what differs

Battery rows are partial: they exclude runs whose thermal pressure left
Nominal, which on battery is most of the protocol.

| install | case | phase | battery | AC |
| --- | --- | --- | --- | --- |
| Gemma | short-explanation | decode | 16.53 W, 0.3702 (n=2) | 16.66 W, 0.3838 |
| Gemma | medium-review | decode | 18.58 W, 0.4568 (n=1) | 17.83 W, 0.4465 |
| Gemma | long-synthesis | decode | no clean run | 17.77 W, 0.4975 |
| Qwen | short-explanation | decode | 14.94 W, 0.3553 (n=1) | 14.30 W, 0.3513 |
| Qwen | medium-review | decode | 14.99 W, 0.3737 (n=2) | 13.78 W, 0.3517 |
| Qwen | long-synthesis | decode | no clean run | 14.88 W, 0.4337 |

**This answers AGENTS.md Gotcha 22, which had stood as a precaution rather
than a measured effect: the same binary had never been run on both power
sources.** It now has, and the answer has two halves.

**Energy is not the axis that moves.** Watts and joules-per-token differ
by a few percent with no consistent sign: Gemma `short-explanation` is
3.7% worse on AC, Qwen `medium-review` 5.9% better. That is the size of
ordinary cross-session drift in this repo, so the precaution is still the
right policy, but nobody should expect a large power-source correction.

**Thermal headroom is the axis that moves, and it is decisive.** On AC,
50 of 50 arms stayed Nominal. On battery, `long-synthesis` left Nominal on
every run of both installs, and `medium-review` (Gemma) and
`short-explanation` (Qwen) each lost one of two runs. That is why the
battery column above has holes and the AC column does not.

Throughput was 2.2-2.8% lower on AC across all four comparable cases
(Gemma short-explanation 41.80 -> 40.70 tok/s, Qwen medium-review 38.84 ->
37.76). The direction is consistent, but this is a cross-session
comparison of absolute numbers, which this repo's own record says
repeatedly fails to reproduce, and it is confounded: the surviving battery
rows are biased toward the start of that session, when the machine was
coldest. Do not carry it as a power-source effect.

## Thermal saturation, and why its direction is a trap

On battery the protocol saturates the machine before it finishes.
`long-synthesis` prefills ~3,000 tokens for 63-72 s before it decodes a
single token, and it left Nominal on every run of both installs.

The reason this needs a warning rather than a footnote is the direction.
Gemma `medium-review` happened to run once clean and once under Heavy
pressure, same binary and prompt:

| | tok/s | watts | J/token |
| --- | ---: | ---: | ---: |
| Nominal | 39.34 | 18.58 | 0.4568 |
| Heavy (throttled) | 31.47 | 10.21 | 0.3169 |

Giving up 20% of throughput bought 31% less energy per token, because
voltage-frequency scaling is superlinear. **A throttled arm therefore does
not look broken in a power table, it looks good.** The harness warns on
every non-Nominal run and its summary still averages that run in;
exclusion is by hand, from the per-arm rows in `rows.tsv`. This paragraph
claimed the opposite until 2026-08-16, when the Muse Glimmer captures above
warned on eight measured arms of eight and printed summaries built entirely
from them;
`scripts/power.sh` prints its table before the warnings and drops nothing.
Nothing else in this repo checks thermal state (AGENTS.md Gotcha 28).

That is also Phase P2's premise measured a phase early: the token-rate
limiter is a deliberate version of what the thermal governor did here
involuntarily. It remains one accidental pair, not an experiment, and P2
owes its own interleaved measurement.

## Hygiene audit

The audit asked whether anything on the token path polls. The design says
no -- the router readback is a blocking command buffer wait chosen
explicitly over spinning on `signaledValue`, and the `read_pool` workers
park on a condvar -- but that had never been checked against a power
counter.

**The stated failure signature was a flat pegged GPU.** It is not flat:

| window | samples | mean | min | max | sd/mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma short-explanation decode (AC) | 63 | 9,220 mW | 64 | 12,744 | 37% |
| Gemma long-synthesis decode (AC) | 369 | 11,326 mW | 64 | 15,893 | 21% |
| Gemma short-explanation decode (battery) | 61 | 9,119 mW | 13 | 12,853 | 39% |

GPU power swings across three orders of magnitude within a decode window,
which is the per-token phase structure showing through, not a busy-wait.
The audit passes on both power sources.

The rest of the expected profile holds: CPU package power sits at 3.4-4.6 W
during decode against 10-14 W of GPU, so the host side is cheap and this
is a GPU- and I/O-bound decoder.

**Do not draw conclusions from the E-cluster residency column.** It is
system-wide rather than per-process, and it is not stable across
invocations: the identical Gemma `short-explanation` decode, in the same
battery session at the same settings, read 70.4 / 71.2 in one run
of the harness and 87.3 / 87.1 / 87.6 in another. A 16-point swing on
identical work means the column cannot support a claim about where this
process runs. It is reported for context only. (M4 Max also has two
performance clusters, P0 and P1, which the harness averages into one P
column.)

## The efficiency profile: measured, and it works

ROADMAP Phase P2's gate. Gemma, `short-explanation`, AC, 3 pairs, arms
alternating within each pair. Every arm held Nominal thermal pressure, so
nothing is filtered. `performance` is the shipped default (no cap, no
thermal stepping); `efficiency` caps decode at reading speed, 10 tok/s.

| pair | performance J/tok | efficiency J/tok | delta |
| --- | ---: | ---: | ---: |
| 1 | 0.4593 | 0.2368 | -48.4% |
| 2 | 0.3616 | 0.2333 | -35.5% |
| 3 | 0.3798 | 0.2248 | -40.8% |

**Decode energy per token falls by a third to a half for a 4.1x slowdown**
(41.1 -> 10.0 tok/s). Read the worst pair, -35.5%: pair 1's
`performance` arm is a high outlier (0.4593 against 0.3616/0.3798, and
39.02 tok/s against 41.09/41.12), which is the first measured process of
the session paying a DVFS ramp the per-process warmup does not cover.

The mechanism is visible in the split: GPU power drops 11.88 -> 1.27 W
while CPU drops 5.07 -> 1.07 W. The GPU is genuinely idle between paced
steps, which is what the sleep's placement in `raw_completion::decode`
(after the continue decision, before the next `produce`) is for. A cap
implemented inside a forward pass would show the throughput loss and none
of the energy win.

**No `performance`-mode regression.** Its clean pairs read 0.3616 and
0.3798 (mean 0.3707) against the 0.3838 frozen above, and 41.09/41.12
tok/s against 40.70. That comparison is cross-session and cross-binary,
though (Phase G, Phase S and the Gotcha 27 determinism fix all landed
between the two captures), so it is corroboration, not proof. The proof
that the default path is unchanged is structural: `RateControl::default()`
leaves both fields `None`, `RateControl::is_active` is false, and the
loop executes the identical statement sequence.

**Do not quote this run's prefill rows.** The summary reports
`efficiency` prefill at 0.2497 J/token against `performance`'s 0.3870,
which reads like a 35% prefill win and is an artifact. Prefill is not
paced at all (the `Pacer` is constructed inside `decode`, after prefill
has finished), and the measurement agrees: prefill takes the same 1.3-1.6
s over the same 61 tokens in both arms. Same work in the same time cannot
cost less energy. What happens instead is window attribution: the
prefill/decode boundary is computed arithmetically from the footer's
prefill seconds, a prefill window is only 5-8 samples at 200 ms, and on
the `efficiency` arm the far side of that boundary is 2.3 W rather than
16 W, so a single straddling sample drags the short window down hard. The
same leak exists on the `performance` arm and is invisible there because
both sides of the boundary draw about the same. The tell that these rows
are noise regardless: `performance` prefill alone ranges 0.3011 to 0.5143
across three pairs, a 71% spread.

## Read-pool QoS: measured, and not wired

Rust std threads carry no QoS class at all, while Swift's I/O pool runs at
`.utility` on E-cores. `TURBOSPARK_READ_QOS=utility`
(`crates/streaming/src/read_pool.rs`) puts the 8 `read_pool` workers on
`QOS_CLASS_UTILITY`. It is off by default.

Gemma, `short-explanation`, 3 pairs, arms alternating within each pair
rather than as two consecutive batches:

| pair | J/token, AC | decode s, AC | J/token, battery | decode s, battery |
| --- | --- | --- | --- | --- |
| 1 | 0.3710 -> 0.3746 (+1.0%) | +0.08% | 0.3927 -> 0.3991 (+1.6%) | +0.16% |
| 2 | 0.3757 -> 0.3792 (+0.9%) | +0.08% | 0.4008 -> 0.4039 (+0.8%) | +0.88% |
| 3 | 0.3753 -> 0.3724 (-0.8%) | +0.08% | 0.3937 -> 0.4283 (+8.8%) | +2.34% |

**On AC this is a null result, and the AC session is the one to believe.**
Energy changes by +1.0% / +0.9% / -0.8%; the sign flips, so the effect
is not distinguishable from noise. Decode time is +0.08% in all three
pairs, which is 0.01 s on 12.5 s: consistently non-negative, and at the
resolution floor.

The battery session read this as a clear loss on both axes. It was not:
its pair 3 (+8.8% energy, +2.3% time) is thermal drift, and the two
sessions disagree by more than the effect being measured. Correcting that
reading is itself the lesson -- an A/B this small needs the thermally
stable power source.

**It is still not wired**, because a knob that measurably does nothing is
complexity without payment. What can be said about the mechanism is only
structural, not measured: these threads sit on the decode critical path
(`run_batch` blocks until every claim drops) doing a page-cache memcpy
that measured 32 GiB/s, so there is no idle to reclaim and the upside was
always going to be small. An earlier draft of this page claimed the
E-residency column showed utility failing to obtain E-cores; that claim is
withdrawn, because the column is too unstable to support it (above).

The seam stays in the tree, off, as a documented dead end so that Phase P2
does not re-derive it. A profile that wants to reach for thread QoS needs
a different lever than this one.

## Caveats

- **Watts are CPU+GPU+ANE, not wall.**
- **There is no published wall-power figure.** The battery gauge (`ioreg`
  `InstantAmperage` x `Voltage`, no root) read a standard deviation of
  20-40% of its own mean across a run, and reported 48.8 W against 64.0 W
  for two arms doing identical work. It is good enough to say the machine
  draws roughly 50-70 W under load against ~20 W idle, and not good enough
  for a wall joules-per-token. The gap between ~17 W of CPU+GPU and ~60 W
  at the battery is mostly display and rest-of-SoC and stays
  unattributed. On AC the column is empty by construction.
- **Two measured runs per arm**, three pairs for the QoS A/B. Enough for
  the shape; not enough for a few percent.
- **The E-cluster residency column supports no conclusions.** See the
  audit section.
- **The timeline is reconstructed, and its drift is measured.**
  `powermetrics` timestamps samples only to the second, so the harness
  accumulates each sample's reported elapsed figure from a wall-clock
  start. Drift is printed every run and warned on past 2 s. It measured
  -0.6 to -0.8 s over spans of 205 to 820 s, so window alignment is not a
  live concern at this session length.
- **The two families' figures are comparable to each other** (same
  machine, session, protocol and slot count), unlike their perplexity
  numbers, which are not.

## Still owed

- **Per-process attribution.** Everything here is system-wide. The runs
  were made with no other model process up (the harness refuses to start
  otherwise), but that is isolation by convention, not by measurement. It
  is also how one battery row was caught and dropped by hand rather than
  by the thermal flag: Qwen `long-synthesis` p1 reported `cpu_W = 18.40`
  against roughly 4.3 W everywhere else, which is a competing process.
  **That refusal only covers other model processes, and the gpt-oss
  capture of 2026-08-13 shows what the gap costs**: an ordinary desktop
  UI, well under the 18.40 W that made the Qwen row obvious, moved a
  decode row 37% with every arm Nominal. See the gpt-oss section and
  AGENTS.md Gotcha 43.
- ~~**Ornith-1.5 35B-A3B against Qwen 3.6 in ONE session.**~~ DESCOPED
  2026-09-16 (user decision): the two installs had gone missing from disk
  and the A/B wanted ~37 GB of re-pulls for one comparison row. The
  separate 0.4731 and 0.3513 rows stand with their cross-session caveat --
  two weeks and one binary apart, shared architecture, which is why the gap
  was always the least trustworthy number on this page. Re-open by pulling
  both installs and running one interleaved capture.
- **The other two Ornith cases**, and a capture for `ornith9b`. The row above
  is `short-explanation` alone.
- **A wall-power number**, which needs an external meter rather than the
  battery gauge.
- **A rate-cap SWEEP.** `READING_SPEED_TOK_PER_SEC` is 10.0 and the A/B above
  prices it at 14.9% of the energy for 51.5% of the throughput, which is a
  poor trade -- but whether the cap is badly PLACED or the idea is badly
  SHAPED cannot be told from two points on a curve. `scripts/power.sh` takes
  numeric arms since 2026-08-18 (`ARMS=default,30,20,15,10`), so the sweep is
  one interleaved capture under pinned fans. Run it on `gemma4` rather than
  Muse Glimmer: the constant is global and gemma4 decodes ~44 tok/s, so the
  arms span 4.4x against 1.9x. `SERIOUS_TOK_PER_SEC` and
  `CRITICAL_TOK_PER_SEC` are a THERMAL ladder and this does not settle them --
  their job is shedding heat, not saving joules, and Gotcha 28's trap means an
  energy curve cannot be read as a ladder placement.

**A THIRD ENTRY WAS RETIRED 2026-08-22.** "A dispersion line in
`scripts/power.sh`'s summary" asked for a min/max column so that two arms 37%
apart could not read as one number. It is there: the summary carries a
`J/tok±` spread column and warns per group over 10%, and beside it a
`contamination floor` line that reports the minimum CPU power anywhere in the
log and warns over 2,000 mW. The floor is the addition the entry did not ask
for and the Ornith captures showed was the necessary one -- dispersion is
blind to a STEADY load by construction. Calibrated on three real captures
here: 94 mW and 392 mW on the two clean DFlash2 ones against 3,361 mW on a
contaminated one, and verified silent on the former before being believed.

Two entries were retired here on 2026-08-18 rather than left standing, because
the cooled A/B answered both and a "still owed" item that has quietly been
paid is the rot mode this repo has already audited for once. **A sustained
J/token row for Muse Glimmer 30B** and **a fair `performance,efficiency` A/B
on it** both named the same missing condition -- hardware that can hold
Nominal in performance mode for a whole case -- and forced cooling supplied
it. See "Forced cooling: the A/B, run" above. What remains genuinely open on
that install is a row on a chassis that holds Nominal on its OWN, since a
pinned-fan row is an upper-headroom operating point no user occupies.
