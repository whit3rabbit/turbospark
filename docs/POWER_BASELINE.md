# Power baseline

Watts and joules-per-token for both real installs, over the frozen
community protocol, on AC and on battery. ROADMAP Phase P1.

This is NOT a parity claim. Swift was never measured for power, here or
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
  QOS=default,utility scripts/power.sh 3
```

## Run provenance

| | |
| --- | --- |
| Date | 2026-08-07, two sessions: 23:58-00:40 UTC (battery) and 00:56-01:26 UTC (AC) |
| Chip | Apple M4 Max, 36 GB |
| macOS | 26.5.2 (25F84) |
| Installs | `~/models/gemma4.gturbo`, `~/models/qwen36.gturbo` |
| Expert-cache slots | 16 (protocol default) |
| Protocol | frozen `real-generation-v1`, seeds 20260721-23, temp 0.2, top-k 64, top-p 0.95, 1024 new-token budget, 4K context |
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
pressure -- 50 of 50 sampled arms -- so nothing here is filtered and every
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
by a few percent with NO consistent sign: Gemma `short-explanation` is
3.7% worse on AC, Qwen `medium-review` 5.9% better. That is the size of
ordinary cross-session drift in this repo, so the precaution is still the
right policy, but nobody should expect a large power-source correction.

**Thermal headroom is the axis that moves, and it is decisive.** On AC,
50 of 50 arms stayed Nominal. On battery, `long-synthesis` left Nominal on
every run of BOTH installs, and `medium-review` (Gemma) and
`short-explanation` (Qwen) each lost one of two runs. That is why the
battery column above has holes and the AC column does not.

Throughput was 2.2-2.8% LOWER on AC across all four comparable cases
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
not look broken in a power table, it looks good.** The harness flags every
non-Nominal run and refuses to average it in; nothing else in this repo
checks thermal state (AGENTS.md Gotcha 28).

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
battery session at the same settings, read 70.4 / 71.2 in one invocation
of the harness and 87.3 / 87.1 / 87.6 in another. A 16-point swing on
identical work means the column cannot support a claim about where this
process runs. It is reported for context only. (M4 Max also has TWO
performance clusters, P0 and P1, which the harness averages into one P
column.)

## Read-pool QoS: measured, and NOT wired

Rust std threads carry no QoS class at all, while Swift's I/O pool runs at
`.utility` on E-cores. `MFERENCE_READ_QOS=utility`
(`crates/streaming/src/read_pool.rs`) puts the 8 `read_pool` workers on
`QOS_CLASS_UTILITY`. It is off by default.

Gemma, `short-explanation`, 3 pairs, arms alternating WITHIN each pair
rather than as two consecutive batches:

| pair | J/token, AC | decode s, AC | J/token, battery | decode s, battery |
| --- | --- | --- | --- | --- |
| 1 | 0.3710 -> 0.3746 (+1.0%) | +0.08% | 0.3927 -> 0.3991 (+1.6%) | +0.16% |
| 2 | 0.3757 -> 0.3792 (+0.9%) | +0.08% | 0.4008 -> 0.4039 (+0.8%) | +0.88% |
| 3 | 0.3753 -> 0.3724 (-0.8%) | +0.08% | 0.3937 -> 0.4283 (+8.8%) | +2.34% |

**On AC this is a null result, and the AC session is the one to believe.**
Energy changes by +1.0% / +0.9% / -0.8% -- the sign flips, so the effect
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
E-residency column showed UTILITY failing to obtain E-cores; that claim is
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
- **A wall-power number**, which needs an external meter rather than the
  battery gauge.
