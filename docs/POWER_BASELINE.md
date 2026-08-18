# Power baseline

Watts and joules-per-token for both real installs, over the frozen
community protocol, on AC and on battery. ROADMAP Phase P1.

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
It takes `default`/`utility`, which set `MFERENCE_READ_QOS`, or
`performance`/`balanced`/`efficiency`, which pass `--power-profile` to the
bench (ROADMAP Phase P2). The two kinds cannot be mixed in one run: the
arm name is a single column of `rows.tsv` and a single grouping key in the
summary, so mixing them would compare two different seams under one label.

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
`.utility` on E-cores. `MFERENCE_READ_QOS=utility`
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
- **A dispersion line in `scripts/power.sh`'s summary.** It prints a mean
  per case and phase, so two arms 37% apart on identical work look like
  one number. The per-arm values are in `rows.tsv` and reading them is
  currently a manual step; a min/max column would make contamination
  visible where the row is published.
- **A wall-power number**, which needs an external meter rather than the
  battery gauge.
- **A sustained J/token row for Muse Glimmer 30B.** Its unconstrained cost is
  established (n=2, ~2.02-2.04 J/token) and its governed cost is unstable;
  what is missing is the cost on a machine that can hold Nominal for a whole
  case, which needs more thermal headroom than this laptop has.
- **A fair `performance,efficiency` A/B on Muse Glimmer 30B.** Run
  2026-08-16 and inconclusive: the performance arm throttled on every pair and
  its 25% spread swallows the efficiency arm's range. Needs hardware that can
  hold Nominal in performance mode for a whole case. What the run did settle
  is that the cap holds 10.00 tok/s to 0.01% and keeps the machine out of
  thermal governance entirely.
