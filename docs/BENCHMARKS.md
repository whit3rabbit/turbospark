# Benchmarks: this port against the Swift original

One machine, one model install, one session, both engines. Every other
figure in this repo and in `docs/BENCHMARKING.md` is this port measured
against its own past self; this is the parity number.

Reproduce with `scripts/parity.sh`. Read `docs/BENCHMARKING.md` for the
harness itself (the three `mference-bench` modes, the memory oracle, and
how memory is sampled).

## Run provenance

| | |
| --- | --- |
| Date | 2026-08-07 |
| Chip | Apple M4 Max, 36 GB |
| macOS | 26.5.2 (25F84) |
| Power | AC |
| This port | `ef4e953` plus the partial-ranking sampler change |
| Swift (`../Mference`) | `1bb585c`, Swift 6.3.3, release build |
| Install | `~/models/gemma4.gturbo`, Gemma 4 26B-A4B, written by this port's repack |
| `manifest.json` sha256 | `d4eb5607509240363c126e743abf7b2f6040c7f3e92347044e8b2f7df878ce9f` |

Both engines read the SAME install directory. The Swift `MferenceCLI`
opens this port's `.gturbo` output unmodified, under its default
`.fullSha256` integrity policy, so no format difference is hiding in the
comparison.

Protocol: frozen `real-generation-v1`, three cases, seeds 20260721-23,
temperature 0.2, top-k 64, top-p 0.95, 1024 new-token budget, 4K context,
16 expert-cache slots (both engines' default). The three source prompt
JSONs hash to the SHA-256s recorded in `crates/bench/src/protocol.rs`, so
both engines tokenize identical bytes. One discarded warmup per engine per
case, then two measured runs, arms interleaved. Every run reported
`stop=endOfTurn`.

## Decode throughput

The comparable column. Decode-only, `new_tokens / decode_seconds`, the
same definition both engines' footers use.

| Case | Prompt tok | Swift tok/s | This port tok/s | Ratio |
| --- | ---: | ---: | ---: | ---: |
| short-explanation | 61 | 41.241 / 40.885 | 40.763 / 40.668 | 0.99 |
| medium-review | 430 | 38.565 / 38.639 | 38.417 / 38.468 | 1.00 |
| long-synthesis | 3,015 | 34.315 / 34.360 | 34.556 / 34.637 | 1.01 |

**This port decodes at Swift's rate, within 1 percent, on the same
hardware and the same install.** The two runs within each arm agree to
0.1 tok/s or better, and the spread between engines is smaller than that
band on two of the three cases, so neither engine is measurably ahead.

Generated token counts differ between engines (516/780/617 for Swift
against 510/691/565 here) because the two samplers walk different RNG
streams. Both stop at `endOfTurn` on coherent text, and tok/s is a rate,
so this does not bias the comparison.

### What this replaces, and why the first number was so wrong

The first parity run on this hardware (same script, same install, six
days' worth of the same code) measured **0.64 to 0.67 of Swift**, and
recorded the cause as unattributed. It has since been attributed and
fixed, and the finding is worth keeping because of what hid it:

`selection::select` full-sorted the entire candidate domain to rank it.
At Gemma 4's vocabulary of 262,144 that sort cost **~18.9 ms per token**,
measured directly (`cargo test -p mrefrust-selection --release --test
rank_top_k -- --ignored --nocapture`), against a whole forward pass of
roughly 25 ms. Both truncation steps only ever keep a PREFIX of the ranked
order, so with `top_k` enabled everything past rank 64 was sorted and
thrown away. Replacing the sort with a partial selection
(`truncation::rank_top_k`, `select_nth_unstable_by` + a sort of the
surviving 64) took that to 2.05 ms and decode from 25.5 to 39.6 tok/s on
a fixed prompt, with byte-identical output.

Three things made this expensive to find, all of them general:

- **It is not in the engine.** Every phase bucket
  (`MFERENCE_PHASES=1`), every GPU-busy attribution, and every dispatch
  ranking this port has ever printed measures the inside of
  `LogitProducer::produce`. The sampler runs in the decode loop AFTER
  `produce` returns, so it appeared in NONE of them. The tell was
  arithmetic, not instrumentation: the phase report's own total came to
  26.1 s against a 41.3 s decode wall clock, and nobody had subtracted
  those two numbers before.
- **Greedy could not see it.** The repo's greedy smoke test passes
  `--temperature 0.0001`, which is not exactly zero, so it took the
  sampled path and paid the same sort. The argmax fast path (exactly
  `--temperature 0`) ran at 45.2 tok/s the whole time.
- **The suspects were all GPU-side.** `DEVIATIONS.md` named the
  `MTLSharedEvent` overlap and GPU-side sampling (`logit.metal`'s
  `sample`) as the unported decode items. The first was already bought
  another way and measured; the second was the right neighbourhood for
  the wrong reason. Swift samples on the GPU, so it never pays a host
  sort -- but the fix here was not to port that kernel, it was to stop
  sorting 262,080 candidates nobody would look at.

## Prefill

Not the same algorithm on both sides, so this is a scope difference, not a
regression. Swift chunks prefill (`--prefill-chunk`, default 128); this
port runs one sequential forward pass per prompt token because the tile
kernels are descoped (`DEVIATIONS.md`).

| Case | Prompt tok | Swift prefill | This port prefill |
| --- | ---: | ---: | ---: |
| short-explanation | 61 | 5.27 / 5.55 s | 1.23 / 1.24 s |
| medium-review | 430 | 7.41 / 7.36 s | 8.13 / 8.13 s |
| long-synthesis | 3,015 | 27.51 / 27.48 s | 64.59 / 64.62 s |

Fitting the two endpoints:

- Swift: about 5.1 s fixed plus 7.5 ms per prompt token.
- This port: no measurable fixed cost, 21.4 ms per prompt token.

Unchanged by the sampler fix, as it must be: prefill selects no tokens.
That per-token figure independently reproduces the 21 ms this port
measured for itself on 2026-08-06 (CLAUDE.local.md's prefill attribution
table), from a completely different measurement path.

The crossover is near 350 prompt tokens. Below it this port is faster to
first token, because Swift pays a fixed startup this port does not; above
it Swift pulls ahead and keeps going, because a chunk of 128 amortizes
weight reads across 128 tokens where this port re-reads per token.
**Prefill is now the only measured gap against Swift**, and it is a known
scope decision rather than an open question.

## Memory

Peak `phys_footprint` is the headline counter: it is what
`../Mference/docs/BENCHMARKS.md` reports, what the Swift README's "26B
total, ~3.88B active per token, in ~2 GB of memory" claim rests on, and
what this port's `AppMemorySampler` samples. `/usr/bin/time -l` reports it
for any process as `peak memory footprint`, so it is available for BOTH
engines even though the Swift CLI prints no memory line of its own.

| Case | Swift footprint | This port footprint | Delta |
| --- | ---: | ---: | ---: |
| short-explanation | 2,218 / 2,217 MiB | 2,182 / 2,181 MiB | -36 / -36 MiB |
| medium-review | 2,235 / 2,219 MiB | 2,180 / 2,108 MiB | -55 / -111 MiB |
| long-synthesis | 2,235 / 2,218 MiB | 2,180 / 2,180 MiB | -55 / -38 MiB |

**This port holds the ~2 GB working set, and does it in about 2 to 5
percent less peak footprint than Swift on the same machine and install.**
Both engines land in the 2.1 to 2.2 GiB band on a 26B model with a 14 GB
install, which is the property the design exists to deliver.

The harness asymmetry works AGAINST this port here, so the delta is if
anything understated: one `mference-bench --case` launch runs the
protocol's discarded warmup AND the measured run in the same process, so
its figure is a peak over two generations, while each Swift figure covers
one. Swift's number is also notably flat near 2,235 MiB, which reads like
a ceiling its allocator reaches and holds rather than a workload-driven
peak.

Two secondary observations:

- **The in-process sampler is validated.** `AppMemorySampler`'s reported
  session peak matched the kernel's own high-water mark from
  `/usr/bin/time -l` to 0.1 MiB on all six runs. Sampling every 8th token
  is not missing a transient peak on this workload.
- **RSS goes the other way and is the less useful counter.** Peak RSS was
  1,682 to 1,831 MiB for Swift against 1,991 to 1,993 MiB here. RSS counts
  resident pages including clean file-backed ones, so it moves with how
  much of the 14 GB mapped install each engine happens to be touching;
  footprint is the counter that tracks what the process actually costs the
  system. Reported for completeness, not as a gap.

Published Swift rows for other hardware, for context: 2,126 to 2,142 MiB
on a 24 GB M5 Pro, 1,776 to 1,971 MiB on an 8 GB M2. Swift reads slightly
higher here (2,217 to 2,235) than its own published M5 Pro band, on a
different chip and OS build, so do not treat the M4 Max numbers above as
transferable to those rows.

## Expert-cache slots: the one runtime control that moves this

Both engines default to 16 slots and the table above is measured there.
`mference-bench --model` can now vary it (`--expert-cache-slots`, allowed
8/16/24/32, matching `MferenceCLI`'s flag), which is what the comparison
needed to be honest about the default. Same case, same session,
interleaved pairs:

| Slots | Decode tok/s | Peak `phys_footprint` |
| ---: | ---: | ---: |
| 16 | 40.988 / 40.381 | 2,180 / 2,109 MiB |
| 32 | 47.051 / 46.827 | 3,728 / 3,654 MiB |

32 slots buys about 15 percent decode and costs about 1.5 GB, which
leaves the ~2 GB working-set claim behind entirely. That is why 16 is
both engines' default, why every published number here is measured at 16,
and why the memory oracle's ceiling only means anything at 16
(`protocol::PROTOCOL_EXPERT_CACHE_SLOTS`). Output is NOT identical across
slot counts: the hit/miss split permutes the phase-2 reduce order and FP
addition is not associative. Compare within one slot count.

This also reconciles a discrepancy that stood open in `DEVIATIONS.md`:
42.6 tok/s recorded on this checkpoint against the 25.6 the first parity
run measured. The two are separated by both axes above -- the sampler
(worth ~15 tok/s at this vocabulary) and the slot count (worth ~6) -- and
the 42.6 sits inside the range they span. The settings behind the 42.6
were not recorded, so it is retired rather than re-explained.

## Quality

The one axis with no Swift column. The Swift original publishes no
perplexity, no KL divergence, and no golden output, so there is nothing to
compare against there; ROADMAP Phase Q exists to build the axis anyway,
before Phase S touches quantization.

Read the sections below in two groups. The perplexity, the digests, the
constrained-cache arm, and the sensitivity curve are all this port measured
against ITS OWN PAST -- regression sentinels, and no row in them is or can
be a parity claim. The cross-engine section at the end is the exception and
the only external reference in this document's Quality half: it compares
this port against mlx-lm on the same quantized bytes.

Reproduce with the two gates (about a minute each), which assert these
values on this chip and print them on any other:

```sh
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test quality_gate --release -- --ignored --nocapture
```

| Install | Reference perplexity | Greedy digest | Sampled digest | Greedy at 8 slots |
| --- | ---: | --- | --- | --- |
| Gemma 4 26B-A4B | 37.3105 | `4f5cba92` | `cde6012a` | `a50ed69d` |
| Qwen 3.6 35B-A3B | 6.2536 | `c5b52f77` | `525cadbc` | `c5b52f77` |

Digests are the leading 8 hex characters of the SHA-256 of the generated
text; the full values live in each gate's `BASELINES` row. Measured
2026-08-07 on the machine in the provenance table above, on AC, at 16
expert-cache slots. Two runs per install in separate processes agreed on
every digit and every hex character.

Four things to know before reading those numbers:

- **Only assistant-position tokens are scored.** Both checkpoints are
  instruction-tuned, and instruction tuning masks the loss on the prompt.
  Teacher-forcing the Gemma install over PROMPT text measured a mean NLL of
  15.3 nats against a uniform-distribution bound of 12.5, which is worse
  than guessing, while assistant-side tokens in the same sequence scored
  0.000. The measurement was not at fault: replaying the model's own greedy
  output reproduced 39 of 40 tokens, the one miss a genuine near-tie at
  0.96 nats. So the corpus is a fixed reference ANSWER
  (`crates/bench/prompts/quality-v1/assistant-reference.txt`, original
  prose) placed in the assistant slot after the frozen protocol's first
  prompt.
- **The two perplexities are not comparable to each other.** Gemma 4's chat
  template opens a `<|channel>thought` block before the assistant slot, so
  its number scores the reference answer as internal reasoning; Qwen 3.6's
  template opens no channel, so its number scores the same passage as a
  reply. That, not model quality, is most of the 37.31 against 6.25. Each
  number is only comparable to its own past.
- **Neither is comparable to a published perplexity.** One install, one
  passage, this port's tokenizer and template. It is a regression sentinel.
- **Digests depend on expert-cache state**, so the gate's run order is part
  of the protocol: warmup, measure, measure, warmup, measure. Each layer's
  routed slots are ordered cache misses first then hits, which permutes the
  phase-2 reduce order, and FP addition is not associative. A cold-cache
  generation does not match a warm one, measured directly here. Everything
  is pinned to 16 slots for the same reason the tok/s rows are.

### Constrained working set

The last column halves the routed-expert cache to 8 slots, which is this
port's memory knob (16 -> 32 slots cost 1.5 GiB; see the sweep above), and
repeats the greedy generation. Upstream's acceptance proof for that case is
"byte-identical output at unchanged throughput under a constrained working
set". Measured here, on the same day and machine as the rows above:

| Install | tok/s at 16 slots | tok/s at 8 slots | Ratio | Bytes identical |
| --- | ---: | ---: | ---: | --- |
| Gemma 4 26B-A4B | 47.620 | 41.120 | 0.86x | no |
| Qwen 3.6 35B-A3B | 43.966 | 40.174 | 0.91x | yes |

Throughput degrades rather than collapsing, on both. Byte identity splits
by family, and the reason is mechanical: `real_forward_gemma4.rs` orders a
layer's routed slots misses first then hits (so the hits' phase-1 GEMV can
ride its own command buffer), that order feeds phase 2's reduce, and FP
addition is not associative, so changing the hit/miss split changes Gemma's
bytes. `real_forward_qwen.rs` does no such reordering and comes out
identical. The ordering is unconditional, so `MFERENCE_HIT_CB=0` does not
explain or remove the difference: run that way, all four Gemma values above
are unchanged.

The gate therefore freezes a second Gemma digest rather than asserting
identity across slot counts, which would be asserting FP associativity.

### Sensitivity: what the perplexity number can actually see

A frozen number with a 2% band is only worth its band if real damage lands
outside it. `crates/bench/tests/quality_sensitivity.rs` measures that
directly: it APFS-clones the install (`clonefile`, 13 GB in ~8 ms, so only
written pages cost disk), flips ONE quantization level in a strided subset
of the routed-expert blobs, and re-measures. XOR `0x01` into an int4 byte
moves that weight by one of its sixteen levels and cannot make a NaN or an
infinity even if it lands on an FP16 scale, so what is being measured is
degradation, not breakage. Nothing outside `packed_experts/` is touched, so
the move is attributable to routed-expert weights alone.

Gemma 4, clean perplexity 37.3105, 2026-08-07:

| Expert bytes touched | Damaged perplexity | Drift | Verdict |
| ---: | ---: | ---: | --- |
| 12.5% | 12,249,392 | +3e7% | model destroyed |
| 0.195% | 51.3597 | +37.7% | detected, 19x the band |
| 0.0122% | 41.2186 | +10.5% | detected, 5x the band |
| 0.0015% | 37.5118 | +0.54% | NOT detected, inside the band |

So the gate's floor sits between 0.0015% and 0.0122% of expert bytes at one
quantization level, and Phase S's expected damage (whole percent) is orders
of magnitude above it. The test asserts the 0.195% row, chosen for margin
rather than for being the smallest detectable damage, so it cannot flake.
Every number here reproduced exactly across runs.

### Cross-engine: token-level KL divergence against mlx-lm

The table above establishes that the metric responds to damage. It cannot
say whether the undamaged starting point is RIGHT, because every number in
it is this port measured against itself. That is what this section adds:
the same corpus, the same token ids, and the same quantized checkpoint run
through a second engine.

`crates/bench/tests/logit_dump.rs` writes this port's full-vocabulary
logits for the quality corpus plus the exact id sequence it walked;
`scripts/kld.py` replays those IDS (never the prose, so no tokenizer or
chat-template difference can masquerade as a numerics gap) through mlx-lm
on `mlx-community/gemma-4-26b-a4b-it-4bit` at revision `0d77464e`, the
exact repo `~/models/gemma4.gturbo` was repacked from. Both heads return
`softcap * tanh(z / softcap)` at softcap 30 and neither normalizes, which
was read out of `mlx_lm/models/gemma4_text.py` rather than assumed.

Gemma 4, 550 positions, 16 expert-cache slots, 2026-08-07, on battery:

| Comparison | Mean KL | Median | p99 | Top-1 agree |
| --- | ---: | ---: | ---: | ---: |
| this port cold vs this port warm | 0.0019 | 0.00007 | 0.026 | 99.1% |
| **this port vs mlx-lm, both cached** | **0.0264** | **0.0022** | **0.640** | **95.6%** |
| mlx-lm batched vs mlx-lm cached | 0.0352 | 0.0028 | 0.938 | 96.0% |

**The cross-engine number is smaller than mlx-lm's disagreement with
itself.** That third row is the whole reason the second one is readable: a
KL between two engines has no natural scale, so mlx-lm is run against its
own two forward shapes (one batched pass over the sequence, and the same
sequence stepped token by token through a cache), which holds the weights,
the kernels, and the engine fixed and varies only the reduce shape. At 4
bits that alone costs 0.0352 mean nats and 4% of the argmaxes. This port
lands under it. There is no kernel gap detectable at this resolution, so a
Phase S quality delta is attributable to the quantization.

The first row is the matching floor from this port's side, and it is
independently useful: cache state alone (Gemma's misses-first slot order
permuting phase 2's reduce, AGENTS.md Gotcha 27) moves the distribution by
0.0019 mean nats. The KL is reported both ways; the forward and reverse
means agree to within 5% on every row above, so none of this is an artifact
of which distribution is treated as the reference.

Perplexity on the same four passes, which cross-validates the whole
pipeline end to end:

| Reading | Perplexity |
| --- | ---: |
| this port, cold cache | 37.3105 |
| this port, warm cache | 37.5059 |
| mlx-lm, batched | 37.4479 |
| mlx-lm, cached | 37.5301 |

A 0.6% spread across two engines and two cache states. The cold reading
reproduces `quality_gate`'s frozen row **to the last digit**, which is what
proves the dump is measuring the same thing the gate is: the gate takes its
perplexity first thing in the process, so its number is a cold one, and
`MREFRUST_LOGIT_DUMP_COLD=1` reproduces that condition. This port's own two
cache states are 0.52% apart, essentially the +0.54% of the 0.0015% damage
row above that the gate does NOT detect: cache state alone sits at the
gate's detection floor, which is a second and independent bound on it.

Everything here reproduced exactly across separate processes: the port's
logit dump is byte-identical run to run (SHA-256 `ee22f854...`), and
`kld.py`'s output diffs clean.

Caveats. One corpus, one family, one machine. mlx-lm returns bfloat16,
whose 8 mantissa bits are strictly coarser than this port's f16 storage at
these softcapped magnitudes, so there is no f16 storage floor to subtract
(measured: 3.5e-22 nats) and mlx is the lower-precision side, not this
port. Qwen 3.6 has no cross-engine number: `logit_dump.rs` accepts
`MREFRUST_QWEN36_INSTALL_DIR` and would produce one, but `kld.py`'s
reference is pinned to the Gemma repo.

## Power

NOT A PARITY CLAIM. Swift was never measured for power, here or upstream;
this is this port measuring itself, like the Quality section above.

**Full write-up, method, hygiene audit and caveats: `docs/POWER_BASELINE.md`.**
Reproduce with `scripts/power.sh`. ROADMAP Phase P1.

Measured 2026-08-07 across two sessions, AC and battery, one binary. 16
expert-cache slots, frozen protocol, `powermetrics` at 200 ms windowed to
the measured run alone by the `[power-window ...]` markers
`mference-bench` emits. Watts are CPU+GPU+ANE, not wall. The AC rows below
are the baseline: every run of both installs held Nominal thermal
pressure, so all are n=2 and none is filtered.

| install | case | tok/s | watts | J/token |
| --- | --- | ---: | ---: | ---: |
| Gemma 4 26B-A4B | short-explanation | 40.70 | 16.66 | 0.3838 |
| Gemma 4 26B-A4B | medium-review | 38.40 | 17.83 | 0.4465 |
| Gemma 4 26B-A4B | long-synthesis | 34.75 | 17.77 | 0.4975 |
| Qwen 3.6 35B-A3B | short-explanation | 38.83 | 14.30 | 0.3513 |
| Qwen 3.6 35B-A3B | medium-review | 37.76 | 13.78 | 0.3517 |
| Qwen 3.6 35B-A3B | long-synthesis | 33.78 | 14.88 | 0.4337 |

Qwen is the more efficient engine here, 0.35 J/token against Gemma's
0.38-0.45, almost entirely from GPU power (10.3 W against 12.3-13.6 W):
the hybrid linear-attention design showing up on the power axis the way it
already does on memory. Energy per token grows with context on both.

Three results worth carrying, each detailed in `docs/POWER_BASELINE.md`:

- **AC vs battery answers AGENTS.md Gotcha 22, which had stood unmeasured.**
  Energy is NOT the axis that moves: watts and J/token differ by a few
  percent with no consistent sign. THERMAL HEADROOM is: on AC 50 of 50
  arms held Nominal, while on battery `long-synthesis` left Nominal on
  every run of both installs and two further runs were lost the same way,
  so the battery column has holes the AC column does not.
- **Throttling BUYS efficiency, and so flatters a power table.** The same
  Gemma case, clean against Heavy pressure: 39.34 tok/s at 0.4568 J/token
  against 31.47 tok/s at 0.3169. Twenty percent less throughput for 31%
  less energy per token. That is Phase P2's premise, measured by accident,
  and AGENTS.md Gotcha 28.
- **Nothing is spinning.** GPU power over a decode window swings from
  64 mW to 15,893 mW (standard deviation 21-39% of mean, on both power
  sources), which is the per-token phase structure rather than a
  busy-wait. `MFERENCE_READ_QOS=utility` on the read pool measured as a
  NULL result on AC (+1.0% / +0.9% / -0.8% energy, sign flipping) and is
  NOT wired. The battery session read it as a clear loss; that reading was
  thermal drift.

## Caveats worth repeating

- **The power numbers are on BATTERY and every other number in this file
  is on AC.** Do not mix them. Sampler overhead is real too:
  `powermetrics` wakes 5x/s and its own CPU time lands in the counters it
  reads, equally across arms, so paired ratios are clean and absolute
  watts carry a small inflation.
- **The wall-power column is directional only.** The battery gauge
  (`ioreg` `InstantAmperage` x `Voltage`, no root) read a standard
  deviation of 20-40% of its own mean across a run, and reported 48.8 W
  against 64.0 W for two arms doing identical work. It is good enough to
  say the machine draws roughly 50-70 W under load against ~20 W idle,
  and not good enough to publish a wall joules-per-token. That gap
  between ~17 W of CPU+GPU and ~60 W at the battery is mostly display and
  rest-of-SoC, and remains unattributed.
- Two measured runs per arm. Enough to show the 1.5x gap that used to be
  here, and enough to show it is gone; not enough to claim a 2 percent
  difference in either direction.
- One machine, one chip, one session, on AC. Absolute numbers here have
  repeatedly failed to transfer across sessions in this repo; the ratio is
  what to carry forward.
- Both engines ran alone (`scripts/parity.sh` refuses to start if another
  model process is up), with no profiler or trace mode active.
