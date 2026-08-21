# Benchmarks: this port against the Swift original

One machine, one model install, one session, both engines. Every other
figure in this repo and in `docs/BENCHMARKING.md` is this port measured
against its own past self; this is the parity number.

Reproduce with `scripts/parity.sh`. Read `docs/BENCHMARKING.md` for the
harness itself (the three `turbospark-bench` modes, the memory oracle, and
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
measured directly (`cargo test -p turbospark-selection --release --test
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
anything understated: one `turbospark-bench --case` launch runs the
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

### Every install, side by side

The table the README's summary is drawn from. All M4 Max, AC, release, 16
expert-cache slots; each row is the session peak its memory oracle asserts,
and the ceiling beside it is that oracle's bound (deliberately ~8-13% above
the reading, so allocator jitter cannot flake it).

| Install | On disk | Context | Measured peak | Oracle ceiling | Decode tok/s | Streams? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gemma 4 26B-A4B, MLX INT4 | 13 GB | 4,096 | 2,108 - 2,182 MiB | 2,250 | 34.6 - 40.7 | yes, 128 experts |
| Gemma 4 26B-A4B, IQ3 GGUF | 12 GB | 4,096 | ~1,850 MiB | -- | 22.9 - 25.4 | yes |
| Qwen 3.6 35B-A3B, MLX INT4 | 18 GB | 4,096 | 1,587 - 1,610 MiB | 1,700 | 32.6 - 38.0 | yes, 256 experts |
| Qwen3-30B-A3B, Q4_K_M | 17 GB | 4,096 | 2,741 - 2,753 MiB | 2,900 | 16.0 - 28.1 | yes, 128 experts |
| gpt-oss-20b, MXFP4 | 11 GB | 8,192 | 5,417 - 5,421 MiB | 5,700 | 22.9 - 30.4 | yes, 32 experts |
| **Qwen3.8-27B, MLX INT4** | 14 GB | 4,096 | **660.0 - 660.3 MiB** | 750 | 18.6 - 21.1 | **no, dense** |
| Mistral 7B, Q4_K_M | 4.1 GB | 8,192 | 1,201 - 1,203 MiB | 1,300 | 16.3 - 30.4 | no, dense |
| **Ternary-Bonsai-27B, MLX 2-bit** | 7.6 GB | 4,096 | **657.8 - 661.6 MiB** | 750 | 12.5 - 13.9 | **no, dense** |
| Bonsai-27B, MLX 1-bit | 3.9 GB | -- | not measured | -- | ~18.3 | no, dense |

**Read the `Streams?` column before comparing any two rows**, because the
two groups are measuring different things and only one of them is a result
about this engine.

- **Streaming rows**: the peak is `resident core + KV + slot cache`, and the
  slot term (`slots x layers x expert_stride`) dominates. That is the
  engineering claim -- a 13 GB model held in 2.1 GB. It also explains the
  spread: gpt-oss is not a regression at 5.4 GiB, it is 24 layers of a much
  larger expert, and Qwen3-30B-A3B sits at 2.7 GiB on depth alone (48
  layers) despite having the smallest expert here (Gotcha 36).
- **Dense rows**: nothing streams, and the peak is small for an entirely
  different reason -- `phys_footprint` does not count the memory-mapped
  weights at all (Gotcha 40). Qwen3.8-27B's 660 MiB sits beside 15.1 GB of
  weights that are mapped and pinned and simply not counted. **The two
  `qwen3_5` rows are the cleanest demonstration of that in this table**: the
  same architecture at INT4 and at 2 bits, the same 4,096 window, 15.1 GB of
  weights against 7.6, and their peaks differ by 1.3 MiB. A dense row is
  therefore NOT a claim that the model runs in that much RAM; the practical
  requirement is closer to its size on disk. The counted figure is a leak
  sentinel, not a capacity number.

And the CONTEXT column is load-bearing on the dense rows in particular: KV
is most of what they measure, so Mistral reads 1,201 MiB at 8,192 and 684
MiB at 4,096 for the same install and the same model.

## Expert-cache slots: the one runtime control that moves this

Both engines default to 16 slots and the table above is measured there.
`turbospark-bench --model` can now vary it (`--expert-cache-slots`, allowed
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
(`protocol::PROTOCOL_EXPERT_CACHE_SLOTS`). Output IS identical across slot
counts, which it was not when this page was first written: the routed
dispatch order used to follow cache state, so the phase-2 reduce order (and
therefore the generated bytes) moved with it. Fixed by dispatching in the
router's own ranking; `quality_common` now asserts that identity across
8/16/32 rather than freezing a digest per slot count. See AGENTS.md
Gotcha 27.

This also reconciles a discrepancy that stood open in `DEVIATIONS.md`:
42.6 tok/s recorded on this checkpoint against the 25.6 the first parity
run measured. The two are separated by both axes above -- the sampler
(worth ~15 tok/s at this vocabulary) and the slot count (worth ~6) -- and
the 42.6 sits inside the range they span. The settings behind the 42.6
were not recorded, so it is retired rather than re-explained.

### What the slot count cannot fix: expert granularity

The slot cache is `slots x layers x expert_stride`, so the memory result
above belongs to FINE-GRAINED MoE rather than to MoE as such. Measured by
arithmetic off the repack walk's own reported stride (ROADMAP Phase M2,
AGENTS.md Gotcha 36):

| | Gemma 4 26B-A4B | Mixtral 8x7B | Qwen3-30B-A3B |
| --- | ---: | ---: | ---: |
| layers | 30 | 32 | 48 |
| experts per layer | 128 (top-8) | 8 (top-2) | 128 (top-8) |
| one expert blob | ~3.2 MiB | 108.9 MiB | 2.53-2.92 MiB |
| whole expert table | 12 GiB | 27.2 GiB | 16.36 GiB |
| slot cache at 16 slots | 1.5 GiB | 54.5 GiB | 2.04 GiB |
| slot cache at `slots == experts` | n/a | 27.2 GiB (the whole table) | n/a |

Mixtral is the SMALLER model by parameter count and cannot stream here at any
useful slot count. The real published Q4_K_M install runs and is correct --
both smokes are coherent -- at 0.15 to 0.17 tok/s on this machine at 8 slots,
because each token re-reads most of the expert table. No memory-oracle or
quality-gate row is published for it: its footprint is this arithmetic rather
than a regression signal, and both gates belong to a fine-grained checkpoint.

Both inputs to that multiplication (`expert_count`, `feed_forward_length`) sit
in the GGUF header, so the answer is available before any download.

**Qwen3-30B-A3B is the third column and it is what picking on granularity
first buys.** Measured 2026-08-10 on the real streamed `qwen3moe` install,
frozen protocol, AC, release, 16 slots, all three cases stopping endOfTurn:

| | value |
| --- | ---: |
| decode, short / medium / long | 27.3 / 24.0 / 16.0 tok/s |
| peak `phys_footprint` | 2,748 MiB (2,751 on the oracle's longer session) |
| reference-answer perplexity | 14.7576 |
| replay growth | +0.23 MiB |

**Read the footprint as arithmetic, not as a regression, and note that it
leaves the published band.** At 2.75 GiB this checkpoint sits above the
~1.6-2.2 GiB the other two families hold, because the slot term is
`slots x layers x expert_stride` and depth counts as much as expert size: 48
layers at ~2.9 MiB is 2,094 MiB of slot capacity at 16 slots, against Gemma's
30 layers at ~3.2 MiB. Add a 916 MiB resident core, which is mapped AND
pinned. It is still a streaming install -- 2.04 GiB of a 16.36 GiB expert
table is ever resident -- which is precisely what Mixtral could not do. The
lesson for the next family is that "fine-grained" is necessary and still not
the whole product; compute `slots x layers x stride` rather than reading the
expert size alone.

The perplexity is not comparable across families (each chat template puts the
reference answer in a different position; see the Quality section), so 14.76
against Gemma's 37.42 ranks nothing. It is a frozen row to compare against its
own future.

## Quality

The one axis with no Swift column. The Swift original publishes no
perplexity, no KL divergence, and no golden output, so there is nothing to
compare against there; ROADMAP Phase Q exists to build the axis anyway,
before Phase S touches quantization.

Read the sections below in two groups. The perplexity, the digests, the
constrained-cache arm, and the sensitivity curve are all this port measured
against its own past, regression sentinels, and no row in them is or can
be a parity claim. The cross-engine section at the end is the exception and
the only external reference in this document's Quality half: it compares
this port against mlx-lm on the same quantized bytes.

Reproduce with the two gates (about a minute each), which assert these
values on this chip and print them on any other:

```sh
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_gate --release -- --ignored --nocapture
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
- **Digests no longer depend on expert-cache state.** They did until
  2026-08-08, when a layer's routed slots were dispatched cache misses
  first then hits: that permuted the phase-2 reduce order, FP addition is
  not associative, and so a cold-cache generation did not match a warm one
  (and, worse, two warm runs need not match each other -- AGENTS.md Gotcha
  27). Slots are now dispatched in the router's own ranking. The gate keeps
  its warmup-then-measure run order anyway, because throughput still wants
  a warm cache. Everything is pinned to 16 slots for the same reason the
  tok/s rows are.

### Constrained working set

The last column halves the routed-expert cache to 8 slots, which is this
port's memory knob (16 -> 32 slots cost 1.5 GiB; see the sweep above), and
repeats the greedy generation. Upstream's acceptance proof for that case is
"byte-identical output at unchanged throughput under a constrained working
set". Measured here, on the same day and machine as the rows above:

| Install | tok/s at 16 slots | tok/s at 8 slots | Ratio | Bytes identical |
| --- | ---: | ---: | ---: | --- |
| Gemma 4 26B-A4B | 44.602 | 39.985 | 0.90x | yes |
| Qwen 3.6 35B-A3B | 40.163 | 29.696 | 0.74x | yes |

Throughput degrades rather than collapsing, and byte identity now holds on
BOTH families, so the gate asserts the 8-slot digest EQUALS the 16-slot one
rather than freezing a second golden.

It did not hold on Gemma until 2026-08-08. `real_forward_gemma4.rs` used to
order a layer's routed slots misses first then hits (so the resident hits'
phase-1 GEMV could ride its own command buffer); that order fed phase 2's
reduce, and FP addition is not associative, so Gemma's bytes moved with the
hit/miss split. The deeper problem was that the hit/miss split is a
function of cache state rather than of the prompt, which made repeated warm
runs of one prompt non-reproducible (AGENTS.md Gotcha 27). Slots are now
dispatched in the router's own ranking. The rows above are re-measured
after that change and are not comparable to the pre-2026-08-08 ones.

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
independently useful: cache state alone moved the distribution by 0.0019
mean nats, under the misses-first slot order that AGENTS.md Gotcha 27
records and that has since been removed. The KL is reported both ways; the forward and reverse
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
`TURBOSPARK_LOGIT_DUMP_COLD=1` reproduces that condition. This port's own two
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
`TURBOSPARK_QWEN36_INSTALL_DIR` and would produce one, but `kld.py`'s
reference is pinned to the Gemma repo.

### Cross-engine: llama.cpp on the same GGUF

The section above audits the INT4 path, against the checkpoint
`~/models/gemma4.gturbo` was repacked from. Nothing audited the GGUF path,
and it had one number that read as a defect: the Q8_0 install scores
39.8808 perplexity against the INT4 install's 37.4176. The
higher-precision side scoring 6.6% WORSE is backwards, and 6.6% sits inside
the band the sensitivity table above proves the metric can see. Two
readings fit: Q8_0 genuinely loses on this corpus, or the GGUF repack
carries residual error. A second engine on the same GGUF separates them.

`scripts/kld_llamacpp.py` replays the same ids through llama.cpp b10310 on
`ggml-org/gemma-4-26B-A4B-it-GGUF`'s `Q8_0` file, the exact bytes
`~/models/gemma4-gguf.gturbo` was streamed from, via a small harness
(`scripts/llamacpp_logits.c`) built against llama.cpp's own header. Ids in,
full-vocabulary logits out; llama.cpp's tokenizer is never asked to encode
anything. Both heads softcap at 30 and neither normalizes, verified rather
than assumed: llama.cpp's max |logit| reads 29.9993. The vocabularies are
checked to line up two ways, since a mismatch there would read as a
numerics gap: `llama_vocab_n_tokens` must equal the dump's 262,144, and the
harness detokenizes the first eight ids, which come back as
`<bos>|<|turn>|user|\n|Explain| how| coastal| wetlands|` -- the frozen
protocol's own `short-explanation` prompt under Gemma's turn markup.

Gemma 4, 550 positions, 16 expert-cache slots, 2026-08-08, on AC:

| Comparison | Mean KL | Median | p99 | Max | Top-1 agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp batched vs cached, both Metal (shape floor) | 0.00134 | 0.00001 | 0.012 | 0.330 | 99.5% |
| **this port vs llama.cpp, same bytes, both Metal, both cached** | **0.00845** | **0.00034** | **0.153** | **1.839** | **98.2%** |
| llama.cpp Metal vs llama.cpp CPU, both cached (backend floor) | 0.05510 | 0.00180 | 0.636 | 8.042 | 95.1% |
| this port on INT4 vs llama.cpp on Q8_0 (different weights) | 0.57748 | 0.09218 | 7.874 | 24.174 | 78.0% |

| Reading | Perplexity |
| --- | ---: |
| this port, GGUF Q8_0 install | 39.8808 |
| **llama.cpp, same GGUF, Metal, cached** | **39.8541** |
| llama.cpp, same GGUF, Metal, batched | 40.0673 |
| llama.cpp, same GGUF, CPU, cached | 39.1419 |
| this port, MLX INT4 install | 37.4176 |

**Q8_0 really is worse than the MLX INT4 checkpoint on this corpus, and
this port's GGUF path is not the reason.** An independent engine on the
same bytes reads 39.8541 against this port's 39.8808, 0.067% apart, while
the INT4 install sits 6.2% away from both. The distribution says the same
thing an order of magnitude more sharply: 0.00845 nats between the two
engines against 0.57748 for a genuine weight difference, 68x.

**Match the backend, not just the bytes.** This was first measured against
llama.cpp on CPU, because a 26.9 GB model looks like it will not fit under
a 36 GB machine's Metal wired limit (it does, and runs 5x faster there).
That reading was **0.05838** nats at 95.1% top-1, which is 40x the shape
floor and reads as a real defect in this port. It is not: ggml's own CPU
and Metal paths disagree with each other by 0.05510 nats and 4.9% of the
argmaxes on this model, and matching the backend collapses the headline to
0.00845 at 98.2%. The whole apparent gap was in the reference. Feeding two
engines the same bytes is not enough when one is running different
arithmetic; a cross-engine comparison needs a backend floor beside its
shape floor, and this one is 41x the larger of the two.

The KL is reported both ways, and unlike the mlx-lm section above the two
directions do NOT agree to within 5% here: reverse means are 0.00144 /
0.00706 / 0.04485 / 0.61091 against the forward 0.00134 / 0.00845 / 0.05510
/ 0.57748, i.e. 16 to 19% apart on the two middle rows. The ordering of the
four rows is identical either way and no conclusion above turns on the
direction, but quote the direction with the number.

Cost, for planning: the 550-position cached walk is **26 s on Metal**
against **2m21 on CPU**, and the model loads in both cases. `-ngl 99` is
the default in the driver for that reason as much as for the accuracy one.
A batched pass is 26 s and 2m09 respectively. Add ~8 min for the one-time
26.9 GB download.

Everything here is deterministic: llama.cpp's Metal cached pass is
byte-identical across processes (SHA-256 `2e5b3f47...`), as is this port's
dump. Treat any movement as a real change, not noise.

Two sibling facts fall out. This port's cold and warm dumps of the GGUF
install are now byte-identical, where the mlx-lm section above measured
0.0019 nats between them; that is AGENTS.md Gotcha 27's fix, and it holds
on the MLX install too (warm now reads 37.4176, which used to be the cold
number). And llama.cpp's own CPU and Metal perplexities differ by 1.8% on
identical bytes, which is a useful calibration for `PERPLEXITY_REL_TOLERANCE`
at 2%.

Caveats. One corpus, one family, one machine, one llama.cpp build. This
says the two engines agree on these bytes; it does not say either is close
to the unquantized model, which would need a bf16 reference nobody has run
here. And nothing in the standing gate runs this: it is a script, and the
26.9 GB GGUF it needs is not kept on disk.

### Cross-engine: llama.cpp on the same GGUF, `qwen3moe`

The section above audits Gemma's GGUF path. This is the same question for
the third family (ROADMAP M3, Qwen3-30B-A3B), and it is the ONLY instrument
that can answer the two things `crates/runtime/tests/real_forward_qwen3moe.rs`
records itself as unable to see: **the order of the per-head q/k norms
relative to RoPE**, and **the RMS epsilon's own value**. RoPE is a rotation
and preserves per-head RMS, so both norm orders differ from "no norm" by
about as much, and a fixture cannot rank them; the epsilon is invisible to
any test whose weights are untrained. Both belong here or nowhere.

Same method as Gemma's: `scripts/kld_llamacpp.py` replays this port's own id
sequence through llama.cpp b10330 on `Qwen/Qwen3-30B-A3B-GGUF`'s `Q4_K_M`
file, the exact 18,556,685,824 bytes `~/models/qwen3moe-gguf.gturbo` was
streamed from. 562 positions, 16 expert-cache slots, 2026-08-10, on AC.

| Comparison | Mean KL | Median | p99 | Max | Top-1 agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp batched vs cached, both Metal (shape floor) | 0.00135 | 0.00039 | 0.018 | 0.033 | 99.1% |
| **this port vs llama.cpp, same bytes, both Metal, both cached** | **0.00320** | **0.00078** | **0.040** | **0.208** | **97.9%** |
| llama.cpp Metal vs llama.cpp CPU, both cached (backend floor) | 0.00988 | 0.00461 | 0.071 | 0.293 | 96.4% |

| Reading | Perplexity |
| --- | ---: |
| **this port, Q4_K_M GGUF install** | **14.7576** |
| llama.cpp, same GGUF, Metal, cached | 14.6167 |
| llama.cpp, same GGUF, Metal, batched | 14.9835 |
| llama.cpp, same GGUF, CPU, cached | 14.2399 |

**The headline sits BETWEEN the two floors and nearer the lower one.** At
0.00320 nats it is 2.4x the shape floor and less than a third of the backend
floor: this port and llama.cpp agree on identical bytes more closely than
ggml's own two backends agree with each other. Gemma's headline is 6.3x its
shape floor by the same arithmetic, so `qwen3moe` is the tighter of the two
families this has been run on.

**That settles the norm order and the epsilon.** Neither is a rounding-scale
effect: a q/k norm applied on the wrong side of RoPE, or an epsilon off by a
factor of ten, perturbs every one of 48 layers systematically and compounds
into the head. Nothing of that shape fits 3x under a backend floor. The
perplexity says it again on a different axis -- 14.7576 against the
reference's 14.6167 is 0.96% apart, comfortably inside the 2.65% by which
llama.cpp's own Metal and CPU paths disagree on these same bytes.

Unlike Gemma's section the two KL directions DO agree here, within 3.3% on
every row (reverse means 0.00137 / 0.00330 / 0.00978), so no conclusion
turns on the choice. Quote the direction anyway.

**A guard in the driver had to be fixed first, and the failure was the
interesting part.** `kld_llamacpp.py` refused any run whose max |logit|
exceeded a literal `30.0` -- Gemma's `final_logit_softcapping`, which every
previous caller happened to share. This family declares
`finalLogitSoftcap: 0.0` and both engines read max |logit| ~51.9, so the
unfixed script aborts a correct run and blames the heads. The bound now
comes from the install's own manifest, and both engines' maxima are reported
side by side (51.93 llama.cpp against 51.97 this port) so the no-softcap
case still has an observable rather than an invented tolerance. See AGENTS.md
Gotcha 38.

Everything here is deterministic, like the other quality numbers and unlike
any tok/s figure: this port's dump is SHA-256 `16bfad2b...` and llama.cpp's
Metal cached arm `7a6a1fbf...`. Treat movement as a real change.

One sibling fact worth keeping. This port's WARM dump reads perplexity
14.757589 against the frozen COLD row of 14.7576 in
`qwen3moe_quality_gate.rs`. Those are the same number, which is Gotcha 27's
fix holding on a third family, and it is what licenses comparing a warm
dump against a cold frozen row at all.

Cost, for planning: the 562-position cached walk is **~30 s on Metal** and
**~40 s on CPU** -- far closer than Gemma's 26 s against 2m21, because only
3B of 30B parameters are active per token. Add ~12 min for the one-time
17.3 GB download, which is the whole expense: llama.cpp cannot stream the
file the way the repack walk did (Gotcha 34).

Caveats, the same ones as Gemma's. One corpus, one family, one machine, one
llama.cpp build. This says the two engines agree on these bytes; it does not
say either is close to the unquantized model, which still needs a bf16
reference nobody has run here. Nothing in the standing gate runs it, and the
GGUF it needs is not kept on disk.

### Cross-engine: llama.cpp on the same GGUF, `gpt-oss`

The same question for the sixth family (ROADMAP M5, gpt-oss-20b), and the
one instrument that can answer the two things
`crates/runtime/tests/real_forward_gptoss.rs` records itself as unable to
see: **whether YaRN's magnitude scale is the right VALUE** (`mscale =
1.3465736`, which scales `q.k` by 1.8133 -- the flow tests can catch it
CHANGING but nothing in this repo can say the number is right), and **the
attention sink's exact placement** (one learned logit per q head, added to
the softmax denominator and to nothing else). Both perturb every one of 24
layers systematically; both belong here or nowhere.

Same method as the two sections above: `scripts/kld_llamacpp.py` replays
this port's own id sequence through llama.cpp b10360 on
`ggml-org/gpt-oss-20b-GGUF`'s `gpt-oss-20b-MXFP4.gguf`, the exact
12,109,566,624 bytes `~/models/gptoss-20b.gturbo` was streamed from. 612
positions, 16 expert-cache slots, chat date pinned to the quality gate's
own `2026-01-01` (the Harmony template reads a clock; bench crate
Gotcha 14), the reference answer behind `<|channel|>final<|message|>`
(Gotcha 13, the same framing the gate scores). 2026-08-12, on AC.

| Comparison | Mean KL | Median | p99 | Max | Top-1 agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp batched vs cached, both Metal (shape floor) | 0.00258 | 0.0000055 | 0.072 | 0.393 | 99.2% |
| **this port vs llama.cpp, same bytes, both Metal, both cached** | **0.00978** | **0.00117** | **0.215** | **0.458** | **97.5%** |
| llama.cpp Metal vs llama.cpp CPU, both cached (backend floor) | 0.01181 | 0.00212 | 0.259 | 0.434 | 96.9% |

| Reading | Perplexity |
| --- | ---: |
| **this port, MXFP4 GGUF install** | **12.0801** |
| llama.cpp, same GGUF, Metal, cached | 12.1526 |
| llama.cpp, same GGUF, Metal, batched | 12.1676 |
| llama.cpp, same GGUF, CPU, cached | 12.1456 |

**The headline sits BETWEEN the two floors, under the backend floor.** At
0.00978 nats it is 3.8x the shape floor and 0.83x the backend floor: this
port agrees with llama.cpp-on-Metal more closely than ggml's own two
backends agree with each other on these same bytes. For scale, Gemma reads
6.3x its shape floor and `qwen3moe` 2.4x.

**That settles the mscale value and the sink placement.** A YaRN factor
applied twice (what the `mscale` division in llama.cpp exists to prevent),
not at all, or at the wrong magnitude scales every attention in the model;
a sink added to the numerator instead of the denominator reweights every
softmax. Nothing of either shape fits under a backend floor. Perplexity
says it again on a separate axis: 12.0801 against the reference's 12.1526
is 0.60% apart -- and this family's llama.cpp CPU and Metal perplexities
agree to 0.06%, so the head really is the same function on both sides
(max |logit| 53.10 llama.cpp against 53.53 this port, no softcap declared,
reported rather than checked per AGENTS.md Gotcha 38).

The two KL directions agree within 10% on every row (reverse means
0.00230 / 0.00887 / 0.01290), so no conclusion turns on the choice. One
reported-not-chased observation: the shape floor (0.00258) is ~2x the
other two families' (~0.00135), i.e. ggml's batched and cached paths
disagree more on this model; it does not change any reading here.

Everything is deterministic, like the other quality numbers and unlike any
tok/s figure: this port's dump is SHA-256 `1761d134...` (byte-identical
across two processes) and llama.cpp's Metal cached arm `65a6200d...`. Treat
movement as a real change.

One sibling fact worth keeping. This port's WARM dump reads perplexity
12.080073 against the frozen COLD row of 12.0801 in
`gptoss_quality_gate.rs`. Those are the same number: Gotcha 27's fix
holding on a fourth family, and only because the dump pins the SAME chat
date the gate pins -- an unpinned dump encodes a different prompt every day
and reproduces nothing.

Cost, for planning: the 612-position cached walk is well under a minute on
either backend (3.6B of 21B parameters active per token, like `qwen3moe`
and unlike Gemma). Add ~7 min for the one-time 12.1 GB download, which is
the whole expense: llama.cpp cannot stream the file the way the repack walk
did (Gotcha 34).

Caveats, the same ones as the two sections above. One corpus, one family,
one machine, one llama.cpp build; nothing here says how far MXFP4 sits from
the unquantized model.

### Cross-engine: MLX on the same bytes, the 1-BIT family

The seventh family (ROADMAP's 1-bit entry, `prism-ml/Bonsai-27B-mlx-1bit`)
and the first whose reference is MLX rather than llama.cpp, because the
checkpoint is MLX-native and there is no GGUF of it. **It is also the
tightest agreement measured in this repo, by two orders of magnitude.**

`scripts/kld_mlx_affine.py` replays this port's own id sequence through
`mlx-lm==0.31.2` on the exact 5,129,115,752 bytes `~/models/bonsai27b.gturbo`
was streamed from. 573 positions, warm cache, 2026-08-14 on AC. The
reference runs on `github.com/PrismML-Eng/mlx@prism` built from source: **the
comparison is impossible on upstream mlx**, which refuses `bits=1` at the
API level rather than merely lacking a Metal kernel ("The supported bits are
2, 3, 4, 5, 6 and 8"), on every device.

| Comparison | Mean KL | Median | p99 | Max | Top-1 agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX batched vs cached, both Metal (shape floor) | 0.0000074 | 0.0000034 | 0.000041 | 0.000086 | 100.0% |
| **this port vs MLX, same bytes, both Metal, both cached** | **0.0000157** | **0.0000104** | **0.000084** | **0.000119** | **100.0%** |
| MLX Metal vs MLX CPU (backend floor) | not measured | | | | |

| Reading | Perplexity |
| --- | ---: |
| **this port, 1-bit install** | **8.3554** |
| MLX, same bytes, Metal, cached | 8.3587 |
| MLX, same bytes, Metal, batched | 8.3576 |

**The headline is 2.1x the shape floor and ~200x SMALLER in absolute terms
than any other family's** (Gemma 0.00845, `qwen3moe` 0.00320, gpt-oss
0.00978, IQ3 0.00440). Top-1 agreement is 100.0% on all 573 positions,
where every other family reads 97.5-98.2%. Perplexity agrees to 0.04%, and
max |logit| is 28.171875 on BOTH sides -- the same value to the last bit.

**Both numbers being tiny is the reading, and it is consistent rather than
suspicious.** The other families' floors are dominated by MoE: batched and
cached passes route and reduce experts differently, and FP addition is not
associative. This model is DENSE with 75% linear attention, so its shape
floor has almost nothing to be made of -- which is why the floor is ~5,000x
smaller than Gemma's mlx-self floor (0.0352) and the headline shrinks with
it. The RATIO is what transfers between families; the absolutes do not.

**What it settles.** Three things this port assumed and could not otherwise
check. The **mrope reduction** -- though the reference SOURCE had already
settled that one for free (see ROADMAP step 5: the checkpoint declares
`rope_type: "default"`, so `mrope_section` is read by nothing on the text
path). The **RMS epsilon** at 1e-6 and the **q/k norm order** relative to
RoPE, both of which perturb all 64 layers systematically and neither of
which could hide under a floor this small. And the **F16-to-BF16 norm
narrowing** this port does and MLX does not (AGENTS.md Gotcha 45): the
step-4 judgement call was that losing 19.5% of the norm values at up to
2^-8 relative would not be the limiting error in a model whose weight
matrices are one bit. At 1.6e-5 nats and 100% top-1 that is now measured
rather than argued, and the FP16-weight kernel variant it was weighed
against stays unbuilt on evidence.

Everything is deterministic: the report reproduces to the last digit across
two processes, and this port's dump is SHA-256 `f2970b98...`. Treat movement
as a real change.

Two things NOT measured, named rather than skipped. There is **no backend
floor** -- the same engine's CPU against its Metal, which AGENTS.md Gotcha 34
records as the axis that made a correct Gemma run look broken. This model is
dense, so every one of its 24.8B backbone weights is read per token, where
the families with a cheap CPU arm activate 3B; the arm is unaffordable rather
than forgotten. A missing floor is a missing scale, and the headline here is
read against the shape floor alone. And there is **no bf16 reference**, so
nothing here says how far 1-bit sits from the unquantized model -- the
checkpoint's own README claims ~90% of FP16 on 15 benchmarks, which is a
different measurement by different people.

Cost, for planning: ~3 min to build the mlx fork from source, ~7 min for the
4.78 GB reference download, ~24 s for both arms plus the KL. The port's dump
is 72 s and 271 MiB.

### Sub-4-bit candidate survey (ROADMAP Phase S)

This is not a measurement of this port. This port cannot ingest IQ3_XXS, so there
is no arm for it here; llama.cpp runs the candidate and this port's frozen
dumps stand in for the two quantizations it does run. What the section
answers is whether a 3-bit checkpoint is worth building kernels FOR, which
is a question about the checkpoint, not about a kernel that does not exist.

The candidate is `unsloth/gemma-4-26B-A4B-it-GGUF`'s `UD-Q3_K_M`, whose
routed experts are IQ3_XXS (`ffn_gate_up_exps`) and IQ4_NL
(`ffn_down_exps`). Despite the name it contains no Q3_K; see ROADMAP Phase
S for the per-tensor table and for why the static build of the same recipe
is a different file entirely. Same harness, same 550 ids, same machine and
day as the section above.

| Comparison | Mean KL | Top-1 agree |
| --- | ---: | ---: |
| candidate batched vs cached, both Metal (shape floor) | 0.00051 | 100.0% |
| candidate Metal vs CPU, both cached (backend floor) | 0.03741 | 94.7% |
| candidate vs this port's Q8_0 install | 0.15275 | 90.5% |
| candidate vs this port's MLX INT4 install | 0.68365 | 77.1% |
| *[frozen above]* INT4 vs llama.cpp Q8_0 | 0.57748 | 78.0% |

| Reading | Perplexity | vs MLX INT4 |
| --- | ---: | ---: |
| this port, MLX INT4 install | 37.4176 | -- |
| **llama.cpp, candidate IQ3_XXS, Metal, cached** | **38.0997** | **+1.82%** |
| llama.cpp, candidate IQ3_XXS, Metal, batched | 37.9122 | +1.32% |
| llama.cpp, candidate IQ3_XXS, CPU, cached | 37.6485 | +0.62% |
| llama.cpp, Q8_0, Metal, cached | 39.8541 | +6.51% |
| this port, GGUF Q8_0 install | 39.8808 | +6.58% |

**The 3-bit checkpoint beats the 8-bit one and ties the 4-bit one.** Its
+1.82% against the incumbent INT4 install is inside
`PERPLEXITY_REL_TOLERANCE`, which is itself calibrated on llama.cpp's own
1.8% CPU/Metal spread, so on this corpus the two are not distinguishable.
Against Q8_0 it is 4.4% BETTER. The imatrix calibration is doing real work,
and the ordering is not an artifact of one arm: the candidate wins on all
three of its own arms.

The distribution says the same thing and sharpens it. Among the three
quantizations the INT4 install is the OUTLIER, not the 3-bit one: the
candidate sits 0.15275 nats from Q8_0 while INT4 sits 0.57748 from the same
reference, 3.8x further. A 3-bit imatrix build tracks the 8-bit reference
more closely than the 4-bit MLX one does.

Read with the same two floors as above, which were re-measured on this file
rather than carried over: 0.00051 shape and 0.03741 backend. The headline
0.15275 is 4x the larger of them, as a genuine weight difference should be.

Caveats, and they matter more here than above because this is a decision
input rather than a parity claim. One corpus of 550 positions, perplexity
and KL only, no generation judged. It measures llama.cpp's IQ3_XXS decode,
so it says the CHECKPOINT is sound and says nothing about a kernel this
port has yet to write. And the win is a mixture, not "3-bit experts":
`ffn_down_exps` stays at IQ4_NL and one layer of each expert tensor sits a
level higher, which is exactly why the file is 9.10 GiB of experts rather
than a true 3-bit 7 GiB.

### The 3-bit install, measured (ROADMAP Phase S)

The section above measured llama.cpp on the candidate, because this port
could not ingest it. It can now. Everything below is this port's own IQ3_XXS
and IQ4_NL kernels, on an install streamed from the identical file, measured
2026-08-08 on AC, same machine, same 550 ids, same corpus.

**Is the decode faithful?** Against llama.cpp on the same bytes, both Metal,
both cached:

| Comparison | Mean KL | Top-1 agree |
| --- | ---: | ---: |
| candidate batched vs cached, both Metal (shape floor) | 0.00051 | 100.0% |
| **this port's IQ install vs llama.cpp** | **0.00440** | **97.5%** |
| candidate Metal vs CPU, both cached (backend floor) | 0.03741 | 94.7% |

Read it against the floors and not on its own, which is AGENTS.md Gotcha 34's
whole point. The port's disagreement with llama.cpp is 8.5x SMALLER than
ggml's own disagreement with itself across backends, and 8.6x larger than the
shape floor: the same shape as the Q8_0 result above and tighter in absolute
terms (0.00440 against that one's 0.00845).

**What does it cost?**

| Reading | Perplexity | vs MLX INT4 |
| --- | ---: | ---: |
| this port, MLX INT4 install (incumbent) | 37.4176 | -- |
| llama.cpp, candidate, Metal, cached | 38.0997 | +1.82% |
| **this port, IQ3_XXS/IQ4_NL install** | **38.3753** | **+2.56%** |
| this port, GGUF Q8_0 install | 39.8808 | +6.58% |

The port reads +0.72% against llama.cpp on the same bytes, inside
`PERPLEXITY_REL_TOLERANCE` and inside llama.cpp's own 1.2% Metal/CPU spread
on this file, so the gap to the incumbent is the QUANTIZATION rather than the
kernels. The 3-bit install still beats the 8-bit one by 3.8%.

**What does it buy?** `packed_experts/` is 9.6 GiB against the MLX install's
12 GiB, **-20%**, and the whole install is 12 GB against 13 GB. Peak
`phys_footprint` over the frozen protocol is **1,850 MiB** against the MLX
install's 2,108-2,182, **-13 to -15%**, which makes this the leanest Gemma 4
configuration measured here. All three protocol cases stop at `endOfTurn`.

| install | peak `phys_footprint` | protocol decode |
| --- | ---: | ---: |
| MLX INT4 (incumbent) | 2,108 - 2,182 MiB | 34.674 - 40.687 tok/s |
| **IQ3_XXS / IQ4_NL** | **1,850 MiB** | **22.874 - 25.410 tok/s** |

The footprint drop is smaller than the 20% disk drop, and that is the
accounting working as documented: only the resident weights are mapped, and
`phys_footprint` also carries KV, the expert slot capacity and the process
baseline. The expert cache holds a fixed slot count, so slots shrink with the
blob but do not disappear.

That -20% is the phase's premise and it is not automatic. Layer 29's expert
blob is 4,212,736 bytes against the other twenty-nine's 2,632,960, so padding
every layer to the model-wide maximum -- which is what the writer did before
this phase -- would have written **16.23 GB of experts instead of 10.33**, a
35% REGRESSION against the install it exists to shrink. The per-layer stride
is a prerequisite, not a tuning step. Both numbers are printed by
`gguf_iq_install_network.rs` and asserted there.

Decode is 22.9-25.4 tok/s on the protocol against the MLX install's
34.7-40.7, about -35%: the codebook kernels cost more throughput than the
smaller reads save. Measured, untuned, and no attempt has been made to close
it. Prefill is also slower (108 s for the 3,015-token case against ~64 s),
which is the same per-token kernel cost paid 3,015 times.

Frozen and reproducible: `iq3_quality_gate.rs` carries the row, and its
perplexity and both digests reproduced to the last digit and last hex
character in a second process. Output is byte-identical across 8, 16 and 32
expert-cache slots (`gguf_nondeterminism_probe`), so nothing here depends on
cache state.

MEASURED 2026-08-09, and the phase's motivating axis is a LOSS: decode
joules-per-token roughly doubles against the INT4 install (0.925-0.996
against 0.384-0.498 on the AC protocol, 2.0-2.4x), because GPU watts
nearly double (22.8-25.0 against 12.3-13.6) while decode runs 35% slower.
The cost is entirely GPU-side codebook dequant: host cpu W falls. Fewer
expert bytes per miss was an energy claim and the measurement refutes it
at this size. Full rows and the cross-session caveat:
`docs/POWER_BASELINE.md`, "The 3-bit install". ROADMAP dead end 12.

### The sixth family: `gpt-oss-20b` MXFP4 (ROADMAP M5)

Not a parity claim. Swift has no GGUF intake, so every number here is this
port measuring itself. Measured 2026-08-12 on AC, release, 16 expert-cache
slots, against `~/models/gptoss-20b.gturbo` streamed from
`ggml-org/gpt-oss-20b-GGUF`.

| | value |
| --- | --- |
| reference-answer perplexity | 12.0801 |
| one expert blob | 12.64 MiB |
| slot cache at 16 slots | 4.74 GiB (24 layers) |
| peak `phys_footprint` | 5,421 MiB at 8,192 context |
| decode, protocol cases | 31.343 / 26.710 / 23.472 tok/s |
| constrained (8 slots) | 0.83x, digest byte-identical |

**The footprint is the highest of any family here and it is arithmetic, not
a regression.** `slots x layers x expert_stride` is 4,854 MiB of slot
capacity on its own, against Qwen3-30B-A3B's 2,094 and Gemma 4's ~1,500, so
this install sits far above the 1.6-2.2 GiB band the README quotes. It is
still STREAMING, which is the whole reason the family was chosen: the expert
table is 9.5 GiB and only 4.7 of it is ever resident, where Mixtral's
108.9 MiB experts wanted 54.5 GiB and could not run here at all (AGENTS.md
Gotcha 36).

**Its row moves two protocol parameters and both are forced**, so it is
comparable to no other row without reading them. The budget is 3,072 new
tokens rather than 1,024 because Harmony puts the model's reasoning in an
`analysis` channel BEFORE its answer: the three cases need 818 / 2,153 /
1,108 tokens to reach `<|return|>`, and at the shared budget two of three
stop on `maxTokens`, which the validity gate refuses. The window is 8,192
rather than 4,096 because `2839 + 3072` does not fit. Worth carrying: a
budget derived from a greedy probe understated it (that read 1,780 for
`medium-review` against the protocol's sampled 2,153) -- a reasoning model's
answer length is a distribution, and it lengthens under sampling.

**The perplexity needed a family-specific assistant prefix, and the raw
number looked like a broken model.** Harmony's generation prompt ends at
`<|start|>assistant`, where the next token must be `<|channel|>`; splicing
the reference answer's prose straight in scores the model's surprise at
prose-instead-of-marker and reads **148,421.76**, against 6-38 for every
other family and 255,409 for the genuinely broken Qwen of `5279c88`. With
`<|channel|>final<|message|>` in front it reads 12.0801. The contradiction
that gives it away is that the generations were coherent throughout -- a
model that cannot predict its own output does not write fluent prose. This
is crate Gotcha 7's rule (score only positions the model was trained to
predict) arriving from a direction that gotcha did not anticipate: not the
wrong tokens, but the right tokens in a position the family's framing does
not put them in.

**The gate pins a date.** Harmony's template writes `Current date: ` into
its system preamble via transformers' `strftime_now`, so this is the first
gate here whose prompt reads a clock; without the pin both digests would
expire at midnight and read as a numerics regression the next morning. The
renderer itself uses the real clock, matching transformers, vLLM and
llama.cpp -- only the measurement asks for determinism.

The cross-engine KL HAS since been run and passed (2026-08-12, the
"Cross-engine: llama.cpp on the same GGUF, `gpt-oss`" section below):
0.00978 mean nats at 97.5% top-1, under the backend floor, which settles
the YaRN `mscale` and the sink placement. The `scripts/power.sh` capture
landed 2026-08-12 on one case and was RE-TAKEN across all three on
2026-08-13, once `turbospark-bench --model` learned to resolve this
family's 8,192/3,072 protocol parameters: decode 32.92 / 30.09 / 29.44 W
at 1.0569 / 1.1109 / 1.2813 J/token and 30.4 / 27.0 / 22.9 tok/s. Still
the highest-wattage install measured here. The superseded one-case row
read 36.67 W and the whole 3.75 W difference is `cpu W` (4.32 against
1.48): system-wide counters had attributed a busy desktop UI to the
decode loop. Rows, the diagnosis and what it does to the AC-throttle
reading: `docs/POWER_BASELINE.md`, "gpt-oss-20b".

### The seventh checkpoint, and the first controlled quantization pair: `Qwen/Qwen3.8-27B`

Not a parity claim. Swift has no `qwen3_5` support at all, so every number
here is this port measuring itself.

Measured 2026-08-14 on the machine in the provenance table, on AC, release,
16 expert-cache slots (inert on a dense install), 4,096 context. Install
streamed from `mlx-community/Qwen3.8-27B-4bit` into ~15.1 GB.

| | value |
| --- | ---: |
| peak `phys_footprint` | 660.3 / 660.0 / 660.2 MiB (three readings) |
| resident weights | 15,132,916,736 bytes, and NONE of it counted |
| decode, short / medium / long | 19.0 / 18.7 / 16.8 tok/s |
| reference perplexity | 4.9432 |
| greedy digest | `c3df0095` |
| sampled digest | `f272437c` |
| greedy at 8 slots | `c3df0095` (equal, as it must be) |
| replay growth | +0.02 MiB |

**Throughput re-measured 2026-08-17** after the INT4 function-constant
specialization (`46617c6`; see `docs/DECODE_BUDGET.md`, "The dense 27B").
Quiet unattended capture, AC, two full protocol runs: 20.3 / 19.8 / 18.6
and 21.1 / 20.7 / 18.7 tok/s (long-synthesis agreeing to 0.7% across
runs), peak 660.0 MiB both. The change itself was isolated with an
interleaved paired A/B in the same capture: +8.0 / +8.1 / +8.8% over the
pre-change binary with all six outputs one md5, so every digest and the
perplexity above are UNTOUCHED -- this is a throughput-only re-freeze,
and the summary table's decode range is updated to 18.6 - 21.1. The
1/2/4-bit triple's INT4 point moves to ~20.7 accordingly; its
compute-bound reading (decode does not track weight bytes, even
monotonically) is unchanged, since the 1-bit and 2-bit GEMVs were not
specialized.

**This is not a new family, and that is the interesting part.** Qwen3.8-27B
and `prism-ml/Bonsai-27B-mlx-1bit` are ONE architecture: their
`text_config`s agree on 33 of 35 keys, both have 2,180 tensors and 333
`vision_tower.` tensors, and both parse to the same `ArchConfig`
(`qwen_gdn_dense_27b()`) -- asserted offline, without the network, by
`every_published_checkpoint_parses_to_one_baseline`. The two that differ,
`eos_token_id` and the quantization block, reach no field of it. So no
`ArchConfig` field, no kernel and no decode flow changed to support this
checkpoint; the repack walk needed nothing either.

What that buys is the comparison this document has been unable to make
anywhere else. Every other quantization number here varies the checkpoint
and the quantization together (the Q8_0-against-INT4 Gemma pair is two
different producers; the Phase S IQ3 pair is a different file). **Bonsai at
1 bit and Qwen3.8 at INT4 hold the architecture, the layer graph, the
tokenizer and the decode flow fixed and vary the quantization alone** --
1-bit group 128 with FP16 companions against INT4 group 64 with BF16 ones.
Two readings from it, both to be taken with the caveat that these are
different TRAINED weights and not two quantizations of one training run
(Bonsai is a QAT checkpoint of its own):

- **Decode is 19.0 tok/s here against Bonsai's 18.3**, on 3.9x the weight
  bytes (15.1 GB against 3.9). If either were bandwidth-bound that could
  not happen, so this flow is compute-bound at both widths -- which is what
  the 1-bit entry suspected ("well below the MLX INT4 families' 35-44 ...
  nobody has profiled it") and had no second point to check against. The
  64-layer depth and the gated-DeltaNet recurrence, not the weight reads,
  are what set the rate.
- **660 MiB of counted footprint on a 27B model**, because AGENTS.md Gotcha
  40 holds here too. (The ternary section below turns this into the
  strongest form of that evidence: the same architecture at HALF the weight
  bytes reads 661.6 MiB.) That gotcha was measured on a dense GGUF install a
  quarter this size and explicitly said to re-derive it per install shape;
  re-derived on a dense SAFETENSORS install of 15.1 GB, the weights are
  still absent from the counter. The accounting that is left closes on KV
  (256.0 MiB at 4,096), the fixed delta-rule state (144.0 MiB, which does
  not grow with context) and the conv tail (7.5 MiB).

Two further notes on the perplexity, which at 4.9432 is the lowest in this
document. It is NOT a ranking against the other families: the corpus is the
frozen protocol's, chosen for Gemma, and each family's template puts the
reference answer in a different position. And it needed NO assistant prefix
even though this checkpoint's assistant slot is structured -- its template
opens a `<think>` block, which is exactly the shape that made gpt-oss read
148,421.76 without one. It needs none because `apply_chat_template` renders
with `enable_thinking: false` and this template's non-thinking branch emits
a closed, empty block (`<think>\n\n</think>\n\n`), so the reference answer
already lands in the answer position. Splicing a `</think>` in would have
written a second close and measured the model's surprise at that: the same
error as omitting one, from the other side.

The run also discharged a written-down unknown. `protocol_parameters` had
put this family in the shared 4,096/1,024 group on the TOKENIZER's evidence
and flagged it "UNVERIFIED until an install exists". Confirmed:
`long-synthesis` tokenizes to 2,940 and generates 637 more, so all three
cases stop `endOfTurn` with 3,577 of 4,096 used.

### The eighth checkpoint, and the ternary operating point: `prism-ml/Ternary-Bonsai-27B-mlx-2bit`

Not a parity claim. Swift has no `qwen3_5` support at all, so every number
here is this port measuring itself.

Measured 2026-08-15 on the machine in the provenance table, on AC, release,
16 expert-cache slots (inert on a dense install), 4,096 context. Install
streamed from the 8,490,785,104-byte published artifact into 7.57 GB of
resident weights, in 13.8 minutes.

| | value |
| --- | ---: |
| resident weights | 7,569,161,216 bytes |
| packed-expert files | 0 (dense) |
| unquantized tensors narrowed to BF16 | 161 tensors, 546,104 values |
| reference perplexity | 6.8350 |
| greedy digest | `6a99d870` |
| sampled digest | `7ea1f8d9` |
| greedy at 8 slots | `6a99d870` (equal, as it must be) |
| decode, greedy / sampled smoke | 14.2 / 13.8 tok/s |
| peak `phys_footprint` | 661.6 / 657.8 / 659.3 MiB (three readings) |
| decode, short / medium / long | 13.8 / 13.6 / 12.7 tok/s |
| replay growth | +0.00 / +0.02 MiB |

**The third checkpoint of one architecture, and it needed no `ArchConfig`
field, no baseline, no parser and no decode flow.** Its `text_config` is
Bonsai-27B's to the KEY -- the same `eos_token_id` 248046 and all -- so the
two files differ in their `quantization` object alone, which is a stronger
statement than the Qwen3.8 pair makes (that one differs in two keys).
`every_published_checkpoint_parses_to_one_baseline` asserts all three parse
to `qwen_gdn_dense_27b()` offline, without the network.

What it cost was a WIDTH, not a family: a CPU reference (`quant_2bit.rs`),
two Metal kernels (a GEMV and an embedding lookup), a `(2, fp16, 128)` arm
in three gates, a `DTYPE_INT2_AFFINE` tag, and two dispatch arms. The
symmetric-fast-path kernel the 1-bit entry built has no analogue here on
purpose: the checkpoint IS ternary, so one could exist, and it would
reassociate the sum exactly as the 1-bit one does -- which is why that one
is reachable from no decode flow, and why a second was not built.

Three things measured off the real bytes before any of it was written:

- **`bias == -scale` in every probed group** (1,920 of them), and the level
  histogram over 245,760 elements is `{0: 93895, 1: 57398, 2: 94467}` --
  **level 3 never occurs**. So the three levels in use are `-s`, `0`, `+s`:
  a ternary grid in a 2-bit affine word, which is what "1.58 bits" names.
  ROADMAP predicted that grid with a ZERO bias and that is wrong -- `q = 0`
  has to reach `-s`.
- **The level histogram proves nothing about field order**, and the module
  states it as a test. Permuting the four 2-bit fields inside a word
  permutes their multiset without changing it, so "no level 3" survives any
  wrong order untouched -- the 2-bit form of the popcount trap at one bit.
  Only `mx.dequantize` can see the order, and LSB-first reproduces it on all
  245,760 elements.
- **Neither property is baked in.** The container permits any
  `(scale, bias)` pair and all four levels, so `is_ternary_symmetric` and
  `uses_fourth_level` MEASURE them, and the GEMV decodes level 3 correctly
  (`the_fourth_level_is_decoded_not_clamped` is what stops a kernel written
  from the ternary description from masking the top bit).

#### Cross-engine: MLX on the same bytes, at TWO bits

Same driver as the 1-bit family's (`scripts/kld_mlx_affine.py`, one file
serving both widths), same 573 positions, warm cache, 2026-08-15 on AC,
replayed through `mlx-lm==0.31.2` on the exact 8,490,785,104 bytes the
install was streamed from. **Unlike the 1-bit arm this one runs on UPSTREAM
mlx** (0.32.0, out of a `uv run` ephemeral env): upstream refuses `bits=1`
at the API level and accepts `bits=2`, so no fork is needed and the whole
measurement is two commands.

| Comparison | Mean KL | Median | p99 | Max | Top-1 agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX batched vs cached, both Metal (shape floor) | 0.0000072 | 0.0000032 | 0.000038 | 0.000052 | 99.8% |
| **this port vs MLX, same bytes, both Metal, both cached** | **0.0000171** | **0.0000130** | **0.000086** | **0.000141** | **99.8%** |
| MLX Metal vs MLX CPU (backend floor) | not measured | | | | |

| Reading | Perplexity |
| --- | ---: |
| **this port, 2-bit install** | **6.8350** |
| MLX, same bytes, Metal, cached | 6.8355 |
| MLX, same bytes, Metal, batched | 6.8327 |

**2.4x the shape floor, against the 1-bit family's 2.1x**, and both are
~500x smaller in absolute terms than any MoE family's (Gemma 0.00845,
`qwen3moe` 0.00320, gpt-oss 0.00978, IQ3 0.00440). That is the MoE term's
absence rather than a better kernel, exactly as the 1-bit row records:
batched-vs-cached expert routing and reduce order is what makes the other
floors big, and this model is dense with 75% linear attention. **The RATIO
transfers between families and the absolutes do not**, which is why the two
sub-4-bit rows are worth reading against each other and not against the
rest of the table.

The reference is asserted to be running the PACKED weights, counted rather
than spot-checked: 497 `QuantizedLinear` plus 1 `QuantizedEmbedding`, all at
`bits=2, group_size=128`, equal to the checkpoint header's 498 `.scales`
tensors. Without that guard the comparison could quietly become "this port's
2-bit kernels against MLX's fp16 kernels on dequantized weights", whose tell
would be a suspiciously SMALL divergence -- the direction nobody
investigates. The driver additionally REQUIRES the checkpoint name on the
command line rather than defaulting: the 1-bit and 2-bit checkpoints have
the same module count and the same shapes, so pairing a dump with the wrong
reference passes every check inside the script and reads as a kernel bug.

One difference from the 1-bit row worth noting rather than explaining away:
max |logit| is 32.75 here against MLX's 32.78125, where the 1-bit pair
agreed to the last bit. The port's dump is float16 and 32.75 is the nearer
representable value below 32.78125, so this is the dump WIDTH and not a
head difference -- at one bit the maximum happened to land on a value f16
represents exactly.

**The peak is Qwen3.8-27B's number on half the weights, and that is the
point.** 661.6 MiB here on 7.57 GB of dense weights against 660.3 MiB there
on 15.1 GB -- same architecture, same 4,096 window, 1.3 MiB of difference.
AGENTS.md Gotcha 40 says a dense install's resident weights are absent from
`phys_footprint` and says to re-derive it per install shape; this is that
re-derivation on a pair that varies nothing but the quantization, which is
the cleanest form the evidence has taken. What is left closes on KV (256.0
MiB at 4,096), the fixed delta-rule state (144.0) and the conv tail (7.5) --
all functions of the architecture and the window, none of the width.

**The perplexity is 6.8350 against Qwen3.8-27B's 4.9432 on the same
architecture, the same corpus and the same framing, and that is NOT a clean
quantization ablation.** Ternary-Bonsai is prism-ml's own QAT checkpoint and
Qwen3.8-27B is Qwen's release quantized by mlx-community, so a TRAINING
separates the two numbers as well as a width. What the pair does say
cleanly is on the throughput axis, where the third point completes a
picture the 1-bit entry could only suspect: 14.2 tok/s here against
Bonsai's 18.3 at one bit and Qwen3.8's 19.0 at four, on 3.9 / 1.0 / 7.7 GB
of weights respectively. **Decode does not track the weight bytes at all**,
which is the compute-bound reading the Qwen3.8 pair first supported; the
2-bit GEMV is simply doing more per byte than either neighbour (four
elements a byte against eight, and no `+/-1` shortcut).

### Ornith-1.5: three installs of two checkpoints

Not a parity claim. Swift has no `qwen3_5` support at all, so every number
here is this port measuring itself.

Measured 2026-08-20 on the machine in the provenance table, on AC, release,
16 expert-cache slots, 4,096 context. **Neither checkpoint is a new family**:
`Ornith-1.5-35B-A3B` derives `qwen_gdn_moe_35b_a3b()` field for field from
its HF config AND from llama.cpp's GGUF metadata independently, and
`Ornith-1.5-9B` is the dense half at a new shape.

| | 9B (GGUF Q8_0, dense) | 35B-A3B (MLX INT4) | 35B-A3B (GGUF Q8_0) |
| --- | ---: | ---: | ---: |
| install on disk | 8.87 GiB | 18.21 GiB | 34.32 GiB |
| stream wall clock | 10.2 min | 22.4 min | 29.6 min |
| resident tensors | 427 | 613 | 613 |
| expert stride | 0 (dense) | 1,769,472 B | 3,342,336 B |
| slot cache at 16 | 0 | 1,080 MiB | 2,040 MiB |
| peak phys_footprint | 438 MiB | 1,578 MiB | not frozen |
| reference perplexity | 6.0503 | 6.2298 | not frozen |
| greedy digest | `37c9bbaf` | `8bab7013` | -- |
| sampled digest | `db6538f6` | `8ff05e36` | -- |
| 8-slot digest | equal, 1.00-1.02x | equal, 0.91-0.94x | -- |
| decode, three cases | 23.9-25.2 tok/s | 32.2-42.3 tok/s | see below |

The two 8-slot ratios are the pair worth reading together.
`--expert-cache-slots` sizes a ROUTED-expert cache, so the dense 9B reads
1.00x while its MoE sibling reads 0.91x on the same day and the same
prompt. That contrast is what makes either number a statement rather than
an absence. Both digests being slot-invariant is the reduce-order fix
(AGENTS.md Gotcha 27) holding on a fourth family.

**The 9B's recurrent state is larger than its KV**, which inverts the usual
reading of a footprint. Three quarters of its layers are gated-DeltaNet, and
a recurrent state is fixed in the window where a KV layer's is nothing but
the window: 24 linear layers give 144.0 MiB of delta-rule `S` against 8 full
layers' 128.0 MiB of KV at 4,096. Its 8.9 GB of resident weights are absent
from the 438 MiB entirely, which is AGENTS.md Gotcha 40 re-derived on a
fourth install shape.

#### INT4 against Q8_0, same model, same architecture

Three interleaved pairs at 16 slots, warmup discarded, `--max-new 300`,
greedy, both streams captured to files:

| pair | Q8_0 | INT4 |
| ---: | ---: | ---: |
| 1 | 22.650 | 36.896 |
| 2 | 21.947 | 36.241 |
| 3 | 22.089 | 36.805 |

**1.63-1.67x**, spreads 3.2% and 1.8%. Half the expert stride is the
honest half of the mechanism; the rest is that the INT4 routed pair is the
MLX-affine one rather than the GGUF one.

An earlier unpaired reading of this said 4.95x and was **cold against
warm**: a 34 GB install's first run reads 8.5 tok/s while it populates the
page cache, against 22 warm. Gotcha 20's discard-a-warmup rule is written
for a cold GPU after a build and applies at least as hard to a cold page
cache after an install, where the error is 2.6x rather than 1.5x.

#### What these installs cannot do

The 35B's published checkpoint carries a 785-tensor multi-token-prediction
head, and none of the three installs has it. The GGUF walk skips every
`blk.<n>` at or above the trunk count by name (a head is ingested from
safetensors, not from a GGUF), and the publisher's own MLX 4-bit conversion
drops `mtp.*` outright -- the same thing mlx-community's Qwen3.8 conversion
does, now observed on a second publisher. So the INT4 install clears
`speculation_blocker`'s dtype arm and still cannot draft, and the head is
coupled to a batched MoE verify rather than to an ingest: `MtpState`'s
required tensors name a DENSE FFN, while this head's is MoE.

## Batched verify and speculative decoding

Not a parity claim. Swift has no speculative decoding. **Full write-up,
method, every measurement and the standing decision:
`docs/SPECULATIVE_DECODING.md`.** Summarized here because it is a
throughput result and this is where throughput results are indexed.

Measured 2026-08-10 on AC, both real MLX INT4 installs, 16 slots, before
committing to a drafter. The question: does a batched verify of M proposed
tokens cost less, in decode-steps, than the tokens it gets accepted?

| term | measured | instrument |
| --- | --- | --- |
| expert union at M=8 | 3.78-4.70 x `top_k` | `MFERENCE_ROUTER_TRACE` + `scripts/router_window.py` |
| `c(8)`, batched vs sequential per token | 0.44 (expert shape) to 0.79 | `gemv_bandwidth_bench.rs` |
| share of compute that cannot amortize | 19% | `MFERENCE_DISPATCH_PROFILE=1` |
| accept length, n-gram drafter | 2.76 at block 8 | `accept_length_probe.rs` |
| accept length, DFlash | 4.26 at block 8 (published) | arXiv 2602.06036, 2607.07409 |

Against break-even, a trained DFlash drafter reads **1.14x at block 4**,
0.97x at block 8 and 0.87x at block 16 -- so on this engine the optimum is a
SMALL block and the win is about 1.1x, inverting the datacenter result where
verify is nearly free and bigger blocks always win. Standing decision: not
worth building at that margin, and the lever is `c(M)` rather than the
drafter. See the write-up for why, and for the two kernel optimizations that
lost.

Losslessness is settled independently of the economics: every block size
produces a token stream byte-identical to the same generation with
speculation switched off.

**QUALIFIED 2026-08-20, ON A DIFFERENT DRAFTER AND FAMILY.** That claim is this probe's, at ITS generation length, and the DFlash2 work found the limit it cannot see: byte-identity to a sequential decode holds for a few hundred tokens and then fails, because a batched verify and a decode step run DIFFERENT KERNELS (`dequant_int4_gemm_simd` against `dequant_int4_gemv_simd`) whose accumulation orders differ, so the streams part at the first near-tie. Nobody has re-run THIS probe long enough to say whether the same happens here; the mechanism is shared, so assume it does until measured. What survives exactly either way is that every BLOCK SIZE produces the same text. See `docs/DFLASH2.md`.

### DFlash2, the block drafter, measured 2026-08-19

Not a parity claim, and not the same question as the row above: that one
prices a drafter this port did not have, from published accept lengths. This
one runs a real one. Install `~/models/qwen38-27b-dflash2.gturbo` (the dense
`qwen3_5` trunk plus `incoai/Qwen3.8-27B-DFlash2`), 32-token prompt, 256
greedy tokens, 16 slots, against a ~21-22 tok/s non-speculative reference.
Full write-up: `docs/DFLASH2.md`.

**The acceptance column is deterministic** (greedy, fixed prompt, fixed
weights); the seconds are NOT, and are omitted here on purpose -- the
machine was running an interactive session throughout, which Gotcha 43
measured as an 11% error on a published row.

| block | accepted/round | committed/round | rollbacks | vs off |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4.56 | 5.56 | 73 of 109 | 0.83x |
| 7 | 4.37 | 5.37 | 66 of 112 | 0.92x |
| 4 | 2.98 | 3.98 | 58 of 151 | 1.06x |
| 2 | 1.74 | 2.74 | 38 of 219 | **1.35x** |

**AT 600 GENERATED TOKENS. The 256-token version of this table, published
here for a day, read 8.09 committed per round at block 8 and 1.43x** -- the
first ~250 tokens of that answer are a code block, far more predictable than
the prose that follows, so a short generation measures the easy part and
reports it as the whole. Accept length and speedup are functions of
GENERATION LENGTH as well as of the prompt.

**NOT byte-identical to a non-speculative decode over a long generation**, which an earlier draft of this row claimed. Acceptance is exact -- a proposal is kept only when it equals the target's argmax -- but the committed token comes from a BATCHED verify row, and `dequant_int4_gemm_simd` accumulates differently from the decode path's `dequant_int4_gemv_simd`, so at the first near-tie the streams part. Measured: they agree for ~200 tokens on the protocol prompt, then take different but equally fluent continuations, both stopping on endOfTurn. Every BLOCK SIZE does produce identical text to every other, which is the exact invariant that survives and is what identifies the kernel pair rather than the batch width as the cause. `docs/DFLASH2.md` carries the measurement.

**READ THE ACCEPT LENGTH AGAINST ITS PROMPT AND ITS LENGTH.** 5.56 committed
per round now sits beside vLLM's published 5.34 and llama.cpp's 4.92-5.08
rather than above them. The second workload that was owed here has since
been run and it lands the other way: on the protocol's prose case the same
drafter at block 2 measures 0.88x throughput and +17.4% J/token
(`docs/DFLASH2.md`). Speculation on this engine pays on predictable
continuations and costs on ordinary prose.

**Measured through the REAL generation loop, which is what a user runs**
(`MFERENCE_SPEC_STATS=1`, 200 greedy tokens, ~22.1 tok/s non-speculative
arm, same install). The probe above hand-rolls its own round; this is
`run_raw_completion_speculative`:

| prompt | block | acceptance | rollbacks | vs off |
| --- | ---: | --- | ---: | ---: |
| code | 2 | 0.93 0.98 | 9% | **1.47x** |
| code | 8 | 0.96 avg | 27% | 1.34x |
| prose | 2 | 0.80 0.66 | 48% | 0.97x |
| prose | 4 | 0.83 0.67 | 73% | 0.73x |
| prose | 8 | 0.78 0.64 | 98% | **0.48x** |

The loop reproduces the probe on the probe's own prompt, so the spread is the
WORKLOAD and not the loop. Throughput tracks the ROLLBACK RATE: a rejected
batched round on this recurrent family restores a gated-DeltaNet snapshot and
replays, and that is what a bigger block buys more of. Small blocks win here
too, and this drafter's serving default is therefore 2 rather than its
trained 8 -- the same number the MTP head reached independently.

**Do not quote an accept length without its prompt.** The 8.09 committed per
round above is one predictable prose answer; on the standing smoke's prompt
the same drafter at the same block is a 2x loss.

## Power

Not a parity claim. Swift was never measured for power, here or upstream;
this is this port measuring itself, like the Quality section above.

**Full write-up, method, hygiene audit and caveats: `docs/POWER_BASELINE.md`.**
Reproduce with `scripts/power.sh`. ROADMAP Phase P1.

Measured 2026-08-07 across two sessions, AC and battery, one binary. 16
expert-cache slots, frozen protocol, `powermetrics` at 200 ms windowed to
the measured run alone by the `[power-window ...]` markers
`turbospark-bench` emits. Watts are CPU+GPU+ANE, not wall. The AC rows below
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

Four results worth carrying, each detailed in `docs/POWER_BASELINE.md`:

- **AC vs battery answers AGENTS.md Gotcha 22, which had stood unmeasured.**
  Energy is not the axis that moves: watts and J/token differ by a few
  percent with no consistent sign. Thermal headroom is: on AC 50 of 50
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

## Checkpoint intake: the Xet bridge, and what parallel range reads buy

Measured 2026-08-14 on AC, this machine. **This is the one section here that
measures the NETWORK rather than the machine**, so it is the least
reproducible page in the file: the link, the CDN edge and the time of day all
move it. Read the ratios, re-measure before quoting an absolute, and do not
compare a row here against a row taken in another session.

Hugging Face replaced Git LFS with Xet, which is a storage and transfer layer
rather than a format: content-defined chunking and dedup underneath the Hub,
with the file reconstructed byte-identically at the client. Nothing about
GGUF or safetensors parsing, the `.gturbo` install, the kernels or MLX
changes, and no compatibility work was needed. What did need measuring is
throughput, because the LFS-compatible bridge every `resolve/...` URL
redirects to is SINGLE-STREAM: one connection to one CloudFront edge, and
`xet-core` issue #821 documents 65-75% of those edges capped at 8.7 MB/s.

### The cap, against this repo's own recorded repack times

End-to-end wall clock of a streamed repack, so these bound the transfer rate
from below rather than measuring it (each also does transcode and write work):

| checkpoint | streamed | wall clock | effective |
|---|---|---|---|
| Gemma 4 26B-A4B Q8_0 | 26.9 GB | 24 min | 18.7 MB/s |
| Qwen 3.6 35B-A3B Q4_K_M | 20 GB | 23 min | 14.5 MB/s |
| Qwen3-30B-A3B Q4_K_M | 17.3 GB | 24 min | 12.0 MB/s |
| Mistral 7B Q4_K_M | 4.1 GB | 5 min | 13.7 MB/s |
| gemma4 UD-Q3_K_M | 12 GB | 17.5 min | 11.4 MB/s |
| **gpt-oss-20b MXFP4** | 12.1 GB | 25 min | **8.1 MB/s** |
| **Bonsai-27B 1-bit** | 5.13 GB | 9.7 min | **8.8 MB/s** |

The two bold rows sitting on 8.1 and 8.8 against a documented 8.7 is what
first made the cap worth measuring directly rather than inferred.

### Serial against parallel, at the size the walk dispatches

Against the real `gpt-oss-20b-MXFP4.gguf`, one connection per stream. The
64 MiB row is the operative one: that is `MAX_RANGE_BYTES`, and 512 MiB is
about the size of one routed tensor.

| chunk | arm | wall clock | rate |
|---|---|---|---|
| 64 MiB | serial, 8 chunks | 60.4 s | 8.9 MB/s |
| 64 MiB | 8-way, same 512 MiB | 17.6 s | **30.5 MB/s** |
| 16 MiB | 1 stream | | 10.4 MB/s |
| 16 MiB | 4 streams | | 16.0 MB/s |
| 16 MiB | 8 streams | | 23.4 MB/s |

**3.4x at 64 MiB**, and the serial arm landing on 8.9 against the documented
8.7 is the reading that says the cap is what is being measured rather than
the link. Scaling is sublinear, which is why `RANGE_CONCURRENCY` is 8 and not
higher: past that the shared link is the limit and more streams buy only more
sockets to drop.

### What the 3.4x does and does not cover

The concurrency engages only when ONE `read_range` exceeds `MAX_RANGE_BYTES`,
and `gguf_checkpoint::read_tensor` issues one call per TENSOR. So:

- **MoE checkpoints get it.** A routed tensor is a layer's whole expert table
  (Gemma's `ffn_gate_up_exps` is ~410 MiB, seven chunks), which is the
  dominant share of those files' bytes.
- **Dense checkpoints get essentially nothing.** Their largest tensor is
  under the cap. TinyLlama re-streamed in 3:28 against a recorded ~3 min,
  unchanged, because not one of its 201 tensors chunked.

Extending it to dense checkpoints means reading several tensors concurrently,
which is a change to the walk rather than to `ranged_download.rs`. No
full-walk MoE timing has been taken yet; the 3.4x is measured on the wire.

### Lessons

- **The concurrency is not the optimization; `http1_only()` is.** The bridge
  speaks HTTP/2, and reqwest will multiplex every concurrent range GET onto
  ONE connection, hence one edge, hence the same cap. There is no error and
  nothing in any log to say the knob did nothing, only the old wall clock.
  Any future change to that client has to confirm the connections really are
  distinct before a number from it is believed.
- **The cheapest fixture was the one that could not see the effect.**
  TinyLlama was picked for the end-to-end gate because it is the cheapest
  real walk. It was the right CORRECTNESS gate (`model_weights.bin` came out
  SHA-256-identical to the install already on disk) and the wrong THROUGHPUT
  one, for the same reason it is cheap: it is small, so nothing chunks. Same
  species as the tidy one-line prompt that missed the `trim` (AGENTS.md
  Gotcha 41).
- **A quality question about a transport does not need the transport
  adopted.** Native `hf-xet` (1.6.0, Apache-2.0) was costed and declined: it
  pulls tokio and a large tree into a crate that is `#![forbid(unsafe_code)]`
  plus a `xet-read-token` auth flow, to buy adaptive concurrency (had far
  more cheaply above) and chunk dedup, which is worth nothing when every
  checkpoint here is streamed exactly once, never kept, and two
  quantizations of one model share no chunks.
- **`x-linked-etag` on a resolve URL is exactly the SHA-256 of the file
  content**, and `x-linked-size` its byte length, both free from headers with
  no auth and no download (verified against
  `Bonsai-27B-mlx-1bit/tokenizer.json`). The streamed walks verify nothing
  about their source bytes today. Not wired; recorded because it is the
  cheapest integrity anchor available to this repo.

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

### The seventh family: `Muse-Glimmer-30B` (MLX INT4)

`mlx-community/Muse-Glimmer-30B-4bit` @ `3e7677d7`, streamed into a 15 GB
install (resident region 15,670,395,904 bytes) and run on `families/museglimmer/`,
the sixth decode flow. Apple M4 Max, AC, release, 16 expert-cache slots
(inert -- the model is dense), 2026-08-15.

**Read these with the window and the budget.** This family runs the protocol
at **8,192 context and a 2,048 generation budget**, not the shared
4,096/1,024, and it is the second family to move both. It reasons before
answering: its template writes `Reasoning strength: high.` into the system
preamble and the model emits a `to=self` message before its `to=user` one, so
at 1,024 even the SHORT case stops on `maxTokens`. A row at one window says
nothing about another (`crates/bench` Gotchas 11 and 12).

| case | prompt | new | tok/s | stop |
|---|---|---|---|---|
| short-explanation | 102 | 1,132 | 13.291 | endOfTurn |
| medium-review | 464 | 1,498 | 15.341 | endOfTurn |
| long-synthesis | 2,820 | 1,552 | 14.483 | endOfTurn |

Peak `phys_footprint` **535 MiB**, replay +1.19 MiB.

**The 15.7 GB of resident weights are absent from that peak**, which is the
THIRD independent re-derivation of AGENTS.md Gotcha 40 (Mistral 7B: 4.07 GiB
of weights, 684 MiB peak; Qwen3.8-27B: 15.1 GB, 660 MiB). The accounting,
computed from shapes before the run:

| term | value |
|---|---|
| KV, 13 full layers x 8,192 | 104.0 MiB |
| KV, 39 sliding layers x 2,176 (window 2,048 + 128 chunk headroom) | 82.9 MiB |
| sum | 186.9 MiB |

leaving ~349 MiB of process baseline and host scratch at a 202,048-wide
vocabulary. Note the KV term is SMALLER than Qwen3.8's 256 MiB despite twice
the window: 2 kv heads at 128 against 4 at 256, and three quarters of the
layers ring rather than running the full window. The window alone does not
tell you the KV bill.

Quality, from TWO fresh processes agreeing to the last digit and the last hex
character:

| | value |
|---|---|
| reference-answer perplexity | 6.2826 |
| greedy digest | `fc1e4e58a6fd9975...` |
| sampled digest | `24fe355decf2f389...` |
| 8-slot digest | EQUAL to the 16-slot one |
| constrained arm | 1.00x / 0.99x |

The perplexity is the number that says the assistant prefix is right. This
family's generation prompt ends at `<|start|>assistant` and the model's next
emission is a recipient, so the reference answer spliced in raw lands in no
message at all -- the position that made gpt-oss read 148,421.76. A healthy
single digit beside coherent generations rules that out. It is NOT a ranking
against the other families: the corpus is the frozen protocol's, chosen for
Gemma and reused verbatim, and each family's template puts the reference
answer somewhere different.

The constrained arm reads ~1.00x rather than the MoE families' 0.85-0.94x
because `--expert-cache-slots` sizes a routed-expert cache and a dense model
has none; the two arms differ only in noise.

**Not measured:** no cross-engine KL against mlx-vlm.

**POWER: the unconstrained cost is measured, the sustained cost is not.**
Three AC captures (2026-08-16, quiet machine, at three starting temperatures)
agree to 1.5%: decode **37.69-38.24 W at 2.0135-2.0444 J/token** (n=3 across three
sessions), which is the highest draw recorded in `docs/POWER_BASELINE.md`. That number comes
from the only Nominal windows either capture produced -- this install
saturates thermally within ~2 minutes whatever its starting temperature, so
the harness's measured pairs are all governed and their J/token wanders 10%
between runs. The governor's own descent is the interesting part: 27.46 W at
18.347 tok/s against 37.69 W at 18.593, i.e. **26% less energy per token for
1.3% less throughput**. The `performance,efficiency` A/B this argued for was then RUN and is
inconclusive: the performance arm throttles on every pair and its 25% spread
swallows the efficiency arm's. What it did settle is that the rate cap holds
10.00 tok/s to 0.01% and keeps the machine out of thermal governance
entirely, where the performance arm never is. Full write-up:
`docs/POWER_BASELINE.md`.

