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

**MATCH THE BACKEND, NOT JUST THE BYTES.** This was first measured against
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

### Sub-4-bit candidate survey (ROADMAP Phase S)

NOT A MEASUREMENT OF THIS PORT. This port cannot ingest IQ3_XXS, so there
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

## Power

NOT A PARITY CLAIM. Swift was never measured for power, here or upstream;
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
