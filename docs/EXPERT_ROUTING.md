# Expert Routing: Domain Concentration (Measured Negative)

The question this page answers: can a domain-restricted expert set,
profiling which experts a coding corpus routes to and then pruning the
cold ones from the install or pinning the hot ones in the cache, save
memory or win throughput over the streaming expert cache that already
ships?

The answer, measured 2026-08-08 on the real Gemma 4 26B-A4B install, is
no. Routing under a coding workload is tilted by domain but nowhere near
concentrated by it, and both halves of the idea (pruned install, pinned
warm set) lose to the existing LFU cache. Recorded here so the idea is not
re-derived from first principles; it also appears as ROADMAP dead end 11
and in DEVIATIONS.md's MoE entry next to the Swift prefetch dead end.

## Where the question came from

Abliteration-style tooling (e.g. NousResearch/llm-abliteration) invites
the mental model "identify the region of the model that does X, keep only
that". Abliteration itself does not do that: it finds one DIRECTION in the
residual stream (difference of mean activations over paired prompt sets)
and orthogonalizes the weights against it. The weights stay the same size
and every layer still runs -- it edits behavior, it does not localize a
capability into a prunable region. In a dense model there is no "coding
portion" to skip at all.

In an MoE model there is a real candidate for the region: the routed
experts. Gemma 4 26B-A4B routes each token through the top 8 of 128
experts per layer, and this engine already exploits that dynamically:
routed experts are not resident, `crates/streaming` keeps 16 of
128 per layer in pinned slots (more where the machine has memory to spare;
16 is the floor and what this measurement used) and streams misses by
`pread`. The open
question was whether a static, domain-specific expert set could beat that
dynamic mechanism. That is an empirical question about routing statistics,
so it was measured before anything was built.

## Instrument

`TURBOSPARK_ROUTER_HIST=/path.json` on `RealForwardRunner`
(`crates/runtime/src/router_hist.rs`): counts, per layer, how often each
expert appears in the router's top-k, over every forward pass (prefill and
decode alike, the `TURBOSPARK_PHASES` divisor convention). Host-side count
taken after `router_topk_gemma4` returns; it touches no decode math, adds
nothing when the env var is unset, and dumps one JSON on runner drop.
Sanity invariant on any capture: per-layer counts sum to
`top_k * forward_passes` (held exactly: 1848 = 8 x 231).

`scripts/router_hist.py` (stdlib only) reads one or two capture groups and
reports, per layer: the smallest expert set covering 95% and 99% of routed
mass, the never-routed count, and (with two groups) the Jaccard overlap of
the two 95%-mass hot sets.

## Measurement

Real 26B install (`~/models/gemma4.gturbo`), greedy (`--seed 1
--temperature 0.0001 --top-k 1`), 200 new tokens per prompt, four coding
prompts (Rust CSV parser, Python perf fix, B-tree indexes, TypeScript
debounce) vs four general prompts (wetlands, monarch migration, French
Revolution, marathon training), ~920 forward passes per corpus. All
numbers deterministic for a fixed corpus and install; per-layer detail
reproduces from the captures.

Mean over the 30 MoE layers, 128 experts each:

| | coding | general |
|---|---|---|
| experts covering 95% of routed mass | 66.8 | 66.5 |
| experts covering 99% of routed mass | 86.6 | 85.7 |
| never-routed experts | 21.5 | 23.9 |
| Jaccard overlap of the two 95% hot sets | 0.455 | |

Per-layer spread: n95 ranges 48-96, Jaccard ranges 0.213-0.790 (highest at
the first and last layers, lowest mid-stack).

## Reading

- **Not concentrated.** 95% of routed mass needs ~52% of the expert table,
  on both corpora. There is no small "coding region" to keep.
- **Tilted, though.** Jaccard 0.455 between two ~67-expert sets is real
  divergence: coding routes differently, just not narrowly. The union of
  the two hot sets is ~92 of 128 experts.
- **A pinned coding set loses to the cache.** A static 95%-mass set is ~67
  slots per layer; the LFU cache reads ~84% hit rate from 32 slots by
  exploiting temporal locality within a generation, which domain
  statistics cannot see.
- **A pruned install loses worse.** The 5% tail is whole experts of routed
  mass. The quality gate's sensitivity curve reads +10.5% perplexity from
  flipping one quantization level in 0.0122% of expert bytes
  (`docs/BENCHMARKS.md`); zeroing entire routed experts is orders of
  magnitude more damage than its detection floor.
- **Corpus-size caveat, bounded.** ~920 passes per side is small; the
  never-routed counts would shrink with more tokens. The n95 numbers
  cannot: more data spreads coverage further, it does not concentrate it,
  so the conclusion is not rescued by a bigger corpus.

## Standing decision

Pruned installs and pinned domain warm sets are closed (ROADMAP dead end
11) unless a corpus measurement contradicts the table above. The
instrument stays wired as a diagnostic: re-running the whole measurement
is two capture batches and one script invocation, so any future claim of
domain-concentrated routing (a different family, a domain far narrower
than "coding") should arrive with its own histogram.

What remains open, and is a different mechanism entirely, is
abliteration's actual trick applied at repack time: a directional weight
edit for behavior steering. That is ROADMAP Later/Optional ("Directional
Weight Steering"); it changes behavior rather than memory and would be
judged by the Phase Q perplexity/digest gate and the logit-dump KL
machinery.

## The prefetch ceiling, and the death of the previous-token predictor

A second, independent question reached this page on 2026-08-29, prompted by
`JustVugg/colibri`'s PILOT: it runs layer L+1's router GEMV on layer L's
post-attention residual and reports 71.6-75.8% recall, against 41.3% for a
previous-token predictor. `DEVIATIONS.md` already rejects expert prefetch,
but on a DIFFERENT predictor -- one that copies layer L's selected expert
ids (Jaccard 0.039). An id-copy predictor and a stale-state router
evaluation are not the same experiment, so that record does not close this.

Before building a predictor, the CEILING was measured, because a prefetcher
can only ever convert a miss into a hit and the miss count is a property of
the trace and the cache policy alone.

### Instrument

`scripts/pilot_ceiling.py` replays an `TURBOSPARK_ROUTER_TRACE` capture through
a port of `ExpertCache`'s LFU policy and reports hit rate, miss composition,
and what a previous-token predictor would have covered. Stdlib only.

Capture: real 26B install, greedy (`--seed 1 --temperature 0.0001 --top-k
1`), 21 prompt + 200 generated tokens, 220 passes on all 30 routed layers,
36.6 tok/s. The capture reconciles exactly (52,800 = 220 x 30 x 8) and the
script refuses one that does not.

### The simulator reproduces the engine's own recorded numbers

This is what makes the rest of the table worth reading, and it was checked
against figures nothing in the script can see:

| quantity | recorded | this simulation |
|---|---|---|
| misses per layer, 32 slots | 1.3 (`crates/streaming/CLAUDE.md` Gotcha 3) | 1.23 decode-only, 1.37 all-pass |
| bytes per token, 32 slots | 125 MiB (same Gotcha) | 118 MiB decode-only, 132 MiB all-pass |
| hit rate, 32 slots | ~84% (this page, above) | 84.6% decode-only |

Expert stride is the install's own: `layer_00.bin` is 429,916,160 bytes over
128 experts, so 3.203 MiB each, which is where the MiB column comes from
rather than from Gotcha 36's rounded ~3.2.

### Measurement

Decode only (prefill warms the cache but is excluded, per AGENTS.md Gotcha
21's divisor rule):

| slots | hit rate | misses/layer/pass | cold | evicted | covered by previous pass |
|---|---|---|---|---|---|
| 16 | 66.3% | 2.70 | 6.0% | 94.0% | **0.0%** |
| 32 | 84.6% | 1.23 | 13.1% | 86.9% | **0.0%** |

### Three findings

**The ceiling is large enough to be worth chasing.** At 32 slots the misses
are 118 MiB/token, which `crates/streaming/CLAUDE.md` Gotcha 3 times at
5.26 ms for its own 125 MiB at that same slot count. Read that against a
decode step AT 32 SLOTS and not against this capture's 36.6 tok/s, which was
taken at the pinned 16 (Gotcha 58's rule: a frozen number belongs to its
configuration). At the ~42 tok/s that 32 slots buys (`DEVIATIONS.md`'s +15%)
the step is ~24 ms, so expert io is about a fifth of decode. At 16 slots the
miss count is 2.2x higher. A perfect prefetcher hides at most that, and
`TURBOSPARK_ROUTED_PIPELINE` has already taken 1.1 ms of it, so the remaining
prize is real but smaller than the raw bucket suggests.

**The previous-token predictor is dead STRUCTURALLY, not statistically.**
Zero of 7,359 misses at 32 slots, and zero of 16,100 at 16, were named by
the previous pass at that layer. Not approximately zero. The reason is a
property of the policy rather than of the data: an expert the previous pass
requested was inserted with a fresh clock and an incremented count, so it
cannot be the LFU victim one pass later. No tuning rescues this predictor,
which is the sharper form of the claim `DEVIATIONS.md:793` inherited from
upstream and is now independently re-derived here.

**Routing turnover matches colibri's model closely enough that its PILOT
number may transfer.** Previous-token recall reads 41.3% all-pass and 41.5%
decode-only, against random chance of 8/128 = 6.25%. colibri reports 41.3%
for the same predictor on GLM-5.2, a different architecture (256 experts,
top-8) under a different engine. The agreement to the decimal is
coincidence; the agreement in MAGNITUDE is the finding, and it says the
~59% of the top-k that turns over every token is the set a stale-state
router would have to name.

## PILOT measured: the predictor works, and it still loses

`TURBOSPARK_PILOT_PROBE=1` runs layer L+1's router on layer L's
post-attention residual and records the guess beside the actual selection.
It costs ONE extra GEMV per MoE layer rather than a norm plus a GEMV,
because Gemma's router pre-norm is `encode_rms_norm_no_scale` -- weightless,
carrying no per-layer tensor, so the `router_x` layer L already computed IS
the input layer L+1's router would see. A family whose router norm has a
weight owes its own norm encode.

### The predictor reproduces colibri's number

**70.6% top-k recall**, against colibri's reported 71.6% on GLM-5.2 -- a
different architecture (256 experts against 128), a different engine, and
an independent implementation. Together with the 41.3% previous-token
agreement above, the routing structure PILOT exploits is clearly present
here.

### And it reads more bytes than it saves, at every width

Coverage is not the metric; a prediction naming an expert that is already
resident buys nothing, and at 84.6% hit rate most of the top-k is resident.
Priced in reads, at 32 slots:

| PILOT_K | miss coverage | reads per hit saved | total bytes vs baseline |
|---|---|---|---|
| 1 | 12.5% | 1.26 | **1.03x** |
| 2 | 20.7% | 1.39 | 1.08x |
| 4 | 35.9% | 1.65 | 1.23x |
| 6 | 49.0% | 1.96 | 1.47x |
| 8 (full) | 60.4% | 2.40 | **1.85x** |

Nothing dips below 1.00x. The same sweep at the pinned 16 slots runs 1.02x
to 1.59x, so the shape is not an artifact of one cache size.

**The mechanism, which is the part that generalises.** A prefetcher's COST
scales with its prediction width (it reads top-k experts whether or not they
were going to miss) while its BENEFIT scales with the MISS RATE. Here the
LFU cache already answers 84.6% of requests, so the engine is guessing 8
experts wide to catch 1.23 misses, and the ~2.4 of 8 guesses that are wrong
each cost a full 3.2 MiB read. colibri wins the same trade because its cache
is far colder -- it streams 370 GB from NVMe with a small resident fraction,
so its miss rate is near 100% and almost every prefetched byte is a byte it
needed anyway.

**The second half of the trade is what kind of cost the read is.** colibri
spends bandwidth to hide NVMe LATENCY, which is a good trade when a demand
read blocks. This engine's expert read is a page-cache memcpy at ~23.8 GiB/s
(`crates/streaming/CLAUDE.md` Gotcha 3), so the cost is BANDWIDTH, on a
unified-memory machine where the GPU's matmuls are competing for it. Buying
overlap with 1.03x to 1.85x the bytes is the wrong direction.

### Standing decision, and the condition that would reverse it

Router-lookahead prefetch is closed on this engine WARM, on the terms above
rather than on the id-copy predictor's terms that `DEVIATIONS.md:793`
closed. The honest bound at the best operating point (`PILOT_K=1`): 12.5% of
a 5.26 ms io bucket is 0.66 ms of a ~24 ms step, so 2.7% BEFORE subtracting
the extra 3% of bytes and the per-layer GEMV dispatch. That is inside the
noise of the +2.5% `TURBOSPARK_ROUTED_PIPELINE` has already banked, and
distinguishing it from zero would need an interleaved A/B that the ceiling
does not justify building.

It would reverse on a machine where the expert read is genuinely disk-bound
rather than a page-cache memcpy -- a cold cache, or a host too memory-tight
to hold the expert table -- because that changes the cost from bandwidth to
latency and raises the miss rate, which is both of the terms above at once.
The probe stays wired for exactly that re-measurement.

**THAT CONDITION IS NOW CREATABLE ON DEMAND, AND WHICH ONE A RUN WAS IN IS
NOW READABLE.** Both were prose until 2026-08-29 (`crates/streaming/CLAUDE.md`
Gotchas 8 and 3): nothing in the tree could establish the disk-bound arm, and
nothing could say afterwards whether it had been established, which is the
same gap AGENTS.md Gotcha 28 records for thermal pressure and Gotcha 43 for
background load. The two seams are

```sh
TURBOSPARK_PHASES=1 TURBOSPARK_EXPERT_DISK_IO=1 TURBOSPARK_EXPERT_NOCACHE=1 \
  ./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 200 --expert-cache-slots 16
```

and the `expert bytes` row of the phase footer reports requested MiB/token,
physical MiB/token and their ratio. Read the physical number rather than the
flag: `F_NOCACHE` prevents RETENTION and does not evict, so a blob a previous
run already faulted in stays resident and needs `sudo purge` beside it.

**The condition has been established once, and the arithmetic above is
confirmed to describe the warm case only.** Real Gemma 4, 16 slots: warm
reads 0.0 MiB/token physical against 274.9 requested, and the purged
bypassed arm reads 274.9 of 274.9 (1.00x), i.e. every routed byte off the
device. That 1.00x is also the answer to a question this page never asked --
`F_RDADVISE` is not over-reading, since the disk-bound arm pulls exactly the
strides the cache asked for and no more.

**Re-running the PILOT sweep under that condition is OWED and has not been
done.** The cost model above is a warm-cache measurement throughout, and both
of its terms are expected to move: the miss rate rises, and each miss stops
being a ~23.8 GiB/s memcpy. Until that sweep exists, the standing decision is
scoped to the warm case and says nothing about the cold one.

Note also what does NOT reopen it. `garnermccloud/sglang-ssd-stream` reports
164.7 tok/s streaming a 47.68 GiB lookup table from NVMe and hiding the reads
behind GPU compute, which reads like a refutation and is not one: its row
addresses are a function of the INPUT TOKEN IDS and are therefore known
before the consuming block, so its prefetch wastes no bytes by construction.
A routed expert set is the router's OUTPUT for that layer, so lookahead here
is a PREDICTION and pays the cost model above. Their result is evidence about
overlap, not about prediction.

### The instrument nearly reported a false negative

The first wiring read **7.7% recall against a 6.25% random baseline** --
indistinguishable from "PILOT does not transfer", and it would have been
published as a negative. It was an off-by-one: the probe buffer written
during layer L's attention holds the guess about L+1, and it was being filed
against layer L. AGENTS.md Gotcha 57's rule caught it (near-random means
UNRELATED, so suspect the instrument before believing the finding).

`TURBOSPARK_PILOT_PROBE=self` is the guard that now exists because of it: it
aims the probe at the layer it is already running in, so the prediction
reproduces the production router and recall MUST read 100%. It does, and the
same run pins the analysis script's cost accounting at exactly 1.00 read per
hit -- a case whose answer is known a priori, which caught a second bug where
residency was sampled AFTER the plan had already installed the misses and
every prefetch therefore looked free.

### Mutation check

The simulator was mutated three ways before its numbers were believed.
Swapping LFU for pure LRU moves the answer (84.6% -> 83.2%, 7,359 -> 8,002
misses), so the policy port is load-bearing. Two mutations SURVIVED, both
because an invariant does the work rather than because the measurement is
weak, and both are findings about the Rust rather than about the script:

- Removing the empty-slots-first arm changes nothing, because an occupied
  slot always holds an expert whose count was incremented when it was
  placed, so `count >= 1` sorts it behind an empty slot's 0 anyway. Verified
  on the real trace: zero occupied slots ever carry `use_count == 0`.
- Incrementing the use counts BEFORE computing the eviction order instead of
  after changes nothing, because no evictable slot can hold an expert this
  pass requested: requested experts are either hits (whose slots are
  reserved) or misses (resident nowhere).

Both details are worth keeping in `expert_cache.rs` for readability, but
neither can change an answer, so neither is worth defending in a test.
