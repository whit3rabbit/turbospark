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

The same captures price cache POLICY: `scripts/expert_policy_replay.py`
replays them back to back through the engine's LFU, LRU, decayed LFU, a
cross-layer pool and Belady's offline optimum. What it is for, and the
long-lived-runner question none of the numbers on this page could see, are
in `docs/MOE_PRIOR_ART.md`.

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
| misses per layer, 32 slots | 1.3 (`crates/streaming/AGENTS.md` Gotcha 3) | 1.23 decode-only, 1.37 all-pass |
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
are 118 MiB/token, which `crates/streaming/AGENTS.md` Gotcha 3 times at
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
(`crates/streaming/AGENTS.md` Gotcha 3), so the cost is BANDWIDTH, on a
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
NOW READABLE.** Both were prose until 2026-08-29 (`crates/streaming/AGENTS.md`
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

**The PILOT lookahead sweep under that condition is still OWED.** The cost
model above is a warm-cache measurement throughout, and both of its terms are
expected to move: the miss rate rises, and each miss stops being a ~23.8 GiB/s
memcpy. Until that sweep exists, the standing decision is scoped to the warm
case and says nothing about the cold one.

**What the cold arm costs, and what does not fix it (2026-10-12, a reading,
not a frozen row).** Gemma 4 26B-A4B INT4, streamed residency, one prompt,
200 greedy tokens, AC power, 3 interleaved warm/cold pairs per slot count
(host load 3.0 to 5.5, no other benchmark running). Cold is `sudo purge` plus
`TURBOSPARK_EXPERT_NOCACHE=1`; every cold run read 1.00x physical. Warm is
one discarded pass, then the measured pass.

| slots | hit rate | requested MiB/token | cold tok/s (range) | cold pread ms/token | warm tok/s |
|---|---|---|---|---|---|
| 16 | 64.3% | 274.2 | 10.64 (10.60-10.83) | 50.9 | 45.6 |
| 32 | 81.0% | 146.3 | 16.04 (15.98-16.58) | 29.0 | 50.8 |
| 64 | 93.5% | 50.1 | 39.23 (39.02-39.79) | 10.1 | 52.3 |

- The cold arm is disk-bound and reads about 5.3 GiB/s effective at 16 slots.
  Cold decode is 4.3x slower than warm at 16 slots and 1.3x slower at 64.
- The lever is the miss count, not the bytes per miss. 16 to 64 slots cut
  requested bytes 5.5x and lifted cold decode 3.7x, while warm decode moved
  15%. 64 slots cost 6.0 GiB of pinned cache against 1.5 GiB at 16.
  `ExpertCacheSlots::Auto` already climbs with headroom (48 slots on a 36 GB
  machine with this install), so a machine like this one sits between the 32
  and 64 rows by default. The 16-slot row is what a 16 GB machine gets, since
  Auto never resolves below 16.
- Shrinking the blobs does not compete. Lossless zstd -3 on three 256 MiB
  samples of an INT4 g64 `model_weights.bin` saved about 10% (241 of 268 MB).
  Even with free decode that removes about 5 ms of the 94 ms cold token at 16
  slots, and about 1 ms of 25.5 ms at 64. Decode would have to sustain more
  than the disk's 5.3 GiB/s to break even, which was not measured. A 3x byte
  reduction is not available losslessly, and `gemma4-iq3` is 12.8 GB against
  13.0 GB for INT4, so a lower-bit catalog row does not shrink this family's
  experts either. This closes the AirLLM "block-wise compression, 3x faster
  loading" idea for this engine: AirLLM's gain comes from re-reading the whole
  model per token, which this engine never does.
- A cold-only anomaly stays unexplained: `gpu busy: cb1` was 13.0 ms/token
  cold against 5.1 warm at 16 slots, 10.9 at 32, and 6.0 at 64 (warm 5.1
  throughout). It tracks miss bytes and persisted with the host at load 3, so
  it is not the earlier concurrent benchmark. The mechanism is not identified,
  so the cold pread figure above may include time that is not purely the SSD.
- Reproduce: `docs/BENCHMARKING.md`, Measurement hygiene (purge before every
  cold arm, discarded refill pass before every warm arm, confirm 1.00x per
  cold run). The install was `turbospark pull gemma4`. The 18 measured runs
  and their load logs lived in a session scratch directory and are not
  committed; the table above is the record.
- Scope: one prompt, one 200-token generation, one machine and one model.
  Hit rates are prompt-dependent (an earlier doc figure was about 84% at 16
  slots on a different run), and the first warm 16-slot pass still read 18 MiB
  physical (40.9 tok/s against 45.6 on the other two). Not run: a quality
  comparison, a 16 GB machine, or the mapped residency arm under the same cold
  condition.

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

## Cache policy: cumulative LFU forgets nothing

Prompted 2026-10-13 by `skeggsguy/Flash-next-ssd`, a llama.cpp fork that
streams Qwen3.8-Flash-Next's experts from SSD. Its slot cache evicts by
hotness with every counter halved each 1024 tokens, and its replay simulator
put pure LRU at +11% misses for that model. This engine's `ExpertCache` LFU
never decays, which no measurement here had priced, because every capture on
this page is ONE request replayed through a FRESH cache. A server or app
runner keeps its streamers for its whole life, so a later request inherits
every count an earlier one left.

### Instrument

`scripts/pilot_ceiling.py --policies lfu,lru,lfu-decay:N` replays one or more
`TURBOSPARK_ROUTER_TRACE` captures through ONE cache per layer, in order, so
several captures read as one session. `lfu-decay:N` halves every count after
each `N` committed plans at that layer, which is the shape
`ExpertCachePolicy::LfuDecay` implements. The port was checked against a
synthetic drifting hot set (cumulative LFU 74.8% hits, LRU 99.8%, decay 98.1%)
and still reproduces this page's single-capture numbers (68.0% / 85.0% at
16 / 32 slots for a 400-token capture, against 66.3% / 84.6% recorded for
the 200-token one).

Capture: real Gemma 4 install (`~/.turbospark/models/text/gemma4.gturbo`),
greedy, 400 new tokens each, eight prompts (four coding, four general, the
same topics as the hot-set measurement above), interleaved coding and general.
3,467 passes on 30 routed layers. The session sweep skips the first capture's
435 passes so it measures inherited state.

### Measurement

Session, eight requests through one cache:

| slots | cumulative LFU | LRU | decay 16 | decay 32 | decay 128 | decay 1024 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 55.3% | 61.6% | **64.5%** | 63.9% | 60.3% | 56.0% |
| 32 | 71.4% | 80.4% | **81.6%** | 81.3% | 77.4% | 72.4% |
| 64 | 90.9% | 95.1% | | **95.2%** | 93.8% | 91.5% |

At 32 slots that is 220 MiB of expert reads per pass under cumulative LFU
against 144 MiB at decay 32, a 35% cut, from a policy change alone.

Single requests, decode only, fresh cache (the case this page already
measured, and the reason LFU was chosen over LRU):

| prompt | slots | LFU | LRU | decay 16 | decay 32 |
|---|---:|---:|---:|---:|---:|
| c1 (Rust CSV) | 32 | 85.1% | 84.4% | 85.6% | 86.0% |
| g1 (wetlands) | 32 | 83.4% | 82.2% | 83.7% | 84.2% |
| c3 (B-trees) | 32 | 80.5% | 78.7% | 80.9% | 81.4% |
| g4 (marathon) | 32 | 80.6% | 78.9% | 80.4% | 80.8% |

At 16 slots the same ordering holds within 1.5 points.

### Reading

- **Cumulative LFU is the worst policy measured once a runner serves more
  than one request**, below plain LRU by 6 to 9 points. An expert that was
  hot for the last topic keeps its count and outranks the current topic's
  experts, which then evict each other.
- **Decay at 16 to 32 plans is at least as good as both alternatives in
  both regimes.** Inside one request it keeps LFU's edge over LRU; across
  requests it beats LRU slightly. Flash-Next's 1024-token interval does not
  transfer: on Gemma 4 it is barely better than no decay.
- This is a hit-rate result. Converting it to tok/s needs the expert-io
  share of the step (`docs/DECODE_BUDGET.md`: about a sixth of decode at 32
  slots, warm), and on a disk-bound install every miss avoided is an SSD
  read, which is where it should matter most.

### Real-install check (2026-10-13, Apple M4 Max, AC)

`TURBOSPARK_EXPERT_CACHE_DECAY=32`, one request (c3), 16 slots, 200 tokens:

| arm | greedy md5 | sampled md5 (`--temperature 0.7 --seed 1`) | greedy hits |
|---|---|---|---:|
| pre-change binary | `7cf23377586f9ecb23e83d3fd579e984` | `c65e3683df4cabd571cad42b3a751dc8` | 34,554 (61.5%) |
| post-change, seam unset | `7cf23377586f9ecb23e83d3fd579e984` | `c65e3683df4cabd571cad42b3a751dc8` | 34,554 (61.5%) |
| post-change, decay 32 | `7cf23377586f9ecb23e83d3fd579e984` | `c65e3683df4cabd571cad42b3a751dc8` | 35,149 (62.6%) |

Byte-identical across all three arms in both modes, as expected structurally:
the policy moves which slot holds an expert, never which experts run or in
what order. The default arm's hit count equals the pre-change binary's
exactly, so the unset seam did not move. The decay arm's +1.1 points matches
the simulator's +1.0 for that prompt, which is the proof it engaged. No
throughput is quoted: the host was running a build during the first arm, and
a one-request +1 point is below run-to-run spread anyway.

### Standing decision

`TURBOSPARK_EXPERT_CACHE_DECAY` is a seam, default off. Flipping the default
to decay 32 is supported by the hit-rate evidence on Gemma 4 and owes:

1. A multi-request replay on a real runner (server or `--chat`), measuring
   hits and tok/s with interleaved arms, since that is where the gain is.
2. One more family with a different expert count, ideally the 512-expert
   `qwen4exp` on an SSD-bound install, before assuming the interval carries
   over. The interval counts plans per layer, so it does not scale with
   layer count, but it may with experts per layer.

## Disk-bound baseline: qwen4exp on an external SSD (2026-10-14)

The measurement the prefetch and cache sections above said was OWED: an
install whose expert table does not fit in RAM, read from a device rather
than the page cache.

### Setup

- Install: `qwen4exp-swift-iq2-xs` (48 layers x 512 experts, top-10, expert
  stride 1,523,712 B), 64 GiB on a Samsung T7 at `/Volumes/T7`: 33 GiB of
  `packed_experts`, a 27 GiB `ngram_table`, 4.2 GiB `model_weights.bin`.
- Host: Apple M4 Max, 36 GB, AC, Spotlight disabled on the volume. Binary
  built from a clean export of `3c0ea885`. Greedy, 200 new tokens, one
  34-token prompt, 2,048 context.
- Link: the drive first negotiated USB 2.0 (480 Mb/s) on a charge-only
  cable and capped random expert reads at 47-61 MB/s (0.24 tok/s decode).
  With a 10 Gb/s cable, a raw `F_NOCACHE` random-read bench of whole
  experts read 576 MB/s at one thread and 778-786 MB/s at 16-32 threads;
  the engine's own pattern (8 threads, 840 KiB chunks) read 617 MB/s.
- Contamination: other sessions ran `cargo` builds throughout. Each arm
  records the 1-minute load and whether any `target/*/deps/` process was
  live at its end; only arms with neither are quoted for time below. Swap
  sat at 9-12 GB used for the whole capture, which is a property of this
  host that day and depresses every arm.

### Measurement

Clean arms only (two per slot count, forward and reverse order):

| slots | hit rate | requested MiB/tok | physical MiB/tok | expert io ms/tok | cb1 wait ms/tok | tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 47.3% | 349.8 | 88-92 | 149-154 | 119-125 | 3.56-3.70 |
| 32 | 58.3% | 276.7 | 91-103 | 153-169 | 124-127 | 3.40-3.59 |
| 128 | 83.1% | 111.9 | 92-93 | 150-152 | 138-150 | 3.42-3.47 |
| 16, `F_NOCACHE` | 47.3% | 349.8 | 349.8 | 585 | 196 | 1.29 |

Every 64-slot arm was contaminated; its hit rate (70.7%) is still valid,
since the routing is deterministic.

### Reading

- **The slot cache and the page cache are the same RAM counted twice.**
  From 16 to 128 slots the requested bytes fall 3.1x while the bytes read
  from the device stay at ~90 MiB/token and throughput stays flat. The page
  cache was already holding what the extra slots hold. This is the reason
  Flash-Next bypasses the page cache and sizes one explicit cache instead.
- **The read path runs at the device's rate.** The `F_NOCACHE` arm moved
  349.8 MiB/token in 585 ms, ~627 MB/s, the raw bench's figure for the same
  chunking. There is no software loss to recover inside the current
  pattern; the bench says whole-expert reads at 16 threads would be ~26%
  faster on this device.
- **A second, cache-independent bucket is as large as expert io.** The
  layer cb1 wait (attention plus router) is 120-150 ms/token at every slot
  count, ~40% of the step. The catalog row for this same install records
  12.2-12.8 tok/s from the internal SSD, ~80 ms for a WHOLE token, so this
  one bucket alone exceeds an internal-SSD token here. The likely cause is page faults on the mapped trunk and the 27 GiB n-gram
  table from the external device. This is INFERRED from the bucket's
  independence of slot count and has not been isolated; Flash-Next's
  128 MiB `--ple-shelf` targets the same table.

### Replay: what a larger explicit cache would buy

`scripts/pilot_ceiling.py --policies` over four routing traces from this
install (924 passes as one session, first request excluded). The decay and
LRU rows reproduce the engine's measured single-request hit rates within a
point:

| slots | pinned | cumulative LFU | LRU | decay 16 | decay 32 | MiB/tok at decay 16 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.1 GiB | 41.4% | 46.8% | 48.2% | 47.2% | 361.5 |
| 32 | 2.2 GiB | 49.5% | 59.6% | 59.9% | 58.5% | 279.8 |
| 64 | 4.5 GiB | 62.5% | 71.3% | 72.0% | 71.1% | 195.4 |
| 128 | 8.9 GiB | 78.7% | 84.2% | 84.1% | 84.0% | 111.1 |
| 192 | 13.4 GiB | 88.1% | 90.5% | 90.5% | 90.5% | 66.0 |
| 256 | 17.9 GiB | 93.6% | 94.4% | 94.4% | 94.4% | 39.2 |

- Cumulative LFU again loses across requests (6-10 points at 16-64 slots),
  the same finding as the Gemma 4 section above, on a 512-expert model.
- At 256 slots with `F_NOCACHE`, 39 MiB/token at the measured ~620 MB/s is
  ~65 ms of expert io against today's ~150 ms. That is a PROJECTION from
  hit rate and device rate, not a measurement, and it ignores that pinning
  17.9 GiB takes RAM the page cache currently gives to the trunk and the
  n-gram table, i.e. the cb1 bucket above may grow.

### What this changes

- `ALLOWED_CACHE_SLOTS` stopped at 128. The replay suggested the payoff
  was at 192-256 slots AND page-cache bypass together. **That projection
  was then measured and did not hold on this machine**; see the next
  section. 192 and 256 were added to the allowed set so the arms could be
  run at all.
- The cb1 bucket needs its own instrument before anyone optimizes it:
  split trunk faults from n-gram-table faults.
- Decay should default on once a clean multi-request tok/s pair exists;
  hit rates now support it on two families with different expert counts.

## Larger explicit cache vs page-cache bypass, measured (2026-10-15)

The Flash-Next shape (one explicit cache plus `F_NOCACHE`) on the same
install, device and host as the section above.

### Protocol

`turbospark-check --chat` reading four prompts from stdin in ONE process,
200 new tokens each, so the cache carries across requests. Turn 1 is the
warmup; the rows below are turns 2-4 from the cumulative phase counters.
Arms: 128 and 192 slots, each with and without `TURBOSPARK_EXPERT_NOCACHE=1`,
one forward and one reverse pass, plus a discarded warmup session. Each arm
waits for other sessions' `target/*/deps/` processes to finish, samples load
and those processes every 5 s while it runs, and records `vm_stat` swap
counts. 256 slots (17.4 GiB pinned) was not run: on this host's ~15 GiB of
free plus inactive memory it can only swap. Binary: `3c0ea885` plus the
192/256 rungs. Scripts and logs: `/Volumes/T7/turbospark/phase1/`.

### Measurement

| arm | hit rate | tok/s (fwd / rev) | expert io ms/pass | contamination |
|---|---:|---:|---:|---|
| 128, page cache | 78.9% | 4.46 / 4.96 | 85-105 | both runs overlapped other builds; fwd swapped 1.8 GiB |
| 128, `F_NOCACHE` | 78.9% | **2.91** / 2.97 | 168-198 | fwd clean; rev overlapped a build |
| 192, page cache | 88.4% | **4.28** / 5.30 | 73-78 | fwd clean (155 MiB swapped); rev swapped 3.1 GiB |
| 192, `F_NOCACHE` | 88.4% | 3.53 / 3.57 | 119-128 | fwd swapped 1.0 GiB; rev overlapped a build |

Output: byte-identical across the bypass arms at each slot count. 128 and
192 slots produce different text from each other, as
`crates/bench/src/real_model_open.rs` documents for any slot-count change.

### Reading

- **Page-cache bypass loses on a 36 GB machine, at both sizes**, by 25-35%,
  in every pair including the clean ones. The OS page cache is a free
  second tier holding experts the slots evicted; `F_NOCACHE` discards it.
  Flash-Next's setting assumes a machine where the explicit cache can be
  sized near the working set (28 GiB on 64 GB). It does not transfer here.
- **192 slots does not clearly beat 128 with the page cache on.** Hits rise
  9.5 points and expert io falls ~25%, but tok/s overlaps (4.3-5.3 against
  4.5-5.0). The extra 4.4 GiB pinned comes out of the page cache that also
  serves the trunk and the n-gram table, and two 192 runs swapped.
- **Under bypass, size does pay**: 192 beats 128 by ~21% (3.55 against
  2.94). That is the Flash-Next regime, and it still ends below the page
  cache arms.
- The replay's hit-rate projections were optimistic by 4-6 points on the
  real 4-turn session (78.9 / 88.4% against 84.1 / 90.5%): a chat session's
  turns share history, so this is the same workload shape but not the same
  traces.

### Standing decision

- Keep the page cache in the loop. Do not ship a production `F_NOCACHE`
  mode for this class of machine; it stays the measurement seam it was.
- Keep 192 and 256 in `ALLOWED_CACHE_SLOTS`. `Auto` reaches them only on
  machines with the memory, and a machine with 64 GB or more is exactly
  where the bypass result could reverse. That reversal is unmeasured.
- The bucket that did not move with any cache setting is cb1 (95-145
  ms/pass). It is now the larger lever on this install, and it needs the
  trunk-versus-n-gram split noted above before anyone optimizes it.
