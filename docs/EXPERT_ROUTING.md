# Expert Routing: Domain Concentration (Measured Negative)

The question this page answers: can a domain-restricted expert set --
profile which experts a CODING corpus routes to, then prune the cold ones
from the install or pin the hot ones in the cache -- save memory or win
throughput over the streaming expert cache that already ships?

Answer, measured 2026-08-08 on the real Gemma 4 26B-A4B install: **no**.
Routing under a coding workload is domain-TILTED but nowhere near
domain-CONCENTRATED, and both halves of the idea (pruned install, pinned
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
experts per layer, and this engine already exploits that dynamically --
routed experts are not resident, `crates/streaming` keeps 16 of
128 per layer in pinned slots (more where the machine has memory to spare;
16 is the floor and what this measurement used) and streams misses by
`pread`. The open
question was whether a STATIC, domain-specific expert set could beat that
dynamic mechanism. That is an empirical question about routing statistics,
so it was measured before anything was built.

## Instrument

`MFERENCE_ROUTER_HIST=/path.json` on `RealForwardRunner`
(`crates/runtime/src/router_hist.rs`): counts, per layer, how often each
expert appears in the router's top-k, over every forward pass (prefill and
decode alike, the `MFERENCE_PHASES` divisor convention). Host-side count
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
  on BOTH corpora. There is no small "coding region" to keep.
- **Tilted, though.** Jaccard 0.455 between two ~67-expert sets is real
  divergence -- coding routes DIFFERENTLY, just not NARROWLY. The union of
  the two hot sets is ~92 of 128 experts.
- **A pinned coding set loses to the cache.** A static 95%-mass set is ~67
  slots per layer; the LFU cache reads ~84% hit rate from 32 slots by
  exploiting temporal locality within a generation, which domain
  statistics cannot see.
- **A pruned install loses worse.** The 5% tail is whole experts of routed
  mass. The quality gate's sensitivity curve reads +10.5% perplexity from
  flipping one quantization level in 0.0122% of expert BYTES
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
