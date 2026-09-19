# Where a decoded token's time actually goes

Measured 2026-08-16 on this machine (Apple M4 Max, AC, quiet), real Gemma 4
26B-A4B install, release, greedy, short prompt and 600 generated tokens so the
divisor is decode (AGENTS.md Gotcha 21). Two unprofiled runs agreed within 1%
on every bucket.

If you are about to propose work on the command-buffer scheduling gap or on
the decode loop's host/GPU overlap, read this first. It refutes the "~5
ms/token scheduling gap" figure that had been circulating, and it relocates
the lever.

## The sweep

`--expert-cache-slots` is the one knob that moves the expert `pread` without
touching anything else, which makes it the instrument rather than the
subject. `gpu busy` is `cb1 + routed + final` from `TURBOSPARK_PHASES=1`'s
`GPUStartTime`/`GPUEndTime` attribution; `+shared` adds the shared-expert
command buffer at 1.73 ms/token.

| slots | hit rate | ms/token | expert io | gpu busy | + shared | GPU idle |
|---|---|---|---|---|---|---|
| 8 | 42.4% | 25.07 | 7.62 | 10.26 | 11.99 | 13.08 (52%) |
| 16 | 64.2% | 22.61 | 5.54 | 10.24 | 11.97 | 10.64 (47%) |
| 32 | 82.7% | 19.54 | 3.13 | 9.99 | 11.72 | 7.82 (40%) |

## Three findings

**1. GPU busy time is constant; the expert `pread` is the whole variable.**
10.26 / 10.24 / 9.99 ms/token across a sweep that changes throughput by 28%.
The GPU does the same work per token at every slot count, as it must -- the
same experts are dispatched either way, only the host's cost of *fetching*
them changes.

**2. Wall clock tracks the `pread` slightly better than 1:1.** 8 -> 16 slots
removes 2.08 ms/token of `expert io` and 2.46 ms/token of wall; 16 -> 32
removes 2.41 and 3.07 (ratios 1.18 and 1.27, the excess coming from the
smaller bind and marginally lower GPU busy that a higher hit rate also
buys). **So the exposed `pread` sits on the critical path essentially
whole.** Every millisecond taken out of it is a millisecond off the token.
The shared-expert command buffer already covers ~1.73 ms/token of it by
design (`TURBOSPARK_SHARED_CB`); the remainder is not covered by anything.

**3. The command-buffer scheduling gap is ~1.1 ms/token, not ~5.** At 32
slots the `gpu wait (layer cb1)` bucket reads 11.74 ms/token while the three
buffers executing inside that window -- `cb1` 5.19, `routed` 3.73, shared
1.73 -- total 10.65. The bucket is ~91% real device time. The earlier ~5 ms
figure came from comparing the wait bucket against `cb1` alone, which omits
the routed and shared buffers that are queued on the same queue and must
drain before the wait returns.

## What that means for the ranked work

At 32 slots, against a 19.54 ms token:

| | ms/token | share |
|---|---|---|
| exposed expert `pread` | 3.13 | 16.0% |
| encode + logit readback | 1.48 | 7.6% |
| host sampler (`select`, outside `produce`) | ~1.2 | 6.1% |
| command-buffer scheduling gap | 1.09 | 5.6% |
| routed bind + upload | 0.74 | 3.8% |

An `MTLSharedEvent` passive-wait rewrite, the named tool for the
scheduling gap, is therefore chasing at most 5.6%, and only if the gap
goes to zero, which it will not. That is a real but ordinary optimization,
not the largest one available, and it should be scoped against 1.09 rather
than against 5.

**And prefill, not decode, is where the `pread` is worth a phase.** A
prefill chunk of M tokens reads the UNION of their routes rather than the
sum, which no decode-side change can imitate: see
`docs/BATCHED_PREFILL.md`, where the same bucket is 25.2% and the union
cuts it 3.3x at M=16.

The `pread` is the item that is worth a phase, and its cheapest lever needs
no kernel work at all: slot count. 32 slots buys +28% decode over 8 and
+15.7% over 16, for pinned host memory of `slots x layers x expert_stride`
(AGENTS.md Gotcha 36) -- which is exactly the "adaptive expert-cache slots"
item, and this is the measurement that prices it. Covering more of the
exposed `pread` with GPU work is the other direction, and the
shared-expert overlap is the existing proof that the shape works.

**LANDED 2026-08-16.** `--expert-cache-slots` defaults to `auto` now
(`crates/runtime/src/expert_cache_policy.rs`): the largest allowed count
whose slot cache fits a quarter of `physical - resident - 4 GiB`, floored at
16 so it can only ever climb. On the machine this page was measured on it
resolves to 32. Every harness keeps its pinned 16 through a separate entry
point, so the rows above and every row in `docs/BENCHMARKS.md` still
describe 16 slots and are still reproducible with
`--expert-cache-slots 16`.

## Two more dead ends, measured 2026-08-16

Both were "measure before believing" items left open by the session above.
Both came back null, and both are recorded here so nobody re-derives them.

**The routed command buffer's retire does not hide a `pread` overlap.**
`families/gemma4/mod.rs` retires layer N-1's routed CB before encoding
layer N's routed MoE, which contains the expert `pread` -- so moving the
retire past the `pread` looks like free overlap. The bucket that bounds it
is `routed cb retire`, and it reads **0.26 and 0.28 ms/token** over two
runs at 32 slots (1.4% and 1.3% of the token), against 0.23 in the
2026-08-06 prefill attribution. Layer N-1's routed work has essentially
completed by the time cb1's wait returns, which is what
`TURBOSPARK_ROUTED_PIPELINE` already claims and this confirms. The ceiling is
1.4% and the achievable part is less.

Worth recording alongside it, because it bounds any future attempt: the
retire can move past the `pread` but **not** past the bind. Everything from
`t_bind` onward in `families/gemma4/moe.rs` writes `scratch.routing_w`,
`scratch.moe_acts` and the routed argument buffer, all of which live in
`DecodeScratch` and are shared across layers, so a host write there while
layer N-1's CB is still reading them is a data race producing fluent wrong
text. Covering the bind too needs those double-buffered, which is a
different and larger change.

**GDN function constants 90-94: superseded 2026-08-16, and the answer
inverted when the install that reaches them came back.** The paragraph
that stood here concluded specialization was not worth measuring, from
the one GDN install then on disk (`ternary27b`, whose 2-bit branch never
dispatches the fused kernel). With `qwen38-27b` re-streamed, the fused
`gdn_in_proj_gemv_simd` is 16.5% of the token and the plain INT4 GEMV
another 72.3%, and baking M/N in is worth having; see "The dense 27B"
below. The general lesson survives the reversal: the 2%-of-the-token
reading was correct for the install that produced it, and a share
measured on one family's branch does not transfer to the other branch of
the same flow.

## The dense 27B (qwen38): profile, the FC win, and two closures

Measured 2026-08-16 on this machine (M4 Max 36 GB, AC, not quiet; the
session's desktop load was present throughout, so every absolute tok/s
here is qualified by Gotcha 43; the deltas are interleaved pairs and the
shares are within-run, which that load does not contaminate). Install:
the catalog-pulled `qwen38-27b.gturbo`, which reproduced the frozen
quality row (perplexity 4.9432, both digests) and the memory oracle (660
MiB of 750, replay +0.00) before anything was measured on it.

**The family's first dispatch profile, and it accounts for the token.**
One command buffer per token, ~978 dispatches, host encode 0.65 ms/token
and sampler+detok ~1.0 against a ~49.5 ms GPU wait: the token is the GPU.
Within the sampled buffer: `dequant_int4_gemv_simd` 72.3%,
`gdn_in_proj_gemv_simd` 16.5% (the same GEMV body, fused), norms and
elementwise ~5.6%, attention plus GDN state ~4.4%. A dense token reads
~14.4 GB of weights (FFN 9.6 + GDN in_proj 2.3 + out_proj 0.9 +
attention 0.9 + head 0.7 -- the 12.3 GB figure that circulated undercounts
by omitting out_proj, attention and the scale/bias planes), which at the
observed ~50 ms/token is ~290 GB/s effective against the kernel's ~375
GB/s saturation. The GEMV itself is not occupancy-bound at any decode
shape: `gemv_bandwidth_bench.rs::int4_gemv_headroom_at_qwen38_shapes`
reads 0.90-1.11x of the same kernel's large-shape reference on every row.
The residue is the ~11% non-GEMV work plus serial-encoder gaps. There is
no large hidden lever; there was one small one:

**Baking M/N as function constants: +3.0-5.2% isolated, +4.5-8.4% end to
end, output byte-identical.** `specialized_constants` in
`dequant_int4_gemv.rs` and `in_proj_pipeline` in `gdn.rs` bake each
dispatch's shape and key the pipeline cache on it. Cold-pool isolated
deltas per shape sit at +3.0-5.2%; three interleaved end-to-end pairs
read +4.5/+8.4/+7.7% (15.2 -> 16.3 tok/s under load). Every gate held
without motion: greedy and sampled stdout byte-identical across the
pre/post binaries on both families (qwen38 and gemma4 -- the fast-math
reassociation worry did not materialize), both quality gates exact to
the last hex character, both memory oracles green. The probe's first run
is a standing caution, recorded in the bench: a shared pipeline-cache key
across shapes reused the first-compiled pipeline for every later shape,
and the wrong-N kernel read a third of each row and printed +322%
(`crates/gpu` Gotcha 1 -- the key must carry the baked values).

**Mid-token command-buffer split: closed without building it.** Its whole
ceiling is the 0.65 ms/token of host encode that a split could overlap,
~1.3% of the token. Under the plan's own 1 ms threshold; skipped.

**Layer streaming: closed by arithmetic, recorded so it is not proposed
again.** Expert streaming works because a token touches ~8 of 128 experts
and temporal locality gives the slot cache ~84% hits. A dense token
touches 100% of the weights once each, so a layer cache smaller than the
model has a structural 0% hit rate (each layer is evicted before its
next use -- and "streaming layers" degenerates to re-reading ~14 GB from
SSD every token: sub-1 tok/s against the current ~19. There is no cache
policy that fixes a working set equal to the model.

**The weight mapping is wired while the model runs, and that settles the
small-machine question.** Measured with `vm_stat` across process exit:
system wired read 17.93 GB while decoding and 3.27 GB the moment the
process exited -- a 14.7 GB delta that is the resident region plus the
Metal buffers. Under `memory_pressure -S -l critical` (free pages driven
to ~140 MB) the process survived and throughput did not move, because
wired pages cannot be evicted: the OS squeezes everything else. The two
claims that looked contradictory are both true: `phys_footprint` does not
count the mapping (the process ledger shows it as 1.4 MB of clean
"mapped file"; AGENTS.md Gotcha 40), and Metal's `newBufferWithBytesNoCopy`
wires it (`crates/model-io` AGENTS.md Gotcha 1). "Not counted" never
meant "reclaimable". Consequence: budget a dense install's full disk size
in physical RAM -- a 16 GB machine is hard-blocked from this model, not
gracefully degraded, and the catalog row's "budget its size on disk in
free RAM" note is the correct guidance. (Whether open() fails cleanly or
thrashes on a too-small machine is unmeasurable from this 36 GB one.)

## Caveats

The shared-expert buffer's 1.73 ms/token is the one number here NOT read
off an unprofiled run: it is dropped unwaited by design, so only
`TURBOSPARK_DISPATCH_PROFILE=1` resolves it, and that mode inflates (it read
1.990, deflated here by the 5.916/5.14 ratio its own `cb1` shows against
the unprofiled `cb1`). **The conclusion is insensitive to that
correction**: taking 1.990 verbatim makes the scheduling gap 0.83 ms/token
instead of 1.09, and taking 0 makes it 2.82 -- still nowhere near 5.

Every number is this machine's. The ratios transfer; the absolutes do not
(CLAUDE.local.md's standing rule). Reproduce with:

```sh
TURBOSPARK_PHASES=1 ./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/phase.json --max-new 600 --seed 1 \
  --temperature 0.0001 --top-k 1 --expert-cache-slots 32
```

where `/tmp/phase.json` is a short single-turn prompt. Discard a warmup run
(Gotcha 20) and check the machine is quiet first (Gotcha 43): with the
desktop app busy, the same sweep read a 2.78x constrained-slot ratio where
a quiet machine reads 0.87x.
