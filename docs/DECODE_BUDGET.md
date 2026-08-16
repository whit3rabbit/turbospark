# Where a decoded token's time actually goes

Measured 2026-08-16 on this machine (Apple M4 Max, AC, quiet), real Gemma 4
26B-A4B install, release, greedy, short prompt and 600 generated tokens so the
divisor is decode (AGENTS.md Gotcha 21). Two unprofiled runs agreed within 1%
on every bucket.

Read this before proposing work on the command-buffer scheduling gap or on
the decode loop's host/GPU overlap. It refutes the "~5 ms/token scheduling
gap" figure that had been circulating, and it relocates the lever.

## The sweep

`--expert-cache-slots` is the one knob that moves the expert `pread` without
touching anything else, which makes it the instrument rather than the
subject. `gpu busy` is `cb1 + routed + final` from `MFERENCE_PHASES=1`'s
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
design (`MFERENCE_SHARED_CB`); the remainder is not covered by anything.

**3. THE COMMAND-BUFFER SCHEDULING GAP IS ~1.1 ms/token, NOT ~5.** At 32
slots the `gpu wait (layer cb1)` bucket reads 11.74 ms/token while the three
buffers executing inside that window -- `cb1` 5.19, `routed` 3.73, shared
1.73 -- total 10.65. The bucket is ~91% real device time. The earlier ~5 ms
figure came from comparing the wait bucket against `cb1` ALONE, which omits
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

An `MTLSharedEvent` passive-wait rewrite -- the named tool for the
scheduling gap -- is therefore chasing at most 5.6%, and only if the gap
goes to zero, which it will not. That is a real but ordinary optimization,
not the largest one available, and it should be scoped against 1.09 rather
than against 5.

The `pread` is the item that is worth a phase, and its cheapest lever needs
no code at all: slot count. 32 slots buys +28% decode over 8 and +15.7% over
the default 16, for pinned host memory of `slots x layers x expert_stride`
(AGENTS.md Gotcha 36) -- which is exactly the "adaptive expert-cache slots"
item, and this is the measurement that prices it. Covering more of the
exposed `pread` with GPU work is the other direction, and the
shared-expert overlap is the existing proof that the shape works.

## Caveats

The shared-expert buffer's 1.73 ms/token is the one number here NOT read
off an unprofiled run: it is dropped unwaited by design, so only
`MFERENCE_DISPATCH_PROFILE=1` resolves it, and that mode inflates (it read
1.990, deflated here by the 5.916/5.14 ratio its own `cb1` shows against
the unprofiled `cb1`). **The conclusion is insensitive to that
correction**: taking 1.990 verbatim makes the scheduling gap 0.83 ms/token
instead of 1.09, and taking 0 makes it 2.82 -- still nowhere near 5.

Every number is this machine's. The RATIOS transfer; the absolutes do not
(CLAUDE.local.md's standing rule). Reproduce with:

```sh
MFERENCE_PHASES=1 ./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/phase.json --max-new 600 --seed 1 \
  --temperature 0.0001 --top-k 1 --expert-cache-slots 32
```

where `/tmp/phase.json` is a short single-turn prompt. Discard a warmup run
(Gotcha 20) and check the machine is quiet first (Gotcha 43): with the
desktop app busy, the same sweep read a 2.78x constrained-slot ratio where
a quiet machine reads 0.87x.
