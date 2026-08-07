# Benchmarks: this port against the Swift original

One machine, one model install, one session, both engines. This is the
first parity number in this repo; every other figure here and in
`docs/BENCHMARKING.md` is this port measured against its own past self.

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
| This port | `98e9cf2` plus the `--case` flag this run added |
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
| short-explanation | 61 | 39.213 / 40.160 | 25.594 / 25.535 | 0.64 |
| medium-review | 430 | 38.182 / 38.447 | 24.635 / 24.630 | 0.64 |
| long-synthesis | 3,015 | 34.485 / 34.396 | 23.150 / 23.125 | 0.67 |

**This port decodes at 64 to 67 percent of Swift on the same hardware and
the same install.** The ratio is flat across a 50x span of prompt length,
so the gap is per-token decode work, not a context-scaling problem. The
two runs within each arm agree to 0.06 tok/s or better, far tighter than
the ~1.5 tok/s run-to-run spread this port shows across sessions, so the
gap is not measurement noise.

Generated token counts differ between engines (516/780/617 for Swift
against 510/691/565 here) because the two samplers walk different RNG
streams. Both stop at `endOfTurn` on coherent text, and tok/s is a rate,
so this does not bias the comparison. It does mean the routing workload is
not token-for-token identical; Swift generated MORE tokens per case and
was still faster.

## Prefill

Not the same algorithm on both sides, so this is a scope difference, not a
regression. Swift chunks prefill (`--prefill-chunk`, default 128); this
port runs one sequential forward pass per prompt token because the tile
kernels are descoped (`DEVIATIONS.md`).

| Case | Prompt tok | Swift prefill | This port prefill |
| --- | ---: | ---: | ---: |
| short-explanation | 61 | 5.59 / 5.67 s | 1.27 / 1.26 s |
| medium-review | 430 | 7.43 / 7.37 s | 8.04 / 8.15 s |
| long-synthesis | 3,015 | 27.62 / 27.72 s | 63.83 / 64.49 s |

Fitting the two endpoints:

- Swift: about 5.1 s fixed plus 7.5 ms per prompt token.
- This port: no measurable fixed cost, 21.2 ms per prompt token.

That per-token figure independently reproduces the 21 ms this port
measured for itself on 2026-08-06 (CLAUDE.local.md's prefill attribution
table), from a completely different measurement path.

The crossover is near 350 prompt tokens. Below it this port is faster to
first token, because Swift pays a fixed startup this port does not; above
it Swift pulls ahead and keeps going, because a chunk of 128 amortizes
weight reads across 128 tokens where this port re-reads per token.

## Memory

Peak `phys_footprint` is the headline counter: it is what
`../Mference/docs/BENCHMARKS.md` reports, what the Swift README's "26B
total, ~3.88B active per token, in ~2 GB of memory" claim rests on, and
what this port's `AppMemorySampler` samples. `/usr/bin/time -l` reports it
for any process as `peak memory footprint`, so it is available for BOTH
engines even though the Swift CLI prints no memory line of its own.

| Case | Swift footprint | This port footprint | Delta |
| --- | ---: | ---: | ---: |
| short-explanation | 2,235 / 2,219 MiB | 2,189 / 2,168 MiB | -46 / -51 MiB |
| medium-review | 2,235 / 2,216 MiB | 2,195 / 2,191 MiB | -40 / -25 MiB |
| long-synthesis | 2,235 / 2,235 MiB | 2,167 / 2,187 MiB | -68 / -48 MiB |

**This port holds the ~2 GB working set, and does it in about 1 to 3
percent less peak footprint than Swift on the same machine and install.**
Both engines land in the 2.1 to 2.2 GiB band on a 26B model with a 14 GB
install, which is the property the design exists to deliver.

The harness asymmetry works AGAINST this port here, so the delta is if
anything understated: one `mference-bench --case` launch runs the
protocol's discarded warmup AND the measured run in the same process, so
its figure is a peak over two generations, while each Swift figure covers
one. Swift's number is also notably flat at 2,235 MiB across five of six
runs, which reads like a ceiling its allocator reaches and holds rather
than a workload-driven peak.

Two secondary observations:

- **The in-process sampler is validated.** `AppMemorySampler`'s reported
  session peak matched the kernel's own high-water mark from
  `/usr/bin/time -l` to 0.1 MiB on all six runs. Sampling every 8th token
  is not missing a transient peak on this workload.
- **RSS goes the other way and is the less useful counter.** Peak RSS was
  1,576 to 1,830 MiB for Swift against 1,977 to 2,004 MiB here. RSS counts
  resident pages including clean file-backed ones, so it moves with how
  much of the 14 GB mapped install each engine happens to be touching;
  footprint is the counter that tracks what the process actually costs the
  system. Reported for completeness, not as a gap.

Published Swift rows for other hardware, for context: 2,126 to 2,142 MiB
on a 24 GB M5 Pro, 1,776 to 1,971 MiB on an 8 GB M2. Swift reads slightly
higher here (2,216 to 2,235) than its own published M5 Pro band, on a
different chip and OS build, so do not treat the M4 Max numbers above as
transferable to those rows.

## What this does not change

`crates/bench/tests/memory_oracle.rs` keeps its `Apple M4 Max` row
self-measured: ceiling 2,300 MiB, floor 15.0 tok/s, source "this port,
measured locally". The floor stays a regression guard against this port's
own past behaviour. Turning it into a Swift parity gate would make the
oracle fail by design until the 0.64 gap closes, which is a separate
decision from measuring the gap.

## Caveats worth repeating

- Two measured runs per arm. Enough to show a 1.5x gap; not enough to
  claim a 2 percent one.
- One machine, one chip, one session, on AC. Absolute numbers here have
  repeatedly failed to transfer across sessions in this repo; the ratio is
  what to carry forward.
- Both engines ran alone (`scripts/parity.sh` refuses to start if another
  model process is up), with no profiler or trace mode active.
