# Benchmarking

How to measure this port's throughput and memory, and how the numbers
compare to the Swift original (`../Mference`, public at
<https://github.com/drumih/turbo-fieldfare>; `docs/BENCHMARKS.md` there).

For the measured head-to-head against Swift on one machine, see
[`BENCHMARKS.md`](BENCHMARKS.md) (2026-08-07: decode at parity within 1
percent on an M4 Max, same install; peak footprint 2 to 5 percent lower;
prefill the one remaining gap, and a known scope difference).
`scripts/parity.sh` reproduces it, and `scripts/phasediff.sh` diffs the
two engines' decode phase splits bucket by bucket. Everything else in this
file is this port measuring itself.

Everything here lives in `crates/bench`: the `turbospark-bench` binary, a
small library the binary and the oracle test share, and
`tests/memory_oracle.rs`.

## The three modes

| Mode | Command | What it measures |
| --- | --- | --- |
| Scripted (default) | `turbospark-bench <tokenizer-dir>` | This port's prefill+decode *loop* overhead. No model. |
| Synthetic real | `turbospark-bench <tokenizer-dir> --real` | The real GPU dispatch path over a tiny synthetic install. |
| Real install | `turbospark-bench --model <install-dir> [--case <id>]` | Real Gemma 4 throughput and peak memory. The Swift-comparison number. |

Only the third mode is comparable to anything published. The first two
exist so the loop and the dispatch path can be timed without a
multi-gigabyte checkout.

### Scripted

```bash
cargo run -p turbospark-bench --bin turbospark-bench -- crates/tokenizer/tests/fixtures/ChatMLTokenizer
```

Three fixed prompts, fixed seed 42, one discarded warmup per prompt, all
driven through the real `run_raw_completion` loop against a
`ScriptedLogitProducer` (a fixed replayed logit sequence). The tok/s
printed is tokenizer + sampler + detokenizer + stop-matcher cost, not
inference. Portable; runs on Linux.

### Synthetic real (macOS)

```bash
cargo run -p turbospark-bench --bin turbospark-bench -- <tokenizer-dir> --real
```

Builds a tiny dense `.gturbo` install (deterministic INT4 weights, vocab
sized to the tokenizer) in a temp dir and drives the same three prompts
through `RealForwardRunner`: real Metal kernels, real KV cache, real
zero-copy resident weights. The model is far too small for the number to
mean anything as throughput; it measures the dispatch path.

### Real install (macOS)

```bash
cargo run --release -p turbospark-bench --bin turbospark-bench -- --model ~/models/gemma4.gturbo
```

This is the frozen community protocol, vendored from the Swift repo so
both engines run the identical workload:

- Prompts: `crates/bench/prompts/real-generation-v1/*.txt`, the byte-exact
  `content` strings from Swift's
  `docs/benchmark-prompts/real-generation-v1/*.json` (source SHA-256s are
  recorded in `crates/bench/src/protocol.rs`).
- Seeds: `short-explanation` 20260721, `medium-review` 20260722,
  `long-synthesis` 20260723.
- Sampling: temperature 0.2, top-k 64, top-p 0.95, repetition penalty 1.0.
- Budget and context window: **per family**, resolved from the install's own
  `manifest.json` by `real_model::protocol_parameters` and printed in the
  header. 4,096 context / 1,024 new tokens for `gemma4`, `qwen36` and
  `qwen3moe`; 8,192 / 1,024 for the dense `llama` half; 8,192 / 3,072 for
  `gpt-oss`; 8,192 / 2,048 for `muse_glimmer`. Neither is a knob, and both are the same numbers the memory
  oracles freeze their rows at (the oracle targets assert the agreement at
  compile time). Why they differ is the point: the protocol freezes the
  prose, so its token count belongs to the checkpoint's tokenizer
  (`long-synthesis` is 3,444 tokens under Mistral's 32k vocab and does not
  fit 4,096 at all), and how many tokens a model spends answering is the
  model's property (Harmony puts gpt-oss's reasoning in an `analysis`
  channel before its answer, so two of three cases stop on `maxTokens` at
  1,024). Raising the shared constants was not an option in either
  direction: KV is sized at open, so a wider window moves every already
  frozen peak, and a larger budget lets every other family generate further.
- **Read a peak or a tok/s row with the window and the budget.** The same
  Mistral install reads 684 MiB at 4,096 and ~1,200 MiB at 8,192, because on
  a dense install KV is nearly all of the counted footprint. Any dense
  `llama` bench number taken before 2026-08-12 is at the old shared 4,096.
- Each case runs one discarded warmup, then one measured run.
- Prompts are chat-templated exactly as the CLI templates them. The IT
  checkpoint needs its turn markup; a raw prompt babbles.

Use `--release`. A debug build's numbers are meaningless.

Output per case, on stdout:

```
turbospark-bench: real install ~/models/gemma4.gturbo on Apple M4 Max, frozen protocol real-generation-v1
  family=gemma4 context=4096 max_new=1024 expert_cache_slots=16
case               prompt_tok  prefill_s  new_tok  decode_s    tok_s  peak_mib
short-explanation          61       1.55     1024     43.14   23.736    2710.8
```

and, on stderr, the Swift CLI's own footer for `grep -h '^\[stop='`
parity with the Swift protocol:

```
[stop=maxTokens prefill=61tok/1.55s new=1024tok decode=43.14s tok/s=23.736]
```

`prefill_s` and `decode_s` come from `RawDecodeResult`'s separate fields,
not wall clock. `tok_s` is decode-only (`new_tokens / decode_seconds`),
the same definition Swift's footer uses.

`--case <id>` restricts the run to one protocol case. That is the
protocol's fresh-process leg: Swift's CLI launches once per case, so any
cross-engine comparison has to match that shape. Without it all three
cases share one process and one sampler, which is what the memory oracle
wants (its steady-state guard needs the same runner).

## Comparing against Swift

```bash
cargo build --release -p turbospark-bench
scripts/parity.sh [pairs]        # default 2 measured pairs per case
```

Runs the protocol through `../Mference/.build/release/MferenceCLI` and
this port's `turbospark-bench --case`, against the same install directory,
one fresh process per run, arms interleaved pair by pair. Discards a
warmup per engine per case, rejects any run that does not stop
`endOfTurn`, refuses to start if another model process is up, and records
chip, macOS, power source, and both git revisions at the top of its
output. By default, results land in a new private directory under
`${TMPDIR:-/tmp}` whose path is printed at the end. Set `OUT` to reuse a
specific caller-managed directory.

Memory comes from `/usr/bin/time -l` around each launch. Its `peak memory
footprint` line IS `phys_footprint`, the counter both engines' published
numbers use, and the kernel reports it for any process, so it covers the
Swift side too even though the Swift CLI prints no memory line. Peak RSS
is captured alongside it as a secondary column.

That also gives a free check on `crates/bench/src/memory.rs`: this port's
in-process `AppMemorySampler` peak matched the kernel's high-water mark to
0.1 MiB on all six runs of the 2026-08-07 session.

Published results: [`BENCHMARKS.md`](BENCHMARKS.md).

## How memory is measured

`crates/bench/src/memory.rs` is a port of the Swift app's
`AppMemorySampler`: `task_info(mach_task_self_, TASK_VM_INFO)` read for
`phys_footprint`, in bytes, with peak tracking. Sampled before prefill, on
every 8th decoded token, and once after the run, the same cadence the
published Swift rows used.

Two things this deliberately is not:

- Not `resident_size`. Swift's docs report both "Peak RSS / footprint";
  only the footprint column is what either engine's code measures.
- Not a Metal counter. Neither engine reads
  `MTLDevice.currentAllocatedSize`. On unified memory `phys_footprint`
  already covers wired Metal allocations, while mmap'd weight pages are
  file-backed and only partly counted, which is why a healthy footprint
  (~2 GB) sits far below the 14 GB install.

Sampling is in-process, so it covers one leg of the protocol. "Fresh
processes" is left to the caller (a shell loop around the binary).

**Every number on this page is at 16 expert-cache slots, and that is pinned
rather than defaulted into.** `turbospark-bench`, both memory oracles, every
quality gate and `logit_dump` pass `protocol::PROTOCOL_EXPERT_CACHE_SLOTS`
explicitly, so none of them can pick up the `--expert-cache-slots auto`
default that `turbospark-check` and `turbospark-server` now carry. That is
deliberate and is the same rule AGENTS.md Gotcha 35 states for the power
profile: a harness that measures a knob is exactly the caller that must not
sense it, because a frozen footprint row taken at whatever the machine felt
like that morning is not a row. The consequence for a reader: a figure here
is not what the CLI will print on a machine with memory to spare, which
climbs to 24 or 32 slots and trades roughly 1.5 GB of peak for 16% of decode
(`docs/DECODE_BUDGET.md` has the sweep). Reproduce a row with
`--expert-cache-slots 16`.

## The memory oracle

```bash
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture
```

`crates/bench/tests/memory_oracle.rs` runs the same protocol in-process
and turns it into assertions. `#[ignore]`d (it needs a real ~14.6 GB
install and takes several minutes); skips with a printed note if
`TURBOSPARK_GEMMA4_INSTALL_DIR` is unset.

What it asserts, in order:

1. **Stop reason.** Every measured case must stop with `endOfTurn`. A run
   that dies on `maxTokens` is not comparable to the published rows, so
   this gates everything after it.
2. **Peak footprint** at or under the Swift ceiling. Asserted always;
   memory sizing does not depend on the chip.
3. **Decode tok/s** at or above the Swift floor, per case, only when the
   chip brand matches a baseline row.

Most rows come from Swift's `docs/BENCHMARKS.md` Gemma 4 table. Chip
detection is `sysctl machdep.cpu.brand_string`, matched by substring,
most specific row first (so `Apple M4 Max` must precede any future bare
`Apple M4`).

| Chip | Source peak footprint | Ceiling used | Source decode | Floor used | Source |
| --- | --- | --- | --- | --- | --- |
| Apple M5 Pro (24 GB) | 2,126-2,142 MiB | 2,250 MiB | 31.01-35.17 tok/s | 31.0 | Swift docs |
| Apple M4 Max (36 GB) | 2,100-2,197 MiB | 2,300 MiB | 23.07-25.06 tok/s | 15.0 | THIS PORT |
| Apple M2 (8 GB) | 1,776-1,971 MiB | 2,070 MiB | 5.10-6.30 tok/s | 5.1 | Swift docs |
| Anything else | - | 2,250 MiB | - | reported, not asserted | - |

**The M4 Max row is not a parity claim.** Swift publishes no M4 Max row
and has not been run on that machine, so both numbers are this port
measuring itself and the row only catches a regression against its own
past behaviour. The `ChipBaseline::source` field carries this, and it is
printed on every run and quoted in the failure message, so a red build
says which kind of number it broke. Replace both with Swift's if that run
ever happens; a real parity number would very likely be tighter.

Its ceiling (2,300 MiB) lands above the generic 2,250 default rather than
below it, because the measured peak spread on that machine is 77 MiB of
expert-slot warming and a 2,250 ceiling would flake on the spread alone.
Its floor sits about 25 percent under the slowest measured case, covering
the run-to-run spread this port shows (wider than the ~1 percent Swift
reports, so a Swift-style verbatim-minimum floor would be too tight here).
The floor was 10.0 until 2026-08-06, when wiring split-KV took the slowest
case from 11.6 to about 20 tok/s and left the old floor unable to catch
losing the whole change. A floor set generously below a number that later
doubles stops being a gate; re-check it whenever a change moves the
slowest case.

Swift-sourced ceilings are the documented peak plus about 5 percent. That
is Swift's own cross-run variance (its repeat table spans 1,388-1,464 MiB
on identical runs); more headroom than that would hide a regression the
size of a single KV layer. Their throughput floors take the documented
minimum verbatim.

A tok/s failure on a Swift-sourced row means the port decodes slower than
Swift on that hardware. It is a finding, not a broken test.

## Static KV accounting

`crates/gpu/tests/kv_cache.rs` has a cheap deterministic companion to the
oracle: for the real Gemma 4 shape at 4K it asserts the fp16 SWA ring caps
the 25 sliding-window layers at 1,152 rows (`sliding_window` 1024 plus a
128-token prefill chunk, the Swift sizing) while the 5 full-attention
layers stay linear at 4,096, for 319,815,680 bytes of KV total. Linear
everywhere would be 922,746,880 bytes. This runs in the normal suite, so a
regression in KV sizing fails in milliseconds instead of waiting on a
multi-minute oracle run.

## Current measured state (2026-08-06)

Apple M4 Max 36 GB, real `gemma4.gturbo` install. All cases stop
`endOfTurn`; the oracle passes. The `after` columns are two runs taken
once split-KV was wired (`chunks_for` in
`crates/gpu/src/attention_decode.rs`); the `before` columns are the three
runs at merge a772b67 that the baseline row was originally derived from.

| case | prompt tok | before (3 runs) | + split-KV (2 runs) | + read pool (1 run) |
| --- | ---: | ---: | ---: | ---: |
| short-explanation | 61 | 20.35 / 20.27 / 20.40 | 22.73 / 23.52 | 25.06 |
| medium-review | 430 | 15.97 / 15.77 / 15.78 | 21.13 / 19.98 | 23.69 |
| long-synthesis | 3,015 | 11.71 / 11.60 / 11.62 | 20.71 / 21.48 | 23.07 |

The last column adds the expert-read chunking and persistent read pool
(`crates/streaming/src/read_pool.rs`): the routed-expert `pread` is a
page-cache memcpy, and reading one miss on one thread left roughly half
the achievable bandwidth unused. See `DEVIATIONS.md`'s MoE entry.

The gain scales with context because attention was the only per-token cost
growing with it, and at one chunk the decode attention kernel ran on
`num_q_heads` (16) threadgroups regardless of how much KV it had to read.
See `DEVIATIONS.md`'s split-KV entry.

Peak footprint 2,120 / 2,197 / 2,197 MiB before and 2,126 / 2,125 after,
around the band Swift publishes for the M5 Pro (2,126-2,142 MiB) despite
this being a different chip. Split-KV costs 512 KiB of extra attention
scratch, which is inside the measurement noise.

Two problems this document previously tracked as open are closed, and
neither was what the numbers suggested:

1. **Word salad after ~40 tokens, no case reaching `endOfTurn`.** The
   output head returned probabilities where the host sampler expects
   logits, so `selection::select` softmaxed an already-normalized vector.
   Over V=262144 that flattens the distribution to near-uniform, and
   because softmax is monotone the ranking survived, so greedy decoding
   stayed byte-identical to correct and only sampling showed it. The head
   now applies the logit softcap alone. See `AGENTS.md` Gotcha 16.
2. **Peak footprint 5,332 MiB against a 2,250 ceiling, with ~2 GiB
   unexplained by static accounting.** Command buffers and encoders are
   autoreleased Objective-C objects and a Rust binary has one autorelease
   pool, around `main`, so every command buffer the process created stayed
   alive: ~180 KiB per decoded token, linear, which reads as "grows with
   prompt length". Decode now runs one pool per token. That closed the
   entire unexplained gap. See `AGENTS.md` Gotcha 17.

The earlier suspicion that the growth was "host-side, since per-token
Metal buffer allocation is already asserted flat" was right about the
location and wrong about the owner: the allocations were Metal's, just
not ours to count.

## The quality gates

A sibling of the memory oracle, same shape and same gating: `#[ignore]`d,
one target per family, keyed on `TURBOSPARK_<FAMILY>_INSTALL_DIR`, asserting
per-chip rows that record their own provenance. What it measures is
quality rather than memory or speed, which is the one axis with no Swift
column at all (the original publishes no perplexity, no KLD, no golden
output).

```sh
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_gate --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_quality_gate --release -- --ignored --nocapture

# Proof the gate above can see quantization damage, rather than assuming it.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_sensitivity --release -- --ignored --nocapture
```

Four arms per install, about 80 seconds: teacher-forced perplexity of a
fixed reference answer, a greedy digest, a sampled digest at the protocol
seed, and the greedy digest repeated with the expert cache halved to 8
slots. Three traps that shaped it, all measured rather than reasoned:

- **Score only assistant-position tokens.** SFT masks the loss on the
  prompt, so teacher-forcing prompt text scores worse than a uniform
  distribution.
- **A golden digest needs a warm expert cache**, so the run order (warmup,
  measure, measure, warmup, measure) is part of the protocol.
- **Byte identity across slot counts holds on both families** as of
  2026-08-08, so the quality gate asserts the 8-slot digest equals the
  16-slot one instead of freezing one per count. Before then the Gemma
  flow dispatched routed slots misses-first, which made phase 2's reduce
  order follow expert-cache state (AGENTS.md Gotcha 27).
- **Hand the second engine token ids, never prose.** A tokenizer or
  chat-template difference would surface as a divergence and read as a
  numerics gap. `meta.json` carries the exact sequence this port walked.
- **Measure a floor in the same run.** A cross-engine KL has no natural
  scale, so `kld.py` also runs mlx-lm against itself in its two forward
  shapes (batched vs token-by-token through a cache). Measured, that
  intra-engine floor is larger than the cross-engine number.
- **The assistant-slot rule does not carry over.** It exists because SFT
  masks prompt loss; a distribution comparison between two engines is
  valid at every position, and prompt positions are free.

## The power harness

`scripts/power.sh` reports average watts and joules-per-token over the
frozen protocol, split prefill vs decode (ROADMAP Phase P1). Numbers,
hygiene audit and caveats: `docs/POWER_BASELINE.md` (summary table also
in `docs/BENCHMARKS.md`).

```sh
LABEL=battery OUT=/tmp/power-gemma MODEL=~/models/gemma4.gturbo scripts/power.sh 2
# Interleaved A/B of the read-pool QoS seam, arms alternating within a pair:
LABEL=battery MODEL=~/models/gemma4.gturbo CASES=short-explanation \
  QOS=default,utility scripts/power.sh 3
```

It needs root, because `powermetrics` does. One sampler runs for the
whole script and every arm is windowed out of that single log.

Four things about it are load-bearing:

- **The window is marker-driven, and without the markers the number is
  meaningless.** `turbospark-bench --model` opens a 13 GB mmap, compiles
  Metal pipelines, and runs a discarded 1024-token warmup before the
  measured run. Wrapping the process would fold all of that into the
  energy total. `run_model_mode` therefore emits
  `[power-window case=... phase=start|end unix_ms=...]` on stderr around
  the measured run alone. The prefill/decode split inside that window
  needs no third marker: the `[stop=...]` footer already carries both
  durations.
- **The timeline is reconstructed, so its drift is measured, not
  assumed.** `powermetrics` timestamps samples only to the second, so the
  script rebuilds an absolute timeline by accumulating each sample's
  reported elapsed figure from a wall-clock start. Drift against the wall
  clock is printed every run and warned on past 2 s; it measured -0.7 s
  over 742 s.
- **Watts are CPU+GPU+ANE, not wall.** `Combined Power` excludes DRAM,
  SSD, and display. The battery-gauge column (`ioreg`, no root) is the
  wall figure, and it is too noisy to publish -- see the caveats in
  `docs/BENCHMARKS.md`.
- **Contaminated runs are flagged, never averaged in silently.** Any run
  whose thermal pressure leaves Nominal, and any window integrated from
  under 3 samples, prints a warning. On battery this fires often: the
  long-context case saturates thermally on both installs.

## Measurement hygiene

Carried over from the Swift protocol, worth repeating:

- Release builds only.
- No other model process running (`pgrep -fl 'mference|mlx'`).
- Interleave A/B variants pair by pair. Run-to-run throughput spread on
  this port is around 1.5 tok/s, wider than most single changes, and
  consecutive batches carry thermal drift.
- No profiler, trace, or experimental control active during a run that
  will be quoted.
- **Record the PAGE-CACHE STATE of a streamed install, the way a power row
  records its power source and its cooling.** An MoE install's expert reads
  are a page-cache memcpy at ~23.8 GiB/s when warm and real device I/O when
  not, and the difference is not a margin: `docs/BENCHMARKS.md` records a
  34 GB install reading **8.5 tok/s cold against 22 warm**, a 2.6x error,
  found by accident rather than by an instrument. AGENTS.md Gotcha 20's
  discard-a-warmup rule was written for a cold GPU and applies at least as
  hard here.

  Since 2026-08-29 this is measurable rather than inferred. Add
  `TURBOSPARK_EXPERT_DISK_IO=1` to a `TURBOSPARK_PHASES=1` run and the
  `expert bytes` row reports requested MiB/token, physical MiB/token and
  their ratio; near-zero physical is the warm regime and anything
  approaching the requested figure is the disk-bound one. It costs a syscall
  pair per read batch, so it is off by default and does not belong in a run
  being quoted for tok/s.

  `TURBOSPARK_EXPERT_NOCACHE=1` establishes the disk-bound arm deliberately,
  in the shape of `scripts/power.sh COOLING=max`. Two caveats that make it a
  SEPARATE published row rather than a replacement: it is an operating point
  no user occupies, and `F_NOCACHE` prevents retention without evicting, so
  a blob already faulted in stays resident and needs `sudo purge` beside it.
  Confirm from `bytes_physical`, never from the flag
  (`crates/streaming/AGENTS.md` Gotcha 8). Measured on the real Gemma 4
  install at 16 slots: warm reads 0.0 MiB/token physical against 274.9
  requested, and `F_NOCACHE` after a purge reads 274.9 of 274.9 (1.00x).

  **DO NOT INTERLEAVE THOSE TWO ARMS, which is an exception to the rule
  three bullets up.** A caching run repopulates the buffer cache and
  `F_NOCACHE` does not evict, so a bypassed run that FOLLOWS a warm one is
  served from memory and reads 0.00x -- the comparison destroys its own
  condition. Observed in one sitting: 1.00x, 1.00x, then two caching runs,
  then 0.00x, 0.00x. Run the bypassed arms consecutively after a purge and
  the warm arms afterwards, and carry Gotcha 22's cross-capture caveat.
