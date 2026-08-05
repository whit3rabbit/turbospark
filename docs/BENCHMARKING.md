# Benchmarking

How to measure this port's throughput and memory, and how the numbers
compare to the Swift original (`../Mference`, `docs/BENCHMARKS.md` there).

Everything here lives in `crates/bench`: the `mference-bench` binary, a
small library the binary and the oracle test share, and
`tests/memory_oracle.rs`.

## The three modes

| Mode | Command | What it measures |
| --- | --- | --- |
| Scripted (default) | `mference-bench <tokenizer-dir>` | This port's prefill+decode *loop* overhead. No model. |
| Synthetic real | `mference-bench <tokenizer-dir> --real` | The real GPU dispatch path over a tiny synthetic install. |
| Real install | `mference-bench --model <install-dir>` | Real Gemma 4 throughput and peak memory. The Swift-comparison number. |

Only the third mode is comparable to anything published. The first two
exist so the loop and the dispatch path can be timed without a
multi-gigabyte checkout.

### Scripted

```bash
cargo run -p mrefrust-bench --bin mference-bench -- crates/tokenizer/tests/fixtures/ChatMLTokenizer
```

Three fixed prompts, fixed seed 42, one discarded warmup per prompt, all
driven through the real `run_raw_completion` loop against a
`ScriptedLogitProducer` (a fixed replayed logit sequence). The tok/s
printed is tokenizer + sampler + detokenizer + stop-matcher cost, not
inference. Portable; runs on Linux.

### Synthetic real (macOS)

```bash
cargo run -p mrefrust-bench --bin mference-bench -- <tokenizer-dir> --real
```

Builds a tiny dense `.gturbo` install (deterministic INT4 weights, vocab
sized to the tokenizer) in a temp dir and drives the same three prompts
through `RealForwardRunner`: real Metal kernels, real KV cache, real
zero-copy resident weights. The model is far too small for the number to
mean anything as throughput; it measures the dispatch path.

### Real install (macOS)

```bash
cargo run --release -p mrefrust-bench --bin mference-bench -- --model ~/models/gemma4.gturbo
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
- Budget: `--max-new 1024`, 4K context.
- Each case runs one discarded warmup, then one measured run.
- Prompts are chat-templated exactly as the CLI templates them. The IT
  checkpoint needs its turn markup; a raw prompt babbles.

Use `--release`. A debug build's numbers are meaningless.

Output per case, on stdout:

```
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

## The memory oracle

```bash
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture
```

`crates/bench/tests/memory_oracle.rs` runs the same protocol in-process
and turns it into assertions. `#[ignore]`d (it needs a real ~14.6 GB
install and takes several minutes); skips with a printed note if
`MREFRUST_GEMMA4_INSTALL_DIR` is unset.

What it asserts, in order:

1. **Stop reason.** Every measured case must stop with `endOfTurn`. A run
   that dies on `maxTokens` is not comparable to the published rows, so
   this gates everything after it.
2. **Peak footprint** at or under the Swift ceiling. Asserted always;
   memory sizing does not depend on the chip.
3. **Decode tok/s** at or above the Swift floor, per case, only when the
   chip brand matches a baseline row.

Baselines come from Swift's `docs/BENCHMARKS.md` Gemma 4 rows. Chip
detection is `sysctl machdep.cpu.brand_string`, matched by substring,
most specific row first.

| Chip | Swift peak footprint | Ceiling used | Swift decode | Floor used |
| --- | --- | --- | --- | --- |
| Apple M5 Pro (24 GB) | 2,126-2,142 MiB | 2,250 MiB | 31.01-35.17 tok/s | 31.0 |
| Apple M2 (8 GB) | 1,776-1,971 MiB | 2,070 MiB | 5.10-6.30 tok/s | 5.1 |
| Anything else | - | 2,250 MiB | - | reported, not asserted |

Ceilings are the documented peak plus about 5 percent. That is Swift's
own cross-run variance (its repeat table spans 1,388-1,464 MiB on
identical runs); more headroom than that would hide a regression the size
of a single KV layer. Throughput floors take the documented minimum
verbatim, since Swift's cross-run throughput spread is around 1 percent.

A tok/s failure means the port decodes slower than Swift on that
hardware. It is a finding, not a broken test.

## Static KV accounting

`crates/gpu/tests/kv_cache.rs` has a cheap deterministic companion to the
oracle: for the real Gemma 4 shape at 4K it asserts the fp16 SWA ring caps
the 25 sliding-window layers at 1,152 rows (`sliding_window` 1024 plus a
128-token prefill chunk, the Swift sizing) while the 5 full-attention
layers stay linear at 4,096, for 319,815,680 bytes of KV total. Linear
everywhere would be 922,746,880 bytes. This runs in the normal suite, so a
regression in KV sizing fails in milliseconds instead of waiting on a
multi-minute oracle run.

## Current measured state (2026-08-05)

Apple M4 Max, real `gemma4.gturbo` install, oracle run:

| case | prompt tok | tok/s | cumulative peak |
| --- | ---: | ---: | ---: |
| short-explanation | 61 | 23.7 | 2,711 MiB |
| medium-review | 430 | 19.2 | 3,404 MiB |
| long-synthesis | 3,015 | 13.8 | 5,332 MiB |

The oracle fails on this install, for two reasons, both real:

1. **No case reaches `endOfTurn`.** Generation degrades into word salad
   after roughly 40 tokens and runs to the 1024-token budget. Degradation
   starts around sequence length 100, far below the 1,152-row ring, so it
   is not KV-ring related; suspicion is on the real-checkpoint decode flow
   in `crates/runtime/src/real_forward_gemma4.rs`.
2. **Peak footprint is over budget**, 5,332 MiB against a 2,250 MiB
   ceiling. Static accounting explains only about 3.1 GiB (1.50 GiB of
   expert slot capacity at 16 slots x 30 layers x 3,358,720-byte stride,
   1.26 GiB of resident weights, 305 MiB of KV); the rest, and the growth
   with prompt length, is unexplained. Per-token Metal buffer allocation
   is already asserted flat, so the growth is host-side.

Both are tracked as follow-up work, not regressions from the benchmark
harness itself.

## Measurement hygiene

Carried over from the Swift protocol, worth repeating:

- Release builds only.
- No other model process running (`pgrep -fl 'mference|mlx'`).
- Interleave A/B variants pair by pair. Run-to-run throughput spread on
  this port is around 1.5 tok/s, wider than most single changes, and
  consecutive batches carry thermal drift.
- No profiler, trace, or experimental control active during a run that
  will be quoted.
