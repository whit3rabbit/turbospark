# Inference engine (Rust port)

A behavior-compatible Rust port of the Mference local inference engine and
its command-line/server surface. The Swift original is public at
<https://github.com/drumih/turbo-fieldfare>; its experiment inventory
(`docs/experiments/EXPERIMENT_INVENTORY.md` there) is cross-referenced
against this port's own measurements in `DEVIATIONS.md`. See `ROADMAP.md`
for phase-by-phase scope and `DEVIATIONS.md` for what is fully wired
versus scaffolded.

This workspace is built and tested with cargo.

## Where it stands

Gemma 4 26B-A4B runs end to end: a real Metal forward pass per token, with
each layer's routed experts streamed from SSD instead of held in RAM. The
model does not have to fit in memory, only its working set does.

On 2026-08-07 the port and the Swift original were measured back to back
on one machine, reading the same model directory, through the same frozen
benchmark protocol. Apple M4 Max, 36 GB, on AC power. Rust at `ef4e953`
plus the sampler change that run measured, Swift at `1bb585c`.

| | This port | Swift original |
| --- | ---: | ---: |
| Decode | 34.6 to 40.7 tok/s | 34.3 to 41.1 tok/s |
| Peak memory | 2,108 to 2,182 MiB | 2,217 to 2,235 MiB |
| Install on disk | 14 GB | same directory |

Two things to take from that:

- **Memory holds.** A 26B model with roughly 3.9B parameters active per
  token, served out of about 2.1 GiB of peak footprint. That is the whole
  point of the design, and this port delivers it in 2 to 5 percent less
  memory than the original.
- **Decode is at parity**, within 1 percent on all three cases. An earlier
  run of the same script measured this port at 0.64 of Swift; that gap was
  a full sort of all 262,144 candidates in the host sampler, costing more
  per token than the entire GPU forward pass. `docs/BENCHMARKS.md` records
  what it was and why every profiler in the repo was blind to it.

Prefill splits the other way: this port is faster below roughly 350 prompt
tokens (no fixed startup cost) and slower above it (the original batches
prompt tokens in chunks of 128; this port runs one forward pass per token,
because those tile kernels are out of scope).

Full numbers, provenance, and caveats: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
Reproduce with `scripts/parity.sh`. How the harness works, and how this
port tracks itself over time: [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Layout

- `crates/core`: shared primitives and the public runtime configuration.
- `crates/compute`: CPU reference kernels (RmsNorm, RoPE, attention,
  int4/int8 quant, MoE, sampling) and the destination compute strategy.
- `crates/invocation`: command-line argument translation, usage text, and
  process-status/stream routing decisions (pure data, no I/O).
- `crates/selection`: candidate selection under a validated shaping
  configuration.
- `crates/window-fit`: conversation-window fitting (turn dropping).
- `crates/tokenizer`: tokenizer wrapper, chat templates, streaming
  detokenizer, tool-call parsing.
- `crates/model-io`: manifest/arch validation, packed-expert layout,
  resident tensor index, SHA-256 verification, install receipt.
- `crates/streaming`: routed-expert `pread` streamer and its LFU/LRU cache
  policy.
- `crates/gpu`: Metal pipeline cache and kernel dispatch (macOS only).
- `crates/runtime`: the raw-completion prefill+decode loop.
- `crates/cli`: the `mference-check` process entry point.
- `crates/repack`: safetensors header parsing, ranged-download planning,
  and quantization repack.
- `crates/server`: OpenAI Chat Completions and Anthropic Messages server on loopback.

## Build and test

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

## License

MIT.
