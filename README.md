# Inference engine (Rust port)

A behavior-compatible Rust port of the Mference local inference engine and
its command-line/server surface. The Swift original is public at
<https://github.com/drumih/turbo-fieldfare>; its experiment inventory
(`docs/experiments/EXPERIMENT_INVENTORY.md` there) is cross-referenced
against this port's own measurements in `DEVIATIONS.md`. See `ROADMAP.md`
(gitignored, local) for the forward roadmap and descope record, and
`DEVIATIONS.md` for what is fully wired versus scaffolded.

This workspace is built and tested with cargo.

## Where it stands

Two model families run end to end, Gemma 4 26B-A4B and Qwen 3.6 35B-A3B:
a real Metal forward pass per token, with each layer's routed experts
streamed from SSD instead of held in RAM. The model does not have to fit
in memory, only its working set does.

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

Qwen 3.6 35B-A3B, proven the same day on the real checkpoint: 32.6 to
38.0 tok/s decode on the same frozen protocol, peak footprint 1,587 to
1,610 MiB on an 18 GB install. That is roughly 500 MiB under Gemma
despite the larger model, because 30 of its 40 layers are gated-DeltaNet
linear attention carrying ~2 MiB of fixed recurrent state each instead
of a KV cache, so context growth touches 10 layers instead of 30. No
Swift side-by-side has been run for this family yet.

Full numbers, provenance, and caveats: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
Reproduce with `scripts/parity.sh`. How the harness works, and how this
port tracks itself over time: [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Layout

- `crates/core`: shared primitives and the public runtime configuration.
- `crates/compute`: CPU reference kernels (RmsNorm, RoPE, attention,
  int4/int8 affine quant, GGUF Q8_0 and Q4_K block quant, MoE, sampling)
  and the
  destination compute strategy. Every GPU kernel is parity-tested against
  one of these before it is trusted.
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
- `crates/repack`: safetensors and GGUF header parsing, ranged-download
  planning, and quantization repack. GGUF intake runs both real published
  files end to end -- Gemma 4's Q8_0 and Qwen 3.6's mixed Q4_K_M, each
  streamed from Hugging Face without the 20-27 GB checkpoint ever landing
  on disk -- and refuses Q4_0 at open, on purpose and twice over, until its
  kernels land (ROADMAP Phase G): a block type needs a resident GEMV, an
  embedding lookup and a routed-expert decode pair before it executes.
  Q6_K is the exception with a GEMV alone, which is all any real file asks
  of it. The resident core is transcoded at repack
  time from the F32 llama.cpp writes into the BF16 and INT8 the existing
  kernels read, which was measured to be lossless for norms rather than
  assumed to be. Qwen additionally needs its source CONVENTIONS undone
  there (llama.cpp orders V heads differently and stores `-exp(A_log)`),
  which is a repack-time byte permutation and not a kernel.
- `crates/server`: OpenAI Chat Completions and Anthropic Messages server on loopback.
- `crates/bench`: throughput benchmark harness, the memory oracle tests, and
  the per-install quality gates (perplexity plus frozen output digests, a
  proof that the perplexity responds to quantization damage, and a logit
  dump feeding `scripts/kld.py`'s cross-engine KL against mlx-lm).

## Build and test

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

## License

MIT.
