# Testing

What the suite covers, how it is gated, and how to run each part.

## The default suite

```sh
cargo test --workspace
```

326 tests as of 2026-08-06, all passing, plus 4 that are `#[ignore]`d (see
below). On macOS this includes every Metal test, which needs a real
Metal-capable device and Xcode's `metal` toolchain
(`xcrun -sdk macosx metal`). On Linux `crates/gpu` compiles to nothing and
the GPU-dependent test files compile away with it, so the same command
stays green.

The full pre-handoff gate:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

`make check` runs fmt-check + clippy + test-debug.

## What each crate's tests prove

| Crate | Covers |
| --- | --- |
| `core` | Runtime-config allowed sets and their panicking setters, chunk-size resolution. |
| `compute` | CPU reference kernels (RmsNorm, WHT, RoPE, causal attention, int4/int8 quant + GEMV, embedding, MoE FFN, softcap-softmax). These are the numerical ground truth the GPU is checked against. |
| `invocation` | Argument parsing: outcomes, ordering, failures, defaults, usage text, exit status. Pure, no I/O. |
| `selection` | Sampling contract: shaping, truncation, penalty, choose. |
| `window-fit` | Conversation-window turn dropping. |
| `tokenizer` | Dialect resolution, chat templates (including the real vendored Qwen ChatML template through minijinja), streaming detokenizer, stop matching, tool-call parsing. |
| `model-io` | Manifest decode and field validation, packed-expert layout, resident index, SHA-256, install receipt. |
| `streaming` | Expert cache eviction policy against scripted access traces (no real install needed). |
| `gpu` | Per-kernel parity against the matching `compute` reference on real hardware, plus KV cache sizing and the resident zero-copy proof. macOS only. |
| `runtime` | The raw-completion loop against `ScriptedLogitProducer`, plus real end-to-end forward passes over synthetic installs. |
| `cli` | Black-box binary invocation, including real generation in all three modes. |
| `repack` | Safetensors parsing, quantization, `.gturbo` assembly round-tripped through every `model-io` loader, install verification. |
| `server` | OpenAI-compatible endpoint shapes, full-response and SSE, the `--model` argument parser, and (gated) the real `RealForwardRunner` backend end to end. |
| `bench` | Protocol constants and footer format, the memory sampler, and the binary's black-box output. |

## Gating conventions

Three gates are in use. Match the existing idiom when adding a test.

### macOS

Metal-dependent test files carry a crate-level attribute as line 1:

```rust
#![cfg(target_os = "macos")]
```

All of `crates/gpu/tests/*`, `crates/runtime/tests/{real_forward,
real_forward_gemma4,real_forward_qwen,golden_tokens}.rs`, `crates/cli/tests/real_generation.rs`,
and `crates/bench/tests/memory_oracle.rs` do this. Inside an otherwise
portable file, gate the single item instead
(`crates/cli/tests/mference_check.rs`).

Source-side, `crates/gpu/src/lib.rs` gates each `mod` and `pub use`
individually, and `crates/bench/src/main.rs` uses the cfg'd function-pair
idiom (real implementation plus a stub that exits 2).

### Ignored (expensive or needs external data)

Six tests, each with a reason string and a module doc giving the exact
command:

```sh
# Real ~14.6 GB Gemma 4 checkpoint download plus full repack.
cargo test -p mrefrust-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture

# Real ~20.4 GB Qwen 3.6 checkpoint download plus full repack. The only
# test that covers the multi-shard walk on that family: its synthetic
# fixture is a single in-memory shard, so a companion tensor living in a
# different shard than its weight is unexercised in the default suite.
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-repack --test qwen36_checkpoint_network --release -- --ignored --nocapture

# Real ~270 MB HF checkpoint download through the Llama-family mapping.
cargo test -p mrefrust-repack --test hf_checkpoint_network --release -- --ignored --nocapture

# The memory oracle (see docs/BENCHMARKING.md).
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture

# Split-KV chunk-count sweep on the decode attention kernel. Needs no
# model install: it is the kernel alone at the real Gemma 4 shapes, and
# it reports speedup ratios rather than absolute times so it stays
# readable on a throttled machine. Takes seconds.
cargo test -p mrefrust-gpu --test attention_chunk_bench --release -- --ignored --nocapture

# The server's real backend end to end: one model open, one non-streaming
# and one streaming request through the bound loopback server.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-server --test real_backend --release -- --ignored --nocapture
```

Run all ignored tests at once with `cargo test --workspace -- --ignored`,
but note that three of them download many gigabytes.

A performance measurement is `#[ignore]`d rather than left out because it
is the evidence behind a constant in `src/` (`MAX_CHUNKS`), and a constant
whose justification cannot be re-run is a constant nobody will ever dare
change. It asserts correctness (every chunk count computes the same
attention) so it cannot rot silently; only the timings are advisory.

### Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `MREFRUST_GEMMA4_INSTALL_DIR` | `gemma4_checkpoint_network`, `memory_oracle`, `real_backend` | Where the real `.gturbo` install lives. The oracle and the server test skip (with a note) when unset; the repack test falls back to a temp dir. |
| `MREFRUST_QWEN36_INSTALL_DIR` | `qwen36_checkpoint_network` | Where to keep the repacked Qwen 3.6 install. Falls back to a temp dir when unset. Deliberately a second variable rather than a generalized one: the two installs coexist, and the oracle's protocol is still Gemma-shaped. |
| `MFERENCE_PHASES=1` | `mference-check` | Prints the per-phase decode breakdown (GPU wait, router readback, expert pread, routed bind, cache hit rate). |
| `MFERENCE_DISPATCH_PROFILE=1` | `mference-check`, `gpu::PassEncoder` | Ranks the individual dispatches inside each command buffer. Encodes one compute encoder per dispatch (Apple GPUs sample counters only at encoder boundaries) and waits on every buffer, so it perturbs the run it measures: a ranking aid, not a throughput number. Covered by `crates/gpu/tests/dispatch_profile.rs`. |
| `MFERENCE_SHARED_CB=0` | `RealForwardRunner` | Reverts the shared-expert branch to encoding after the expert pread instead of on its own overlapping command buffer. The A/B seam for any throughput claim. |
| `MFERENCE_HIT_CB=0` | `RealForwardRunner` | Reverts the cache-hit experts' phase-1 GEMV to the main pass instead of its own command buffer dispatched before the pread. |
| `MFERENCE_ROUTED_PIPELINE=0` | `RealForwardRunner` | Reverts a layer's routed-expert command buffer to rolling uncommitted into the next layer's first buffer instead of committing at the end of the layer (the one-layer pipeline). |

## Test-writing notes

- **Never hardcode a token id from fixture JSON.** The vendored tokenizer
  fixtures carry high placeholder ids (e.g. `248044`) that the `tokenizers`
  crate renumbers at load time. Resolve ids from a loaded `MfTokenizer`
  (`token_to_id`, `end_of_turn_id`).
- **Never assert specific generated text.** Synthetic installs have
  deterministic but untrained weights, so output is structurally real and
  semantically meaningless, and a short generation can decode to the empty
  string. Assert that generation ran: token counts, stop reason, log lines.
- **Build test installs with the helpers**, not a hand-written
  `ArchConfig`: `repack::build_synthetic_gemma4_install` (dense) or its
  `_swa` / `_moe` / `_moe_streamed` variants, or
  `build_synthetic_gemma4_real_install` (verbatim checkpoint naming, which
  is what selects the runner's real Gemma 4 decode flow). Their non-shape
  fields are pinned to `gemma4_26b_a4b()`'s values on purpose.
- **Prefer exact assertions over thresholds.** The steady-state memory
  guarantee is asserted as "zero Metal buffers allocated per token"
  (`dense_decode_allocates_no_gpu_buffers_per_token`), not as an RSS
  threshold, because the exact form cannot go flaky. The one place a
  threshold is unavoidable is the memory oracle, which is why it carries a
  documented headroom policy.
- **Temp dirs** follow the repeated idiom: a `static AtomicU64` counter
  plus `std::env::temp_dir().join(format!("<label>-{pid}-{n}"))`.
- **Metal tests run serially by shared device state** in practice; the
  Swift original runs `swift test --no-parallel` for the same reason. If a
  GPU test starts flaking under parallelism, that is the first thing to
  suspect.

## Benchmarks

Throughput and memory measurement, including the memory oracle and the
Swift baseline table it asserts against, are documented separately in
[BENCHMARKING.md](BENCHMARKING.md).
