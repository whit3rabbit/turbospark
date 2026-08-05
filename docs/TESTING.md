# Testing

What the suite covers, how it is gated, and how to run each part.

## The default suite

```sh
cargo test --workspace
```

316 tests as of 2026-08-05, all passing, plus 3 that are `#[ignore]`d (see
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
| `server` | OpenAI-compatible endpoint shapes, full-response and SSE. |
| `bench` | Protocol constants and footer format, the memory sampler, and the binary's black-box output. |

## Gating conventions

Three gates are in use. Match the existing idiom when adding a test.

### macOS

Metal-dependent test files carry a crate-level attribute as line 1:

```rust
#![cfg(target_os = "macos")]
```

All of `crates/gpu/tests/*`, `crates/runtime/tests/{real_forward,
real_forward_gemma4,golden_tokens}.rs`, `crates/cli/tests/real_generation.rs`,
and `crates/bench/tests/memory_oracle.rs` do this. Inside an otherwise
portable file, gate the single item instead
(`crates/cli/tests/mference_check.rs`).

Source-side, `crates/gpu/src/lib.rs` gates each `mod` and `pub use`
individually, and `crates/bench/src/main.rs` uses the cfg'd function-pair
idiom (real implementation plus a stub that exits 2).

### Ignored (expensive or needs external data)

Three tests, each with a reason string and a module doc giving the exact
command:

```sh
# Real ~14.6 GB Gemma 4 checkpoint download plus full repack.
cargo test -p mrefrust-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture

# Real ~270 MB HF checkpoint download through the Llama-family mapping.
cargo test -p mrefrust-repack --test hf_checkpoint_network --release -- --ignored --nocapture

# The memory oracle (see docs/BENCHMARKING.md).
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture
```

Run all ignored tests at once with `cargo test --workspace -- --ignored`,
but note the first two download many gigabytes.

### Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `MREFRUST_GEMMA4_INSTALL_DIR` | `gemma4_checkpoint_network`, `memory_oracle` | Where the real `.gturbo` install lives. The oracle skips (with a note) when unset; the repack test falls back to a temp dir. |
| `MFERENCE_PHASES=1` | `mference-check` | Prints the per-phase decode breakdown (GPU wait, router readback, expert pread, routed bind, cache hit rate). |
| `MFERENCE_SHARED_CB=0` | `RealForwardRunner` | Reverts the shared-expert branch to encoding after the expert pread instead of on its own overlapping command buffer. The A/B seam for any throughput claim. |

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
