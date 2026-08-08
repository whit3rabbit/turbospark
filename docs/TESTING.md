# Testing

What the suite covers, how it is gated, and how to run each part.

## The default suite

```sh
cargo test --workspace
```

439 tests as of 2026-08-07, all passing, plus 17 that are `#[ignore]`d (see
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
| `bench` | Protocol constants and footer format, the memory sampler, and the binary's black-box output. Its gated targets carry the quality axis: per-install perplexity and golden digests, the damage-sensitivity proof, and the logit dump feeding the cross-engine KLD. |

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

Ten of them, each with a reason string and a module doc giving the exact
command (the eleventh, `crates/selection`'s `rank_top_k`, is a sampler
microbenchmark documented in `docs/BENCHMARKS.md`):

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

# The three GGUF checks (ROADMAP Phase G). All are `*_network` but NONE of
# them downloads a checkpoint: each reads a few KB to a few MB off a
# 20-27 GB remote file over range requests, in seconds. They are grouped
# here rather than above for that reason -- do not budget a download for
# them, and prefer a ranged read when adding the next one.
#
# 1. The header, three ways: this port's parser against llama.cpp's
#    converter, every tensor name maps, and the ArchConfig derived from
#    GGUF metadata equals the one the .gturbo install declares.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-repack --test gguf_checkpoint_network --release -- --ignored --nocapture

# 2. Which half of Gemma's fused ffn_gate_up_exps is the gate. Correlates a
#    dequantized layer 0 expert 0 against the MLX install; doubles as a
#    real-data check on the Q8_0 dequant reference.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-repack --test gguf_fused_gate_network --release -- --ignored --nocapture

# 3. The evidence behind the repack-time transcode decision: GGUF's F32
#    norms are upcast BF16 and narrow back bit-exactly, and INT8-transcoding
#    its F32 router does not move the routing decision. Needs no install.
cargo test -p mrefrust-repack --test gguf_f32_transcode_network --release -- --ignored --nocapture

# The memory oracle (see docs/BENCHMARKING.md). One target per model
# family: the footprint assertion is a whole-session peak, so two
# families in one process cannot each have a ceiling.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-bench --test qwen36_memory_oracle --release -- --ignored --nocapture

# The quality gate (ROADMAP Phase Q; numbers in docs/BENCHMARKS.md). Split
# per family for the same one-model-per-process reason as the oracle.
# Takes about 80 seconds each (four arms: perplexity, greedy, sampled, and
# a constrained-working-set repeat at 8 expert-cache slots).
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test quality_gate --release -- --ignored --nocapture
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-bench --test qwen36_quality_gate --release -- --ignored --nocapture

# Proof the gate above can see quantization damage: clone the install
# (APFS clonefile, original untouched), shift one quantization level in a
# strided subset of the routed experts, re-measure. About 30 seconds.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test quality_sensitivity --release -- --ignored --nocapture

# The cross-engine half of Phase Q, and the one test here whose second
# step is NOT cargo: it dumps this port's full-vocab logits plus the exact
# token ids it walked (~275 MiB, ~30 s), then replays those IDS through
# mlx-lm and prints the KL. Needs the 14.6 GB reference checkpoint
# (`hf download mlx-community/gemma-4-26b-a4b-it-4bit --revision
# 0d77464eeb233a2da68ebf9d7dc4edaac7db956d`). mlx-lm runs in a uv
# ephemeral env, so it is never installed globally and never enters this
# workspace's dependency graph. MREFRUST_LOGIT_DUMP_COLD=1 skips the
# warmup walk and reproduces quality_gate's frozen perplexity exactly,
# which is the cross-check that the dump measures what the gate measures.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
MREFRUST_LOGIT_DUMP_DIR=/tmp/kld/mrefrust \
  cargo test -p mrefrust-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with mlx-lm --with numpy scripts/kld.py /tmp/kld/mrefrust

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
| `MREFRUST_GEMMA4_INSTALL_DIR` | `gemma4_checkpoint_network`, `memory_oracle`, `quality_gate`, `quality_sensitivity`, `logit_dump`, `real_backend`, `gguf_checkpoint_network`, `gguf_fused_gate_network` | Where the real `.gturbo` install lives. The oracle, the quality gate, the logit dump, and the server test skip (with a note) when unset; the repack test falls back to a temp dir. The two GGUF tests use the install as an independent REFERENCE rather than as a subject: `gguf_checkpoint_network` cross-checks names and `ArchConfig` against it and skips those two checks when unset, and `gguf_fused_gate_network` correlates against its expert weights and skips entirely. Both accept a leading `~/`. |
| `MREFRUST_QWEN36_INSTALL_DIR` | `qwen36_checkpoint_network`, `qwen36_memory_oracle`, `qwen36_quality_gate`, `logit_dump` | Where the repacked Qwen 3.6 install lives. The oracle and the quality gate skip (with a note) when unset; the repack test falls back to a temp dir. Deliberately a second variable rather than a generalized one, so both installs can coexist and each target asserts its own family's row. `logit_dump` takes it as a fallback when the Gemma variable is unset, though `scripts/kld.py`'s reference is pinned to the Gemma repo. |
| `MREFRUST_LOGIT_DUMP_DIR` | `logit_dump` | Where to write `logits.f16` and `meta.json` (~275 MiB on either family). Required: the target skips when unset, since a few hundred MB is not something to write to a default path. |
| `MREFRUST_LOGIT_DUMP_COLD=1` | `logit_dump` | Skips the warmup walk, so the dump comes off a COLD expert cache. That is the condition `quality_gate` takes its perplexity under, so this reproduces the frozen row exactly; without it the dump is warm and reads about 0.5% higher. How the warm/cold difference gets measured rather than assumed. |
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
