# Testing

What the suite covers, how it is gated, and how to run each part.

## The default suite

```sh
cargo test --workspace
```

458 tests as of 2026-08-08, all passing, plus 18 that are `#[ignore]`d (see
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

Sixteen targets carry `#[ignore]`d tests, 18 functions between them
(`cargo test --workspace` prints the count; two GGUF network targets carry
more than one). Each has a reason string and a module doc with the exact
command. The commands below are the ones that are GATES. The two that are
not are documented in `docs/BENCHMARKS.md` instead: `crates/selection`'s
`rank_top_k` (a sampler microbenchmark) and `crates/gpu`'s
`attention_chunk_bench` (the split-KV chunk sweep).

```sh
# Real ~14.6 GB Gemma 4 checkpoint download plus full repack.
cargo test -p turbospark-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture

# Real ~20.4 GB Qwen 3.6 checkpoint download plus full repack. The only
# test that covers the multi-shard walk on that family: its synthetic
# fixture is a single in-memory shard, so a companion tensor living in a
# different shard than its weight is unexercised in the default suite.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test qwen36_checkpoint_network --release -- --ignored --nocapture

# Real ~270 MB HF checkpoint download through the Llama-family mapping.
cargo test -p turbospark-repack --test hf_checkpoint_network --release -- --ignored --nocapture

# The four GGUF checks (ROADMAP Phase G). All are `*_network` but NONE of
# them downloads a checkpoint: each reads a few KB to a few MB off a
# 20-27 GB remote file over range requests, in seconds. They are grouped
# here rather than above for that reason -- do not budget a download for
# them, and prefer a ranged read when adding the next one.
#
# 1. The header, three ways: this port's parser against llama.cpp's
#    converter, every tensor name maps, and the ArchConfig derived from
#    GGUF metadata equals the one the .gturbo install declares.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- --ignored --nocapture

# 2. Which half of Gemma's fused ffn_gate_up_exps is the gate. Correlates a
#    dequantized layer 0 expert 0 against the MLX install; doubles as a
#    real-data check on the Q8_0 dequant reference.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-repack --test gguf_fused_gate_network --release -- --ignored --nocapture

# 3. The evidence behind the repack-time transcode, which is now LANDED
#    (`gguf_checkpoint.rs::transcode_f32`): GGUF's F32 norms are upcast
#    BF16 and narrow back bit-exactly, and INT8-transcoding its F32 router
#    does not move the routing decision. Needs no install. The transcode's
#    own behaviour is covered by the unit tests in
#    `crates/repack/tests/gguf_checkpoint.rs`, on the synthetic fixture;
#    this one is why those tests are allowed to assume what they assume.
cargo test -p turbospark-repack --test gguf_f32_transcode_network --release -- --ignored --nocapture

# 4. The Q4_K reference against real published bytes. Correlates a
#    dequantized layer 0 expert 0 gate row of the real Qwen 3.6 Q4_K_M
#    against the same row in the install (+0.9930), with the unrelated `up`
#    matrix as the control (+0.12). This is the only check that can catch a
#    decoder and the fixture quantizer that feeds it being wrong TOGETHER,
#    which is the failure mode `crates/compute`'s unit tests structurally
#    cannot see. It skips all-zero rows and asserts the two sides agree on
#    which those are: see AGENTS.md Gotcha 30 before reading a low number
#    out of it.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_q4_k_network --release -- --ignored --nocapture

# 5. Qwen's SOURCE CONVENTIONS, which are not a format question: llama.cpp
#    orders V heads differently and stores -exp(A_log). Checks every tensor
#    on that axis on every layer, and with TURBOSPARK_QWEN_PATCH=1 rewrites
#    them in place, which is how a whole-model coherence test costs seconds
#    instead of a ~21-minute repack. Numbers in crates/repack/CLAUDE.md
#    Gotcha 7; the trap that made this eight tensors rather than three is
#    AGENTS.md Gotcha 33.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_convention_patch --release -- --ignored --nocapture

# 6. The diagnostic that found the five QUANTIZED tensors on that axis,
#    which a BF16-only probe structurally cannot see. Recovers the
#    permutation outright where a tensor has one row per head.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_quant_probe --release -- --ignored --nocapture

# The memory oracle (see docs/BENCHMARKING.md). One target per model
# family: the footprint assertion is a whole-session peak, so two
# families in one process cannot each have a ceiling.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_memory_oracle --release -- --ignored --nocapture

# The quality gate (ROADMAP Phase Q; numbers in docs/BENCHMARKS.md). Split
# per family for the same one-model-per-process reason as the oracle.
# Takes about 80 seconds each (four arms: perplexity, greedy, sampled, and
# a constrained-working-set repeat at 8 expert-cache slots).
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_gate --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_quality_gate --release -- --ignored --nocapture

# Proof the gate above can see quantization damage: clone the install
# (APFS clonefile, original untouched), shift one quantization level in a
# strided subset of the routed experts, re-measure. About 30 seconds.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_sensitivity --release -- --ignored --nocapture

# The cross-engine half of Phase Q, and the one test here whose second
# step is NOT cargo: it dumps this port's full-vocab logits plus the exact
# token ids it walked (~275 MiB, ~30 s), then replays those IDS through
# mlx-lm and prints the KL. Needs the 14.6 GB reference checkpoint
# (`hf download mlx-community/gemma-4-26b-a4b-it-4bit --revision
# 0d77464eeb233a2da68ebf9d7dc4edaac7db956d`). mlx-lm runs in a uv
# ephemeral env, so it is never installed globally and never enters this
# workspace's dependency graph. TURBOSPARK_LOGIT_DUMP_COLD=1 skips the
# warmup walk and reproduces quality_gate's frozen perplexity exactly,
# which is the cross-check that the dump measures what the gate measures.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/turbospark \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with mlx-lm --with numpy scripts/kld.py /tmp/kld/turbospark

# The same shape against llama.cpp, on a GGUF-derived install and the GGUF
# bytes it was streamed from. This is what closed Phase G's last gate
# clause. Needs brew's llama.cpp (the driver compiles
# scripts/llamacpp_logits.c against its header) and a LOCAL copy of the
# 26.9 GB GGUF: llama.cpp cannot stream it the way the repack walk can, so
# unlike every other target here it is gated on disk rather than on time.
# Runs llama.cpp three times, and all three are needed to read the result:
# two shapes for the shape floor, two backends for the backend floor
# (AGENTS.md Gotcha 34 -- CPU is NOT interchangeable with Metal here).
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/gguf-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/gemma-4-26B-A4B-it-Q8_0.gguf /tmp/kld/gguf-warm

# Split-KV chunk-count sweep on the decode attention kernel. Needs no
# model install: it is the kernel alone at the real Gemma 4 shapes, and
# it reports speedup ratios rather than absolute times so it stays
# readable on a throttled machine. Takes seconds.
cargo test -p turbospark-gpu --test attention_chunk_bench --release -- --ignored --nocapture

# The server's real backend end to end: one model open, one non-streaming
# and one streaming request through the bound loopback server.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-server --test real_backend --release -- --ignored --nocapture
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
| `TURBOSPARK_GEMMA4_INSTALL_DIR` | `gemma4_checkpoint_network`, `memory_oracle`, `quality_gate`, `quality_sensitivity`, `logit_dump`, `real_backend`, `gguf_checkpoint_network`, `gguf_fused_gate_network` | Where the real `.gturbo` install lives. The oracle, the quality gate, the logit dump, and the server test skip (with a note) when unset; the repack test falls back to a temp dir. The two GGUF tests use the install as an independent REFERENCE rather than as a subject: `gguf_checkpoint_network` cross-checks names and `ArchConfig` against it and skips those two checks when unset, and `gguf_fused_gate_network` correlates against its expert weights and skips entirely. Both accept a leading `~/`. |
| `TURBOSPARK_QWEN36_INSTALL_DIR` | `qwen36_checkpoint_network`, `qwen36_memory_oracle`, `qwen36_quality_gate`, `logit_dump` | Where the repacked Qwen 3.6 install lives. The oracle and the quality gate skip (with a note) when unset; the repack test falls back to a temp dir. Deliberately a second variable rather than a generalized one, so both installs can coexist and each target asserts its own family's row. `logit_dump` takes it as a fallback when the Gemma variable is unset, though `scripts/kld.py`'s reference is pinned to the Gemma repo. |
| `TURBOSPARK_LOGIT_DUMP_DIR` | `logit_dump` | Where to write `logits.f16` and `meta.json` (~275 MiB on either family). Required: the target skips when unset, since a few hundred MB is not something to write to a default path. |
| `TURBOSPARK_LOGIT_DUMP_COLD=1` | `logit_dump` | Skips the warmup walk, so the dump comes off a COLD expert cache. That is the condition `quality_gate` takes its perplexity under, so this reproduces the frozen row exactly; without it the dump is warm and reads about 0.5% higher. How the warm/cold difference gets measured rather than assumed. |
| `MFERENCE_PHASES=1` | `turbospark-check` | Prints the per-phase decode breakdown (GPU wait, router readback, expert pread, routed bind, cache hit rate). |
| `MFERENCE_DISPATCH_PROFILE=1` | `turbospark-check`, `gpu::PassEncoder` | Ranks the individual dispatches inside each command buffer. Encodes one compute encoder per dispatch (Apple GPUs sample counters only at encoder boundaries) and waits on every buffer, so it perturbs the run it measures: a ranking aid, not a throughput number. Covered by `crates/gpu/tests/dispatch_profile.rs`. |
| `MFERENCE_SHARED_CB=0` | `RealForwardRunner` | Reverts the shared-expert branch to encoding after the expert pread instead of on its own overlapping command buffer. The A/B seam for any throughput claim. |
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
