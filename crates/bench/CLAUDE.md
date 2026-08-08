# mrefrust-bench

Throughput benchmark harness (`mference-bench`), mach memory sampler (`memory.rs`), frozen community benchmark protocol (`protocol.rs`), real model benchmark runner (`real_model.rs`), and memory oracle integration test (`tests/memory_oracle.rs`).

## Directory & File Structure

```
crates/bench/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library entry point (mrefrust_bench)
|   +-- main.rs             # Binary entry point (mference-bench)
|   +-- memory.rs           # Mach memory sampler for physical footprint tracking
|   +-- protocol.rs         # Frozen community benchmark protocol definitions
|   \-- real_model.rs       # Real model benchmark runner driving RealForwardRunner
+-- tests/
|   +-- logit_dump.rs       # Full-vocab logit dump for the cross-engine KLD (scripts/kld.py)
|   +-- memory_oracle.rs    # Memory oracle asserting peak footprint ceiling & steady state (Gemma 4)
|   +-- mference_bench.rs   # Benchmark harness integration smoke test
|   +-- oracle_common/      # Shared memory oracle assertion helpers (mod.rs)
|   +-- quality_common/     # Shared quality gate evaluation helpers (mod.rs)
|   +-- gguf_nondeterminism_probe.rs # Repeated warm greedy runs must produce ONE output
|   +-- quality_gate.rs     # Quality gate integration test (Gemma 4)
|   +-- quality_sensitivity.rs # Proof the gate sees quantization damage (Gemma 4)
|   +-- qwen36_memory_oracle.rs # Memory oracle for Qwen 3.6 family
|   \-- qwen36_quality_gate.rs  # Quality gate for Qwen 3.6 family
\-- prompts/
    +-- quality-v1/         # Quality gate reference prompt fixtures
    |   \-- assistant-reference.txt
    \-- real-generation-v1/ # Standardized benchmark protocol prompt fixtures
        +-- short-explanation.txt
        +-- medium-review.txt
        \-- long-synthesis.txt
```

## Key Modules

- `main.rs`: Handles CLI benchmark modes (scripted producer overhead vs real model protocol).
- `memory.rs`: Mach kernel task info sampler for tracking peak physical memory footprint (`phys_footprint`).
- `protocol.rs`: Frozen benchmark protocol case definitions and step evaluators.
- `real_model.rs`: Runs protocol cases against real `.gturbo` installs using `RealForwardRunner`.
- `tests/memory_oracle.rs`: Ignored test gated on `MREFRUST_GEMMA4_INSTALL_DIR` asserting per-chip memory ceilings and zero leak growth for Gemma 4.
- `tests/qwen36_memory_oracle.rs`: Ignored test gated on `MREFRUST_QWEN36_INSTALL_DIR` asserting per-chip memory ceilings and steady state for Qwen 3.6.
- `tests/quality_gate.rs` & `tests/qwen36_quality_gate.rs`: Quality gate evaluation verifying generated output metrics against reference fixtures.

## Development & Test Commands

```sh
# Run fast unit and integration tests for mrefrust-bench
cargo test -p mrefrust-bench

# Run scripted producer throughput benchmark
cargo run -p mrefrust-bench --bin mference-bench -- <tokenizer-dir>

# Run real install benchmark (macOS, release mode required)
cargo run --release -p mrefrust-bench --bin mference-bench -- --model ~/models/gemma4.gturbo

# One protocol case per process (the protocol's fresh-process leg, and what
# a cross-engine comparison needs since Swift's CLI launches once per case)
cargo run --release -p mrefrust-bench --bin mference-bench -- \
  --model ~/models/gemma4.gturbo --case short-explanation

# Vary the routed-expert cache size (allowed 8/16/24/32, default 16, the
# same set and default MferenceCLI takes). 32 buys ~15% decode and ~1.5 GB
# of footprint, so it leaves the ~2 GB working-set claim behind; every
# published number is measured at the default.
cargo run --release -p mrefrust-bench --bin mference-bench -- \
  --model ~/models/gemma4.gturbo --case short-explanation --expert-cache-slots 32

# Head-to-head against ../Mference's MferenceCLI, same install, interleaved
# arms, one fresh process per run. Results: docs/BENCHMARKS.md
scripts/parity.sh

# Bucket-level decode phase diff against the Swift engine (both print a
# split under MFERENCE_PHASES=1, but Swift's is decode-only and this
# port's divides by all forward passes -- hence the short-prompt rule
# baked into the script). Not a benchmark; an attribution aid.
scripts/phasediff.sh [pairs] [slots]

# Run memory oracle test for Gemma 4 (macOS, takes ~10 mins, requires model env var)
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture

# Run memory oracle test for Qwen 3.6 (macOS, separate process target)
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-bench --test qwen36_memory_oracle --release -- --ignored --nocapture

# Quality gate: reference-answer perplexity + frozen output digests
# (ROADMAP Phase Q). One target per family, ~1 min each.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test quality_gate --release -- --ignored --nocapture
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-bench --test qwen36_quality_gate --release -- --ignored --nocapture

# Cross-engine KLD against mlx-lm (Phase Q's last item). Step 1 dumps this
# port's full-vocab logits plus the token ids; step 2 replays those IDS
# through mlx-lm in a uv ephemeral env. Needs the 14.6 GB reference
# checkpoint. MREFRUST_LOGIT_DUMP_COLD=1 skips the warmup walk and
# reproduces quality_gate's frozen perplexity exactly.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
MREFRUST_LOGIT_DUMP_DIR=/tmp/kld/mrefrust \
  cargo test -p mrefrust-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with mlx-lm --with numpy scripts/kld.py /tmp/kld/mrefrust

# Sensitivity proof for the gate above: APFS-clone the install, shift one
# quantization level in a strided subset of the routed experts, re-measure.
# ~30 s. Response curve and detection floor in docs/BENCHMARKS.md.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test quality_sensitivity --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Footprint Accounting**: `phys_footprint` includes resident weight mapping (`mmap` pinned by Metal `newBufferWithBytesNoCopy`) + KV cache + expert slot capacity + process baseline.
2. **Cold GPU Benchmark Artifacts**: The first run after a build executes on a cold GPU at low DVFS clock states (up to 53% slower). Always discard at least one warmup run.
3. **Power Source & Cross-Session Ratios**: Thermal throttling and battery state (`pmset -g ps`) alter absolute tok/s. Always measure ratios back-to-back in the same session.
4. **Slot count is pinned, not defaulted-into**: `protocol::PROTOCOL_EXPERT_CACHE_SLOTS` (16) is what `docs/BENCHMARKS.md`, the memory oracle's per-chip rows, and Swift's own default all sit at. `--expert-cache-slots` exists so a Swift comparison can match a non-default setting, not so the protocol can drift; the oracle passes the constant explicitly for that reason. Output IS identical across slot counts on both families as of 2026-08-08 (routed slots dispatch in router rank; AGENTS.md Gotcha 27), so a slot-count change is a throughput axis only.
5. **The phase report does not account for a decode run**: `MFERENCE_PHASES=1` covers the inside of `produce` only; the sampler and detokenizer are outside it (AGENTS.md Gotcha 23). Subtract the phase total from the footer's `decode=` seconds before trusting a phase table as a full attribution.
6. **Perplexity is only meaningful on ASSISTANT-position tokens**: both supported checkpoints are instruction-tuned, and instruction tuning masks the loss on the prompt, so the model was never trained to predict user-turn text or the markup closing it. Measured on the real Gemma 4 install, teacher-forcing the PROMPT gave a mean NLL of 15.3 nats against a uniform bound of 12.5 (worse than guessing) while assistant-side tokens in the same sequence scored 0.000; a replay of the model's own greedy output agreed 39/40, so the measurement was sound and the corpus was not. `quality_common` therefore teacher-forces a fixed reference answer into the assistant slot and scores only that. Sibling note: digests used to be irreproducible from a cold expert cache (slots were ordered misses-first, permuting the phase-2 reduce order); slots now dispatch in router rank, so cold and warm agree. The warmup before each measured generation is kept for throughput, not for the digest.
7. **A cross-engine KL number is meaningless without a floor measured beside it.** `scripts/kld.py` runs mlx-lm TWICE on purpose: once token-by-token through a cache (the shape this port runs in, and the headline comparison) and once as a single batched pass. The divergence between those two holds the weights, the kernels, and the engine fixed and varies only the reduce shape, and at 4 bits it still costs 0.0352 mean nats and 4% of the argmaxes -- MORE than this port's 0.0264 against mlx-lm. Drop that second run and the headline has no scale to be read against. Two sibling traps. Do NOT carry `quality_common`'s assistant-slot-only rule over to the KL: that rule exists because SFT masks prompt loss, while a distribution comparison is valid at every position, so all 550 are used. And do NOT budget for an f16 storage floor: mlx-lm returns bfloat16, whose 8 mantissa bits are coarser than f16's 10 at these softcapped magnitudes, so the round trip through this port's dump width is lossless (measured 3.5e-22 nats) and mlx is the lower-precision side.
8. **Byte-identity under a constrained expert cache holds on BOTH families, and the gate asserts it rather than freezing a second digest**: the fourth arm reopens the install at 8 slots, repeats the greedy digest, and requires it to equal the 16-slot one. That assertion is the regression test for AGENTS.md Gotcha 27 -- it fails if a layer's routed slots are ever again dispatched in an order that depends on cache state rather than on the route. It did fail on Gemma before 2026-08-08, which is why the row used to carry a `constrained_digest` field; that field is gone.
