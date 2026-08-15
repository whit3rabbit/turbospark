# turbospark: High-Efficiency Apple Silicon Inference in Rust

[![CI](https://github.com/whit3rabbit/turbospark/actions/workflows/ci.yml/badge.svg)](https://github.com/whit3rabbit/turbospark/actions/workflows/ci.yml)
[![Release](https://github.com/whit3rabbit/turbospark/actions/workflows/release.yml/badge.svg)](https://github.com/whit3rabbit/turbospark/actions/workflows/release.yml)
[![crates.io](https://img.shields.io/crates/v/turbospark-cli.svg)](https://crates.io/crates/turbospark-cli)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20arm64-lightgrey.svg)](#)
[![MSRV](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](rust-toolchain.toml)

`turbospark` is a high-performance, behavior-compatible Rust inspired port of the [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) local LLM inference engine.

It is specifically designed for **Apple Silicon (macOS Metal)** to execute large language models (LLMs) with **extremely low memory overhead**. Instead of holding full model parameters in unified RAM/VRAM, `turbospark` streams routed expert weights directly from high-speed SSD storage into a lean working memory footprint.

This enables Mac users with limited memory (8 GB, 16 GB, 24 GB, or 36 GB) to run large models like **Gemma 4 26B-A4B** and **Qwen 3.6 35B-A3B** locally without exhausting system memory.

---

## Key Benefits for macOS / Apple Silicon Users

- **Extreme Memory Efficiency**: Runs large 26B-35B parameter Mixture-of-Experts (MoE) models using only **~1.6 GiB to 2.2 GiB of peak RAM/VRAM**. Users with 16 GB or 36 GB Macs no longer need 64 GB+ memory configurations to run 26B-35B models.
- **Direct GGUF Streaming Intake (New / WIP)**: Native intake for published GGUF formats (Gemma 4 Q8_0, Qwen 3.6 mixed Q4_K_M, and sub-4-bit IQ imatrix builds). Streams directly from Hugging Face or parses local GGUFs into optimized `.gturbo` format without requiring the 12-27 GB raw model payload to be loaded in RAM.
- **Sub-4-bit Option for the Tightest Budgets**: A published IQ3_XXS/IQ4_NL Gemma 4 build runs at **~1.8 GiB peak**, the leanest configuration here, verified against `llama.cpp` on the same bytes. Slower than INT4, and the tradeoff is spelled out below rather than buried.
- **Architecture Registry with Honest Refusals**: GGUF `general.architecture` and Hugging Face `model_type` strings resolve through one table, every key of which was read off a real published file. An architecture this port recognizes but cannot yet run says so, names the missing work, and points at the bring-up checklist, instead of failing as "unknown". Mixtral-style `llama` MoE checkpoints run, as do `qwen3moe` ones (Qwen3-30B-A3B) through the same decode flow; the dense half of the `llama` string is refused by name (it has no routed experts to stream).
- **Zero-Copy Metal Execution**: Utilizes zero-copy `MTLBuffer` memory mappings (`newBufferWithBytesNoCopy`) and native Metal compute shaders for high-throughput generation.
- **Low Memory Overhead vs standard MLX / LLM tools**: Standard MLX or llama.cpp setups load full weights into system memory (requiring 16 to 32+ GB RAM). `turbospark` streams expert layers on demand and caps physical memory usage tightly under ~2.2 GB for Gemma 4 and ~1.6 GB for Qwen 3.6.
- **Built-in OpenAI & Anthropic API Server**: Includes a local server providing OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) endpoints for drop-in integration with CLI tools (e.g., `claude-code`), Web UIs, and applications.

---

## Memory Footprint & Benchmark Parity

Everything below was measured on one machine: an **Apple M4 Max, 36 GB unified memory, on mains power**. Numbers do not transfer to other chips.

### What this actually buys you

A conventional runner keeps the whole model in RAM. This engine keeps only a small resident core in RAM and streams the mixture-of-experts weights off SSD on demand, so **the model on disk can be far bigger than the RAM it occupies while running**.

The plain version: on a 36 GB laptop, a 26B-parameter model that would normally need ~13 GB of RAM runs in about **2 GB**.

| Model | Model size (a normal runner such as mlx-lm or llama.cpp holds ~all of this in RAM) | RAM while generating | Speed | Power draw | Energy per token |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Gemma 4 26B-A4B** (INT4) | 13 GB | **~2.1 GB** | 35 - 41 tok/s | 17 W | 0.4 - 0.5 J |
| **Gemma 4 26B-A4B** (3-bit) | 12 GB | **~1.8 GB** | 23 - 25 tok/s | 27 W | ~1.0 J |
| **Qwen 3.6 35B-A3B** (INT4) | 18 GB | **~1.6 GB** | 33 - 38 tok/s | 14 W | ~0.4 J |
| **Qwen3-30B-A3B** (Q4_K_M) | 17 GB | ~2.7 GB | 16 - 25 tok/s | 21 W | ~0.8 - 1.4 J |
| **gpt-oss-20b** (MXFP4) | 11 GB | ~5.4 GB | 23 - 30 tok/s | 30 - 33 W | ~1.1 - 1.3 J |

**The first three rows are the point of the project**: a 26B model in ~2.1 GB and a 35B model in ~1.6 GB, against 13 GB and 18 GB on disk. The last two rows are honest counter-examples that still stream but land higher, and the reason is arithmetic rather than a defect - see the note on expert size below.

Dense models (no experts to stream) work too, but the memory story is different and the table above does not apply to them:

| Model | Size on disk | RAM while generating | Speed | Note |
| --- | ---: | ---: | ---: | --- |
| **Qwen3.8-27B** (INT4) | 14 GB | 660 MB counted | 17 - 19 tok/s | see caveat |
| **Mistral 7B** (Q4_K_M) | 4.1 GB | 1.2 GB counted | 16 - 30 tok/s | measured at 8k context |
| **Bonsai-27B** (1-bit) | 3.9 GB | not yet measured | ~18 tok/s | |

> **Caveat, and please read it before quoting the 660 MB.** Nothing streams in a dense model. That figure is what macOS *counts* against the process; the 14 GB of weights are memory-mapped and simply are not counted. You still need a machine that can hold and page them, so treat a dense model as needing roughly its **size on disk** in free RAM, not its counted footprint. The counted number is useful for spotting leaks, not for capacity planning.

**How to read the other columns.** "Power draw" is the engine's own CPU + GPU draw while generating, not the whole machine - the laptop as a whole measured roughly 50 - 70 W under load, most of the difference being the display. "Energy per word" is joules per generated token, so at ~0.4 J a thousand tokens costs about 400 J, roughly 0.1 Wh. Speed and power vary by prompt length; the ranges span three fixed benchmark prompts of increasing size. Power figures exist for five installs and are simply absent for the rest.

**Memory is compared at a fixed context window.** The MoE rows are at 4,096 tokens; gpt-oss runs at 8,192 and Mistral at 8,192, because their tokenizers or their reasoning output need it. A footprint number without its window is not comparable to another one - on a dense model the KV cache is most of what is being measured, and doubling the window roughly doubles the figure.

*Why the memory result is about EXPERT SIZE, not about MoE.* The expert cache is `slots x layers x expert_size`, so what matters is how finely the model splits. Gemma 4 has 128 experts of ~3.2 MiB and lands at 2.1 GB; Qwen3-30B-A3B has smaller experts (2.5 MiB) but 48 layers, so it lands at 2.7 GB; gpt-oss is deeper still and reaches 5.4 GB. A coarse mixture like Mixtral 8x7B has 8 experts of ~109 MiB and cannot stream usefully at any setting - it runs correctly here and is simply not what this engine is for. Compute the product before assuming a new model will be small.

*Why Qwen 3.6 beats Gemma 4 on memory despite being the larger model.* 30 of its 40 layers use gated-DeltaNet linear attention, which carries ~2 MiB of fixed recurrent state per layer instead of a KV cache that grows with context.

*On the 3-bit row.* It is the leanest Gemma 4 configuration and the slowest: about 15% less peak memory and 20% fewer expert bytes on disk, for roughly 35% of the decode speed and 2.6% worse perplexity. INT4 remains the default; choose 3-bit only when memory or disk is the binding constraint. Its quality is verified against `llama.cpp` on identical bytes rather than asserted.

### Parity with Swift Original (Gemma 4 26B-A4B)

| Metric | `turbospark` (Rust) | Swift Original | Notes |
| --- | ---: | ---: | --- |
| **Decode Speed** | 34.6 to 40.7 tok/s | 34.3 to 41.1 tok/s | Decode throughput within 1% parity |
| **Peak RAM Footprint** | **2,108 to 2,182 MiB** | 2,217 to 2,235 MiB | `turbospark` uses **2-5% less memory** |
| **Install Disk Size** | 14 GB | 14 GB | Identical disk model layout read by both |

Full benchmarks, quality verification, KL divergence vs `mlx-lm`, and power consumption measurements are available in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`docs/POWER_BASELINE.md`](docs/POWER_BASELINE.md).

---

## GGUF Intake & Custom `.gturbo` Format

### Why the Custom `.gturbo` Format?
Standard GGUF files store all tensors inside a single monolithic binary file (often 20 GB to 27 GB). Loading large GGUFs in traditional tools requires reading or mapping the entire multi-gigabyte file into system RAM.

`turbospark` uses the **`.gturbo`** model install directory format specified in upstream [`SYSTEM_DESIGN.md`](https://github.com/drumih/turbo-fieldfare/blob/main/docs/SYSTEM_DESIGN.md):
- **`resident_core.bin`**: Non-expert resident tensors (norms, linear attention states, router projections, embeddings) loaded once via zero-copy `MTLBuffer` memory mappings.
- **`packed_experts/`**: Transcoded MoE expert blobs organized for fast sequential `pread` streaming and LFU/LRU caching during token generation.

This separation is what enables `turbospark` to execute 26B-35B models in **~1.6 GiB to 2.2 GiB of peak physical memory** instead of 20+ GB.

For full binary layouts, header byte specifications, and streaming mechanics, see [`docs/GTURBO.md`](docs/GTURBO.md).

### Compatibility with Upstream `turbo-fieldfare`
`turbospark` is a 100% behavior-compatible Rust port of upstream [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) (Mference). `.gturbo` model directories produced by `turbospark-repack` can be executed interchangeably by both Swift `MferenceCLI` and Rust `turbospark-check`: the parity numbers in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) were produced by the Swift CLI opening this port's install unmodified.

### Streaming GGUF Intake Without Large RAM Allocation
`turbospark` includes a native GGUF intake engine in `crates/repack`:
1. **HTTP Range Streaming**: Streams raw GGUF files directly from Hugging Face layer-by-layer using HTTP range requests (`HttpRangeSource`).
2. **Zero Large Memory Allocation**: The 20-27 GB GGUF checkpoint is **never** fully downloaded to disk or loaded into RAM. `repack` reads header ranges, extracts layer tensors, transcodes resident core norms to BF16 and routers to INT8, and writes out the `.gturbo` directory.
3. **Execution Memory Stats**: Once repacked into `.gturbo`, inference runs under the exact same tight memory ceiling (**~1.6 GiB for Qwen 3.6 GGUF, ~2.2 GiB for Gemma 4 GGUF**).



---

## Supported Features & Models

### Supported Models

Six architecture families run end to end, each with a real decode flow rather than a config entry:

| Family | Example checkpoints | Shape |
| --- | --- | --- |
| **Gemma 4** | `gemma-4-26B-A4B-it` (MLX INT4, Q8_0 GGUF, sub-4-bit IQ GGUF) | Sliding-window + full attention, 128 streamed experts |
| **Qwen 3.6** (`qwen3_5_moe`) | `Qwen3.6-35B-A3B` (MLX INT4, Q4_K_M GGUF) | Gated-DeltaNet linear attention + 256 streamed experts |
| **Qwen 3.5/3.8 dense** (`qwen3_5`) | `Qwen/Qwen3.8-27B`, `prism-ml/Bonsai-27B-mlx-1bit` | Same hybrid attention, dense FFN. **Two checkpoints, one architecture** |
| **Qwen3-MoE** (`qwen3moe`) | `Qwen3-30B-A3B` | Plain GQA + 128 streamed experts |
| **Llama** (`llama`) | Mixtral 8x7B, Mistral 7B, TinyLlama 1.1B | Plain GQA; covers both the MoE and dense halves of one architecture string |
| **gpt-oss** (`gptOss`) | `gpt-oss-20b` MXFP4 | GQA with attention sinks, YaRN rope, Harmony reasoning channels |

Two things worth knowing. **A new checkpoint is usually not a new family**: `Qwen/Qwen3.8-27B` shipped in August 2026 and needed no engine change at all, because its architecture is identical to a checkpoint already supported - which is asserted by a test that parses both configs, not assumed. And **the family is chosen from the architecture string, never from tensor names**, because several of these families use identical tensor naming and picking the wrong flow produces fluent, wrong output rather than an error.

### Checkpoints & GGUF Support
- **GGUF Intake**: Native parsing and direct streaming intake for published GGUF checkpoints:
  - Gemma 4 Q8_0 GGUF (`ggml-org/gemma-4-26B-A4B-it-GGUF`).
  - Qwen 3.6 mixed Q4_K_M GGUF (Q4_K experts/embeddings, Q8_0 attention, Q6_K output).
  - Gemma 4 sub-4-bit imatrix GGUF (`unsloth/gemma-4-26B-A4B-it-GGUF` `UD-Q3_K_M`): IQ3_XXS routed gate/up over IQ4_NL down, Q6_K tied embedding.
  - Streams directly from Hugging Face without writing 12-27 GB checkpoint files to disk.
- **`.gturbo` Format**: High-speed packed expert layout optimized for sequential SSD streaming and mmap execution. Expert stride is per layer, so a checkpoint whose layers carry different block types is not padded to its widest one.

### Quantization Support
- **INT4 / INT8**: Affine quantized expert weights and resident core.
- **GGUF Block Quantizations**: Native GPU kernels for Q8_0, Q4_K, Q6_K, plus INT8/FP16 execution paths.
- **Sub-4-bit IQ Codebooks**: IQ3_XXS, IQ4_NL and IQ4_XS, with port-local Metal kernels validated against `llama.cpp` on identical bytes (0.0044 mean nats KL, 97.5% top-1, against a 0.0374 backend floor). Shrinks Gemma 4's expert table from 12 GiB to 9.6 GiB and its peak footprint to ~1.8 GiB. **A tradeoff, not a strict upgrade**: it costs ~2.6% perplexity and ~35% decode throughput, so INT4 remains the default. See the table above.
- **Per-Tensor Mixing**: Block type is resolved per tensor, and for routed experts per layer AND per phase, so a checkpoint that uses a different quantization for `gate`/`up` than for `down`, or for one layer than for the rest, executes as published.

### Server & Interfaces
- **Interactive REPL & CLI**: `turbospark-check` binary for interactive chat (`--chat`), raw prompt (`--prompt`), or JSON message history (`--messages-file`).
- **HTTP Server**: `turbospark-server` serving OpenAI Chat Completions (`/v1/chat/completions`), Anthropic Messages (`/v1/messages`), and `/v1/models`.
- **Configurable Expert Cache**: Adjust expert cache slot counts (8, 16, 24, 32) to tune performance vs memory footprint.

### Limitations & Out of Scope
- **Apple Silicon Acceleration Only**: Metal GPU acceleration requires macOS (`xcrun -sdk macosx metal`). On non-macOS platforms, crates compile CPU stubs.
- **Sequential Prompt Prefill**: Prompt tokens are processed sequentially per token (prefill tile kernels descoped; see [`DEVIATIONS.md`](DEVIATIONS.md)).
- **Q4_0 GGUF Quantization**: Refused at open until dedicated Q4_0 resident GEMV and embedding kernels land.
- **Sub-4-bit Costs Energy, Not Just Throughput**: The IQ path is a memory and disk win only (-15% peak footprint, -20% expert bytes). Measured on the power harness it draws roughly 2x the joules per decoded token of the INT4 install (codebook dequant nearly doubles GPU watts while running 35% slower). Use it when memory is the constraint; INT4 remains the default on every other axis. See [`docs/POWER_BASELINE.md`](docs/POWER_BASELINE.md).
- **No Speculative Decoding (DFlash)**: Measured before building, and it does not pay here yet. A batched verify of M tokens has to cost less, in decode-steps, than the tokens it gets accepted; on this engine 19% of decode compute is per-token work with no weights to amortize, so verify cost scales almost linearly in M. Against a trained DFlash drafter's published accept lengths that lands at **1.14x at block 4, 0.97x at block 8 and 0.87x at block 16** -- so small blocks win and large ones lose, inverting the datacenter result where verify is nearly free. About 1.1x is not worth the drafter, and the break-even column already assumes a batched MoE kernel that does not exist. The lever is that kernel rather than the drafter. Full arithmetic, the five measurement surfaces, and what would change the answer: [`docs/SPECULATIVE_DECODING.md`](docs/SPECULATIVE_DECODING.md).


---

## Architecture & Repository Layout

The workspace is organized into modular Rust crates:

- **`crates/core`**: Primitive types, token definitions, and `RuntimeConfig`.
- **`crates/compute`**: CPU reference kernels for math, quantization (Q8_0, Q4_K, INT4/INT8), norms, RoPE, and sampling.
- **`crates/gpu`**: macOS Metal shaders and execution pipeline dispatch (macOS only).
- **`crates/streaming`**: SSD streamer for routed expert weights with LFU/LRU caching.
- **`crates/model-io`**: Model manifest parsing, tensor indexes, and file verification.
- **`crates/repack`**: GGUF/Safetensors intake and transcode pipeline into `.gturbo`.
- **`crates/tokenizer`**: Fast tokenization, chat template application, streaming detokenizer, and tool-call parsing.
- **`crates/runtime`**: Core execution engine for prefill and decode loops.
- **`crates/server`**: Local OpenAI and Anthropic compatible HTTP server (`turbospark-server`).
- **`crates/cli`**: Command-line application binary (`turbospark-check`).
- **`crates/selection`**, **`crates/window-fit`**, **`crates/invocation`**: Context window management and candidate sampling.
- **`crates/bench`**: Throughput benchmark harness, memory oracle tests, and quality gate suite.

---

## Quick Start

### Install

```sh
# Homebrew (Apple Silicon; installs turbospark-check and turbospark-server)
brew install --cask whit3rabbit/tap/turbospark

# Or from crates.io
cargo install turbospark-cli turbospark-server
```

Both install the binaries only. Model installs (`.gturbo` directories) are built
separately by `crates/repack`; see [`docs/GTURBO.md`](docs/GTURBO.md).

### Build & Run Tests

```sh
# Build all workspace crates
cargo build --workspace --release

# Run workspace test suite
cargo test --workspace

# Check formatting and lints
cargo fmt --check
cargo clippy --workspace --tests
```

### Running the CLI

```sh
# Run interactive chat against a model install
cargo run --release -p turbospark-cli --bin turbospark-check -- \
  --model ~/models/gemma4.gturbo \
  --chat

# Run prompt via JSON messages file
cargo run --release -p turbospark-cli --bin turbospark-check -- \
  --model ~/models/gemma4.gturbo \
  --messages-file prompt.json
```

### Running the HTTP Server

```sh
# Start local OpenAI / Anthropic compatible HTTP server
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo
```

Point Anthropic-compatible clients (like `claude-code`) directly to loopback:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=unused claude
```

---

## Documentation

- [`docs/MODEL_FAMILY.md`](docs/MODEL_FAMILY.md): Supported model families, automatic detection, and parity matrix.
- [`docs/GTURBO.md`](docs/GTURBO.md): Comprehensive specification of the `.gturbo` installation format.
- [`DEVIATIONS.md`](DEVIATIONS.md): Wired features vs scaffolded scope.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): Benchmark results, quality gates, and parity analysis.
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md): Harness documentation and memory oracle details.
- [`docs/POWER_BASELINE.md`](docs/POWER_BASELINE.md): Power metrics (Watts, Joules/token).
- [`docs/SPECULATIVE_DECODING.md`](docs/SPECULATIVE_DECODING.md): DFlash and batched verify, measured marginal (~1.1x, small blocks only) and why it is not shipped.
- [`docs/EXPERT_ROUTING.md`](docs/EXPERT_ROUTING.md): Domain-restricted expert sets, measured negative.
- [`ROADMAP.md`](ROADMAP.md): Project roadmap and status.



---

## License

MIT

