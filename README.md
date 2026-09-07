<div align="center">

<img src="assets/TurboSpark_README_LOGO.png" alt="TurboSpark Logo" width="600" />

# turbospark: Low-Memory LLM Inference for Apple Silicon, in Rust

[![CI](https://github.com/whit3rabbit/turbospark/actions/workflows/ci.yml/badge.svg)](https://github.com/whit3rabbit/turbospark/actions/workflows/ci.yml)
[![Release](https://github.com/whit3rabbit/turbospark/actions/workflows/release.yml/badge.svg)](https://github.com/whit3rabbit/turbospark/actions/workflows/release.yml)
[![crates.io](https://img.shields.io/crates/v/turbospark-cli.svg)](https://crates.io/crates/turbospark-cli)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20arm64-lightgrey.svg)](#limitations--out-of-scope)
[![MSRV](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](rust-toolchain.toml)

</div>

`turbospark` is a behavior-compatible **Rust port** of [Mference](https://github.com/NeelM0906/Mference/tree/main), a Swift LLM inference engine for Apple Silicon, built to the design and philosophy published in [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare). It is the only pure Rust inference engine for Apple Silicon Metal, pairing high-performance Rust internals with native Swift bindings (`crates/ffi` + SwiftPM) for its macOS chat app (`swift/TurboSparkApp`).

### What Makes TurboSpark Special
What makes `turbospark` unique is its architecture based on the philosophy of [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) and [Mference](https://github.com/NeelM0906/Mference/tree/main): instead of holding full model parameters in unified RAM/VRAM, `turbospark` streams routed expert weights on demand directly from high-speed SSD storage into a lean working-memory slot cache. This enables Macs with limited unified memory (8 GB, 16 GB, 24 GB, or 36 GB) to load and run large Mixture-of-Experts (MoE) models like **Gemma 4 26B-A4B** and **Qwen 3.6 35B-A3B** locally in as little as **~1.6 GiB to 2.2 GiB of peak RAM** without exhausting system memory.

The port is tested against the original rather than assumed compatible. Decode throughput lands within 1% of the Swift engine on the same install, every family carries a memory oracle asserting a peak-footprint ceiling, every family but the dense `llama` one carries a frozen quality gate (teacher-forced perplexity plus output digests), and the numerics are cross-checked against `mlx-lm`, `llama.cpp`, and MLX on identical bytes. What the suite proves is in [`docs/TESTING.md`](docs/TESTING.md), and the frozen numbers are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### Looking for More Polished Alternatives?
If you are looking for more polished, general-purpose local LLM runners, GUI desktop applications, or MLX/Python serving frameworks, consider these established alternatives in the ecosystem:

- [Ollama](https://ollama.com/) - Popular CLI, background service, and API for running local models.
- [LM Studio](https://lmstudio.ai/) - Polished visual desktop application for discovering, downloading, and running LLMs locally.
- [omlx](https://github.com/jundot/omlx) - Fast MLX-based inference server and desktop client for Apple Silicon.
- [Unsloth](https://github.com/unslothai/unsloth) - Ultra-fast, memory-efficient LLM fine-tuning and inference framework.
- [MLX Studio](https://mlx.studio/) - Dedicated graphical interface and workspace for Apple MLX models.
- [mlxserve](https://mlxserve.com/) - High-performance MLX model serving engine.
- [RapidMLX](https://rapidmlx.com/) - Accelerated MLX inference framework and utilities.
- [vMLX](https://vmlx.net/) - Efficient local MLX inference interface and serving tool.
- [MTPLX](https://github.com/youssofal/MTPLX) - Multi-Token Prediction (MTP) speculative inference engine built on MLX.

> **Where the low-memory win comes from, and where it does not.** The ~2 GB
> figures are a property of MIXTURE-OF-EXPERTS models: their routed experts
> are streamed from SSD through a bounded slot cache instead of being held
> in RAM. Dense models (Mistral, TinyLlama, Qwen3.8-27B, Bonsai, Ornith-1.5 9B) run
> correctly on this engine, but they get none of that benefit: a dense
> token touches every weight once, so the whole mapping is wired for the
> GPU and the RAM requirement is the install's full size on disk. This was
> measured, not inferred (the mapping survives critical memory pressure
> untouched), and no quantization changes it -- a smaller quant only
> shrinks what gets wired, and "streaming layers" for a dense model is
> closed by arithmetic (a layer cache has a structural 0% hit rate). The
> measurements and the closures are in
> [`docs/DECODE_BUDGET.md`](docs/DECODE_BUDGET.md). Future model support
> therefore targets fine-grained MoE checkpoints, where one expert is
> small enough for the slot cache to hold the working set.

## Table of Contents

- [What Makes TurboSpark Special](#what-makes-turbospark-special)
- [Looking for More Polished Alternatives?](#looking-for-more-polished-alternatives)
- [In a hurry](#in-a-hurry)
- [Key Benefits for macOS / Apple Silicon Users](#key-benefits-for-macos--apple-silicon-users)
- [Memory Footprint & Benchmark Parity](#memory-footprint--benchmark-parity)
  - [What this actually buys you](#what-this-actually-buys-you)
  - [With this engine against without it](#with-this-engine-against-without-it)
  - [Does this help if the model is dense, and what about the 1-bit checkpoint](#does-this-help-if-the-model-is-dense-and-what-about-the-1-bit-checkpoint)
  - [Parity with the Swift Original (Gemma 4 26B-A4B)](#parity-with-the-swift-original-gemma-4-26b-a4b)
  - [Compared to slotstream, a Swift engine for the same model family](#compared-to-slotstream-a-swift-engine-for-the-same-model-family)
- [GGUF Intake & Custom `.gturbo` Format](#gguf-intake--custom-gturbo-format)
  - [Why the Custom `.gturbo` Format?](#why-the-custom-gturbo-format)
  - [Compatibility with Upstream `turbo-fieldfare`](#compatibility-with-upstream-turbo-fieldfare)
  - [Streaming GGUF Intake Without Large RAM Allocation](#streaming-gguf-intake-without-large-ram-allocation)
- [Supported Features & Models](#supported-features--models)
  - [Supported Models](#supported-models)
  - [Checkpoints & GGUF Support](#checkpoints--gguf-support)
  - [Quantization Support](#quantization-support)
  - [Server & Interfaces](#server--interfaces)
  - [Limitations & Out of Scope](#limitations--out-of-scope)
- [Architecture & Repository Layout](#architecture--repository-layout)
- [Getting Started](#getting-started)
  - [1. Install](#1-install)
    - [macOS Desktop App (TurboSpark.app)](#macos-desktop-app-turbosparkapp)
  - [2. Pull a model](#2-pull-a-model)
  - [3. Talk to it](#3-talk-to-it)
  - [4. Serve it](#4-serve-it)
  - [5. Point Claude Code at it](#5-point-claude-code-at-it)
  - [Installing something not in the catalog](#installing-something-not-in-the-catalog)
  - [Building from source](#building-from-source)
- [Swift Bindings & macOS App](#swift-bindings--macos-app)
- [Documentation](#documentation)
- [License](#license)

---

## In a hurry

```sh
brew install --cask whit3rabbit/tap/turbospark
turbospark-model pull qwen38-27b
turbospark-check --model qwen38-27b --chat
```

Then serve it to anything speaking the OpenAI or Anthropic API, Claude Code included:

```sh
turbospark-server --model qwen38-27b
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=unused \
  CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=true claude
```

The long version, including installing without Homebrew and choosing a different model, is under [Getting Started](#getting-started).

---

## Key Benefits for macOS / Apple Silicon Users

- **26B-35B Models in ~2 GB**: Runs large Mixture-of-Experts (MoE) models using only **~1.6 GiB to 2.2 GiB of peak RAM/VRAM**. Users with 16 GB or 36 GB Macs no longer need 64 GB+ memory configurations to run 26B-35B models. That band is the FLOOR, and the figure every published number here is measured at: the routed-expert cache at 16 slots per layer. The cache trades RAM for throughput. A machine with memory to spare climbs to 24 or 32 slots on its own (`--expert-cache-slots auto`, the default), trading roughly 1.5 GB of RAM for about 16% more decode throughput on this M4 Max, and the gain will vary by machine. `--expert-cache-slots 16` pins the band back for a machine without the headroom, which is never moved off it.
- **Live Directional Steering (Runtime Obliteration)**: Apply an abliteration-style directional edit to the residual stream: `ablate` a concept out, `add` it, `clamp` its coefficient, or `renorm` through the edit for graceful degradation. Toggle it on and off between two generations in the SAME process, with no weight byte ever written and no second copy of the model held. `--steering <vector.gguf>` (plus `--steering-mode`, `--steering-scale`, `--steering-layers START:END`, `--steering-target`, `--steering-gate`) works on both `turbospark-check` and `turbospark-server`. It reads the same `.gguf` control-vector layout `llama.cpp`/`repeng` write, so a published direction runs as-is, or extract your own from this engine's own captured activations (`scripts/extract_direction.py`). Wired for five of eight families: Qwen 3.6, the dense Qwen3.5/3.8 line, Mixtral, Qwen3-30B-A3B, and (since 2026-08-24) Gemma 4 including its chunked prefill driver. It composes losslessly with speculative decoding, and costs ~1.7% of decode throughput with every layer of a 64-layer model steered, ~0.75% over a 26-layer band. See [`docs/OBLITERATION.md`](docs/OBLITERATION.md).
- **Direct GGUF Streaming & Dynamic Quant Intake**: Native intake for published GGUF formats, including Gemma 4 Q8_0, Qwen 3.6 mixed Q4_K_M, and Unsloth dynamic imatrix builds (`UD-Q3_K_M` with IQ3_XXS / IQ4_NL). It streams layer-by-layer directly from Hugging Face, or parses local GGUFs into the optimized `.gturbo` format. Neither path requires 12-27 GB raw model downloads in RAM.
- **Ultra-Low 1-bit & 2-bit (Ternary) Quantization**: Dedicated GPU dequantization GEMV kernels for extreme low-bitwidth models. Runs 27B-parameter models in 3.9 GB (`Bonsai-27B` 1-bit) and 7.6 GB (`Ternary-Bonsai-27B` 2-bit).
- **TurboQuant KV-Cache Quantization**: A second, independent quantization axis from the weight formats above -- `--kv-bits off|2|3|3.5|4` shrinks the attention KV cache itself, on the eligible full-attention layers of any of the eight families, to trade a small quality cost for a smaller footprint at longer contexts. Ports mlx-vlm's `_TurboQuantMSECodec` (Lloyd-Max codebook, randomized Hadamard rotation) with a real fused Metal dequant kernel inside attention rather than a naive per-token unpack. Off by default everywhere; an install whose head dimension or layer layout does not qualify refuses the flag by name rather than silently running FP16. The macOS app defaults to Auto, which asks for 4-bit only on checkpoints it has already checked would accept it. Real-install memory and quality numbers are not yet measured. See [`docs/TRUBOQUANT.md`](docs/TRUBOQUANT.md).
- **Speculative Decoding with MTP & DFlash2 (Dynamic Sloth)**: Zero-overhead speculative decoding on dense architectures, using the checkpoint's own weights. Two drafters are wired: the native **multi-token-prediction (MTP) head** (step-wise), and the **DFlash2 (Dynamic Sloth v2) block drafter**, which proposes a full block in a single forward pass and verifies it in a batched pass. Configure either via `--speculative auto` and `--speculative-drafter auto|mtp|dflash`. Acceptance is mathematically exact at temperature 0. See [`docs/DFLASH2.md`](docs/DFLASH2.md) and [`docs/MTP_SPECULATIVE.md`](docs/MTP_SPECULATIVE.md).
- **Sub-4-bit Option for the Tightest Budgets**: A published IQ3_XXS/IQ4_NL Gemma 4 build runs at **~1.8 GiB peak**, the leanest streaming configuration here. It's verified against `llama.cpp` on the same bytes, and it's slower than INT4. The tradeoff is spelled out below rather than buried.
- **Chunked Prefill for Long Prompts**: Bounded GPU command-buffer execution via chunked prefill (`--prefill-chunk`). It prevents watchdog timeouts and manages prompt compute across the Gemma 4 and dense Llama/Mistral families. See [`docs/BATCHED_PREFILL.md`](docs/BATCHED_PREFILL.md).
- **Long Agent Runs That Finish ([SKILL.state](https://arxiv.org/abs/2608.26263), implemented)**: A tool-using agent normally re-sends its whole transcript every step, so on a 4,096-token window the run dies partway through. The macOS app can instead carry a compact JSON state the model patches each step, opt-in per project and off by default. Measured on three real installs at 50 tool steps: the transcript loop **stops between step 30 and 35**, the bounded one **finishes all 50 using 3-5x fewer tokens**, and the gap compounds with run length. Short runs are unaffected and should stay on the default. See [`docs/SKILL_STATE.md`](docs/SKILL_STATE.md).
- **Hardware-Aware Model Recommendations & Probe**: `turbospark-model recommend` inspects unified memory headroom against context window allocations to rank which models fit your Mac. `turbospark-model probe` reads Hugging Face repo headers in seconds, with no weights downloaded.
- **Architecture Registry with Honest Refusals**: GGUF `general.architecture` and Hugging Face `model_type` strings resolve through one table, every key of which was read off a real published file. An architecture this port recognizes but cannot yet run says so. It names the missing work and points at the bring-up checklist, instead of failing as "unknown". One string can cover two model shapes: `llama` covers Mixtral-style MoE and dense Mistral/Llama alike. Both halves run through the same decode flow, told apart by `num_experts` rather than by tensor names.
- **Zero-Copy Metal Execution**: The resident weights are mapped once and wrapped in a single `MTLBuffer` through `newBufferWithBytesNoCopy`, so the GPU reads them in place. Nothing is copied into a staging buffer per token.
- **Low Memory Overhead vs standard MLX / LLM tools, on mixture-of-experts models**: `mlx-lm` or `llama.cpp` keep full weights resident, so a 13 GB checkpoint wants roughly 13 GB. This engine streams routed experts on demand and holds under ~2.2 GB for Gemma 4 and ~1.6 GB for Qwen 3.6. The condition is load-bearing. A DENSE model has no experts to stream and gets none of this, and a coarse mixture like Mixtral gets none of it either. Both cases are in the comparison table below rather than left out of it.
- **98 Metal Compute Kernels & 12 Quant Formats**: A full custom Metal Shading Language (MSL) kernel suite. It covers 12 quantization formats (INT1, INT2 ternary, INT4, INT8, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ4_XS, IQ4_NL, and MXFP4). It also covers six compute primitives: Gated DeltaNet, split-KV attention, centered norms, directional steering, DFlash2 convolution, and vision transformer ops. See [`docs/KERNELS.md`](docs/KERNELS.md).
- **Built-in OpenAI & Anthropic API Server**: A local server providing OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) endpoints. Drop it in for CLI tools (e.g., `claude-code`), web UIs, and applications.
- **In-Process Forge Tool-Call Guardrails**: Active by default on the server (`--guardrails on|off`). It rescues malformed tool calls that the model emitted as raw text, validates JSON arguments against the request's own schema, and re-asks once with a targeted nudge. It runs in-process, with zero proxy latency and no heavy dependencies. See [`docs/FORGE_GUARDRAILS.md`](docs/FORGE_GUARDRAILS.md).
- **Native Swift & C ABI Bindings**: Embed `turbospark` directly into macOS apps (`crates/ffi` + SwiftPM package). You get async token streaming, non-blocking cancellation, reasoning channel extraction, and model catalog management, all in-process with no HTTP overhead. See [Swift Bindings](#swift-bindings) and [`docs/SWIFT_BINDINGS.md`](docs/SWIFT_BINDINGS.md).

---

## Memory Footprint & Benchmark Parity

Everything below was measured on one machine: an **Apple M4 Max, 36 GB unified memory, on mains power**. Numbers do not transfer to other chips.

### What this actually buys you

A conventional runner keeps the whole model in RAM. This engine keeps only a small resident core in RAM and streams the mixture-of-experts weights off SSD on demand, so **the model on disk can be far bigger than the RAM it occupies while running**.

The plain version: on a 36 GB laptop, a 26B-parameter model that would normally need ~13 GB of RAM runs in about **2 GB**.

| Model | Model size (a normal runner such as mlx-lm or llama.cpp holds ~all of this in RAM) | RAM while generating | Speed | Power draw | Energy per token |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Gemma 4 26B-A4B** (INT4) | 13 GB | **~2.1 GB** | 35 to 41 tok/s | 17 W | 0.4 to 0.5 J |
| **Gemma 4 26B-A4B** (3-bit) | 12 GB | **~1.8 GB** | 23 to 25 tok/s | 27 W | ~1.0 J |
| **Qwen 3.6 35B-A3B** (INT4) | 18 GB | **~1.6 GB** | 33 to 38 tok/s | 14 W | ~0.4 J |
| **Ornith-1.5 35B-A3B** (INT4) | 18 GB | **~1.6 GB** | 32 to 42 tok/s | 21 W (1 case) | ~0.47 J (1 case) |
| **Qwen3-30B-A3B** (Q4_K_M) | 17 GB | ~2.7 GB | 16 to 28 tok/s | 21 W | ~0.8 to 1.4 J |
| **gpt-oss-20b** (MXFP4) | 11 GB | ~5.4 GB | 23 to 30 tok/s | 29 to 33 W | ~1.1 to 1.3 J |
| **Qwen3.8-Flash-Next REAP-288** (INT4) | 68 GB | **~2.5 GB** | 6.9 to 11.2 tok/s | not measured | not measured |

**The first four rows are the point of the project**: a 26B model in ~2.1 GB and two 35B models in ~1.6 GB, against 13 GB and 18 GB on disk. The next two rows are honest counter-examples that still stream but land higher, and the reason is arithmetic rather than a defect: see the note on expert size below. **The last row is a different shape of counter-example**: an expert-pruned variant of a 125B-parameter architecture (288 of the original 512 experts per layer kept, top-10 routing unchanged) in ~2.5 GB, the largest model in this table by a wide margin, at this table's slowest speed and this repo's widest run-to-run spread (6.9 to 11.2 tok/s case to case) -- a 16-slot cache against 288 top-10-routed experts misses on most tokens, and which experts are already resident when a case starts moves its own hit rate more than it does for any other row here. It also runs at a 2,048-token context window rather than this table's usual 4,096, because the checkpoint's own query-sparse indexer is not yet implemented and this port refuses to run dense attention above the budget it was trained under. See [`docs/QWEN4_EXP.md`](docs/QWEN4_EXP.md) and the [slotstream comparison](#compared-to-slotstream-a-swift-engine-for-the-same-model-family) below.

Ornith-1.5 35B-A3B lands on Qwen 3.6's memory and speed because it *is* Qwen 3.6's architecture, retrained: its config derives the same internal baseline field for field, so it needed no new kernel and no new decode flow.

**Its two power cells are one case, not three**, where every other row in this table spans the protocol's three prompts. On that case it reads 21.0 W and 0.473 J/token against Qwen 3.6's 14.3 W and 0.351, about 35% more energy per generated token, while decoding *faster* (43.0 against 38.8 tok/s), and the gap is mostly CPU rather than GPU. Two weeks and one engine build separate the two measurements, so treat that as a difference worth chasing rather than a settled result: cross-session absolutes are the thing that has most often failed to reproduce on this hardware, which is why [`docs/POWER_BASELINE.md`](docs/POWER_BASELINE.md) now carries running the pair in one session as an open item.

Dense models (no experts to stream) work too, but the memory story is different and the table above does not apply to them:

| Model | Size on disk | RAM while generating | Speed | Note |
| --- | ---: | ---: | ---: | --- |
| **Qwen3.8-27B** (INT4) | 14 GB | 660 MB counted | 19 to 21 tok/s | see caveat |
| **Ornith-1.5 9B** (Q8_0) | 8.9 GB | 438 MB counted | 24 to 25 tok/s | same caveat |
| **Mistral 7B** (Q4_K_M) | 4.1 GB | 1.2 GB counted | 16 to 30 tok/s | measured at 8k context |
| **Bonsai-27B** (1-bit) | 3.9 GB | not yet measured | ~18 tok/s | |
| **Ternary-Bonsai-27B** (2-bit) | 7.6 GB | 660 MB counted | 13 to 14 tok/s | same caveat |
| **Muse Glimmer 30B** (INT4) | 15 GB | 535 MB counted | 13 to 15 tok/s | measured at 8k context, reasons before answering |

> **Caveat, and please read it before quoting the counted figures.** Nothing streams in a dense model. Those figures are what macOS *counts* against the process. The weights are memory-mapped and simply are not counted -- most starkly for Muse Glimmer, where 535 MB is counted against 15 GB of weights. You still need a machine that can hold and page them, so treat a dense model as needing roughly its **size on disk** in free RAM, not its counted footprint. The counted number is useful for spotting leaks, not for capacity planning.

### With this engine against without it

The saving comes from streaming routed experts, so it exists only where there are routed experts to stream.

The right-hand column is arithmetic rather than a measurement. A conventional runner keeps every weight resident, so it needs roughly the file size plus a KV cache. If a model is dense, both columns hold the same number and this engine buys nothing on memory.

| Install | Format | Bits | On disk | RAM here | A conventional runner needs | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next REAP-288 | MLX affine | 4 | 68 GB | **~2.5 GB** | ~68 GB | **~27x** |
| Qwen 3.6 35B-A3B | MLX affine | 4 | 18 GB | **1.6 GB** | ~18 GB | **11x** |
| Ornith-1.5 35B-A3B | MLX affine | 4 | 18 GB | **1.6 GB** | ~18 GB | **11x** |
| Gemma 4 26B-A4B | MLX affine | 4 | 13 GB | **2.1 GB** | ~13 GB | **6x** |
| Gemma 4 26B-A4B | GGUF IQ3_XXS/IQ4_NL | ~3 | 12 GB | **1.8 GB** | ~12 GB | **6.5x** |
| Qwen3-30B-A3B | GGUF Q4_K_M | 4 | 17 GB | **2.7 GB** | ~17 GB | **6x** |
| gpt-oss-20b | GGUF MXFP4 | 4 | 11 GB | **5.4 GB** | ~11 GB | **2x** |
| Mixtral 8x7B | GGUF Q4_K_M | 4 | 26 GB | runs, does not stream usefully | ~26 GB | **1x** |
| Qwen3.8-27B | MLX affine | 4 | 14 GB | ~14 GB | ~14 GB | **1x** |
| Ornith-1.5 9B | GGUF Q8_0 | 8 | 8.9 GB | ~8.9 GB | ~8.9 GB | **1x** |
| Ternary-Bonsai-27B | MLX affine | 2 | 7.6 GB | ~7.6 GB | ~7.6 GB | **1x** |
| Bonsai-27B | MLX affine | 1 | 3.9 GB | ~3.9 GB | ~3.9 GB | **1x** |
| Mistral 7B | GGUF Q4_K_M | 4 | 4.1 GB | ~4.1 GB | ~4.1 GB | **1x** |
| Muse Glimmer 30B | MLX affine | 4 | 15 GB | ~15 GB | ~15 GB | **1x** |

The six dense rows say 1x on purpose. Their *counted* footprints are 660 MB, 438 MB, 660 MB, not measured, 1.2 GB and 535 MB respectively, and quoting those as the RAM requirement would be wrong for the reason the caveat above gives. Muse Glimmer is the sharpest illustration: 535 MB counted against 15 GB of weights, because the resident mapping is not charged to the process and three quarters of its layers use a 2,048-token sliding window rather than the full context. Ratios are rounded, and the ones above 1x are the whole engineering claim of this project.

Mixtral is the interesting failure. It is a mixture of experts and it still gets no benefit, because the expert cache is `slots x layers x expert_size` and its 8 experts of ~109 MiB each want 54 GiB at the default 16 slots.

Fine-grained mixtures stream and coarse ones do not. Compute that product before assuming a new checkpoint will be small.

### Does this help if the model is dense, and what about the 1-bit checkpoint

On memory, no. A dense model has nothing to stream, so this engine and `mlx-lm` both need roughly the weights in RAM, and any table that claims otherwise is quoting a counter rather than a requirement.

For `prism-ml/Bonsai-27B-mlx-1bit` the advantage is different in kind, and it is narrow. **Upstream `mlx` refuses `bits=1` at the API level**, on every device, not merely on Metal: `mx.quantize` reports "The supported bits are 2, 3, 4, 5, 6 and 8".

Running that checkpoint under stock `mlx-lm` is therefore not slow, it is impossible, and the reference implementation has to be built from source (`github.com/PrismML-Eng/mlx@prism`). This engine reads it natively with a 1-bit GEMV and a 1-bit embedding lookup, so a working install needs no fork and no build.

That argument does not extend to two bits. Upstream `mlx` supports `bits=2`, and this project's own cross-engine check for the ternary checkpoint runs against stock `mlx` 0.32.0. If you want the ternary model and already have `mlx-lm` working, this engine offers you the server, the GGUF intake, and the frozen quality gates, and it does not offer you less memory.

**How to read the other columns.** "Power draw" is the engine's own CPU + GPU draw while generating, not the whole machine: the laptop as a whole measured roughly 50 to 70 W under load, most of the difference being the display. "Energy per token" is joules per generated token, so at ~0.4 J a thousand tokens costs about 400 J, roughly 0.1 Wh. Speed and power vary by prompt length, and the ranges span three fixed benchmark prompts of increasing size, except the Ornith row, whose power is one case and says so. Power figures exist for six installs and are simply absent for the rest.

A seventh, **Muse Glimmer 30B, is deliberately not in a table**: it draws ~38 W and saturates this laptop thermally within about two minutes, so every arm the harness measures is being driven by the thermal governor rather than by the workload, and their energy-per-token wanders 25% run to run. Its unconstrained cost is known (~2.02 J/token, reproduced across three sessions to 1.5%), but its sustained cost on this hardware is not knowable. The direction of that error is the trap worth knowing: a throttled run is slower AND cheaper per token, so it does not look broken in a power table, it looks good.

**Memory is compared at a fixed context window.** The MoE rows are at 4,096 tokens. gpt-oss, Mistral and Muse Glimmer run at 8,192, because their tokenizers or their reasoning output need it. A footprint number without its window is not comparable to another one: on a dense model the KV cache is most of what is being measured, and doubling the window roughly doubles the figure.

*Why the memory result is about EXPERT SIZE, not about MoE.* The expert cache is `slots x layers x expert_size`, so what matters is how finely the model splits. Gemma 4 has 128 experts of ~3.2 MiB and lands at 2.1 GB. Qwen3-30B-A3B has smaller experts (2.5 MiB) but 48 layers, so it lands at 2.7 GB, and gpt-oss is deeper still and reaches 5.4 GB.

A coarse mixture like Mixtral 8x7B has 8 experts of ~109 MiB and cannot stream usefully at any setting. It runs correctly here and is simply not what this engine is for. Compute the product before assuming a new model will be small.

*Why Qwen 3.6 beats Gemma 4 on memory despite being the larger model.* 30 of its 40 layers use gated-DeltaNet linear attention, which carries ~2 MiB of fixed recurrent state per layer instead of a KV cache that grows with context.

*On the 3-bit row.* It is the leanest Gemma 4 configuration and the slowest: about 15% less peak memory and 20% fewer expert bytes on disk, for roughly 35% of the decode speed and 2.6% worse perplexity. INT4 remains the default. Choose 3-bit only when memory or disk is the binding constraint. Its quality is verified against `llama.cpp` on identical bytes rather than asserted.

### Parity with the Swift Original (Gemma 4 26B-A4B)

Both engines opened the same install on the same machine. The Swift arm is [Mference](https://github.com/NeelM0906/Mference)'s own CLI, unmodified.

| Metric | `turbospark` (Rust) | Swift Original | Notes |
| --- | ---: | ---: | --- |
| **Decode Speed** | 34.6 to 40.7 tok/s | 34.3 to 41.1 tok/s | Decode throughput within 1% parity |
| **Peak RAM Footprint** | **2,108 to 2,182 MiB** | 2,217 to 2,235 MiB | `turbospark` uses **2-5% less memory**, but on one counter only, and peak RSS runs the other way |
| **Install Disk Size** | 14 GB | 14 GB | Identical disk model layout read by both |

The 2-5% is one machine, one install, and one counter. It is `phys_footprint` on an M4 Max, and this port's process covered two generations to the Swift CLI's one, which if anything favours Swift. Peak RSS runs the other way (1,991 to 1,993 MiB here against 1,682 to 1,831), and that counter moves with how much of the mapped install happens to be resident, so neither number alone settles the question. Do not read a 2% gap as an engineering result. Read it as evidence the two engines do the same work.

Full benchmarks, quality verification, KL divergence vs `mlx-lm`, and power consumption measurements are available in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`docs/POWER_BASELINE.md`](docs/POWER_BASELINE.md).

### Compared to slotstream, a Swift engine for the same model family

[slotstream](https://github.com/carloslfu/slotstream) is a Swift/MLX engine that streams **Qwen3.8-Flash-Next** experts from SSD, the same architecture family as this port's `qwen4exp` support. It is not a controlled A/B: different checkpoint, different chip, and neither project has published a same-machine side-by-side yet (slotstream's own README says as much about every engine it lists, this one included). Read the numbers below as two independent projects' own measurements, not a race result.

| Metric | This port (`qwen4exp`) | slotstream |
| --- | --- | --- |
| Checkpoint | `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`, expert-pruned to 288 of 512 experts per layer | `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit`, the full 512-expert checkpoint |
| On disk | 68 GB | 105 GB |
| Chip measured | Apple M4 Max, 36 GB | Apple M5 Pro, 48 GB |
| Peak memory | **~2.5 GB** (fixed 16-slot cache) | 8.1 GB floor, auto-sizes up to 33 GB on a 48 GB+ Mac |
| Decode | 6.9 to 11.2 tok/s | ~3 tok/s at its 8.1 GB floor, ~4 at 16 GB, ~12 at 33 GB (warm) |
| Context window | 2,048 tokens (checkpoint's own QSA indexer not yet implemented) | 32,768 tokens |
| Speculative decoding | not wired for this family yet (an ingestible head artifact exists but is unattached -- see below) | native MTP head, 1.24-1.33x measured |
| Prefix-KV reuse | implemented in the engine, not yet enabled for this family | measured, cuts an 8th-turn prefill from 25.8s to 6.0s |

Three things worth reading past the raw tok/s row before drawing a conclusion:

- **The checkpoints are not the same size for a reason that matters to the comparison, and it is coincidence rather than agreement that both read "68 GB".** REAP-288 prunes 224 of the 512 experts per layer before quantizing, so this engine's 16-slot cache is choosing among 288 candidates where slotstream's is choosing among 512 -- a smaller haystack per token, independent of anything either engine's cache policy does. slotstream's own README puts its *routed-expert* bytes alone at 68 GB, on top of a separate 32 GB n-gram table and 3.8 GB trunk (105 GB total). This project's install (`du -sh` on the actual `.gturbo` directory) is 36 GB of pruned routed experts, 30 GB of n-gram table, and 2.6 GB of trunk -- 68 GB total, but a DIFFERENT 68 GB: the n-gram table and trunk are unaffected by expert pruning and read almost the same size both projects measure (30 GB vs slotstream's 32, 2.6 GB vs slotstream's 3.8), while the pruned expert table (36 GB) is a little over half of slotstream's unpruned 68 GB, consistent with keeping 288 of 512 experts.
- **The memory strategies are different points on the tradeoff curve, not the same point measured twice.** slotstream auto-sizes toward whatever the machine can spare (its own example: 33 GB target, ~152 of 512 experts resident per layer, about 30% cache coverage) and treats a bigger cache as a bigger, faster machine. This port's expert cache defaults to a fixed 16 slots regardless of available RAM (`--expert-cache-slots` goes up to 128, but nothing has tuned this specific 288-expert routing profile past the default yet -- see [`docs/QWEN4_EXP.md`](docs/QWEN4_EXP.md)'s open item). So the fair reading of the tok/s row is "this port at ~2.5 GB lands above slotstream's 8 GB and 16 GB tiers, and in the neighborhood of its 24 GB one" -- a real result at a much smaller footprint, but against a checkpoint with 224 fewer experts to route among per token, which should make this port's number look better than a matched comparison of the same checkpoint would. The chip difference (M4 Max against M5 Pro) is a separate confound whose direction is not known here -- a Max-tier chip typically has more memory bandwidth than a same-generation Pro one, which is the resource this workload is bound by, but nobody has measured how that trades off against one generation's improvements.
- **slotstream is further along on two axes this port has not built for this family**: 16x the context window (32,768 against 2,048, gated on this port's missing QSA indexer above 2,048) and a working speculative-decode path for this exact checkpoint -- and the second is no longer blocked on a missing artifact, though it is still a real engine-side gap rather than a small one. `sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` (same publisher as this port's own checkpoint, verified 2026-09-04) repacks the base model's `mtp.*` tensors standalone and states compatibility with any expert count, its own install example naming this port's exact checkpoint -- but the head it repacks is a complete QSA-plus-512-expert-MoE decoder layer with its own hyper-connections, not a small drafter: wiring it needs a second repack path, a second independent expert cache, and (per this port's own Phase 0 notes) an unresolved question about whether it clears this engine's own rollback-cost bar at all. This port's *existing* MTP drafter is a different, unrelated artifact for the *dense* `qwen3_5` family; neither is wired to `qwen4exp` yet. Detail: [`docs/QWEN4_EXP.md`](docs/QWEN4_EXP.md)'s MTP section.

A same-machine, same-checkpoint run is the only way to settle the tok/s and memory questions properly; nobody has published one yet, including slotstream's own comparisons against the other engines it lists.

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
`turbospark` is a behavior-compatible Rust port of upstream [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) ([Mference](https://github.com/NeelM0906/Mference)). `.gturbo` model directories produced by `turbospark-repack` can be executed interchangeably by both the Swift `MferenceCLI` and the Rust `turbospark-check`: the parity numbers in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) were produced by the Swift CLI opening this port's install unmodified.

### Streaming GGUF Intake Without Large RAM Allocation
`turbospark` includes a native GGUF intake engine in `crates/repack`:
1. **HTTP Range Streaming**: Streams raw GGUF files directly from Hugging Face layer-by-layer using HTTP range requests (`HttpRangeSource`).
2. **Zero Large Memory Allocation**: The 20-27 GB GGUF checkpoint is **never** fully downloaded to disk or loaded into RAM. `repack` reads header ranges, extracts layer tensors, transcodes resident core norms to BF16 and routers to INT8, and writes out the `.gturbo` directory.
3. **Execution Memory Stats**: Once repacked into `.gturbo`, inference runs under the exact same tight memory ceiling (**~1.6 GiB for Qwen 3.6 GGUF, ~2.2 GiB for Gemma 4 GGUF**).



---

## Supported Features & Models

### Supported Models

Eight architecture families run end to end, each with a real decode flow rather than a config entry. Every checkpoint named below has been installed from its published bytes and generated text through `turbospark-check` on this machine. None is a projection from a config file. **Verified** or **unverified** marks whether that specific artifact also has a frozen quality-gate or memory-oracle row backing it, explained below the table.

| Family | Checkpoints that run today | Shape |
| --- | --- | --- |
| **Gemma 4** | `gemma-4-26B-A4B-it` at MLX INT4 (verified), Q8_0 GGUF (unverified), and sub-4-bit IQ GGUF (verified) | Sliding-window + full attention, 128 streamed experts |
| **Qwen 3.6** (`qwen36`) | `Qwen3.6-35B-A3B` at MLX INT4 (verified) and Q4_K_M GGUF (unverified), `Ornith-1.5-35B-A3B` at MLX INT4 (verified) | Gated-DeltaNet linear attention + 256 streamed experts |
| **Qwen 3.5/3.8 dense** (`qwen35`) | `Qwen/Qwen3.8-27B` (4-bit, verified), `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (2-bit, verified), `prism-ml/Bonsai-27B-mlx-1bit` (1-bit, unverified), `Ornith-1.5-9B` (Q8_0 GGUF, verified) | Same hybrid attention, dense FFN. **Four checkpoints, one architecture** |
| **Qwen3-MoE** (`qwen3moe`) | `Qwen3-30B-A3B` at Q4_K_M GGUF (verified) | Plain GQA + 128 streamed experts |
| **Qwen3.8-Flash-Next** (`qwen4exp`) | `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` at MLX INT4 (verified) | Hyper-connections, GDN + QSA-as-dense attention, fine-grained MoE (288 experts, sigmoid-gated); **safetensors only, no GGUF path**; `--max-context` capped at 2048 (no QSA indexer yet) |
| **Llama** (`llama`) | Mixtral 8x7B (unverified), Mistral 7B (unverified), TinyLlama 1.1B (unverified) | Plain GQA, one architecture string covering a MoE half and a dense half, both running |
| **gpt-oss** (`gptOss`) | `gpt-oss-20b` MXFP4 (verified) | GQA with attention sinks, YaRN rope, Harmony reasoning channels |
| **Muse Glimmer** (`museGlimmer`) | `Muse-Glimmer-30B` at MLX INT4 (verified) | Dense GQA, 3-sliding/1-full window, **NoPE on the full layers**, separate attention output gate, reasons before answering |

**Verified** means the catalog carries a frozen quality-gate and/or memory-oracle row for that exact artifact in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): a specific perplexity, output digest, or peak-footprint ceiling that a future change can redden. **Unverified** checkpoints installed and generated coherent text on this machine too, but nothing pins a number to them yet, for different reasons per row. Gemma 4's Q8_0 GGUF and Qwen 3.6's Q4_K_M GGUF simply have no gates written yet. Bonsai-27B (1-bit) has none either, and likely can't get one cheaply: upstream `mlx` refuses `bits=1` outright (see below), so there's no independent reference to check its perplexity against. Mistral 7B and TinyLlama 1.1B were brought up to prove the `llama` family's dense/MoE split rather than to freeze a number. Qwen3.8-Flash-Next now has both: a memory oracle (2,503-2,509 MiB peak) and a quality gate (reference-answer perplexity 8.7224, frozen digests, reproducing across three independent processes -- [`docs/QWEN4_EXP.md`](docs/QWEN4_EXP.md) has the determinism bug this took root-causing and fixing before either could be trusted).

Mixtral 8x7B is the deepest "unverified": it installs and decodes correctly but cannot stream its experts usefully at all (see the memory tables further down), so a quality gate for it would be measuring a configuration nobody would actually run. Treat an unverified row as "runs, unmeasured," not as "broken."

DeepSeek-V4-Flash is recognized and refused at open. Its kernels are unported, and the refusal names them rather than reporting the architecture as unknown.

Two things worth knowing. **A new checkpoint is usually not a new family**: `Qwen/Qwen3.8-27B` shipped in August 2026 and needed no engine change at all, because its architecture is identical to a checkpoint already supported, which is asserted by a test that parses every published config against one baseline rather than assumed. The ternary checkpoint went further and differs from `Bonsai-27B` in its `quantization` block alone. And **the family is chosen from the architecture string, never from tensor names**, because several of these families use identical tensor naming and picking the wrong flow produces fluent, wrong output rather than an error.

### Checkpoints & GGUF Support
- **GGUF Intake**: Native parsing and direct streaming intake for published GGUF checkpoints:
  - Gemma 4 Q8_0 GGUF (`ggml-org/gemma-4-26B-A4B-it-GGUF`).
  - Qwen 3.6 mixed Q4_K_M GGUF (Q4_K experts/embeddings, Q8_0 attention, Q6_K output).
  - Gemma 4 sub-4-bit imatrix GGUF (`unsloth/gemma-4-26B-A4B-it-GGUF` `UD-Q3_K_M`): IQ3_XXS routed gate/up over IQ4_NL down, Q6_K tied embedding.
  - Streams directly from Hugging Face without writing 12-27 GB checkpoint files to disk.
- **`.gturbo` Format**: High-speed packed expert layout optimized for sequential SSD streaming and mmap execution. Expert stride is per layer, so a checkpoint whose layers carry different block types is not padded to its widest one.

### Quantization Support

Two container families, twelve types, backed by **98 specialized Metal compute kernels**. Whether a type runs is decided per type and per ROLE, not per format: a kernel that decodes a weight matrix is not the same kernel as one that decodes a routed expert or an embedding row, and several types have only the ones their real checkpoint needed. For the complete kernel inventory, threadgroup layouts, and host dispatch mapping, see [`docs/KERNELS.md`](docs/KERNELS.md).

**MLX affine** (a packed plane beside per-group scales and biases). The bit width, the group size, and the companion dtype travel together and are checked as one shape, because the cross-products are combinations no published file has:

| Bits | Group | Companions | Roles | Example checkpoint |
| ---: | ---: | --- | --- | --- |
| 1 | 128 | FP16 | matrix, embedding | `prism-ml/Bonsai-27B-mlx-1bit` |
| 2 | 128 | FP16 | matrix, embedding | `prism-ml/Ternary-Bonsai-27B-mlx-2bit` |
| 4 | 64 | BF16 | matrix, embedding, routed experts | `gemma-4-26B-A4B-it` 4bit |
| 8 | 64 | BF16 | matrix, embedding, routed experts | routers, shared experts |

**GGUF block quants**, every row read off a real published file rather than off the format spec:

| Type | Bits | Matrix | Embedding | Routed gate/up | Routed down | Why it stops there |
| --- | ---: | :-: | :-: | :-: | :-: | --- |
| Q8_0 | 8 | yes | yes | yes | yes | full set |
| Q4_K | 4 | yes | yes | yes | yes | full set |
| Q6_K | 6 | yes | yes | no | yes | no real file puts it in gate/up |
| Q5_K | 5 | yes | no | no | no | Mixtral puts it on `attn_output` alone |
| IQ3_XXS | ~3 | yes | no | yes | no | the imatrix build puts it in gate/up |
| IQ4_XS | ~4 | yes | no | yes | no | same, on one layer |
| IQ4_NL | ~4 | yes | no | no | yes | the same file's `down` |
| MXFP4 | 4 | no | no | yes | yes | `gpt-oss` puts it in `ffn_*_exps` and nowhere else |
| Q4_0 | 4 | no | no | no | no | refused at open, by name |

If a checkpoint uses a type in a role that has no kernel, it passes the manifest gate and is refused at the dispatch with the tensor named, rather than decoding to something plausible and wrong. Widening either list means landing kernels, not editing a list.

- **Sub-4-bit IQ Codebooks**: validated against `llama.cpp` on identical bytes (0.0044 mean nats KL, 97.5% top-1, against a 0.0374 backend floor). Shrinks Gemma 4's expert table from 12 GiB to 9.6 GiB and its peak footprint to ~1.8 GiB. **A tradeoff, not a strict upgrade**: it costs ~2.6% perplexity and ~35% decode throughput, so INT4 remains the default. See the table above.
- **Per-Tensor Mixing**: Block type is resolved per tensor, and for routed experts per layer AND per phase, so a checkpoint that uses a different quantization for `gate`/`up` than for `down`, or for one layer than for the rest, executes as published.
- **KV-Cache Quantization (TurboQuant), a Separate Axis from All of the Above**: everything in the two tables above quantizes WEIGHTS. `--kv-bits off|2|3|3.5|4` instead quantizes the attention K/V cache at decode time, on any of the eight families, and composes with whatever weight format the checkpoint already uses. Ported from mlx-vlm's `_TurboQuantMSECodec`: a per-row norm, a randomized Hadamard rotation (one fixed sign vector per K or V, for the whole model), then a 1-D Lloyd-Max codebook fit to the rotated coordinate's own density. A real fused Metal kernel dequantizes inside the attention pass itself rather than as a separate step. Off by default everywhere -- every existing memory-oracle and quality-gate row is unaffected by this option's mere existence -- and an install whose head dimension is not a power of two in 32..512, or whose layer layout leaves nothing eligible, refuses the flag by name at open rather than silently falling back to FP16. See [`docs/TRUBOQUANT.md`](docs/TRUBOQUANT.md).

### Server & Interfaces
- **Interactive REPL & CLI**: `turbospark-check` binary for interactive chat (`--chat`), raw prompt (`--prompt`), or JSON message history (`--messages-file`).
- **HTTP Server**: `turbospark-server` serving OpenAI Chat Completions (`/v1/chat/completions`), Anthropic Messages (`/v1/messages`), and `/v1/models`.
- **Configurable Expert Cache**: Adjust expert cache slot counts (8, 16, 24, 32) to tune performance vs memory footprint.
- **Tool Calling, With Guardrails On By Default**: Both endpoints render a request's `tools` through the checkpoint's own chat template and return calls as OpenAI `tool_calls` / Anthropic `tool_use`. On top of that, a reliability layer aimed at what a SMALL local model actually gets wrong: a call emitted in a dialect the checkpoint's own template did not teach it (bare JSON, Qwen XML, Mistral `[TOOL_CALLS]`) is **rescued** out of the raw text instead of reaching the client as prose, a call's arguments are **validated** against the schema the request itself sent, and a failure is **re-asked once** with a corrective nudge. All of it runs **in process against the local model**: the pure half of [`forge-guardrails`](https://crates.io/crates/forge-guardrails) is compiled in (3 added crates, no network crate among them), there is no proxy, no second process, and no outbound call of any kind. One behaviour change to know: a request carrying tools is buffered rather than streamed while guardrails are on, because a verdict needs the whole turn. Requests without tools stream exactly as before. `--guardrails off` restores the previous path. See [`docs/FORGE_GUARDRAILS.md`](docs/FORGE_GUARDRAILS.md).

- **Bounded-State Agent Loop (SKILL.state), Implemented and Opt-In**: An agent that runs many tool steps normally re-sends its whole transcript every step, so the prompt grows without bound and the run dies when it outgrows the context window. [SKILL.state](https://arxiv.org/abs/2608.26263) replaces that transcript with a compact JSON execution state the model patches each step. **Implemented in the macOS app** as a per-project toggle, default off, so the existing loop is byte-identical when it is unset. Measured here on three real installs at 50 tool steps each: the append-only loop **stops between step 30 and 35** with `prompt + max_new exceeds max_context`, while the bounded loop **finishes all 50 at 3-5x fewer tokens**, and the gap compounds (1.5x at ten steps, 5.1x at forty). Two findings the paper did not predict: no local install produced a single JSON syntax error, and the paper's dominant failure class (premature state overwrites, 68% of its open-weight errors) never occurred, so the grammar-constrained decoding it recommends turned out not to be needed. It saves no memory and moves no benchmark row. Measurement, scale caveats, and what would reverse the verdict: [`docs/SKILL_STATE.md`](docs/SKILL_STATE.md).

### Limitations & Out of Scope
- **Apple Silicon Acceleration Only**: Metal GPU acceleration requires macOS (`xcrun -sdk macosx metal`). On non-macOS platforms, crates compile CPU stubs.
- **Sequential Prompt Prefill**: Prompt tokens are processed sequentially per token (prefill tile kernels descoped, see [`DEVIATIONS.md`](DEVIATIONS.md)).
- **Q4_0 GGUF Quantization**: Refused at open until dedicated Q4_0 resident GEMV and embedding kernels land.
- **Sub-4-bit Costs Energy, Not Just Throughput**: The IQ path is a memory and disk win only (-15% peak footprint, -20% expert bytes). Measured on the power harness it draws roughly 2x the joules per decoded token of the INT4 install (codebook dequant nearly doubles GPU watts while running 35% slower). Use it when memory is the constraint. INT4 remains the default on every other axis. See [`docs/POWER_BASELINE.md`](docs/POWER_BASELINE.md).
- **No Speculative Decoding on the MoE Families**: Measured before building, and it does not pay there. A batched verify of M tokens has to cost less, in decode-steps, than the tokens it gets accepted. On this engine 19% of decode compute is per-token work with no weights to amortize, so verify cost scales almost linearly in M. Against a trained DFlash drafter's published accept lengths that lands at **1.14x at block 4, 0.95x at block 8 and 0.78x at block 16**, so small blocks win and large ones lose, inverting the datacenter result where verify is nearly free. About 1.1x is not worth the drafter, and the break-even column already assumes a batched MoE kernel that does not exist. The lever is that kernel rather than the drafter. Full arithmetic, the five measurement surfaces, and what would change the answer: [`docs/SPECULATIVE_DECODING.md`](docs/SPECULATIVE_DECODING.md). **That verdict is the MoE family's.** The DENSE `qwen3_5` family has no expert-union term and a much smaller un-amortizable floor, and there a checkpoint's own drafter does pay: a native MTP head measures 1.44x-1.66x ([`docs/MTP_SPECULATIVE.md`](docs/MTP_SPECULATIVE.md)) and the DFlash2 block drafter accepts 7.09 of 8 proposals per round ([`docs/DFLASH2.md`](docs/DFLASH2.md)). Cost a drafter per family, not per engine.


---

## Architecture & Repository Layout

The workspace is organized into modular Rust crates:

- **`crates/core`**: Primitive types, token definitions, allowed runtime-knob sets, and the directional-steering mode.
- **`crates/compute`**: CPU reference kernels for math, quantization (Q8_0, Q4_K, INT4/INT8), norms, RoPE, and sampling.
- **`crates/gpu`**: macOS Metal shaders and execution pipeline dispatch (macOS only).
- **`crates/streaming`**: SSD streamer for routed expert weights with LFU/LRU caching.
- **`crates/model-io`**: Model manifest parsing, tensor indexes, and file verification.
- **`crates/repack`**: GGUF/Safetensors intake and transcode pipeline into `.gturbo`.
- **`crates/catalog`**: The curated model table, the header-only Hugging Face probe, the fit-and-evidence recommendation engine, and the install driver behind `turbospark-model`.
- **`crates/tokenizer`**: Fast tokenization, chat template application, streaming detokenizer, and tool-call parsing.
- **`crates/runtime`**: Core execution engine for prefill and decode loops.
- **`crates/server`**: Local OpenAI and Anthropic compatible HTTP server (`turbospark-server`).
- **`crates/cli`**: Command-line application binaries (`turbospark-check`, `turbospark-model`).
- **`crates/selection`**, **`crates/window-fit`**, **`crates/invocation`**: Context window management and candidate sampling.
- **`crates/bench`**: Throughput benchmark harness, memory oracle tests, and quality gate suite.
- **`crates/ffi`**: C ABI over the engine for the Swift/native GUI host (staticlib + `turbospark.h`).
- **`crates/vision-io`**: Portable vision preprocessing (decode, resize, patchify, position tables).

---

## Getting Started

Four steps, start to finish: install the binaries, pull a model, talk to it, then serve it to your own tools. Needs an Apple Silicon Mac on macOS. The walkthrough uses **Qwen3.8-27B**. See the note under step 2 for why you might want a different one.

### 1. Install

Three ways in. Homebrew is the shortest, and all three put the same three binaries on `PATH`: `turbospark-check` (generate), `turbospark-model` (find and install models), and `turbospark-server` (HTTP).

```sh
# Homebrew. The desktop app in /Applications, plus all three binaries.
brew install --cask whit3rabbit/tap/turbospark

# ...or the command-line tools alone, no app.
brew install --cask whit3rabbit/tap/turbospark-cli
```

Either cask uninstalls in one step, app and commands together. Add `--zap` to also remove the app's settings and chat archive. Neither touches your models under `~/.turbospark`.

```sh
brew uninstall --cask turbospark
```

```sh
# crates.io. `turbospark-cli` carries turbospark-check AND turbospark-model.
cargo install turbospark-cli turbospark-server
```

Or grab the [latest release](https://github.com/whit3rabbit/turbospark/releases) directly. The CLI asset is one zip per version, built for `aarch64-apple-darwin`, with a `SHA256SUMS` beside it:

```sh
VER=0.1.0
curl -LO "https://github.com/whit3rabbit/turbospark/releases/download/v${VER}/turbospark-${VER}-macos-arm64.zip"
unzip "turbospark-${VER}-macos-arm64.zip" -d ~/bin

# macOS quarantines anything downloaded by a browser or curl. Without this
# the first run dies with "cannot be opened because the developer cannot be
# verified" rather than anything about turbospark.
xattr -d com.apple.quarantine ~/bin/turbospark-* 2>/dev/null
```

#### macOS Desktop App (TurboSpark.app)

`brew install --cask whit3rabbit/tap/turbospark` above installs the app. To do it by hand, every release also carries a `.dmg` for Apple Silicon on macOS 14 (Sonoma) or later, on the [Releases](https://github.com/whit3rabbit/turbospark/releases) page:

1. Download `TurboSpark-<version>-arm64.dmg`.
2. Open the DMG and drag `TurboSpark.app` to `/Applications`.
3. Clear the quarantine flag. **The app is ad-hoc signed and not notarized**, so Gatekeeper refuses to open it until you do, however you installed it:
   ```sh
   xattr -dr com.apple.quarantine /Applications/TurboSpark.app
   ```
   Or right-click (Control-click) `TurboSpark.app` in Finder and choose **Open**. Installing through Homebrew, `brew install --cask --no-quarantine whit3rabbit/tap/turbospark` avoids the step entirely.

The app carries the same three command-line binaries inside its bundle, and the cask links them onto `PATH` from there, so installing the app is a superset of installing the CLI. That is why the two casks conflict: pick one.

However you installed it, this should now print the model catalog:

```sh
turbospark-model list
```

And this ranks it for the machine you are sitting at:

```sh
turbospark-model recommend
```

Two size columns, because they fail differently: **ALLOCS** is what the engine
allocates up front (expert-cache slots plus KV) and blowing that is a failed
open, while **ON DISK** is the whole install -- which for a mixture-of-experts
model *streams*, so it need not fit at all. A 13 GB install runs on a 16 GB
machine. A 27 GB one can still be refused, on its slot cache rather than its
size. Rows that have been through a memory oracle here quote what it measured.
Rows that have not say so, rather than guessing.

Add `--discover` to rank the
most-downloaded GGUF repositories on Hugging Face through the same gates.

### 2. Pull a model

```sh
turbospark-model pull qwen38-27b
```

That streams `mlx-community/Qwen3.8-27B-4bit` from Hugging Face in ranges and writes a `.gturbo` install to `~/.turbospark/models/qwen38-27b.gturbo`. About 15 GiB moves over the network and ~14 GiB lands on disk. The original checkpoint is **never written to disk whole**. Budget 20 minutes on a fast connection.

Two things worth knowing before you start it:

- **A pull cannot resume.** One that dies 12 GB in starts again from zero. The command says so before it begins.
- **Qwen3.8-27B is dense, so it does not stream.** It needs roughly its 14 GiB of weights in memory while running, and it is the walkthrough model because it is well-behaved and current, not because it is the low-memory demo. If your machine is tight on RAM, or you want the thing this project is actually for, pull `gemma4` instead: a 26B mixture-of-experts model that runs in **~2.1 GB** because its routed experts stream off SSD. Same commands from here on, with `gemma4` in place of `qwen38-27b`.

```sh
turbospark-model list             # the whole table, and what each row's numbers are backed by
turbospark-model recommend        # ...ranked for this machine, with the fit arithmetic
turbospark-model info qwen38-27b  # one row in full
turbospark-model path qwen38-27b  # where it landed
```

Every catalog row names a repository and a revision that were actually streamed and run on this port, and a `verified` status means a frozen quality-gate or memory-oracle row in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) asserts something about that exact artifact.

### 3. Talk to it

`--model` takes a catalog alias or a path. An existing directory always wins over an alias, so nothing that used to work changes.

```sh
# Interactive REPL.
turbospark-check --model qwen38-27b --chat
```

```sh
# Or one shot, through the checkpoint's own chat template.
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json
turbospark-check --model qwen38-27b --messages-file /tmp/p.json
```

Use `--messages-file` or `--chat` rather than `--prompt` on an instruction-tuned model. `--prompt` sends raw text with no chat framing, which makes these checkpoints babble. That's the template missing, not a decode bug.

```sh
# Ask the model to think first. Default is off.
turbospark-check --model qwen38-27b --messages-file /tmp/p.json --reasoning xhigh
```

`--reasoning off|low|medium|high|xhigh` is rendered into the prompt by the checkpoint's own chat template, so the levels a model accepts are its own: Qwen 3.8 takes `xhigh`/`medium`/`low`, gpt-oss and Muse Glimmer take `high`/`medium`/`low`, and a checkpoint whose template has only an on/off switch (Gemma 4, the Qwen3.5-era 27Bs) turns thinking on and says so. Where the reasoning can be separated from the answer, the **answer goes to stdout and the reasoning to stderr**, so `2>/dev/null` leaves you the answer alone. Servers take the same thing as OpenAI's `reasoning_effort` field on either endpoint, and hand the reasoning back as `reasoning_content` / an Anthropic `thinking` block.

The default is `off` rather than whatever the vendor advertises, and the difference is worth knowing: a model card's "reasons at xhigh by default" describes what you get by sending no setting at all, which is not what this engine has ever sent.

### 4. Serve it

```sh
turbospark-server --model qwen38-27b
```

Loopback on port 8080, serving OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`. One runner per process, so requests are answered one at a time. `--model` takes an alias or a directory here exactly as it does for `turbospark-check`, and the startup line prints which directory an alias resolved to. In-process tool-call guardrails (`--guardrails on|off`, default `on`) automatically rescue malformed tool calls, validate arguments against the schema, and retry once with a nudge ([`docs/FORGE_GUARDRAILS.md`](docs/FORGE_GUARDRAILS.md)). In a script that would rather fail early than serve the wrong thing, `turbospark-model path <alias>` prints the install directory and exits non-zero if the model is not installed.

```sh
curl -s localhost:8080/v1/models | python3 -m json.tool
```

### 5. Point Claude Code at it

The Anthropic endpoint is native, so there is no proxy in between:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_API_KEY=unused \
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=true \
  claude
```

`ANTHROPIC_API_KEY` is required by the client and ignored by the server, which has **no authentication and no TLS**: it is a loopback service. The model discovery flag makes Claude Code ask `/v1/models` instead of assuming Anthropic's hosted names. The server advertises one id, the install directory's own name (`qwen38-27b.gturbo` here). Anything else speaking either API works the same way, e.g. `OPENAI_BASE_URL=http://127.0.0.1:8080/v1`.

To reach it from another machine on your Tailnet, add `--bind tailnet`. That binds this machine's Tailscale IPv4 address, and the Tailnet ACL is then the only access control there is.

### Installing something not in the catalog

Ask before downloading. `probe` reads headers only, costs KB and seconds, and answers whether a checkpoint would run before a byte of it moves:

```sh
turbospark-model probe Qwen/Qwen3-30B-A3B-GGUF \
  --file Qwen3-30B-A3B-Q4_K_M.gguf --sidecar-repo Qwen/Qwen3-30B-A3B

turbospark-model pull --repo owner/name --alias mine --sidecar-repo owner/original
```

It reports the architecture verdict (naming what an unported one would need), each block type against the kernels that exist, which tokenizer sidecars the repository actually has, and the expert-slot arithmetic that decides whether a mixture-of-experts model fits here at all. That last one is worth reading before any large download: Mixtral 8x7B installs and decodes correctly and wants 54.5 GiB of pinned expert cache, because what this engine can hold is decided by how finely a model splits its experts rather than by its size.

`probe` exits 0 only on a runnable verdict, so `probe X && pull --repo X ...` works. Full guide, including how a pull orders its steps and how to add a catalog row: [`docs/MODELS.md`](docs/MODELS.md). Format details: [`docs/GTURBO.md`](docs/GTURBO.md).

Each checkpoint also still has its own `crates/repack` integration-test target, listed with its environment variable in [`AGENTS.md`](AGENTS.md). Those are what the catalog rows were built from.

### Building from source

```sh
# Build all workspace crates
cargo build --workspace --release

# Run workspace test suite
cargo test --workspace

# Check formatting and lints
cargo fmt --check
cargo clippy --workspace --tests
```

The binaries land in `target/release/`, and `cargo run -p turbospark-cli --bin turbospark-check -- ...` works in place of an installed `turbospark-check` throughout the walkthrough above.

Setting the tree up for the first time, or building the macOS app alongside the engine, is [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md): prerequisites, the order the two halves build in, the test and run loop, and what CI does not cover.

The workspace suite runs on any platform and covers the structural contracts. The heavier proof is env-gated and opt-in, because it needs a real model install: per-family quality gates freeze teacher-forced perplexity plus greedy and sampled output digests, memory oracles assert peak footprint against a per-chip ceiling and re-run a warm case to catch growth, and a determinism probe runs one greedy generation six times and requires exactly one distinct output.

The quality gate is calibrated rather than decorative. Shifting one quantization level in 0.0122% of Gemma 4's expert bytes moves its perplexity +10.5%, so the gate sees damage far below what reads as coherent by eye. Gating conventions and test-writing rules are in [`docs/TESTING.md`](docs/TESTING.md).

---

## Swift Bindings & macOS App

A C ABI (`crates/ffi`) and a SwiftPM package over it, so a native macOS app
drives the engine in-process rather than over HTTP. `swift/TurboSparkApp` is
a full-featured SwiftUI chat and model management desktop app. See
[`swift/README.md`](swift/README.md) for package and build details.

```bash
make swift-lib                                    # build the staticlib + stage the header
make swift-app-build                              # build the SwiftUI app (debug)
make swift-app-release                            # build the SwiftUI app (release)
make swift-app                                    # run the SwiftUI app
make swift-test                                   # ABI checks, no model needed
make swift-test-real MODEL=~/models/gemma4.gturbo # end to end against a real install
make clean-swift                                  # clean Swift artifacts and staged headers
```

```swift
let session = try await TurboSparkSession(modelPath: "~/models/gemma4.gturbo")

for try await event in session.generate([ChatMessage(role: .user, content: "Hello")]) {
    switch event {
    case .prefill(let done, let total): print("reading prompt \(done)/\(total)")
    case .content(let text):            print(text, terminator: "")
    case .reasoning(let text):          print("thinking: \(text)")
    case .finished(let result):         print("\n\(result.newTokens) tokens, \(result.stopReason)")
    }
}

session.cancel()   // safe from any thread, never blocks
```

Three things about the shape are worth knowing before building on it.

**`cancel()` never blocks.** Generation holds the engine lock for a whole
turn, so the cancel flag deliberately lives outside it. Put it inside and a
Stop button only takes effect once the model has finished on its own, which
a user experiences as a frozen window rather than as a bug. Cancelling is not
an error: the turn returns normally with `stopReason == .cancelled`, the
partial text intact, and a KV cache that describes itself honestly, so the
conversation continues from where it stopped.

**Options and results are JSON across the boundary**, which is what keeps
`turbospark.h` to 27 functions and makes adding a knob something
other than an ABI break. The per-token path carries no JSON: it is a pointer
and a length.

**Reasoning arrives separately from the reply.** `.content` is the assistant
turn to keep. `.reasoning` is for display only, because the checkpoints that
produce it drop prior-turn thinking from their own history, and replaying it
sends the model something it was never trained to read.

Model management (`TurboSparkCatalog`) needs no Metal device, so its C ABI
calls work on any platform, including ones that cannot then run a model. The
Swift package itself targets macOS 13+ only. Installs stream gigabytes and
**cannot resume**, so tell the user before starting rather than after failing.

The full API, the C ABI for non-Swift hosts, the threading contract, and the
list of what is deliberately not supported are in
[`docs/SWIFT_BINDINGS.md`](docs/SWIFT_BINDINGS.md).

---

## Documentation

- [`docs/CLI.md`](docs/CLI.md): Every flag on `turbospark-check`, `turbospark-model`, and `turbospark-server`, including a full steering/obliteration walkthrough.
- [`docs/MODELS.md`](docs/MODELS.md): The model catalog, the header-only probe, `turbospark-model pull`, and how to install something not in the table.
- [`docs/MODEL_FAMILY.md`](docs/MODEL_FAMILY.md): Supported model families, automatic detection, and parity matrix.
- [`docs/GTURBO.md`](docs/GTURBO.md): Comprehensive specification of the `.gturbo` installation format.
- [`DEVIATIONS.md`](DEVIATIONS.md): Wired features vs scaffolded scope.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): Benchmark results, quality gates, and parity analysis.
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md): Harness documentation and memory oracle details.
- [`docs/POWER_BASELINE.md`](docs/POWER_BASELINE.md): Power metrics (Watts, Joules/token).
- [`docs/SPECULATIVE_DECODING.md`](docs/SPECULATIVE_DECODING.md): DFlash and batched verify, measured marginal (~1.1x, small blocks only), and why it is not shipped.
- [`docs/OBLITERATION.md`](docs/OBLITERATION.md): Live directional steering (runtime abliteration) -- the CLI flags, the four edit modes, family coverage, measured cost, and what is still open.
- [`docs/EXPERT_ROUTING.md`](docs/EXPERT_ROUTING.md): Domain-restricted expert sets, measured negative.
- [`docs/QWEN4_EXP.md`](docs/QWEN4_EXP.md): Qwen3.8-Flash-Next (`qwen4_exp`) bring-up lessons learned -- intake, decode wiring, memory policy, the router/shared-expert-gate dtype bug and fix, first real-hardware decode.
- [`docs/FORGE_GUARDRAILS.md`](docs/FORGE_GUARDRAILS.md): Tool-call rescue, argument validation and the one-retry loop: how a verdict is reached, why a tool request is buffered, and why none of it leaves the process.
- [`docs/KERNELS.md`](docs/KERNELS.md): Complete technical catalog of all 98 Metal compute kernels, 12 supported quantization formats, and kernel optimizations.
- [`docs/SWIFT_BINDINGS.md`](docs/SWIFT_BINDINGS.md): Driving the engine from a native app: the Swift API, the C ABI, threading, and what is not supported.



---

## License

MIT. See [LICENSE](LICENSE).

