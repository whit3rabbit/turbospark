# turbospark: Low-Memory LLM Inference for Apple Silicon, in Rust

[![CI](https://github.com/whit3rabbit/turbospark/actions/workflows/ci.yml/badge.svg)](https://github.com/whit3rabbit/turbospark/actions/workflows/ci.yml)
[![Release](https://github.com/whit3rabbit/turbospark/actions/workflows/release.yml/badge.svg)](https://github.com/whit3rabbit/turbospark/actions/workflows/release.yml)
[![crates.io](https://img.shields.io/crates/v/turbospark-cli.svg)](https://crates.io/crates/turbospark-cli)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20arm64-lightgrey.svg)](#limitations--out-of-scope)
[![MSRV](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](rust-toolchain.toml)

`turbospark` is a behavior-compatible **Rust port** of [Mference](https://github.com/NeelM0906/Mference), a Swift LLM inference engine for Apple Silicon, built to the design published in [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare). There is no Swift in this tree: the engine, the expert streamer, the repack pipeline, and the server are all Rust, and the only non-Rust source is the vendored Metal shader code both engines dispatch.

It is specifically designed for **Apple Silicon (macOS Metal)** to execute large language models (LLMs) with **extremely low memory overhead**. Instead of holding full model parameters in unified RAM/VRAM, `turbospark` streams routed expert weights directly from high-speed SSD storage into a lean working memory footprint. This enables Mac users with limited memory (8 GB, 16 GB, 24 GB, or 36 GB) to run large models like **Gemma 4 26B-A4B** and **Qwen 3.6 35B-A3B** locally without exhausting system memory.

The port is tested against the original rather than assumed compatible. Decode throughput lands within 1% of the Swift engine on the same install, every family carries a memory oracle asserting a peak-footprint ceiling, every family but the dense `llama` one carries a frozen quality gate (teacher-forced perplexity plus output digests), and the numerics are cross-checked against `mlx-lm`, `llama.cpp`, and MLX on identical bytes. What the suite proves is in [`docs/TESTING.md`](docs/TESTING.md), and the frozen numbers are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

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

- [In a hurry](#in-a-hurry)
- [Key Benefits for macOS / Apple Silicon Users](#key-benefits-for-macos--apple-silicon-users)
- [Memory Footprint & Benchmark Parity](#memory-footprint--benchmark-parity)
- [GGUF Intake & Custom `.gturbo` Format](#gguf-intake--custom-gturbo-format)
- [Supported Features & Models](#supported-features--models)
- [Architecture & Repository Layout](#architecture--repository-layout)
- [Getting Started](#getting-started)
- [Swift Bindings](#swift-bindings)
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

- **26B-35B Models in ~2 GB**: Runs large Mixture-of-Experts (MoE) models using only **~1.6 GiB to 2.2 GiB of peak RAM/VRAM**. Users with 16 GB or 36 GB Macs no longer need 64 GB+ memory configurations to run 26B-35B models. That band is the FLOOR and the figure every published number here is measured at: it is the routed-expert cache at 16 slots per layer. Since the cache is a straight RAM-for-throughput trade, a machine with memory to spare climbs to 24 or 32 slots on its own (`--expert-cache-slots auto`, the default) and trades ~1.5 GB for ~16% more decode; `--expert-cache-slots 16` pins the band back. A machine without the headroom is never moved off 16.
- **Direct GGUF Streaming Intake (New / WIP)**: Native intake for published GGUF formats (Gemma 4 Q8_0, Qwen 3.6 mixed Q4_K_M, and sub-4-bit IQ imatrix builds). Streams directly from Hugging Face or parses local GGUFs into optimized `.gturbo` format without requiring the 12-27 GB raw model payload to be loaded in RAM.
- **Sub-4-bit Option for the Tightest Budgets**: A published IQ3_XXS/IQ4_NL Gemma 4 build runs at **~1.8 GiB peak**, the leanest streaming configuration here, verified against `llama.cpp` on the same bytes. Slower than INT4, and the tradeoff is spelled out below rather than buried.
- **Architecture Registry with Honest Refusals**: GGUF `general.architecture` and Hugging Face `model_type` strings resolve through one table, every key of which was read off a real published file. An architecture this port recognizes but cannot yet run says so, names the missing work, and points at the bring-up checklist, instead of failing as "unknown". One string can cover two model shapes: `llama` covers Mixtral-style MoE and dense Mistral/Llama alike, and both halves run through the same decode flow, told apart by `num_experts` rather than by tensor names.
- **Zero-Copy Metal Execution**: The resident weights are mapped once and wrapped in a single `MTLBuffer` through `newBufferWithBytesNoCopy`, so the GPU reads them in place. Nothing is copied into a staging buffer per token.
- **Low Memory Overhead vs standard MLX / LLM tools, on mixture-of-experts models**: `mlx-lm` or `llama.cpp` keep full weights resident, so a 13 GB checkpoint wants roughly 13 GB. This engine streams routed experts on demand and holds under ~2.2 GB for Gemma 4 and ~1.6 GB for Qwen 3.6. The condition is load-bearing: a DENSE model has no experts to stream and gets none of this, and a coarse mixture like Mixtral gets none of it either. Both cases are in the comparison table below rather than left out of it.
- **Speculative Decoding on the Dense Families, Lossless and Opt-In**: Two drafters ride the checkpoint's own weights, so there is no separate draft model to download or keep resident: the checkpoint's own **multi-token-prediction head** (step-wise) and the **DFlash2 block drafter**, which proposes a whole block in ONE forward pass and verifies it in one batched pass. `--speculative auto` reads the install's own index, and DETECTING a drafter is not the same as ENABLING it: an MTP head is switched on, while a DFlash2 drafter is reported with the flag that runs it (`--speculative-drafter dflash`) and left off, because on ordinary prose it is a slowdown rather than a speedup (numbers below). Nothing speculates without either carrying a head or being asked. Acceptance is **exact** -- a proposal is kept only when it equals the target's own argmax, never approximated -- and a sampled run is REFUSED rather than served, because that exactness holds only at temperature 0 and approximating it would quietly bias the output toward the mode. **It is not, however, bit-identical to a sequential decode over a long generation**: the verify pass is BATCHED, and a batched row differs from a one-row pass in the last bits, so at a near-tie the argmax can fall the other way. Measured, the two greedy streams agree for 154 tokens on prose (and for all 600 on code and math) and then take different, equally valid continuations. Both remain the model's own greedy output; neither is degraded. **That difference is a SHAPE FLOOR every engine has, not a defect in this one** -- 1e-5 nats with the argmax agreeing, against the 7.4e-6 that MLX's own batched and cached passes differ by on this same architecture. If you need a stream reproducible token-for-token against non-speculative decoding, leave speculation off. **WHAT IT IS WORTH DEPENDS ON THE WORKLOAD, and the honest numbers are both signs**: measured through the real generation loop on `Qwen3.8-27B` over 600-token generations on an idle machine, DFlash2 at block 2 runs **1.33x** on a predictable code prompt and **1.47x** on arithmetic working, against **0.90x** on ordinary prose; the drafter's own trained block of 8 runs 0.82x / 1.07x / 0.42x on the same three. Short generations flatter it -- the same sweep over 256 tokens read 1.43x at block 8, because the predictable opening of an answer is not the whole of it. So the shipped default is the SMALL block, which inverts the datacenter answer -- a rejected batched round on these recurrent architectures restores a gated-DeltaNet snapshot and replays, and the odds of paying that climb from 9% to 98% across those blocks. A small block wins where speculation pays and bounds the loss where it does not. Read [`docs/DFLASH2.md`](docs/DFLASH2.md) and [`docs/MTP_SPECULATIVE.md`](docs/MTP_SPECULATIVE.md) before carrying any of these numbers; note also that the same arithmetic on the MoE families lands at ~1.1x and is not shipped.
- **Built-in OpenAI & Anthropic API Server**: Includes a local server providing OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) endpoints for drop-in integration with CLI tools (e.g., `claude-code`), Web UIs, and applications.

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
| **Ornith-1.5 35B-A3B** (INT4) | 18 GB | **~1.6 GB** | 32 to 43 tok/s | 21 W (1 case) | ~0.47 J (1 case) |
| **Qwen3-30B-A3B** (Q4_K_M) | 17 GB | ~2.7 GB | 16 to 25 tok/s | 21 W | ~0.8 to 1.4 J |
| **gpt-oss-20b** (MXFP4) | 11 GB | ~5.4 GB | 23 to 30 tok/s | 30 to 33 W | ~1.1 to 1.3 J |

**The first four rows are the point of the project**: a 26B model in ~2.1 GB and two 35B models in ~1.6 GB, against 13 GB and 18 GB on disk. The last two rows are honest counter-examples that still stream but land higher, and the reason is arithmetic rather than a defect: see the note on expert size below.

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
| **Muse Glimmer 30B** (INT4) | 15 GB | 535 MB counted | 13 to 19 tok/s | measured at 8k context; reasons before answering |

> **Caveat, and please read it before quoting the counted figures.** Nothing streams in a dense model. Those figures are what macOS *counts* against the process. The weights are memory-mapped and simply are not counted — most starkly for Muse Glimmer, where 535 MB is counted against 15 GB of weights. You still need a machine that can hold and page them, so treat a dense model as needing roughly its **size on disk** in free RAM, not its counted footprint. The counted number is useful for spotting leaks, not for capacity planning.

### With this engine against without it

The saving comes from streaming routed experts, so it exists only where there are routed experts to stream.

The right-hand column is arithmetic rather than a measurement. A conventional runner keeps every weight resident, so it needs roughly the file size plus a KV cache. If a model is dense, both columns hold the same number and this engine buys nothing on memory.

| Install | Format | Bits | On disk | RAM here | A conventional runner needs | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
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

**How to read the other columns.** "Power draw" is the engine's own CPU + GPU draw while generating, not the whole machine: the laptop as a whole measured roughly 50 to 70 W under load, most of the difference being the display. "Energy per token" is joules per generated token, so at ~0.4 J a thousand tokens costs about 400 J, roughly 0.1 Wh. Speed and power vary by prompt length, and the ranges span three fixed benchmark prompts of increasing size, except the Ornith row, whose power is one case and says so. Power figures exist for six installs and are simply absent for the rest. A seventh, **Muse Glimmer 30B, is deliberately not in a table**: it draws ~38 W and saturates this laptop thermally within about two minutes, so every arm the harness measures is being driven by the thermal governor rather than by the workload, and their energy-per-token wanders 25% run to run. Its unconstrained cost is known (~2.02 J/token, reproduced across three sessions to 1.5%); its sustained cost on this hardware is not knowable. The direction of that error is the trap worth knowing: a throttled run is slower AND cheaper per token, so it does not look broken in a power table, it looks good.

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

Seven architecture families run end to end, each with a real decode flow rather than a config entry. Every checkpoint named below has been installed from its published bytes and generated text through `turbospark-check` on this machine. None is a projection from a config file.

| Family | Checkpoints that run today | Shape |
| --- | --- | --- |
| **Gemma 4** | `gemma-4-26B-A4B-it` at MLX INT4, Q8_0 GGUF, and sub-4-bit IQ GGUF | Sliding-window + full attention, 128 streamed experts |
| **Qwen 3.6** (`qwen3_5_moe`) | `Qwen3.6-35B-A3B` at MLX INT4 and Q4_K_M GGUF | Gated-DeltaNet linear attention + 256 streamed experts |
| **Qwen 3.5/3.8 dense** (`qwen3_5`) | `Qwen/Qwen3.8-27B` (4-bit), `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (2-bit), `prism-ml/Bonsai-27B-mlx-1bit` (1-bit) | Same hybrid attention, dense FFN. **Three checkpoints, three widths, one architecture** |
| **Qwen3-MoE** (`qwen3moe`) | `Qwen3-30B-A3B` at Q4_K_M GGUF | Plain GQA + 128 streamed experts |
| **Llama** (`llama`) | Mixtral 8x7B, Mistral 7B, TinyLlama 1.1B | Plain GQA, one architecture string covering a MoE half and a dense half, both running |
| **gpt-oss** (`gptOss`) | `gpt-oss-20b` MXFP4 | GQA with attention sinks, YaRN rope, Harmony reasoning channels |
| **Muse Glimmer** (`museGlimmer`) | `Muse-Glimmer-30B` at MLX INT4 | Dense GQA, 3-sliding/1-full window, **NoPE on the full layers**, separate attention output gate, reasons before answering |

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

Two container families, twelve types. Whether a type runs is decided per type and per ROLE, not per format: a kernel that decodes a weight matrix is not the same kernel as one that decodes a routed expert or an embedding row, and several types have only the ones their real checkpoint needed.

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

### Server & Interfaces
- **Interactive REPL & CLI**: `turbospark-check` binary for interactive chat (`--chat`), raw prompt (`--prompt`), or JSON message history (`--messages-file`).
- **HTTP Server**: `turbospark-server` serving OpenAI Chat Completions (`/v1/chat/completions`), Anthropic Messages (`/v1/messages`), and `/v1/models`.
- **Configurable Expert Cache**: Adjust expert cache slot counts (8, 16, 24, 32) to tune performance vs memory footprint.
- **Tool Calling, With Guardrails On By Default**: Both endpoints render a request's `tools` through the checkpoint's own chat template and return calls as OpenAI `tool_calls` / Anthropic `tool_use`. On top of that, a reliability layer aimed at what a SMALL local model actually gets wrong: a call emitted in a dialect the checkpoint's own template did not teach it (bare JSON, Qwen XML, Mistral `[TOOL_CALLS]`) is **rescued** out of the raw text instead of reaching the client as prose, a call's arguments are **validated** against the schema the request itself sent, and a failure is **re-asked once** with a corrective nudge. All of it runs **in process against the local model**: the pure half of [`forge-guardrails`](https://crates.io/crates/forge-guardrails) is compiled in (6 added crates, no network crate among them), there is no proxy, no second process, and no outbound call of any kind. One behaviour change to know: a request carrying tools is buffered rather than streamed while guardrails are on, because a verdict needs the whole turn; requests without tools stream exactly as before. `--guardrails off` restores the previous path. See [`docs/FORGE_GUARDRAILS.md`](docs/FORGE_GUARDRAILS.md).

### Limitations & Out of Scope
- **Apple Silicon Acceleration Only**: Metal GPU acceleration requires macOS (`xcrun -sdk macosx metal`). On non-macOS platforms, crates compile CPU stubs.
- **Sequential Prompt Prefill**: Prompt tokens are processed sequentially per token (prefill tile kernels descoped, see [`DEVIATIONS.md`](DEVIATIONS.md)).
- **Q4_0 GGUF Quantization**: Refused at open until dedicated Q4_0 resident GEMV and embedding kernels land.
- **Sub-4-bit Costs Energy, Not Just Throughput**: The IQ path is a memory and disk win only (-15% peak footprint, -20% expert bytes). Measured on the power harness it draws roughly 2x the joules per decoded token of the INT4 install (codebook dequant nearly doubles GPU watts while running 35% slower). Use it when memory is the constraint. INT4 remains the default on every other axis. See [`docs/POWER_BASELINE.md`](docs/POWER_BASELINE.md).
- **No Speculative Decoding on the MoE Families**: Measured before building, and it does not pay there. A batched verify of M tokens has to cost less, in decode-steps, than the tokens it gets accepted. On this engine 19% of decode compute is per-token work with no weights to amortize, so verify cost scales almost linearly in M. Against a trained DFlash drafter's published accept lengths that lands at **1.14x at block 4, 0.97x at block 8 and 0.87x at block 16**, so small blocks win and large ones lose, inverting the datacenter result where verify is nearly free. About 1.1x is not worth the drafter, and the break-even column already assumes a batched MoE kernel that does not exist. The lever is that kernel rather than the drafter. Full arithmetic, the five measurement surfaces, and what would change the answer: [`docs/SPECULATIVE_DECODING.md`](docs/SPECULATIVE_DECODING.md). **That verdict is the MoE family's.** The DENSE `qwen3_5` family has no expert-union term and a much smaller un-amortizable floor, and there a checkpoint's own drafter does pay: a native MTP head measures 1.44x-1.66x ([`docs/MTP_SPECULATIVE.md`](docs/MTP_SPECULATIVE.md)) and the DFlash2 block drafter accepts 7.09 of 8 proposals per round ([`docs/DFLASH2.md`](docs/DFLASH2.md)). Cost a drafter per family, not per engine.


---

## Architecture & Repository Layout

The workspace is organized into modular Rust crates:

- **`crates/core`**: Primitive types, token definitions, and `RuntimeConfig`.
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

---

## Getting Started

Four steps, start to finish: install the binaries, pull a model, talk to it, then serve it to your own tools. Needs an Apple Silicon Mac on macOS. The walkthrough uses **Qwen3.8-27B**; see the note under step 2 for why you might want a different one.

### 1. Install

Three ways in. Homebrew is the shortest, and all three put the same three binaries on `PATH`: `turbospark-check` (generate), `turbospark-model` (find and install models), and `turbospark-server` (HTTP).

```sh
# Homebrew. One cask, all three binaries.
brew install --cask whit3rabbit/tap/turbospark
```

```sh
# crates.io. `turbospark-cli` carries turbospark-check AND turbospark-model.
cargo install turbospark-cli turbospark-server
```

Or grab the [latest release](https://github.com/whit3rabbit/turbospark/releases) directly. The asset is one zip per version, built for `aarch64-apple-darwin`, with a `SHA256SUMS` beside it:

```sh
VER=0.1.0
curl -LO "https://github.com/whit3rabbit/turbospark/releases/download/v${VER}/turbospark-${VER}-macos-arm64.zip"
unzip "turbospark-${VER}-macos-arm64.zip" -d ~/bin

# macOS quarantines anything downloaded by a browser or curl. Without this
# the first run dies with "cannot be opened because the developer cannot be
# verified" rather than anything about turbospark.
xattr -d com.apple.quarantine ~/bin/turbospark-* 2>/dev/null
```

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
machine; a 27 GB one can still be refused, on its slot cache rather than its
size. Rows that have been through a memory oracle here quote what it measured;
rows that have not say so rather than guessing. Add `--discover` to rank the
most-downloaded GGUF repositories on Hugging Face through the same gates.

### 2. Pull a model

```sh
turbospark-model pull qwen38-27b
```

That streams `mlx-community/Qwen3.8-27B-4bit` from Hugging Face in ranges and writes a `.gturbo` install to `~/.turbospark/models/qwen38-27b.gturbo`. About 15 GiB moves over the network and ~14 GiB lands on disk; the original checkpoint is **never written to disk whole**. Budget 20 minutes on a fast connection.

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

Use `--messages-file` or `--chat` rather than `--prompt` on an instruction-tuned model. `--prompt` sends raw text with no chat framing, which makes these checkpoints babble; that is the template missing, not a decode bug.

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

Loopback on port 8080, serving OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`. One runner per process, so requests are answered one at a time. `--model` takes an alias or a directory here exactly as it does for `turbospark-check`, and the startup line prints which directory an alias resolved to. In a script that would rather fail early than serve the wrong thing, `turbospark-model path <alias>` prints the install directory and exits non-zero if the model is not installed.

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

`ANTHROPIC_API_KEY` is required by the client and ignored by the server, which has **no authentication and no TLS**: it is a loopback service. The model discovery flag makes Claude Code ask `/v1/models` instead of assuming Anthropic's hosted names; the server advertises one id, the install directory's own name (`qwen38-27b.gturbo` here). Anything else speaking either API works the same way, e.g. `OPENAI_BASE_URL=http://127.0.0.1:8080/v1`.

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

Each checkpoint also still has its own `crates/repack` integration-test target, listed with its environment variable in [`AGENTS.md`](AGENTS.md); those are what the catalog rows were built from.

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

The workspace suite runs on any platform and covers the structural contracts. The heavier proof is env-gated and opt-in, because it needs a real model install: per-family quality gates freeze teacher-forced perplexity plus greedy and sampled output digests, memory oracles assert peak footprint against a per-chip ceiling and re-run a warm case to catch growth, and a determinism probe runs one greedy generation six times and requires exactly one distinct output.

The quality gate is calibrated rather than decorative. Shifting one quantization level in 0.0122% of Gemma 4's expert bytes moves its perplexity +10.5%, so the gate sees damage far below what reads as coherent by eye. Gating conventions and test-writing rules are in [`docs/TESTING.md`](docs/TESTING.md).

---

## Swift Bindings

A C ABI (`crates/ffi`) and a SwiftPM package over it, so a native macOS app
drives the engine in-process rather than over HTTP. `swift/TurboSparkDemo` is
a small SwiftUI chat app that exercises the whole thing.

```bash
make swift-lib                                    # build the staticlib + stage the header
make swift-demo                                   # run the demo chat app
make swift-test                                   # ABI checks, no model needed
make swift-test-real MODEL=~/models/gemma4.gturbo # end to end against a real install
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
`turbospark.h` to about twenty functions and makes adding a knob something
other than an ABI break. The per-token path carries no JSON: it is a pointer
and a length.

**Reasoning arrives separately from the reply.** `.content` is the assistant
turn to keep; `.reasoning` is for display only, because the checkpoints that
produce it drop prior-turn thinking from their own history and replaying it
sends the model something it was never trained to read.

Model management (`TurboSparkCatalog`) works on any platform, including ones
that cannot then run a model. Installs stream gigabytes and **cannot resume**,
so tell the user before starting rather than after failing.

The full API, the C ABI for non-Swift hosts, the threading contract, and the
list of what is deliberately not supported are in
[`docs/SWIFT_BINDINGS.md`](docs/SWIFT_BINDINGS.md).

---

## Documentation

- [`docs/MODELS.md`](docs/MODELS.md): The model catalog, the header-only probe, `turbospark-model pull`, and how to install something not in the table.
- [`docs/MODEL_FAMILY.md`](docs/MODEL_FAMILY.md): Supported model families, automatic detection, and parity matrix.
- [`docs/GTURBO.md`](docs/GTURBO.md): Comprehensive specification of the `.gturbo` installation format.
- [`DEVIATIONS.md`](DEVIATIONS.md): Wired features vs scaffolded scope.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): Benchmark results, quality gates, and parity analysis.
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md): Harness documentation and memory oracle details.
- [`docs/POWER_BASELINE.md`](docs/POWER_BASELINE.md): Power metrics (Watts, Joules/token).
- [`docs/SPECULATIVE_DECODING.md`](docs/SPECULATIVE_DECODING.md): DFlash and batched verify, measured marginal (~1.1x, small blocks only), and why it is not shipped.
- [`docs/EXPERT_ROUTING.md`](docs/EXPERT_ROUTING.md): Domain-restricted expert sets, measured negative.
- [`docs/FORGE_GUARDRAILS.md`](docs/FORGE_GUARDRAILS.md): Tool-call rescue, argument validation and the one-retry loop: how a verdict is reached, why a tool request is buffered, and why none of it leaves the process.
- [`docs/SWIFT_BINDINGS.md`](docs/SWIFT_BINDINGS.md): Driving the engine from a native app: the Swift API, the C ABI, threading, and what is not supported.



---

## License

MIT. See [LICENSE](LICENSE).

