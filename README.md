# TurboSpark

TurboSpark is a native macOS app and Rust workspace for running local language models on Apple Silicon. Its main advantage is fine-grained Mixture-of-Experts (MoE) streaming: routed expert weights stay on SSD until they are needed, so supported large MoE models can run with a small bounded memory footprint.

The app provides desktop chat and agent tools. The Rust crates provide the Metal inference engine, model installer, command-line tools, HTTP server, and Swift/C bindings.

[![CI](https://github.com/whit3rabbit/turbospark/actions/workflows/ci.yml/badge.svg)](https://github.com/whit3rabbit/turbospark/actions/workflows/ci.yml)
[![Release](https://github.com/whit3rabbit/turbospark/actions/workflows/release.yml/badge.svg)](https://github.com/whit3rabbit/turbospark/actions/workflows/release.yml)
[![crates.io](https://img.shields.io/crates/v/turbospark-cli.svg)](https://crates.io/crates/turbospark-cli)
[![GitHub stars](https://img.shields.io/github/stars/whit3rabbit/turbospark)](https://github.com/whit3rabbit/turbospark/stargazers)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](docs/RELEASE.md)

## Install

TurboSpark targets Apple Silicon. The desktop app and Homebrew casks require macOS 14 Sonoma or newer.

### Homebrew, recommended

Install the app and CLI tools:

```sh
brew install --cask whit3rabbit/tap/turbospark
```

This installs `TurboSpark.app` in `/Applications` and puts these commands on your `PATH`:

```text
turbospark-check    generate text
turbospark-model    find and install models
turbospark-server   run the local API server
```

For CLI tools only:

```sh
brew install --cask whit3rabbit/tap/turbospark-cli
```

The two casks conflict. Choose the app cask or the CLI-only cask.

### Rust CLI

Install the published CLI and server crates with Rust stable:

```sh
cargo install turbospark-cli turbospark-server
```

To build the whole workspace from source, install Xcode Command Line Tools and run:

```sh
git clone https://github.com/whit3rabbit/turbospark.git
cd turbospark
cargo build --workspace --release
```

The source build produces the same three user-facing binaries under `target/release/`. See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the full build and test setup.

### Manual DMG

Every [GitHub Release](https://github.com/whit3rabbit/turbospark/releases) includes an Apple Silicon DMG named `TurboSpark-<version>-arm64.dmg`.

1. Download the DMG from the release assets.
2. Open it and drag `TurboSpark.app` to `/Applications`.
3. If macOS blocks the first launch, Control-click the app and choose **Open**. If that option is not enough, clear the quarantine attribute:

   ```sh
   xattr -dr com.apple.quarantine /Applications/TurboSpark.app
   ```

The release app is ad-hoc signed and not notarized, so the quarantine step may be required for a manual download. Homebrew can remove the quarantine during installation:

```sh
brew install --cask --no-quarantine whit3rabbit/tap/turbospark
```

The app bundle contains the CLI binaries. The Homebrew app cask also links them onto `PATH`.

## Quickstart

Install a catalog model, then start an interactive chat:

```sh
turbospark-model pull gemma4
turbospark-check --model gemma4 --chat
```

To use another model, inspect what fits this Mac first:

```sh
turbospark-model list
turbospark-model recommend
turbospark-model pull qwen36
```

Run the local OpenAI- and Anthropic-compatible server:

```sh
turbospark-server --model gemma4
```

The default server listens on `127.0.0.1:8080` and provides `/v1/chat/completions`, `/v1/messages`, `/v1/models`, and `/health`. See [`docs/CLI.md`](docs/CLI.md) for server flags and client setup.

## Supported models

The catalog in [`crates/catalog/src/models.json`](crates/catalog/src/models.json) is the source of truth. Use `turbospark-model list` for current aliases and evidence status.

| Family | Catalog checkpoints |
| --- | --- |
| Gemma 4 | MLX INT4, GGUF Q8_0, and UD-Q3_K_M/IQ3 sub-4-bit builds |
| Qwen 3.6 and Ornith-1.5 35B-A3B | MLX INT4 and Q4_K_M/Q8_0 GGUF builds |
| Qwen 3.5/3.8 dense | MLX INT4, MTP, vision, Bonsai 1-bit, and Ternary-Bonsai 2-bit builds |
| Qwen3-MoE | Qwen3-30B-A3B Q4_K_M |
| Qwen3.8-Flash-Next | REAP-288 MLX INT4 |
| gpt-oss | gpt-oss 20B MXFP4 |
| DeepSeek V2 | DeepSeek-V2-Lite 16B Q8_0 GGUF |
| Other supported families | Muse Glimmer, Qwen3-VL, Spark-X2.5, Mistral, TinyLlama, and related catalog rows |

New checkpoints are not automatically supported just because their architecture name matches. The catalog probe checks the checkpoint header, tokenizer sidecars, tensor types, and memory shape before installation. Read [`docs/MODELS.md`](docs/MODELS.md) before adding a model outside the catalog.

## Memory and benchmark results

These rows are measured `phys_footprint` peaks on one Apple M4 Max with 36 GB unified memory. They use 16 expert-cache slots and the listed context window. Peak footprint is process memory, not the model's on-disk size. Results vary by chip, context, cache size, and workload.

| Model | Context | Peak footprint | Decode | Install on disk |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 26B-A4B, MLX INT4 | 4,096 | 2,175 MiB | 33.0-45.6 tok/s | about 13.0 GB |
| Qwen 3.6 35B-A3B, MLX INT4 | 4,096 | 1,610 MiB | 32.6-38.0 tok/s | about 18.0 GB |
| Qwen3-30B-A3B, GGUF Q4_K_M | 4,096 | 2,748 MiB | 16.0-27.3 tok/s | about 18.6 GB |
| Qwen 3.8 27B, MLX INT4, dense | 4,096 | 661 MiB | 16.8-19.0 tok/s | about 15.2 GB |
| gpt-oss 20B, GGUF MXFP4 | 8,192 | 5,422 MiB | 23.4-31.3 tok/s | about 12.2 GB |

The low-memory result is primarily an MoE result. Dense models run, but they do not get the same routed-expert streaming benefit. For the complete measured rows, quality gates, cross-engine comparisons, and methodology, see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Supported features

| Feature | Support |
| --- | --- |
| Desktop app | Native SwiftUI chat, multi-chat persistence, model management, settings, and accessibility support |
| Agent runtime | Native shell, file, web, git, MCP, and indexed code-search tools with permission gates |
| MoE streaming | Routed expert weights stream from SSD through a bounded Metal slot cache |
| CLI generation | Interactive chat, raw prompts, JSON message files, reasoning output, cancellation, and model recommendations |
| Local API | OpenAI-compatible and Anthropic-compatible endpoints from `turbospark-server` or the app |
| Model intake | Catalog installs, Hugging Face header probes, GGUF intake, MLX quantized checkpoints, and `.gturbo` packing |
| Vision and images | Supported vision checkpoints plus validated Z-Image-Turbo generation in the CLI and app |
| Performance controls | Context and load guards, expert-cache sizing, chunked prefill, KV-cache quantization, and eligible speculative decoding |
| Steering | Directional residual steering for supported model flows, without modifying model weights |

## More details

- [Model catalog and installation](docs/MODELS.md)
- [CLI and server reference](docs/CLI.md)
- [Benchmark rows and quality evidence](docs/BENCHMARKS.md)
- [Release assets, Homebrew casks, and packaging](docs/RELEASE.md)
- [Swift bindings and app development](docs/SWIFT_BINDINGS.md)
- [Vision and image generation](docs/VISION.md) and [image generation details](docs/IMAGE_GENERATION.md)
- [Testing and evidence gates](docs/TESTING.md)

## License

[MIT](LICENSE)
