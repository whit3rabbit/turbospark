# turbospark-cli

[![crates.io](https://img.shields.io/crates/v/turbospark-cli.svg)](https://crates.io/crates/turbospark-cli)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/whit3rabbit/turbospark/blob/main/LICENSE)

Command-line binaries for [turbospark](https://github.com/whit3rabbit/turbospark), a native Rust LLM and diffusion inference engine for Apple Silicon.
- `turbospark-model`: Finds, inspects, and installs models.
- `turbospark-check`: Generates text and runs interactive chat.
- `turbospark-image`: Generates native diffusion images via Z-Image-Turbo.
- `turbospark`: Unified front end over all subcommands, the HTTP server, and external coding agents.

Generation is macOS-only and requires a Metal-capable Apple Silicon device. On other platforms, the binaries parse arguments and validate configuration.

## Installation

```sh
# Install from crates.io
cargo install turbospark-cli

# Or install via Homebrew cask (includes CLI tools, server, and GUI app)
brew install --cask whit3rabbit/tap/turbospark
```

## Quickstart

```sh
# 1. Pull a tested model into ~/.turbospark/models
turbospark-model pull tinyllama

# 2. Start an interactive terminal chat session
turbospark-check --model tinyllama --chat
```

`--model` accepts either a curated catalog alias (e.g. `tinyllama`, `gemma4`) or a direct path to a `.gturbo` directory.

---

## The Binaries

### 1. `turbospark-model`

Catalog discovery, remote Hugging Face inspection, and streaming install:

```sh
turbospark-model list                         # Browse curated catalog
turbospark-model info gemma4                  # Inspect catalog entry details
turbospark-model probe Qwen/Qwen3-30B-A3B-GGUF # Probe remote HF repo (reads KB, no download)
turbospark-model pull gemma4                  # Stream and repack directly into ~/.turbospark/models
turbospark-model recommend --context 8192     # Rank models that fit system memory
turbospark-model path gemma4                  # Print resolved filesystem path
turbospark-model rm gemma4                    # Delete local install
turbospark-model auth                         # Check or set Hugging Face access token
```

### 2. `turbospark-check`

Inference runner with streaming output and full sampling controls:

```sh
# Raw prompt (no chat template applied)
turbospark-check --model gemma4 --prompt "Explain quantum decoherence in one sentence."

# Formatted conversation using the model's native chat template
turbospark-check --model gemma4 --messages-file ./messages.json

# Interactive REPL trimming history to fit the context window
turbospark-check --model gemma4 --chat

# Directional steering edit (ablate, add, clamp, renorm)
turbospark-check --model gemma4 --chat --steer layer=12:mode=renorm:scale=1.5:vector=steering.bin
```

### 3. `turbospark-image`

Native diffusion image generation:

```sh
turbospark-image --model ~/models/z-image-turbo \
  --prompt "A cinematic photo of an astronaut on Mars during sunset" \
  --output mars.png --steps 9
```

### 4. `turbospark` (Unified Front End)

Front end routing commands to peer binaries:

```sh
turbospark run gemma4                         # Alias for turbospark-check --chat
turbospark run gemma4 "hello"                 # Alias for turbospark-check --prompt
turbospark image --prompt "mars"              # Alias for turbospark-image
turbospark serve                              # Launches turbospark-server
turbospark start                              # Starts background server daemon
turbospark stop / restart / status            # Manages daemon lifecycle
turbospark start claude                       # Configures Claude Code to use local server
turbospark list / pull / info / probe / auth   # Subcommands routed to turbospark-model
turbospark bench                              # Runs turbospark-bench
```

The `start <agent>` subcommand connects external coding agents (`claude`, `codex`, `opencode`, `hermes`, `openclaw`, `dsh`) directly to the local server via `ANTHROPIC_BASE_URL` or `OPENAI_BASE_URL`.

---

## Memory vs Throughput: `--expert-cache-slots`

MoE expert weights stream from NVMe storage on demand. Setting the cache slot count trades RAM for decode throughput:

| slots | peak RAM | decode |
|---|---|---|
| 16 | ~2.1 GB | ~44 tok/s |
| 32 | ~3.7 GB | ~51 tok/s |

- **Default is `auto`**: Automatically computes the largest slot count fitting a quarter of remaining system memory after base weights and a 4 GiB reserve.
- **Pinning for Benchmarks**: Specify `--expert-cache-slots 16` to pin reproducible memory ceilings.

---

## Key Modules

- `main.rs`: Entry point for `turbospark-check`.
- `generate/`: Non-interactive text generation and streaming detokenization loop.
- `chat.rs`: Interactive terminal REPL integrating `turbospark-window-fit`.
- `agent.rs`: External coding agent environment setup and configuration overlays.
- `daemon.rs`: Server background daemon process management.
- `bin/turbospark.rs`: Unified front-end command dispatcher.
- `bin/model.rs` & `bin/model_cmd/`: Subcommands for `turbospark-model`.
- `bin/image.rs`: Entry point for `turbospark-image`.

## Development & Test Commands

```sh
# Run CLI test suite
cargo test -p turbospark-cli

# Run real-model generation test (macOS, release mode)
cargo test -p turbospark-cli --test real_generation --release -- --ignored --nocapture
```

## Tests

- `tests/model_cli.rs`: Tests `turbospark-model` subcommands, alias resolution, and probing.
- `tests/real_generation.rs`: End-to-end inference tests verifying greedy and sampled generation.
- `tests/image_cli.rs`: CLI tests for `turbospark-image` argument validation and output paths.
- `tests/turbospark_cli.rs`: Tests unified front-end command routing.
- `tests/mference_check.rs`: Backward-compatibility tests for legacy CLI flags.

## Crate Gotchas

1. **Greedy vs Sampled Verification**: Greedy `argmax` selection is invariant under any monotone transformation of the logit distribution. A mathematical bug that ruins sampling distributions can still appear correct under greedy generation. Always verify both greedy and sampled outputs when testing numerics.
2. **Raw Prompt Babbling**: Invoking `--prompt` directly bypasses the checkpoint's Jinja chat template, which causes instruction-tuned models to hallucinate or babble. Use `--messages-file` or `--chat` for instruction models.
3. **Auto Expert Slots**: Because `--expert-cache-slots` defaults to `auto`, machines with different RAM sizes will report different slot counts and throughputs. Pin the slot count when comparing benchmarks.
