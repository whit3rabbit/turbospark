---
description: Official documentation for TurboSpark on Apple Silicon macOS.
icon: book-open
---

# TurboSpark Documentation

Welcome to the official TurboSpark documentation. TurboSpark is a native macOS application and high-performance Rust workspace engineered specifically for local inference on Apple Silicon.

## Key Capabilities

* **Mixture-of-Experts (MoE) Streaming**: Routed expert weights stream directly from SSD through a bounded Metal slot cache, enabling 20B to 35B models (Gemma 4, Qwen 3.6, Swift-1.5) to run with a ~1.5 to 2.1 GiB process memory footprint.
* **Native Desktop App**: Native SwiftUI interface with multi-session chat, reasoning controls, agent tooling, and encrypted profiles.
* **Audio Suite**: End-to-end local sound suite: Whisper, Qwen3-ASR, and Parakeet speech-to-text with speaker naming and meeting notes, Fish Audio Drama 3, OmniVoice, and MegaTTS3 synthesis and voice cloning, and MiniMax Music 3 track generation.
* **Image Generation**: Fast local diffusion generation with Z-Image Turbo and Qwen-Image 2.1 Turbo via Swift MLX.
* **Decision Models**: Typed boolean, categorical, and scoring decisions via the bundled OpenKind loopback daemon.
* **Local HTTP API**: Drop-in OpenAI, Anthropic, and Ollama compatibility with streaming SSE support.

---

## Explore the Documentation

### [Getting Started](getting-started/install.md)
* [Installation](getting-started/install.md): Install via Homebrew cask, manual DMG, or Cargo.
* [Quickstart](getting-started/quickstart.md): Model recommendations and first chat in under 5 minutes.
* [Desktop App Guide](getting-started/desktop.md): Features, parameters, and settings in `TurboSpark.app`.

### [Models & Hardware Acceleration](models/models.md)
* [Supported Models Catalog](models/models.md): Complete list of catalog aliases, quants, and evidence statuses.
* [Model Families and Architecture](models/model-families.md): Exhaustive guide to all 15 supported model families.
* [Hardware Acceleration Matrix](models/hardware-acceleration.md): Metal GPU, MLX, Core ML, and CPU support per model and task.
* [Memory and Capacity](models/memory-and-capacity.md): How MoE streaming works, slot cache math, and context memory scaling.

### [Multimodal & Image Generation](multimodal/vision-models.md)
* [Vision Models](multimodal/vision-models.md): Image comprehension with Qwen3-VL 4B and deepstack mergers.
* [Image Generation](multimodal/image-generation.md): Local diffusion generation with Z-Image Turbo and Qwen-Image 2.1 Turbo.

### [Audio Suite](audio/overview.md)
* [Audio Overview](audio/overview.md): Unified sound processing workspace and encrypted profile storage.
* [Speech to Text](audio/speech-to-text.md): Whisper, Qwen3-ASR, and Parakeet transcription, speaker naming, meeting notes, and realtime WebSocket streaming.
* [Text to Speech & Voice Cloning](audio/text-to-speech.md): Fish Audio Drama 3, OmniVoice, MegaTTS3, and Kokoro with consent requirements and the full HTTP request surface.
* [Music Generation](audio/music-generation.md): MiniMax Music 3 track generation, flow profiles, prompt builder, and lyrics tags.

### [Decision Models](decision-models/typesafe-decisions.md)
* [TypeSafe Decision Models](decision-models/typesafe-decisions.md): Strongly typed JSON classifications (Noul, Choice, Score) via OpenKind.

### [Features & Tools](features/humanize.md)
* [Humanize Text Rewriter](features/humanize.md): Rewrite drafts naturally with `humanizer-12b` and automated fact audits.
* [Agent Tools and Guardrails](features/agents-and-tools.md): Sandboxed tools, interactive permission gates, and tool-call repair tiers.

### [Command Line & Local API](command-line-and-local-api/cli.md)
* [CLI Reference](command-line-and-local-api/cli.md): Commands reference (`turbospark`, `turbospark-check`, `turbospark-server`, etc.).
* [Server Configuration](command-line-and-local-api/server.md): Flags, ports, Tailnet binding, and authentication.
* [Calling the Local API](command-line-and-local-api/call-the-local-api.md): Client integration examples in Python, TypeScript, and curl.

### [API Reference](../../api-reference/http.md)
* [HTTP REST API](../../api-reference/http.md): Complete wire endpoints reference.
* [OpenAPI Schemas](../../api-reference/http/schemas.md): Downloadable OpenAPI 3.1.0 specifications for unified and dedicated services.
