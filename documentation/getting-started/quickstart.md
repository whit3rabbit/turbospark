---
description: Start chatting, pulling models, and running the local API server in under 5 minutes.
icon: rocket
---

# Quickstart

Get up and running with TurboSpark on your Mac.

## 1. Discover Models for Your Mac

Different Macs have different memory capacities. Run `turbospark-model recommend` to inspect which catalog models fit your machine:

```sh
turbospark-model recommend
```

The recommendation tool inspects your Apple Silicon chip, available unified memory, and context window requirements, ranking models by fit and evidence status.

To list all catalog entries:

```sh
turbospark-model list
```

---

## 2. Pull Your First Model

Install the recommended model (for example, `gemma4` or `qwen36`):

```sh
# Pull Gemma 4 26B-A4B (MLX INT4, 16 expert cache slots, ~2.1 GB process footprint)
turbospark-model pull gemma4
```

The installer verifies sha256 checksums, downloads required tokenizer sidecars, packs resident weights, and writes a verified `.gturbo` installation receipt.

---

## 3. Interactive CLI Chat

Start a conversational chat session directly from your terminal:

```sh
turbospark-check --model gemma4 --chat
```

You can now type prompts, ask questions, or request code. Use `Ctrl+C` to exit.

To run a single prompt:

```sh
turbospark-check --model gemma4 --prompt "Explain unified memory on Apple Silicon in three bullet points."
```

---

## 4. Run the Local API Server

Launch the local server to connect tools, scripts, or client applications:

```sh
turbospark-server --model gemma4
```

The server binds to `http://127.0.0.1:8080` by default and provides:
* OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
* Anthropic-compatible endpoints: `/v1/messages`
* Health probes: `/health`

Test your server in another terminal window:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Hello local TurboSpark!"}]
  }'
```

---

## 5. Next Steps

* [Desktop App Guide](desktop.md): Explore the native macOS interface with multi-tab chat, encrypted profiles, and image generation.
* [Supported Models](../models/models.md): Discover all 15 model families, quantization levels, and hardware backends.
* [Audio Suite](../audio/overview.md): Transcribe speech, clone voices, and generate music.
* [Call the Local API](../command-line-and-local-api/call-the-local-api.md): Code examples for Python, TypeScript, and curl.
