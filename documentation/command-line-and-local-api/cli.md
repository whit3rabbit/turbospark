---
description: Complete command-line interface reference for TurboSpark tools and utilities.
icon: terminal
---

# Command-Line Tools Reference

Installing `turbospark-cli` or the Homebrew cask provides six dedicated CLI utilities on your `PATH`.

## Summary of Commands

| Binary | Primary Function |
| --- | --- |
| `turbospark` | Front-end command: runs subcommands for models, chat, humanize, image, and serve. |
| `turbospark-check` | Generates text, interactive chat sessions, and runs JSON prompt files. |
| `turbospark-model` | Explores catalog, recommends models for this Mac, and installs models. |
| `turbospark-server` | Starts the local HTTP API server for OpenAI, Anthropic, Ollama, and audio routes. |
| `turbospark-clone` | Clones target voices and speaks text via MegaTTS3 (requires `--consent`). |
| `turbospark-music` | Generates 44.1 kHz stereo music tracks using MiniMax Music 3. |
| `turbospark-image` | Native Metal image diffusion generator for Z-Image-Turbo. |

---

## 1. `turbospark-model`

Manage local model installations:

```sh
# List all catalog entries
turbospark-model list

# Recommend models ranked for this Mac
turbospark-model recommend --context 4096

# Inspect full metadata, checksums, and measured gates for an alias
turbospark-model info gemma4

# Download and install a model into ~/.turbospark/models/
turbospark-model pull gemma4

# Probe an unlisted Hugging Face checkpoint before downloading
turbospark-model probe owner/repo@revision --file model.Q4_K_M.gguf
```

---

## 2. `turbospark-check`

Run local model inference directly:

```sh
# Start interactive conversational chat
turbospark-check --model gemma4 --chat

# Execute a single text prompt
turbospark-check --model gemma4 --prompt "Explain the difference between MQA and GQA."

# Execute a JSON conversation message file
turbospark-check --model gemma4 --messages-file prompt.json

# Adjust sampling parameters
turbospark-check --model gemma4 \
  --prompt "Write three haikus." \
  --temperature 0.8 \
  --top-p 0.9 \
  --max-tokens 150
```

---

## 3. `turbospark-server`

Run the local API server:

```sh
# Launch with a chat model on default port 8080
turbospark-server --model gemma4

# Launch with chat, speech-to-text, text-to-speech, and music models
turbospark-server \
  --model gemma4 \
  --stt-model whisper-base-en \
  --tts-model kokoro-82m \
  --music-model minimax-music3-4bit \
  --port 8080 \
  --api-key "$TURBOSPARK_API_KEY"
```

---

## 4. `turbospark-clone`

Voice cloning and synthesis:

```sh
turbospark-clone speak \
  --model megatts3-f32 \
  --reference reference.wav \
  --text "Hello, this is a cloned voice generated on Apple Silicon." \
  --quality balanced \
  --output out.wav \
  --consent
```

---

## 5. `turbospark-music`

Generate music tracks:

```sh
turbospark-music generate \
  --model minimax-music3-4bit \
  --prompt "epic cinematic orchestral trailer music with brass swells" \
  --length 30 \
  --output trailer.wav
```
