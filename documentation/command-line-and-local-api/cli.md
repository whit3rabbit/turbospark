---
description: Complete command-line interface reference for TurboSpark tools and utilities.
icon: terminal
---

# Command-Line Tools Reference

Installing `turbospark-cli` or the Homebrew cask provides these utilities on your `PATH`:

| Binary | Primary Function |
| --- | --- |
| `turbospark` | Front-end command: chat and one-shot runs, the server daemon (`serve`/`start`/`stop`/`restart`/`status`), model management, image generation, OCR, humanize, benchmarks, and launching coding agents against the local server. Forwards model subcommands to `turbospark-model` and `ocr` to `turbospark-ocr`. |
| `turbospark-check` | Generates text, interactive chat sessions, and runs JSON prompt files. Owns the full `--speculative` grammar, including `adaptive`. |
| `turbospark-model` | Explores the catalog, recommends models for this Mac, and installs models: text, image, vision, and every pinned audio row. |
| `turbospark-server` | Starts the local HTTP API server for OpenAI, Anthropic, Ollama, embeddings, and audio routes. |
| `turbospark-clone` | Clones target voices and speaks text via MegaTTS3 (requires `--consent`). |
| `turbospark-speak` | Speaks text with Fish Audio Drama 3: free-reference speech or consented cloning. |
| `turbospark-steer` | Trait steering: extracts a behavior direction (refusals, slop, humor, sycophancy, emotional tone) from a model's own activations and checks it. |
| `turbospark-music` | Generates 44.1 kHz stereo music tracks with MiniMax Music 3. |
| `turbospark-image` | Native Metal image diffusion generator for Z-Image-Turbo. |
| `turbospark-ocr` | Reads page images to text with the Core ML Unlimited OCR pipeline (macOS 15+). |

---

## 1. `turbospark-model`

Manage local model installations:

```sh
# List catalog entries, whisper rows, and every pinned audio model
turbospark-model list
turbospark-model list-speech
turbospark-model list-audio

# Recommend models ranked for this Mac
turbospark-model recommend --context 4096

# Inspect full metadata, checksums, and measured gates for an alias
turbospark-model info gemma4

# Download and install a model into ~/.turbospark/models/
turbospark-model pull gemma4

# Install a pinned audio model; terms print and need each --accept id
turbospark-model pull-audio omnivoice-8bit --accept omnivoice_higgs_tokenizer
turbospark-model pull-audio whisper-base

# Install Fish Audio Drama 3 by converting the pinned MLX checkpoint in place
turbospark-model pull-drama3 --out ~/models/drama3 --accept-fish-license

# Probe an unlisted Hugging Face checkpoint before downloading
turbospark-model probe owner/repo@revision --file model.Q4_K_M.gguf

# Other commands: path, rm, pull-image, pull-vision, auth
```

`pull-audio` covers every pinned audio model: speech, TTS, diarization, and the app-only FluidAudio Core ML rows, so a headless box or script can put the bytes where the app reads them. Installs land in `~/.turbospark/models/audio/<alias>.gturbo` with the receipt the app verifies before loading. `path` and `rm` resolve audio aliases too, and deleting an audio install goes through the catalog's ownership-checked removal.

An install streams multi-GB weights and cannot resume: a failure restarts the walk from the beginning. Set `TURBOSPARK_DISABLE_DOWNLOAD_CACHE=1` to lower peak disk use at the cost of refetching interrupted ranges.

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

# Speculative decoding: auto (default), off, a block 1-15, or adaptive[:MAX]
turbospark-check --model qwen38-27b-mtp --speculative 2 --chat
turbospark-check --model qwen38-27b-mtp --speculative adaptive:6 --prompt "..."
```

See [Speculative Decoding](../advanced/speculative-decoding.md) for the full grammar and the measured trade-offs.

---

## 3. `turbospark-server`

Run the local API server:

```sh
# Launch with a chat model on default port 8080
turbospark-server --model gemma4

# Launch with chat, speech-to-text, text-to-speech, and music models
turbospark-server \
  --model gemma4 \
  --stt-model whisper-base \
  --tts-model kokoro-82m-bf16 \
  --music-model minimax-music3-4bit \
  --port 8080 \
  --api-key "$TURBOSPARK_API_KEY"
```

Audio model flags repeat, and each one opens through the catalog: `--stt-model` takes Whisper, Qwen3-ASR, or Parakeet Redux/Ultra (Core ML, macOS 15+); `--tts-model` takes Kokoro, MegaTTS3, OmniVoice, or Drama 3; `--music-model` takes a MiniMax Music 3 row. See [Server Configuration](server.md).

---

## 4. `turbospark-clone`

Voice cloning and synthesis with MegaTTS3:

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

## 5. `turbospark-speak`

Speak text with Fish Audio Drama 3. Without a reference the model picks the voice; with `--reference`, `--reference-text` and `--consent` it clones. Output is mono 44.1 kHz, Metal only.

```sh
# The model picks the voice; the drawn seed prints to the terminal
turbospark-speak \
  --model fish-drama3-4bit \
  --text "Local speech with [whispers] delivery cues." \
  --output take.wav

# Consented cloning from a 1 to 30 second reference
turbospark-speak \
  --model ~/models/drama3 \
  --reference voice_sample.wav \
  --reference-text "The exact words spoken in the reference." \
  --consent \
  --text "Hello from my own voice." \
  --seed 7 \
  --output clone.wav
```

`--temperature`, `--top-p`, `--top-k` and `--max-frames` override Fish Audio's sampler recipe, and `--wav-format pcm16|float32` picks the output sample format. A run that stopped early (the model went silent at a pause and hit the frame cap) prints a warning with the seed used.

---

## 6. `turbospark-music`

Generate music tracks:

```sh
turbospark-music generate \
  --model minimax-music3-4bit \
  --caption "epic cinematic orchestral trailer music with brass swells" \
  --lyrics "[instrumental]" \
  --duration 30 \
  --output trailer.wav
```

`--flow-profile fast|exact|turbo` selects the flow integration; `--lyrics-file`, `--steps`, `--seed`, `--precision`, `--wav-format` and `--timings` are also available. See [Music Generation](../audio/music-generation.md).

---

## 7. `turbospark-image`

Generate images:

```sh
turbospark-image generate \
  --model z-image-turbo \
  --prompt "a red panda astronaut, watercolor" \
  --output panda.png
```

`--image` is a vision input flag on `turbospark-check`, not a generator: it attaches image files to the prompt (repeatable, all in one turn, `--image-batch` runs the prompt once per image). It refuses `--chat`, so a vision prompt is a one-shot run. For generation, use `turbospark-image`.

---

## 8. `turbospark-ocr`

Read page images (PNG or JPEG) as one document with the Core ML parts of Unlimited OCR. Requires macOS 15 or newer; the first run compiles the decoder for this machine.

```sh
turbospark-ocr --model unlimited-ocr-coreml page1.png page2.jpg

# Keep the layout markup (<PAGE>, <|det|> boxes) instead of plain text
turbospark-ocr --model unlimited-ocr-coreml --raw scan.png

# Cap generated tokens
turbospark-ocr --model unlimited-ocr-coreml --max-new-tokens 2048 scan.png
```

The unified binary forwards the same command as `turbospark ocr`.

---

## The Unified `turbospark` Front End

`turbospark` exposes the most-used pieces as subcommands:

```sh
# Chat REPL or a one-shot prompt
turbospark run gemma4
turbospark run gemma4 "explain quantum computing"

# Rewrite a draft so it reads like a person wrote it (audit goes to stderr)
turbospark humanize humanizer-12b draft.md > rewrite.md

# Server lifecycle
turbospark serve         # foreground
turbospark start         # managed background daemon
turbospark status        # health check
turbospark stop          # stop the daemon

# Launch a coding agent against the local server
turbospark gemma4 claude

# Model management, image, OCR, benchmarks
turbospark list
turbospark pull gemma4
turbospark image generate --prompt "sunset over Tokyo skyline" --output tokyo.png
turbospark ocr --model unlimited-ocr-coreml page.png
turbospark bench
```

Model subcommands (`list`, `pull`, `info`, `rm`, `probe`, `recommend`, `path`, `auth`) forward to `turbospark-model`, and `list-audio`, `list-speech` and `pull-audio` forward the same way.
