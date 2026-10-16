---
description: "Unified audio processing suite: speech-to-text, text-to-speech, zero-shot voice cloning, and music generation."
icon: headphones
---

# Audio Suite Overview

TurboSpark integrates a comprehensive local audio processing suite. Everything runs entirely on device using Apple Silicon Metal acceleration, with zero network audio streaming.

## Capabilities & Models Summary

| Task | Catalog Models | Hardware Engine | Supported Interfaces |
| --- | --- | --- | --- |
| **Speech-to-Text** | Whisper (tiny, base, small), Qwen3-ASR 0.6B | Metal GPU, MLX, CPU | App Audio Studio, CLI, HTTP REST & WebSocket |
| **Text-to-Speech** | Kokoro-82M | Metal GPU, CPU | App Audio Studio, `turbospark-server` |
| **Voice Design & TTS** | Qwen3-TTS 0.6B & 1.7B (Base, CustomVoice, VoiceDesign) | Swift MLX | App Audio Studio |
| **Zero-Shot Voice Cloning** | MegaTTS3 | Metal GPU (Default), CPU (`--cpu`) | App Audio Studio, `turbospark-clone`, `turbospark-server` |
| **Music Generation** | MiniMax Music 3 (4-bit) | Metal GPU | App Audio Studio, `turbospark-music`, `turbospark-server` |

---

## Audio Studio in the Desktop App

The **Audio** workspace in `TurboSpark.app` provides an integrated sound workstation:

1. **Recording Section**:
   * Record microphone input, system audio loopback, or both mixed together.
   * View live audio waveforms and volume meters.
   * Live streaming draft transcription appears as you speak.

2. **Synthesis & Voice Cloning Section**:
   * Text-to-speech generation with Kokoro and Qwen3-TTS voices.
   * Upload or record a short reference clip to clone a voice using MegaTTS3.
   * Built-in consent gate ensures compliance with ethical voice replication standards.

3. **Music Generation Section**:
   * Enter natural language song prompts (e.g. "90s grunge guitar with energetic drums").
   * Interactive prompt builder helps assemble genre, mood, tempo, and instrumentation tags.
   * Structured lyrics editor supporting section markers (`[verse]`, `[chorus]`, `[solo]`, `[outro]`).

---

## Privacy & Storage

* All audio recordings, generated takes, transcripts, and voice clones are stored in your encrypted local profile directory (`~/.turbospark/profiles/`).
* Microphone and screen/system audio recording permissions are requested only after you click **Record** in the app.
* No audio data is ever transmitted to cloud servers or third parties.
