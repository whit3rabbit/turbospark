---
description: "Unified audio processing suite: speech-to-text, text-to-speech, zero-shot voice cloning, speaker diarization, meeting notes, and music generation."
icon: headphones
---

# Audio Suite Overview

TurboSpark bundles a complete local audio suite. Transcription, live text, speaker intelligence, meeting notes, synthesis, cloning and music all run on device, on Apple Silicon Metal, Core ML and the Neural Engine, with zero network audio streaming.

## Capabilities & Models Summary

Run `turbospark-model list-audio` for the full pinned table; the `RUNS IN` column marks which rows the Rust engine runs and which are app-only.

| Task | Catalog Models | Hardware Engine | Supported Interfaces |
| --- | --- | --- | --- |
| **Speech-to-Text** | Whisper (`whisper-base` and the `list-speech` layouts), Qwen3-ASR 0.6B, Parakeet Redux and Ultra on Core ML | Metal GPU, Core ML, CPU | App Audio, CLI, HTTP REST & WebSocket |
| **Live Transcription** | Parakeet Unified on Core ML (int8 encoder on the Neural Engine) | Core ML / ANE | App Record page, `ts_audio_stream_*` C ABI |
| **Speaker Diarization** | pyannote community-1 on Core ML (`speaker-diarization-coreml`) | Core ML | App, native audio surface |
| **Text-to-Speech** | Fish Audio Drama 3, OmniVoice, Kokoro-82M, PocketTTS (install-only) | Metal GPU, portable CPU | App Audio, `turbospark-speak`, `turbospark-server` |
| **Voice Design & TTS** | Qwen3-TTS 0.6B & 1.7B (Base, CustomVoice, VoiceDesign), Kokoro on the ANE | Swift MLX; Core ML | App Audio |
| **Zero-Shot Voice Cloning** | Fish Audio Drama 3, OmniVoice, MegaTTS3 | Metal GPU (default), CPU (`--cpu`) | App Voice Cloning, `turbospark-clone`, `turbospark-speak`, `turbospark-server` |
| **Music Generation** | MiniMax Music 3 (4-bit, Turbo 8-step, bf16, 8/6-bit, MXFP8/MXFP4/NVFP4) | Metal GPU | App Audio, `turbospark-music`, `turbospark-server` |

Rows with license terms (MegaTTS3's community encoder, OmniVoice's Higgs tokenizer, Drama 3's Fish Audio license, Paraformer's FunASR license) print them and need an `--accept <id>` on install. See [Speech to Text](speech-to-text.md), [Text to Speech and Voice Cloning](text-to-speech.md) and [Music Generation](music-generation.md) for each surface.

---

## Audio in the Desktop App

The **Audio** workspace in `TurboSpark.app` provides an integrated sound workstation:

1. **Record & Transcribe**: three modes (Voice Notes, Meeting, Any Audio), each with its own takes and models. Live text appears while recording, finished takes get speaker labels you can name and merge, and a meeting take grows a Notes view with a summary, timestamp chips and a model recommendation.
2. **Voiceover, Voice Cloning & Voice Design**: speak a script in a voice the model picks (Drama 3, OmniVoice), clone a voice from a consented reference (Drama 3, OmniVoice, MegaTTS3, Qwen3-TTS Base), or describe a voice in words (Qwen3-TTS VoiceDesign). Built-in consent gates cover every reference, and consent is never carried over between reference regions.
3. **Music Generation**: enter natural language prompts (e.g. "90s grunge guitar with energetic drums"), assemble genre, mood, tempo and instrumentation with the prompt builder, and structure lyrics with `[verse]`, `[chorus]` and `[instrumental]` markers.

---

## Privacy & Storage

* All audio recordings, generated takes, transcripts, and voice clones are stored in your encrypted local profile directory (`~/.turbospark/profiles/`).
* Microphone and screen/system audio recording permissions are requested only after you choose **Allow** in the access sheet, and a declined permission never blocks features that do not need it. On macOS 14.2+, system audio is captured with a CoreAudio process tap that avoids the Screen Recording prompt entirely.
* No audio data is ever transmitted to cloud servers or third parties. The FluidAudio Core ML rows run with `ModelHub.offlineMode = true`, so their assets come only from the TurboSpark catalog.
