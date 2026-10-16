---
description: Synthesize speech and clone voices with Kokoro, Qwen3-TTS, and MegaTTS3 on Apple Silicon.
icon: volume-high
---

# Text to Speech and Voice Cloning

TurboSpark provides fast local speech synthesis and zero-shot voice cloning.

## Supported Models

1. **Kokoro-82M** (`kokoro-82m`):
   * Ultra-fast text-to-speech model producing 24 kHz mono audio.
   * Default voice: `af_heart`.
   * Runs natively on Metal GPU or CPU.

2. **Qwen3-TTS** (0.6B and 1.7B):
   * High-expressivity TTS and voice design models in the desktop app.
   * Supports CustomVoice, VoiceDesign, and Base configurations via Swift MLX.

3. **MegaTTS3** (`megatts3-f32`):
   * Zero-shot voice cloning diffusion model.
   * Clones a speaker's voice from a short reference recording (1 to 28 seconds).
   * Runs the diffusion transformer and WavVAE decoder on Native Metal GPU by default.

---

## Ethical Consent Requirement for Voice Cloning

Voice cloning with MegaTTS3 requires **explicit caller consent** on every run:
* In the CLI: You must supply the `--consent` flag.
* In the Desktop App: You must confirm an acknowledgment dialog stating you have the legal right or authorization to replicate the target voice.
* Over HTTP: You must include `"consent": true` in the JSON request body.

*Rationale*: Consent is strictly enforced because the community WavVAE encoder's provenance is unverified. In the app, consent covers one exact reference region and is never carried over implicitly between sessions.

---

## Zero-Shot Voice Cloning with MegaTTS3

### CLI Command (`turbospark-clone`)

Clone a voice from a reference WAV file and generate speech:

```sh
turbospark-clone speak \
  --model megatts3-f32 \
  --reference voice_sample.wav \
  --text "Welcome to TurboSpark local voice synthesis." \
  --quality balanced \
  --output cloned_speech.wav \
  --consent
```

#### Quality Presets

* `--quality fast`: 8 diffusion steps (fastest generation).
* `--quality balanced`: 16 diffusion steps (default recommendation).
* `--quality best`: 32 diffusion steps (highest quality and voice similarity).
* Custom steps: Pass `--steps <1-100>`.

#### Tuning Parameters

* `--p-w <value>`: Intelligibility weight (default: 1.4). Higher values improve phonetic clarity.
* `--t-w <value>`: Similarity weight (default: 3.0). Higher values pull closer to the reference speaker's timbre.
* `--seed <N>`: Deterministic seed for repeatable synthesis.
* `--cpu`: Forces portable CPU execution instead of Metal GPU. *Note: Metal is ~5x faster and highly recommended.*

---

## Speech Synthesis via Local HTTP API

### Standard TTS (Kokoro-82M)

```sh
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "kokoro-82m",
    "input": "TurboSpark delivers high performance local AI on Apple Silicon.",
    "voice": "af_heart",
    "response_format": "wav"
  }' \
  -o output.wav
```

### Voice Cloning via HTTP (MegaTTS3)

```sh
REF_B64=$(base64 -i reference.wav)

curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "megatts3-f32",
    "input": "This voice was synthesized locally on my Mac.",
    "reference_audio": "'"$REF_B64"'",
    "consent": true,
    "steps": 16,
    "p_w": 1.4,
    "t_w": 3.0
  }' \
  -o clone.wav
```
