---
description: Synthesize speech and clone voices with Fish Audio Drama 3, OmniVoice, MegaTTS3, Qwen3-TTS and Kokoro on Apple Silicon.
icon: volume-high
---

# Text to Speech and Voice Cloning

TurboSpark provides fast local speech synthesis and zero-shot voice cloning. Several families run in the Rust engine (CLI, HTTP server and the app), and two run only inside the desktop app.

## Supported Models

Run `turbospark-model list-audio` for the current table; the `RUNS IN` column says whether a row runs in the Rust engine or only in the app.

| Model | Catalog alias | Runs in | Output | Cloning |
| --- | --- | --- | --- | --- |
| **Fish Audio Drama 3** | `fish-drama3-4bit` (22.1 GiB) | Rust (Metal only) | 44.1 kHz mono | Yes, with the reference's transcript and consent |
| **OmniVoice** | `omnivoice-8bit` (1.3 GiB, default), `omnivoice-bf16` (1.9 GiB) | Rust (Metal, CPU fallback) | Multilingual, 24 languages | Yes, with consent |
| **MegaTTS3** | `megatts3-f32` (4.1 GiB) | Rust (Metal, `--cpu` fallback) | English | Yes, with consent (always needs a reference) |
| **Kokoro-82M** | `kokoro-82m-bf16` (312 MiB) | Rust, portable CPU path | 24 kHz mono, voice `af_heart` | No |
| **Qwen3-TTS** 0.6B and 1.7B | `qwen3-tts-12hz-*-8bit` (Base, CustomVoice, VoiceDesign) | App only (Swift MLX) | Voice design and expressive TTS | Base rows clone; CustomVoice uses preset voices; VoiceDesign describes a voice in words |
| **Kokoro on the Neural Engine** | `kokoro-82m-ane-coreml` (92 MiB) | Installs only; no adapter yet | 24 kHz | No |
| **PocketTTS** | `pocket-tts-coreml` (493 MiB) | Installs only; no adapter yet | Small on-device TTS | No |

Notes on the rows:

* **Fish Audio Drama 3** runs its Qwen3.5-MoE text backbone as a headless trunk with routed experts streamed from disk, so a job peaks near 5.6 GB instead of the 26 GB the reference implementation needs. It takes delivery cues inline in the text, such as `[whispers]` or `[pause]`, has no speed control, and picks its own voice when no reference is sent. Its license requires the "Built with Fish Audio" notice the app shows.
* **OmniVoice** clones only when a reference is sent; without one it is a multilingual TTS voice. The catalog installs it with the Boson Higgs tokenizer's community-license acknowledgment.
* **Kokoro-82M**'s Metal runtime is pending. The standalone server and CLI run it on the portable CPU reference path.
* The Qwen3-TTS rows install from the CLI like any other row, but only the app's Swift MLX provider executes them; the Rust session API and the HTTP server do not.
* `kokoro-82m-ane-coreml` and `pocket-tts-coreml` are pinned FluidAudio Core ML exports that install and verify like any audio row, so a script can put the bytes in place, but no adapter runs them yet in the app or the Rust engine. PocketTTS leaves out the `cosette` and `jean` voices, whose license is non-commercial.

### Installing

```sh
# Rows with license terms print them and need each --accept id
turbospark-model pull-audio omnivoice-8bit --accept omnivoice_higgs_tokenizer
turbospark-model pull-audio megatts3-f32 --accept megatts3_community_encoder

# Drama 3 converts the upstream checkpoint on install and has its own command.
# It streams the 20 GB source by range read, so it is never stored whole.
turbospark-model pull-drama3 --out ~/models/drama3 --accept-fish-license
```

A Drama 3 install made this way is a plain directory for `turbospark-speak --model`. The app keeps its own receipt-backed `fish-drama3-4bit` install in the managed store.

---

## Ethical Consent Requirement for Voice Cloning

Every clone requires **explicit caller consent** on every run, stating that you have the legal right or authorization to replicate the target voice:

* In the CLI: supply `--consent` (and, for Drama 3, `--reference-text`).
* In the Desktop App: confirm the acknowledgment beside the reference region. Consent covers one exact reference region and is never carried over by reusing or re-recording a reference.
* Over HTTP: include `"consent": true` in the JSON request body.

MegaTTS3 always needs a reference and consent, because the community WavVAE encoder's provenance is unverified. OmniVoice and Drama 3 clone only when `reference_audio` is sent; Drama 3 also needs the reference's exact transcript. A reference sent to a family that cannot clone (Kokoro) is refused.

---

## Fish Audio Drama 3 (`turbospark-speak`)

`turbospark-speak` speaks through the same audio session the server uses. Output is mono 44.1 kHz WAV, Metal only.

```sh
# The model picks the voice
turbospark-speak \
  --model ~/models/drama3 \
  --text "Welcome to TurboSpark. [whispers] Everything here runs on your Mac." \
  --output welcome.wav

# Clone a voice from a 1 to 30 second reference
turbospark-speak \
  --model ~/models/drama3 \
  --reference voice_sample.wav \
  --reference-text "The exact words spoken in the reference recording." \
  --consent \
  --text "This voice was synthesized locally on my Mac." \
  --output clone.wav
```

`--model` takes an installed audio alias (`fish-drama3-4bit`) or a Drama 3 install directory. `--text-file PATH` reads the text from a file.

### Sampling and seeds

Drama 3 samples with Fish Audio's serving recipe by default. The four sampler flags override it, and are refused for every other family:

* `--temperature <X>`: default 1.0.
* `--top-p <X>`: default 0.9.
* `--top-k <N>`: default 30.
* `--max-frames <N>`: default and maximum 1024; one frame is 46.4 ms.
* `--seed <N>`: makes a run repeatable. Without it each run draws its own seed and prints it, so running again gives a new take. A seed you name, 0 included, is kept.
* `--wav-format pcm16|float32`: output sample format.

Some takes pause at a sentence gap, a turn change or a `[pause]` and would stay silent until the frame cap. The runtime holds a pause's frames back, and if it reaches 128 frames it rolls back to the pause's start and draws again, up to 6 times per chunk. A take that still hit its frame cap without finishing, or held 160 silent frames (7.4 s), is reported as **stopped early**; `turbospark-speak` prints a warning, and the app retakes it with a fresh seed at most twice before saving the last take.

---

## Zero-Shot Voice Cloning with MegaTTS3 (`turbospark-clone`)

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

Start the server with one or more `--tts-model` rows, then call `POST /v1/audio/speech`. The response is WAV by default (`mp3` is refused by name because the server cannot encode it). `stream: true` returns raw `pcm` in chunks, one per synthesized segment, so the first audio leaves before the last segment is generated.

### Request fields

| Field | Families | Description |
| --- | --- | --- |
| `model`, `input` | all | The attached alias and the text (up to 4,096 characters). |
| `voice` | Kokoro | Voice name; default `af_heart`. |
| `speed` | families with a speed control | 0.5 to 2.0. Drama 3 has none and refuses it. |
| `response_format` | all | `wav` (default) or `pcm`. |
| `language`, `instruction` | OmniVoice | Language hint and a style instruction. Drama 3 refuses `instruction`: it takes delivery cues inline. |
| `reference_audio` | MegaTTS3, OmniVoice, Drama 3 | Base64 WAV of the voice to clone (up to about 24 MB encoded). Its presence makes the request a clone. |
| `reference_text` | Drama 3 | The exact words spoken in the reference. Required for a Drama 3 clone. |
| `consent` | cloning requests | Must be `true` whenever `reference_audio` is sent. |
| `steps`, `p_w`, `t_w` | MegaTTS3 | Diffusion steps and the intelligibility and similarity weights. |
| `seed` | all | Repeatable synthesis. A Drama 3 request without one draws a fresh seed, and the response names it in the `x-turbospark-seed` header. |
| `temperature`, `top_p`, `top_k`, `max_frames` | Drama 3 only | The sampler recipe; refused for every other family. |

### Standard TTS (Kokoro-82M)

```sh
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "kokoro-82m-bf16",
    "input": "TurboSpark delivers high performance local AI on Apple Silicon.",
    "voice": "af_heart",
    "response_format": "wav"
  }' \
  -o output.wav
```

### Fish Audio Drama 3 via HTTP

```sh
curl -D headers.txt http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "fish-drama3-4bit",
    "input": "Local speech, [whispers] generated on this Mac.",
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 30
  }' \
  -o drama3.wav

grep -i x-turbospark-seed headers.txt
```

To clone with Drama 3, add `reference_audio`, `reference_text` and `"consent": true`.

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

### Managing attached audio models

`GET /v1/audio/models` lists the attached audio models with their task, `POST /v1/audio/models` with `{"model": "<installed alias>"}` attaches another installed row (a filesystem path is refused so an API caller cannot probe directories), and `DELETE /v1/audio/models/{id}` detaches one. Audio models also appear in `GET /v1/models` with a `capabilities` field.

---

## Voices in the Desktop App

The Audio workspace has four pages that speak:

* **Voiceover** speaks a script without a reference. Drama 3 and OmniVoice pick their own voice here (no PCM, consent or transcript leaves the page; Drama 3 runs at speed 1.0 and OmniVoice keeps its speed slider), and the Qwen3-TTS CustomVoice rows also appear. A Voiceover take with seed 0 draws its own seed, so **Generate** again is a new take.
* **Voice Cloning** offers the models that clone with native Metal or MLX acceleration: Drama 3 (which asks for the reference's transcript beside consent), OmniVoice, MegaTTS3 and the Qwen3-TTS Base rows.
* **Voice Design** describes a voice in words with the Qwen3-TTS VoiceDesign model.
* **Advanced** exposes every attached synthesis model with its full parameter surface.

Each page's model picker rescans installed models when it opens, shows **Load** and **Unload** actions with a Loaded badge, and hides the download button for a row that is already installed.
