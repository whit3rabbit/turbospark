---
description: Text-to-music generation with MiniMax Music 3 on Apple Silicon GPU.
icon: music
---

# Music Generation

TurboSpark generates complete music tracks from text prompts and structured lyrics using **MiniMax Music 3** (4-bit quantization). Inference is accelerated natively on Apple Silicon GPU via Metal compute kernels.

## Supported Model

* **MiniMax Music 3 (4-bit)** (`minimax-music3-4bit`):
  * Generates 44.1 kHz stereo 16-bit WAV music tracks.
  * Supports instrumental tracks or vocal tracks following structured lyrics tags.
  * Metal GPU acceleration enables local music composition without cloud fees.

---

## Music Generation in the Desktop App

The **Music** tab in Audio Studio features a dedicated composition workbench:

1. **Prompt Builder**:
   * **Genre**: Ambient, Classical, Electronic, Hip Hop, Jazz, Pop, Rock, Synthwave, etc.
   * **Mood**: Uplifting, Melancholic, Energetic, Relaxed, Dark, Dreamy.
   * **Tempo**: BPM and rhythmic feel (e.g. 120 BPM, driving beat, half-time).
   * **Instrumentation**: Acoustic guitar, analog synths, orchestral strings, 808 bass, piano.
2. **Lyrics & Structure Editor**:
   Use standard section markers to control track arrangement:
   ```text
   [intro]
   Gentle piano chords and soft strings

   [verse]
   The quiet screen begins to glow
   Local thoughts that start to flow

   [chorus]
   TurboSpark running free
   On Silicon naturally

   [outro]
   Fading piano chords
   ```
   For instrumental music, simply leave the lyrics as `[instrumental]`.
3. Click **Generate Track**. Progress is reported in real time, and the resulting stereo WAV can be played back or exported.

---

## Music Generation via CLI (`turbospark-music`)

Generate music directly from your terminal:

```sh
turbospark-music generate \
  --model minimax-music3-4bit \
  --prompt "lo-fi chillhop beat with dusty vinyl crackle and smooth jazz chords" \
  --lyrics "[instrumental]" \
  --length 30 \
  --output lofi_track.wav
```

### Options

* `--length <seconds>`: Target track duration (e.g. 15, 30, 60 seconds).
* `--lyrics <text>`: Lyrics or section arrangement.
* `--steps <1-30>`: Diffusion inference steps (default: 20).
* `--seed <N>`: Deterministic seed for repeatable generation.

---

## Music Generation via Local HTTP API

Because full music generation requires substantial GPU computation, use `async: true` when calling over HTTP:

```sh
curl http://127.0.0.1:8080/v1/audio/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "minimax-music3-4bit",
    "prompt": "upbeat acoustic folk melody with harmonica and fingerpicked guitar",
    "lyrics": "[instrumental]",
    "audio_length": 30,
    "async": true
  }'
```

Response (`202 Accepted`):

```json
{
  "job_id": "job_01j7abc123",
  "status": "running"
}
```

Poll job status until complete:

```sh
curl http://127.0.0.1:8080/v1/audio/jobs/job_01j7abc123 \
  -H "Authorization: Bearer $KEY"
```

Once `"status": "succeeded"`, download the finished audio:

```sh
curl http://127.0.0.1:8080/v1/audio/jobs/job_01j7abc123/result \
  -H "Authorization: Bearer $KEY" \
  -o generated_song.wav
```
