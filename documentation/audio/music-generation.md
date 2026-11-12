---
description: Text-to-music generation with MiniMax Music 3 on Apple Silicon GPU.
icon: music
---

# Music Generation

TurboSpark generates complete music tracks from text prompts and structured lyrics using **MiniMax Music 3**. Inference is accelerated natively on Apple Silicon GPU via Metal compute kernels, and the 4-bit profiles are bit-identical to MLX end to end.

## Supported Models

All rows install with `turbospark-model pull-audio <ALIAS>` and generate 44.1 kHz stereo WAV:

| Alias | Size | Notes |
| --- | --- | --- |
| `minimax-music3-4bit` | 8.6 GiB | The standard 4-bit profile. Bit-identical to MLX 0.32.3 end to end (hiddens, logits, sampled codes, flow noise, waveform). |
| `minimax-music3-turbo-4bit-8step` | 11.8 GiB | The experimental 8-step Turbo LoRA checkpoint. Defaults to an 8-step Euler profile with both guidance branches on every step. Quality is one blind 15-second listen at one seed, not a gate. |
| `minimax-music3-bf16` | - | Full precision reference. |
| `minimax-music3-8bit` / `minimax-music3-6bit` | - | Intermediate quantizations. |
| `minimax-music3-mxfp8` / `minimax-music3-mxfp4` / `minimax-music3-nvfp4` | - | MXFP8, MXFP4 and NVFP4 formats. |

Tracks support instrumental output or vocals following structured lyrics tags, and generate locally with no cloud fees.

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
   For instrumental music, pass `[instrumental]` explicitly. Empty lyrics produce an `[instrumental]` track.
3. Click **Generate Track**. Progress is reported in real time, and the resulting stereo WAV can be played back or exported.

---

## Music Generation via CLI (`turbospark-music`)

Generate music directly from your terminal:

```sh
turbospark-music generate \
  --model minimax-music3-4bit \
  --caption "lo-fi chillhop beat with dusty vinyl crackle and smooth jazz chords" \
  --lyrics "[instrumental]" \
  --duration 30 \
  --output lofi_track.wav
```

### Options

| Flag | Description |
| --- | --- |
| `--model <ALIAS_OR_DIR>` | An installed Music 3 alias or a local converted model directory. |
| `--caption <TEXT>` | The text description of the track. Required. |
| `--lyrics <TEXT>` / `--lyrics-file <PATH>` | Lyrics or section arrangement. Required; use `[instrumental]` explicitly for instrumental output. |
| `--duration <SECONDS>` | Target track duration. |
| `--steps <N>` | Diffusion inference steps. Defaults to the chosen flow profile's step count. |
| `--seed <N>` | Deterministic seed for repeatable generation. |
| `--precision checkpoint\|float32` | Weight precision; `float32` upcasts the checkpoint. |
| `--wav-format pcm16\|float32` | `pcm16` (default) rounds to 16-bit integers; `float32` writes the generated samples unrounded. |
| `--timings` | Print per-stage wall time and the real-time factor to stderr. |
| `--flow-profile fast\|exact\|turbo` | The flow integration profile. See below. |
| `--flow-uncond-interval <N>` / `--flow-uncond-warmup <N>` / `--flow-solver euler\|ab2` | Override one part of the chosen profile. An explicit `--steps` is always used. |

### Flow profiles

* **fast** (default for ordinary checkpoints): second-order Adams-Bashforth, reuses the unconditional guidance branch (interval 3, warmup 2), and runs 21 steps when `--steps` is not given: 30 DiT forwards instead of 60, about half the DiT time. On one 8-second prompt it ran in about 55 percent of the exact time with a 0.974 correlation against the exact 30-step audio. It changes the audio and does not follow MLX's trajectory.
* **exact**: the reference profile. Euler, both guidance branches every step, 30 steps by default. Restores the MLX-identical trajectory.
* **turbo**: 8-step Euler with both guidance branches, the profile the marked Turbo checkpoint declares. Any checkpoint can be asked to run it.

---

## Music Generation via Local HTTP API

Start the server with `--music-model minimax-music3-4bit`, then call `POST /v1/audio/generate`. Full music generation runs for minutes, so `async: true` is the practical mode: it returns `202 Accepted` with a job id instead of holding the connection open.

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

Request fields: `model` and `prompt` are required; `lyrics` is optional (`[instrumental]` when omitted); `audio_length` is the target duration in seconds (default 60); `max_frames` sets the duration in frames instead; `num_inference_steps`, `seed` and `response_format` (`wav` or `pcm`) pass through to the runner.

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
