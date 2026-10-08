# MOSS-Music 8B Thinking (music understanding, lyrics ASR)

This module ports `mlx_audio/stt/models/moss_music/` (moss_music.py,
config.py, processor.py, audio.py) from mlx-audio 0.5.7 at source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Overview

Music understanding and lyrics transcription over a Qwen3 text backbone:

1. Whisper-family frontend: centered reflect STFT (400 FFT, 160 hop,
   periodic Hann window), power spectrum, Slaney 128-band mel, drop the
   last frame, log10 with a 1e-10 clamp, a global peak - 8 floor, and
   `(x + 4) / 4` scaling.
2. A 32-layer audio encoder: three 3x3 stride-2 Conv2d stages with GELU
   (128 mels downsampled to 16 frequency rows, 480 channels), a stem
   projection from 480*16 to 1280, sinusoidal positions, and full
   attention. Hidden states at layers 8, 16, and 24 are captured as
   deepstack features.
3. A gated-MLP adapter projects the encoder output to the text width; three
   further gated-MLP mergers project the deepstack captures. Both are
   quantized 4-bit group-64 in the pinned distribution (only the audio
   encoder stays full precision).
4. The prompt wraps the clip in `<|audio_bos|>...<|audio_eos|>` with a
   digit time marker inserted every two seconds (25 audio tokens at 12.5
   tokens per second), then the transcription prompt. Audio embeddings
   replace the audio placeholder tokens; the deepstack rows are added to
   the decoder hidden states after layers 0, 1, and 2 during prefill only.
5. Greedy decode over the shared Qwen3 decoder until EOS, `<think>` blocks
   stripped.

The upstream default prompt asks for a musical description and produces no
lyrics for a speech clip, so this family pins the explicit transcription
prompt `Please transcribe the lyrics of this clip.` (the fixture records
it).

## Pinned profile

- Hugging Face repository: `mlx-community/MOSS-Music-8B-Thinking-4bit`
- Immutable revision: `b14123a419b6254b92d9e55b8a5b6f7285e05c70`
- Input: mono f32 PCM at 16 kHz
- Audio encoder: 32 layers, d_model 1280, 20 heads, full precision
- Text backbone: Qwen3, 36 layers, hidden 4096, 32/8 heads, head dim 128,
  vocab 151936, untied LM head, 4-bit group-64
- Rust profile: `MOSS_MUSIC_8B_THINKING_4BIT`

Artifact sizes are bytes; SHA-256 digests were computed from the downloaded
snapshot at the pinned revision.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 6017835521 | `17560d5e601103ec84981673692f69602ed78ffeea07b522d731649cc9711a44` |
| `config.json` | 2823 | `09c79f9ca6708bcd81d996bfb6e84dce37d42ed9575e77b29e71d662039eb3f9` |
| `tokenizer.json` | 11423404 | `fe09df9d0df539ba3c7c8812750bda67907c889d0fbcc500454ba7037cc34466` |
| `tokenizer_config.json` | 6114 | `0869e41f5d123ff144a811f0d83c5d18871dcd4b4064f46bf9def194bfbc6f41` |
| `vocab.json` | 2776833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | 1671853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `added_tokens.json` | 801 | `1fabcc983ee0051b922009b1b5a7af1844385f38783e62679ac35d6dd2aa3355` |
| `special_tokens_map.json` | 613 | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| `generation_config.json` | 120 | `7914bf1e97eebd99700a1e0fa689a2cc66449ee748bf0d8eba69ea8470a5a83b` |
| `preprocessor_config.json` | 130 | `365bb03467925a83c979779d04a8767a7dc057f6e138691c06820884a70849b6` |
| `chat_template.jinja` | 4116 | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` |

## Verification evidence

`generate_moss_music_fixture.py` runs the pinned checkpoint through the
reference classes (with the loader's 4-bit quantization applied) on the
smoke clip and records the input features, full prompt token ids (including
the time-marker digits), audio mask count, greedy token ids, transcript, and
segments into `crates/audio/testdata/moss_music_reference.json` (force-added;
the workspace `.gitignore` covers testdata directories). Tests never
download model files.

Regenerate on an Apple Silicon host with the mlx-audio reference
environment:

```sh
~/.venv-mlxaudio/bin/python crates/audio/scripts/generate_moss_music_fixture.py \
  --model-dir ~/models/moss-music-8b-thinking-4bit \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/moss_music_reference.json \
  --revision b14123a419b6254b92d9e55b8a5b6f7285e05c70
```

- **Python reference transcript (pinned checkpoint, greedy):**
  `The quick brown fox jumps over the lazy dog.` (16 generated tokens
  including the `<think>` wrapper; prompt 67 tokens with one "2" marker).
- **Rust always-on fixture tests:** the frontend matches the recorded
  reference features within 2.0e-3 absolute, and the reconstructed prompt
  token sequence (system/user wrappers, audio placeholders, the time-marker
  digit, the lyrics prompt, and the assistant opening) matches the recorded
  reference prompt token-for-token. The timestamp-marker construction and
  `<think>` stripping are pinned by unit tests.

## Checkpoint gate status

The checkpoint-gated release test is checked in behind
`TURBOSPARK_MOSS_MUSIC_MODEL_DIR` but has NOT been run to completion on this
machine: the 8B model dequantizes to roughly 40 GiB of resident f32 weights,
which exceeds this host's 36 GiB (the run is killed by the OS). The Rust
port is fixture-verified end to end up to the decoder weights; a host with
at least 48 GiB must run the gated test before any real-checkpoint claim
is made. Port traps found while wiring (each fixed and covered by the
fixture tests): the checkpoint stores audio attention projections without
the reference sanitize's `.self_attn` segment, stores conv kernels in MLX
`[out, k, k, in]` order, and quantizes the adapter and deepstack mergers
even though they sit outside the audio encoder.

## Remaining gates

Real-checkpoint Rust transcription (memory-sized host), quality on actual
music/lyrics audio, long-clip chunking beyond one 400-frame window, the
streaming surface, performance and memory, runtime, catalog, FFI, and Swift
integration remain separate work.
