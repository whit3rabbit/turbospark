# GLM-ASR-Nano-2512

This module ports the pinned mlx-audio GLM-ASR inference path: a Whisper audio
encoder with traditional RoPE, a merge-four MLP adaptor, and a Llama text
decoder.

## Pinned profile

- Hugging Face repository: `mlx-community/GLM-ASR-Nano-2512-4bit`
- Immutable revision: `35553fa5bebfcc3ece3ce7d47b98827cb0ac9eef`
- mlx-audio reference: 0.5.7 at `e1b19b9054bf163f5d812221a54fcc346f1890e9`
- Input: mono f32 PCM at 16 kHz
- Audio frontend: 128-bin Slaney log-mel, 400-sample window, 160-sample hop
- Audio encoder: 32 Whisper blocks, width 1280, 20 heads, traditional RoPE
- Text decoder: 28 Llama blocks, width 2048, affine 4-bit weights, group size 64

The Rust implementation accepts a completed local snapshot and dequantizes
weights to f32. It supports one clip up to 30 seconds; longer audio chunking
and streaming are not implemented.

Download the exact revision with:

```sh
hf download mlx-community/GLM-ASR-Nano-2512-4bit \
  --revision 35553fa5bebfcc3ece3ce7d47b98827cb0ac9eef \
  --local-dir /models/glmasr-nano-2512-4bit
```

## Reference and parity

The pinned mlx-audio 0.5.7 model returned `The quick brown fox jumps over the
lazy dog.` on `testdata/qwen3_forced_aligner_reference.wav`. Transformers
reported a `glmasr` versus `Glmasr` model-type warning during loading; inference
completed successfully. `generate_glmasr_fixture.py` records the mel frontend,
convolutions, first and last encoder blocks, normalization, merge, adaptor, and
transcript from that MLX run.

The ignored checkpoint test compares selected early-stage values within
`0.01` and final encoder/adapter values within `0.03` to allow small f32 backend
differences accumulated over 32 layers. It also requires an exact transcript
match. The Rust debug test passed on the pinned checkpoint in 365.89 seconds.
This is one English clip and does not qualify recognition quality, other
languages, long-form transcription, latency, memory, or product integration.

Regenerate the fixture and run the checkpoint test with the pinned model
directory:

```sh
../mlx-audio/.venv/bin/python crates/audio/scripts/generate_glmasr_fixture.py \
  --model-dir /models/glmasr-nano-2512-4bit \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/glmasr_reference.json

TURBOSPARK_GLMASR_DIR=/models/glmasr-nano-2512-4bit \
  cargo test --locked -p turbospark-audio \
  stt::glmasr::tests::pinned_checkpoint_matches_mlx_stages_and_transcript \
  --lib -- --ignored --nocapture --test-threads=1
```

The fixture is checked in, and ordinary unit tests do not download checkpoints.
