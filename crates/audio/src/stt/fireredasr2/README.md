# FireRedASR2-AED

This module ports the FireRedASR2-AED English encoder-decoder from
`mlx_audio/stt/models/fireredasr2/` at mlx-audio commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9` (version 0.5.7).

## Pinned profile

- Hugging Face repository: `mlx-community/FireRedASR2-AED-mlx`
- Revision: `f3212eacfa49b851130b97c63653c8e06ee09bdb`
- Profile: 80-bin FBANK, 16 Conformer encoder blocks, 16 Transformer decoder
  blocks, 1280 hidden width, 20 attention heads, 8667 dictionary entries.
- Input: mono normalized f32 PCM at 16 kHz.

The Rust loader reads the local snapshot and does not download checkpoints.
Call `FireRedAsr2::open(model_dir)` and then `transcribe(samples)`. Beam size,
length limit, softmax smoothing, length penalty, and EOS penalty are available
through `transcribe_with_options`.

## Verification

The committed MLX fixture records the direct-float reference WAV, FBANK and
CMVN features, selected first-block stages, final encoder values, and output
transcript. The ignored checkpoint-gated test compares the full frontend and
selected stages, runs all encoder and decoder layers, and checks the exact
transcript. The first Conformer block also has a focused stage-parity test.

Run the checkpoint-gated comparison with:

```sh
TURBOSPARK_FIREREDASR2_DIR=/path/to/pinned/snapshot cargo test -p turbospark-audio pinned_checkpoint_matches_mlx_stages_and_transcript -- --ignored
```

The checked-in evidence is one English clip. It does not qualify broader
transcription quality, memory use, performance, or runtime/catalog/FFI/Swift
integration.
