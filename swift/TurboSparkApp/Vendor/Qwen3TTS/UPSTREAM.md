# Qwen3-TTS provider provenance

This package contains the Qwen3-TTS subset of
[mlx-audio-swift at a0bd39c923e1359d5deb3d049c89fa5585cf479a](https://github.com/Blaizzy/mlx-audio-swift/tree/a0bd39c923e1359d5deb3d049c89fa5585cf479a).
The upstream MIT license is retained in LICENSE. The copied files cover the
talker, code predictor, speaker encoder, speech tokenizer, generation types,
DSP support, and the tokenizer's Mimi convolution/transformer components.
Other upstream model families and download code are excluded.

MLX Swift stays on the app's 0.30.x dependency line, with MLXLMCommon 2.30.6.
This avoids importing the upstream package's unrelated models and its newer
MLX dependency into the app. GenerateParameters is a local value type because
the compatible MLXLMCommon release does not expose all Qwen sampling controls.

The numerical reference is
[mlx-audio 0.5.7 at e1b19b9054bf163f5d812221a54fcc346f1890e9](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/qwen3_tts).

Local changes:

- Load local folders only. TurboSpark's catalog owns downloads, immutable
  file plans, receipts, and byte verification.
- Check family, variant, geometry, codec configuration, and affine quantization
  before constructing the model. Refuse incomplete checkpoints.
- Generate tokenizer.json in private temporary storage, leaving the source
  checkpoint unchanged. Never delete a malformed source folder.
- Require the Base speaker encoder and require the audio encoder for cloning
  with a reference transcript. Missing conditioning must fail.
- Use the reference's symmetric Hann window, reflect padding, Slaney magnitude
  mels, and natural logarithm for speaker extraction. Whisper normalization is
  a different frontend. The golden fixture records the pinned Python result.
- Keep the cached reference MLXArray alive so a recycled object address cannot
  return another recording's conditioning.
- Expose a synchronous synthesize operation for the app's detached producer.
  Check cancellation during loading, generation, and bounded codec decoding.
- Refuse a take that exhausts its token limit instead of silently cutting it.
- Decode bounded chunks using the reference's left context. Codec token zero
  is valid; generated sequences contain no padding. Trim reference audio by
  its exact codec frame count.
- Map ISO language codes to the checkpoint's language names.

Regenerate speaker-mel.json from the pinned Python mel_spectrogram using the
fixture's samples. Tests never download checkpoints. Qwen3TTSSmoke is an
explicit real-model runner; its usage and evidence limits are documented in
[the project guide](../../../../docs/QWEN3_TTS.md).
