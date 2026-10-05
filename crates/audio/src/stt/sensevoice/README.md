# SenseVoice Small

SenseVoice Small is a non-autoregressive CTC recognizer with language,
emotion, and event tags. The Rust implementation follows
`mlx_audio/stt/models/sensevoice/config.py` and `sensevoice.py` in
mlx-audio 0.5.7 at source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Pinned profile

- Hugging Face repository: `mlx-community/SenseVoiceSmall`
- Immutable revision: `8ddd966bd96243cff196422f81f0c5d955814792`
- Model file: `model.safetensors`, F32, 936,100,124 bytes
- Required sidecars: `config.json`, `am.mvn`, and
  `chn_jpn_yue_eng_ko_spectok.bpe.model`
- Supported architecture: 560 input features, 50 SANM encoder blocks, 20 TP
  blocks, 512 output features, 4 attention heads, and a 25,055-token CTC head

The Rust loader consumes an already materialized snapshot. It does not fetch
weights, convert precision, or accept packed quantization. Inputs are mono
16 kHz float samples. Resample and mix down before calling `transcribe`.

## Usage

```rust,no_run
use std::path::Path;
use turbospark_audio::stt::SenseVoiceSmall;

let model = SenseVoiceSmall::load(Path::new("/models/SenseVoiceSmall"))?;
let text = model.transcribe(&mono_16khz_samples)?;
```

`transcribe_with_options` accepts `auto`, `zh`, `en`, `yue`, `ja`, `ko`, or
`nospeech`, and returns recognized text plus language, emotion, and event
tags. `use_itn` selects the model's text-normalization query token.

## Reference and checks

The pinned Python MLX model returned `the quick brown fox jumps over the lazy dog`
on `testdata/qwen3_forced_aligner_reference.wav`, SHA-256
`ea38d350100b5d532fbc3e3517499dbd9915b9b0890d9e819bae384fa1384bcb`.
The fixture generator reads the WAV as float32 and passes the sample array
directly to the MLX model, matching the Rust samples API. Frontend values and
selected tensors at four encoder stages are recorded in
`testdata/sensevoice_reference.json`. The unit test checks raw FBANK and
post-CMVN features within `0.01`; the checkpoint test checks selected values
at each recorded stage within `0.01` and verifies the full CTC token sequence.

Regenerate the fixture with the `mlx-audio` 0.5.7 checkout at the pinned source
commit and a local snapshot of the pinned model:

```sh
HF_HOME=/tmp/turbospark-sensevoice-hf \
PYTHONPATH=../mlx-audio \
../mlx-audio/.venv/bin/python crates/audio/scripts/generate_sensevoice_fixture.py \
  --model-dir /path/to/SenseVoiceSmall \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/sensevoice_reference.json
```

The unit frontend parity test requires no model download. The real-checkpoint
test is ignored by default. Run it with:

```sh
TURBOSPARK_SENSEVOICE_MODEL_DIR=/path/to/SenseVoiceSmall \
cargo test -p turbospark-audio pinned_checkpoint_matches_mlx_transcript_and_tags -- --ignored
```

This smoke witness covers one English clip and does not establish broad
transcription quality, latency, memory qualification, or runtime integration.
