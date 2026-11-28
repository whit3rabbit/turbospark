# Breeze TTS 2 in Audio

TurboSpark runs Breeze TTS 2 (BreezeBlue, 3B, English and Chinese) through
native Swift MLX, in the vendored Qwen3TTS package
(`swift/TurboSparkApp/Vendor/Qwen3TTS/Sources/Qwen3TTS/Breeze/`). Breeze
bundles the Qwen3-TTS 12 Hz codec unchanged under `audio_tokenizer/`, so the
port shares that package's codec encoder and decoder. The Rust port in
`crates/audio/src/tts/breeze_tts` ends at codebook frames, so the Rust
AudioSession and the HTTP Audio API do not execute this family
(`http_api: false`; there is no server speech bridge for it yet).

The numerical reference is mlx-audio 0.5.7 `tts/models/breeze_tts` at
`e1b19b9054bf163f5d812221a54fcc346f1890e9`. Local deviations are listed in
the vendor `UPSTREAM.md`.

## License

The weights are under the BreezeBlue Research and Non-Commercial License
Agreement: research and non-commercial use only, with commercial use needing
a separate written license. The license also forbids cloning or imitating a
real person's voice without that person's explicit consent. The catalog
profile carries the `breezeblue_noncommercial_license` acknowledgment, which
the app shows before download, and the install keeps `LICENSE` and `NOTICE`.

## Destinations

One profile, `breeze-tts-2-mxfp4`, serves three pages:

| Destination | Inputs |
|---|---|
| Voice Cloning | Text, reference region (1 to 30 s) with its exact transcript, consent for that region, optional style (voice direction) |
| Voice Design | Text, voice description |
| Text-to-Speech | Text, preset speaker `S0` to `S9` |

Breeze speaks a mismatched transcript, so cloning requires the reference's
exact words; there is no embedding-only mode. Consent covers one exact
reference region, as for the other cloners. The request comes from the
shared native-clone path (`nativeCloneRequest`), and the Breeze job refuses a
reference without consent again on its own (`BreezeTTSSpeech.modelRequest`).
An instruction (design or direction) runs classifier-free guidance at 4, the
upstream recommendation; without one guidance does not run. There is no speed
control. A take whose segment reaches 750 frames without the model's end
class is retaken with a fresh seed, like a Drama 3 early stop.

## Model and loading

`breeze-tts-2-mxfp4` pins `mlx-community/Breeze-TTS-2-mlx-4bit` at
`3a06d26b172ea4ae1da2f42d708383e9c79d5526` (about 3.0 GB): `mxfp4` weights,
group 32, for the backbone, text encoder, depth decoder and embeddings. The
codec weights are byte-identical to the pinned Qwen3-TTS
`speech_tokenizer/model.safetensors`. A managed install loads only after its
bytes verify, inside `AudioModelOpening.perform`, as for Qwen3-TTS. Breeze
has its own resident slot (`breezeModel`), released wherever the Qwen model
is.

The loader also accepts the Gust export (`Shoof/Gust-Breeze-TTS-2-MLX-Realtime`:
BF16 with an int8 affine depth decoder, group 64) from a chosen folder. It is
not a catalog profile yet and has not been run here.

## Evidence

Measured on an M4 Max with the pinned 4-bit checkpoint, debug builds:

- Greedy frames match the Python reference exactly until the first bf16
  one-ulp tie (frame 3, codebook 12 for the tested prompt). Python mlx 0.32.3
  and mlx-swift 0.30.6 kernels round differently there, so later greedy
  frames diverge.
- Sampled plain, design, clone, direction and Chinese runs each stopped on
  the end class at a plausible length (3.5 to 5.4 s for one sentence). Peak
  MLX memory was 5.1 to 5.9 GB.
- Not verified: intelligibility by a speech-to-text pass, speaker similarity,
  release-build speed, and the app end to end in the running UI.

Smoke runner:

```sh
cd swift/TurboSparkApp/Vendor/Qwen3TTS
swift run Qwen3TTSSmoke breeze /path/to/Breeze-TTS-2-mlx-4bit /tmp/out.wav \
  'Text to speak.' --ref /path/to/reference.wav --ref-text 'Exact words.'
```

Options: `--instruct`, `--cfg`, `--speaker N`, `--greedy`, `--stream`,
`--max N`, `--all-codes`. The real-checkpoint test needs
`TURBOSPARK_TEST_BREEZE=/path/to/checkpoint swift test`.
