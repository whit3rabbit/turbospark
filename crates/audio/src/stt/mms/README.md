# MMS 1B FL102 English

This profile ports the English CTC path from `mlx_audio/stt/models/mms/` and
its shared `mlx_audio/stt/models/wav2vec/` backbone in mlx-audio 0.5.7 at
source commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Pinned profile

- Hugging Face repository: `facebook/mms-1b-fl102`
- Immutable revision: `d483345545bea550895b1aa0c6ba40236b9f1e22`
- Language adapter: English (`adapter.eng.safetensors`)
- Input: mono f32 PCM at 16 kHz
- Base architecture: Wav2Vec2, 48 stable-layer-norm encoder blocks
- Rust profile: `MMS_1B_FL102_ENGLISH`

The reference profile loaded in mlx-audio 0.5.7 with the explicit
`model_type="mms"` override and transcribed the generated 16 kHz smoke clip as
`the quick bround fox jumps over the laisy dog`. Automatic routing sees the
checkpoint's `model_type="wav2vec2"` and fails because there is no standalone
`stt.models.wav2vec2` module. The checkpoint directory was removed after the
reference run. This one clip is a smoke test, not a recognition-quality
measurement.

The pin's relevant artifact sizes are bytes. Large-file hashes come from the
Hub LFS metadata at the immutable revision; JSON hashes were verified from
the downloaded files.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 3859131696 | `38de641c5ac2ea23f27d672ccfc7dd671a919696cc4750fe4a64a6368aca4993` |
| `adapter.eng.safetensors` | 9039376 | `14a55521004fd3adb526cf128764df496ab6d2a5a0b3e0b6f04b7f387b6cd714` |
| `config.json` | 2039 | `226c04148b448eaf64410235d974dbe46f7f34b91da9fd49b72fdd160f331dbb` |
| `preprocessor_config.json` | 254 | `f724fdad65341e79bd9741888fe4e8ce1bb2e7b44fc8216c28c591afb1e202f4` |
| `tokenizer_config.json` | 397 | `42a65547cf0654d12263ece7c909c4b0c574b660c78658f806c772cc223bc01b` |
| `special_tokens_map.json` | 96 | `9046da57c270c8e74d0f38832b4adce269c9d914ef21d2a0925e7772152dd793` |
| `vocab.json` | 350797 | `0566ed8d1df0e87db829b9e4001ce061ba0bfb29469e50fba0e8ef8c2596c30e` |

## Inference contract

- The waveform is normalized to zero mean and unit variance with an epsilon of
  `1e-7` before Wav2Vec2 feature extraction.
- The seven valid convolutions use layer normalization, then a feature
  projection, grouped weight-normalized positional convolution, and 48
  stable-layer-norm transformer blocks. Each block applies the language adapter
  from `adapter.eng.safetensors`.
- The English CTC head comes from the same adapter file. Greedy decoding
  collapses adjacent repeats, removes blank token 0, concatenates vocabulary
  pieces, and converts `|` to a space.
- Other MMS language adapters and `facebook/mms-1b-all` are not part of this
  profile.

## Verification status

The upstream MLX checkpoint smoke and immutable artifact pins are recorded
above. The initial Rust Wav2Vec2 encoder and MMS wrapper are implemented, and
the focused config/profile/decoder tests pass. A Rust release-mode run returned
the exact same transcript on the same WAV. The checked-in MLX fixture captures
sparse stage values for diagnosis; it does not establish numerical parity. The
largest sampled absolute difference grew from `8.4e-5` at the first raw
convolution to `0.489` at encoder layer 47 and `1.159` at sampled CTC logits.
The exact transcript is a one-clip smoke result, not broader quality evidence.
The checkpoint-gated transcript test is ignored in ordinary test runs because
the pinned base checkpoint is 3.86 GB.

On the Apple Silicon reference machine, download only the immutable revision,
generate the fixture with the same `../mlx-audio` checkout used for the port,
then run the checkpoint test:

```sh
export TURBOSPARK_MMS_CACHE=/tmp/turbospark-mms-hf-cache
export TURBOSPARK_MMS_DIR="$(../mlx-audio/.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("facebook/mms-1b-fl102", revision="d483345545bea550895b1aa0c6ba40236b9f1e22", cache_dir="/tmp/turbospark-mms-hf-cache"))')"
../mlx-audio/.venv/bin/python crates/audio/scripts/generate_mms_fixture.py \
  --model-dir "$TURBOSPARK_MMS_DIR" \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/mms_reference.json
CARGO_TARGET_DIR=/tmp/turbospark-audio-mms-target CARGO_BUILD_JOBS=1 \
  cargo test --release -p turbospark-audio --lib \
  models::stt::mms::tests::pinned_english_checkpoint_matches_mlx_reference \
  -- --ignored --nocapture --test-threads=1
python3 -c 'from pathlib import Path; import shutil; shutil.rmtree(Path("/tmp/turbospark-mms-hf-cache"), ignore_errors=False)'
```

The Rust path is a CPU implementation; this command does not qualify Metal
performance. Sparse stage values are printed as diagnostics, not asserted as a
parity gate. Resolving the growing numeric drift, other language adapters,
quality evaluation, runtime/catalog integration, FFI, Swift, performance, and
memory qualification remain open.
