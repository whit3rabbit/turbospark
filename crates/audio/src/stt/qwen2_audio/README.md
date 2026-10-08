# Qwen2-Audio (offline transcription)

Offline single-window transcription for the Qwen2-Audio multimodal
audio-understanding family: a whisper-family log-mel frontend, a conv +
attention-pooling audio encoder, a linear projector, and the Qwen2 language
model over a spliced audio-and-text chat prompt with greedy decoding.

- Upstream reference: `mlx_audio/stt/models/qwen2_audio/`
  (`qwen2_audio.py`, 590 lines, `config.py`) and
  `mlx_audio/lm/models/qwen2.py`, at mlx-audio 0.5.7, commit
  [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9).
- Module files: `mod.rs` (config, frontend, prompt, splice, greedy),
  `encoder.rs` (tower + projector), `language_model.rs` (Qwen2 decoder).
- Fixture generator: `crates/audio/scripts/generate_qwen2_audio_fixture.py`
  (run with the mlx-audio venv Python).
- Fixture: `crates/audio/testdata/qwen2_audio_reference.json`.

## Scope

Ported: the offline single-audio transcription path of upstream
`Model.generate` only.

Refused, with the enforcing check in the module:

| Feature | Why refused |
|---|---|
| Multi-audio batching (`List[audio]`) | The pinned transcription path is one window; batching changes prompt assembly ("Audio 2:") and embedding concatenation that no fixture verifies. |
| Streaming decode (`stream=True`) | Upstream yields per-token `StreamingResult`s; the offline port returns the full transcript. No streaming protocol was verified. |
| Task prompts beyond the default | Captioning, emotion, editing, and other audio-understanding prompts change the instruction text; only `Please transcribe the speech.` is verified. `build_prompt_string` accepts a user prompt for future use, but no task quality gate exists. |
| Any other quantization than 4-bit group 64 | `Qwen2AudioConfig::from_json` refuses other `quantization` values; the dequantizer refuses scheme mismatches at row width. |
| Decoder variants (`tie_word_embeddings=true`, `attention_bias=false`, `rope_scaling`, sliding-window attention, non-silu MLP) | Unverified geometry changes the decode flow; each knob has an explicit refusal in the config parser. |
| Audio geometry deviations | Any present `audio_config` key must equal the pinned value; absent keys take the pinned default. |

## Pinned checkpoint profile

Repository `mlx-community/Qwen2-Audio-7B-Instruct-4bit`, revision
`c65570002626f41b4dc08b7b54f42f99f3e82e7f`. Total 6,574,659,991 bytes
across the 9 repository files below (sizes in bytes):

| File | Size | SHA-256 |
|---|---|---|
| weights.safetensors | 6,562,540,479 | `0967cde270ad62aa4824f0bbce283d0a2a6da2825ccf587640753c527d4174da` |
| tokenizer.json | 7,028,015 | `f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8` |
| vocab.json | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| merges.txt | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| tokenizer_config.json | 638,335 | `c738158e70eeecf25736a6c4d8e5f34cfce683079980757edfb0665a7b2457ed` |
| README.md | 1,716 | `1cddc760431c9c2c043ed4d17d9c17716a1f81978e6093389420cd46e7f2ce6d` |
| .gitattributes | 1,519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |
| config.json | 913 | `39493dad1678533cc2a7ba2de0a573cc926a78ef4c7abef72a6b660159de94b7` |
| preprocessor_config.json | 342 | `4cd7c6c061fe79244c57b0c320b2873f3ee5acce2277f7cc3aced042725680f2` |

Geometry: audio tower d_model 1280, 32 layers, 20 heads, FFN 5120, 128 mel
bands, 1500 max source positions; decoder hidden 4096, 32 layers, 32
heads, 32 KV heads, vocab 156032, RoPE theta 10000, RMSNorm eps 1e-5;
every decoder linear (including the untied `lm_head`) is MLX
affine-quantized 4-bit with 64-value groups; the audio tower and projector
are BF16.

Install:

```sh
hf download mlx-community/Qwen2-Audio-7B-Instruct-4bit \
  --revision c65570002626f41b4dc08b7b54f42f99f3e82e7f \
  --local-dir ~/models/qwen2-audio-7b-instruct-4bit
```

## Inference contract

1. Audio must be mono f32 at 16 kHz. It is zero padded or truncated to the
   fixed 480,000-sample window.
2. Frontend (upstream `_extract_features`): reflect padding of 200, 400-point
   frames at hop 160 (3001 frames), periodic Hann window, f64 power
   spectrum, projection onto the inline HTK-mel filterbank (integer bins
   `floor(401 * hz / 16000)`, no area normalization; 21 low bands have
   all-zero weights by construction), `log10` with a 1e-10 floor, global
   peak clamp at `max - 8`, `(x + 4) / 4`.
3. Tower: conv1 (kernel 3, pad 1) + GELU, conv2 (kernel 3, stride 2, pad 1)
   + GELU, additive sinusoid positions, 32 pre-norm layers with full
   bidirectional self-attention (k_proj has no bias), non-causal pair
   average to 750 rows, final LayerNorm. Conv kernels load from the
   checkpoint's MLX layout `[out, kernel, in]` and transpose once to the
   PyTorch layout.
4. Projector: one biased linear to the decoder width (750 x 4096).
5. Prompt (upstream `_build_prompt` + the pinned chat template):
   `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|>x750<|audio_eos|>\nPlease transcribe the speech.<|im_end|>\n<|im_start|>assistant\n`,
   encoded to 783 ids. Token loading needs the reconciliation described
   below. The 750 `<|AUDIO|>` (151646) positions take the projected feature
   rows; every other position embeds its id (placeholders embed id 0
   before being overwritten).
6. Decode: causal prefill, then greedy argmax until `<|im_end|>` (151645,
   the tokenizer's `eos_token`); special tokens are skipped in the
   detokenization.

Tokenizer note: the pinned `tokenizer.json` carries only the three Qwen
chat tokens (151643-151645). The audio and timestamp control tokens
(151646 through 154930) exist only in `tokenizer_config.json
added_tokens_decoder`. The loader adds the missing tokens in id order from
that block, which is contiguous, and refuses if any added token does not
retain its id.

## Divergences from the reference

- The reference evaluates the tower and the spliced embeddings in bfloat16
  (checkpoint dtype); this port computes in f32 from the same
  bf16-rounded weights. Fixture gates absorb the difference; generated
  token ids and the transcript match exactly.
- The 400-point real FFT runs as a plain f64 DFT instead of numpy's
  pocketfft (agreement far inside the fixture gates).
- The Rust transcript is the detokenized text; upstream returns the same
  string without stripping (identical on the pinned run).

## Verification evidence

All numbers from the pinned profile and the shared smoke WAV
`crates/audio/testdata/qwen2_audio_reference.wav` (actually
`qwen3_forced_aligner_reference.wav`, SHA-256
`ea38d350100b5d532fbc3e3517499dbd9915b9b0890d9e819bae384fa1384bcb`).

Python reference (venv, mlx-audio at the pinned commit):
`mlx_audio.stt.utils.load_model` + `Model.generate` on the smoke WAV
returned `"The quick brown fox jumps over the lazy dog."` (783 prompt
tokens, 10 generated tokens).

Rust checkpoint run (release, `stt::qwen2_audio` gated test, 320 s wall
clock on an M4 Max, peak roughly 31 GB resident for the dequantized f32
decoder), fixture worst absolute spot differences:

| Stage | Worst diff | Gate |
|---|---|---|
| input_features (mel) | 1.93e-4 | 5.0e-3 |
| encoder_output (attention pooling) | 4.77e-2 | 1.0e-1 |
| projected | 6.44e-2 | 1.5e-1 |
| inputs_embeds rows | 2.13e-2 | 1.0e-1 |
| prefill hidden last row | 2.61e-1 | 5.0e-1 |
| first logits watch values | 1.81e-1 | 5.0e-1 |
| per-step top log-prob | 3.61e-2 | 2.5e-1 |

The residual is the documented bf16-vs-f32 divergence: it grows through
the 32-layer tower and decoder but stays at a few percent of the value
magnitudes, while the discrete decisions are unaffected. Generated token
ids `[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]`, all 783
prompt ids, all audio positions, every greedy step argmax, and the
transcript `"The quick brown fox jumps over the lazy dog."` match the
fixture exactly.

Always-on tests (no checkpoint): 9 tests cover the profile pin, fixture
provenance (including the WAV digest), transcript/greedy shape, config
parsing with refusal of every unverified knob, prompt string assembly,
silence-floor frontend behavior, and filterbank edge fidelity.

Regenerate the fixture:

```sh
/Users/whit3rabbit/Documents/GitHub/.venv-mlxaudio/bin/python \
  crates/audio/scripts/generate_qwen2_audio_fixture.py \
  --model-dir ~/models/qwen2-audio-7b-instruct-4bit \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/qwen2_audio_reference.json \
  --revision c65570002626f41b4dc08b7b54f42f99f3e82e7f
```

Run the checkpoint gate (minutes on CPU f32; the dequantized decoder needs
about 31 GB of resident memory):

```sh
TURBOSPARK_QWEN2_AUDIO_MODEL_DIR=~/models/qwen2-audio-7b-instruct-4bit \
  cargo test --release -p turbospark-audio --lib stt::qwen2_audio \
  -- --ignored --nocapture --test-threads=1
```

## Verification status and remaining gates

| Gate | Status |
|---|---|
| Implementation + always-on tests | Done (this module; 9 tests). |
| Fixture parity, frontend/tower/projector/prompt/prefill/greedy | Done via the gated test against the Python reference fixture. |
| Checkpoint end-to-end transcript parity | Done on the pinned profile (exact match, release, 320 s). |
| Portable cross-target compile | Done: `cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio` passes with this family. |
| Task quality beyond the smoke WAV (WER, other languages, long audio) | Not run. |
| Runtime/Metal, catalog, FFI, Swift integration | Not started; the family is CPU f32 inside `crates/audio` only. |
| Multi-audio, streaming, task prompts | Refused (see above), no gates. |
