# VibeVoice-ASR

This module ports the Microsoft VibeVoice-ASR speech-to-text family from
`mlx_audio/stt/models/vibevoice_asr/` (vibevoice_asr.py, audio_encoder.py,
config.py) in mlx-audio 0.5.7 at source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`. The language model follows
`mlx_audio/lm/models/qwen2.py` in the same snapshot.

## Overview

VibeVoice-ASR transcribes speech with an encode-and-splice design:

1. Two fully convolutional tokenizer encoders (`TokenizerEncoder` in
   audio_encoder.py) run over the raw 24 kHz waveform. Each is a stride-1
   stem convolution, then six stages of strided downsample convolution plus
   pre-norm blocks (causal depthwise kernel-7 convolution mixer and a
   GELU-erf feed-forward, each with a per-channel layer scale), then a
   stride-1 head projection. The declared ratios `[8, 5, 5, 4, 2, 2]` are
   reversed for encoding into strides `[2, 2, 4, 5, 5, 8]` (product 3200),
   so one latent frame covers 3200 samples. The acoustic encoder emits 64
   latent channels, the semantic encoder 128.
2. Each latent is projected to the LM width by a `Linear -> RMSNorm ->
   Linear` SpeechConnector, and the two projections are added.
3. The chat prompt splices the combined features into the positions of
   repeated `<|box_start|>` tokens between `<|object_ref_start|>` and
   `<|object_ref_end|>` (VibeVoice repurposes Qwen2.5 special tokens). The
   default offline prompt asks for JSON-shaped transcription with the keys
   `Start time, End time, Speaker ID, Content`; the pinned model answers
   with inline `Speaker N:`-marked text instead.
4. A Qwen2 1.5B decoder (28 layers, hidden 1536, GQA 12/2, intermediate
   8960, RoPE theta 1e6, RMSNorm eps 1e-6) prefills the merged embeddings
   and decodes greedily until `<|endoftext|>` (151643) or `<|im_end|>`
   (151645). `tie_word_embeddings` is true, so the token embedding matrix
   is the logits head; the checkpoint's stored `lm_head` tensor is ignored,
   matching mlx_lm's sanitize.
5. Generated ids are decoded with special tokens skipped and the text is
   stripped (upstream `STTOutput.text` behavior). `parse_speaker_segments`
   splits the result on `Speaker N:` markers; upstream's own
   `parse_transcription` only handles JSON output and returns nothing for
   this format, so the splitter is a port-side utility pinned by tests.

The input frontend is part of the port: mlx-audio resamples non-native
input with `scipy.signal.resample_poly` and a kaiser-best FIR (385 taps at
the 3/2 rate pair, rolloff 0.9475937167399596, beta 14.769656459379492,
`padtype="edge"`). `resample_poly.rs` reproduces that exact pipeline in
f64 with an f32 rounding at the end; the Rust output is bitwise equal to
the scipy reference on the pinned smoke clip (see the fixture digest gate)
and on every rate pair and signal length cross-checked during development.

## The upstream load path is broken for the pinned checkpoint

Automatic routing cannot load `microsoft/VibeVoice-ASR-Streaming-1.5B` in
the reference environment (reproduced at the pinned commit):

1. `config.json` carries `model_type: "vibevoice"`. The installed
   transformers maps that to its native `VibeVoiceConfig`, whose
   architecture validation rejects the 1.5B geometry
   (`diffusion_head_config.hidden_size` 1536 must match `text_config`
   4096), so `AutoTokenizer.from_pretrained` raises before any weights
   load.
2. Upstream `post_load_hook` swallows that error and falls back to a
   remote `Qwen/Qwen2.5-7B` tokenizer, which lacks the streaming
   `<|text_chunk_end|>` token; the stock loader then fails with
   "This VibeVoice streaming checkpoint is missing the <|text_chunk_end|>
   tokenizer token".

The verified path, which both the Python fixture script and this Rust port
use: read the raw safetensors tensors with shape checks (PyTorch conv
layout `[out, in, kernel]`, exactly what the portable conv kernel takes,
so no checkpoint tensor is transposed), build the prompt string from the
chat template the upstream hook installs, and load the tokenizer from the
checkpoint's own `tokenizer.json`. The Python reference loads the tokenizer
as a direct `Qwen2Tokenizer` to skip the broken AutoConfig validation.

## Scope: the offline single-window path only

The pinned smoke path is the offline branch of upstream `generate` on one
clip with default prompts and no hotwords: resample, encode both
tokenizers, splice, prefill, greedy decode. Everything else the upstream
file implements is refused or out of scope for this port:

- Streaming chunked transcription (`is_streaming_model`,
  `streaming_generate`, `streaming_generate_step`,
  `init_streaming_state`): out of scope. The pinned checkpoint is trained
  for the streaming protocol (`chunk_frames` 22, `lookahead_frames` 4,
  `<|text_chunk_end|>` 151665), and the stock upstream `generate` takes
  the streaming branch, but this port pins the offline single-window
  branch; the streaming state machine, per-chunk delimiters, and chunk
  text assembly are not ported and are refused by omission.
- Diarization controls, timestamps, and long-form trimming (59-minute
  cap): the model emits speaker markers inline and the transcript keeps
  them, but no segment/time parsing is implemented beyond the marker
  splitter.
- Hotwords and `context` folding (`merge_hotwords`, the `extra info`
  prompt branch): refused; only the no-context prompt suffix is built.
- Sampling controls (temperature, top-p/top-k/min-p, repetition penalty):
  only the greedy temperature-0 path is ported.
- `normalize_audio: true` (the reference loudness normalizer): refused at
  load; the pinned profile trains with normalization off.
- Non-pinned geometry: RoPE scaling, sliding-window attention, untied
  embeddings, non-causal or LayerNorm encoder variants, non-depthwise
  mixers, and MLX affine-quantized tensors (`.scales`/`.biases`) are
  refused with reasons at load.

## Pinned profile

- Hugging Face repository: `microsoft/VibeVoice-ASR-Streaming-1.5B`
- Immutable revision: `4262d23d8a539a6530cf64fbd0b1751ef9a30853`
- Input: mono f32 PCM at 16 kHz (resampled to 24 kHz internally); native
  rate 24 kHz, one latent frame per 3200 samples
- Tokenizer encoders: 7 stages (depths 3-3-3-3-3-3-8), 32 base filters,
  channel growth 32 to 2048, acoustic vae_dim 64, semantic vae_dim 128
- Decoder: Qwen2 1.5B, vocab 151936, tied embeddings
- Precision: plain BF16 checkpoint tensors, dequantized to f32 at load;
  no MLX quantization scheme present or accepted
- Rust profile: `VIBEVOICE_ASR_STREAMING_1_5B`

Artifact sizes are bytes; SHA-256 digests were computed from the downloaded
snapshot at the pinned revision. The sixteen files total 5,645,212,528
bytes.

| File | Size | SHA-256 |
|---|---:|---|
| `.gitattributes` | 1735 | `c50aecad5c712e61fc8e2af657385e3c108df6da345cfa324b4e6d42f520e17d` |
| `README.md` | 2238 | `5e4050aec9e328c5e6ef21973ee6c97660bfcd587da9610a585e028ae97f24c1` |
| `added_tokens.json` | 714 | `d5401f76119bc71e1844d8e909826f8cb3d68e152a058fe228cc37f74603ca84` |
| `config.json` | 2941 | `2b46a311855872193b218bc8d4de499e3252cd3c3c06662128d16e3f9e8bc72f` |
| `figures/VibeVoice_ASR_Streaming_architecture.png` | 191728 | `2b38df286fe4d133d3ccbaf97e3195be349aca09dad011ff463b8d3b656c678f` |
| `figures/VibeVoice_ASR_Streaming_results.png` | 623135 | `fa83f98d6506ea7ed9039e0c886baa375a9614a12ad915d71009bb76b6b15535` |
| `merges.txt` | 1671853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `model-00001-of-00003.safetensors` | 2498354152 | `788e6206a158c614e1e6777ab843baf64ca8a3f555e3cb72c26d5d79b925d4b9` |
| `model-00002-of-00003.safetensors` | 2486576066 | `4d58b3ce47ac6b0ac5d3b142d54e56f10e02a928cf016b493e9e6a3f91beb75d` |
| `model-00003-of-00003.safetensors` | 643458072 | `884b560dcd3024436d8827d32c7e211c545751c099df4ee75101c1ee2196f427` |
| `model.safetensors.index.json` | 120114 | `beac94f964e3358f8c66e7dcdbc700e5518e5cf2a7b7697a0c156ff7bcf89038` |
| `preprocessor_config.json` | 191 | `59ee0cc74687fc2ab8f5f9c2fc3b61cdf66078ebd4f34a61e1fcfc1a644ac31a` |
| `special_tokens_map.json` | 769 | `7093fea72a118fb4016ee85c13cc2f94ecd659d4600c6ce16cdcd3a2c1d96e68` |
| `tokenizer.json` | 11422657 | `d198a051741ee47805797b744e6524c259b08a59b58d1ec159cfd7f1da5f2df7` |
| `tokenizer_config.json` | 9330 | `e80c7714aaff125403ba7c18a4c06da9b18eb81cb6ee4bee711143fa15d94fd6` |
| `vocab.json` | 2776833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

## Inference contract

- `VibeVoiceAsr::transcribe(samples)` takes one mono 16 kHz waveform and
  returns the stripped transcript text. `resample_input`,
  `encode_speech`, and `build_prompt` expose the stages between for
  parity work.
- The resampled 24 kHz window encodes to `ceil(samples / 3200)` latent
  frames (21 for the pinned smoke clip); the prompt contains that many
  `<|box_start|>` placeholders and the model refuses a placeholder count
  that disagrees with the encoder.
- Prompt ids are built from the fixed offline chat template; the gated
  test pins all 83 ids against the reference run, which also pins the
  tokenizer and the `{:.2}` duration formatting.
- Linear weights load in the HF `[out, in]` layout; conv weights stay in
  the stored PyTorch `[out, in, kernel]` layout the portable conv kernel
  consumes. A missing or misshaped tensor is refused by name.
- Token ids crossing the tokenizer interface are signed 32-bit; generated
  ids convert to unsigned only for the decode call, mirroring the other
  Qwen families.

## Reference and parity

`generate_vibevoice_asr_fixture.py` runs the pinned checkpoint through the
reference classes (manual sanitize, tied-head lm_head dropped, direct
`Qwen2Tokenizer`) and records the resampled waveform digest and spots, the
acoustic and semantic tokenizer latents, both connector outputs, the
combined speech features, the prompt text and ids, the prefill hidden row,
the first-decision logits, every greedy step with its top log-probability,
the generated ids, and the transcript into
`crates/audio/testdata/vibevoice_asr_reference.json` (force-added; the
workspace `.gitignore` covers testdata directories). Tests never download
model files.

Regenerate on an Apple Silicon host with the mlx-audio reference
environment:

```sh
/Users/whit3rabbit/Documents/GitHub/.venv-mlxaudio/bin/python \
  crates/audio/scripts/generate_vibevoice_asr_fixture.py \
  --model-dir ~/models/vibevoice-asr-streaming-1.5b \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/vibevoice_asr_reference.json \
  --revision 4262d23d8a539a6530cf64fbd0b1751ef9a30853
TURBOSPARK_VIBEVOICE_ASR_MODEL_DIR=~/models/vibevoice-asr-streaming-1.5b \
  cargo test --release -p turbospark-audio --lib stt::vibevoice -- \
  --ignored --nocapture --test-threads=1
```

- **Python reference transcript (pinned checkpoint):**
  `Speaker 0:The quick brown fox jumps over the lazy dog.` (the raw decode
  is ` Speaker 0:...` with one leading space; the strip matches upstream's
  `STTOutput.text`). The stock upstream `generate()` returns the same
  transcript through its streaming branch on this clip; the offline branch
  is the pinned contract.
- **Rust checkpoint run:** identical transcript byte for byte, identical
  83-token prompt, and identical 14 generated token ids ending in the
  streaming chunk delimiter 151665 the text decoder drops. Release wall
  time about 13 s including weight load and the full transcribe pass.
- **Stage parity (f32 port vs the reference run, measured worst spot):**
  the resampled 24 kHz waveform matches bitwise (the fixture's f32 byte
  digest reproduces exactly), tokenizer encoder latents `1.6e-5` acoustic
  and `2.4e-5` semantic, combined speech features `3.3e-6`, prefill hidden
  row `5.8e-5`, first-decision logits `6.1e-5` on ~15-magnitude values,
  and per-step top log-probabilities `4.3e-2` where bf16 drift
  accumulates over the decode steps. The 83 prompt ids and all 14 greedy
  decisions are exact. Always-on gates: resampled spots `1.0e-6` plus the
  digest, encoder spots `5.0e-4`, combined `5.0e-3`, prefill hidden
  `2.0e-2`, first logits `2.0e-4`, step log-probabilities `5.0e-1`.

Always-on tests (no checkpoint required) pin the config refusals, the
speaker-marker parser, the fixture provenance and transcript, the prompt
contract, the prefill/decode cache equivalence on a tiny Qwen2, the
scipy-aligned resampler design values, and the resampled 24 kHz waveform
against the fixture spots and its f32 byte digest.

## Verification status

Both the Python reference and the Rust release-mode checkpoint run were
executed on this machine against the pinned revision. One short English
clip does not qualify recognition quality, multi-speaker behavior, long
audio, or the streaming protocol this port refuses. Performance, memory,
runtime, catalog, FFI, and Swift gates remain separate work.
