# Higgs Audio v3 STT

Boson AI speech-to-text: a whisper-style audio encoder feeds an MLP
projector into a Qwen3 text backbone that emits the plain lowercase
transcription. On the shared smoke clip the pinned checkpoint produces:

```
the quick brown fox jumps over the lazy dog.
```

The model adds the final period despite the no-punctuation default prompt;
the fixture records the reference output verbatim.

## Overview

Three parts:

- `HiggsAudioEncoder` (`encoder.rs`): the 128-band whisper-style tower. Two
  GELU convolutions (kernel 3, padding 1; the second with stride 2), learned
  position embeddings added on the time axis, 32 encoder blocks with
  per-layer pre-norm (`self_attn_layer_norm` -> attention -> residual, then
  `final_layer_norm` -> GELU fc1 -> fc2 -> residual), pairwise time-mean
  pooling, and a final layer norm. d_model 1280, 20 heads, FFN 5120, 1500
  source positions. The attention carries a q bias and no k bias; v and out
  projections carry biases.
- `FeatureProjector` (`encoder.rs`): depthwise stride-2 conv (kernel 3,
  padding 1, 1280 groups) halving the time axis, then Linear(1280, 2048),
  ReLU, Linear(2048, hidden 2048).
- Text backbone: the shared `qwen3_asr` decoder (hidden 2048, 28 layers,
  16 query heads, 8 key-value heads, head dim 128, RoPE theta 1e6, q/k
  RMS norms) with an UNTIED LM head. The HF config flag says
  `tie_word_embeddings: true`, but the reference forces the flag off for the
  Qwen3 args and the checkpoint ships the separate LM head under
  `audio_decoder_proj.text_lm_head.weight`; the port loads the untied head.

The generate path (`mod.rs`) mirrors the reference `Model.generate`:

1. optional VAD chunking (`chunk_size_seconds` 4.0): the Silero backend
   (`vad.py` semantics, threshold 0.5, 250 ms minimum speech, 100 ms minimum
   silence, 30 ms padding) provides speech ranges; ranges merge into
   waveform-covering spans (`split_vads: false`) whose last span always
   reaches the clip end, and every span tiles into chunks of at most 4
   seconds starting from its own start. A missing or failing backend falls
   back to a uniform split of the whole clip, exactly like the reference's
   empty-cuts path. Shorter chunks are zero-padded to the longest chunk
   before the frontend.
2. per chunk: the 128-band log-mel frontend (centered STFT with reflect
   padding and the symmetric Hann window the reference `hanning()` produces,
   Slaney filterbank, final STFT frame dropped, `log10` with a 1e-10 floor,
   global peak minus 8 clamp, `(x + 4) / 4`), encode, pool, project.
3. prompt assembly: `<|im_start|>user\n` + prompt + `<|audio_bos|>` + one
   `<|AUDIO|>` placeholder (151672) per chunk + `<|audio_eos|>` +
   `<|im_end|>\n` + `<|im_start|>assistant\n`; the placeholder ids are
   dropped from the embedding sequence and the projected audio rows are
   spliced in their place.
4. one prefill over the injected embeddings, then greedy argmax decoding,
   stopping at the Qwen EOS ids 151645 (`<|im_end|>`) and 151643
   (`<|endoftext|>`) before emission.
5. output cleanup (`_parse_output`): complete `<think>...</think>` spans are
   removed (non-greedy, newlines allowed), an unclosed `<think>` keeps only
   the text after the marker, every same-line `<|...|>` span is removed, and
   the result is trimmed.

The default prompt is the reference `DEFAULT_PROMPT`:
`Transcribe the speech. Output only the spoken words in lowercase with no punctuation.`

## Upstream reference

- `mlx_audio/stt/models/higgs_audio_3/higgs_audio_3.py`
- `mlx_audio/stt/models/higgs_audio_3/audio.py`
- `mlx_audio/stt/models/higgs_audio_3/config.py`
- `mlx_audio/stt/models/higgs_audio_3/vad.py`
- Dependencies reused by the reference:
  `mlx_audio/lm/models/qwen3.py` (`Qwen3Model`),
  `mlx_audio/dsp.py` (`hanning`, `stft`, `mel_filters`).
- mlx-audio 0.5.7, commit
  [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio).
  The Rust module header of every file names its reference file.

## Pinned checkpoint profile

Repository
[`bosonai/higgs-audio-v3-stt`](https://huggingface.co/bosonai/higgs-audio-v3-stt)
at revision `2ffd1aa39f5a1266931e405cba12e404a9f994b2`. 29 files,
5,367,429,273 bytes total, all tensors BF16 (no quantization). Install
locally; tests never download:

```sh
hf download bosonai/higgs-audio-v3-stt \
  --revision 2ffd1aa39f5a1266931e405cba12e404a9f994b2 \
  --local-dir ~/models/higgs-audio-v3-stt
```

| File | Bytes | SHA-256 |
|---|---|---|
| model-00001-of-00002.safetensors | 4,728,828,400 | `ee2160c035cef428edd4a0dca35fef31cb49531776677f0b597c0fc859599580` |
| model-00002-of-00002.safetensors | 622,354,784 | `ad37c68f8dd3f97d32027b54b1ba7f10428ac2dcd0e675ee7bcafd3ef68ba7c3` |
| model.safetensors.index.json | 65,718 | `717c57575604b803774308efa7b7dfedc5394cc82ad1a293a651840362629f30` |
| tokenizer.json | 11,425,740 | `a94e8e77dc484e3f37cd2a82a940a41bb918918defb7a36f6b517657c1616018` |
| vocab.json | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| merges.txt | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| tokenizer_config.json | 12,656 | `97441a379bb20beb786abf288ef0da54ad1fa2bcf7e0e3bdd3fb57dd3f23cb0e` |
| added_tokens.json | 1,185 | `b9599c6bc310e3af8490040df1782cdef4ecb2d9f08dcad2ba1869cdb94c61ec` |
| special_tokens_map.json | 613 | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| config.json | 5,909 | `3bd04c7ca043c76b4d7812abf395eaffc4e93d2e4fa28d9e417b79ce03b4d783` |
| generation_config.json | 147 | `c1172416c87609377570678ff222c8b18455ceee3dd240f8a87113fabec89ac8` |
| MERGE_PROVENANCE.json | 685 | `623ccbb286c848e8f288cf00f8aa4c36816dcb13b1bf224375a9784fadd50a10` |
| README.md | 6,022 | `8f5eddec2afd097c3a3b7d98b9e1f8e9911ed036fe550c43c1cd6e5a495e2a51` |
| .gitattributes | 1,570 | `34448b82c17d60fecb65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930` |
| attention.py | 8,710 | `ca26f0d2e16fdf57313e8ee639f7728a22833ea37bbb323fe79b72cdbdd04646` |
| common.py | 1,094 | `8fb3fc531d985f5d60ff40cdb0f6214ef0715a0d0643e74e566fa89eeabd49d0` |
| configuration_higgs_audio.py | 9,882 | `5c3584743c71d326d6d4a40389639cfaace8a17ac006bb27e8c522ba44ed18e3` |
| cuda_graph_runner.py | 5,144 | `a4d67d89e23334304dd831ff049d2df357f9e8e25a3761587eb549d093db04d5` |
| custom_modules.py | 6,186 | `eb1a8d99d4aae8475b7a0f06f28d63f9014fcbf953fb31567c1f61fe4882f5c4` |
| higgs_audio_collator.py | 36,643 | `77c72b98bc9daff4669eecf7d0686ded0b1fbc57f8d0fcb404b62bd77242837b` |
| modeling_higgs_audio.py | 110,160 | `6259e90f8f5048e392572a9509e9e595339d097f720d3b004b9e2df199b1174c` |
| modeling_higgs_audio_xcodec.py | 20,185 | `8d7eea957d188243230088316bb237d64e347ae97b9839eed22fef2fbbce8bd` |
| ngram_loop_fix.py | 2,069 | `44010170393bf47891d4176b70aa5141fb575754cd23981c2064f13e9b3b3fd6` |
| transcribe.py | 16,055 | `6c159a8d01bf0f9f49d8a7c9b1e9ebf752540f9d949a4233033e8ba6f42ace5a` |
| utils.py | 36,457 | `4a39b74dd86010cb485ca0c6436b7368ec429250376ed8716c48ef9b73cc863a` |
| open_asr_leaderboard/README.md | 3,010 | `2368e5624cd9d96b3c4b6085a42d97a1bb88c0719f2608c2b169396f4f435a58` |
| open_asr_leaderboard/aggregate.py | 4,171 | `8fb0618a62609232c5aab369cded549bddc84210d03d48f2356331ed4e31707b` |
| open_asr_leaderboard/run_8gpu_parallel.sh | 1,800 | `499f72a092c4bc0e5a5cb6650d4c5c849d07c1cb2cfb049abe0df7ca1e76b1d9` |
| open_asr_leaderboard/run_eval.py | 15,592 | `39e10b4374ab2dd01496ddc13f1284345f9bba1f5d76dd567460e887294c8a6b` |

Checkpoint facts verified at load: `model_type` `higgs_audio_3`; no
`quantization_config` (a quantized variant is refused, and any `.scales`
tensor in the shards is refused); `projector_temporal_downsample` 2;
16 kHz audio; `audio_in_token_idx` 151672 matches the tokenizer's
`<|AUDIO|>`; `audio_eos_token_id` 151670 matches `<|audio_eos|>`; the prompt
tokens `<|im_start|>`, `<|im_end|>`, `<|audio_bos|>`, `<|endoftext|>` exist.
Any other geometry, activation, RoPE scaling, or sample rate is refused
rather than mis-decoded.

## Shared-module extension

The checkpoint stores the Qwen3 backbone under the raw HF names
(`layers.*`, `embed_tokens.weight`, `norm.weight`) across two
`model.safetensors` shards, with the untied LM head in the second shard.
`crates/audio/src/stt/qwen3_asr/decoder.rs` gained a minimal, documented
extension for this: every loader accepts a shard list (`Decoder::load`
forwards single-file callers unchanged), and two weight-prefix candidates
were appended after the existing ones (the bare `model.`-less names, and
`lm_head` -> `audio_decoder_proj.text_lm_head`). Existing families resolve
through the earlier candidates, so their behavior is unchanged.

## Inference contract

1. Audio: mono f32 at 16 kHz, non-empty and finite. Chunks above 4 seconds
   of samples follow the VAD path above; the smoke clip (44,715 samples)
   stays one chunk under every VAD outcome because the merged single span is
   `(0, total)`.
2. Frames: `1 + samples / 160` centered STFT frames minus the dropped tail
   frame; the conv stride then halves twice (encoder, projector), so one
   audio row covers 640 samples. The smoke clip yields 279 mel frames, 140
   encoder frames, 70 pooled rows, and 35 projected rows.
3. Prompt: 20 prefix ids, one `<|AUDIO|>` per chunk, 6 suffix ids; the
   embedding sequence carries `20 + audio_rows + 6` rows. The smoke clip:
   61 prompt rows, 35 audio rows.
4. Decode: untied LM head logits, greedy argmax, stop on 151643/151645.
5. Output: `_parse_output` over the raw decode, trimmed.

What this port refuses: quantization blocks and `.scales` tensors, mel bins
other than 128, non-GELU encoder activation, non-SiLU decoder MLP,
attention bias, RoPE scaling, temporal downsample other than 2, sample rates
other than 16 kHz, and negative token ids.

Divergence note: when `vad_cut` is on and no Silero VAD directory was
provided to `load_with_vad`, the port runs the reference's backend-failure
fallback (uniform 4-second chunks) instead of erroring; the reference
behaves identically when its VAD cannot load.

## Verification evidence

Fixture: `crates/audio/testdata/higgs_audio_3_reference.json`
(schema `turbospark.higgs_audio_3.reference/1`), generated by
`crates/audio/scripts/generate_higgs_audio_3_fixture.py` on the shared
smoke clip `crates/audio/testdata/qwen3_forced_aligner_reference.wav`
(44,715 samples, SHA-256 `ea38d350100b5d532fbc3e3517499dbd9915b9b0890d9e819bae384fa1384bcb`)
through the reference stack (mlx-audio 0.5.7 at the commit above, pinned
revision, Silero VAD active). Regenerate with:

```sh
cd ../mlx-audio && ../.venv-mlxaudio/bin/python \
  ../turbospark/crates/audio/scripts/generate_higgs_audio_3_fixture.py \
  --model-dir ~/models/higgs-audio-v3-stt \
  --audio ../turbospark/crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output ../turbospark/crates/audio/testdata/higgs_audio_3_reference.json \
  --revision 2ffd1aa39f5a1266931e405cba12e404a9f994b2
```

Python reference run (venv mlx-audio, `Model.generate` on the raw float32
samples, Silero VAD cuts `(32, 44715)` merging to one chunk):

```
the quick brown fox jumps over the lazy dog.
```

27 prompt ids, 35 audio rows, 61 prompt embedding rows, 10 generated tokens
(ids 1782, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13), first
greedy token 1782.

Rust release checkpoint run (`cargo test --release -p turbospark-audio
--lib stt::higgs_audio_3 -- --ignored --nocapture --test-threads=1` with
`TURBOSPARK_HIGGS_AUDIO_3_MODEL_DIR` set to the pinned snapshot, 2026-10-05,
macOS arm64) reproduces the contract:

- prompt token ids: 27/27 exact match
- prompt embedding rows: 61/61
- greedy token ids: 9/10 exact match; the tenth is the documented near tie
  below
- transcript: `the quick brown fox jumps over the lazy dog` (the reference
  sentence minus the final period; see the near tie)
- audio rows: 35/35

Final-decision near tie (recorded in the fixture as `final_step_logits`):
the reference bf16 logits for the period (token 13) and `<|im_end|>`
(151645) at the last step are 20.875 vs 20.75, a 0.125 gap that equals one
bf16 ulp at magnitude 20.9. This f32 port lands at 20.865 vs 20.878 (gap
-0.013), so its argmax picks `<|im_end|>` and stops one token early,
producing the same sentence without the period. The other nine decisions
have runner-up gaps of 3 or more logits and match exactly. The test asserts
the exact 9-token prefix, requires both candidates inside the Rust top-16
with the gap within 0.5 of the reference gap, and accepts either the full
reference transcript or its near-tie prefix.

Numeric stage witnesses (the reference computes the whole pipeline in
bfloat16; this port computes f32, so gates are relative while tokens stay
exact; measured values from the release run above):

| Stage | Max abs diff | Gate |
|---|---|---|
| input_features (log-mel) | 1.92e-3 | 3.0e-3 of row max (bf16 rounding bound) |
| encoder_output | 2.49e-2 | 6e-2 of row max |
| audio_embeddings (projector) | 3.00e-4 | 6e-2 of row max |
| prefill last hidden | 6.44e-1 (relative 1.55e-2) | 6e-2 of row max |
| first-step logits top-8 | 1.53e-1 | 5e-2 relative; top-8 ids survive in top-16 |

Always-on tests (no checkpoint) cover the frontend witness against the
fixture, the prompt shape and contiguous `<|AUDIO|>` span, the fixture
transcript and raw decode, `_parse_output` stripping (think blocks, special
spans, no-DOTALL behavior), the VAD chunking arithmetic (uniform fallback,
merged spans, split_vads, tiling), the symmetric Hann window, the channel
witness spots, and config parsing plus refusals.

Gates run on 2026-10-05 (macOS, arm64, CPU f32):

- `cargo test -p turbospark-audio --lib stt`: all stt tests green. This
  family adds 19 tests: 18 always-on plus the checkpoint-gated release run
  documented above (about 21 s wall in release on the smoke clip).
- `cargo fmt -p turbospark-audio --check`: clean on this family's files.
- `cargo clippy -p turbospark-audio --tests`: no diagnostics mentioning
  `stt::higgs_audio_3`.

## Remaining gates

- Task quality beyond the smoke clip (the checkpoint ships an
  `open_asr_leaderboard` harness upstream; no multi-clip quality run has
  been made here), runtime integration, catalog probes, FFI, and Swift
  surfaces are not started.
- The Rust release run above exercised the fallback chunking path (no VAD
  directory given; identical single-chunk outcome for the smoke clip). The
  Silero-backed Rust path (`load_with_vad`, reusing the verified
  `vad/silero_vad` family) is implemented and its chunking arithmetic is
  unit-tested, but an end-to-end multi-chunk transcription against a real
  Silero checkpoint is still open.
- `cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio` could
  not run on this machine: `onig_sys` (a transitive dependency of the
  `tokenizers` crate that predates this family) needs a linux cross-gcc
  which is absent. The port uses portable std + crate-internal APIs only.
- Metal offload and streaming are unverified. The checkpoint is 5.4 GB BF16;
  the f32 CPU resident set is about 10.7 GB plus activations.
