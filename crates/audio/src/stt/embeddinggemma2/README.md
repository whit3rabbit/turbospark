# EmbeddingGemma 2 word CTC (experiment)

Status: experiment on branch `experiment/embeddinggemma2-ctc`. The portable
half of the pipeline is implemented and tested here; a macOS Core ML backend
for the encoder, reader, and scorer lives in `turbospark-runtime`
(`coreml::embeddinggemma2_ctc`). Real-audio transcripts have been checked on
four short English clips only (one checked-in, three macOS `say`). No WER, throughput, latency,
or memory claim beyond those clips is made here.

The full writeup, covering how the reference works, what was verified in its
programs and binary, and what this port measured, is
[docs/EMBEDDINGGEMMA2_WORD_CTC.md](../../../../../docs/EMBEDDINGGEMMA2_WORD_CTC.md).

## Overview

Haku's "Embedding Gemma Transcribe" (`ai.haku.eg2demo`) recognizes speech
without an autoregressive decoder:

1. Gemma4 log-mel frontend: 20 s windows, `[2000, 128]`.
2. `eg2_audio_l11`: the EmbeddingGemma 2 audio tower, all 12 conformer
   blocks. `l11` is the zero-indexed last block's normalized output,
   `[500, 1024]` (40 ms frames).
3. `reader_P6`: a trained reader (86.45 M parameters per its sidecar) that
   outputs a unit vector `vec [500, 768]` in the EmbeddingGemma 2 text
   embedding space and a `blank [500, 1]` logit.
4. Vocabulary search: a 128-d block shortlist (16 blocks of 4 words) and
   exact 768-d reranking of the 64 candidates against 229457 FP16 word
   rows.
5. Greedy word CTC: emit the best word when
   `39.8566 * cos - 0.4626 > blank`, collapse adjacent equal ids, drop
   blanks.
6. Punctuation and casing (`punct_headonly`, `punct_case.json`).

This crate implements steps 1, 4, and 5 plus the window schedule and
stitching, and defines the `AcousticBackend` boundary for steps 2 and 3.
`crates/runtime/src/coreml/embeddinggemma2_ctc.rs` implements that boundary
and a `BlockShortlister` over the three compiled programs (macOS 15 or
newer). Step 6 is not ported.

| Module | Contents |
|---|---|
| `frontend.rs` | Gemma4 log-mel contract, shipped-table loader |
| `windows.rs` | 20 s window / 18 s step schedule, `[25, 475)` keep range, stitching |
| `vocabulary.rs` | `reader_words.json`, `word_unit_f16.bin` (kept as FP16 bits) |
| `scorer.rs` | CPU reference of the compiled block scorer, MIL blob parser, exact rerank, exhaustive search |
| `ctc.rs` | Calibration, blank decision, collapse, `<rep>` expansion, timestamps |
| `pipeline.rs` | `AcousticBackend` trait, the end-to-end `Transcriber`, the 0.5 s tail padding |
| `runtime: coreml/embeddinggemma2_ctc.rs` | Core ML encoder/reader backend and scorer, program-description checks, per-program compute units |
| `runtime: examples/eg2_ctc_transcribe.rs` | WAV to transcript with timings; `--cpu-scorer`, `--recall` |

## Sources and pins

| Source | Pin |
|---|---|
| Reference app and asset manifest | `https://haku-models.tslfiles.org/eg2demo/v5/manifest.json` (version 5, created 2026-10-12, 714975724 bytes) |
| Frontend reference | transformers `Gemma4AudioFeatureExtractor` @ `28034545c35acac05793695bd5c878f66e6b544e` |
| Frontend configuration | `google/embeddinggemma-2` `preprocessor_config.json` (frame 320, hop 160, FFT 512, 128 mels, 0..8000 Hz, mel floor 1e-3, no preemphasis) |
| Audio tower configuration | `google/embeddinggemma-2` `config.json` audio section: `gemma4_audio`, 12 layers, hidden 1024 |

`EG2DEMO_V5` in `mod.rs` records the size and sha256 of every file the
portable path reads. The compiled programs carry these manifest hashes:

| File | Size | sha256 |
|---|---|---|
| `eg2_audio_l11.mlmodelc/weights/weight.bin` | 227280704 | `1781b3491deef798302ff0594cf89af84512dca0baaee093f71dada0985f0aab` |
| `reader_P6.mlmodelc/weights/weight.bin` | 65689984 | `b0697e6d084048918dc2ddde822da156e1b2400ff585bc63d4929d8e9721adbf` |
| `scorer_blk_d128_k16_b4_t500.mlmodelc/model.mil` | 3583 | `3992faf39a83111a1664a8cbe35bee0a1427b79eadc33d85f83b49e888a59790` |
| `punct_headonly.mlmodelc/weights/weight.bin` | 6772352 | `c84e2dc00eb63ad6c800c1f25fb2250a8929553d10565725fb5eb67cf20fbfcc` |

The assets are not committed. The app's NOTICE states they are modified
derivatives of EmbeddingGemma 2 (Apache-2.0) and asks deployments to follow
the Gemma Prohibited Use Policy. Confirm redistribution terms before any
catalog or download integration.

## Evidence

Checked in this branch against the pinned files:

| Finding | Evidence |
|---|---|
| The frontend is the Gemma4 extractor | Shipped `hann_320.f32` equals a periodic Hann (max diff 2.9e-8); shipped `mel_filters_257x128.f32` equals the f64 HTK, unnormalized, 0..8000 Hz bank in `[bin][mel]` layout (max diff 3.0e-8; the `[mel][bin]` reading differs by 0.99). The ignored asset test reproduces both within 1e-6. |
| Rust frontend matches the reference | `testdata/embeddinggemma2_features.json` from the pinned transformers extractor: frame count and mask exact, mel energy within 3e-6 absolute plus 1e-6 relative, log-mel within 5e-4 (f32 FFT round-off magnified near the floor). |
| Scorer vocabulary is the word table in identity order | For the first and last 2000 words (numpy check) the word row times the `[768,128]` projection matches the scorer's `[128,229460]` column (cosine >= 0.9999994, max abs 4.8e-4, FP16 rounding). So block `b` holds word ids `4b..4b+3`. The ignored Rust asset test repeats the check on every 997th word within 2e-3. |
| The 768 to 128 projection has orthonormal columns | `P^T P` off-diagonal max 4.4e-5, diagonal 1 +- 6e-5. This is consistent with a PCA basis but does not prove how it was built. |
| Scorer padding is not zero | Columns 229457..229459 are three copies of one vector (norm 0.436). Reranking skips ids past the table, so a padding column can only cost a shortlist slot. |
| Word table layout | `word_unit_f16.bin` is exactly `229457 * 768 * 2` bytes, headerless; sampled rows have unit norm within 2e-3. |
| Vocabulary contents | 229457 entries, 229304 distinct strings (153 strings appear twice under different ids), last entry `<rep>`. Layout: a 320-entry head block (ids 0..319, plain and apostrophe-prefixed common and programming words), an alphabetically sorted main list (ids 320..229425, ending at `zzzz`), 30 appended custom words (ids 229426..229455: `haku`, `coreml`, `mlx`, `parakeet`, ...), then `<rep>` (id 229456). |
| The word table is the stock text model, no prompt | Every word embedded with `google/embeddinggemma-2` @ `914f7f89142e33e77833254d9c9b90c3cef7303b` in float32, no task prefix, mean pooling, L2 norm (`scripts/probe_embeddinggemma2_word_table.py`): median cosine to its row 0.99999, 99% of rows >= 0.99996, 229014 of 229456 rows >= 0.999. On a 64-word sample every task prefix scores worse (best: `task: search result \| query: `, mean 0.90). The 30 appended custom words match at >= 0.99997, so Haku added words by embedding the raw string. |
| Some rows were changed | 442 rows are below 0.999: 142 in the head block (cosine 0.88 to 0.97, the first id of most duplicate pairs, whose second id is the stock row) and 300 frequent function words in the main list (`and`, `the`, `can`, ...; 0.980 to 0.995). `<rep>` is not an embedding of its string (cosine 0.02). These rows look trained, but how is not known. |
| Calibration | `reader_P6.json` (`reader_G795_L11_big.json` upstream): scale 39.85666275024414, bias -0.4626207649707794, `arm.layers = ["L11"]`, d 768, 12 layers, 12 heads, lr 3e-4. It names no loss. |
| Encoder and reader interfaces | `metadata.json`: encoder `input_features [1,2000,128]` f32 and `feature_mask [1,2000]` to `l11 [1,500,1024]` f16; reader `l11 [1,500,1024]` and `mask [1,500]` to `vec [1,500,768]` and `blank [1,500,1]` f16. The reader op histogram (1 reduceSum, pow, sqrt, realDiv) is consistent with an in-graph L2 normalization of `vec`. |
| The scorer's top-k returns sorted descending uint16 block ids | `model.mil`: `topk(k=16, ascending=false, sort=true, output_indices_dtype="uint16")`, cast to int32. |
| The encoder holds all 12 blocks | `model.mil` names `layers_0` to `layers_11` (q/k/v, light conv with depthwise kernel 5, two feed-forward modules each) and ends at `layers_11_norm_out`. "L11" is the zero-indexed last block, not a cut before it. About 290.7 M weights. |
| The reader is a 12-layer transformer | `model.mil`: input LayerNorm (1024), linear to 768, `x + DepthwiseConv1d(x)` (kernel 15) for position, 12 pre-LN layers (fused QKV, 12 heads of 64, scale 0.125, FFN 3072, exact GELU), key-padding mask of -10000, final LayerNorm, word head with in-graph L2 norm, linear blank head. Weight shapes sum to 86,448,897, the sidecar's 86.45 M. No in-layer convolution, so not a Conformer. |
| The decision rule is in the app binary | `EG2Transcribe` (app 1.6, sha256 `7bcdc008...`), `EG2Transcriber.back` at 0x1000757d0: `fmul`, `fadd`, `fcmp` with the blank logit, `csinv ... hi`. |
| The projection is not Matryoshka truncation | Its first 128 input rows hold 32.2 of its 128 units of energy. Truncation would hold all 128. |
| The app has a GPU path | `Front(folder:units:readerUnits:...)` takes compute units from callers, one `EG2Transcriber` site passes `cpuAndGPU`, and `transcribeShared(_:gpu:...)` takes a GPU front. Run-time choices were not traced. |

Checked on macOS 26.6.2, Apple M4 Max, with the Core ML backend and every
program on `ComputeUnits::All` (`eg2_ctc_transcribe`, release build):

| Finding | Evidence |
|---|---|
| End-to-end transcripts | `testdata/qwen3_forced_aligner_reference.wav` (2.79 s): `the quick brown fox jumps over the lazy dog`, exact (ignored runtime test). Three macOS `say -v Samantha` clips rendered at 16 kHz f32: 6.0 s, 0 of 24 words wrong; 47.0 s over 3 windows, 2 of 133 (`and` read as `an`, `counts` as `count`), no word lost or doubled at a seam; 7.5 s of back-to-back repeats, 0 of 29. Total 2 of 195. One synthetic voice, clean audio: not a WER claim. |
| Shortlist recall | Over 1527 kept frames of three clips (the 6 s and 47 s clips before tail padding was added, the repeats clip with it), the 64 shortlisted candidates held the exhaustive best word in 1442 frames (94.4%) and in 189 of 189 frames whose exhaustive winner beats blank. The Core ML (FP16) and CPU (f32) scorers made the same decision on every frame, and no decision differed from the exhaustive search. |
| `<rep>` is the immediate-repeat marker | On the repeats clip the reader emitted id 229456 exactly where a word repeats with no other word between (`no <rep> <rep>`, `very <rep>`, `that <rep>`, the last two on adjacent frames with no blank), and the plain id for `walked and walked`. This supports rendering `<rep>` as the preceding word. |
| Duplicate ids | Across the four transcripts no head-block id (0..319) was emitted; duplicated strings (`a`, `with`, `for`, ...) always came from their main-list id, and no adjacent pair alternated between two ids of one string. |
| Word spikes trail the word | Every word is a one- or two-frame spike. Against the forced-aligner spans of the reference clip, spikes land 0.04 to 0.32 s after the word ends, so `start_seconds` is a spike time, not a word onset. |
| The last word needs trailing audio | Without padding, the repeats clip lost its last word (`night`, spike at 7.52 s, audio ends at 7.498 s). Appending 0.25 s or 0.5 s of zeros recovered it on all three `say` clips with no other change; 1 s also flipped an earlier frame (`that that` read as `think think`). The encoder sees the whole window, so trailing context moves earlier outputs too. |
| Planned placement (`MLComputePlan`, `coreml_plan --units all`) | Encoder: 994 ops on the Neural Engine (61% of estimated cost), 701 on the GPU (28%). Reader: 292 Neural Engine ops (87%), 4 GPU. Scorer: both matmuls on the Neural Engine (26%); reshape, reduce_max, topk, and cast on the GPU (73%). reshape, topk, and cast list no Neural Engine support; reduce_max does but prefers the GPU. A plan, not an execution trace. |
| Speed, indicative only | One warm run each: 47.0 s in 0.38 to 0.46 s (RTFx 103 to 123), 6.0 s in 0.10 s. The first run after load includes Core ML specialization (cold: 0.7 s for 47 s). Opening the encoder and reader took 11.7 s the first time. Not a benchmark-protocol measurement. |

Corrections to the earlier analysis this evidence forces:

- The audio encoder may not be frozen. The app NOTICE says EmbeddingGemma
  2's audio encoder was fine-tuned, while the project site says it "stays
  exactly as it is". Until the program weights are compared with the stock
  tower, a controlled "frozen encoder" comparison cannot reuse
  `eg2_audio_l11` as its frozen baseline.
- The encoder is not truncated before its last layer (see Evidence).
- The vocabulary is not a plain word list: `<rep>` and the 153 duplicate
  strings change how collapse and rendering must work (below).
- A 20 s window yields 1999 valid mel frames under the reference contract,
  not 2000; the 2000th encoder input frame is masked padding here.

Reported elsewhere and not reproduced here:

- 288x real time and about 5% WER are the developer's numbers.
- The earlier report that reduce_max is not Neural Engine supported does not
  hold on the host above (it is supported there, but the plan prefers the
  GPU).

## Open questions and this port's choices

| Question | Current choice |
|---|---|
| What `<rep>` means | Rendered as a repeat of the preceding word and dropped at the start. The repeats clip supports this reading (see Evidence); the reference binary was not inspected for it. Because it repeats whatever precedes it, a lost word before `<rep>` repeats the wrong one. |
| Collapse by id or by string | By id (the reference stores indices). The duplicates are deliberate (an adjusted head row and a stock main row for the same word), so adjacent frames that alternate between the two ids would render the word twice. No alternation and no head-block id appeared on the four clips; keep by id until real recordings show otherwise. |
| Adding words | Embed the raw lowercase string with the stock text model (no prompt, unit norm, FP16), as Haku did for its 30 custom words. The compiled scorer has a fixed `[128, 229460]` shape, so new words need a separate CPU shortlist path or a rebuilt scorer program. Core ML text conversions (FluidInference, OpenInfinity int8, FilBona int8) could do this on device after their fp16/int8 drift is measured against these rows, and without their default task prefixes. |
| First and last window edges | The first window keeps from frame 0 and the last keeps through its last valid frame. Interior windows keep `[25, 475)`. `Transcriber` appends `TAIL_PADDING_SAMPLES` (0.5 s) of zeros to non-empty input first, so a final spike that trails the audio is kept (the `tail_padding` field, 0 disables it). The edge rule matches the app binary (`transcribeShared` and `Stream.advance(final:)`: keep start is 0 for the first window, keep end is the frame count for the last). The reference's trailing-audio handling is unknown. |
| Encoder mask subsampling | `ceil(valid_mel / 4)` valid encoder frames (the Gemma audio every-4th-frame convention). The transcripts above are consistent with it; the compiled encoder was not inspected for it. |
| Rerank similarity | Plain dot product with the unit rows; equals the cosine up to FP16 rounding because `vec` is normalized in-graph. |
| Shortlist precision | The CPU scorer runs in f32; Core ML runs FP16, so near-tied blocks may order differently. On the measured clips both kept every emitted word's block and made identical decisions; `eg2_ctc_transcribe --recall` repeats the measurement on any WAV. |

## Verification

```sh
cargo test -p turbospark-audio --lib embeddinggemma2
cargo fmt -p turbospark-audio --check
cargo clippy -p turbospark-audio --tests

# Asset checks (manifest layout, downloaded separately, not by tests):
TURBOSPARK_EG2_ASSET_DIR=/path/to/eg2demo \
  cargo test -p turbospark-audio --lib embeddinggemma2::asset_tests -- --ignored

# Core ML backend (macOS 15+): unit tests, then the reference-clip
# transcript with both scorers.
cargo test -p turbospark-runtime --lib embeddinggemma2_ctc
TURBOSPARK_EG2_ASSET_DIR=/path/to/eg2demo cargo test --release \
  -p turbospark-runtime --lib embeddinggemma2_ctc -- --ignored

# Any WAV; placement per program through
# TURBOSPARK_EG2_CTC_{ENCODER,READER,SCORER}_UNITS (cpu, cpu_gpu, cpu_ane, all).
cargo run --release -p turbospark-runtime --example eg2_ctc_transcribe -- \
  /path/to/eg2demo clip.wav [--cpu-scorer] [--recall]
```

The asset directory is the manifest layout: every file listed in
`v5/manifest.json`, at its `path`, verified against its `sha256`. The host
refuses Python's default urllib user agent (HTTP 403); `curl` works.

Regenerate the frontend fixture with
`crates/audio/scripts/generate_embeddinggemma2_features_fixture.py` (install
command in its header).

Gates:

| Gate | Status |
|---|---|
| Frontend fixture parity | Passed (Linux, CPU; macOS) |
| Shipped tables, word table, scorer layout, calibration | Passed against the pinned files (Linux, CPU; macOS 26.6.2) |
| Window stitching, tail padding, and CTC logic | Synthetic tests, including mutation checks |
| Encoder/reader/scorer Core ML backend | Implemented in `turbospark-runtime`; program descriptions checked at open |
| End-to-end transcript on real audio | Reference clip exact (ignored runtime test); three `say` clips, 2 of 195 words wrong overall (M4 Max) |
| Word-table provenance | Passed: full-table comparison with the stock text model (Linux, CPU, float32) |
| Shortlist recall | Measured on three clips: every emitted word's block kept, Core ML and CPU decisions identical |
| Planned placement | `MLComputePlan` read on one host (above); no execution trace |
| WER on real recordings, latency protocol, memory | Not run |
| Catalog, CLI, server, FFI, Swift | Not started |

## Next steps

1. WER on real recordings (several speakers, noise, accents) against a
   reference such as LibriSpeech test-clean, with the same clips run through
   an existing recognizer for comparison.
2. Latency and memory under the benchmark protocol, per program and per
   compute-unit choice, including the first-load cost.
3. Map the word table instead of reading it (352 MB resident as FP16 bits;
   the CPU scorer adds 117 MB of f32 columns).
4. Port punctuation and casing only after the recognizer is measured.
5. Confirm redistribution terms before any catalog or download integration.
