# EmbeddingGemma 2 word CTC (experiment)

Status: experiment on branch `experiment/embeddinggemma2-ctc`. The portable
half of the pipeline is implemented and tested. No encoder or reader backend
exists yet, so this module cannot transcribe real audio on its own. No
quality, throughput, latency, memory, or device-placement claim is made
here.

## Overview

Haku's "Embedding Gemma Transcribe" (`ai.haku.eg2demo`) recognizes speech
without an autoregressive decoder:

1. Gemma4 log-mel frontend: 20 s windows, `[2000, 128]`.
2. `eg2_audio_l11`: the EmbeddingGemma 2 audio tower truncated after layer
   11, `[500, 1024]` (40 ms frames).
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
Step 6 is not ported.

| Module | Contents |
|---|---|
| `frontend.rs` | Gemma4 log-mel contract, shipped-table loader |
| `windows.rs` | 20 s window / 18 s step schedule, `[25, 475)` keep range, stitching |
| `vocabulary.rs` | `reader_words.json`, `word_unit_f16.bin` (kept as FP16 bits) |
| `scorer.rs` | CPU reference of the compiled block scorer, MIL blob parser, exact rerank, exhaustive search |
| `ctc.rs` | Calibration, blank decision, collapse, `<rep>` expansion, timestamps |
| `pipeline.rs` | `AcousticBackend` trait and the end-to-end `Transcriber` |

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
| Vocabulary contents | 229457 entries, 229304 distinct strings (153 strings appear twice under different ids), last entry `<rep>`. Layout: a 320-entry head block (ids 0..319, plain and apostrophe-prefixed common and programming words), an alphabetically sorted main list (ids 320..229426), 30 appended custom words (`haku`, `coreml`, `mlx`, `parakeet`, ...), then `<rep>`. |
| The word table is the stock text model, no prompt | Every word embedded with `google/embeddinggemma-2` @ `914f7f89142e33e77833254d9c9b90c3cef7303b` in float32, no task prefix, mean pooling, L2 norm (`scripts/probe_embeddinggemma2_word_table.py`): median cosine to its row 0.99999, 99% of rows >= 0.99996, 229014 of 229456 rows >= 0.999. On a 64-word sample every task prefix scores worse (best: `task: search result \| query: `, mean 0.90). The 30 appended custom words match at >= 0.99997, so Haku added words by embedding the raw string. |
| Some rows were changed | 442 rows are below 0.999: 142 in the head block (cosine 0.88 to 0.97, the first id of most duplicate pairs, whose second id is the stock row) and 300 frequent function words in the main list (`and`, `the`, `can`, ...; 0.980 to 0.995). `<rep>` is not an embedding of its string (cosine 0.02). These rows look trained, but how is not known. |
| Calibration | `reader_P6.json` (`reader_G795_L11_big.json` upstream): scale 39.85666275024414, bias -0.4626207649707794, `arm.layers = ["L11"]`, d 768, 12 layers, 12 heads, lr 3e-4. It names no loss. |
| Encoder and reader interfaces | `metadata.json`: encoder `input_features [1,2000,128]` f32 and `feature_mask [1,2000]` to `l11 [1,500,1024]` f16; reader `l11 [1,500,1024]` and `mask [1,500]` to `vec [1,500,768]` and `blank [1,500,1]` f16. The reader op histogram (1 reduceSum, pow, sqrt, realDiv) is consistent with an in-graph L2 normalization of `vec`. |
| The scorer's top-k returns sorted descending uint16 block ids | `model.mil`: `topk(k=16, ascending=false, sort=true, output_indices_dtype="uint16")`, cast to int32. |

Corrections to the earlier analysis this evidence forces:

- The audio encoder is not frozen. The app NOTICE says EmbeddingGemma 2's
  audio encoder was fine-tuned. A controlled "frozen encoder" comparison
  cannot reuse `eg2_audio_l11` as its frozen baseline.
- The vocabulary is not a plain word list: `<rep>` and the 153 duplicate
  strings change how collapse and rendering must work (below).
- A 20 s window yields 1999 valid mel frames under the reference contract,
  not 2000; the 2000th encoder input frame is masked padding here.

Reported elsewhere and not reproduced here:

- The `MLComputePlan` placement (matmuls ANE-preferred; reshape, reduce_max,
  topk not ANE-supported) was observed on a user's Apple Silicon host. It is
  anticipated placement, not an execution trace, and nothing in this branch
  re-ran it.
- 288x real time and about 5% WER are the developer's numbers.

## Open questions and this port's choices

| Question | Current choice |
|---|---|
| What `<rep>` means | Rendered as a repeat of the preceding word and dropped at the start. Unverified: it is the obvious reading of a word-level CTC repeat marker, but the reference binary was not inspected for it. |
| Collapse by id or by string | By id (the reference stores indices). The duplicates are deliberate (an adjusted head row and a stock main row for the same word), so adjacent frames that alternate between the two ids render the word twice. Collapsing by string avoids that but would also merge a true repeat with no blank between; decide with real-audio paths. |
| Adding words | Embed the raw lowercase string with the stock text model (no prompt, unit norm, FP16), as Haku did for its 30 custom words. The compiled scorer has a fixed `[128, 229460]` shape, so new words need a separate CPU shortlist path or a rebuilt scorer program. Core ML text conversions (FluidInference, OpenInfinity int8, FilBona int8) could do this on device after their fp16/int8 drift is measured against these rows, and without their default task prefixes. |
| First and last window edges | The first window keeps from frame 0 and the last keeps through its last valid frame; interior windows keep `[25, 475)`. The reference runtime's edge handling is unknown. |
| Encoder mask subsampling | `ceil(valid_mel / 4)` valid encoder frames (the Gemma audio every-4th-frame convention), unverified for the compiled encoder. |
| Rerank similarity | Plain dot product with the unit rows; equals the cosine up to FP16 rounding because `vec` is normalized in-graph. |
| Shortlist precision | The CPU scorer runs in f32; Core ML runs FP16, so near-tied blocks may order differently. Measure shortlist recall against `exhaustive_best` before relying on either. |

## Verification

```sh
cargo test -p turbospark-audio --lib embeddinggemma2
cargo fmt -p turbospark-audio --check
cargo clippy -p turbospark-audio --tests

# Asset checks (manifest layout, downloaded separately, not by tests):
TURBOSPARK_EG2_ASSET_DIR=/path/to/eg2demo \
  cargo test -p turbospark-audio --lib embeddinggemma2::asset_tests -- --ignored
```

Regenerate the frontend fixture with
`crates/audio/scripts/generate_embeddinggemma2_features_fixture.py` (install
command in its header).

Gates:

| Gate | Status |
|---|---|
| Frontend fixture parity | Passed (Linux, CPU) |
| Shipped tables, word table, scorer layout, calibration | Passed against the pinned files (Linux, CPU) |
| Window stitching and CTC logic | Synthetic tests, including mutation checks |
| Encoder/reader backend | Not implemented |
| End-to-end transcript on real audio | Not run |
| Word-table provenance | Passed: full-table comparison with the stock text model (Linux, CPU, float32) |
| Shortlist recall, WER, latency, memory, ANE/GPU placement | Not run |
| Runtime, catalog, CLI, FFI, Swift | Not started |

## Next steps

1. A Core ML backend in `crates/runtime` (macOS only) implementing
   `AcousticBackend` over `eg2_audio_l11` and `reader_P6`, plus a scorer
   `BlockShortlister`, then a local WAV-to-transcript command with per-stage
   timings.
2. Shortlist recall: the share of frames whose `exhaustive_best` word is in
   the 64 shortlisted candidates, for both the Core ML and CPU scorers.
3. Map the word table instead of reading it (352 MB resident as FP16 bits;
   the CPU scorer adds 117 MB of f32 columns).
4. Port punctuation and casing only after the recognizer is measured.
