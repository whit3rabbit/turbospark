# Core ML Model Pipelines

## Overview
Each submodule turns one FluidInference Core ML export into a TurboSpark capability: Parakeet TDT and Parakeet Unified streaming speech recognition, pyannote community-1 speaker diarization, EmbeddingGemma 2 text/audio/image embeddings, and Unlimited OCR. The Objective-C and Core ML work lives in `turbospark_coreml`; this directory supplies the host side the exports leave out (windowing, greedy decode loops, clustering, masks, tokenization, config validation). `mod.rs` also recognizes which pipeline a model directory holds by its files, since these exports carry no `model_type`. Compiled packages are cached inside the install directory.

## Interface
- `CoreMlLayout` [mod.rs:CoreMlLayout]: enum of recognized layouts (ParakeetTdt, ParakeetUnified, EmbeddingGemma2, SpeakerDiarization) with `family()` string
- `detect_layout` [mod.rs:detect_layout]: classifies a model directory by required files/dirs, `None` if unknown
- `cache_root` [mod.rs:cache_root]: `<install>/.compiled`, where compiled packages are cached
- `ParakeetCoreMl`, `ParakeetCoreMlOptions` [parakeet.rs:ParakeetCoreMl]: batch TDT transcription (`open`, `transcribe`, `decode` with progress/cancel); `for_model` picks encoder placement from declared weight format
- `default_encoder_units`, `encoder_weight_format`, `words` [parakeet.rs:default_encoder_units]: placement policy, config weight-format read, token-to-word grouping
- `ParakeetUnifiedCoreMl`, `UnifiedOptions` [parakeet_unified.rs:ParakeetUnifiedCoreMl]: streaming-only English encoder/decoder/joint; also `EncoderBundle`, `parse_encoder_bundle`, `encoder_bundles`, `unified_mel`
- `UnifiedStream`, `StreamUpdate`, `StreamWord` [parakeet_unified_stream.rs:UnifiedStream]: platform-free streaming session (`append`, `finish`) over a `StreamingModel`
- `Windower`, `Geometry`, `RnntStepper`, `StreamingModel`, `rnnt_greedy` [parakeet_unified_stream.rs:Windower]: window planning and plain greedy RNNT, testable without Core ML
- `DiarizationCoreMl`, `DiarizationOptions`, `Diarization`, `SpeakerTurn` [diarization.rs:DiarizationCoreMl]: offline diarization (`open`, `diarize` with progress/cancel); VBx clustering and exclusive reconstruction on the host
- `EmbeddingGemma2CoreMl`, `EmbeddingGemma2Options` [embedding_gemma2.rs:EmbeddingGemma2CoreMl]: text embeddings (`embed_texts`, `token_ids`), Matryoshka dims via `truncate_normalize`
- `AudioWindowEmbedding`, `MediaTowers` [embedding_gemma2_media.rs:AudioWindowEmbedding]: optional audio/vision towers (`embed_audio`, `embed_image`, `*_file` variants)
- `UnlimitedOcrCoreMl`, `OcrOptions`, `OcrPage`, `OcrResult`, `OcrEvent`, `OcrStats`, `OcrConfig`, `plain_text` [unlimited_ocr.rs:UnlimitedOcrCoreMl]: multi-page OCR (`read`, `read_limited`) with prefill/decode buckets and no-repeat n-gram ban
- Constants per pipeline (`REQUIRED_DIRS`, `VOCAB_FILE`, `CONFIG`, `TOKENIZER`, `TEXT_PACKAGE`, `SAMPLE_RATE`, ...) are the file contract used by `detect_layout` [mod.rs:detect_layout]

## Invariants
- Every runner refuses to open below macOS 15 (or lacking MultiFunction/State capability) [parakeet.rs:ParakeetCoreMl::open] [diarization.rs:DiarizationCoreMl::open] [embedding_gemma2.rs:EmbeddingGemma2CoreMl::open] [unlimited_ocr.rs:UnlimitedOcrCoreMl::open]
- A GPU-capable placement for an unverified (int8 or unlisted weight format) encoder or the diarization embedding is refused at open unless `TURBOSPARK_COREML_ALLOW_QUANTIZED_GPU=1`, because Core ML can abort the process [parakeet.rs:check_encoder_units] [parakeet_unified.rs:check_units] [diarization.rs:check_embedding_units]
- Unlimited OCR page encoder is pinned to the GPU (wrong output on the Neural Engine) and a Neural Engine decoder is rejected [unlimited_ocr.rs:UnlimitedOcrCoreMl::open]
- Unknown `*_UNITS` env values are errors, never silent defaults [parakeet.rs:ParakeetCoreMlOptions::for_model] [diarization.rs:DiarizationOptions::from_env]
- Audio input must be nonempty, finite, 16 kHz mono; wrong sample rate is an error, not a resample [parakeet.rs:ParakeetCoreMl::decode] [diarization.rs:DiarizationCoreMl::diarize]
- Parakeet Unified refuses a `config.json` whose `model_id`, sample rate, subsampling, mel dims or vocab/blank counts differ, and requires exactly one streaming encoder bundle [parakeet_unified.rs:ParakeetUnifiedCoreMl::open]
- Vocabulary ids must be contiguous from 0 and blank equals vocab length [parakeet.rs:read_vocabulary] [parakeet_unified.rs:read_vocabulary]
- Parakeet Unified streaming: append is 1..=32000 finite samples; after a failed or cancelled step the stream refuses further work; carried decoder state is updated only after a whole pass succeeds [parakeet_unified_stream.rs:UnifiedStream::append] [parakeet_unified_stream.rs:rnnt_greedy]
- TDT loop: blank advances at least one frame; steps capped at `max_symbols * frames`; token beyond blank is an error [parakeet.rs:tdt_greedy_redux]
- Compiled shapes are cross-checked against each other and the declared config (mel/encoder/decoder/joint dims, OCR `pixels` 1x3x1024x1024, `page_embeds` shape) [parakeet.rs:read_dims] [parakeet_unified.rs:ParakeetUnifiedCoreMl::open] [unlimited_ocr.rs:encode_page]
- Memory-mapped tables must be exactly `rows*cols*2` bytes; out-of-range token ids error [embedding_gemma2.rs:Bf16Table::open] [unlimited_ocr.rs:F16Table::open] [embedding_gemma2_media.rs:MediaTowers::position_table]
- EmbeddingGemma `embed_scale` must equal sqrt(hidden_size); Matryoshka dimension must be one of the configured sizes [embedding_gemma2.rs:parse_config] [embedding_gemma2.rs:check_dimension]
- OCR config requires `vocab_half*2 == vocab_size`, `image_size == 1024`, nonzero chunk/ring, `ngram_size >= 2`; decode mask hides the ring slot about to be overwritten [unlimited_ocr.rs:OcrConfig::parse] [unlimited_ocr.rs:DecodeMask]
- Small JSON config files are read with a 1 MiB cap [diarization.rs:read_psi] [parakeet.rs:read_config] [parakeet_unified.rs:read_json]

## Side Effects
- Reads `TURBOSPARK_PARAKEET_COREML_ENCODER_UNITS`, `TURBOSPARK_PARAKEET_UNIFIED_ENCODER_UNITS`, `TURBOSPARK_DIARIZATION_COREML_EMBEDDING_UNITS`, `TURBOSPARK_EMBEDDINGGEMMA2_COREML_UNITS`, and `TURBOSPARK_COREML_ALLOW_QUANTIZED_GPU` [parakeet.rs:ParakeetCoreMlOptions::for_model] [parakeet_unified.rs:UnifiedOptions::from_env] [diarization.rs:DiarizationOptions::from_env] [embedding_gemma2.rs:EmbeddingGemma2Options::from_env]
- Compiles `.mlpackage`/`.mlmodelc` programs into `<install>/.compiled` on first load (can take minutes) [mod.rs:cache_root] [embedding_gemma2.rs:EmbeddingGemma2CoreMl::open]
- Loads Core ML models onto CPU/GPU/Neural Engine; a bad placement can abort the process [parakeet.rs:ParakeetCoreMl::open]
- Memory-maps bf16/f16 token and position tables via `ResidentBuffer` [embedding_gemma2.rs:Bf16Table::open] [unlimited_ocr.rs:F16Table::open]
- Lazy per-instance model caches behind `Mutex` (function per length, audio/vision towers, OCR buckets) with poison recovery [embedding_gemma2.rs:EmbeddingGemma2CoreMl] [embedding_gemma2_media.rs:MediaTowers] [unlimited_ocr.rs:UnlimitedOcrCoreMl::bucket]
- Reads model-directory files and, for `embed_audio_file` / `embed_image_file` / `OcrPage::from_file`, user media files; no writes outside the compile cache [embedding_gemma2_media.rs:embed_audio_file] [unlimited_ocr.rs:OcrPage::from_file]

## Verification
- Run: `cargo test -p turbospark-runtime coreml`
