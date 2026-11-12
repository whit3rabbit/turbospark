# Runtime Fixture Tools

## Overview
Three Python 3 helpers that prepare or regenerate test fixtures for the Core ML gates in crates/runtime/tests. Tests never run them; they are manual, and the two generators pin their dependencies through `uv run --with`.

## Interface
- `fetch_sound_recall_clips.py` [fetch_sound_recall_clips.py]: downloads pinned clips listed in tests/fixtures/sound_recall/clips.json, verifies SHA-256, converts to 16 kHz mono WAV with ffmpeg, writes converted.json; arg `[OUT_DIR]`
- `gen_ocr_pages.py` [gen_ocr_pages.py]: renders two known-text pages and pages.json into testdata/coreml_ocr/ using Pillow and Arial
- `ref_embeddinggemma2_coreml.py` [ref_embeddinggemma2_coreml.py]: reference vectors for the Core ML EmbeddingGemma 2 host via coremltools on CPU; arg model dir; writes testdata/coreml_embeddinggemma2/ (reference.json, manifest.json, pattern.png, chirp.wav)
- `__pycache__/` holds compiled bytecode, not source.

## Invariants
- A clip whose SHA-256 differs from the pin aborts, as does an oversize download or a missing ffmpeg [fetch_sound_recall_clips.py:SystemExit]
- Downloaded files are only hashed and passed to ffmpeg, never executed [fetch_sound_recall_clips.py]
- Reference preprocessing is written independently of the Rust port [ref_embeddinggemma2_coreml.py]
- gen_ocr_pages needs macOS (system Arial path) [gen_ocr_pages.py]

## Side Effects
- Network download (urllib) of the pinned clip URLs [fetch_sound_recall_clips.py]
- Writes under OUT_DIR (default `$TURBOSPARK_SOUND_RECALL_CLIPS`, then `~/models/sound-recall-clips`) [fetch_sound_recall_clips.py]
- Overwrites checked-in testdata files [gen_ocr_pages.py:OUT] [ref_embeddinggemma2_coreml.py:OUT]
- Spawns ffmpeg [fetch_sound_recall_clips.py]

## Verification
- Run: `python3 -I -m py_compile crates/runtime/tools/fetch_sound_recall_clips.py crates/runtime/tools/gen_ocr_pages.py crates/runtime/tools/ref_embeddinggemma2_coreml.py`
- Running them needs the pinned uv dependencies and network; `--help` is not available for gen_ocr_pages or ref_embeddinggemma2_coreml.
