# OmniVoice duration (test-only directory)

## Overview
Contains only `tests.rs`, the test module for the sibling `../duration.rs`. It holds no production source. The production duration logic lives in `../duration.rs` and is summarized in `../EGM.md`.

## Interface
- None (no non-test source files; `../mod.rs` declares `mod duration;` privately).

## Invariants
- None

## Side Effects
- None

## Verification
- Run: `cargo test -p turbospark-audio tts::omnivoice`
