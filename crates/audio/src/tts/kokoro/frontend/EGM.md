# Kokoro English Frontend
## Overview
Self-contained US English text frontend for the Kokoro TTS model. It normalizes text, tags parts of speech, looks up phonemes in bundled Misaki-derived gold and silver dictionaries, and splits the result into checked phoneme segments that fit the Kokoro context. It also holds the vocabulary and voice-pack validation types. No Python or external process is used [mod.rs:EnglishFrontend].

## Interface
- `EnglishFrontend::new`, `EnglishFrontend::prepare(request, vocab)` returning `Vec<SynthesisSegment>` [mod.rs:EnglishFrontend]
- `SynthesisRequest` (text, language, voice, speed) with `new` and `validate` [mod.rs:SynthesisRequest]
- `Voice` (only `AfHeart`), `as_str`, `FromStr` [mod.rs:Voice]
- `PhonemeVocabulary::new`, `PhonemeVocabulary::from_config` [mod.rs:PhonemeVocabulary]
- `SynthesisSegment::from_phonemes`, `text`, `phonemes`, `ids`, `style_row` [mod.rs:SynthesisSegment]
- `VoicePack::from_values`, `open`, `voice`, `style_for` [mod.rs:VoicePack]
- `MAX_PHONEME_CHARS` (510) [mod.rs:MAX_PHONEME_CHARS]
- `bundled::*` static byte/str copies of resources and licenses, for catalog digest pinning [mod.rs:bundled]
- Crate-private helpers: `Lexicon`, `PerceptronTagger`, `normalize::{validate_text,text,tokens}`, `load_us_gold`, `load_us_silver` [lexicon.rs:Lexicon] [tagger.rs:PerceptronTagger] [normalize.rs:tokens] [data.rs:load_us_gold]

## Invariants
- Only language `en-US` and a finite, positive speed are accepted [mod.rs:SynthesisRequest::validate]
- Input must be nonblank, at most 65536 bytes, and limited to ASCII alphanumerics plus a fixed punctuation set; other characters are rejected [normalize.rs:validate_text]
- Every vocabulary ID is nonzero and below both embedding row counts; usable length is `min(510, max_positions - 2)` [mod.rs:PhonemeVocabulary::new]
- A segment is 1..=max_chars phonemes, framed by blank token 0 on both sides; unknown phonemes are refused [mod.rs:PhonemeVocabulary::checked_ids]
- Style row is phoneme character count minus one [mod.rs:SynthesisSegment::style_row]
- Voice pack shape must be [510, 1, 256] with finite values [mod.rs:VoicePack::from_values]
- A word plus attached punctuation larger than the context is an error; segments flush at sentence-ending punctuation or on overflow [mod.rs:prepare]
- Unsupported notations (time, slash, oversized numbers, bad decimals) return errors instead of being dropped [normalize.rs:tokens]
- Segment fields are private so callers only see checked segments [mod.rs:SynthesisSegment]

## Side Effects
- Dictionaries and POS model are embedded at compile time via `include_str!`/`include_bytes!`; no runtime disk read for them [data.rs:load_us_gold] [mod.rs:bundled]
- Dictionary and tagger are parsed once into a process-wide `OnceLock` shared across requests; malformed bundled JSON panics via `expect` [mod.rs:ENGINE] [data.rs:load_us_gold] [tagger.rs:PerceptronTagger::new]
- `VoicePack::open` reads a safetensors file from disk [mod.rs:VoicePack::open]
- No env vars, threads, RNG, or logging found.

## Verification
- Run: `cargo test -p turbospark-audio tts::kokoro::frontend`
