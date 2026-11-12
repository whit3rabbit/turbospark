# NeMo Alignment Utilities

## Overview
Portable utility library (not a loadable model) holding the timestamped alignment data model shared by NeMo transducer families such as Parakeet. It provides token, sentence, and result types, sentence grouping at terminal punctuation, and two strategies for merging overlapping chunk alignments.

## Interface
- `AlignedToken` (with `new`) [alignment.rs:AlignedToken]
- `AlignedSentence` (with `new`) [alignment.rs:AlignedSentence]
- `AlignedResult` [alignment.rs:AlignedResult]
- `tokens_to_sentences` [alignment.rs:tokens_to_sentences]
- `sentences_to_result` [alignment.rs:sentences_to_result]
- `merge_longest_contiguous` [alignment.rs:merge_longest_contiguous]
- `merge_longest_common_subsequence` [alignment.rs:merge_longest_common_subsequence]
- All re-exported from [mod.rs:alignment]

## Invariants
- `AlignedToken.end` is always `start + duration` when built via `new` [alignment.rs:AlignedToken].
- `AlignedSentence::new` sorts tokens by start and derives start/end from first/last token; empty input gives a zero interval [alignment.rs:AlignedSentence].
- `AlignedResult.text` is trimmed while sentence texts keep leading whitespace [alignment.rs:AlignedResult].
- A period closes a sentence only if the next piece contains a space or none follows; `!`, `?`, and CJK terminators always close [alignment.rs:ends_sentence].
- Empty inputs pass through, time-ordered chunks concatenate, and fewer than 2 overlap tokens fall back to a midpoint cutoff split [alignment.rs:merge_longest_contiguous] [alignment.rs:merge_longest_common_subsequence].
- `merge_longest_contiguous` returns `SpeechError::Input` if the best matching run is shorter than half the overlap token count [alignment.rs:merge_longest_contiguous].
- Merges append the entire remainder of `b` past the last matched index, not only the overlap [alignment.rs:merge_longest_contiguous].

## Side Effects
- None. Pure in-memory functions.

## Verification
- Run: `cargo test -p turbospark-audio stt::nemo`
