//! Shared NeMo-family utilities.
//!
//! Reference: `mlx_audio/stt/models/nemo/` at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. Upstream hosts the alignment
//! data model shared by the NeMo transducer families (Parakeet today); the
//! module is a utility library rather than a loadable model family.

pub mod alignment;

pub use alignment::{
    merge_longest_common_subsequence, merge_longest_contiguous, sentences_to_result,
    tokens_to_sentences, AlignedResult, AlignedSentence, AlignedToken,
};
