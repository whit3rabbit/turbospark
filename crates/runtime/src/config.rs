//! Generation-level configuration bundling the validated sampling shape
//! ([`selection::ShapingConfig`]) with the loop-level knobs (`max_new`,
//! stop strings, extra stop token ids) that `selection` itself has no
//! opinion about. Ported from the parts of `GenerationConfig.swift` that
//! `Runtime/Generation/RawCompletion.swift` reads directly.

use crate::power::RateControl;
use foundation::TokenId;
use selection::ShapingConfig;

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub shaping: ShapingConfig,
    pub max_new_tokens: u32,
    pub stop_strings: Vec<String>,
    pub extra_stop_tokens: Vec<TokenId>,
    /// Decode-rate cap and thermal stepping (ROADMAP Phase P2). The
    /// [`Default`] is uncapped, which is the pre-P2 loop exactly.
    pub rate: RateControl,
}

impl GenerationConfig {
    /// A pure-greedy config always agrees with the single highest-scoring
    /// candidate, independent of the seed.
    pub fn is_pure_greedy(&self) -> bool {
        self.shaping.temperature() == 0.0 && self.shaping.repetition_penalty() == 1.0
    }
}
