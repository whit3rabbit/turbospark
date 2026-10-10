//! Shared corpus and helpers for the two humanizer gates (the memory
//! oracle and the quality gate). The model is a RAW COMPLETION rewriter:
//! every prompt here goes through [`foundation::humanizer`]'s byte-exact
//! wrapper and encodes WITH the BOS, never through a chat template --
//! applying this model's (absent) template is the one intake mistake its
//! publisher leads with.
//!
//! The drafts and their reference rewrites are a FROZEN CORPUS: compare
//! each row only against its own past. The reference rewrites are
//! deliberately uneven, the way the model is trained to write -- varied
//! sentence length, no hedging, every fact kept.

use selection::ShapingConfig;
use tokenizer::MfTokenizer;

/// A short AI-flavored draft, three sentences, two numbers.
pub const DRAFT_SHORT: &str = "We are pleased to inform you that the migration has been successfully completed. All 14 services were migrated over the course of 3 days, and the total downtime was approximately 47 minutes. Furthermore, it should be noted that performance improvements of up to 30 percent were observed post-migration.";

/// The frozen reference rewrite of [`DRAFT_SHORT`].
#[allow(dead_code)] // consumed by the quality gate target, not this one
pub const REFERENCE_SHORT: &str = "The migration is done. Fourteen services moved in 3 days; we were down about 47 minutes in total, and things run up to 30 percent faster now.";

/// A medium draft with names, dates and a quota, the shape the publisher's
/// own fact-slip statistics point at.
pub const DRAFT_MEDIUM: &str = "On October 12, 2026, the platform team, led by Mara Chen, rolled out the new rate limiter to all production clusters. The previous limit of 1,200 requests per minute was raised to 5,000 requests per minute for paid tiers, while free-tier users remain at 300 requests per minute. It is worth noting that the rollout was completed without any reported incidents, and the team believes that this change will significantly improve customer satisfaction going forward. Additionally, latency at the 99th percentile dropped from 840 milliseconds to 610 milliseconds during internal load testing.";

/// The frozen reference rewrite of [`DRAFT_MEDIUM`].
#[allow(dead_code)] // consumed by the quality gate target, not this one
pub const REFERENCE_MEDIUM: &str = "Mara Chen's platform team shipped the new rate limiter to every production cluster on October 12, 2026. Paid tiers now get 5,000 requests per minute instead of 1,200; free tier stays at 300. No incidents were reported during the rollout. Internal load tests put the 99th-percentile latency at 610 milliseconds, down from 840.";

/// The digest draft: short enough that 256-token generations terminate
/// well inside the budget, long enough to be a real rewrite.
pub const DRAFT_DIGEST: &str = "Our company has always been deeply committed to leveraging cutting-edge technologies in order to facilitate seamless collaboration across all departments. In order to achieve this goal, we have implemented a comprehensive suite of tools that enable team members to synergize effectively. It is important to note that adoption has been growing at a rate of approximately 15 percent month over month since the initiative launched in January.";

/// The frozen reference rewrite of [`DRAFT_DIGEST`].
#[allow(dead_code)] // consumed by the quality gate target, not this one
pub const REFERENCE_DIGEST: &str = "We build tools so teams can work together without friction. Since January, usage has grown about 15 percent a month.";

/// The publisher's recipe: temperature 1.0, top_p 0.95, top_k off,
/// identity repetition penalty. The sampler boundary in this engine takes
/// raw logits and softmaxes once; these are the model's documented values.
pub fn humanizer_shaping(seed: u64) -> ShapingConfig {
    ShapingConfig::new(
        foundation::humanizer::HUMANIZER_TEMPERATURE,
        foundation::humanizer::HUMANIZER_TOP_K,
        Some(foundation::humanizer::HUMANIZER_TOP_P),
        foundation::humanizer::HUMANIZER_REPETITION_PENALTY,
        Some(seed),
    )
    .expect("the humanizer recipe constructs")
}

/// The wrapper prompt's token ids: raw completion encoding WITH the BOS
/// (`run_prompt`'s contract), never a template.
pub fn prompt_ids(tokenizer: &MfTokenizer, draft: &str) -> Vec<i32> {
    tokenizer.encode(&foundation::humanizer::build_humanize_prompt(draft), true)
}
