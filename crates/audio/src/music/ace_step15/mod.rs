//! ACE-Step 1.5: flow-matching music generation over Oobleck VAE
//! latents, with text2music, cover (restyle from source audio) and
//! repaint tasks.
//!
//! Reference: upstream `ACE-Step/ACE-Step-1.5` at commit
//! [`ca1e85fe9430179831e6bc6be790c332190a3866`](https://github.com/ACE-Step/ACE-Step-1.5/tree/ca1e85fe9430179831e6bc6be790c332190a3866)
//! (MIT) and the checkpoint `ACE-Step/Ace-Step1.5` at revision
//! `19671f406d603126926c1b7e2adc169acbcade22` (MIT). The VAE is the
//! diffusers `AutoencoderOobleck` that upstream loads (0.37.1 per its
//! uv.lock).
//!
//! Status: phase 2 of the port, CPU f32 only: the VAE ([`vae`]), the
//! Qwen3-Embedding text encoder ([`text`]), the condition encoder, the
//! DiT and the turbo sampler ([`dit`]), and the FSQ tokenizer and
//! detokenizer that make cover hints ([`tokenizer`]). See
//! `README.md` in this directory for the phase plan, the gates, and
//! what is deliberately not ported.

pub mod dcw;
pub mod dit;
pub mod pipeline;
pub mod prompt;
pub mod silence;
pub mod text;
pub mod tokenizer;
pub mod transformer;
pub mod vae;

pub use dcw::{Dcw, DcwMode};
pub use dit::{
    cover_switch_step, sample_turbo, turbo_schedule, ConditionEncoder, Conditioning, Dit,
    StepCondition,
};
pub use pipeline::{gaussian, AceStep15, Generated, Stage, Text2Music, Timings, SAMPLE_RATE};
pub use prompt::{caption_prompt, lyric_prompt, Metas, PromptTokenizer, Task};
pub use text::TextEncoder;
pub use tokenizer::{cover_hints, AudioDetokenizer, AudioTokenizer, Fsq};
pub use transformer::{AceConfig, Loader};
pub use vae::{Moments, OobleckConfig, OobleckVae};

/// Pinned upstream ACE-Step commit this port is transcribed from.
pub const REFERENCE_COMMIT: &str = "ca1e85fe9430179831e6bc6be790c332190a3866";

/// Pinned Hugging Face revision of `ACE-Step/Ace-Step1.5`.
pub const CHECKPOINT_REVISION: &str = "19671f406d603126926c1b7e2adc169acbcade22";

#[cfg(test)]
mod dcw_tests;
#[cfg(test)]
mod dit_tests;
#[cfg(test)]
mod pipeline_tests;
#[cfg(test)]
mod prompt_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod text_tests;
