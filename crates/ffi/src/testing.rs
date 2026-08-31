//! Test fixtures and scripted engine scaffolding.
//!
//! Excluded from `turbospark.h` and unreachable from the C ABI.
//! Used by this crate's integration tests to verify channel splitting,
//! cancellation, error propagation, and telemetry without requiring
//! a multi-gigabyte installed model.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::session::{self, Session, SessionCore};
use crate::wire;

/// A session over a scripted producer.
///
/// **Deliberately absent from `turbospark.h`, so it is unreachable from C.**
/// It exists so this crate's own tests can drive the whole generate path --
/// the channel split, the cancel plumbing, the event callback, the result
/// JSON -- on any platform and with no multi-gigabyte install. Without it the
/// threading contract this crate's design turns on would be covered by
/// nothing but a manual click of a Stop button.
#[doc(hidden)]
pub fn session_for_testing(
    tokenizer: tokenizer::MfTokenizer,
    steps: Vec<Vec<foundation::LogitValue>>,
    vocab_size: usize,
    max_context: u32,
) -> Session {
    Session::new(SessionCore {
        engine: Mutex::new(session::Engine::Scripted(Box::new(
            runtime::ScriptedLogitProducer::new(steps),
        ))),
        cancel: Arc::new(AtomicBool::new(false)),
        // No install behind a scripted session, so nothing can read a
        // preprocessor config off it -- which is the same state an install
        // with no vision tower is in, and the refusal names it either way.
        model_dir: std::path::PathBuf::new(),
        max_context,
        rate: runtime::RateControl::default(),
        // A scripted producer replays logits and implements no drafter, so
        // there is nothing to speculate WITH. `None` rather than a policy
        // decision: this is the absence of a capability, not a caller's
        // choice, which is why the reported `reason` is null too.
        speculation_block: None,
        info: wire::SessionInfo {
            model_path: "<scripted>".to_string(),
            family: "<scripted>".to_string(),
            max_context,
            trained_context: None,
            past_trained_context: false,
            expert_cache_slots: 0,
            vocab_size,
            dialect: format!("{:?}", tokenizer.dialect),
            reasoning_support: "none".to_string(),
            steering: wire::SteeringInfo::default(),
            speculation: wire::SpeculationInfo::default(),
            // No tower behind a scripted producer, and `default()` is the
            // absence of a capability rather than a refusal a caller could
            // act on -- which is why `reason` stays null here too.
            vision: wire::VisionInfo::default(),
            special_tokens: wire::SpecialTokensInfo {
                bos_id: (tokenizer.bos_id >= 0).then_some(tokenizer.bos_id),
                eos_id: (tokenizer.eos_id >= 0).then_some(tokenizer.eos_id),
                pad_id: (tokenizer.pad_id >= 0).then_some(tokenizer.pad_id),
                end_of_turn_id: (tokenizer.end_of_turn_id >= 0).then_some(tokenizer.end_of_turn_id),
                stop_token_ids: tokenizer.stop_token_ids.iter().copied().collect(),
                think_start_id: tokenizer.think_start_id,
                think_end_id: tokenizer.think_end_id,
            },
        },
        tokenizer,
    })
}
