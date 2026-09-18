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
    session_for_testing_named(tokenizer, steps, vocab_size, max_context, "<scripted>")
}

/// [`session_for_testing`] with a chosen `model_path`.
///
/// **THE SERVER'S MODEL ID IS DERIVED FROM `model_path`**
/// (`api/server.rs::model_id_of`), so a test that attaches two scripted
/// sessions to one server needs two paths or both entries collide -- and
/// the collision is itself worth testing, which is why the default stays
/// `<scripted>` and this is the variant rather than the other way round.
#[doc(hidden)]
pub fn session_for_testing_named(
    tokenizer: tokenizer::MfTokenizer,
    steps: Vec<Vec<foundation::LogitValue>>,
    vocab_size: usize,
    max_context: u32,
    model_path: &str,
) -> Session {
    Session::new(SessionCore {
        engine: Mutex::new(session::Engine::Scripted(Box::new(
            runtime::ScriptedLogitProducer::new(steps),
        ))),
        cancel: Arc::new(AtomicBool::new(false)),
        max_context,
        rate: runtime::RateControl::default(),
        // A scripted producer replays logits and implements no drafter, so
        // there is nothing to speculate WITH. `None` rather than a policy
        // decision: this is the absence of a capability, not a caller's
        // choice, which is why the reported `reason` is null too.
        speculation_block: None,
        info: wire::SessionInfo {
            model_path: model_path.to_string(),
            family: "<scripted>".to_string(),
            max_context,
            trained_context: None,
            past_trained_context: false,
            expert_cache_slots: 0,
            expert_residency: "streamed".to_string(),
            vocab_size,
            dialect: format!("{:?}", tokenizer.dialect),
            // ASKED rather than asserted, and that is a fix rather than
            // tidying. This read `"none"` unconditionally while
            // `generate` gates a requested level on
            // `tokenizer.reasoning_support()` directly -- so a scripted
            // session over a fixture that DOES ship a template reported a
            // capability it had, as absent, to the one field a GUI reads.
            reasoning_support: match tokenizer.reasoning_support() {
                tokenizer::ReasoningSupport::Level => "level",
                tokenizer::ReasoningSupport::ToggleOnly => "toggleOnly",
                tokenizer::ReasoningSupport::None => "none",
            }
            .to_string(),
            reasoning_levels: tokenizer
                .accepted_reasoning_levels()
                .into_iter()
                .map(|level| level.as_str().to_string())
                .collect(),
            // A scripted producer runs no decode flow, so there is no
            // boundary for the edit to sit on and `supported` is honestly
            // false rather than defaulted into. `open` refuses a vector on a
            // scripted session for the same reason.
            steering: wire::SteeringInfo {
                reason: Some(
                    "a scripted session runs no decode flow, so there is no residual stream \
                     to steer"
                        .to_string(),
                ),
                ..Default::default()
            },
            // ASKED rather than asserted, for the reason the reasoning field
            // above records: the dialect is real even when the producer is
            // not, and a fixture whose dialect DOES frame tool calls should
            // not report otherwise.
            tool_calling: wire::ToolCallingInfo {
                native: tokenizer.dialect.tool_call_support() == tokenizer::ToolCallSupport::Native,
                reason: tokenizer.dialect.tool_call_unsupported_reason(),
            },
            speculation: wire::SpeculationInfo::default(),
            // No tower behind a scripted producer, and `default()` is the
            // absence of a capability rather than a refusal a caller could
            // act on -- which is why `reason` stays null here too.
            vision: wire::VisionInfo::default(),
            // A scripted producer's KV cache does not exist to quantize, so
            // this reads exactly what a real session with `kvBits` absent
            // would: `off`.
            kv_bits: "off".to_string(),
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
        // A scripted producer has no install to budget against and no
        // tower to attach a pixel ceiling for -- `LoadPolicy::default()`
        // and zero committed/KV bytes are the same "absence of a
        // capability" answer `vision: wire::VisionInfo::default()` above
        // already gives.
        load_policy: runtime::LoadPolicy::default(),
        committed_bytes: 0,
        kv_bytes: 0,
    })
}
