//! Muse Glimmer recipient-frame handling for `StructuredAssistantDecoder`.
//!
//! **THIS FAMILY REASONS ON EVERY TURN, whatever `--reasoning` says**, which
//! is why its arm is built unconditionally like Harmony's rather than only
//! when a level is asked for. The checkpoint's template calls its
//! `render_reasoning()` macro from the system message with no gate at all and
//! defaults the strength to `high` when the caller sets nothing, so a plain
//! `turbospark-check --model museglimmer-30b.gturbo` prompt already asks for
//! reasoning. Measured on the real 30B install with NO reasoning flag: the
//! reply opened ` to=selfExplain how coastal wetlands reduce flood damage.`
//! followed by the scratch work, with nothing on the reasoning stream.

/// Which side of a Muse Glimmer message the body belongs on.
///
/// Only a message addressed to the USER is the answer. `to=self` is the
/// model's scratchpad and `to=<tool>` carries an `<atem:function_calls>`
/// block; both are reported as reasoning.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum MuseChannel {
    Answer,
    Reasoning,
}

/// Position in Muse Glimmer's `<|start|>ROLE to=RECIPIENT<|message|>BODY<|eom|>`
/// frame.
///
/// Header/body like Harmony's rather than a bracketing pair, but keyed on
/// different tokens and split on a different word: Harmony's first header word
/// is a CHANNEL NAME and the recipient is an optional extra, where here the
/// recipient IS the channel. `<|eot|>` also closes a body and is deliberately
/// absent from this enum's transitions -- it is a STOP token, so
/// `run_raw_completion` breaks before the progress callback and the decoder
/// never sees it (AGENTS.md Gotcha 49). Only `<|eom|>`, which hands off rather
/// than ending the turn, ever reaches `consume`.
pub(super) enum MuseState {
    /// No frame token has been seen and the prompt left none open. Text passes
    /// through as content, so a model that never emits the frame is reported
    /// rather than silenced.
    Unframed,
    /// Inside a header, accumulating its text. **This is the state a real
    /// generation prompt STARTS in**: the template's `add_generation_prompt`
    /// emits `<|start|>assistant` and stops, so the model's first tokens are
    /// the rest of that header (` to=self`) and no `<|start|>` ever arrives.
    Header(String),
    /// Inside a message body.
    Body(MuseChannel),
    /// After `<|eom|>` and before the next `<|start|>`. Text here is dropped;
    /// in a well-formed stream there is none.
    Between,
}

/// The recipient that means "this is the reply".
///
/// A message with NO recipient is also the reply: the checkpoint's template
/// defaults `recipient` to `user` when a turn names none, so an assistant
/// message rendered without one is an ordinary answer.
const USER_RECIPIENT: &str = "user";

/// Which channel a header addresses.
///
/// ANYTHING THAT IS NOT `to=user` IS REASONING, including an unrecognized
/// recipient, and the default direction is the same judgement Harmony's
/// `parse_header` makes for the same reason: a body misreported as reasoning
/// shows up in the wrong place, while one misreported as the answer corrupts
/// the reply. That also routes `to=<toolname>` -- this dialect's tool calls are
/// `<atem:function_calls>` PLAIN TEXT with no parser wired, so putting an
/// unparseable call on the reasoning stream is strictly better than emitting
/// its markup as the reply.
pub(super) fn parse_header(header: &str) -> MuseChannel {
    match header
        .split_whitespace()
        .find_map(|w| w.strip_prefix("to="))
    {
        None => MuseChannel::Answer,
        Some(USER_RECIPIENT) => MuseChannel::Answer,
        Some(_) => MuseChannel::Reasoning,
    }
}

use super::StructuredAssistantDecoder;
use crate::StructuredAssistantEvent;

impl<'a> StructuredAssistantDecoder<'a> {
    /// One token of a Muse Glimmer generation.
    ///
    /// Keys on TOKEN IDS and never on text, like the Harmony arm and for the
    /// same reason: the detokenizer renders `<|start|>`, `<|message|>` and
    /// `<|eom|>` to the EMPTY STRING, so a text-keyed reader would see no
    /// transitions at all (AGENTS.md Gotcha 44).
    ///
    /// `<|eot|>` is absent on purpose -- it is a STOP token, so the loop breaks
    /// before the callback and it never arrives. Nothing here needs it: it ends
    /// a turn rather than opening a channel, and `finish` has nothing to flush
    /// for this dialect because every body is emitted delta by delta as it
    /// arrives.
    pub(super) fn consume_muse(
        &mut self,
        token_id: i32,
        delta: &str,
    ) -> Vec<StructuredAssistantEvent> {
        if token_id == self.tokenizer.channel_start_id {
            self.muse = MuseState::Header(String::new());
            return Vec::new();
        }
        if token_id == self.tokenizer.message_start_id {
            if let MuseState::Header(header) = &self.muse {
                self.muse = MuseState::Body(parse_header(header));
            }
            return Vec::new();
        }
        if token_id == self.tokenizer.message_end_id {
            self.muse = MuseState::Between;
            return Vec::new();
        }
        match &mut self.muse {
            // The header is markup, never output. A recipient split across
            // deltas (`` to=``, ``self``) accumulates here and parses whole.
            MuseState::Header(header) => {
                header.push_str(delta);
                Vec::new()
            }
            MuseState::Between => Vec::new(),
            MuseState::Body(channel) => {
                let channel = *channel;
                if delta.is_empty() {
                    return Vec::new();
                }
                vec![match channel {
                    MuseChannel::Reasoning => {
                        StructuredAssistantEvent::Reasoning(delta.to_string())
                    }
                    MuseChannel::Answer => StructuredAssistantEvent::Content(delta.to_string()),
                }]
            }
            MuseState::Unframed => {
                if delta.is_empty() {
                    Vec::new()
                } else {
                    vec![StructuredAssistantEvent::Content(delta.to_string())]
                }
            }
        }
    }
}
