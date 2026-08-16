//! Harmony channel and stream state handling for StructuredAssistantDecoder.

/// Which Harmony channel a message body belongs to.
///
/// Only `final` is the answer. `analysis` is the model's reasoning, and
/// `commentary` carries tool calls and preambles; both are reported as
/// reasoning rather than as content, so nothing but the final channel is ever
/// presented as the reply.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum HarmonyChannel {
    Final,
    Reasoning,
}

/// Position in Harmony's `<|channel|>HEADER<|message|>BODY<|end|>` frame.
///
/// Harmony is the one dialect here whose channels do not BRACKET: `<|channel|>`
/// opens a header, `<|message|>` ends that header and opens the body, and
/// `<|end|>` closes the body. So the start/end token pair every other arm keys
/// on cannot express it, and reusing the Gemma arm would open a channel label
/// on the first `<|channel|>` and never close it, swallowing the whole reply.
pub(super) enum HarmonyState {
    /// Before the first `<|channel|>`. Text passes through as content, so a
    /// model that never emits the frame is reported rather than silenced.
    Unframed,
    /// Inside a header, accumulating its text.
    Header,
    /// Inside a message body.
    Body(HarmonyChannel),
    /// After `<|end|>` and before the next `<|channel|>`. Everything here is
    /// dropped: what falls in this gap is the `<|start|>assistant` that opens
    /// the next message, and passing it through would emit the bare word
    /// "assistant" into the reply.
    Between,
}

/// Which channel a Harmony message header names.
///
/// A header is the text between `<|channel|>` and `<|message|>`, and it is not
/// always one word: a tool call reads `commentary to=functions.get_weather
/// <|constrain|>json`. The channel is the first whitespace-delimited word, and
/// the rest is deliberately IGNORED here -- Harmony frames a tool call as a
/// recipient in this header rather than as the bracketing token pair
/// [`StructuredAssistantDecoder`]'s tool contract describes, which is why
/// `resolve_harmony` leaves every tool id `NO_SUCH_TOKEN_ID` and why decoding
/// them is its own item.
///
/// ANYTHING THAT IS NOT `final` IS REASONING, including an unrecognized
/// channel name. The default direction matters: a new channel misreported as
/// reasoning is visible in the wrong place, while one misreported as the
/// answer corrupts the reply.
pub(super) fn harmony_channel(header: &str) -> HarmonyChannel {
    match header.split_whitespace().next() {
        Some("final") => HarmonyChannel::Final,
        _ => HarmonyChannel::Reasoning,
    }
}
