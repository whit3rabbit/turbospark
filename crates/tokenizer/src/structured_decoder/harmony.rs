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
    /// Inside the body of a message whose header named a tool. The body is raw
    /// JSON rather than a DSL, so it is accumulated verbatim and handed to
    /// [`crate::json_value::JsonValue::parse`] when the span closes.
    ///
    /// **The span usually closes at [`StructuredAssistantDecoder::finish`]
    /// rather than at a token**, which is the opposite of every other dialect's
    /// tool span: Harmony ends a call with `<|call|>`, `<|call|>` is in the stop
    /// set, and `run_raw_completion` breaks before the progress callback, so the
    /// decoder never sees the token that terminates the thing it is parsing.
    Tool { name: String, body: String },
    /// After `<|end|>` and before the next `<|channel|>`. Everything here is
    /// dropped: what falls in this gap is the `<|start|>assistant` that opens
    /// the next message, and passing it through would emit the bare word
    /// "assistant" into the reply.
    Between,
}

impl HarmonyState {
    /// Removes an open tool span and returns its `(name, body)`, leaving the
    /// machine between messages. `None` for every other state, which is left
    /// untouched.
    pub(super) fn take_tool(&mut self) -> Option<(String, String)> {
        match std::mem::replace(self, HarmonyState::Between) {
            HarmonyState::Tool { name, body } => Some((name, body)),
            other => {
                *self = other;
                None
            }
        }
    }
}

/// What a Harmony message header names: a channel, and optionally a tool.
pub(super) struct HarmonyHeader {
    pub(super) channel: HarmonyChannel,
    /// The bare function name a `to=functions.NAME` recipient named, with the
    /// namespace stripped. `None` when the header names no recipient, or names
    /// one outside the `functions` namespace (see [`parse_header`]).
    pub(super) recipient: Option<String>,
}

/// The namespace Harmony renders CALLER-SUPPLIED tools into. Builtin tools live
/// in their own namespaces (`browser`, `python`) and are deliberately NOT
/// reachable through this: they are not the caller's tools, `python`'s body is
/// source code rather than JSON, and treating one as a call would fail to parse
/// and poison the stream rather than degrade.
const FUNCTIONS_NAMESPACE: &str = "functions.";

/// What a Harmony message header names.
///
/// A header is the text between `<|channel|>` and `<|message|>`, and it is not
/// always one word: a tool call reads `commentary to=functions.get_weather
/// <|constrain|>json`. The channel is the first whitespace-delimited word and
/// the recipient is the `to=` one; `<|constrain|>` is a special token that
/// detokenizes to nothing, so the constraint arrives as a bare `json` word and
/// is ignored (this decoder validates the body by parsing it, not by trusting
/// the header's claim about it).
///
/// ANYTHING THAT IS NOT `final` IS REASONING, including an unrecognized
/// channel name. The default direction matters: a new channel misreported as
/// reasoning is visible in the wrong place, while one misreported as the
/// answer corrupts the reply.
///
/// THE RECIPIENT IS ONLY EVER READ OFF THIS HEADER. Harmony also allows a
/// `to=` after the role in a `<|start|>` header, but that form addresses a
/// message TO the assistant (a tool's response), not a call FROM it, so a
/// generation never contains one.
pub(super) fn parse_header(header: &str) -> HarmonyHeader {
    let mut words = header.split_whitespace();
    let channel = match words.next() {
        Some("final") => HarmonyChannel::Final,
        _ => HarmonyChannel::Reasoning,
    };
    let recipient = header
        .split_whitespace()
        .find_map(|w| w.strip_prefix("to="))
        .and_then(|to| to.strip_prefix(FUNCTIONS_NAMESPACE))
        .map(str::to_string);
    HarmonyHeader { channel, recipient }
}
