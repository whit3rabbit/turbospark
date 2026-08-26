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

use super::StructuredAssistantDecoder;
use crate::json_value::JsonValue;
use crate::tool_call::{is_valid_function_name, ParsedToolCall};
use crate::{StructuredAssistantEvent, ToolCallParserError};

impl<'a> StructuredAssistantDecoder<'a> {
    /// `gpt-oss`'s Harmony frame. A real assistant turn reads
    ///
    /// ```text
    /// <|channel|>analysis<|message|>REASONING<|end|>
    /// <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>
    /// ```
    ///
    /// and a call reads
    ///
    /// ```text
    /// <|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Oslo"}<|call|>
    /// ```
    ///
    /// **Every transition keys on a TOKEN ID, never on text**, the way the
    /// Gemma arm does: that is what makes the state machine independent of
    /// whether the detokenizer renders special tokens, and of how the header's
    /// words happen to be split into tokens.
    ///
    /// The three tokens that CLOSE a turn -- `<|return|>`, `<|call|>` and
    /// `<|endoftext|>` -- never reach here at all. They are in the dialect's
    /// stop set, and `run_raw_completion` breaks before the progress callback,
    /// so the only frame token this sees inside a body is `<|end|>`, which is
    /// deliberately not a stop (it also closes the PROMPT's system and user
    /// turns). **That is why a tool call is emitted from
    /// [`Self::finish`]**: the token that terminates the call is the one token
    /// the decoder is structurally unable to see.
    ///
    /// A HEADER NAMING A TOOL THE CALLER DID NOT OFFER IS NOT AN ERROR, it is
    /// an ordinary body. Every other dialect fails an unknown tool, but every
    /// other dialect only ever builds this decoder when a request carried
    /// tools; Harmony builds it for the reasoning split too, with an empty
    /// allowlist, so failing here would turn the CLI's every gpt-oss tool call
    /// into a lost turn. Falling back to the channel rule reports the body as
    /// reasoning, which is what a caller that offered no tools can use.
    ///
    /// ONE ORDERING CONSTRAINT IS INHERITED RATHER THAN ENFORCED HERE. A
    /// consumer mapping [`StructuredAssistantEvent::Reasoning`] onto an
    /// Anthropic `thinking` block gets a well-formed stream because Harmony
    /// emits `analysis` BEFORE `final` within one response. This arm reports
    /// whatever order the model produced; it does not reorder to protect a
    /// downstream state machine.
    pub(super) fn consume_harmony(
        &mut self,
        token_id: i32,
        delta: &str,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        if token_id == self.tokenizer.channel_start_id {
            self.label.clear();
            self.harmony = HarmonyState::Header;
            return Ok(Vec::new());
        }
        if token_id == self.tokenizer.message_start_id {
            let header = parse_header(&self.label);
            self.label.clear();
            self.harmony = match header.recipient.filter(|n| self.accepts_tool(n)) {
                Some(name) => HarmonyState::Tool {
                    name,
                    body: String::new(),
                },
                None => HarmonyState::Body(header.channel),
            };
            return Ok(Vec::new());
        }
        if token_id == self.tokenizer.message_end_id {
            // A tool body closed by `<|end|>` rather than by `<|call|>` is
            // still a complete call. Reachable only from a model that framed
            // one that way; the emit path is shared with `finish` so the two
            // cannot drift.
            let events = self.close_harmony_tool()?;
            self.harmony = HarmonyState::Between;
            return Ok(events);
        }
        Ok(match &mut self.harmony {
            HarmonyState::Header => {
                self.label.push_str(delta);
                Vec::new()
            }
            HarmonyState::Tool { body, .. } => {
                body.push_str(delta);
                if body.len() > crate::tool_call::MAXIMUM_BYTES {
                    self.failed = true;
                    return Err(ToolCallParserError::Oversized);
                }
                Vec::new()
            }
            HarmonyState::Between => Vec::new(),
            HarmonyState::Body(_) | HarmonyState::Unframed if delta.is_empty() => Vec::new(),
            HarmonyState::Body(HarmonyChannel::Reasoning) => {
                vec![StructuredAssistantEvent::Reasoning(delta.to_string())]
            }
            HarmonyState::Body(HarmonyChannel::Final) | HarmonyState::Unframed => {
                vec![StructuredAssistantEvent::Content(delta.to_string())]
            }
        })
    }

    /// Whether a header's recipient names a tool this decoder may emit. The
    /// name check is the same one the three DSL parsers apply, so a namespace
    /// separator that survived stripping cannot reach a caller as a function
    /// name.
    fn accepts_tool(&self, name: &str) -> bool {
        is_valid_function_name(name) && self.allowed_tools.contains(name)
    }

    /// Closes an open Harmony tool span, if there is one, parsing its
    /// accumulated body as the call's arguments.
    ///
    /// A body that will not parse is [`ToolCallParserError::Malformed`], which
    /// is the same verdict the Gemma arm reaches on an unterminated tool span.
    /// The common way to get one is a generation that hit its token budget
    /// partway through the JSON.
    pub(super) fn close_harmony_tool(
        &mut self,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        let Some((name, body)) = self.harmony.take_tool() else {
            return Ok(Vec::new());
        };
        let arguments = match JsonValue::parse(body.trim()) {
            // Arguments are an object in every wire format this feeds, and
            // every other parser here builds one. A bare array or scalar is a
            // malformed call rather than a call with odd arguments.
            Ok(value @ JsonValue::Object(_)) => value,
            _ => {
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            }
        };
        let call = ParsedToolCall {
            id: (self.id_generator)(),
            name,
            arguments_json: arguments.encoded(),
            arguments,
        };
        self.emitted_calls += 1;
        Ok(vec![StructuredAssistantEvent::ToolCall(call)])
    }
}
