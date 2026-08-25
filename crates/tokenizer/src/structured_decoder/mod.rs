//! Streaming assistant-output decoder: splits generated tokens/text into
//! visible content and parsed tool calls per dialect. Ported from
//! `Tokenization/StructuredAssistantDecoder.swift`.

mod chatml;
mod deepseek;
mod harmony;
mod muse;

use std::collections::HashSet;

use self::harmony::{HarmonyChannel, HarmonyState};
use self::muse::{MuseChannel, MuseState};
use crate::dialect::{ChatDialect, MfTokenizer};
use crate::error::ToolCallParserError;
use crate::json_value::JsonValue;
use crate::tool_call::{is_valid_function_name, GemmaToolCallParser, ParsedToolCall};

/// Decoded event emitted by the structured assistant output decoder.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuredAssistantEvent {
    /// Text content chunk for display.
    Content(String),
    /// Reasoning the model produced on its way to the answer, separated from
    /// the answer itself.
    ///
    /// The Harmony, ChatML and Gemma arms all emit this. Harmony always did,
    /// because `gpt-oss` puts most of its generated tokens in that channel;
    /// the other two started when `--reasoning` gave a caller a way to turn
    /// thinking ON, at which point discarding the body destroyed exactly what
    /// had been asked for. What still differs per dialect is the FRAME
    /// (Harmony's header/body triple, ChatML's `<think>` pair, Gemma's
    /// bracketing channel pair) -- and, for ChatML, whether the generation
    /// prompt already opened that frame. See [`StructuredAssistantDecoder::new`].
    Reasoning(String),
    /// Parsed tool call invocation.
    ToolCall(ParsedToolCall),
}

#[derive(PartialEq)]
enum Channel {
    Thought,
    Visible,
    Label,
}

/// Streaming assistant output decoder splitting tokens into visible content and tool calls.
pub struct StructuredAssistantDecoder<'a> {
    tokenizer: &'a MfTokenizer,
    allowed_tools: HashSet<String>,
    id_generator: Box<dyn FnMut() -> String + 'a>,
    channel: Channel,
    harmony: HarmonyState,
    muse: MuseState,
    label: String,
    tool_tokens: Option<Vec<i32>>,
    held_text: String,
    dsml_text: Option<String>,
    emitted_calls: usize,
    failed: bool,
}

/// Where `prompt_ids` leaves a Muse Glimmer generation, read off the last
/// frame token in it.
///
/// THE SAME QUESTION [`prompt_opens_thought`] ASKS FOR ChatML, on the other
/// frame shape, and the answer is load-bearing for the same reason: the
/// checkpoint's `add_generation_prompt` emits `<|start|>assistant` and STOPS,
/// so a real generation begins inside a header and no `<|start|>` ever
/// arrives. Starting [`MuseState::Unframed`] instead would pass the header
/// remainder (` to=self`) and then the whole scratchpad through as the reply,
/// which is exactly what this dialect did before it had an arm at all.
///
/// The `<|message|>` case cannot arise from this checkpoint's own template and
/// is handled rather than defaulted, because guessing a channel there picks
/// between losing the answer and corrupting it: the header is recovered by
/// decoding back to the `<|start|>` that opened it.
fn muse_state_for(tokenizer: &MfTokenizer, prompt_ids: &[i32]) -> MuseState {
    let start = tokenizer.channel_start_id;
    let message = tokenizer.message_start_id;
    let eom = tokenizer.message_end_id;

    for (i, &id) in prompt_ids.iter().enumerate().rev() {
        if id == start {
            // Seed the accumulator with the header text already rendered (the
            // role word), so a recipient the model appends to it parses in one
            // piece.
            return MuseState::Header(tokenizer.decode(&prompt_ids[i + 1..], true));
        }
        if id == message {
            let header_start = prompt_ids[..i]
                .iter()
                .rposition(|&p| p == start)
                .map(|p| p + 1)
                .unwrap_or(0);
            let header = tokenizer.decode(&prompt_ids[header_start..i], true);
            return MuseState::Body(muse::parse_header(&header));
        }
        if id == eom {
            return MuseState::Between;
        }
    }
    MuseState::Unframed
}

/// True when `prompt_ids` ends INSIDE an open thought frame, i.e. the last
/// `<think>`/`</think>` in the rendered generation prompt is the opening one.
///
/// Scans backwards and stops at the first of the two it finds, so the
/// balanced pairs a tool preamble puts in its instructions ("use the
/// `<think></think>` block to plan") cannot outvote the tail.
///
/// Always false for a dialect whose thought frame is not this pair (Gemma,
/// Harmony, Muse Glimmer): both ids are `None` there, so nothing matches.
fn prompt_opens_thought(tokenizer: &MfTokenizer, prompt_ids: &[i32]) -> bool {
    for &id in prompt_ids.iter().rev() {
        if tokenizer.think_start_id == Some(id) {
            return true;
        }
        if tokenizer.think_end_id == Some(id) {
            return false;
        }
    }
    false
}

impl<'a> StructuredAssistantDecoder<'a> {
    /// Creates a structured assistant decoder over the turn whose generation
    /// prompt is `prompt_ids`.
    ///
    /// **THE PROMPT IS AN ARGUMENT BECAUSE A CHATML PROMPT CAN OPEN THE
    /// THOUGHT FRAME ITSELF, AND THEN THE MODEL NEVER EMITS `<think>`.** Qwen's
    /// own template ends `<|im_start|>assistant\n<think>\n` when thinking is on
    /// and `<|im_start|>assistant\n<think>\n\n</think>\n\n` when it is off, so
    /// the FIRST token of a thinking turn is already scratchpad. A decoder that
    /// always starts in [`Channel::Visible`] therefore stays there -- the
    /// `</think>` that arrives later flips Visible to Visible, a no-op -- and
    /// the whole scratchpad is reported as the answer, with the reasoning
    /// stream empty. That is what `--reasoning` did on every ChatML checkpoint
    /// until this parameter existed.
    ///
    /// Passing `&[]` means "nothing was prefilled" and reproduces the old
    /// behaviour exactly; it is what the unit tests below drive their own
    /// synthetic token streams with.
    ///
    /// Deriving this from the RENDERED PROMPT rather than from the reasoning
    /// level is what makes it unable to be wrong: a checkpoint whose template
    /// enables thinking without prefilling the tag is read correctly by the
    /// same scan, where a level-keyed guess would open a frame the model is
    /// about to open again and swallow the reply.
    pub fn new(
        tokenizer: &'a MfTokenizer,
        allowed_tools: HashSet<String>,
        id_generator: impl FnMut() -> String + 'a,
        prompt_ids: &[i32],
    ) -> Self {
        Self {
            tokenizer,
            allowed_tools,
            id_generator: Box::new(id_generator),
            channel: if prompt_opens_thought(tokenizer, prompt_ids) {
                Channel::Thought
            } else {
                Channel::Visible
            },
            harmony: HarmonyState::Unframed,
            muse: muse_state_for(tokenizer, prompt_ids),
            label: String::new(),
            tool_tokens: None,
            held_text: String::new(),
            dsml_text: None,
            emitted_calls: 0,
            failed: false,
        }
    }

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
    fn consume_muse(&mut self, token_id: i32, delta: &str) -> Vec<StructuredAssistantEvent> {
        if token_id == self.tokenizer.channel_start_id {
            self.muse = MuseState::Header(String::new());
            return Vec::new();
        }
        if token_id == self.tokenizer.message_start_id {
            if let MuseState::Header(header) = &self.muse {
                self.muse = MuseState::Body(muse::parse_header(header));
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

    /// Returns true if at least one tool call has been parsed and emitted.
    pub fn has_tool_calls(&self) -> bool {
        self.emitted_calls > 0
    }

    /// Consumes flushed text snippet during stream decoding.
    pub fn consume_flushed_text(
        &mut self,
        text: &str,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        self.consume(-1, text)
    }

    /// Consumes a token ID and text delta, returning any parsed events.
    pub fn consume(
        &mut self,
        token_id: i32,
        delta: &str,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        if self.failed {
            return Err(ToolCallParserError::Malformed);
        }
        match self.tokenizer.dialect {
            ChatDialect::ChatMl => return self.consume_chatml(token_id, delta),
            ChatDialect::Deepseek => return self.consume_deepseek(token_id, delta),
            ChatDialect::MuseGlimmer => return Ok(self.consume_muse(token_id, delta)),
            ChatDialect::Gemma => {}
            // Nothing to decode: this checkpoint has no tool-call or
            // thinking markup, and its channel/tool ids are all
            // `NO_SUCH_TOKEN_ID`. Falling through to the Gemma arm would
            // compare every token against that sentinel, which is harmless
            // but says something untrue about the dialect.
            // Same reasoning as Mistral immediately above: no tool-call or
            // thinking markup in this dialect's table, so its channel/tool
            // ids are all `NO_SUCH_TOKEN_ID` and falling through to the
            // Gemma arm below would compare every token against that
            // sentinel harmlessly but say something untrue about the dialect.
            ChatDialect::Mistral | ChatDialect::Llama3 => {
                return Ok(if delta.is_empty() {
                    Vec::new()
                } else {
                    vec![StructuredAssistantEvent::Content(delta.to_string())]
                })
            }
            ChatDialect::Harmony => return self.consume_harmony(token_id, delta),
        }

        if token_id == self.tokenizer.channel_start_id {
            self.label.clear();
            self.channel = Channel::Label;
            return Ok(Vec::new());
        }
        if token_id == self.tokenizer.channel_end_id {
            self.channel = Channel::Visible;
            return Ok(Vec::new());
        }
        if token_id == self.tokenizer.tool_call_start_id {
            if self.tool_tokens.is_some() {
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            }
            self.tool_tokens = Some(Vec::new());
            return Ok(Vec::new());
        }
        if token_id == self.tokenizer.tool_call_end_id {
            let Some(tokens) = self.tool_tokens.take() else {
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            };
            let text = self.tokenizer.decode(&tokens, false);
            match GemmaToolCallParser::new().parse(
                &text,
                &self.allowed_tools,
                &(self.id_generator)(),
            ) {
                Ok(call) => {
                    self.emitted_calls += 1;
                    return Ok(vec![StructuredAssistantEvent::ToolCall(call)]);
                }
                Err(e) => {
                    self.failed = true;
                    return Err(e);
                }
            }
        }
        if token_id == self.tokenizer.tool_response_id
            || token_id == self.tokenizer.tool_response_end_id
        {
            if self.emitted_calls == 0 || self.tool_tokens.is_some() {
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            }
            return Ok(Vec::new());
        }
        if let Some(tokens) = &mut self.tool_tokens {
            tokens.push(token_id);
            if tokens.len() * 4 > crate::tool_call::MAXIMUM_BYTES {
                self.failed = true;
                return Err(ToolCallParserError::Oversized);
            }
            return Ok(Vec::new());
        }
        match self.channel {
            // EMITTED, NOT DISCARDED, for the reason the ChatML arm's
            // `<think>` body is: a caller that asked for reasoning must be
            // able to see it, and the alternative is not "hidden" but
            // "destroyed". Discarding was right while no knob could turn
            // thinking on; `--reasoning` / `reasoning_effort` is that knob.
            //
            // The CHANNEL LABEL is still swallowed either way (it is parsed
            // in the `Label` arm and never emitted), which is what stops a
            // bare `thought` appearing in the stream -- the exact leak this
            // arm's caller produced the first time a level was asked for on
            // a real Gemma install.
            Channel::Thought => Ok(if delta.is_empty() {
                Vec::new()
            } else {
                vec![StructuredAssistantEvent::Reasoning(delta.to_string())]
            }),
            Channel::Visible => {
                if delta.is_empty() {
                    Ok(Vec::new())
                } else {
                    Ok(vec![StructuredAssistantEvent::Content(delta.to_string())])
                }
            }
            Channel::Label => {
                self.label.push_str(delta);
                let Some(newline) = self.label.find('\n') else {
                    return Ok(Vec::new());
                };
                let name = self.label[..newline].trim().to_lowercase();
                let content = self.label[newline + 1..].to_string();
                self.channel = if name == "final" || name == "answer" {
                    Channel::Visible
                } else {
                    Channel::Thought
                };
                self.label.clear();
                if self.channel == Channel::Visible && !content.is_empty() {
                    Ok(vec![StructuredAssistantEvent::Content(content)])
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }

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
    fn consume_harmony(
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
            let header = harmony::parse_header(&self.label);
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
    fn close_harmony_tool(&mut self) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
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

    /// Release any tail withheld as a potential DSML-open prefix.
    pub fn drain(&mut self) -> Vec<StructuredAssistantEvent> {
        if self.failed || self.dsml_text.is_some() || self.held_text.is_empty() {
            return Vec::new();
        }
        let visible = std::mem::take(&mut self.held_text);
        vec![StructuredAssistantEvent::Content(visible)]
    }

    /// Ends the stream, releasing anything the decoder was still holding.
    ///
    /// **A HARMONY TOOL CALL IS EMITTED HERE AND NOWHERE ELSE IN PRACTICE**,
    /// which inverts every other dialect's tool span. `<|call|>` terminates a
    /// Harmony call and `<|call|>` is a stop token, so `run_raw_completion`
    /// breaks out of its loop before the progress callback ever sees it: a
    /// consumer that never calls this gets the reasoning and the content and
    /// silently drops every call. For the other dialects an open tool span at
    /// this point is instead the error it looks like -- their terminator is an
    /// ordinary token that arrived or did not.
    pub fn finish(&mut self) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        let released = self.drain();
        if self.failed || self.tool_tokens.is_some() || self.dsml_text.is_some() {
            return Err(ToolCallParserError::Malformed);
        }
        let mut events = self.close_harmony_tool()?;
        events.extend(released);
        Ok(events)
    }
}
