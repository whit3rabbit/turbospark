//! Streaming assistant-output decoder: splits generated tokens/text into
//! visible content and parsed tool calls per dialect. Ported from
//! `Tokenization/StructuredAssistantDecoder.swift`.

mod chatml;
mod deepseek;
mod harmony;
mod mistral;
mod muse;

use std::collections::HashSet;

use self::harmony::HarmonyState;
use self::muse::MuseState;
use crate::dialect::{ChatDialect, MfTokenizer};
use crate::error::ToolCallParserError;
use crate::tool_call::{GemmaToolCallParser, ParsedToolCall};

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
    /// The buffered body of the span the decoder gave up on, held as token
    /// ids and decoded only when [`Self::take_failed_span_text`] asks. Every
    /// other route out of a tool span (a parsed call, a clean close) empties
    /// `tool_tokens` into the parser; this is what happens to those tokens
    /// on the one route that used to drop them instead.
    failed_span_tokens: Option<Vec<i32>>,
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
            failed_span_tokens: None,
            held_text: String::new(),
            dsml_text: None,
            emitted_calls: 0,
            failed: false,
        }
    }

    /// Returns true if at least one tool call has been parsed and emitted.
    pub fn has_tool_calls(&self) -> bool {
        self.emitted_calls > 0
    }

    /// Releases the decoded body of the tool span the decoder failed on, if
    /// it was holding one.
    ///
    /// A parser error inside a tool span is permanent (`consume` answers
    /// `Err` for every later token), and the bytes the span had already
    /// buffered are NOT parser output -- they are ordinary model output the
    /// span framing happened to swallow. A consumer that degrades to raw
    /// text on an error should release this beside the erroring event, or a
    /// pre-3.5 Qwen writing bare JSON inside `<tool_call>` / `</tool_call>`
    /// loses the call twice over: the decoder refuses it AND the text the
    /// server's rescue layer would parse it out of never reaches
    /// `generated.text`. `None` when the error came from somewhere with no
    /// buffered span (a stray end token, a failed `finish`).
    pub fn take_failed_span_text(&mut self) -> Option<String> {
        let tokens = self.failed_span_tokens.take()?;
        let text = self.tokenizer.decode(&tokens, false);
        (!text.is_empty()).then_some(text)
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
            ChatDialect::MiniMax => {
                if Some(token_id) == self.tokenizer.think_start_id {
                    self.channel = Channel::Thought;
                } else if Some(token_id) == self.tokenizer.think_end_id {
                    self.channel = Channel::Visible;
                } else if !delta.is_empty() {
                    let event = if self.channel == Channel::Thought {
                        StructuredAssistantEvent::Reasoning(delta.to_string())
                    } else {
                        StructuredAssistantEvent::Content(delta.to_string())
                    };
                    return Ok(vec![event]);
                }
                return Ok(Vec::new());
            }
            // Spark SHARES the DeepSeek arm on purpose: its reasoning frame
            // is the same think-open/think-close pair resolved into the same
            // ids, so the channel split is identical. What differs is only
            // what the arm never sees -- Spark has no DSML tool markers, so
            // its `<tool_call>` markup flows through as ordinary content
            // (`Prompted`, per `tool_call_support`), which is the honest
            // route while that DSL has no parser.
            ChatDialect::Deepseek | ChatDialect::Spark => {
                return self.consume_deepseek(token_id, delta)
            }
            // V2-era tables carry NO structural markup at all: plain text in,
            // plain text out, exactly the Llama3 arm's contract (all ids are
            // NO_SUCH_TOKEN_ID, nothing can open a span).
            ChatDialect::DeepseekV2 => {
                return Ok(if delta.is_empty() {
                    Vec::new()
                } else {
                    vec![StructuredAssistantEvent::Content(delta.to_string())]
                })
            }
            ChatDialect::MuseGlimmer => return Ok(self.consume_muse(token_id, delta)),
            ChatDialect::Gemma => {}
            // Mistral's `[TOOL_CALLS]` arm: the marker is a SPECIAL token, so
            // it keys on the id like the Gemma arm, but the span it opens has
            // no closing token -- the call body runs to end of turn -- and
            // that span is closed by `finish`, not here.
            ChatDialect::Mistral => return self.consume_mistral(token_id, delta),
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
            ChatDialect::Llama3 => {
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
            if let Some(tokens) = self.tool_tokens.take() {
                // A second start inside an open span abandons the first one;
                // what it had buffered is still model output.
                self.failed_span_tokens = Some(tokens);
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
                    debug_assert_eq!(
                        self.tokenizer.dialect.tool_call_support(),
                        crate::ToolCallSupport::Native,
                        "a dialect that hands a parsed tool call over must answer Native to \
                         `tool_call_support`; the two matches have drifted"
                    );
                    return Ok(vec![StructuredAssistantEvent::ToolCall(call)]);
                }
                Err(e) => {
                    // The span's body was decoded for this parse and is what
                    // the caller most wants back: release it beside the
                    // error rather than drop it with the failed attempt.
                    self.failed_span_tokens = Some(tokens);
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
            // Checked BEFORE pushing `token_id`, so the clone below -- the
            // span body `turn_stream::feed` releases on this error -- never
            // includes the token that tripped the limit. That token is
            // ordinary content (not a special/marker token, unlike the
            // "second start"/parse-error arms above), so `feed` also emits
            // ITS OWN delta right after releasing the span; including it in
            // both would print its text twice.
            if (tokens.len() + 1) * 4 > crate::tool_call::MAXIMUM_BYTES {
                if self.failed_span_tokens.is_none() {
                    self.failed_span_tokens = Some(tokens.clone());
                }
                self.failed = true;
                return Err(ToolCallParserError::Oversized);
            }
            tokens.push(token_id);
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
    /// silently drops every call. **MISTRAL JOINS IT FOR A STRUCTURALLY
    /// IDENTICAL REASON, and worse**: a `[TOOL_CALLS]` span has no closing
    /// token at all -- the body runs to end of turn, and end of turn is
    /// `</s>`, a stop token -- so `finish` is the only place the span can be
    /// closed and parsed. For the REMAINING dialects an open tool span at
    /// this point is instead the error it looks like -- their terminator is
    /// an ordinary token that arrived or did not.
    pub fn finish(&mut self) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        let released = self.drain();
        if self.failed || self.dsml_text.is_some() {
            return Err(ToolCallParserError::Malformed);
        }
        let mut events = match self.tokenizer.dialect {
            ChatDialect::Harmony => self.close_harmony_tool()?,
            ChatDialect::Mistral => self.close_mistral_tool()?,
            _ => {
                if self.tool_tokens.is_some() {
                    return Err(ToolCallParserError::Malformed);
                }
                Vec::new()
            }
        };
        events.extend(released);
        Ok(events)
    }
}
