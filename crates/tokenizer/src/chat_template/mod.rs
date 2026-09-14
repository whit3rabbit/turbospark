//! Text-only chat-template rendering for the four dialects, plus DeepSeek's
//! hand-rolled (non-Jinja) native tool chat. Ported from the chat-template
//! section of `Tokenization/Tokenizer.swift`.
//!
//! These per-dialect renderers are now the FALLBACK rather than the primary
//! path: `apply_chat_template` prefers the checkpoint's own installed Jinja
//! template (see its doc comment for why, and [`crate::jinja_chat_template`]
//! for the render). They still serve every checkpoint that ships no
//! template -- all the synthetic fixtures here, and DeepSeek, whose native
//! tool chat is plain string composition and is ported in full below.

mod chatml;
mod deepseek;
mod gemma;
mod llama3;
mod mistral;

use crate::dialect::{
    ChatDialect, MfTokenizer, HARMONY_END_MARK, HARMONY_MESSAGE_MARK, HARMONY_START_MARK,
    MUSE_EOT_MARK, MUSE_MESSAGE_MARK, MUSE_START_MARK, SPARK_BOS_MARK, SPARK_BOT_MARK,
    SPARK_EOS_MARK, SPARK_USER_MARK,
};
use crate::error::TokenizerError;
use crate::json_value::JsonValue;
use crate::reasoning::ReasoningEffort;

/// Message sender role discriminator for chat templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// System instructions role.
    System,
    /// Developer instructions role.
    Developer,
    /// User input role.
    User,
    /// Assistant response role.
    Assistant,
    /// Tool execution result role.
    Tool,
}

impl Role {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::Developer => "developer",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// Recorded historical tool call invocation in a chat turn.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalToolCall {
    /// Tool call ID string.
    pub id: String,
    /// Called function name.
    pub name: String,
    /// Function arguments JSON structure.
    pub arguments: JsonValue,
}

/// Function schema definition for tool use.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition {
    /// Defined tool function name.
    pub name: String,
    /// Description of tool function behavior.
    pub description: String,
    /// JSON schema describing expected function parameters.
    pub parameters: JsonValue,
}

/// One ordered piece of a multimodal message's content (ROADMAP M-V6).
///
/// **An image is a PLACEHOLDER here and carries no pixels.** A template
/// renders it as one marker run and the tower's rows are injected later, by
/// position (`runtime::vision::PromptVision`), so the two halves meet at the
/// id sequence rather than inside this type. That is also why there is no
/// path, no bytes and no grid on this variant: everything about the image
/// except WHERE it sits is somebody else's business.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    /// Literal text.
    Text(String),
    /// One image, in prompt order.
    Image,
}

/// Single chat message in a conversation sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// Sender role.
    pub role: Role,
    /// Text content string if present.
    ///
    /// On a message built by [`Message::with_parts`] this is the TEXT-ONLY
    /// view, kept so a consumer that summarizes or counts still sees prose.
    /// It is not what renders -- see `content_parts`.
    pub content: Option<String>,
    /// Ordered content parts, EMPTY on every text-only message.
    ///
    /// **Non-empty is what makes this a multimodal message, and then this
    /// field is the one that renders**: `content` becomes a summary and is
    /// ignored by the template. Two fields rather than one because
    /// `content: Option<String>` is read in over fifty places that have
    /// nothing to do with vision, and widening it would put a content-part
    /// match in all of them to serve one family.
    pub content_parts: Vec<ContentPart>,
    /// Tool calls invoked by assistant.
    pub tool_calls: Vec<HistoricalToolCall>,
    /// Tool call ID if role is Tool.
    pub tool_call_id: Option<String>,
    /// Optional name of function or tool sender.
    pub name: Option<String>,
}

impl Message {
    /// Constructs a chat message with a role and text content string.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            content_parts: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Constructs a multimodal message from ordered content parts.
    ///
    /// `content` is set to the parts' TEXT joined, so a consumer reading it
    /// for a summary sees prose rather than `None`; the parts are what the
    /// template renders.
    pub fn with_parts(role: Role, parts: Vec<ContentPart>) -> Self {
        let text: String = parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.as_str()),
                ContentPart::Image => None,
            })
            .collect();
        Self {
            role,
            content: Some(text),
            content_parts: parts,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Images this message carries, in order.
    pub fn image_count(&self) -> usize {
        self.content_parts
            .iter()
            .filter(|p| matches!(p, ContentPart::Image))
            .count()
    }
}

impl MfTokenizer {
    /// Formats a sequence of messages into a chat template string.
    ///
    /// THE CHECKPOINT'S OWN TEMPLATE WINS when it ships one, because framing
    /// is a property of the CHECKPOINT and the dialect is only a property of
    /// its special-token table. The two are not in one-to-one correspondence:
    /// TinyLlama-1.1B-Chat and Mistral-7B-Instruct present the identical
    /// `<s>`/`</s>` table, resolve to the same [`ChatDialect::Mistral`], and
    /// are trained on Zephyr and `[INST]` framing respectively. Fed the wrong
    /// one, a model echoes the markup back instead of answering -- fluent,
    /// and not an answer.
    ///
    /// The dialect keeps everything it is actually evidence for: BOS/EOS and
    /// turn ids, the stop set, and the fallback render for a checkpoint that
    /// ships no template at all (every synthetic fixture here, and DeepSeek,
    /// whose native tool chat is hand-rolled in this file).
    ///
    /// The renderers below differ from a real template by exactly one
    /// thing, and it is worth knowing before changing either: they
    /// `trim()` message content unconditionally, where a template trims
    /// only if it says `| trim`. Gemma's and Qwen 3.6's do, so those rows
    /// are byte-identical; Qwen3-30B-A3B's does not, and that one trailing
    /// newline re-froze its quality-gate row. `tests/installed_template.rs`
    /// pins the behaviour per family.
    ///
    /// A template that fails to render ERRORS rather than falling back.
    /// The fallback would be a renderer this checkpoint is known not to
    /// match, and its output is fluent -- the whole failure mode above.
    /// A visible error is the better of the two.
    pub fn apply_chat_template(&self, messages: &[Message]) -> Result<String, TokenizerError> {
        self.apply_chat_template_with_reasoning(messages, ReasoningEffort::default())
    }

    /// [`Self::apply_chat_template`], asking the checkpoint's own template for
    /// a reasoning level.
    ///
    /// At [`ReasoningEffort::Off`] -- the default, and what
    /// `apply_chat_template` passes -- this renders the same bytes it always
    /// did, which is what leaves `crates/bench`'s frozen digests alone.
    ///
    /// **A LEVEL IS REFUSED RATHER THAN IGNORED when the checkpoint ships no
    /// template**, because the per-dialect fallback renderers below are fixed
    /// strings with no reasoning knob in them: honouring the request is
    /// impossible and dropping it silently would give a fluent answer that
    /// simply did not think, with nothing anywhere saying so. A template that
    /// exists but names no effort key is NOT refused -- see
    /// [`ReasoningSupport::ToggleOnly`], where thinking is still a real
    /// effect and only the level is unexpressible; that case is the caller's
    /// to warn about.
    ///
    /// [`ReasoningSupport::ToggleOnly`]: crate::ReasoningSupport::ToggleOnly
    pub fn apply_chat_template_with_reasoning(
        &self,
        messages: &[Message],
        reasoning: ReasoningEffort,
    ) -> Result<String, TokenizerError> {
        if self.chat_template_source.is_some() {
            return crate::jinja_chat_template::render_generic_chat_template(
                self,
                messages,
                &[],
                true,
                reasoning,
            );
        }
        if reasoning.enable_thinking() {
            return Err(TokenizerError::UnsupportedForDialect(format!(
                "reasoning effort {} was requested, but this checkpoint ships no chat template \
                 and the {:?} fallback renderer has no reasoning knob to set",
                reasoning.as_str(),
                self.dialect
            )));
        }
        self.apply_dialect_chat_template(messages)
    }

    /// The hand-written per-dialect render, bypassing any installed
    /// template. Public so the guard test can compare the two.
    ///
    /// **AN IMAGE PART IS REFUSED HERE RATHER THAN DROPPED** (ROADMAP M-V6).
    /// None of the renderers below emits a vision marker, so a multimodal
    /// message would render as its text alone -- and then the id sequence
    /// carries no `<|image_pad|>` for the splice to expand, `PromptVision`
    /// receives spans that do not exist, and the model answers about a
    /// picture it was never shown. Every real vision install ships its own
    /// template and takes the Jinja path, so this refusal is what a
    /// MALFORMED install gets, exactly as the per-dialect refusals above are.
    pub fn apply_dialect_chat_template(
        &self,
        messages: &[Message],
    ) -> Result<String, TokenizerError> {
        if let Some(index) = messages.iter().position(|m| m.image_count() > 0) {
            return Err(TokenizerError::UnsupportedForDialect(format!(
                "message {index} carries an image, but this checkpoint ships no chat template \
                 and the {:?} fallback renderer has no vision markers to emit",
                self.dialect
            )));
        }
        match self.dialect {
            ChatDialect::Gemma => gemma::gemma_chat_template(messages),
            ChatDialect::ChatMl => chatml::chatml_chat_template(messages),
            ChatDialect::Deepseek => deepseek::deepseek_chat_template(messages),
            ChatDialect::Mistral => mistral::mistral_chat_template(messages),
            // NO FALLBACK RENDERER FOR `muse_glimmer` EITHER, and for Harmony's
            // reason one model over: its template carries an image/video
            // content macro and an `<atem:function_calls>` tool DSL, so a
            // hand-rolled version would be framing the model was not trained
            // on -- fluent output that is not an answer, with no error
            // anywhere (AGENTS.md Gotcha 41). A real install always ships the
            // template, so this refusal is what a MALFORMED install gets.
            ChatDialect::MuseGlimmer => Err(TokenizerError::UnsupportedForDialect(
                "muse_glimmer has no fallback renderer; the install must carry its own \
                 chat_template.jinja (or tokenizer_config.json's chat_template key)"
                    .to_string(),
            )),
            // NO FALLBACK RENDERER FOR HARMONY, ON PURPOSE (ROADMAP M5).
            //
            // Every other arm here is a handful of markers around the content.
            // Harmony is a 17 KB template with a system preamble, a knowledge
            // cutoff, a reasoning-effort knob and a tool namespace written in
            // TypeScript syntax, and a partial re-implementation of it is
            // exactly AGENTS.md Gotcha 41's failure: framing the model was not
            // trained on, which comes back as fluent output that is not an
            // answer, with no error anywhere.
            //
            // Refusing is safe because it is unreachable for a real install --
            // `apply_chat_template` prefers the checkpoint's own template and
            // gpt-oss always ships one. This arm is what a MALFORMED install
            // gets, and saying so beats inventing a prompt.
            ChatDialect::Harmony => Err(TokenizerError::UnsupportedForDialect(
                "the Harmony format has no fallback renderer; a gpt-oss install must carry its                  own chat_template.jinja (or tokenizer_config.json's chat_template key)"
                    .to_string(),
            )),
            ChatDialect::Llama3 => llama3::llama3_chat_template(messages),
            // NO FALLBACK RENDERER FOR SPARK, on the Harmony/Muse doctrine:
            // its template forces the think frame OPEN at the generation
            // point (`<think>` or `</think>` per `enable_thinking`), frames a
            // default system prompt into the opening block, and carries a
            // `<tool_call>`/`<arg_key>` DSL -- all decisions a hand-rolled
            // renderer would have to guess, and guessing one wrong is fluent
            // output that is not an answer. A real install always ships the
            // template; this refusal is what a MALFORMED install gets.
            ChatDialect::MiniMax => Err(TokenizerError::UnsupportedForDialect("MiniMax requires its checkpoint chat template".into())),
            ChatDialect::Spark => Err(TokenizerError::UnsupportedForDialect(
                "spark2_5 has no fallback renderer; the install must carry its own \
                 chat_template.jinja (or tokenizer_config.json's chat_template key)"
                    .to_string(),
            )),
        }
    }

    /// Formats user text continuation and returns encoded token IDs.
    pub fn encode_text_continuation(&self, user_content: &str) -> Vec<i32> {
        let content = user_content.trim();
        let suffix = match self.dialect {
            ChatDialect::Gemma => gemma::gemma_continuation_suffix(content),
            ChatDialect::ChatMl => chatml::chatml_continuation_suffix(content),
            ChatDialect::Deepseek => deepseek::deepseek_continuation_suffix(user_content),
            // No leading newline and no assistant marker: this dialect's
            // generation point is simply the character after `[/INST]`.
            ChatDialect::Mistral => mistral::mistral_continuation_suffix(content),
            // Harmony's TURN FRAME is writable even though its full template
            // is not: a continuation is one user turn and the opening of an
            // assistant one, with no system preamble involved. The channel is
            // left for the model to choose, which is what the checkpoint's own
            // generation prompt does.
            ChatDialect::Harmony => format!(
                "{HARMONY_START_MARK}user{HARMONY_MESSAGE_MARK}{content}{HARMONY_END_MARK}{HARMONY_START_MARK}assistant"
            ),
            // Writable for Harmony's reason: a continuation is one user turn
            // plus the opening of an assistant one, with no system preamble
            // and no channel involved. This family closes a turn with
            // `<|eot|>` where Harmony uses `<|end|>`.
            ChatDialect::MuseGlimmer => format!(
                "{MUSE_START_MARK}user{MUSE_MESSAGE_MARK}{content}{MUSE_EOT_MARK}\
                 {MUSE_START_MARK}assistant{MUSE_MESSAGE_MARK}"
            ),
            ChatDialect::Llama3 => llama3::llama3_continuation_suffix(content),
            // Writable for Harmony's reason: a continuation is one user turn
            // plus the opening of an assistant one, no system preamble and no
            // tools involved. The think frame is opened here to match the
            // checkpoint's own generation prompt at its default
            // (`enable_thinking` true): the template ALWAYS writes one of the
            // two tags at this point, and the bare `<|Bot|>` the model was
            // never trained to continue is the one frame that reads as
            // malformed. `prompt_opens_thought` keys on the rendered tag, so
            // the structured decoder lands in Thought to match.
            ChatDialect::MiniMax => format!("]~b]user\n{content}[e~[\n]~b]ai\n<think>\n"),
            ChatDialect::Spark => format!(
                "{SPARK_BOS_MARK}{SPARK_USER_MARK}{content}{SPARK_EOS_MARK}\
                 {SPARK_BOS_MARK}{SPARK_BOT_MARK}<think>"
            ),
        };
        let mut out = vec![self.end_of_turn_id];
        out.extend(self.encode(&suffix, false));
        out
    }

    /// Confirms `vision_start` and `image_pad` (a `VisionConfig`'s
    /// `vision_start_token_id` / `image_token_id`, as read off a trunk or an
    /// attached sidecar) are USABLE markers for THIS tokenizer, before the
    /// first image is ever processed (vision memory sidecar, Part A4).
    ///
    /// Both ids are meaningless unless they resolve to real tokens in this
    /// tokenizer's vocabulary and the checkpoint's own chat template actually
    /// places exactly one of each around a single image. Without this check
    /// both failures surface instead inside
    /// `turbospark_vision_io::splice_and_walk`, at the FIRST image a caller
    /// attaches, as a bare `PlaceholderMismatch` far from the id pairing that
    /// caused it -- surfacing the same question here, at attach time, is
    /// strictly earlier and names the actual ids.
    ///
    /// Two checks, in order:
    /// - `vision_start` and `image_pad` are DISTINCT and each resolves to a
    ///   real token (`id_to_token` is `Some`), the same reverse-lookup
    ///   `dialect::resolve::special_token_id` already uses to confirm a
    ///   resolved id names what it claims to;
    /// - a minimal `[Image, Text]` user turn, rendered through
    ///   [`Self::apply_chat_template`] and re-encoded, contains EXACTLY ONE
    ///   `vision_start` and EXACTLY ONE `image_pad`. That is the multiplicity
    ///   `turbospark_vision_io::splice_and_walk` assumes for a single image:
    ///   it counts `vision_start`-followed-by-`image_pad` PAIRS to size its
    ///   walk and only expands the placeholder to the image's own
    ///   merged-token count afterwards, so the RENDERED prompt (before that
    ///   expansion) must carry exactly one of each per image.
    ///
    /// A checkpoint with no installed template (every synthetic fixture,
    /// plus a malformed real install) fails here rather than at the first
    /// image, through the same [`TokenizerError::InvalidChatTemplate`]
    /// [`Self::apply_chat_template`] itself would raise.
    pub fn verify_image_markers(
        &self,
        vision_start: i32,
        image_pad: i32,
    ) -> Result<(), TokenizerError> {
        if vision_start == image_pad {
            return Err(TokenizerError::InvalidVisionMarkers(format!(
                "vision_start and image_pad are both token id {vision_start}; a vision config \
                 must name two distinct markers"
            )));
        }
        for (name, id) in [("vision_start", vision_start), ("image_pad", image_pad)] {
            if self.id_to_token(id).is_none() {
                return Err(TokenizerError::InvalidVisionMarkers(format!(
                    "{name} token id {id} does not resolve to a token in this tokenizer's \
                     vocabulary"
                )));
            }
        }

        let probe = [Message::with_parts(
            Role::User,
            vec![
                ContentPart::Image,
                ContentPart::Text("describe this image".to_string()),
            ],
        )];
        let rendered = self.apply_chat_template(&probe).map_err(|e| {
            TokenizerError::InvalidVisionMarkers(format!(
                "could not render a minimal one-image turn to verify vision markers: {e}"
            ))
        })?;
        let ids = self.encode(&rendered, true);
        let starts = ids.iter().filter(|&&id| id == vision_start).count();
        let pads = ids.iter().filter(|&&id| id == image_pad).count();
        if starts != 1 || pads != 1 {
            return Err(TokenizerError::InvalidVisionMarkers(format!(
                "this checkpoint's chat template renders {starts} occurrence(s) of vision_start \
                 ({vision_start}) and {pads} of image_pad ({image_pad}) for one image; \
                 splice_and_walk needs exactly one of each"
            )));
        }
        Ok(())
    }

    /// Full DeepSeek-V4 tool-chat render: tool schemas join the system
    /// message as a `## Tools` section, `tool` results merge into
    /// `<User>` turns as `<tool_result>` blocks, and historical tool calls
    /// render as DSML `<DSML|tool_calls>` blocks.
    pub fn encode_deepseek_tool_chat(
        &self,
        messages: &[Message],
        tools: &[FunctionDefinition],
    ) -> Result<Vec<i32>, TokenizerError> {
        let s = deepseek::render_deepseek_tool_chat(messages, tools)?;
        Ok(self.encode(&s, false))
    }
}
