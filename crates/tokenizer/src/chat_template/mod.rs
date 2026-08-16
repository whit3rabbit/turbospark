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
mod mistral;

use crate::dialect::{
    ChatDialect, MfTokenizer, HARMONY_END_MARK, HARMONY_MESSAGE_MARK, HARMONY_START_MARK,
};
use crate::error::TokenizerError;
use crate::json_value::JsonValue;

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

/// Single chat message in a conversation sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// Sender role.
    pub role: Role,
    /// Text content string if present.
    pub content: Option<String>,
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
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
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
        if self.chat_template_source.is_some() {
            return crate::jinja_chat_template::render_generic_chat_template(
                self,
                messages,
                &[],
                true,
                false,
            );
        }
        self.apply_dialect_chat_template(messages)
    }

    /// The hand-written per-dialect render, bypassing any installed
    /// template. Public so the guard test can compare the two.
    pub fn apply_dialect_chat_template(
        &self,
        messages: &[Message],
    ) -> Result<String, TokenizerError> {
        match self.dialect {
            ChatDialect::Gemma => gemma::gemma_chat_template(messages),
            ChatDialect::ChatMl => chatml::chatml_chat_template(messages),
            ChatDialect::Deepseek => deepseek::deepseek_chat_template(messages),
            ChatDialect::Mistral => mistral::mistral_chat_template(messages),
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
                "{HARMONY_START_MARK}user{HARMONY_MESSAGE_MARK}{content}{HARMONY_END_MARK}                 {HARMONY_START_MARK}assistant"
            ),
        };
        let mut out = vec![self.end_of_turn_id];
        out.extend(self.encode(&suffix, false));
        out
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
