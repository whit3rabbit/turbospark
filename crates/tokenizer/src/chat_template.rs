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

use crate::dialect::{
    ChatDialect, MfTokenizer, DEEPSEEK_BOS_MARK, DEEPSEEK_EOS_MARK, HARMONY_END_MARK,
    HARMONY_MESSAGE_MARK, HARMONY_START_MARK,
};
use crate::error::TokenizerError;
use crate::json_value::JsonValue;
use crate::tool_call::DeepseekToolCallParser;

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

const TURN_OPEN: &str = "<|turn>";
const TURN_CLOSE: &str = "<turn|>";
const BOS_MARK: &str = "<bos>";
const IM_START_MARK: &str = "<|im_start|>";
const IM_END_MARK: &str = "<|im_end|>";
const CHATML_GENERATION_SUFFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
const DEEPSEEK_GENERATION_SUFFIX: &str = "<\u{FF5C}Assistant\u{FF5C}></think>";
const DEEPSEEK_THINK_CLOSE_MARK: &str = "</think>";

/// Mistral / Mixtral instruction format (ROADMAP Phase M2).
///
/// PLAIN TEXT framing, which is what makes this dialect different in kind
/// from the other three rather than in detail: `[INST]` and `[/INST]` are
/// ordinary token sequences, not special tokens, so nothing here can be
/// built out of ids.
///
/// Two rules the reference template enforces and a naive render gets wrong:
/// a SYSTEM message has no turn of its own and is folded into the first
/// user turn, and `</s>` closes each ASSISTANT turn only -- a user turn is
/// closed by `[/INST]`, not by the sentence end.
///
/// FALLBACK ONLY since the installed template took over: every real
/// `<s>`/`</s>` checkpoint on the shelf ships one, so this render now fires
/// solely for a hypothetical template-less one. Which is just as well,
/// because it emits no `<s>` -- it was written expecting `bos_prefix_id` to
/// prepend it, and every call site renders with `add_bos = false` (the
/// convention the other three dialects need, since their templates emit
/// their own BOS as text). Mistral's real template emits `<s>` itself, so
/// the live path is correct and this gap is inert. Fix it here before
/// relying on this function for a real checkpoint.
fn mistral_chat_template(messages: &[Message]) -> Result<String, TokenizerError> {
    let mut out = String::new();
    let mut pending_system: Option<String> = None;
    for message in messages {
        let Some(raw) = &message.content else {
            return Err(TokenizerError::InvalidChatTemplate(
                "text-only messages require content".to_string(),
            ));
        };
        let content = raw.trim();
        match message.role {
            Role::System | Role::Developer => {
                pending_system = Some(match pending_system.take() {
                    Some(prior) => format!("{prior}\n\n{content}"),
                    None => content.to_string(),
                });
            }
            Role::User => {
                let body = match pending_system.take() {
                    Some(system) => format!("{system}\n\n{content}"),
                    None => content.to_string(),
                };
                out.push_str(&format!(" [INST] {body} [/INST]"));
            }
            Role::Assistant => out.push_str(&format!(" {content}</s>")),
            Role::Tool => {
                return Err(TokenizerError::InvalidChatTemplate(
                    "Mixtral 8x7B-Instruct v0.1 has no tool-calling markup".to_string(),
                ));
            }
        }
    }
    // A trailing system message with no user turn after it would otherwise
    // vanish; fold it into an empty instruction rather than dropping it.
    if let Some(system) = pending_system {
        out.push_str(&format!(" [INST] {system} [/INST]"));
    }
    Ok(out)
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
            ChatDialect::Gemma => gemma_chat_template(messages),
            ChatDialect::ChatMl => chatml_chat_template(messages),
            ChatDialect::Deepseek => deepseek_chat_template(messages),
            ChatDialect::Mistral => mistral_chat_template(messages),
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
                "the Harmony format has no fallback renderer; a gpt-oss install must carry its \
                 own chat_template.jinja (or tokenizer_config.json's chat_template key)"
                    .to_string(),
            )),
        }
    }

    /// Formats user text continuation and returns encoded token IDs.
    pub fn encode_text_continuation(&self, user_content: &str) -> Vec<i32> {
        let content = user_content.trim();
        let suffix = match self.dialect {
            ChatDialect::Gemma => format!(
                "\n{TURN_OPEN}user\n{content}{TURN_CLOSE}\n{TURN_OPEN}model\n<|channel>thought\n<channel|>"
            ),
            ChatDialect::ChatMl => {
                format!("\n{IM_START_MARK}user\n{content}{IM_END_MARK}\n{CHATML_GENERATION_SUFFIX}")
            }
            ChatDialect::Deepseek => {
                format!("{DEEPSEEK_USER_MARK}{user_content}{DEEPSEEK_GENERATION_SUFFIX}")
            }
            // No leading newline and no assistant marker: this dialect's
            // generation point is simply the character after `[/INST]`.
            ChatDialect::Mistral => format!(" [INST] {content} [/INST]"),
            // Harmony's TURN FRAME is writable even though its full template
            // is not: a continuation is one user turn and the opening of an
            // assistant one, with no system preamble involved. The channel is
            // left for the model to choose, which is what the checkpoint's own
            // generation prompt does.
            ChatDialect::Harmony => format!(
                "{HARMONY_START_MARK}user{HARMONY_MESSAGE_MARK}{content}{HARMONY_END_MARK}\
                 {HARMONY_START_MARK}assistant"
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
    // The Developer arm always sets `last_turn_was_user` right after
    // `flush_user_turn!`, which may already have set it; harmless double
    // assignment, not worth restructuring the shared macro over.
    #[allow(unused_assignments)]
    pub fn encode_deepseek_tool_chat(
        &self,
        messages: &[Message],
        tools: &[FunctionDefinition],
    ) -> Result<Vec<i32>, TokenizerError> {
        let mut s = DEEPSEEK_BOS_MARK.to_string();
        let mut remaining = messages;
        let mut system_text: Option<String> = None;
        if let Some(first) = remaining.first() {
            if first.role == Role::System {
                system_text = Some(first.content.clone().unwrap_or_default());
                remaining = &remaining[1..];
            }
        }
        if !tools.is_empty() {
            let section = deepseek_tools_section(tools)?;
            system_text = Some(format!(
                "{}\n\n{}",
                system_text.unwrap_or_default(),
                section
            ));
        }
        if let Some(text) = &system_text {
            s.push_str(text);
        }

        let mut pending_user_parts: Vec<String> = Vec::new();
        let mut last_turn_was_user = false;
        macro_rules! flush_user_turn {
            () => {
                if !pending_user_parts.is_empty() {
                    s.push_str(DEEPSEEK_USER_MARK);
                    s.push_str(&pending_user_parts.join("\n\n"));
                    pending_user_parts.clear();
                    last_turn_was_user = true;
                }
            };
        }
        for message in remaining {
            match message.role {
                Role::System => {
                    return Err(TokenizerError::InvalidChatTemplate(
                        "system message must be first".to_string(),
                    ))
                }
                Role::User => {
                    pending_user_parts.push(message.content.clone().unwrap_or_default());
                }
                Role::Tool => {
                    pending_user_parts.push(format!(
                        "<tool_result>{}</tool_result>",
                        message.content.clone().unwrap_or_default()
                    ));
                }
                Role::Developer => {
                    flush_user_turn!();
                    s.push_str(DEEPSEEK_USER_MARK);
                    s.push_str(&message.content.clone().unwrap_or_default());
                    last_turn_was_user = true;
                }
                Role::Assistant => {
                    flush_user_turn!();
                    if last_turn_was_user {
                        s.push_str(DEEPSEEK_GENERATION_SUFFIX);
                    }
                    let mut turn = message.content.clone().unwrap_or_default();
                    if !message.tool_calls.is_empty() {
                        let invokes: Result<Vec<String>, TokenizerError> =
                            message.tool_calls.iter().map(deepseek_invoke).collect();
                        let invokes = invokes?.join("\n");
                        turn.push_str(&format!(
                            "\n\n{}\n{}\n{}",
                            DeepseekToolCallParser::tool_calls_open_mark(),
                            invokes,
                            DeepseekToolCallParser::tool_calls_close_mark()
                        ));
                    }
                    s.push_str(&turn);
                    s.push_str(DEEPSEEK_EOS_MARK);
                    last_turn_was_user = false;
                }
            }
        }
        flush_user_turn!();
        s.push_str(DEEPSEEK_GENERATION_SUFFIX);
        Ok(self.encode(&s, false))
    }
}

const DEEPSEEK_USER_MARK: &str = "<\u{FF5C}User\u{FF5C}>";

fn gemma_chat_template(messages: &[Message]) -> Result<String, TokenizerError> {
    let mut s = BOS_MARK.to_string();
    for (index, message) in messages.iter().enumerate() {
        let Some(raw) = &message.content else {
            return Err(TokenizerError::InvalidChatTemplate(
                "text-only messages require content".to_string(),
            ));
        };
        let content = raw.trim();
        if message.role == Role::System && index != 0 {
            return Err(TokenizerError::InvalidChatTemplate(
                "system message must be first".to_string(),
            ));
        }
        let role = if message.role == Role::Assistant {
            "model"
        } else {
            message.role.as_str()
        };
        s.push_str(TURN_OPEN);
        s.push_str(role);
        s.push('\n');
        s.push_str(content);
        s.push_str(TURN_CLOSE);
        s.push('\n');
    }
    s.push_str(TURN_OPEN);
    s.push_str("model\n<|channel>thought\n<channel|>");
    Ok(s)
}

fn chatml_chat_template(messages: &[Message]) -> Result<String, TokenizerError> {
    let mut s = String::new();
    for (index, message) in messages.iter().enumerate() {
        let Some(raw) = &message.content else {
            return Err(TokenizerError::InvalidChatTemplate(
                "text-only messages require content".to_string(),
            ));
        };
        let content = raw.trim();
        if message.role == Role::System && index != 0 {
            return Err(TokenizerError::InvalidChatTemplate(
                "system message must be first".to_string(),
            ));
        }
        s.push_str(IM_START_MARK);
        s.push_str(message.role.as_str());
        s.push('\n');
        s.push_str(content);
        s.push_str(IM_END_MARK);
        s.push('\n');
    }
    s.push_str(CHATML_GENERATION_SUFFIX);
    Ok(s)
}

/// Text-only, no-tool rendering of the DeepSeek-V4 non-thinking ("chat"
/// mode) encoding: a system message renders bare, EVERY user turn is
/// followed by the assistant transition, every assistant turn opens with
/// its own think-close, content is never trimmed, and the generation prompt
/// is appended only when the last message is not a user turn.
fn deepseek_chat_template(messages: &[Message]) -> Result<String, TokenizerError> {
    let mut s = DEEPSEEK_BOS_MARK.to_string();
    for (index, message) in messages.iter().enumerate() {
        let Some(content) = &message.content else {
            return Err(TokenizerError::InvalidChatTemplate(
                "text-only messages require content".to_string(),
            ));
        };
        if message.role == Role::System && index != 0 {
            return Err(TokenizerError::InvalidChatTemplate(
                "system message must be first".to_string(),
            ));
        }
        match message.role {
            Role::System => s.push_str(content),
            Role::User | Role::Developer => {
                s.push_str(DEEPSEEK_USER_MARK);
                s.push_str(content);
                s.push_str(DEEPSEEK_GENERATION_SUFFIX);
            }
            Role::Assistant => {
                s.push_str(DEEPSEEK_THINK_CLOSE_MARK);
                s.push_str(content);
                s.push_str(DEEPSEEK_EOS_MARK);
            }
            Role::Tool => {
                return Err(TokenizerError::InvalidChatTemplate(
                    "deepseek merges tool results into user turns; use the tool chat encoder"
                        .to_string(),
                ))
            }
        }
    }
    if let Some(last) = messages.last() {
        if !(last.role == Role::User || last.role == Role::Developer) {
            s.push_str(DEEPSEEK_GENERATION_SUFFIX);
        }
    }
    Ok(s)
}

/// One historical tool call as a DSML invoke block. String arguments pass
/// through raw with `string="true"`; everything else serializes to JSON
/// with `string="false"`. Keys render in sorted order (guaranteed by
/// `JsonValue::Object`'s `BTreeMap`) to keep the prompt deterministic.
fn deepseek_invoke(call: &HistoricalToolCall) -> Result<String, TokenizerError> {
    let arguments = call.arguments.as_object().ok_or_else(|| {
        TokenizerError::InvalidChatTemplate(
            "historical tool arguments must be a JSON object".to_string(),
        )
    })?;
    let dsml = crate::tool_call::deepseek_dsml_mark();
    let guard_framable = |text: &str, what: &str| -> Result<(), TokenizerError> {
        if text.contains(&dsml) {
            return Err(TokenizerError::InvalidChatTemplate(format!(
                "historical tool call {what} contains the DSML marker and cannot be re-rendered unambiguously"
            )));
        }
        Ok(())
    };
    guard_framable(&call.name, "name")?;
    let mut parameters = Vec::with_capacity(arguments.len());
    for (key, value) in arguments {
        guard_framable(key, "parameter name")?;
        if let JsonValue::String(raw) = value {
            guard_framable(raw, &format!("argument \"{key}\""))?;
            parameters.push(format!(
                "<{dsml}parameter name=\"{key}\" string=\"true\">{raw}</{dsml}parameter>"
            ));
        } else {
            let encoded = value.encoded();
            guard_framable(&encoded, &format!("argument \"{key}\""))?;
            parameters.push(format!(
                "<{dsml}parameter name=\"{key}\" string=\"false\">{encoded}</{dsml}parameter>"
            ));
        }
    }
    Ok(format!(
        "<{dsml}invoke name=\"{}\">\n{}\n</{dsml}invoke>",
        call.name,
        parameters.join("\n")
    ))
}

/// The `## Tools` system-prompt section, carrying the DSML invoke syntax and
/// the JSON tool schemas.
fn deepseek_tools_section(tools: &[FunctionDefinition]) -> Result<String, TokenizerError> {
    let dsml = crate::tool_call::deepseek_dsml_mark();
    let schemas: Vec<String> = tools
        .iter()
        .map(|tool| {
            let name = JsonValue::String(tool.name.clone()).encoded();
            let description = JsonValue::String(tool.description.clone()).encoded();
            let parameters = tool.parameters.encoded();
            format!("{{\"name\":{name},\"description\":{description},\"parameters\":{parameters}}}")
        })
        .collect();
    let schemas = schemas.join("\n");
    Ok(format!(
        "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{dsml}tool_calls>\" block like the following:\n\n<{dsml}tool_calls>\n<{dsml}invoke name=\"$TOOL_NAME\">\n<{dsml}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{dsml}parameter>\n...\n</{dsml}invoke>\n<{dsml}invoke name=\"$TOOL_NAME2\">\n...\n</{dsml}invoke>\n</{dsml}tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{schemas}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
    ))
}

#[cfg(test)]
mod mistral_template_tests {
    use super::*;

    fn msg(role: Role, content: &str) -> Message {
        Message {
            role,
            content: Some(content.to_string()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Tested here rather than through `MfTokenizer` because there is no
    /// Mistral tokenizer fixture in this repo: the render is pure text and
    /// the dialect's whole content, so it is worth pinning on its own.
    #[test]
    fn a_user_turn_renders_as_an_instruction_block() {
        let out = mistral_chat_template(&[msg(Role::User, "hello")]).unwrap();
        assert_eq!(out, " [INST] hello [/INST]");
    }

    /// THE RULE A NAIVE RENDER GETS WRONG: a system message has no turn of
    /// its own; it is folded into the first user turn. Emitting it as its own
    /// `[INST]` block would leave two instructions in a row, which the model
    /// was never trained on.
    #[test]
    fn a_system_message_folds_into_the_first_user_turn() {
        let out = mistral_chat_template(&[
            msg(Role::System, "be terse"),
            msg(Role::User, "hello"),
            msg(Role::Assistant, "hi"),
            msg(Role::User, "again"),
        ])
        .unwrap();
        assert_eq!(
            out,
            " [INST] be terse\n\nhello [/INST] hi</s> [INST] again [/INST]"
        );
    }

    /// `</s>` closes an ASSISTANT turn only. A user turn is closed by
    /// `[/INST]`, and putting a sentence end there instead is the other half
    /// of the same mistake.
    #[test]
    fn only_assistant_turns_are_closed_with_the_sentence_end() {
        let out =
            mistral_chat_template(&[msg(Role::User, "a"), msg(Role::Assistant, "b")]).unwrap();
        assert_eq!(out.matches("</s>").count(), 1);
        assert!(out.ends_with("b</s>"));
    }

    #[test]
    fn a_tool_message_is_refused_rather_than_invented() {
        assert!(mistral_chat_template(&[msg(Role::Tool, "{}")]).is_err());
    }
}
