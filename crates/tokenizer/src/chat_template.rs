//! Text-only chat-template rendering for the three dialects, plus DeepSeek's
//! hand-rolled (non-Jinja) native tool chat. Ported from the chat-template
//! section of `Tokenization/Tokenizer.swift`.
//!
//! The generic Jinja-templated tool chat that Gemma/ChatML route through the
//! checkpoint's `chat_template.jinja` is out of scope for this port; it needs
//! a full Jinja engine and a real installed tokenizer sidecar to test
//! against. Only DeepSeek's tool framing is plain string composition and is
//! ported in full.

use crate::dialect::{ChatDialect, MfTokenizer, DEEPSEEK_BOS_MARK, DEEPSEEK_EOS_MARK};
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

impl MfTokenizer {
    /// Formats a sequence of messages into a chat template string for the tokenizer's dialect.
    pub fn apply_chat_template(&self, messages: &[Message]) -> Result<String, TokenizerError> {
        match self.dialect {
            ChatDialect::Gemma => gemma_chat_template(messages),
            ChatDialect::ChatMl => chatml_chat_template(messages),
            ChatDialect::Deepseek => deepseek_chat_template(messages),
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
