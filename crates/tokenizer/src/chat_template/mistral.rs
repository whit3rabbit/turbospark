//! Mistral / Mixtral chat template fallback renderer and tests.

use super::{Message, Role};
use crate::error::TokenizerError;

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
pub(super) fn mistral_chat_template(messages: &[Message]) -> Result<String, TokenizerError> {
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

pub(super) fn mistral_continuation_suffix(content: &str) -> String {
    format!(" [INST] {content} [/INST]")
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
