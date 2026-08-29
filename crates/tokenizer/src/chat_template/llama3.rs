//! Meta's Llama-3 family chat template fallback renderer and tests.

use super::{Message, Role};
use crate::error::TokenizerError;

/// Meta's Llama-3 (base and Instruct) header framing.
///
/// The reference template (`meta-llama/Meta-Llama-3-8B-Instruct`'s
/// `tokenizer_config.json`) is `{{- bos_token }}` followed by one
/// `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` block per
/// message and, when a generation prompt is asked for,
/// `<|start_header_id|>assistant<|end_header_id|>\n\n`. Unlike Mistral's
/// fallback (`mistral.rs`, which has a documented, deliberately-inert gap
/// here) this one DOES emit the literal `<|begin_of_text|>` itself, matching
/// `resolve_llama3`'s `bos_prefix_id: None` -- the encoder must not prepend a
/// second one.
///
/// FALLBACK ONLY: a real Llama-3 install always ships a template (it has
/// since the family's release, because `apply_chat_template` needs one
/// upstream too), so this fires solely for a hypothetical template-less one.
/// Every role maps to its own header; there is no folding rule to get wrong
/// the way Mistral's system-into-first-user-turn one is, since headers frame
/// every role including system.
pub(super) fn llama3_chat_template(messages: &[Message]) -> Result<String, TokenizerError> {
    let mut out = String::from("<|begin_of_text|>");
    for message in messages {
        let Some(raw) = &message.content else {
            return Err(TokenizerError::InvalidChatTemplate(
                "text-only messages require content".to_string(),
            ));
        };
        if message.role == Role::Tool {
            return Err(TokenizerError::InvalidChatTemplate(
                "the Llama-3 fallback renderer has no tool-calling markup".to_string(),
            ));
        }
        let content = raw.trim();
        out.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{content}<|eot_id|>",
            message.role.as_str()
        ));
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    Ok(out)
}

pub(super) fn llama3_continuation_suffix(content: &str) -> String {
    format!(
        "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

#[cfg(test)]
mod llama3_template_tests {
    use super::*;

    fn msg(role: Role, content: &str) -> Message {
        Message {
            role,
            content: Some(content.to_string()),
            content_parts: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Tested here rather than through `MfTokenizer` because rendering the
    /// fallback needs no tokenizer at all -- the render is pure text and the
    /// dialect's whole content, same reasoning as Mistral's own test module.
    #[test]
    fn a_user_turn_renders_between_bos_and_the_assistant_header() {
        let out = llama3_chat_template(&[msg(Role::User, "hello")]).unwrap();
        assert_eq!(
            out,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
             hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    /// Unlike Mistral, a system message gets its OWN header rather than
    /// folding into the next user turn -- the reference template has no
    /// special case for it.
    #[test]
    fn a_system_message_gets_its_own_header() {
        let out =
            llama3_chat_template(&[msg(Role::System, "be terse"), msg(Role::User, "hi")]).unwrap();
        assert!(out.contains("<|start_header_id|>system<|end_header_id|>\n\nbe terse<|eot_id|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"));
    }

    #[test]
    fn every_turn_closes_with_eot_id_not_end_of_text() {
        let out = llama3_chat_template(&[msg(Role::User, "a"), msg(Role::Assistant, "b")]).unwrap();
        assert_eq!(out.matches("<|eot_id|>").count(), 2);
        assert!(!out.contains("<|end_of_text|>"));
    }

    #[test]
    fn a_tool_message_is_refused_rather_than_invented() {
        assert!(llama3_chat_template(&[msg(Role::Tool, "{}")]).is_err());
    }
}
