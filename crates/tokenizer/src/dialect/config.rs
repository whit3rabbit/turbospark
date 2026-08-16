//! Tokenizer and generation configuration loading and embedded template resolution.

#[derive(Default, serde::Deserialize)]
pub(crate) struct TokenizerConfig {
    pub(crate) bos_token: Option<String>,
    pub(crate) eos_token: Option<String>,
    /// The pre-`chat_template.jinja` convention. HF allowed either one
    /// template string or a NAMED LIST of them (the `default` /
    /// `tool_use` split some checkpoints ship), so both shapes parse.
    #[serde(default)]
    pub(crate) chat_template: Option<EmbeddedChatTemplate>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddedChatTemplate {
    One(String),
    Named(Vec<NamedChatTemplate>),
}

#[derive(serde::Deserialize)]
pub(crate) struct NamedChatTemplate {
    pub(crate) name: String,
    pub(crate) template: String,
}

impl TokenizerConfig {
    /// The embedded template's source, if any. From a named list this takes
    /// the entry called `default`, falling back to the first: the other
    /// names are tool-use variants, and the tool path here renders through
    /// [`crate::jinja_chat_template`] with `tools` in the context rather
    /// than by selecting a different template.
    pub(crate) fn chat_template_source(&self) -> Option<String> {
        match self.chat_template.as_ref()? {
            EmbeddedChatTemplate::One(source) => Some(source.clone()),
            EmbeddedChatTemplate::Named(entries) => entries
                .iter()
                .find(|entry| entry.name == "default")
                .or_else(|| entries.first())
                .map(|entry| entry.template.clone()),
        }
    }
}

/// The slice of `generation_config.json` this loader reads. HF writes
/// `eos_token_id` as either one integer or an array of them.
#[derive(Default, serde::Deserialize)]
pub(crate) struct GenerationConfig {
    #[serde(default)]
    pub(crate) eos_token_id: Option<EosTokenIds>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
pub(crate) enum EosTokenIds {
    One(i64),
    Many(Vec<i64>),
}

impl GenerationConfig {
    pub(crate) fn eos_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(EosTokenIds::One(id)) => vec![*id as i32],
            Some(EosTokenIds::Many(ids)) => ids.iter().map(|&id| id as i32).collect(),
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod embedded_template_tests {
    use super::*;

    fn parse(json: &str) -> TokenizerConfig {
        serde_json::from_str(json).expect("config parses")
    }

    #[test]
    fn a_plain_string_template_is_read() {
        let config = parse(r#"{"chat_template": "<|user|>\n{{ x }}"}"#);
        assert_eq!(
            config.chat_template_source().as_deref(),
            Some("<|user|>\n{{ x }}")
        );
    }

    /// The other shape the pre-`chat_template.jinja` convention allowed. A
    /// loader that handles only the string form does not FAIL on this one,
    /// it silently reports no template and falls back to the dialect
    /// renderer, which is the failure mode this whole change exists to
    /// remove.
    #[test]
    fn a_named_list_resolves_to_the_default_entry() {
        let config = parse(
            r#"{"chat_template": [
                {"name": "tool_use", "template": "TOOLS"},
                {"name": "default", "template": "PLAIN"}
            ]}"#,
        );
        assert_eq!(config.chat_template_source().as_deref(), Some("PLAIN"));
    }

    #[test]
    fn a_named_list_without_a_default_takes_the_first_entry() {
        let config = parse(r#"{"chat_template": [{"name": "rag", "template": "R"}]}"#);
        assert_eq!(config.chat_template_source().as_deref(), Some("R"));
    }

    #[test]
    fn a_config_without_the_key_reports_no_template() {
        assert!(parse(r#"{"eos_token": "</s>"}"#)
            .chat_template_source()
            .is_none());
    }
}
