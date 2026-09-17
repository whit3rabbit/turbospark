//! Tokenizer and generation configuration loading and embedded template resolution.

/// A special-token reference in `tokenizer_config.json`: either a bare
/// string or HF's `{"__type": "AddedToken", "content": ...}` object. The
/// V2-era DeepSeek tables spell every one of these as the object, and a
/// strict-string field fails the WHOLE config parse -- which silently
/// stripped the embedded chat template too, because `chat_template` lives
/// in the same object.
#[derive(Default, serde::Deserialize)]
#[serde(untagged)]
pub(crate) enum TokenRef {
    Added {
        content: String,
    },
    /// The older convention: the bare string. Newer configs wrap it.
    Plain(String),
    #[default]
    Missing,
}

impl TokenRef {
    pub(crate) fn content(&self) -> Option<&str> {
        match self {
            TokenRef::Added { content } => Some(content),
            TokenRef::Plain(content) => Some(content),
            TokenRef::Missing => None,
        }
    }
}

#[derive(Default, serde::Deserialize)]
pub(crate) struct TokenizerConfig {
    #[serde(default)]
    pub(crate) bos_token: TokenRef,
    #[serde(default)]
    pub(crate) eos_token: TokenRef,
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

pub(crate) enum EosTokenIds {
    One(i64),
    Many(Vec<i64>),
}

pub(crate) const MAX_EOS_TOKEN_IDS: usize = 256;

impl<'de> serde::Deserialize<'de> for EosTokenIds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct EosTokenIdsVisitor;

        impl<'de> serde::de::Visitor<'de> for EosTokenIdsVisitor {
            type Value = EosTokenIds;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    formatter,
                    "an integer or at most {MAX_EOS_TOKEN_IDS} integers"
                )
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(EosTokenIds::One(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i64::try_from(value)
                    .map(EosTokenIds::One)
                    .map_err(|_| E::custom("EOS token id exceeds i64"))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut ids =
                    Vec::with_capacity(seq.size_hint().unwrap_or(0).min(MAX_EOS_TOKEN_IDS));
                while let Some(id) = seq.next_element()? {
                    if ids.len() == MAX_EOS_TOKEN_IDS {
                        return Err(serde::de::Error::custom("too many EOS token ids"));
                    }
                    ids.push(id);
                }
                Ok(EosTokenIds::Many(ids))
            }
        }

        deserializer.deserialize_any(EosTokenIdsVisitor)
    }
}

impl GenerationConfig {
    pub(crate) fn eos_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(EosTokenIds::One(id)) => i32::try_from(*id).into_iter().collect(),
            Some(EosTokenIds::Many(ids)) => ids
                .iter()
                .filter_map(|&id| i32::try_from(id).ok())
                .collect(),
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
