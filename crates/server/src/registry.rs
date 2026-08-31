//! Which backend serves a request, when more than one model is attached.
//!
//! Before this module there was exactly one: [`crate::handler::AppState`] is
//! `Arc<dyn ChatModel>` and [`crate::ChatModel::model_id`]'s own doc says a
//! request's `model` field is echoed back rather than routed on. That is
//! still true of a server with one model, and deliberately so -- see
//! [`ModelRegistry::resolve`].
//!
//! **THE SINGLE-MODEL FALLBACK IS LOAD-BEARING, NOT A COURTESY.** The
//! documented way to point Claude Code at this server (`docs/CLI.md`, and
//! `crates/server/CLAUDE.md` Gotcha 8's worked example) sends
//! `"model": "claude-sonnet-4-6"` at an install that is not named that, and
//! OpenAI SDKs default to a model name of their own. Every one of those
//! callers worked because the name was ignored. Routing strictly on the name
//! would 404 all of them on the first day multi-model shipped, so a request
//! that names nothing this server knows is served by the only model when
//! there IS only one, and refused by name only when the answer is genuinely
//! ambiguous.

use std::sync::Arc;

use crate::model::ChatModel;

/// One row of `GET /v1/models`, and what the FFI reports back to a GUI.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRow {
    /// The id a client puts in a request's `model` field.
    pub id: String,
    /// The resolved context window, which is a property of how the model was
    /// OPENED rather than of the checkpoint (AGENTS.md Gotcha 55), so it is
    /// reported per row rather than assumed.
    pub max_context: u32,
}

/// What [`ModelRegistry::resolve`] decided.
pub enum Resolution {
    /// Serve this backend.
    Model(Arc<dyn ChatModel>),
    /// Several models are attached and the request named none of them.
    /// Carries what IS available, because a 404 that does not say what the
    /// caller should have asked for makes them guess.
    Unknown {
        requested: String,
        available: Vec<String>,
    },
    /// The server is running with nothing attached.
    Empty,
}

/// How a router picks a backend for a request.
pub trait ModelRegistry: Send + Sync {
    /// `requested` is the request's own `model` field, absent on the few
    /// shapes that do not carry one.
    fn resolve(&self, requested: Option<&str>) -> Resolution;
    /// Every attached model, in a stable order.
    fn rows(&self) -> Vec<ModelRow>;
}

/// The one-model registry every pre-registry caller gets, including all of
/// this crate's integration tests: `build_router(model)` builds one of these
/// through [`crate::ServerState`]'s `From` impl, and its `resolve` returns
/// that model for every request whatever the name.
pub struct SingleModel(Arc<dyn ChatModel>);

impl SingleModel {
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self(model)
    }
}

impl ModelRegistry for SingleModel {
    fn resolve(&self, _requested: Option<&str>) -> Resolution {
        Resolution::Model(Arc::clone(&self.0))
    }

    fn rows(&self) -> Vec<ModelRow> {
        vec![ModelRow {
            id: self.0.model_id().to_string(),
            max_context: self.0.max_context(),
        }]
    }
}

/// A fixed set of models, decided when the router is built.
///
/// Enough for a host that knows its models up front and for this crate's own
/// tests. A host that attaches and detaches on a RUNNING server (the FFI)
/// implements [`ModelRegistry`] over its own locked storage instead -- which
/// is the reason `resolve` takes `&self` and the policy lives in
/// [`resolve_among`] rather than in this type.
pub struct StaticRegistry(Vec<Arc<dyn ChatModel>>);

impl StaticRegistry {
    pub fn new(models: Vec<Arc<dyn ChatModel>>) -> Self {
        Self(models)
    }
}

impl ModelRegistry for StaticRegistry {
    fn resolve(&self, requested: Option<&str>) -> Resolution {
        resolve_among(&self.0, requested)
    }

    fn rows(&self) -> Vec<ModelRow> {
        self.0
            .iter()
            .map(|m| ModelRow {
                id: m.model_id().to_string(),
                max_context: m.max_context(),
            })
            .collect()
    }
}

/// Resolves `requested` against a slice of attached models under the policy
/// this module's header describes. Shared so a registry with interior
/// mutability (the FFI's `RwLock<HashMap<..>>` one) states the rule once
/// rather than restating it against its own storage.
pub fn resolve_among(models: &[Arc<dyn ChatModel>], requested: Option<&str>) -> Resolution {
    if let Some(name) = requested {
        if let Some(hit) = models.iter().find(|m| m.model_id() == name) {
            return Resolution::Model(Arc::clone(hit));
        }
    }
    match models {
        [] => Resolution::Empty,
        // The fallback. One model attached means there is no ambiguity to
        // resolve, so an unrecognized name is the caller's label for the
        // conversation rather than a routing instruction.
        [only] => Resolution::Model(Arc::clone(only)),
        several => Resolution::Unknown {
            requested: requested.unwrap_or("").to_string(),
            available: several.iter().map(|m| m.model_id().to_string()).collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScriptedChatModel;

    /// `ScriptedChatModel::model_id` is the constant `"scripted"`, so a
    /// per-id backend needs a wrapper. Naming it after the install
    /// directory mirrors what `RealChatModel::open` and the FFI both do.
    struct Named(ScriptedChatModel, String);

    impl ChatModel for Named {
        fn tokenizer(&self) -> &tokenizer::MfTokenizer {
            self.0.tokenizer()
        }
        fn vocab_size(&self) -> usize {
            self.0.vocab_size()
        }
        fn max_context(&self) -> u32 {
            self.0.max_context()
        }
        fn model_id(&self) -> &str {
            &self.1
        }
        fn with_producer(
            &self,
            f: &mut dyn FnMut(
                &mut dyn runtime::LogitProducer,
            )
                -> Result<runtime::RawDecodeResult, runtime::RuntimeError>,
        ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError> {
            self.0.with_producer(f)
        }
    }

    fn model(id: &str) -> Arc<dyn ChatModel> {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
        let tok = tokenizer::MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer");
        Arc::new(Named(
            ScriptedChatModel::new(tok, 4096, Vec::new()),
            id.to_string(),
        ))
    }

    #[test]
    fn an_exact_id_wins() {
        let models = vec![model("a.gturbo"), model("b.gturbo")];
        match resolve_among(&models, Some("b.gturbo")) {
            Resolution::Model(m) => assert_eq!(m.model_id(), "b.gturbo"),
            _ => panic!("expected the named model"),
        }
    }

    /// The Claude Code case: one model, a name it has never heard of.
    #[test]
    fn one_model_serves_a_name_it_does_not_know() {
        let models = vec![model("gemma4.gturbo")];
        match resolve_among(&models, Some("claude-sonnet-4-6")) {
            Resolution::Model(m) => assert_eq!(m.model_id(), "gemma4.gturbo"),
            _ => panic!("a single model must serve an unrecognized name"),
        }
    }

    #[test]
    fn several_models_refuse_an_unknown_name_and_say_what_is_there() {
        let models = vec![model("a.gturbo"), model("b.gturbo")];
        match resolve_among(&models, Some("nope")) {
            Resolution::Unknown {
                requested,
                available,
            } => {
                assert_eq!(requested, "nope");
                assert_eq!(available, vec!["a.gturbo", "b.gturbo"]);
            }
            _ => panic!("expected a refusal naming both"),
        }
    }

    #[test]
    fn nothing_attached_is_empty_rather_than_unknown() {
        assert!(matches!(
            resolve_among(&[], Some("anything")),
            Resolution::Empty
        ));
    }
}
