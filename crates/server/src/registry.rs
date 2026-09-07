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
    /// Resolves an embedding backend for embedding requests.
    fn resolve_embedding(&self, requested: Option<&str>) -> Resolution {
        self.resolve(requested)
    }
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

    fn resolve_embedding(&self, _requested: Option<&str>) -> Resolution {
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

    fn resolve_embedding(&self, requested: Option<&str>) -> Resolution {
        resolve_embedding_among(&self.0, requested)
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
        if !name.is_empty() {
            if let Some(hit) = models.iter().find(|m| m.model_id() == name) {
                return Resolution::Model(Arc::clone(hit));
            }
        }
    }
    match models {
        [] => Resolution::Empty,
        // The fallback. One model attached means there is no ambiguity to
        // resolve, so an unrecognized name is the caller's label for the
        // conversation rather than a routing instruction.
        [only] => Resolution::Model(Arc::clone(only)),
        several => {
            // When multiple models are attached, but exactly one is generative
            // (i.e. not an embedding-only model), default generative chat requests
            // to that single generative model rather than failing with 404.
            let gen_models: Vec<_> = several
                .iter()
                .filter(|m| !m.supports_embeddings())
                .collect();
            if gen_models.len() == 1 {
                return Resolution::Model(Arc::clone(gen_models[0]));
            }

            Resolution::Unknown {
                requested: requested.unwrap_or("").to_string(),
                available: several.iter().map(|m| m.model_id().to_string()).collect(),
            }
        }
    }
}

/// Resolves `requested` against a slice of attached models for an embedding
/// request.
///
/// An exact id match always wins. Failing that, when EXACTLY ONE attached
/// model supports embeddings, it serves the request whatever name was
/// asked for -- there is no ambiguity to resolve, the same reasoning
/// [`resolve_among`] already applies to a single attached model, scoped
/// here to the embedding-capable subset. This one rule is what a caller
/// asking for a well-known default name (`"text-embedding-3-small"`,
/// `"bge-small"`, ...) needs; a caller asking for genuine garbage gets the
/// same answer, because with only one candidate there IS no other answer.
///
/// With zero or two-or-more embedding-capable candidates there is no
/// single answer to default to -- picking one anyway would silently serve
/// the wrong model's (wrong-dimension) vectors with no error -- so this
/// defers to [`resolve_among`]'s general policy (single attached model, or
/// a 404 naming what is there) instead of guessing.
pub fn resolve_embedding_among(
    models: &[Arc<dyn ChatModel>],
    requested: Option<&str>,
) -> Resolution {
    if let Some(name) = requested {
        if !name.is_empty() {
            if let Some(hit) = models.iter().find(|m| m.model_id() == name) {
                return Resolution::Model(Arc::clone(hit));
            }
        }
    }
    let mut embedding_models = models.iter().filter(|m| m.supports_embeddings());
    if let Some(only) = embedding_models.next() {
        if embedding_models.next().is_none() {
            return Resolution::Model(Arc::clone(only));
        }
    }
    resolve_among(models, requested)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScriptedChatModel;

    /// `ScriptedChatModel::model_id` is the constant `"scripted"`, so a
    /// per-id backend needs a wrapper. Naming it after the install
    /// directory mirrors what `RealChatModel::open` and the FFI both do.
    struct Named {
        inner: ScriptedChatModel,
        id: String,
        is_embedding: bool,
    }

    impl ChatModel for Named {
        fn tokenizer(&self) -> &tokenizer::MfTokenizer {
            self.inner.tokenizer()
        }
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }
        fn max_context(&self) -> u32 {
            self.inner.max_context()
        }
        fn model_id(&self) -> &str {
            &self.id
        }
        fn with_producer(
            &self,
            f: &mut dyn FnMut(
                &mut dyn runtime::LogitProducer,
            )
                -> Result<runtime::RawDecodeResult, runtime::RuntimeError>,
        ) -> Result<runtime::RawDecodeResult, runtime::RuntimeError> {
            self.inner.with_producer(f)
        }
        fn supports_embeddings(&self) -> bool {
            self.is_embedding
        }
    }

    fn model(id: &str) -> Arc<dyn ChatModel> {
        model_with_embedding(id, false)
    }

    fn model_with_embedding(id: &str, is_embedding: bool) -> Arc<dyn ChatModel> {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
        let tok = tokenizer::MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer");
        Arc::new(Named {
            inner: ScriptedChatModel::new(tok, 4096, Vec::new()),
            id: id.to_string(),
            is_embedding,
        })
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

    #[test]
    fn embedding_resolution_defaults_to_embedding_model_when_omitted() {
        let models = vec![
            model_with_embedding("gemma4.gturbo", false),
            model_with_embedding("bge-small-en-v1.5", true),
        ];

        // An embedding request omitting the model defaults to bge-small-en-v1.5
        match resolve_embedding_among(&models, None) {
            Resolution::Model(m) => assert_eq!(m.model_id(), "bge-small-en-v1.5"),
            _ => panic!("expected default embedding model"),
        }
    }

    #[test]
    fn embedding_resolution_handles_generic_defaults_and_exact_match() {
        let models = vec![
            model_with_embedding("gemma4.gturbo", false),
            model_with_embedding("bge-small-en-v1.5", true),
        ];

        // Standard OpenAI default model name text-embedding-3-small routes to bge-small-en-v1.5
        match resolve_embedding_among(&models, Some("text-embedding-3-small")) {
            Resolution::Model(m) => assert_eq!(m.model_id(), "bge-small-en-v1.5"),
            _ => panic!("expected generic embedding alias to route to attached embedding model"),
        }

        // Exact match
        match resolve_embedding_among(&models, Some("bge-small-en-v1.5")) {
            Resolution::Model(m) => assert_eq!(m.model_id(), "bge-small-en-v1.5"),
            _ => panic!("expected exact match"),
        }
    }

    #[test]
    fn mixed_generative_and_embedding_models_route_chat_and_embeddings_transparently() {
        let models = vec![
            model_with_embedding("gemma4.gturbo", false),
            model_with_embedding("bge-small-en-v1.5", true),
        ];

        // Chat requests without model or with unrecognized client name route to the single generative model
        match resolve_among(&models, Some("claude-sonnet-4-6")) {
            Resolution::Model(m) => assert_eq!(m.model_id(), "gemma4.gturbo"),
            _ => panic!("expected chat request to default to single generative model"),
        }

        // Embedding requests default to the embedding model
        match resolve_embedding_among(&models, None) {
            Resolution::Model(m) => assert_eq!(m.model_id(), "bge-small-en-v1.5"),
            _ => panic!("expected embedding request to default to embedding model"),
        }
    }
}
