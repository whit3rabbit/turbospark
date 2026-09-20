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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::model::ChatModel;

/// One attached backend, its public ids, and what the FFI reports back to a GUI.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRow {
    /// The id a client puts in a request's `model` field.
    pub id: String,
    /// Extra public ids that resolve to this same backend. `GET /v1/models`
    /// expands these into first-class response entries because Claude Code's
    /// gateway discovery filters on the entry's id.
    pub aliases: Vec<String>,
    /// The resolved context window, which is a property of how the model was
    /// OPENED rather than of the checkpoint (AGENTS.md Gotcha 55), so it is
    /// reported per row rather than assumed.
    pub max_context: u32,
}

impl ModelRow {
    /// Every id clients may use for this backend, in stable display order.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.id.as_str()).chain(self.aliases.iter().map(String::as_str))
    }
}

/// Builds the registry-facing metadata for one backend. Kept here so the
/// static registry and the FFI's live registry cannot disagree about which
/// identities a model exposes.
pub fn model_row(model: &dyn ChatModel) -> ModelRow {
    ModelRow {
        id: model.model_id().to_string(),
        aliases: model.model_aliases(),
        max_context: model.max_context(),
    }
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
        vec![model_row(&*self.0)]
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
    /// Refuses a duplicate public identity rather than silently shadowing one
    /// entry with another: `resolve_among`'s exact-id match returns the FIRST
    /// hit, so two canonical ids or aliases sharing an id would make the
    /// second permanently unreachable by name.
    pub fn new(models: Vec<Arc<dyn ChatModel>>) -> Result<Self, String> {
        let mut seen = std::collections::HashSet::new();
        for m in &models {
            for id in model_row(&**m).ids() {
                if !seen.insert(id.to_string()) {
                    return Err(format!(
                        "duplicate model identity {id:?}: two attached models cannot share an id or alias"
                    ));
                }
            }
        }
        Ok(Self(models))
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
        self.0.iter().map(|m| model_row(&**m)).collect()
    }
}

/// N backends sharing ONE public identity (ROADMAP P3.6, the multi-runner
/// pool): N opens of the same install, each a fully independent
/// `RealChatModel` with its own KV, its own session pool and -- the property
/// that makes the pool CONCURRENT -- its own one-permit
/// [`crate::queue::GenerationQueue`]. A request is routed to one member and
/// serializes on that member's gate alone, so N requests can generate at
/// once where a single runner would queue them.
///
/// **THE REGISTRY LEVEL IS WHERE THE FAN-OUT LIVES, and that is forced by
/// the trait shape, not chosen for tidiness.** A handler resolves one
/// `Arc<dyn ChatModel>` and later asks THAT handle for its
/// `generation_queue()` -- the gate and the generation must be the same
/// runner, or two requests admitted by different gates would still collide
/// on one runner's mutex. A `ChatModel` wrapper that picked a member
/// internally could not keep the two in step, because the queue is fetched
/// separately from every generation call. Routing per REQUEST, here, hands
/// the handler a single self-consistent backend.
///
/// **MEMBERS ARE CHOSEN LEAST-BUSY-FIRST, ROUND-ROBIN ON TIES**, reading
/// each gate's `load()` (active plus waiting) counter: an idle member is
/// always preferred over a busy one, and among equally-loaded members the
/// cursor spreads arrivals so an all-idle pool distributes rather than
/// stampeding member 0. The counter is advisory at the moment of choice (a
/// member can pick up work between the read and the routing), which is fine:
/// it is a balance heuristic riding on top of gates that are CORRECT whatever
/// it reads.
///
/// `rows()` reports ONE row, because the pool IS one model as far as a
/// client can tell: one id, one advertised context, and `/v1/models` has
/// nothing to say about how many runners stand behind it.
pub struct PoolRegistry {
    members: Vec<Arc<dyn ChatModel>>,
    /// Round-robin cursor; `fetch_add` makes concurrent resolves spread
    /// without a lock. Wrapping is fine -- everything reads it modulo the
    /// member count.
    next: AtomicUsize,
}

impl PoolRegistry {
    /// Every member must advertise THE SAME public identity: the pool is
    /// one model N times, and a member with its own id or alias would be
    /// unreachable by name (exact-id match would find whichever member the
    /// fan-out picked, not the one the name named). Distinct identities are
    /// what [`StaticRegistry`] is for.
    pub fn new(members: Vec<Arc<dyn ChatModel>>) -> Result<Self, String> {
        let Some(first) = members.first() else {
            return Err("a pool needs at least one member".to_string());
        };
        let identity = model_row(&**first);
        for m in members.iter().skip(1) {
            let row = model_row(&**m);
            if row.id != identity.id || row.aliases != identity.aliases {
                return Err(format!(
                    "pool members must share one identity: member 0 is {:?} with aliases                      {:?}, but another member is {:?} with {:?}; use distinct --model                      flags (a static registry) for models that route by name",
                    identity.id, identity.aliases, row.id, row.aliases
                ));
            }
            if row.max_context != identity.max_context {
                return Err(format!(
                    "pool members must be opened at one context: member 0 reports {}, \
                     another reports {}",
                    identity.max_context, row.max_context
                ));
            }
        }
        if members.len() == 1 {
            return Err(
                "a pool of one is not a pool; open the model without --pool-size".to_string(),
            );
        }
        Ok(Self {
            members,
            next: AtomicUsize::new(0),
        })
    }

    /// The member a request should run on: the first idle member scanning
    /// from the round-robin cursor, else the least-busy one in that scan
    /// order. `load()` includes the active permit holder; a member with no
    /// gate (scripted backends in tests) reads as permanently idle.
    fn pick(&self) -> Arc<dyn ChatModel> {
        let n = self.members.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        let mut best = 0usize;
        let mut best_load = usize::MAX;
        for i in 0..n {
            let idx = (start.wrapping_add(i)) % n;
            let load = self.members[idx]
                .generation_queue()
                .map(|q| q.load())
                .unwrap_or(0);
            if load < best_load {
                best_load = load;
                best = idx;
            }
            if best_load == 0 {
                break;
            }
        }
        Arc::clone(&self.members[best])
    }
}

impl ModelRegistry for PoolRegistry {
    fn resolve(&self, requested: Option<&str>) -> Resolution {
        // One identity means the routing policy's three cases collapse into
        // one answer: a request naming the id, an alias, nothing, or
        // something else entirely is all "the pool" -- there is nothing the
        // name could disambiguate.
        let _ = requested;
        Resolution::Model(self.pick())
    }

    fn resolve_embedding(&self, requested: Option<&str>) -> Resolution {
        let _ = requested;
        Resolution::Model(self.pick())
    }

    fn rows(&self) -> Vec<ModelRow> {
        vec![model_row(&*self.members[0])]
    }
}

/// Resolves `requested` against a slice of attached models under the policy
/// this module's header describes. Shared so a registry with interior
/// mutability (the FFI's `RwLock<HashMap<..>>` one) states the rule once
/// rather than restating it against its own storage.
pub fn resolve_among(models: &[Arc<dyn ChatModel>], requested: Option<&str>) -> Resolution {
    resolve_capability(models, requested, false)
}

/// Resolves `requested` against a slice of attached models for an embedding
/// request. See [`resolve_capability`] for the shared policy; scoped here to
/// the embedding-capable subset, which is what lets a caller asking for a
/// well-known default name (`"text-embedding-3-small"`, `"bge-small"`, ...)
/// reach the one attached embedding model whatever name it actually carries.
pub fn resolve_embedding_among(
    models: &[Arc<dyn ChatModel>],
    requested: Option<&str>,
) -> Resolution {
    resolve_capability(models, requested, true)
}

/// The capability-aware resolution both public entry points share.
///
/// **AN EXACT ID MATCH ONLY WINS WHEN THE MATCHED MODEL HAS THE RIGHT
/// CAPABILITY.** Matching by name alone let a chat request naming an
/// embedding model's own id reach `RealEncoderModel::with_producer`, which
/// unconditionally refuses generation -- a `RuntimeError::Producer` that
/// `status_for` maps to 500, for a request that was never going to succeed
/// and should have been refused at ROUTING, not deep inside a generation
/// attempt. A name matching the WRONG kind of model is treated as not
/// found among what this request type can use, exactly like a name
/// matching nothing at all.
///
/// **THE CAPABLE SUBSET, NEVER THE WHOLE ROSTER, DECIDES THE OTHER TWO
/// CASES.** Zero capable models is [`Resolution::Empty`] (503): an
/// embedding request against a chat-only server used to fall through to
/// `resolve_among`'s single-model default, reach `ChatModel::encode`'s
/// trait-default refusal, and surface as a 400 -- blaming the caller's
/// request shape for what is actually a server CONFIGURATION gap (no
/// embedding model is attached at all). Exactly one capable model is served
/// whatever name was asked for, the same "no ambiguity" reasoning a lone
/// attached model already gets. Two or more capable models with no exact
/// match is [`Resolution::Unknown`], naming only the USABLE ones -- listing
/// a model of the wrong kind in "did you mean one of these" would send a
/// caller straight back into this same refusal.
fn resolve_capability(
    models: &[Arc<dyn ChatModel>],
    requested: Option<&str>,
    want_embedding: bool,
) -> Resolution {
    let wants = |m: &Arc<dyn ChatModel>| m.supports_embeddings() == want_embedding;
    if let Some(name) = requested {
        if !name.is_empty() {
            if let Some(hit) = models
                .iter()
                .find(|m| wants(m) && model_row(&***m).ids().any(|id| id == name))
            {
                return Resolution::Model(Arc::clone(hit));
            }
        }
    }
    let capable: Vec<&Arc<dyn ChatModel>> = models.iter().filter(|m| wants(m)).collect();
    match capable.as_slice() {
        [] => Resolution::Empty,
        // The fallback. One capable model means there is no ambiguity to
        // resolve, so an unrecognized name is the caller's label for the
        // conversation rather than a routing instruction.
        [only] => Resolution::Model(Arc::clone(only)),
        several => Resolution::Unknown {
            requested: requested.unwrap_or("").to_string(),
            available: several
                .iter()
                .flat_map(|m| {
                    model_row(&***m)
                        .ids()
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .collect(),
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
    struct Named {
        inner: ScriptedChatModel,
        id: String,
        is_embedding: bool,
        aliases: Option<Vec<String>>,
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
        fn model_aliases(&self) -> Vec<String> {
            if self.is_embedding {
                Vec::new()
            } else {
                self.aliases
                    .clone()
                    .unwrap_or_else(|| vec![format!("claude-turbospark-{}", self.model_id())])
            }
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

    pub(crate) fn model(id: &str) -> Arc<dyn ChatModel> {
        model_with_embedding(id, false)
    }

    pub(crate) fn model_with_embedding(id: &str, is_embedding: bool) -> Arc<dyn ChatModel> {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
        let tok = tokenizer::MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer");
        Arc::new(Named {
            inner: ScriptedChatModel::new(tok, 4096, Vec::new()),
            id: id.to_string(),
            is_embedding,
            aliases: None,
        })
    }

    fn model_with_aliases(id: &str, aliases: &[&str]) -> Arc<dyn ChatModel> {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
        let tok = tokenizer::MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer");
        Arc::new(Named {
            inner: ScriptedChatModel::new(tok, 4096, Vec::new()),
            id: id.to_string(),
            is_embedding: false,
            aliases: Some(aliases.iter().map(|alias| (*alias).to_string()).collect()),
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

    #[test]
    fn an_exact_alias_wins_when_several_chat_models_are_attached() {
        let models = vec![model("a.gturbo"), model("b.gturbo")];
        match resolve_among(&models, Some("claude-turbospark-b.gturbo")) {
            Resolution::Model(m) => assert_eq!(m.model_id(), "b.gturbo"),
            _ => panic!("expected the alias to select b.gturbo"),
        }
    }

    #[test]
    fn embedding_models_do_not_advertise_claude_aliases() {
        let embedding = model_with_embedding("bge-small-en-v1.5", true);
        assert!(model_row(&*embedding).aliases.is_empty());
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
                assert_eq!(
                    available,
                    vec![
                        "a.gturbo",
                        "claude-turbospark-a.gturbo",
                        "b.gturbo",
                        "claude-turbospark-b.gturbo",
                    ]
                );
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

    /// F17: a chat request naming an embedding model's OWN id exactly used
    /// to route there anyway (the name matched), reach
    /// `ChatModel::with_producer`'s trait-default refusal, and surface as a
    /// 500 for a request that could never have succeeded. The name must be
    /// treated as not found among usable chat models, not as a routing
    /// instruction to a backend that cannot serve the request.
    #[test]
    fn a_chat_request_naming_an_embedding_models_own_id_is_not_routed_there() {
        let models = vec![
            model_with_embedding("gemma4.gturbo", false),
            model_with_embedding("bge-small-en-v1.5", true),
        ];
        match resolve_among(&models, Some("bge-small-en-v1.5")) {
            // With exactly one USABLE (chat) model left, the single-model
            // fallback still serves it -- the exact id just does not win.
            Resolution::Model(m) => assert_eq!(m.model_id(), "gemma4.gturbo"),
            Resolution::Unknown { .. } => panic!("expected the fallback chat model, not Unknown"),
            Resolution::Empty => panic!("expected the fallback chat model, not Empty"),
        }
    }

    /// The mirror case: an embedding request naming a CHAT model's exact id
    /// must not be routed to it either.
    #[test]
    fn an_embedding_request_naming_a_chat_models_own_id_is_not_routed_there() {
        let models = vec![
            model_with_embedding("gemma4.gturbo", false),
            model_with_embedding("bge-small-en-v1.5", true),
        ];
        match resolve_embedding_among(&models, Some("gemma4.gturbo")) {
            Resolution::Model(m) => assert_eq!(m.model_id(), "bge-small-en-v1.5"),
            Resolution::Unknown { .. } => {
                panic!("expected the fallback embedding model, not Unknown")
            }
            Resolution::Empty => panic!("expected the fallback embedding model, not Empty"),
        }
    }

    /// F17: an embedding request against a chat-only server used to fall
    /// through to `resolve_among`'s single-model default, reach
    /// `ChatModel::encode`'s trait-default refusal, and surface as a 400 --
    /// blaming the caller's request for a server CONFIGURATION gap. Must be
    /// `Empty` (503): nothing here can serve this request, and it is not
    /// the caller's fault.
    #[test]
    fn an_embedding_request_against_a_chat_only_registry_is_empty_not_a_client_error() {
        let models = vec![model_with_embedding("gemma4.gturbo", false)];
        assert!(matches!(
            resolve_embedding_among(&models, None),
            Resolution::Empty
        ));
        assert!(matches!(
            resolve_embedding_among(&models, Some("text-embedding-3-small")),
            Resolution::Empty
        ));
    }

    /// The mirror case: a chat request against an embedding-only server.
    #[test]
    fn a_chat_request_against_an_embedding_only_registry_is_empty() {
        let models = vec![model_with_embedding("bge-small-en-v1.5", true)];
        assert!(matches!(resolve_among(&models, None), Resolution::Empty));
    }

    /// A `404` naming the usable models must not offer one of the wrong
    /// kind: that would send the caller straight back into the same
    /// refusal.
    #[test]
    fn ambiguous_resolution_lists_only_capability_matching_models() {
        let models = vec![
            model_with_embedding("a.gturbo", false),
            model_with_embedding("b.gturbo", false),
            model_with_embedding("bge-small-en-v1.5", true),
        ];
        match resolve_among(&models, Some("nope")) {
            Resolution::Unknown { available, .. } => {
                assert_eq!(
                    available,
                    vec![
                        "a.gturbo",
                        "claude-turbospark-a.gturbo",
                        "b.gturbo",
                        "claude-turbospark-b.gturbo",
                    ]
                );
            }
            Resolution::Model(_) => panic!("expected an ambiguous refusal, got Model"),
            Resolution::Empty => panic!("expected an ambiguous refusal, got Empty"),
        }
    }

    #[test]
    fn a_duplicate_model_id_is_refused_at_construction() {
        match StaticRegistry::new(vec![model("a.gturbo"), model("a.gturbo")]) {
            Err(err) => assert!(err.contains("a.gturbo"), "{err}"),
            Ok(_) => panic!("expected a duplicate-id refusal"),
        }
    }

    #[test]
    fn a_canonical_id_cannot_collide_with_an_alias() {
        match StaticRegistry::new(vec![
            model_with_aliases("alpha.gturbo", &["shared"]),
            model("shared"),
        ]) {
            Err(err) => assert!(err.contains("shared"), "{err}"),
            Ok(_) => panic!("expected a canonical-to-alias refusal"),
        }
    }

    #[test]
    fn aliases_cannot_collide_across_models() {
        match StaticRegistry::new(vec![
            model_with_aliases("alpha.gturbo", &["shared"]),
            model_with_aliases("beta.gturbo", &["shared"]),
        ]) {
            Err(err) => assert!(err.contains("shared"), "{err}"),
            Ok(_) => panic!("expected an alias-to-alias refusal"),
        }
    }

    #[test]
    fn distinct_model_ids_construct_fine() {
        assert!(StaticRegistry::new(vec![model("a.gturbo"), model("b.gturbo")]).is_ok());
    }

    #[cfg(test)]
    mod pool_tests {
        use super::*;

        struct Gated {
            inner: Arc<dyn ChatModel>,
            queue: Arc<crate::GenerationQueue>,
        }

        impl ChatModel for Gated {
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
                self.inner.model_id()
            }
            fn model_aliases(&self) -> Vec<String> {
                self.inner.model_aliases()
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
            fn generation_queue(&self) -> Option<Arc<crate::GenerationQueue>> {
                Some(Arc::clone(&self.queue))
            }
        }

        fn gated_model(id: &str) -> (Arc<dyn ChatModel>, Arc<crate::GenerationQueue>) {
            let queue = crate::GenerationQueue::shared();
            let model: Arc<dyn ChatModel> = Arc::new(Gated {
                inner: model(id),
                queue: Arc::clone(&queue),
            });
            (model, queue)
        }

        /// N members of one install advertise THE SAME identity, and the pool
        /// exists precisely so that identity can be duplicated. `StaticRegistry`
        /// refuses what this accepts, and the pair of refusals is the contract:
        /// duplicates mean "one model, N runners" here and a routing bug there.
        #[test]
        fn a_pool_allows_what_a_static_registry_refuses() {
            assert!(StaticRegistry::new(vec![model("a.gturbo"), model("a.gturbo")]).is_err());
            assert!(PoolRegistry::new(vec![model("a.gturbo"), model("a.gturbo")]).is_ok());
        }

        #[test]
        fn a_pool_of_one_is_refused() {
            match PoolRegistry::new(vec![model("a.gturbo")]) {
                Err(err) => assert!(err.contains("not a pool"), "{err}"),
                Ok(_) => panic!("a single-member pool is a misconfiguration, not a registry"),
            }
        }

        /// Members with DIFFERENT identities are a static registry's job: the
        /// pool's exact-id match would find whichever member the fan-out
        /// picked, not the one the name named.
        #[test]
        fn members_with_distinct_ids_are_refused() {
            match PoolRegistry::new(vec![model("a.gturbo"), model("b.gturbo")]) {
                Err(err) => assert!(err.contains("share one identity"), "{err}"),
                Ok(_) => panic!("expected a distinct-identity refusal"),
            }
        }

        /// The context window is a property of how a member was OPENED, so two
        /// members of one install opened at different windows would advertise
        /// one row that describes neither faithfully.
        #[test]
        fn members_at_different_contexts_are_refused() {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/ChatMLTokenizer");
            let tok = tokenizer::MfTokenizer::load_from_dir(&dir).expect("fixture tokenizer");
            let big = Arc::new(Named {
                inner: ScriptedChatModel::new(tok, 8192, Vec::new()),
                id: "a.gturbo".to_string(),
                is_embedding: false,
                aliases: None,
            });
            match PoolRegistry::new(vec![model("a.gturbo"), big]) {
                Err(err) => assert!(err.contains("one context"), "{err}"),
                Ok(_) => panic!("expected a context-mismatch refusal"),
            }
        }

        /// `/v1/models` reports ONE row for the pool: one id, one context, N
        /// runners behind it that a client cannot see and does not need to.
        #[test]
        fn rows_report_one_entry_whatever_the_member_count() {
            let pool = PoolRegistry::new(vec![
                model("a.gturbo"),
                model("a.gturbo"),
                model("a.gturbo"),
            ])
            .expect("identical members");
            let rows = pool.rows();
            assert_eq!(rows.len(), 1, "the pool is one model to a client");
            assert_eq!(rows[0].id, "a.gturbo");
        }

        /// With every member idle, arrivals spread round-robin: N resolves hit
        /// N distinct members exactly once each. Compared by `Arc` pointer,
        /// because every member answers to the same id on purpose.
        #[test]
        fn idle_members_are_served_round_robin() {
            let members: Vec<Arc<dyn ChatModel>> =
                vec![model("a.gturbo"), model("a.gturbo"), model("a.gturbo")];
            let pool = PoolRegistry::new(members.clone()).expect("identical members");

            let mut seen: Vec<usize> = Vec::new();
            for _ in 0..members.len() {
                let Resolution::Model(served) = pool.resolve(Some("a.gturbo")) else {
                    panic!("expected a member");
                };
                let idx = members
                    .iter()
                    .position(|m| Arc::ptr_eq(m, &served))
                    .expect("the served backend is one of the members");
                seen.push(idx);
            }
            seen.sort_unstable();
            assert_eq!(
                seen,
                vec![0, 1, 2],
                "an all-idle pool must spread arrivals across every member"
            );
        }

        /// An acquired permit is active load even when nobody is waiting.
        /// Starting the scan at that member must not queue a request there
        /// while the other member is idle.
        #[tokio::test]
        async fn active_member_is_skipped_for_an_idle_member() {
            let (active, active_queue) = gated_model("a.gturbo");
            let (idle, _idle_queue) = gated_model("a.gturbo");
            let pool =
                PoolRegistry::new(vec![active, Arc::clone(&idle)]).expect("identical members");
            let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let _active_permit = active_queue.acquire(&cancel).await.expect("admitted");

            let Resolution::Model(served) = pool.resolve(Some("a.gturbo")) else {
                panic!("expected a member");
            };
            assert!(
                Arc::ptr_eq(&served, &idle),
                "an idle runner must win over an active runner with no waiters"
            );
        }

        /// An unknown name resolves to a member, exactly as a single model
        /// serves one it does not know: with ONE identity attached there is
        /// nothing the name could disambiguate, and this is the property Claude
        /// Code's gateway discovery depends on.
        #[test]
        fn a_pool_serves_a_name_it_does_not_know() {
            let pool = PoolRegistry::new(vec![model("a.gturbo"), model("a.gturbo")])
                .expect("identical members");
            match pool.resolve(Some("claude-sonnet-4-6")) {
                Resolution::Model(_) => {}
                _ => panic!("the single-identity fallback must carry into the pool"),
            }
        }
    }
}
