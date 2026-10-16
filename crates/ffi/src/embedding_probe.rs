//! Resolves an embedding model argument (a pinned alias or a path) and
//! decides whether the runtime can run it, before anything is loaded.
//!
//! One decision for every host-facing embedding entry: the standalone
//! encode calls, the server attach, and `ts_embedding_probe_json`, which
//! lets a host show the same verdict before the user commits to a model.

use std::path::PathBuf;

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct EmbeddingProbe {
    pub input: String,
    pub resolved_path: Option<PathBuf>,
    pub runnable: bool,
    pub family: Option<String>,
    pub backend: Option<&'static str>,
    /// The resolved directory is the verified install of a catalog profile
    /// whose status is `verified`. A path to the same family elsewhere is
    /// runnable but not verified: nothing checked those bytes.
    pub verified: bool,
    pub profile_alias: Option<String>,
    pub refused_because: Option<String>,
}

impl EmbeddingProbe {
    fn refused(input: &str, why: String) -> Self {
        Self {
            input: input.to_string(),
            resolved_path: None,
            runnable: false,
            family: None,
            backend: None,
            verified: false,
            profile_alias: None,
            refused_because: Some(why),
        }
    }
}

pub(crate) fn probe(arg: &str) -> EmbeddingProbe {
    let arg = arg.trim();
    if arg.is_empty() {
        return EmbeddingProbe::refused(arg, "no embedding model was given".into());
    }
    let cat = match catalog::EmbeddingCatalog::embedded() {
        Ok(c) => c,
        Err(e) => return EmbeddingProbe::refused(arg, e),
    };
    let store = catalog::Store::default_store().ok();

    // A catalog alias means its verified install and nothing else, so an
    // older pin's files never answer for the current one. An existing
    // directory of the same name is a path, as everywhere else.
    let as_path = PathBuf::from(arg);
    let dir = match (cat.get(arg), &store) {
        (Some(_), Some(store)) if !as_path.is_dir() => match cat.installed_path(store, arg) {
            Some(dir) => dir,
            None => {
                let why = cat
                    .install_problem(store, arg)
                    .unwrap_or_else(|| format!("{arg} is not installed"));
                return EmbeddingProbe::refused(arg, why);
            }
        },
        _ => catalog::resolve_model_arg(arg),
    };
    let canonical = dir.canonicalize().unwrap_or_else(|_| dir.clone());

    let profile = store.as_ref().and_then(|store| {
        cat.profiles().iter().find(|p| {
            cat.installed_path(store, &p.alias)
                .and_then(|d| d.canonicalize().ok())
                .is_some_and(|d| d == canonical)
        })
    });

    #[cfg(target_os = "macos")]
    {
        let found = runtime::embedding_probe::probe_embedding_dir(&canonical);
        EmbeddingProbe {
            input: arg.to_string(),
            resolved_path: Some(canonical),
            runnable: found.runnable,
            family: found.family,
            backend: found.backend,
            verified: found.runnable
                && profile.is_some_and(|p| p.status == catalog::Status::Verified),
            profile_alias: profile.map(|p| p.alias.clone()),
            refused_because: found.refused_because,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = profile;
        EmbeddingProbe {
            resolved_path: Some(canonical),
            ..EmbeddingProbe::refused(arg, "embedding models run on macOS only".into())
        }
    }
}
