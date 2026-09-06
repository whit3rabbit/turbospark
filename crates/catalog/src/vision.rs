//! Finding an installed vision-tower sidecar that pairs with a given text
//! family and hidden size (vision memory sidecar, part A5).
//!
//! This is the READ side of the store's vision-tower rows; the WRITE side
//! (`turbospark-model pull-vision`) lives in `install.rs` and `stream.rs`.

use std::path::PathBuf;

use model_io::ModelFamily;

use crate::store::Store;

/// Find the ONE installed vision-tower row pairing with `(family,
/// hidden_size)`.
///
/// **NEVER SILENTLY DISAMBIGUATES.** Two installed towers of the same shape
/// are not interchangeable: the revision pin is load-bearing (this
/// feature's own design principle -- a different revision of "the same"
/// tower can carry different trained weights), so picking one over the
/// other by any rule -- alphabetical, most recently installed, first found
/// -- would silently run a tower the caller never named. Zero matches and
/// more than one match are both refused, with messages a reader can tell
/// apart without reading the code.
///
/// Reads back through [`model_io::load_vision_sidecar`] for every candidate
/// row rather than trusting the store's own `family` string, because that
/// loader is the full validated read (the record parses, the manifest loads
/// against the arch the record implies, and the two agree on hidden size)
/// and a store record that merely LOOKS like a tower is not evidence that
/// the directory still is one -- an install can rot on disk after the JSON
/// record was written.
pub fn resolve_vision_sidecar(
    store: &Store,
    family: ModelFamily,
    hidden_size: i64,
) -> Result<PathBuf, String> {
    let wanted = family.as_str();
    let mut matches: Vec<PathBuf> = Vec::new();
    for row in store.installed().values() {
        if row.kind.as_deref() != Some("vision-tower") {
            continue;
        }
        let Ok((record, _vision)) = model_io::load_vision_sidecar(&row.path) else {
            continue;
        };
        if record.pairs_with.family == wanted && record.pairs_with.hidden_size == hidden_size {
            matches.push(row.path.clone());
        }
    }
    match matches.len() {
        0 => Err(format!(
            "no installed vision-tower pairs with family {wanted:?} at hidden_size \
             {hidden_size}. `turbospark-model list` shows installed rows; \
             `turbospark-model pull-vision <ALIAS>` installs one."
        )),
        1 => Ok(matches.remove(0)),
        n => Err(format!(
            "{n} installed vision towers pair with family {wanted:?} at hidden_size \
             {hidden_size}: {}. The revision pin is load-bearing, so this is refused \
             rather than picked between -- remove the ones you do not want, or point \
             --vision-sidecar at a directory directly.",
            matches
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InstalledModel;
    use std::path::Path;

    fn temp_store(tag: &str) -> Store {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-catalog-vision-resolve-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        Store::new(dir)
    }

    /// Writes a real, loadable vision-tower sidecar at
    /// `<store>/models/<alias>.gturbo-vision` and records it, using the same
    /// fixture builder `crates/repack`'s own offline tests use -- so
    /// `model_io::load_vision_sidecar` inside [`resolve_vision_sidecar`] is
    /// exercised against a genuine directory rather than a hand-typed record.
    fn install_tower(store: &Store, alias: &str) -> PathBuf {
        let dir = store.vision_install_path(alias);
        std::fs::create_dir_all(&dir).unwrap();
        repack::build_synthetic_vision_sidecar(&dir, alias).expect("fixture writes");
        store
            .record(&InstalledModel {
                alias: alias.to_string(),
                repo: "test/vision-sidecar-fixture".to_string(),
                revision: "0".repeat(40),
                path: dir.clone(),
                family: ModelFamily::QwenGdnDense.as_str().to_string(),
                install_bytes: 1,
                installed_on: "2026-01-01".to_string(),
                status: "runs".to_string(),
                kind: Some("vision-tower".to_string()),
            })
            .expect("recording the row");
        dir
    }

    fn hidden_size_of(dir: &Path) -> i64 {
        model_io::load_vision_sidecar(dir)
            .unwrap()
            .1
            .out_hidden_size
    }

    #[test]
    fn resolves_exactly_one_match() {
        let store = temp_store("one");
        let dir = install_tower(&store, "tower-a");
        let hidden = hidden_size_of(&dir);
        let found = resolve_vision_sidecar(&store, ModelFamily::QwenGdnDense, hidden)
            .expect("exactly one candidate should resolve");
        assert_eq!(found, dir);
    }

    #[test]
    fn zero_matches_names_the_absence_distinctly_from_a_multiple_match() {
        let store = temp_store("zero");
        install_tower(&store, "tower-a");
        // A family this fixture never pairs with -- no candidate at all.
        let err = resolve_vision_sidecar(&store, ModelFamily::Gemma4, 5120).unwrap_err();
        assert!(err.contains("no installed vision-tower"), "{err}");
        assert!(!err.contains("installed vision towers"), "{err}");
    }

    #[test]
    fn two_matching_towers_are_refused_rather_than_picked_between() {
        let store = temp_store("two");
        let dir_a = install_tower(&store, "tower-a");
        let hidden = hidden_size_of(&dir_a);
        install_tower(&store, "tower-b");
        let err = resolve_vision_sidecar(&store, ModelFamily::QwenGdnDense, hidden).unwrap_err();
        assert!(err.contains("2 installed vision towers"), "{err}");
        assert!(
            err.contains("tower-a") || err.contains("gturbo-vision"),
            "{err}"
        );
    }

    /// A trunk (non-tower) row must never be returned, whatever family or
    /// hidden size happens to match its own record.
    #[test]
    fn a_trunk_row_is_never_returned_as_a_tower_candidate() {
        let store = temp_store("skip-trunk");
        let dir = store.install_path("trunk");
        std::fs::create_dir_all(&dir).unwrap();
        store
            .record(&InstalledModel {
                alias: "trunk".to_string(),
                repo: "owner/name".to_string(),
                revision: "main".to_string(),
                path: dir,
                family: ModelFamily::QwenGdnDense.as_str().to_string(),
                install_bytes: 1,
                installed_on: "2026-01-01".to_string(),
                status: "runs".to_string(),
                kind: None,
            })
            .expect("recording the row");
        let err = resolve_vision_sidecar(&store, ModelFamily::QwenGdnDense, 5120).unwrap_err();
        assert!(err.contains("no installed vision-tower"), "{err}");
    }
}
