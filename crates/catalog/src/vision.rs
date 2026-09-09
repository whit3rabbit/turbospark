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
/// The outcome of matching installed vision-tower rows against a trunk's
/// `(family, hidden_size)`. Zero and ambiguous are kept apart because the
/// two consumers render them differently: an explicit `--vision-sidecar`
/// path refuses both, while `auto` treats zero as "run text-only" and
/// still refuses an ambiguity.
enum SidecarMatch {
    One(PathBuf),
    Zero(String),
    Ambiguous(Vec<PathBuf>),
}

fn match_installed_sidecar(store: &Store, family: ModelFamily, hidden_size: i64) -> SidecarMatch {
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
        0 => SidecarMatch::Zero(format!(
            "no installed vision-tower pairs with family {wanted:?} at hidden_size \
             {hidden_size}. `turbospark-model list` shows installed rows; \
             `turbospark-model pull-vision <ALIAS>` installs one."
        )),
        1 => SidecarMatch::One(matches.remove(0)),
        _ => SidecarMatch::Ambiguous(matches),
    }
}

fn ambiguous_message(family: ModelFamily, hidden_size: i64, dirs: &[PathBuf]) -> String {
    format!(
        "{} installed vision towers pair with family {:?} at hidden_size \
         {hidden_size}: {}. The revision pin is load-bearing, so this is refused \
         rather than picked between -- remove the ones you do not want, or point \
         --vision-sidecar at a directory directly.",
        dirs.len(),
        family.as_str(),
        dirs.iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub fn resolve_vision_sidecar(
    store: &Store,
    family: ModelFamily,
    hidden_size: i64,
) -> Result<PathBuf, String> {
    match match_installed_sidecar(store, family, hidden_size) {
        SidecarMatch::One(dir) => Ok(dir),
        SidecarMatch::Zero(reason) => Err(reason),
        SidecarMatch::Ambiguous(dirs) => Err(ambiguous_message(family, hidden_size, &dirs)),
    }
}

/// What `--vision-sidecar auto` resolves to for one opened trunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoVisionSidecar {
    /// Exactly one installed tower pairs with the trunk; attach it.
    Attached(PathBuf),
    /// Run text-only. Carries the reason a startup line should print, so
    /// every front end reports the same words rather than three paraphrases.
    TextOnly(String),
}

/// Resolve `--vision-sidecar auto` for a trunk that has already opened, by
/// the trunk's own `(family, hidden_size)`.
///
/// The three outcomes, in the order they are decided:
/// - the trunk's own install declares a tower: `auto` is a no-op returning
///   [`AutoVisionSidecar::TextOnly`] saying so. Attaching a sidecar beside
///   it would be a second tower for one session, which the runtime refuses
///   for an explicit path and `auto` must not sneak past.
/// - no installed tower pairs: text-only, carrying the same reason an
///   explicit path would be refused with. This is the `--speculative auto`
///   shape: best effort, reported, never a failed open over an absent
///   optional component.
/// - more than one pairs: REFUSED. The revision pin is load-bearing (see
///   [`resolve_vision_sidecar`]); `auto` never picks between candidates.
pub fn resolve_vision_sidecar_auto(
    store: &Store,
    trunk_has_own_tower: bool,
    family: ModelFamily,
    hidden_size: i64,
) -> Result<AutoVisionSidecar, String> {
    if trunk_has_own_tower {
        return Ok(AutoVisionSidecar::TextOnly(
            "the trunk's own install already declares a vision tower; \
             leaving it in place"
                .to_string(),
        ));
    }
    match match_installed_sidecar(store, family, hidden_size) {
        SidecarMatch::One(dir) => Ok(AutoVisionSidecar::Attached(dir)),
        SidecarMatch::Zero(reason) => Ok(AutoVisionSidecar::TextOnly(reason)),
        SidecarMatch::Ambiguous(dirs) => Err(ambiguous_message(family, hidden_size, &dirs)),
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

    /// `--vision-sidecar auto` attaches when exactly one tower pairs.
    #[test]
    fn auto_attaches_the_one_pairing_tower() {
        let store = temp_store("auto-one");
        let dir = install_tower(&store, "tower-a");
        let hidden = hidden_size_of(&dir);
        let found = resolve_vision_sidecar_auto(&store, false, ModelFamily::QwenGdnDense, hidden)
            .expect("one candidate resolves");
        assert_eq!(found, AutoVisionSidecar::Attached(dir));
    }

    /// A trunk carrying its own tower is a reported no-op, never an attach:
    /// a sidecar beside it would be a second tower for one session.
    #[test]
    fn auto_on_a_trunk_with_its_own_tower_is_a_reported_no_op() {
        let store = temp_store("auto-own");
        let dir = install_tower(&store, "tower-a");
        let hidden = hidden_size_of(&dir);
        let found = resolve_vision_sidecar_auto(&store, true, ModelFamily::QwenGdnDense, hidden)
            .expect("own tower is not an error");
        let AutoVisionSidecar::TextOnly(reason) = &found else {
            panic!("expected TextOnly, got {found:?}");
        };
        assert!(
            reason.contains("already declares a vision tower"),
            "{reason}"
        );
    }

    /// Zero pairing towers: text-only carrying the SAME reason an explicit
    /// path is refused with, never an error. `auto` is best effort.
    #[test]
    fn auto_with_no_pairing_tower_runs_text_only_with_the_refusal_reason() {
        let store = temp_store("auto-zero");
        install_tower(&store, "tower-a");
        let found = resolve_vision_sidecar_auto(&store, false, ModelFamily::Gemma4, 5120)
            .expect("absence is not an error for auto");
        let AutoVisionSidecar::TextOnly(reason) = &found else {
            panic!("expected TextOnly, got {found:?}");
        };
        assert!(reason.contains("no installed vision-tower"), "{reason}");
    }

    /// Ambiguity refuses under `auto` exactly as it does for an explicit
    /// path: the revision pin is load-bearing and auto never picks.
    #[test]
    fn auto_refuses_two_pairing_towers_rather_than_picking() {
        let store = temp_store("auto-two");
        let dir_a = install_tower(&store, "tower-a");
        let hidden = hidden_size_of(&dir_a);
        install_tower(&store, "tower-b");
        let err = resolve_vision_sidecar_auto(&store, false, ModelFamily::QwenGdnDense, hidden)
            .unwrap_err();
        assert!(err.contains("2 installed vision towers"), "{err}");
    }
}
