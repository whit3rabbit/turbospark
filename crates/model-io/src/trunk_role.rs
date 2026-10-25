//! The marker that makes a `.gturbo` install a headless trunk.
//!
//! A headless trunk is a language-model trunk with no LM head, driven by
//! another model that supplies its input rows and reads its final hidden
//! state (Fish Audio Drama 3's text backbone, under its audio install). Its
//! manifest is an ordinary `QwenGdnMoe` manifest, so without this marker a
//! chat open would accept it and fail late on the missing head. The marker
//! makes the refusal early and by name.
//!
//! **Presence alone refuses a chat open.** [`has_marker`] answers on the
//! file existing, not on it parsing: a truncated or hand-edited marker must
//! still keep a chat open out. Only the headless open reads it fully
//! ([`read`]), and that read refuses anything it does not recognize.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ModelError;

/// File name of the marker, beside `manifest.json`.
pub const TRUNK_ROLE_FILE: &str = "trunk_role.json";

/// The only kind this crate writes.
pub const HEADLESS_TRUNK_KIND: &str = "headless_trunk";

/// `trunk_role.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrunkRoleRecord {
    /// Always [`HEADLESS_TRUNK_KIND`] for a marker this crate wrote.
    pub kind: String,
    /// The model that drives this trunk, as its root `config.json`
    /// `model_type` spells it (`fish_drama3`).
    pub owner: String,
}

/// True when `dir` carries a trunk-role marker, readable or not.
pub fn has_marker(dir: &Path) -> bool {
    dir.join(TRUNK_ROLE_FILE).exists()
}

/// Reads and validates the marker in `dir`.
pub fn read(dir: &Path) -> Result<TrunkRoleRecord, ModelError> {
    let path = dir.join(TRUNK_ROLE_FILE);
    let text = std::fs::read_to_string(&path).map_err(|e| ModelError::IoFailed {
        call: format!("read {}", path.display()),
        detail: e.to_string(),
    })?;
    let record: TrunkRoleRecord =
        serde_json::from_str(&text).map_err(|e| ModelError::IndexCorrupt {
            detail: format!("{TRUNK_ROLE_FILE}: {e}"),
        })?;
    if record.kind != HEADLESS_TRUNK_KIND {
        return Err(ModelError::IndexCorrupt {
            detail: format!(
                "{TRUNK_ROLE_FILE} kind {:?} is not {HEADLESS_TRUNK_KIND:?}",
                record.kind
            ),
        });
    }
    if record.owner.is_empty() {
        return Err(ModelError::IndexCorrupt {
            detail: format!("{TRUNK_ROLE_FILE} names no owner"),
        });
    }
    Ok(record)
}

/// Writes a headless-trunk marker naming `owner` into `dir`.
///
/// Written through a temporary file and a rename, so a reader never sees a
/// half-written marker. The repack writer calls this before it streams the
/// trunk; the streamed writer carries existing files into its staging
/// directory, so the marker is published in the same rename as the manifest.
pub fn write_headless(dir: &Path, owner: &str) -> Result<(), ModelError> {
    let record = TrunkRoleRecord {
        kind: HEADLESS_TRUNK_KIND.to_string(),
        owner: owner.to_string(),
    };
    let json = serde_json::to_vec_pretty(&record).map_err(|e| ModelError::IndexCorrupt {
        detail: format!("{TRUNK_ROLE_FILE}: {e}"),
    })?;
    let tmp = dir.join(format!(".{TRUNK_ROLE_FILE}.tmp"));
    let path = dir.join(TRUNK_ROLE_FILE);
    std::fs::write(&tmp, json).map_err(|e| ModelError::IoFailed {
        call: format!("write {}", tmp.display()),
        detail: e.to_string(),
    })?;
    std::fs::rename(&tmp, &path).map_err(|e| ModelError::IoFailed {
        call: format!("rename {}", path.display()),
        detail: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "trunk_role_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn round_trips_a_headless_marker() {
        let dir = scratch("round_trip");
        assert!(!has_marker(&dir));
        write_headless(&dir, "fish_drama3").unwrap();
        assert!(has_marker(&dir));
        let record = read(&dir).unwrap();
        assert_eq!(record.kind, HEADLESS_TRUNK_KIND);
        assert_eq!(record.owner, "fish_drama3");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_marker_still_counts_as_present_and_fails_to_read() {
        let dir = scratch("corrupt");
        std::fs::write(dir.join(TRUNK_ROLE_FILE), b"{\"kind\": \"headl").unwrap();
        assert!(has_marker(&dir), "presence alone must refuse a chat open");
        assert!(read(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_an_unknown_kind_or_an_empty_owner() {
        let dir = scratch("kinds");
        std::fs::write(
            dir.join(TRUNK_ROLE_FILE),
            br#"{"kind": "vision_tower", "owner": "fish_drama3"}"#,
        )
        .unwrap();
        assert!(read(&dir).is_err());
        std::fs::write(
            dir.join(TRUNK_ROLE_FILE),
            br#"{"kind": "headless_trunk", "owner": ""}"#,
        )
        .unwrap();
        assert!(read(&dir).is_err());
        std::fs::write(
            dir.join(TRUNK_ROLE_FILE),
            br#"{"kind": "headless_trunk", "owner": "x", "extra": 1}"#,
        )
        .unwrap();
        assert!(read(&dir).is_err(), "unknown fields are refused");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
