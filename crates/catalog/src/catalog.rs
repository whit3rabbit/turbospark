//! Loading the curated table, and merging a user's own rows over it.
//!
//! **Row-admission rule, the same one `repack::arch_registry` states for its
//! architecture strings: every row in `models.json` names a repository and a
//! revision that were actually streamed and run on real hardware here.** The
//! [`Status`](crate::Status) field records how far, and `gates` names the test
//! targets that assert it. A model somebody expects to work is not a row; a
//! model somebody ran is.
//!
//! That rule is what makes the table worth more than a README list, and it is
//! also why the table is SMALL and will stay small. The answer for everything
//! else is `probe`, which reads a header and decides, rather than a row
//! somebody added optimistically.
//!
//! A user override at `$TURBOSPARK_HOME/models.json` merges by alias, so a
//! local row can be added without a rebuild and an existing alias can be
//! repointed. The override is held to the same structural validation, so a
//! malformed one fails at load rather than a quarter of an hour into a walk.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::entry::CatalogEntry;

/// The embedded curated table.
const EMBEDDED: &str = include_str!("models.json");

/// The schema version this build understands. A file declaring anything else
/// is refused rather than best-effort parsed: the failure mode of guessing is
/// a silently dropped field, and a dropped `sidecars.repo` is a 404 twenty
/// minutes into a stream.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CatalogFile {
    schema_version: u32,
    models: Vec<CatalogEntry>,
}

/// The resolved table: curated rows with any user rows merged over them.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    by_alias: BTreeMap<String, CatalogEntry>,
    /// Aliases the user override replaced or added, for `list` to mark.
    overridden: BTreeMap<String, bool>,
}

impl Catalog {
    /// The curated table alone. Infallible in practice (the embedded file is
    /// covered by `tests/catalog.rs`) and still a `Result`, because a
    /// `panic!` inside a library is a worse diagnostic than a message.
    pub fn embedded() -> Result<Self, String> {
        let mut catalog = Catalog::default();
        catalog.merge_str(EMBEDDED, "embedded models.json", false)?;
        Ok(catalog)
    }

    /// The curated table with `$TURBOSPARK_HOME/models.json` merged over it,
    /// if that file exists. A missing override is not an error; an unreadable
    /// or malformed one is.
    pub fn load(home: &Path) -> Result<Self, String> {
        let mut catalog = Self::embedded()?;
        let user = home.join("models.json");
        match std::fs::read_to_string(&user) {
            Ok(text) => catalog.merge_str(&text, &user.display().to_string(), true)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("reading {}: {e}", user.display())),
        }
        Ok(catalog)
    }

    fn merge_str(&mut self, text: &str, origin: &str, user: bool) -> Result<(), String> {
        let file: CatalogFile =
            serde_json::from_str(text).map_err(|e| format!("parsing {origin}: {e}"))?;
        if file.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "{origin}: schema_version {} is not {SCHEMA_VERSION}",
                file.schema_version
            ));
        }
        for entry in file.models {
            entry.validate().map_err(|e| format!("{origin}: {e}"))?;
            if user {
                self.overridden.insert(entry.alias.clone(), true);
            }
            self.by_alias.insert(entry.alias.clone(), entry);
        }
        Ok(())
    }

    /// The row for `alias`, or `None`.
    pub fn get(&self, alias: &str) -> Option<&CatalogEntry> {
        self.by_alias.get(alias)
    }

    /// Every row, alias order.
    pub fn entries(&self) -> impl Iterator<Item = &CatalogEntry> {
        self.by_alias.values()
    }

    /// Whether `alias` came from the user's override rather than the
    /// embedded table. `list` marks these, so a surprising row is traceable
    /// to the file that introduced it.
    pub fn is_user_row(&self, alias: &str) -> bool {
        self.overridden.contains_key(alias)
    }

    pub fn len(&self) -> usize {
        self.by_alias.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_alias.is_empty()
    }

    /// Aliases whose text contains `needle`, case-insensitively, matched
    /// against the alias, the display name and the repo. Enough for
    /// `list --filter`; it is deliberately not a search over Hugging Face,
    /// which is a different feature with a different failure mode.
    pub fn find(&self, needle: &str) -> Vec<&CatalogEntry> {
        let needle = needle.to_lowercase();
        self.by_alias
            .values()
            .filter(|e| {
                e.alias.to_lowercase().contains(&needle)
                    || e.name.to_lowercase().contains(&needle)
                    || e.source.repo.to_lowercase().contains(&needle)
            })
            .collect()
    }
}
