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

/// Short `family[:variant]` names (`qwen3.6:35b`) for curated aliases, the
/// way a user types a model in ollama. A separate table rather than a field
/// on [`CatalogEntry`] so the row schema the FFI and the app read is
/// unchanged: tags are a front-end spelling, and the alias stays the one
/// identity every install, receipt and server id is keyed on.
const EMBEDDED_TAGS: &str = include_str!("tags.json");

/// The schema version this build understands. A file declaring anything else
/// is refused rather than best-effort parsed: the failure mode of guessing is
/// a silently dropped field, and a dropped `sidecars.repo` is a 404 twenty
/// minutes into a stream.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
struct TagsFile {
    schema_version: u32,
    tags: BTreeMap<String, String>,
}

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
    /// Tag -> alias, from `tags.json`.
    tags: BTreeMap<String, String>,
}

impl Catalog {
    /// The curated table alone. Infallible in practice (the embedded file is
    /// covered by `tests/catalog.rs`) and still a `Result`, because a
    /// `panic!` inside a library is a worse diagnostic than a message.
    pub fn embedded() -> Result<Self, String> {
        let mut catalog = Catalog::default();
        catalog.merge_str(EMBEDDED, "embedded models.json", false)?;
        catalog.load_tags(EMBEDDED_TAGS, "embedded tags.json")?;
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

    /// A tag table is refused whole when any tag points at an alias that is
    /// not a row, or shadows an alias: a tag that silently resolved to a
    /// different model than the alias of the same spelling would be the
    /// worst kind of wrong install.
    fn load_tags(&mut self, text: &str, origin: &str) -> Result<(), String> {
        let file: TagsFile =
            serde_json::from_str(text).map_err(|e| format!("parsing {origin}: {e}"))?;
        if file.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "{origin}: schema_version {} is not {SCHEMA_VERSION}",
                file.schema_version
            ));
        }
        for (tag, alias) in &file.tags {
            if tag.is_empty() || tag.chars().any(char::is_whitespace) {
                return Err(format!("{origin}: tag {tag:?} is empty or has whitespace"));
            }
            if self.by_alias.contains_key(tag) {
                return Err(format!("{origin}: tag {tag:?} shadows an alias"));
            }
            if !self.by_alias.contains_key(alias) {
                return Err(format!(
                    "{origin}: tag {tag:?} names unknown alias {alias:?}"
                ));
            }
        }
        self.tags = file.tags;
        Ok(())
    }

    /// The row a user-typed name means: an alias first, then a tag. Matching
    /// is exact (case-insensitive only for tags), so a typo is a miss the
    /// caller reports rather than a near neighbour it installs.
    pub fn resolve_name(&self, name: &str) -> Option<&CatalogEntry> {
        if let Some(entry) = self.by_alias.get(name) {
            return Some(entry);
        }
        let lower = name.to_lowercase();
        self.tags
            .get(&lower)
            .and_then(|alias| self.by_alias.get(alias))
    }

    /// Every tag naming `alias`, shortest first, for display.
    pub fn tags_for(&self, alias: &str) -> Vec<&str> {
        let mut tags: Vec<&str> = self
            .tags
            .iter()
            .filter(|(_, a)| a.as_str() == alias)
            .map(|(t, _)| t.as_str())
            .collect();
        tags.sort_by_key(|t| (t.len(), *t));
        tags
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

    /// Number of entries currently in the catalog.
    pub fn len(&self) -> usize {
        self.by_alias.len()
    }

    /// Returns true if the catalog contains no entries.
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
                    || self.tags_for(&e.alias).iter().any(|t| t.contains(&needle))
                    || e.source.repo.to_lowercase().contains(&needle)
            })
            .collect()
    }
}
