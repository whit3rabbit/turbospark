//! Network rot guard for the MiniMax Music 3 profile pins.
//!
//! Run manually with:
//! `cargo test -p turbospark-catalog --test music_catalog_network --release -- --ignored --nocapture`
//! It reads repository trees only and downloads no checkpoint payloads.

use turbospark_catalog::{embedded_music_entries, Client, RepoRef};

#[test]
#[ignore = "network: checks the pinned MiniMax Music 3 repository file sets"]
fn pinned_music_profiles_still_have_their_exact_runtime_files() {
    let client = Client::new();
    let mut failures = Vec::new();
    for entry in embedded_music_entries().expect("embedded music catalog") {
        let repo = RepoRef::new(&entry.model_id, &entry.revision);
        let files = match client.file_list(&repo) {
            Ok(files) => files,
            Err(error) => {
                failures.push(format!("{}: cannot list {repo}: {error}", entry.alias));
                continue;
            }
        };
        let missing: Vec<_> = entry
            .required_files
            .iter()
            .filter(|required| !files.iter().any(|found| found == *required))
            .collect();
        if !missing.is_empty() {
            failures.push(format!("{} ({repo}) is missing {missing:?}", entry.alias));
        }
    }
    assert!(
        failures.is_empty(),
        "pinned music catalog drift:\n{}",
        failures.join("\n")
    );
}
