//! The catalog's rot guard: does every row still describe a real artifact?
//!
//! ```sh
//! cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture
//! ```
//!
//! **This is `arch_registry_network.rs`'s discipline applied to the catalog,
//! and for the same reason.** A table of strings about published files decays
//! silently: nothing fails when a repository is renamed, a sidecar list
//! changes, or a `main`-pinned GGUF is re-uploaded a quantization better. The
//! result is a table that reads as authoritative and sends a user into a
//! twenty-minute stream that 404s at the end.
//!
//! Costs a file list and a HEAD per row, seconds each, and downloads NOTHING.
//! That budget is deliberate: a guard that costs a download is a guard nobody
//! runs.
//!
//! Three things it checks, in increasing order of what they would cost to
//! find the hard way:
//!
//! 1. **Every sidecar named in the row EXISTS in the repository it names.**
//!    This is the `qwen38` failure (AGENTS.md Gotcha 47) turned into a
//!    seconds-long test rather than a re-stream.
//! 2. **The weights file exists and is the size the row records.** For a
//!    `main`-pinned row, the size is the ONLY fingerprint there is, so this
//!    is what notices a re-upload at all.
//! 3. **A GGUF row's architecture string is what the row's family implies.**
//!    Reads the header, which is a few MB off a 20 GB file.

use turbospark_catalog::{Catalog, Client, RepoRef, SourceKind};

/// How far a recorded `download_bytes` may sit from the published size before
/// it is a finding rather than drift.
///
/// **Two percent, because every row's figure was read off THIS test's own
/// HEAD requests rather than estimated from a listing page.** An earlier draft
/// carried round numbers and a 10% tolerance, which is a fingerprint that
/// cannot detect anything: `mixtral`'s "26 GB" sat 9.4% from its real
/// 28,448,468,384 and would have absorbed an entire re-quantization. If a row
/// here is ever added by hand, run this test and paste the published figure
/// back rather than widening the tolerance.
const SIZE_TOLERANCE: f64 = 0.02;

#[test]
#[ignore = "network: one file list and a HEAD per catalog row"]
fn every_row_still_names_files_that_exist_at_the_size_it_records() {
    let catalog = Catalog::embedded().expect("catalog");
    let client = Client::new();
    let mut findings = Vec::new();

    for entry in catalog.entries() {
        let weights = RepoRef::new(&entry.source.repo, &entry.source.revision);
        eprintln!("\n=== {} ({weights})", entry.alias);

        let files = match client.file_list(&weights) {
            Ok(f) => f,
            Err(e) => {
                findings.push(format!("{}: cannot list {weights}: {e}", entry.alias));
                continue;
            }
        };

        // The weights themselves.
        let (weight_url, present) = match (&entry.source.kind, &entry.source.file) {
            (SourceKind::Gguf, Some(file)) => {
                (weights.file_url(file), files.iter().any(|f| f == file))
            }
            _ => {
                let shards: Vec<&String> = files
                    .iter()
                    .filter(|f| f.ends_with(".safetensors"))
                    .collect();
                if shards.is_empty() {
                    findings.push(format!("{}: no .safetensors in {weights}", entry.alias));
                    continue;
                }
                (weights.file_url(shards[0]), true)
            }
        };
        if !present {
            findings.push(format!(
                "{}: {:?} is gone from {weights}",
                entry.alias,
                entry.source.file.as_deref().unwrap_or("?")
            ));
            continue;
        }

        // The size, which for a `main`-pinned row is the only fingerprint.
        let published: u64 = match entry.source.kind {
            SourceKind::Gguf => client
                .content_length(&weight_url)
                .ok()
                .flatten()
                .unwrap_or(0),
            SourceKind::Mlx => files
                .iter()
                .filter(|f| f.ends_with(".safetensors"))
                .filter_map(|f| client.content_length(&weights.file_url(f)).ok().flatten())
                .sum(),
        };
        if published > 0 {
            let drift = (published as f64 - entry.download_bytes as f64).abs()
                / entry.download_bytes.max(1) as f64;
            eprintln!(
                "  size      recorded {}, published {} ({:+.1}%)",
                entry.download_bytes,
                published,
                100.0 * (published as f64 - entry.download_bytes as f64)
                    / entry.download_bytes.max(1) as f64
            );
            if drift > SIZE_TOLERANCE {
                findings.push(format!(
                    "{}: download_bytes {} against a published {published} ({:.1}% out). \
                     On a row pinned at `main` this is what a re-upload looks like.",
                    entry.alias,
                    entry.download_bytes,
                    100.0 * drift
                ));
            }
        }

        // The sidecars, in the repository the row names for them.
        let sidecars = RepoRef::new(entry.sidecar_repo(), entry.sidecar_revision());
        match client.file_list(&sidecars) {
            Ok(have) => {
                let missing: Vec<&String> = entry
                    .sidecars
                    .files
                    .iter()
                    .filter(|want| !have.iter().any(|h| h == *want))
                    .collect();
                eprintln!(
                    "  sidecars  {} in {sidecars}{}",
                    entry.sidecars.files.len(),
                    if missing.is_empty() {
                        String::new()
                    } else {
                        format!(", MISSING {missing:?}")
                    }
                );
                if !missing.is_empty() {
                    findings.push(format!(
                        "{}: {sidecars} does not have {missing:?}. A pull would stream \
                         the weights and then fail on this.",
                        entry.alias
                    ));
                }
            }
            Err(e) => findings.push(format!("{}: cannot list {sidecars}: {e}", entry.alias)),
        }
    }

    assert!(
        findings.is_empty(),
        "the catalog has drifted from what is published:\n  {}",
        findings.join("\n  ")
    );
}

/// A GGUF row's architecture string, read off the real header.
///
/// Separate from the test above because it costs a few MB per row rather than
/// a HEAD, and because it is the check that would catch a repository swapping
/// which MODEL sits behind a filename -- the failure the size check cannot
/// see when two models happen to quantize to similar sizes.
#[test]
#[ignore = "network: reads the real GGUF header of every gguf row"]
fn every_gguf_row_still_declares_the_architecture_its_family_implies() {
    let catalog = Catalog::embedded().expect("catalog");
    let client = Client::new();
    let mut findings = Vec::new();

    for entry in catalog.entries() {
        let (SourceKind::Gguf, Some(file)) = (entry.source.kind, &entry.source.file) else {
            continue;
        };
        let weights = RepoRef::new(&entry.source.repo, &entry.source.revision);
        let report = match turbospark_catalog::probe(&client, &weights, Some(file), None) {
            Ok(r) => r,
            Err(e) => {
                findings.push(format!("{}: {e}", entry.alias));
                continue;
            }
        };
        let declared = report.architecture.as_deref().unwrap_or("<none>");
        let family = report.family.map(|f| f.as_str()).unwrap_or("<none>");
        eprintln!(
            "{:<12} declares {declared:<10} -> family {family} (row says {})",
            entry.alias, entry.family
        );
        if family != entry.family {
            findings.push(format!(
                "{}: the file's {declared:?} resolves to family {family}, but the row \
                 says {}",
                entry.alias, entry.family
            ));
        }
    }

    assert!(
        findings.is_empty(),
        "gguf rows disagree with their files:\n  {}",
        findings.join("\n  ")
    );
}
