//! Prints the `source`, `assets` and `required_files` of a pinned audio
//! catalog row (crates/catalog/src/audio_assets.json).
//!
//!     cargo run -p turbospark-catalog --example audio_pin -- \
//!         <owner/repo> <revision|main> [prefix ...]
//!
//! The revision resolves once to the 40-hex commit the Hub reports, and every
//! path, size and SHA-256 comes from that commit. LFS files carry their
//! SHA-256 in the Hub metadata. Every other file is downloaded and hashed, so
//! a row never holds a digest nobody computed. Prefixes select files by path
//! (a `Model.mlmodelc/` directory, a language folder). Without one, every
//! file except `.mlpackage/` sources is listed: Core ML loads the compiled
//! `.mlmodelc` beside them. Run the audio network test after adding a row.

use sha2::{Digest, Sha256};
use turbospark_catalog::{Client, RepoRef};

const USAGE: &str = "usage: audio_pin <owner/repo> <revision|main> [prefix ...]";

fn main() {
    if let Err(e) = run() {
        eprintln!("audio_pin: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let repo = args.next().ok_or(USAGE)?;
    let revision = args.next().ok_or(USAGE)?;
    let prefixes: Vec<String> = args.collect();
    let client = Client::new();
    let meta: serde_json::Value = serde_json::from_slice(&client.get_bounded(
        &RepoRef::new(&repo, &revision).api_url_with_sizes(),
        8 << 20,
    )?)
    .map_err(|e| format!("Hub metadata: {e}"))?;
    let sha = meta["sha"]
        .as_str()
        .filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or("Hub metadata has no 40-hex commit")?
        .to_string();
    let pinned = RepoRef::new(&repo, &sha);
    let siblings = meta["siblings"]
        .as_array()
        .ok_or("Hub metadata lists no files")?;
    let mut assets = Vec::new();
    for sibling in siblings {
        let path = sibling["rfilename"]
            .as_str()
            .ok_or("a listed file has no path")?;
        let selected = if prefixes.is_empty() {
            !path.contains(".mlpackage/")
        } else {
            prefixes.iter().any(|p| path.starts_with(p.as_str()))
        };
        if !selected {
            continue;
        }
        let size = sibling["size"]
            .as_u64()
            .ok_or_else(|| format!("{path} has no size"))?;
        let sha256 = match sibling["lfs"]["sha256"].as_str() {
            Some(digest) => digest.to_string(),
            None => {
                let limit = usize::try_from(size + 1).map_err(|e| e.to_string())?;
                let bytes = client.get_bounded(&pinned.file_url(path), limit)?;
                if bytes.len() as u64 != size {
                    return Err(format!(
                        "{path}: downloaded {} bytes, the Hub lists {size}",
                        bytes.len()
                    ));
                }
                format!("{:x}", Sha256::digest(&bytes))
            }
        };
        assets.push((path.to_string(), size, sha256));
    }
    if assets.is_empty() {
        return Err("no file matched".into());
    }
    assets.sort();
    let rows: Vec<_> = assets
        .iter()
        .map(|(path, size, sha256)| serde_json::json!({"path": path, "size": size, "sha256": sha256}))
        .collect();
    let required: Vec<_> = assets.iter().map(|(path, _, _)| path).collect();
    let out = serde_json::json!({
        "source": {"repository": repo, "revision": sha},
        "assets": rows,
        "required_files": required,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
    );
    Ok(())
}
