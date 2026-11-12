//! Exact source guard for the pinned embedding and OCR profiles, shaped like
//! `audio_catalog_network.rs`. LFS files are checked against the Hub's LFS
//! SHA-256; other files are downloaded at the pin and hashed. The branch is
//! on whether the Hub reports LFS metadata, not on size: the OCR
//! `tokenizer.json` is a 9.98 MB plain git blob.

use sha2::{Digest, Sha256};
use turbospark_catalog::{Client, EmbeddingCatalog, OcrCatalog, PinnedAsset, RepoRef};

fn check(client: &Client, alias: &str, repo: RepoRef, assets: &[PinnedAsset]) {
    let url = format!(
        "https://huggingface.co/api/models/{}/revision/{}?blobs=true",
        repo.repo, repo.revision
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&client.get_bounded(&url, 4 * 1024 * 1024).unwrap()).unwrap();
    assert_eq!(metadata["sha"], repo.revision);
    assert_eq!(metadata["id"], repo.repo);
    let files = metadata["siblings"].as_array().unwrap();
    let mut fetched = 0u64;
    for asset in assets {
        let source = files
            .iter()
            .find(|f| f["rfilename"] == asset.path)
            .unwrap_or_else(|| panic!("{alias} missing {}", asset.path));
        assert_eq!(source["size"].as_u64(), Some(asset.size), "{}", asset.path);
        if let Some(lfs) = source["lfs"]["sha256"].as_str() {
            assert_eq!(lfs, asset.sha256, "{}", asset.path);
        } else {
            let bytes = client
                .get_bounded(&repo.file_url(&asset.path), 16 * 1024 * 1024)
                .unwrap();
            assert_eq!(bytes.len() as u64, asset.size, "{}", asset.path);
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                asset.sha256,
                "{}",
                asset.path
            );
            fetched += asset.size;
        }
    }
    println!(
        "{alias}: {} assets, {} bytes, {fetched} bytes hashed locally, exact source guard passed",
        assets.len(),
        assets.iter().map(|a| a.size).sum::<u64>()
    );
}

#[test]
#[ignore = "network: verifies pinned embedding and OCR file sizes and digests"]
fn pinned_dir_profiles_match_immutable_upstream() {
    let client = Client::new();
    for p in EmbeddingCatalog::embedded().unwrap().profiles() {
        check(
            &client,
            &p.alias,
            RepoRef::new(&p.repository, &p.revision),
            &p.assets,
        );
    }
    for p in OcrCatalog::embedded().unwrap().profiles() {
        check(
            &client,
            &p.alias,
            RepoRef::new(&p.repository, &p.revision),
            &p.assets,
        );
    }
}
