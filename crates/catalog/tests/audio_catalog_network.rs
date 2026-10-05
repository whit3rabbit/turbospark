//! Exact source guard for initial audio profiles. Large payloads use immutable
//! LFS metadata; small files are downloaded and SHA-256 checked byte for byte.

use sha2::{Digest, Sha256};

fn github_bytes(url: &str, expected_size: u64) -> Vec<u8> {
    use std::io::Read;
    // Frontend provenance comes from GitHub; never attach the HF client's token.
    let response = reqwest::blocking::Client::new()
        .get(url)
        .send()
        .unwrap()
        .error_for_status()
        .unwrap();
    let mut bytes = Vec::new();
    response
        .take(expected_size + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes.len() as u64, expected_size);
    bytes
}

use turbospark_catalog::{
    AudioCatalog, CancelFlag, Catalog, Client, DownloadReceipt, RepoRef, Store, VerifiedSourceFile,
    VerifiedSourceGroup,
};

#[test]
#[ignore = "network: verifies immutable audio file sizes, LFS digests, and small asset bytes"]
fn pinned_audio_asset_graph_matches_immutable_upstream() {
    let catalog = AudioCatalog::embedded().unwrap();
    let client = Client::new();
    for profile in catalog.entries() {
        let repo = RepoRef::new(&profile.identity.repository, &profile.identity.revision);
        let url = format!(
            "https://huggingface.co/api/models/{}/revision/{}?blobs=true",
            repo.repo, repo.revision
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&client.get_bounded(&url, 4 * 1024 * 1024).unwrap()).unwrap();
        assert_eq!(metadata["sha"], repo.revision);
        assert_eq!(metadata["id"], repo.repo);
        let files = metadata["siblings"].as_array().unwrap();
        for asset in &profile.assets {
            let source = files
                .iter()
                .find(|f| f["rfilename"] == asset.path)
                .unwrap_or_else(|| panic!("{} missing {}", profile.identity.alias, asset.path));
            assert_eq!(source["size"].as_u64(), Some(asset.size), "{}", asset.path);
            if asset.size > 16 * 1024 * 1024 {
                assert_eq!(source["lfs"]["sha256"], asset.sha256, "{}", asset.path);
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
            }
        }
        if let Some(frontend) = &profile.frontend {
            let url = format!(
                "https://raw.githubusercontent.com/{}/{}/{}",
                frontend.repository, frontend.revision, frontend.license_path
            );
            let bytes = github_bytes(&url, frontend.license_size);
            assert_eq!(bytes.len() as u64, frontend.license_size);
            assert_eq!(
                format!("{:x}", Sha256::digest(bytes)),
                frontend.license_sha256
            );
            for asset in &frontend.resources {
                let url = format!(
                    "https://raw.githubusercontent.com/{}/{}/{}",
                    asset.repository, asset.revision, asset.path
                );
                let bytes = github_bytes(&url, asset.size);
                assert_eq!(bytes.len() as u64, asset.size);
                assert_eq!(format!("{:x}", Sha256::digest(bytes)), asset.sha256);
            }
        }
        println!(
            "{}: {} assets, {} bytes, exact source guard passed",
            profile.identity.alias,
            profile.assets.len(),
            profile.download_bytes
        );
    }
}

#[test]
#[ignore = "real checkpoint: TURBOSPARK_AUDIO_CHECKPOINTS points to a JSON map of exact cache directories"]
fn exact_cached_checkpoints_publish_resolve_verify_and_delete() {
    let paths = std::env::var("TURBOSPARK_AUDIO_CHECKPOINTS")
        .expect("set TURBOSPARK_AUDIO_CHECKPOINTS to exact cache path JSON");
    let paths: serde_json::Value = serde_json::from_slice(&std::fs::read(paths).unwrap()).unwrap();
    let catalog = AudioCatalog::embedded().unwrap();
    let root = std::env::temp_dir().join(format!("catalog-audio-real-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let result = std::panic::catch_unwind(|| {
        let store = Store::new(root.join("store"));
        for profile in catalog.entries() {
            let cached = &paths[&profile.identity.alias];
            assert_eq!(cached["repository"], profile.identity.repository);
            assert_eq!(cached["revision"], profile.identity.revision);
            let cached = std::path::Path::new(cached["path"].as_str().unwrap());
            let staging_root = root.join(format!("stage-{}", profile.identity.alias));
            let source_root = staging_root.join("source-0000");
            let plan = catalog.pinned_plan(&profile.identity).unwrap();
            let mut files = Vec::new();
            for asset in &profile.assets {
                let staged_path = source_root.join(&asset.path);
                std::fs::create_dir_all(staged_path.parent().unwrap()).unwrap();
                // Hard links preserve HF's immutable cache and avoid a second 9 GiB copy.
                std::fs::hard_link(
                    std::fs::canonicalize(cached.join(&asset.path)).unwrap(),
                    &staged_path,
                )
                .unwrap();
                files.push(VerifiedSourceFile {
                    path: asset.path.clone(),
                    staged_path,
                    size: asset.size,
                    sha256: asset.sha256.clone(),
                });
            }
            let receipt = DownloadReceipt {
                owner_id: plan.owner_id,
                staging_root,
                sources: vec![VerifiedSourceGroup {
                    role: plan.sources[0].role.clone(),
                    repo: plan.sources[0].repo.clone(),
                    files,
                }],
            };
            let installed = catalog
                .publish_verified(&store, &profile.identity, &receipt, &CancelFlag::new())
                .unwrap();
            assert_eq!(
                catalog.resolve(&store, &profile.identity).unwrap(),
                installed.path
            );
            assert_eq!(
                catalog.verify(&store, &profile.identity).unwrap(),
                installed.path
            );
            if profile.identity.alias == "whisper-base" {
                turbospark_catalog::verify_speech_install(&installed.path).unwrap();
            }
            catalog.delete(&store, &profile.identity).unwrap();
            assert!(!installed.path.exists());
            assert!(catalog.installed(&store).unwrap().is_empty());
            println!("{}: exact bytes, atomic receipt publication, resolve/full verify/delete passed; no inference run",profile.identity.alias);
        }
    });
    std::fs::remove_dir_all(root).unwrap();
    result.unwrap();
}

#[test]
fn cancelled_audio_download_does_not_publish_or_contact_upstream() {
    let root =
        std::env::temp_dir().join(format!("catalog-audio-pre-cancel-{}", std::process::id()));
    let store = Store::new(&root);
    let audio = AudioCatalog::embedded().unwrap();
    let profile = audio.get("kokoro-82m-bf16").unwrap();
    let cancel = CancelFlag::new();
    cancel.cancel();
    assert!(audio
        .install(
            &Client::new(),
            &Catalog::embedded().unwrap(),
            &store,
            &profile.identity,
            &mut |_| panic!("cancelled download must not report network progress"),
            &cancel
        )
        .unwrap_err()
        .contains("cancelled"));
    assert!(!root.exists());
}
