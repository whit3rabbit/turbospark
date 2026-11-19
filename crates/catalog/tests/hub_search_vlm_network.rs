//! Vision-language checkpoints are filed under `image-text-to-text`, which a
//! `text-generation` query never returns. This port runs their text trunk, so
//! search merges a second listing; a LoRA adapter on such a base must surface
//! too.
//!
//! ```sh
//! cargo test -p turbospark-catalog --test hub_search_vlm_network -- --ignored --nocapture
//! ```

use std::time::Duration;

use turbospark_catalog::{Catalog, Client, HubClient, HubQuery, HubSort, Machine, Refresh, Store};

struct TempRoot(std::path::PathBuf);

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "network: two Hub list requests"]
fn search_finds_the_pi_adapted_mlx_build_and_its_lora_adapter() {
    let root = TempRoot(
        std::env::temp_dir().join(format!("turbospark-hub-vlm-network-{}", std::process::id())),
    );
    let client = Client::with_timeout(Duration::from_secs(30));
    let hub = HubClient::new(
        &client,
        Catalog::embedded().expect("embedded catalog"),
        Machine {
            physical_bytes: 32 * 1024 * 1024 * 1024,
            chip: "network-test".to_string(),
            ..Machine::default()
        },
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(&root.0));
    let page = hub
        .search(
            &HubQuery::new("Qwen3.8-27B-pi-adapted")
                .with_limit(20)
                .with_vision_language(true)
                .with_sort(HubSort::Downloads),
            Refresh::Force,
        )
        .expect("live search");
    let ids: Vec<String> = page
        .entries
        .iter()
        .map(|e| e.repo_id.to_ascii_lowercase())
        .collect();
    for want in [
        "tozuo/qwen3.8-27b-pi-adapted-mlx-4bit",
        "harness-hub/qwen3.8-27b-pi-adapted",
    ] {
        assert!(ids.iter().any(|id| id == want), "{want} not in {ids:?}");
    }
}
