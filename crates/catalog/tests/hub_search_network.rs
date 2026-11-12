//! Does the live Hub still answer the search the app depends on?
//!
//! ```sh
//! cargo test -p turbospark-catalog --test hub_search_network -- --ignored --nocapture
//! ```
//!
//! `hub_validation.rs` proves the parser against fixtures, and a fixture
//! cannot notice the Hub renaming `expand[]=gguf`'s `architecture` key or
//! dropping a field. This reads the real response, in the same spirit as
//! `catalog_network.rs`: a table of facts about someone else's service rots
//! silently. Costs one list request per format plus one repository check, and
//! downloads no weights.

use std::time::Duration;

use turbospark_catalog::{
    arch_hint, check_repo, ArchHint, Catalog, CheckState, Client, HubClient, HubFormat, HubQuery,
    HubSort, Machine, Refresh, RepoRef, Store,
};

struct TempRoot(std::path::PathBuf);

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn hub<'a>(client: &'a Client, root: &TempRoot) -> HubClient<'a> {
    HubClient::new(
        client,
        Catalog::embedded().expect("embedded catalog"),
        Machine {
            physical_bytes: 32 * 1024 * 1024 * 1024,
            chip: "network-test".to_string(),
            ..Machine::default()
        },
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(&root.0))
}

#[test]
#[ignore = "network: two Hub list requests and one repository check"]
fn the_live_hub_returns_the_metadata_the_badges_read() {
    let root = TempRoot(std::env::temp_dir().join(format!(
        "turbospark-hub-search-network-{}",
        std::process::id()
    )));
    let client = Client::with_timeout(Duration::from_secs(30));
    let hub = hub(&client, &root);

    let gguf = hub
        .search(
            &HubQuery::new("qwen")
                .with_limit(20)
                .with_sort(HubSort::Downloads)
                .with_format(HubFormat::Gguf),
            Refresh::Force,
        )
        .expect("live GGUF search");
    let live: Vec<_> = gguf
        .entries
        .iter()
        .filter_map(|entry| entry.live_target.as_ref().map(|live| (entry, live)))
        .collect();
    assert!(
        !live.is_empty(),
        "the Hub answered a popular query with nothing"
    );
    assert!(
        gguf.rejected.is_empty(),
        "the validator rejected live rows: {:?}",
        gguf.rejected
    );
    assert!(
        live.iter().all(|(_, live)| live.meta.gguf),
        "filter=gguf must return only GGUF rows"
    );
    assert!(
        live.iter()
            .any(|(_, live)| live.meta.gguf_architecture.is_some()),
        "expand[]=gguf stopped returning an architecture; the badge has nothing to read"
    );
    for (entry, live) in &live {
        println!(
            "{:<60} arch={:<14} hint={:?} gated={}",
            entry.repo_id,
            live.meta.gguf_architecture.as_deref().unwrap_or("-"),
            arch_hint(&live.meta),
            live.meta.gated
        );
    }

    let mlx = hub
        .search(
            &HubQuery::new("qwen 4bit")
                .with_limit(20)
                .with_sort(HubSort::Downloads)
                .with_format(HubFormat::Mlx),
            Refresh::Force,
        )
        .expect("live MLX search");
    let mlx_live: Vec<_> = mlx
        .entries
        .iter()
        .filter_map(|entry| entry.live_target.as_ref())
        .collect();
    assert!(!mlx_live.is_empty());
    assert!(
        mlx_live.iter().any(|live| live.meta.model_type.is_some()),
        "expand[]=config stopped returning a model_type"
    );

    // One real check on the first ungated row whose architecture this port
    // recognizes. Any state but a transport failure is a pass: this asserts
    // the plumbing, and which repository runs today is not this test's claim.
    let pick = live.iter().find(|(_, live)| {
        !live.meta.gated && matches!(arch_hint(&live.meta), ArchHint::Recognized)
    });
    if let Some((entry, live)) = pick {
        let hint = live
            .meta
            .base_model
            .as_deref()
            .map(|base| RepoRef::new(base, "main"));
        let check = check_repo(
            &client,
            &RepoRef::new(&entry.repo_id, "main"),
            hint.as_ref(),
            None,
        );
        println!(
            "check {}: {:?} reason={:?} file={:?} sidecar={:?}",
            entry.repo_id,
            check.state,
            check.reason,
            check.report.as_ref().and_then(|r| r.file.as_deref()),
            check.sidecar_repo.as_ref().map(|r| r.repo.as_str()),
        );
        assert!(
            matches!(check.state, CheckState::Runnable | CheckState::Refused),
            "a public repository could not be checked: {:?} {:?}",
            check.state,
            check.reason
        );
        assert!(check.report.is_some());
        assert!(
            !check.variants.is_empty(),
            "a GGUF repository lists its files"
        );
    } else {
        println!("no ungated recognized-architecture GGUF row in this page; check skipped");
    }
}
