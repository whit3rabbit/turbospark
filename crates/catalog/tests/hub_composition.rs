use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use turbospark_catalog::{
    Catalog, Client, EntryKind, Evidence, FitVerdict, HubClient, HubQuery, Machine, Refresh,
    RepoRef, Store,
};

static ENDPOINT_LOCK: Mutex<()> = Mutex::new(());
static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);

struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        let sequence = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "turbospark-hub-composition-{}-{sequence}",
            std::process::id()
        )))
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct EndpointReset;

impl Drop for EndpointReset {
    fn drop(&mut self) {
        turbospark_catalog::set_hf_endpoint_override(None);
    }
}

fn fixture_server(bodies: Vec<String>) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture server");
    let address = listener.local_addr().expect("fixture address");
    let handle = thread::spawn(move || {
        bodies
            .into_iter()
            .map(|body| {
                let (mut stream, _) = listener.accept().expect("accept hub request");
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .expect("set request timeout");
                let mut request = String::new();
                BufReader::new(stream.try_clone().expect("clone fixture stream"))
                    .read_line(&mut request)
                    .expect("read request line");
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write fixture response");
                request.trim_end().to_string()
            })
            .collect()
    });
    (format!("http://{address}"), handle)
}

fn hub_row(repo: &str, revision: &str, downloads: u64, likes: u64) -> serde_json::Value {
    serde_json::json!({
        "id": repo,
        "sha": revision,
        "downloads": downloads,
        "likes": likes,
    })
}

#[test]
fn search_and_trending_compose_sources_without_implicit_probes() {
    let _endpoint_lock = ENDPOINT_LOCK.lock().expect("endpoint lock");
    let catalog = Catalog::embedded().expect("embedded catalog");
    let repo = "mlx-community/Qwen3.8-27B-4bit";
    let bundled_entries: Vec<_> = catalog
        .entries()
        .filter(|entry| entry.kind == EntryKind::Model && entry.source.repo == repo)
        .collect();
    assert!(
        bundled_entries.len() > 1,
        "fixture repo exercises bundled aliases"
    );

    let search_revision = "a".repeat(40);
    let special_revision = "b".repeat(40);
    let trend_revision = "c".repeat(40);
    let search_payload = serde_json::json!([
        hub_row(repo, &search_revision, 120, 8),
        hub_row("community/search-only", &search_revision, 90, 2),
        { "id": "malformed/no-commit", "sha": "main", "downloads": 1, "likes": 0 }
    ]);
    let encoded_query_payload =
        serde_json::json!([hub_row("community/query-encoding", &special_revision, 7, 1)]);
    let trending_payload = serde_json::json!([
        hub_row(repo, &trend_revision, 400, 20),
        hub_row("community/trending-only", &trend_revision, 350, 11)
    ]);
    let (endpoint, server) = fixture_server(vec![
        search_payload.to_string(),
        encoded_query_payload.to_string(),
        trending_payload.to_string(),
    ]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;
    let cache_root = TempRoot::new();

    let client = Client::with_timeout(Duration::from_secs(2));
    let hub = HubClient::new(
        &client,
        catalog.clone(),
        Machine {
            physical_bytes: 32 * 1024 * 1024 * 1024,
            chip: "fixture chip".to_string(),
            ..Machine::default()
        },
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(&cache_root.0));

    let search = hub
        .search(&HubQuery::new("Qwen3.8"), Refresh::AllowCache)
        .expect("fixture search");
    let merged = search
        .entries
        .iter()
        .find(|entry| entry.repo_id == repo)
        .expect("live and bundled repo merged");
    assert!(merged
        .provenance
        .contains(&turbospark_catalog::EntryProvenance::LiveHub));
    assert!(merged
        .provenance
        .contains(&turbospark_catalog::EntryProvenance::Bundled));
    assert_eq!(merged.bundled_targets.len(), bundled_entries.len());
    assert_eq!(
        merged.live_target.as_ref().unwrap().repo,
        RepoRef::new(repo, &search_revision)
    );
    assert_eq!(
        merged.live_target.as_ref().unwrap().fit.verdict,
        FitVerdict::Unknown.as_str()
    );
    assert_eq!(merged.live_target.as_ref().unwrap().fit.counted_bytes, None);
    assert!(!merged.live_target.as_ref().unwrap().fit.is_estimate);
    for entry in &bundled_entries {
        let target = merged
            .bundled_targets
            .iter()
            .find(|target| target.alias == entry.alias)
            .expect("curated action retained");
        assert_eq!(target.repo, RepoRef::new(repo, &entry.source.revision));
        assert_eq!(target.file, entry.source.file);
        assert_eq!(target.install_args, vec![entry.alias.clone()]);
        assert_eq!(target.evidence, Evidence::of(entry.status).as_str());
        assert_eq!(target.fit.mapped_bytes, Some(entry.install_bytes));
    }
    assert!(search
        .rejected
        .iter()
        .any(|row| row.entry == "malformed/no-commit"));
    let search_only = search
        .entries
        .iter()
        .find(|entry| entry.repo_id == "community/search-only")
        .expect("unmatched live result retained");
    assert!(search_only.bundled_targets.is_empty());
    assert_eq!(
        search_only.provenance,
        vec![turbospark_catalog::EntryProvenance::LiveHub]
    );

    let encoded = hub
        .search(&HubQuery::new("Qwen3.8&limit=999"), Refresh::Force)
        .expect("encoded fixture search");
    assert_eq!(encoded.entries.len(), 1);

    let trending = hub.trending(Refresh::AllowCache).expect("fixture feed");
    let feed_merged = trending
        .entries
        .iter()
        .find(|entry| entry.repo_id == repo)
        .expect("feed repo merged");
    assert_eq!(feed_merged.bundled_targets.len(), bundled_entries.len());
    assert_eq!(
        feed_merged.live_target.as_ref().unwrap().repo.revision,
        trend_revision
    );
    assert!(trending.entries.iter().any(|entry| {
        entry.repo_id == "community/trending-only"
            && entry.provenance == vec![turbospark_catalog::EntryProvenance::LiveHub]
    }));
    assert!(trending.entries.iter().any(|entry| {
        entry.provenance == vec![turbospark_catalog::EntryProvenance::Bundled]
            && !entry.bundled_targets.is_empty()
    }));
    assert!(trending
        .entries
        .iter()
        .flat_map(|entry| &entry.bundled_targets)
        .all(|target| catalog
            .get(&target.alias)
            .is_some_and(|entry| entry.kind == EntryKind::Model)));

    let requests = server.join().expect("fixture server joins");
    assert_eq!(requests.len(), 3, "only explicit list requests were sent");
    assert!(requests
        .iter()
        .all(|request| request.starts_with("GET /api/models?")));
    let encoded_url = requests[1]
        .split_whitespace()
        .nth(1)
        .expect("request target");
    let parsed = reqwest::Url::parse(&format!("http://fixture{encoded_url}"))
        .expect("parse fixture request target");
    let pairs: Vec<_> = parsed.query_pairs().collect();
    assert_eq!(
        pairs.iter().find(|(key, _)| key == "search").unwrap().1,
        "Qwen3.8&limit=999"
    );
    assert_eq!(pairs.iter().filter(|(key, _)| key == "limit").count(), 1);
}
