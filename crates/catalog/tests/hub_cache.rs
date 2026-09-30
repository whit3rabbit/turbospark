use std::ffi::OsString;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use turbospark_catalog::{Catalog, Client, HubClient, HubQuery, HubSort, Machine, Refresh, Store};

static ENDPOINT_LOCK: Mutex<()> = Mutex::new(());
static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);

struct EndpointReset;

impl Drop for EndpointReset {
    fn drop(&mut self) {
        turbospark_catalog::set_hf_endpoint_override(None);
    }
}

struct AuthEnvironmentReset(Vec<(&'static str, Option<OsString>)>);

impl AuthEnvironmentReset {
    fn with_anonymous_home(home: &Path) -> Self {
        const VARIABLES: [&str; 5] = [
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_HOME",
            "HOME",
            "TURBOSPARK_HOME",
        ];
        let previous = VARIABLES
            .into_iter()
            .map(|name| (name, std::env::var_os(name)))
            .collect();
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
        std::env::set_var("HF_HOME", home);
        std::env::set_var("HOME", home);
        std::env::set_var("TURBOSPARK_HOME", home);
        Self(previous)
    }
}

impl Drop for AuthEnvironmentReset {
    fn drop(&mut self) {
        for (name, value) in self.0.drain(..) {
            if let Some(value) = value {
                std::env::set_var(name, value);
            } else {
                std::env::remove_var(name);
            }
        }
    }
}

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        let sequence = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "turbospark-hub-cache-{label}-{}-{sequence}",
            std::process::id()
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture_server(bodies: Vec<String>) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture server");
    listener
        .set_nonblocking(true)
        .expect("make fixture accept bounded");
    let address = listener.local_addr().expect("fixture address");
    let handle = thread::spawn(move || {
        let mut requests = Vec::new();
        for body in bodies {
            let deadline = Instant::now() + Duration::from_secs(2);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(accepted) => break accepted,
                    Err(error)
                        if error.kind() == ErrorKind::WouldBlock && Instant::now() < deadline =>
                    {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => return requests,
                }
            };
            stream
                .set_nonblocking(false)
                .expect("make accepted fixture stream blocking");
            let mut reader = BufReader::new(stream.try_clone().expect("clone fixture stream"));
            let mut request = String::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).expect("read request header");
                if line.is_empty() {
                    break;
                }
                let done = line == "\r\n" || line == "\n";
                request.push_str(&line);
                if done {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write fixture response");
            requests.push(request);
        }
        requests
    });
    (format!("http://{address}"), handle)
}

fn hub_payload(repo: &str, revision: &str, downloads: u64) -> String {
    serde_json::json!([{
        "id": repo,
        "sha": revision,
        "downloads": downloads,
        "likes": 1,
    }])
    .to_string()
}

fn cache_file(root: &Path) -> PathBuf {
    cache_files(root)
        .into_iter()
        .next()
        .expect("one query cache file")
}

fn cache_files(root: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(root.join("hub-cache"))
        .expect("hub cache directory")
        .map(|entry| entry.expect("cache entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect()
}

fn set_cache_age(path: &Path, age: Duration) {
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("read cache envelope"))
            .expect("parse cache envelope");
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after unix epoch")
        .as_secs();
    envelope["fetched_at"] = serde_json::json!(now.saturating_sub(age.as_secs()));
    std::fs::write(path, serde_json::to_vec(&envelope).unwrap()).expect("age cache envelope");
}

#[test]
fn fresh_queries_hit_cache_and_force_refresh_fetches_again() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("force");
    let first_payload = hub_payload("community/cache-model", &"a".repeat(40), 10);
    let forced_payload = hub_payload("community/cache-model", &"b".repeat(40), 20);
    let (endpoint, server) = fixture_server(vec![first_payload, forced_payload]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;

    let client = Client::with_timeout(Duration::from_secs(2));
    let hub = HubClient::new(
        &client,
        Catalog::embedded().unwrap(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(root.path()));
    let query = HubQuery::new("community/cache-model");

    let first = hub
        .search(&query, Refresh::AllowCache)
        .expect("first fetch");
    assert!(!first.from_cache);
    let repeated = hub
        .search(&query, Refresh::AllowCache)
        .expect("fresh cache hit");
    assert!(repeated.from_cache);
    assert_eq!(repeated.fetched_at, first.fetched_at);
    assert_eq!(
        repeated.entries[0]
            .live_target
            .as_ref()
            .unwrap()
            .repo
            .revision,
        "a".repeat(40)
    );

    let forced = hub.search(&query, Refresh::Force).expect("forced fetch");
    assert!(!forced.from_cache);
    assert_eq!(
        forced.entries[0]
            .live_target
            .as_ref()
            .unwrap()
            .repo
            .revision,
        "b".repeat(40)
    );
    assert_eq!(server.join().unwrap().len(), 2);
}

#[test]
fn cached_responses_are_partitioned_by_token_fingerprint_and_anonymous_client() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("token-partition");
    let anonymous_home = root.path().join("anonymous-home");
    std::fs::create_dir_all(&anonymous_home).unwrap();
    let _auth_environment = AuthEnvironmentReset::with_anonymous_home(&anonymous_home);
    let token_one = "hf_fixture_credential_one";
    let token_two = "hf_fixture_credential_two";
    let response_one = hub_payload("community/first-credential", &"1".repeat(40), 10);
    let response_two = hub_payload("community/second-credential", &"2".repeat(40), 20);
    let response_anonymous = hub_payload("community/anonymous-credential", &"3".repeat(40), 30);
    let (endpoint, server) = fixture_server(vec![response_one, response_two, response_anonymous]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;

    let client_one = Client::with_token(Some(token_one.to_string()));
    let client_two = Client::with_token(Some(token_two.to_string()));
    let anonymous_client = Client::with_token(None);
    assert!(anonymous_client.token().is_none());
    let catalog = Catalog::embedded().unwrap();
    let first_hub = HubClient::new(
        &client_one,
        catalog.clone(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(root.path()));
    let second_hub = HubClient::new(
        &client_two,
        catalog.clone(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(root.path()));
    let anonymous_hub = HubClient::new(
        &anonymous_client,
        catalog,
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(root.path()));
    let query = HubQuery::new("community/token-partition");

    let first = first_hub
        .search(&query, Refresh::AllowCache)
        .expect("first credential fetch");
    let second = second_hub
        .search(&query, Refresh::AllowCache)
        .expect("second credential fetch");
    let anonymous = anonymous_hub
        .search(&query, Refresh::AllowCache)
        .expect("anonymous credential partition fetch");
    let requests = server.join().unwrap();

    assert_eq!(requests.len(), 3, "each credential partition must fetch");
    assert!(requests[0]
        .to_ascii_lowercase()
        .contains(&format!("authorization: bearer {token_one}")));
    assert!(requests[1]
        .to_ascii_lowercase()
        .contains(&format!("authorization: bearer {token_two}")));
    assert!(!requests[2].to_ascii_lowercase().contains("authorization:"));
    assert!(!first.from_cache);
    assert!(!second.from_cache);
    assert!(!anonymous.from_cache);
    assert_eq!(first.entries[0].repo_id, "community/first-credential");
    assert_eq!(second.entries[0].repo_id, "community/second-credential");
    assert_eq!(
        anonymous.entries[0].repo_id,
        "community/anonymous-credential"
    );

    let cache_files = cache_files(root.path());
    assert_eq!(
        cache_files.len(),
        3,
        "credential partitions use distinct files"
    );
    for path in cache_files {
        let serialized_path = path.to_string_lossy().into_owned();
        let bytes = std::fs::read(path).unwrap();
        let serialized_body = String::from_utf8_lossy(&bytes);
        assert!(!serialized_path.contains(token_one));
        assert!(!serialized_path.contains(token_two));
        assert!(!serialized_body.contains(token_one));
        assert!(!serialized_body.contains(token_two));
    }
}

#[test]
fn cache_identity_includes_search_sort_and_limit_for_one_credential() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("request-identity");
    let responses = vec![
        hub_payload("community/request-identity-one", &"1".repeat(40), 1),
        hub_payload("community/request-identity-two", &"2".repeat(40), 2),
        hub_payload("community/request-identity-three", &"3".repeat(40), 3),
        hub_payload("community/request-identity-four", &"4".repeat(40), 4),
    ];
    let (endpoint, server) = fixture_server(responses);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;

    let client = Client::with_token(Some("hf_same_request_credential".to_string()));
    let hub = HubClient::new(
        &client,
        Catalog::embedded().unwrap(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(root.path()));
    let queries = [
        HubQuery::new("owner/request-a")
            .with_sort(HubSort::Downloads)
            .with_limit(12),
        HubQuery::new("owner/request-b")
            .with_sort(HubSort::Downloads)
            .with_limit(12),
        HubQuery::new("owner/request-b")
            .with_sort(HubSort::Likes)
            .with_limit(12),
        HubQuery::new("owner/request-b")
            .with_sort(HubSort::Likes)
            .with_limit(13),
    ];
    let expected_repositories = [
        "community/request-identity-one",
        "community/request-identity-two",
        "community/request-identity-three",
        "community/request-identity-four",
    ];

    let fetched: Vec<_> = queries
        .iter()
        .map(|query| {
            hub.search(query, Refresh::AllowCache)
                .expect("distinct request fetch")
        })
        .collect();
    for (page, expected) in fetched.iter().zip(expected_repositories) {
        assert!(!page.from_cache);
        assert_eq!(page.entries[0].repo_id, expected);
    }

    for (query, expected) in queries.iter().zip(expected_repositories) {
        let cached = hub
            .search(query, Refresh::AllowCache)
            .expect("same request cache hit");
        assert!(cached.from_cache);
        assert_eq!(cached.entries[0].repo_id, expected);
    }

    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 4, "each URL identity must fetch once");
    let request_lines: std::collections::HashSet<_> = requests
        .iter()
        .map(|request| request.lines().next().unwrap())
        .collect();
    assert_eq!(request_lines.len(), 4);
    assert_eq!(cache_files(root.path()).len(), 4);
}

#[test]
fn cache_identity_includes_hf_endpoint_for_the_same_query_and_token() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("endpoint-identity");
    let (endpoint_one, server_one) = fixture_server(vec![hub_payload(
        "community/endpoint-one",
        &"1".repeat(40),
        1,
    )]);
    let (endpoint_two, server_two) = fixture_server(vec![hub_payload(
        "community/endpoint-two",
        &"2".repeat(40),
        2,
    )]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint_one));
    let _reset = EndpointReset;

    let client = Client::with_token(Some("hf_same_endpoint_credential".to_string()));
    let hub = HubClient::new(
        &client,
        Catalog::embedded().unwrap(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(root.path()));
    let query = HubQuery::new("community/same-endpoint-query");

    let first = hub
        .search(&query, Refresh::AllowCache)
        .expect("first endpoint response");
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint_two));
    let second = hub
        .search(&query, Refresh::AllowCache)
        .expect("second endpoint response");

    assert!(!first.from_cache);
    assert!(!second.from_cache);
    assert_eq!(first.entries[0].repo_id, "community/endpoint-one");
    assert_eq!(second.entries[0].repo_id, "community/endpoint-two");
    assert_eq!(server_one.join().unwrap().len(), 1);
    assert_eq!(server_two.join().unwrap().len(), 1);
    let cache_files = cache_files(root.path());
    assert_eq!(cache_files.len(), 2);
    let cached_payloads: Vec<_> = cache_files
        .iter()
        .map(|path| {
            serde_json::from_slice::<serde_json::Value>(&std::fs::read(path).unwrap()).unwrap()
        })
        .collect();
    assert!(cached_payloads
        .iter()
        .any(|envelope| envelope["payload"][0]["id"] == "community/endpoint-one"));
    assert!(cached_payloads
        .iter()
        .any(|envelope| envelope["payload"][0]["id"] == "community/endpoint-two"));
}

#[test]
fn expired_search_refetches_but_trending_keeps_its_longer_window() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("expiry");
    let response = hub_payload("community/cache-model", &"c".repeat(40), 30);
    let (endpoint, server) = fixture_server(vec![response.clone(), response.clone()]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;

    let client = Client::with_timeout(Duration::from_secs(2));
    let hub = HubClient::new(
        &client,
        Catalog::embedded().unwrap(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(root.path()));
    let query = HubQuery::new("community/cache-model");

    let first = hub
        .search(&query, Refresh::AllowCache)
        .expect("search fetch");
    assert!(!first.from_cache);
    let search_cache = cache_file(root.path());
    set_cache_age(&search_cache, Duration::from_secs(15 * 60 - 2));
    let just_fresh = hub
        .search(&query, Refresh::AllowCache)
        .expect("search just below freshness window hits cache");
    assert!(just_fresh.from_cache);
    set_cache_age(&search_cache, Duration::from_secs(15 * 60 + 2));
    let just_expired = hub
        .search(&query, Refresh::AllowCache)
        .expect("search just above freshness window refetches");
    assert!(!just_expired.from_cache);
    assert_eq!(server.join().unwrap().len(), 2);

    let feed_root = TempRoot::new("feed-expiry");
    let feed_payload = hub_payload("community/trending-model", &"d".repeat(40), 40);
    let (endpoint, feed_server) = fixture_server(vec![feed_payload.clone(), feed_payload]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let feed_client = Client::with_timeout(Duration::from_secs(2));
    let feed_hub = HubClient::new(
        &feed_client,
        Catalog::embedded().unwrap(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(feed_root.path()));
    let first_feed = feed_hub.trending(Refresh::AllowCache).expect("feed fetch");
    assert!(!first_feed.from_cache);
    let feed_cache = cache_file(feed_root.path());
    set_cache_age(&feed_cache, Duration::from_secs(60 * 60 - 2));
    let cached_feed = feed_hub
        .trending(Refresh::AllowCache)
        .expect("feed just below freshness window hits cache");
    assert!(cached_feed.from_cache);
    assert!(cached_feed.fetched_at < first_feed.fetched_at);
    set_cache_age(&feed_cache, Duration::from_secs(60 * 60 + 2));
    let expired_feed = feed_hub
        .trending(Refresh::AllowCache)
        .expect("feed just above freshness window refetches");
    assert!(!expired_feed.from_cache);
    assert_eq!(feed_server.join().unwrap().len(), 2);
}

#[test]
fn cache_older_than_twice_search_freshness_is_deleted_before_refetch() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("stale-cleanup");
    let valid_response = hub_payload("community/stale-cache", &"a".repeat(40), 3);
    let (endpoint, server) = fixture_server(vec![valid_response, "not valid JSON".to_string()]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;

    let client = Client::with_timeout(Duration::from_secs(2));
    let hub = HubClient::new(
        &client,
        Catalog::embedded().unwrap(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(root.path()));
    let query = HubQuery::new("community/stale-cache");

    hub.search(&query, Refresh::AllowCache)
        .expect("initial fetch writes a cache");
    let stale_cache = cache_file(root.path());
    set_cache_age(&stale_cache, Duration::from_secs(31 * 60));

    assert!(hub.search(&query, Refresh::AllowCache).is_err());
    assert!(
        !stale_cache.exists(),
        "a stale cache beyond twice its freshness window must be deleted"
    );
    assert_eq!(server.join().unwrap().len(), 2);
}

#[test]
fn hostile_cached_metadata_is_discarded_and_fetched_again() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("hostile");
    let first_payload = hub_payload("community/first-safe", &"e".repeat(40), 1);
    let replacement_payload = hub_payload("community/refetched-safe", &"f".repeat(40), 2);
    let (endpoint, server) = fixture_server(vec![first_payload, replacement_payload]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;

    let client = Client::with_timeout(Duration::from_secs(2));
    let hub = HubClient::new(
        &client,
        Catalog::embedded().unwrap(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(root.path()));
    let query = HubQuery::new("community/hostile-cache");

    let first = hub
        .search(&query, Refresh::AllowCache)
        .expect("initial fetch");
    assert!(!first.from_cache);
    let path = cache_file(root.path());
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    envelope["payload"][0]["id"] = serde_json::json!("../escape");
    std::fs::write(&path, serde_json::to_vec(&envelope).unwrap()).unwrap();

    let refetched = hub
        .search(&query, Refresh::AllowCache)
        .expect("hostile cached row is refetched");
    assert!(!refetched.from_cache);
    assert_eq!(refetched.entries.len(), 1);
    assert_eq!(refetched.entries[0].repo_id, "community/refetched-safe");
    assert_eq!(server.join().unwrap().len(), 2);
}

#[test]
fn cache_write_failure_does_not_hide_a_valid_live_page() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("write-failure");
    std::fs::create_dir_all(root.path()).unwrap();
    let blocked_store_root = root.path().join("store-is-a-file");
    std::fs::write(&blocked_store_root, b"not a directory").unwrap();
    let response = hub_payload("community/cache-write-fallback", &"a".repeat(40), 5);
    let (endpoint, server) = fixture_server(vec![response]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;

    let client = Client::with_timeout(Duration::from_secs(2));
    let hub = HubClient::new(
        &client,
        Catalog::embedded().unwrap(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(blocked_store_root));

    let page = hub
        .search(
            &HubQuery::new("community/cache-write-fallback"),
            Refresh::AllowCache,
        )
        .expect("cache write failure preserves live result");
    assert!(!page.from_cache);
    assert_eq!(page.entries[0].repo_id, "community/cache-write-fallback");
    assert_eq!(server.join().unwrap().len(), 1);
}
