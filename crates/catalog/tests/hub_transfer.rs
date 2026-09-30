use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use turbospark_catalog::{
    CancelFlag, Catalog, Client, HubClient, HubDownloadProgress, Machine, PinnedArtifactPlan,
    PinnedSourceFile, PinnedSourceGroup, RepoRef,
};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
static ENDPOINT_LOCK: Mutex<()> = Mutex::new(());

enum FixtureResponse {
    Json(String),
    Body(Vec<u8>),
    NotFound,
}

fn fixture_server(responses: Vec<FixtureResponse>) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture server");
    let address = listener.local_addr().expect("fixture address");
    listener
        .set_nonblocking(true)
        .expect("set listener nonblocking");
    let handle = thread::spawn(move || {
        let mut requests = Vec::new();
        for response in responses {
            let started = Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(accepted) => break accepted,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && started.elapsed() < Duration::from_secs(2) =>
                    {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        return requests;
                    }
                    Err(error) => panic!("accept hub request: {error}"),
                }
            };
            stream
                .set_nonblocking(false)
                .expect("set accepted stream blocking");
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .expect("set request timeout");
            let mut request = String::new();
            let mut reader = BufReader::new(stream.try_clone().expect("clone fixture stream"));
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
            let response = match response {
                    FixtureResponse::Json(body) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                    .into_bytes(),
                    FixtureResponse::Body(body) => {
                        let mut response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .into_bytes();
                        response.extend(body);
                        response
                    }
                    FixtureResponse::NotFound => {
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                    }
                };
            stream.write_all(&response).expect("write fixture response");
            requests.push(request);
        }
        requests
    });
    (format!("http://{address}"), handle)
}

struct TempRoot(std::path::PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "turbospark-hub-transfer-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create transfer fixture root");
        Self(path)
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

fn set_endpoint(endpoint: String) -> EndpointReset {
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    EndpointReset
}

fn hub_for(client: &Client) -> HubClient<'_> {
    HubClient::new(
        client,
        Catalog::embedded().expect("embedded catalog"),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
}

fn source_group(repo: &str, role: &str, files: Vec<PinnedSourceFile>) -> PinnedSourceGroup {
    PinnedSourceGroup {
        role: role.to_string(),
        repo: RepoRef::new(repo, REVISION),
        files,
    }
}

fn source_file(
    path: &str,
    expected_size: Option<u64>,
    expected_sha256: Option<&str>,
) -> PinnedSourceFile {
    PinnedSourceFile {
        path: path.to_string(),
        expected_size,
        expected_sha256: expected_sha256.map(str::to_string),
    }
}

fn metadata(repo: &str, files: Vec<serde_json::Value>) -> String {
    metadata_at_revision(repo, REVISION, files)
}

fn metadata_at_revision(repo: &str, revision: &str, files: Vec<serde_json::Value>) -> String {
    serde_json::json!({ "id": repo, "sha": revision, "siblings": files }).to_string()
}

fn listed_file(path: &str, size: u64, sha256: Option<&str>) -> serde_json::Value {
    let mut file = serde_json::json!({ "rfilename": path, "size": size });
    if let Some(sha256) = sha256 {
        file["lfs"] = serde_json::json!({ "sha256": sha256 });
    }
    file
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn plan(owner_id: &str, sources: Vec<PinnedSourceGroup>) -> PinnedArtifactPlan {
    PinnedArtifactPlan {
        owner_id: owner_id.to_string(),
        sources,
    }
}

fn assert_staging_parent_empty(path: &std::path::Path) {
    assert_eq!(
        std::fs::read_dir(path)
            .expect("read staging parent")
            .count(),
        0,
        "failed transfer left a staging root behind"
    );
}

#[test]
fn transfer_preserves_opaque_ids_and_stages_exact_files_without_group_collisions() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let path = "weights/shared.bin";
    let bytes_a = b"alpha".to_vec();
    let bytes_b = b"bravo".to_vec();
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/first",
            vec![
                listed_file(path, bytes_a.len() as u64, None),
                listed_file("extra.bin", 5, None),
            ],
        )),
        FixtureResponse::Json(metadata(
            "owner/second",
            vec![listed_file(path, bytes_b.len() as u64, None)],
        )),
        FixtureResponse::Body(bytes_a.clone()),
        FixtureResponse::Body(bytes_b.clone()),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_transfer_success".to_string()));
    let hub = hub_for(&client);
    let root = TempRoot::new("success");
    let plan = plan(
        "opaque:owner/id/../must-not-be-a-path",
        vec![
            source_group(
                "owner/first",
                "role with spaces: adapter/alpha",
                vec![source_file(path, Some(bytes_a.len() as u64), None)],
            ),
            source_group(
                "owner/second",
                "role#two: control/beta",
                vec![source_file(path, Some(bytes_b.len() as u64), None)],
            ),
        ],
    );
    let mut progress = Vec::<HubDownloadProgress>::new();

    let receipt = hub
        .download_pinned_artifacts(
            &plan,
            &root.0,
            &mut |update| progress.push(update),
            &CancelFlag::new(),
        )
        .expect("all exact files transfer");

    assert_eq!(receipt.owner_id, "opaque:owner/id/../must-not-be-a-path");
    assert_eq!(receipt.sources.len(), 2);
    assert_eq!(receipt.sources[0].role, "role with spaces: adapter/alpha");
    assert_eq!(receipt.sources[1].role, "role#two: control/beta");
    assert_eq!(receipt.sources[0].repo.repo, "owner/first");
    assert_eq!(receipt.sources[1].repo.repo, "owner/second");
    assert_eq!(receipt.sources[0].files.len(), 1);
    assert_eq!(receipt.sources[1].files.len(), 1);
    assert_eq!(receipt.sources[0].files[0].path, path);
    assert_eq!(receipt.sources[1].files[0].path, path);
    assert_eq!(receipt.sources[0].files[0].size, bytes_a.len() as u64);
    assert_eq!(receipt.sources[1].files[0].size, bytes_b.len() as u64);
    assert_eq!(receipt.sources[0].files[0].sha256, sha256_hex(&bytes_a));
    assert_eq!(receipt.sources[1].files[0].sha256, sha256_hex(&bytes_b));
    assert_eq!(
        std::fs::read(&receipt.sources[0].files[0].staged_path).unwrap(),
        bytes_a
    );
    assert_eq!(
        std::fs::read(&receipt.sources[1].files[0].staged_path).unwrap(),
        bytes_b
    );
    assert_ne!(
        receipt.sources[0].files[0].staged_path,
        receipt.sources[1].files[0].staged_path
    );
    assert_eq!(
        receipt.staging_root.parent(),
        Some(std::fs::canonicalize(&root.0).unwrap().as_path())
    );
    assert!(progress.iter().any(|update| update.completed_files == 2));
    assert_eq!(progress.last().unwrap().completed_bytes, 10);

    let requests = fixture.join().expect("fixture server joins");
    assert_eq!(requests.len(), 4);
    assert!(
        requests
            .iter()
            .filter(|request| request.starts_with("GET /owner/"))
            .count()
            == 2
    );
    assert!(requests
        .iter()
        .all(|request| !request.contains("extra.bin")));
}

#[test]
fn missing_or_mismatched_source_is_rejected_before_staging() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/missing",
            vec![listed_file("other.bin", 3, None)],
        )),
        FixtureResponse::Json(metadata(
            "owner/mismatch",
            vec![listed_file("weights.bin", 4, None)],
        )),
        FixtureResponse::Json(metadata_at_revision(
            "owner/stale",
            "f".repeat(40).as_str(),
            vec![listed_file("weights.bin", 4, None)],
        )),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_transfer_preflight".to_string()));
    let hub = hub_for(&client);
    let root = TempRoot::new("preflight");

    let missing = plan(
        "owner",
        vec![source_group(
            "owner/missing",
            "missing",
            vec![source_file("wanted.bin", Some(3), None)],
        )],
    );
    assert!(hub
        .download_pinned_artifacts(&missing, &root.0, &mut |_| {}, &CancelFlag::new())
        .is_err());
    assert_staging_parent_empty(&root.0);

    let mismatch = plan(
        "owner",
        vec![source_group(
            "owner/mismatch",
            "mismatch",
            vec![source_file("weights.bin", Some(5), None)],
        )],
    );
    assert!(hub
        .download_pinned_artifacts(&mismatch, &root.0, &mut |_| {}, &CancelFlag::new())
        .is_err());
    assert_staging_parent_empty(&root.0);

    let stale = plan(
        "owner",
        vec![source_group(
            "owner/stale",
            "stale",
            vec![source_file("weights.bin", Some(4), None)],
        )],
    );
    assert!(hub
        .download_pinned_artifacts(&stale, &root.0, &mut |_| {}, &CancelFlag::new())
        .is_err());
    assert_staging_parent_empty(&root.0);

    assert_eq!(fixture.join().unwrap().len(), 3);
}

#[test]
fn extra_repository_siblings_are_not_staged_or_requested() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/extras",
            vec![
                listed_file("selected.bin", 4, None),
                listed_file("unselected.bin", 7, None),
            ],
        )),
        FixtureResponse::Body(b"keep".to_vec()),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_transfer_extras".to_string()));
    let hub = hub_for(&client);
    let root = TempRoot::new("extras");
    let plan = plan(
        "owner",
        vec![source_group(
            "owner/extras",
            "one",
            vec![source_file("selected.bin", Some(4), None)],
        )],
    );

    let receipt = hub
        .download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new())
        .expect("only selected path transfers");

    assert_eq!(receipt.sources[0].files.len(), 1);
    assert!(!receipt
        .staging_root
        .join("source-0000/unselected.bin")
        .exists());
    let requests = fixture.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests
        .iter()
        .all(|request| !request.contains("unselected.bin")));
}

#[test]
fn transfer_keeps_one_hub_endpoint_snapshot_for_preflight_and_all_files() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/endpoint-snapshot",
            vec![
                listed_file("first.bin", 3, None),
                listed_file("second.bin", 4, None),
            ],
        )),
        FixtureResponse::Body(b"one".to_vec()),
        FixtureResponse::Body(b"two2".to_vec()),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_transfer_endpoint_snapshot".to_string()));
    let hub = hub_for(&client);
    let root = TempRoot::new("endpoint-snapshot");
    let plan = plan(
        "owner",
        vec![source_group(
            "owner/endpoint-snapshot",
            "snapshot-role",
            vec![
                source_file("first.bin", Some(3), None),
                source_file("second.bin", Some(4), None),
            ],
        )],
    );

    let receipt = hub
        .download_pinned_artifacts(
            &plan,
            &root.0,
            &mut |update| {
                if update.completed_files == 1 && update.current_path.is_none() {
                    turbospark_catalog::set_hf_endpoint_override(Some(
                        "http://127.0.0.1:9".to_string(),
                    ));
                }
            },
            &CancelFlag::new(),
        )
        .expect("all files use the endpoint that supplied the pinned metadata");

    assert_eq!(receipt.sources[0].files.len(), 2);
    assert_eq!(
        std::fs::read(&receipt.sources[0].files[1].staged_path).unwrap(),
        b"two2"
    );
    assert_eq!(fixture.join().unwrap().len(), 3);
}

#[test]
fn partial_transfer_failure_removes_the_entire_staging_set() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/partial",
            vec![
                listed_file("first.bin", 4, None),
                listed_file("second.bin", 6, None),
            ],
        )),
        FixtureResponse::Body(b"done".to_vec()),
        FixtureResponse::NotFound,
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_transfer_partial".to_string()));
    let hub = hub_for(&client);
    let root = TempRoot::new("partial");
    let caller_file = root.0.join("caller-owned.txt");
    std::fs::write(&caller_file, b"preserve caller-owned content").unwrap();
    let plan = plan(
        "owner",
        vec![source_group(
            "owner/partial",
            "partial-role",
            vec![
                source_file("first.bin", Some(4), None),
                source_file("second.bin", Some(6), None),
            ],
        )],
    );

    assert!(hub
        .download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new())
        .is_err());
    assert_eq!(
        std::fs::read(&caller_file).unwrap(),
        b"preserve caller-owned content"
    );
    assert_eq!(std::fs::read_dir(&root.0).unwrap().count(), 1);
    assert_eq!(fixture.join().unwrap().len(), 3);
}

#[test]
fn streamed_size_and_pinned_digest_failures_remove_staging_root() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let metadata_for_size = metadata("owner/size", vec![listed_file("size.bin", 4, None)]);
    let bytes_for_digest = b"good";
    let metadata_for_digest = metadata(
        "owner/digest",
        vec![listed_file(
            "digest.bin",
            4,
            Some(&sha256_hex(bytes_for_digest)),
        )],
    );
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata_for_size),
        FixtureResponse::Body(b"bad".to_vec()),
        FixtureResponse::Json(metadata_for_digest),
        FixtureResponse::Body(b"evil".to_vec()),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_transfer_integrity".to_string()));
    let hub = hub_for(&client);
    let root = TempRoot::new("integrity");
    let size_plan = plan(
        "owner",
        vec![source_group(
            "owner/size",
            "size",
            vec![source_file("size.bin", Some(4), None)],
        )],
    );
    assert!(hub
        .download_pinned_artifacts(&size_plan, &root.0, &mut |_| {}, &CancelFlag::new())
        .is_err());
    assert_staging_parent_empty(&root.0);

    let digest_plan = plan(
        "owner",
        vec![source_group(
            "owner/digest",
            "digest",
            vec![source_file(
                "digest.bin",
                Some(4),
                Some(&sha256_hex(bytes_for_digest)),
            )],
        )],
    );
    assert!(hub
        .download_pinned_artifacts(&digest_plan, &root.0, &mut |_| {}, &CancelFlag::new())
        .is_err());
    assert_staging_parent_empty(&root.0);
    assert_eq!(fixture.join().unwrap().len(), 4);
}

#[test]
fn cancellation_during_a_file_stream_removes_staging_root() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let body = vec![b'x'; 128 * 1024];
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/cancel",
            vec![listed_file("large.bin", body.len() as u64, None)],
        )),
        FixtureResponse::Body(body),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_transfer_cancel".to_string()));
    let hub = hub_for(&client);
    let root = TempRoot::new("cancel");
    let plan = plan(
        "owner",
        vec![source_group(
            "owner/cancel",
            "cancel-role",
            vec![source_file("large.bin", Some(128 * 1024), None)],
        )],
    );
    let cancel = CancelFlag::new();
    let cancel_on_progress = cancel.clone();

    let result = hub.download_pinned_artifacts(
        &plan,
        &root.0,
        &mut |update| {
            if update.current_path.is_some() && update.completed_bytes > 0 {
                cancel_on_progress.cancel();
            }
        },
        &cancel,
    );

    assert!(
        result.is_err(),
        "cancelled transfer must not return a receipt"
    );
    assert_staging_parent_empty(&root.0);
    assert_eq!(fixture.join().unwrap().len(), 2);
}

#[test]
fn cancellation_between_files_skips_the_next_request_and_removes_staging_root() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/cancel-between",
            vec![
                listed_file("first.bin", 4, None),
                listed_file("second.bin", 6, None),
            ],
        )),
        FixtureResponse::Body(b"done".to_vec()),
        FixtureResponse::NotFound,
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_transfer_cancel_between".to_string()));
    let hub = hub_for(&client);
    let root = TempRoot::new("cancel-between");
    let plan = plan(
        "owner",
        vec![source_group(
            "owner/cancel-between",
            "cancel-between-role",
            vec![
                source_file("first.bin", Some(4), None),
                source_file("second.bin", Some(6), None),
            ],
        )],
    );
    let cancel = CancelFlag::new();
    let cancel_on_progress = cancel.clone();

    let result = hub.download_pinned_artifacts(
        &plan,
        &root.0,
        &mut |update| {
            if update.completed_files == 1 && update.current_path.is_none() {
                cancel_on_progress.cancel();
            }
        },
        &cancel,
    );

    assert!(
        result.is_err(),
        "cancelled transfer must not return a receipt"
    );
    assert_staging_parent_empty(&root.0);
    let requests = fixture.join().unwrap();
    assert_eq!(
        requests.len(),
        2,
        "the second file request must not be issued"
    );
}
