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
    ObservedBody(Vec<u8>, std::sync::mpsc::Sender<()>),
    Retry(u16, u64, std::sync::mpsc::Sender<()>),
    NotFound,
    Raw(Vec<u8>),
    Head(String),
    SlowBody(
        Vec<u8>,
        std::sync::mpsc::Sender<()>,
        std::sync::Arc<std::sync::atomic::AtomicU64>,
    ),
    Source(
        std::sync::Arc<Vec<u8>>,
        String,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
    ),
}

fn fixture_server(responses: Vec<FixtureResponse>) -> (String, thread::JoinHandle<Vec<String>>) {
    fixture_server_with_idle_timeout(responses, Duration::from_secs(2))
}

fn fixture_server_with_idle_timeout(
    responses: Vec<FixtureResponse>,
    idle_timeout: Duration,
) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture server");
    let address = listener.local_addr().expect("fixture address");
    listener
        .set_nonblocking(true)
        .expect("set listener nonblocking");
    let handle = thread::spawn(move || {
        let mut requests = Vec::new();
        let mut head_files = std::collections::HashMap::<String, (u64, String)>::new();
        for response in responses {
            let (mut stream, request) = loop {
                let started = Instant::now();
                let (mut stream, _) = loop {
                    match listener.accept() {
                        Ok(accepted) => break accepted,
                        Err(error)
                            if error.kind() == std::io::ErrorKind::WouldBlock
                                && started.elapsed() < idle_timeout =>
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
                if request.starts_with("HEAD ") && !matches!(&response, FixtureResponse::Head(_)) {
                    let path = request.split_whitespace().nth(1).unwrap();
                    let (size, digest) = head_files
                        .get(path)
                        .expect("LFS fixture HEAD has listed metadata");
                    stream.write_all(format!("HTTP/1.1 200 OK\r\nX-Linked-Size: {size}\r\nX-Linked-ETag: \"{digest}\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
                    continue;
                }
                break (stream, request);
            };
            let metadata_body = match &response {
                FixtureResponse::Json(body) => Some(body.as_str()),
                FixtureResponse::Source(_, body, _) => Some(body.as_str()),
                _ => None,
            };
            if let Some(body) = metadata_body {
                let parsed: serde_json::Value = serde_json::from_str(body).unwrap();
                for file in parsed["siblings"].as_array().unwrap() {
                    if let Some(digest) = file["lfs"]["sha256"].as_str() {
                        let path = format!(
                            "/{}/resolve/{}/{}",
                            parsed["id"].as_str().unwrap(),
                            parsed["sha"].as_str().unwrap(),
                            file["rfilename"].as_str().unwrap()
                        );
                        head_files
                            .insert(path, (file["size"].as_u64().unwrap(), digest.to_string()));
                    }
                }
            }
            let observed = match &response {
                FixtureResponse::ObservedBody(_, ready) => Some(ready.clone()),
                _ => None,
            };
            let response = match response {
                    FixtureResponse::Json(body) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                    .into_bytes(),
                    FixtureResponse::Body(body) | FixtureResponse::ObservedBody(body, _) => {
                        let range = request.lines().find_map(|line| line.to_ascii_lowercase().strip_prefix("range: bytes=").map(str::to_owned));
                        let status = if let Some(range) = range {
                            let (start, end) = range.split_once('-').unwrap();
                            format!("206 Partial Content\r\nContent-Range: bytes {start}-{end}/{}", body.len())
                        } else {
                            "200 OK".to_string()
                        };
                        let mut response = format!(
                            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .into_bytes();
                        response.extend(body);
                        response
                    }
                    FixtureResponse::Source(body, metadata, fail_first_range) => {
                    if request.starts_with("GET /api/") {
                        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{metadata}", metadata.len()).into_bytes()
                    } else {
                        let range = request.lines().find_map(|line| line.to_ascii_lowercase().strip_prefix("range: bytes=").map(str::to_owned)).expect("ranged source request");
                        let (start, end) = range.split_once('-').unwrap();
                        let start: usize = start.parse().unwrap();
                        let end: usize = end.parse().unwrap();
                        if start == 0 && fail_first_range.load(std::sync::atomic::Ordering::Acquire) {
                            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                        } else {
                            let bytes = &body[start..=end];
                            let mut wire = format!("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len(), bytes.len()).into_bytes();
                            wire.extend_from_slice(bytes);
                            wire
                        }
                    }
                }
                FixtureResponse::SlowBody(body, ready, sent) => {
                    stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).as_bytes()).unwrap();
                    for (index, chunk) in body.chunks(1024).enumerate() {
                        if stream.write_all(chunk).is_err() { break; }
                        sent.fetch_add(chunk.len() as u64, std::sync::atomic::Ordering::Release);
                        if index == 0 { ready.send(()).unwrap(); }
                        thread::sleep(Duration::from_millis(10));
                    }
                    Vec::new()
                }
                FixtureResponse::Head(headers) => format!("HTTP/1.1 200 OK\r\n{headers}Content-Length: 0\r\nConnection: close\r\n\r\n").into_bytes(),
                FixtureResponse::Retry(status, seconds, ready) => {
                    stream.write_all(format!("HTTP/1.1 {status} Retry\r\nRetry-After: {seconds}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
                    ready.send(()).unwrap();
                    Vec::new()
                }
                FixtureResponse::Raw(body) => body,
                    FixtureResponse::NotFound => {
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                    }
                };
            stream.write_all(&response).expect("write fixture response");
            if let Some(ready) = observed {
                ready.send(()).unwrap();
            }
            if request.starts_with("GET ") {
                requests.push(request);
            }
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
    let mut file = serde_json::json!({ "rfilename": path, "size": size, "blobId": "a".repeat(40) });
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
            .filter(|entry| entry.as_ref().unwrap().file_name() != ".download-cache")
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
    assert!(
        requests
            .iter()
            .filter(|request| request.starts_with("GET /owner/"))
            .all(|request| request.to_ascii_lowercase().contains("range: bytes=")),
        "pinned payloads must use the ranged downloader"
    );
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
    assert_eq!(
        std::fs::read_dir(&root.0)
            .unwrap()
            .filter(|entry| entry.as_ref().unwrap().file_name() != ".download-cache")
            .count(),
        1
    );
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

#[test]
fn small_non_lfs_pin_is_verified_from_exact_bytes_before_transfer() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let bytes = b"{\"tokenizer\":true}".to_vec();
    let digest = sha256_hex(&bytes);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/small",
            vec![listed_file("tokenizer.json", bytes.len() as u64, None)],
        )),
        FixtureResponse::Body(bytes.clone()),
        FixtureResponse::Body(bytes.clone()),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_small".into()));
    let hub = hub_for(&client);
    let root = TempRoot::new("small");
    let plan = plan(
        "small-owner",
        vec![source_group(
            "owner/small",
            "config",
            vec![source_file(
                "tokenizer.json",
                Some(bytes.len() as u64),
                Some(&digest),
            )],
        )],
    );
    let receipt = hub
        .download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new())
        .expect("non-LFS SHA is verified from bounded exact-pin GET");
    assert_eq!(receipt.sources[0].files[0].sha256, digest);
    assert_eq!(
        std::fs::read(&receipt.sources[0].files[0].staged_path).unwrap(),
        bytes
    );
    let requests = fixture.join().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(requests[1].starts_with(&format!(
        "GET /owner/small/resolve/{REVISION}/tokenizer.json"
    )));
}

fn retry_plan() -> PinnedArtifactPlan {
    plan(
        "retry-owner",
        vec![source_group(
            "owner/retry",
            "weights",
            vec![
                source_file("first.bin", Some(4), Some(&sha256_hex(b"done"))),
                source_file("second.bin", Some(4), Some(&sha256_hex(b"next"))),
            ],
        )],
    )
}

fn retry_metadata() -> String {
    metadata(
        "owner/retry",
        vec![
            listed_file("first.bin", 4, Some(&sha256_hex(b"done"))),
            listed_file("second.bin", 4, Some(&sha256_hex(b"next"))),
        ],
    )
}

fn cached_ranges(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    fn visit(path: &std::path::Path, found: &mut Vec<std::path::PathBuf>) {
        if !path.is_dir() {
            return;
        }
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, found);
            } else if path
                .extension()
                .is_some_and(|extension| extension == "range")
            {
                found.push(path);
            }
        }
    }
    let mut found = Vec::new();
    visit(&root.join(".download-cache"), &mut found);
    found
}

fn assert_progress(updates: &[HubDownloadProgress], bytes: u64, files: u32) {
    assert!(!updates.is_empty());
    for update in updates {
        assert_eq!(update.total_bytes, bytes);
        assert_eq!(update.total_files, files);
        assert!(update.completed_bytes <= bytes);
        assert!(update.completed_files <= files);
    }
    for pair in updates.windows(2) {
        assert!(pair[0].completed_bytes <= pair[1].completed_bytes);
        assert!(pair[0].completed_files <= pair[1].completed_files);
    }
    assert_eq!(updates.last().unwrap().completed_bytes, bytes);
    assert_eq!(updates.last().unwrap().completed_files, files);
}

#[test]
fn pinned_retry_process_worker() {
    let Ok(root) = std::env::var("TURBOSPARK_PINNED_RETRY_TEST_ROOT") else {
        return;
    };
    let _reset = set_endpoint(std::env::var("TURBOSPARK_PINNED_RETRY_TEST_ENDPOINT").unwrap());
    let cancel = CancelFlag::new();
    let first_process = std::env::var("TURBOSPARK_PINNED_RETRY_TEST_PHASE").unwrap() == "interrupt";
    let client = Client::with_token(Some("hf_restart_fixture".into()));
    let hub = hub_for(&client);
    let mut updates = Vec::new();
    let result = hub.download_pinned_artifacts(
        &retry_plan(),
        std::path::Path::new(&root),
        &mut |update| {
            if first_process && update.completed_files == 1 {
                cancel.cancel();
            }
            updates.push(update);
        },
        &cancel,
    );
    if first_process {
        assert!(matches!(
            result,
            Err(turbospark_catalog::HubError::Cancelled)
        ));
        assert_eq!(cached_ranges(std::path::Path::new(&root)).len(), 1);
        assert_staging_parent_empty(std::path::Path::new(&root));
    } else {
        let receipt = result.expect("fresh process reuses verified first file and finishes second");
        assert_progress(&updates, 8, 2);
        assert_eq!(
            std::fs::read(&receipt.sources[0].files[0].staged_path).unwrap(),
            b"done"
        );
        assert_eq!(
            std::fs::read(&receipt.sources[0].files[1].staged_path).unwrap(),
            b"next"
        );
    }
}

#[test]
fn verified_ranges_survive_interruption_and_a_real_process_restart() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, fixture) = fixture_server_with_idle_timeout(
        vec![
            FixtureResponse::Json(retry_metadata()),
            FixtureResponse::Body(b"done".to_vec()),
            FixtureResponse::Json(retry_metadata()),
            FixtureResponse::Body(b"next".to_vec()),
        ],
        Duration::from_secs(30),
    );
    let root = TempRoot::new("process-restart");
    for phase in ["interrupt", "resume"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "pinned_retry_process_worker", "--nocapture"])
            .env("TURBOSPARK_PINNED_RETRY_TEST_ROOT", &root.0)
            .env("TURBOSPARK_PINNED_RETRY_TEST_ENDPOINT", &endpoint)
            .env("TURBOSPARK_PINNED_RETRY_TEST_PHASE", phase)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child {phase} failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let requests = fixture.join().unwrap();
    assert_eq!(requests.len(), 4);
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.starts_with("GET /owner/retry/resolve/")
                && request.contains("first.bin"))
            .count(),
        1,
        "completed first file must not be requested after process restart"
    );
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.starts_with("GET /api/"))
            .count(),
        2,
        "each process must revalidate live exact source"
    );
}

#[test]
fn pause_between_ranges_waits_and_cancel_wakes_paused_transfer() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(retry_metadata()),
        FixtureResponse::Body(b"done".to_vec()),
        FixtureResponse::Body(b"next".to_vec()),
    ]);
    let _reset = set_endpoint(endpoint);
    let root = TempRoot::new("pause-resume");
    let flag = CancelFlag::new();
    let (ready, waiting) = std::sync::mpsc::channel();
    let worker_flag = flag.clone();
    let destination = root.0.clone();
    let worker = thread::spawn(move || {
        let client = Client::with_token(Some("hf_pause_fixture".into()));
        let mut updates = Vec::new();
        let receipt = hub_for(&client)
            .download_pinned_artifacts(
                &retry_plan(),
                &destination,
                &mut |update| {
                    if update.completed_files == 1 && update.current_path.is_none() {
                        assert!(worker_flag.pause());
                        ready.send(()).unwrap();
                    }
                    updates.push(update);
                },
                &worker_flag,
            )
            .unwrap();
        (receipt, updates)
    });
    waiting.recv_timeout(Duration::from_secs(5)).unwrap();
    thread::sleep(Duration::from_millis(50));
    assert!(
        !worker.is_finished(),
        "paused transfer must wait for shared flag resume"
    );
    assert_eq!(
        cached_ranges(&root.0).len(),
        1,
        "pause keeps the completed range"
    );
    assert!(flag.resume());
    let (_, updates) = worker.join().unwrap();
    assert_progress(&updates, 8, 2);
    assert_eq!(fixture.join().unwrap().len(), 3);

    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(retry_metadata()),
        FixtureResponse::Body(b"done".to_vec()),
    ]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let root = TempRoot::new("pause-cancel");
    let flag = CancelFlag::new();
    let (ready, waiting) = std::sync::mpsc::channel();
    let worker_flag = flag.clone();
    let destination = root.0.clone();
    let worker = thread::spawn(move || {
        let client = Client::with_token(Some("hf_pause_fixture".into()));
        hub_for(&client).download_pinned_artifacts(
            &retry_plan(),
            &destination,
            &mut |update| {
                if update.completed_files == 1 {
                    assert!(worker_flag.pause());
                    ready.send(()).unwrap();
                }
            },
            &worker_flag,
        )
    });
    waiting.recv_timeout(Duration::from_secs(5)).unwrap();
    flag.cancel();
    assert!(matches!(
        worker.join().unwrap(),
        Err(turbospark_catalog::HubError::Cancelled)
    ));
    assert_staging_parent_empty(&root.0);
    assert_eq!(cached_ranges(&root.0).len(), 1);
    assert_eq!(fixture.join().unwrap().len(), 2);
}

#[test]
fn corrupted_range_refetches_and_whole_file_sha_failure_invalidates_only_poisoned_source() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(retry_metadata()),
        FixtureResponse::Body(b"done".to_vec()),
        FixtureResponse::Body(b"evil".to_vec()),
        FixtureResponse::Json(retry_metadata()),
        FixtureResponse::Body(b"next".to_vec()),
        FixtureResponse::Json(retry_metadata()),
        FixtureResponse::Body(b"done".to_vec()),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_corruption_fixture".into()));
    let hub = hub_for(&client);
    let root = TempRoot::new("poison");
    let installed = root.0.join("existing-install");
    std::fs::write(&installed, b"published bytes").unwrap();
    let failed =
        hub.download_pinned_artifacts(&retry_plan(), &root.0, &mut |_| {}, &CancelFlag::new());
    assert!(matches!(
        failed,
        Err(turbospark_catalog::HubError::TransferIntegrity {
            rule: "downloaded_sha256_matches_pinned_source",
            ..
        })
    ));
    let ranges = cached_ranges(&root.0);
    assert_eq!(
        ranges.len(),
        1,
        "only the poisoned second source cache is invalidated"
    );
    assert_eq!(std::fs::read(&installed).unwrap(), b"published bytes");
    assert_eq!(
        std::fs::read_dir(&root.0)
            .unwrap()
            .filter(|entry| entry.as_ref().unwrap().file_name() != ".download-cache")
            .count(),
        1
    );
    let receipt = hub
        .download_pinned_artifacts(&retry_plan(), &root.0, &mut |_| {}, &CancelFlag::new())
        .expect("retry fetches corrected second file while reusing first");
    assert_eq!(
        std::fs::read(&receipt.sources[0].files[1].staged_path).unwrap(),
        b"next"
    );
    let mut corrupt = std::fs::read(&ranges[0]).unwrap();
    let original_len = corrupt.len();
    *corrupt.last_mut().unwrap() ^= 1;
    std::fs::write(&ranges[0], &corrupt).unwrap();
    assert_eq!(
        std::fs::metadata(&ranges[0]).unwrap().len(),
        original_len as u64,
        "same-size corruption must be caught by the cached range digest"
    );
    let receipt = hub
        .download_pinned_artifacts(&retry_plan(), &root.0, &mut |_| {}, &CancelFlag::new())
        .expect("corruption forces a fresh ranged request");
    assert_eq!(
        std::fs::read(&receipt.sources[0].files[0].staged_path).unwrap(),
        b"done"
    );
    assert_eq!(std::fs::read(&installed).unwrap(), b"published bytes");
    assert_eq!(fixture.join().unwrap().len(), 7);
}

#[test]
fn changed_remote_validator_and_source_pin_cannot_reuse_previous_ranges() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let file = |blob: char| {
        let mut file = listed_file("same.bin", 4, None);
        file["blobId"] = serde_json::json!(blob.to_string().repeat(40));
        file
    };
    let new_revision = "b".repeat(40);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata("owner/change", vec![file('a')])),
        FixtureResponse::Body(b"old!".to_vec()),
        FixtureResponse::Json(metadata("owner/change", vec![file('b')])),
        FixtureResponse::Body(b"new!".to_vec()),
        FixtureResponse::Json(metadata_at_revision(
            "owner/change",
            &new_revision,
            vec![file('b')],
        )),
        FixtureResponse::Body(b"pin!".to_vec()),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_changed_fixture".into()));
    let hub = hub_for(&client);
    let root = TempRoot::new("changed");
    let mut plan = plan(
        "same-owner",
        vec![source_group(
            "owner/change",
            "role",
            vec![source_file("same.bin", Some(4), None)],
        )],
    );
    for (index, bytes) in [b"old!", b"new!", b"pin!"].iter().enumerate() {
        if index == 2 {
            plan.sources[0].repo.revision = new_revision.clone();
        }
        let receipt = hub
            .download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new())
            .unwrap();
        assert_eq!(
            std::fs::read(&receipt.sources[0].files[0].staged_path).unwrap(),
            *bytes
        );
    }
    let requests = fixture.join().unwrap();
    assert_eq!(requests.len(), 6);
    assert!(requests[3]
        .to_ascii_lowercase()
        .contains(&format!("if-match: \\\"{}\\\"", "b".repeat(40)).replace("\\\"", "\"")));
    assert!(requests[5].contains(&format!("resolve/{new_revision}/")));
}

#[test]
fn bounded_non_lfs_verification_rejects_wrong_size_sha_and_large_unverified_files() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/small-negative",
            vec![listed_file("config.json", 4, None)],
        )),
        FixtureResponse::Body(b"evil".to_vec()),
        FixtureResponse::Json(metadata(
            "owner/small-negative",
            vec![listed_file("config.json", 4, None)],
        )),
        FixtureResponse::Body(b"short".to_vec()),
        FixtureResponse::Json(metadata(
            "owner/small-negative",
            vec![listed_file("config.json", 16 * 1024 * 1024 + 1, None)],
        )),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_negative_fixture".into()));
    let hub = hub_for(&client);
    let root = TempRoot::new("small-negative");
    for expected_size in [4, 4, 16 * 1024 * 1024 + 1] {
        let plan = plan(
            "small",
            vec![source_group(
                "owner/small-negative",
                "config",
                vec![source_file(
                    "config.json",
                    Some(expected_size),
                    Some(&sha256_hex(b"good")),
                )],
            )],
        );
        assert!(matches!(
            hub.download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new()),
            Err(turbospark_catalog::HubError::InvalidResponse { .. })
        ));
        assert_staging_parent_empty(&root.0);
    }
    let requests = fixture.join().unwrap();
    assert_eq!(
        requests.len(),
        5,
        "large unverified files must not be downloaded to discover a digest"
    );
}

#[test]
fn pinned_range_responses_reject_wrong_offset_total_validator_and_ignored_range() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let hash = sha256_hex(b"good");
    let body = |status: &str, headers: &str, bytes: &[u8]| {
        let mut wire = format!(
            "HTTP/1.1 {status}\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        )
        .into_bytes();
        wire.extend_from_slice(bytes);
        FixtureResponse::Raw(wire)
    };
    let mut responses = Vec::new();
    for wire in [
        body(
            "206 Partial Content",
            "Content-Range: bytes 1-4/4\r\n",
            b"good",
        ),
        body(
            "206 Partial Content",
            "Content-Range: bytes 0-3/5\r\n",
            b"good",
        ),
        body(
            "206 Partial Content",
            "Content-Range: bytes 0-3/4\r\nETag: \"changed\"\r\n",
            b"good",
        ),
        body(
            "206 Partial Content",
            "Content-Range: bytes 0-3/4\r\n",
            b"longer",
        ),
        body("200 OK", "", b"good"),
    ] {
        responses.push(FixtureResponse::Json(metadata(
            "owner/responses",
            vec![listed_file("file.bin", 4, Some(&hash))],
        )));
        responses.push(wire);
    }
    let (endpoint, fixture) = fixture_server(responses);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_response_fixture".into()));
    let hub = hub_for(&client);
    let root = TempRoot::new("response-guards");
    let plan = plan(
        "response-owner",
        vec![source_group(
            "owner/responses",
            "weights",
            vec![source_file("file.bin", Some(4), Some(&hash))],
        )],
    );
    for rule in [
        "downloaded_range_matches_pinned_source",
        "downloaded_range_matches_pinned_source",
        "downloaded_validator_matches_pinned_source",
        "downloaded_size_matches_authoritative_source",
        "http_range_required",
    ] {
        let result = hub.download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new());
        if rule == "http_range_required" {
            assert!(matches!(
                result,
                Err(turbospark_catalog::HubError::Network(_))
            ));
        } else {
            assert!(
                matches!(result, Err(turbospark_catalog::HubError::TransferIntegrity { rule: actual, .. }) if actual == rule),
                "expected {rule}, got {result:?}"
            );
        }
        assert_staging_parent_empty(&root.0);
        assert!(cached_ranges(&root.0).is_empty());
    }
    assert_eq!(fixture.join().unwrap().len(), 10);
}

#[test]
fn bounded_small_verification_rejects_oversized_body_and_malformed_blob_identity() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut malformed = listed_file("small.json", 4, None);
    malformed["blobId"] = serde_json::json!("malformed-blob");
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata(
            "owner/oversized",
            vec![listed_file("small.json", 4, None)],
        )),
        FixtureResponse::Raw(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                16 * 1024 * 1024 + 1
            )
            .into_bytes(),
        ),
        FixtureResponse::Json(metadata("owner/oversized", vec![malformed])),
    ]);
    let _reset = set_endpoint(endpoint);
    let client = Client::with_token(Some("hf_oversized_fixture".into()));
    let hub = hub_for(&client);
    let root = TempRoot::new("oversized-guard");
    let plan = plan(
        "small",
        vec![source_group(
            "owner/oversized",
            "config",
            vec![source_file(
                "small.json",
                Some(4),
                Some(&sha256_hex(b"good")),
            )],
        )],
    );
    for expected_rule in [
        "response_size_within_limit",
        "owner_source_file_metadata_valid",
    ] {
        assert!(
            matches!(hub.download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new()), Err(turbospark_catalog::HubError::InvalidResponse { rule, .. }) if rule == expected_rule)
        );
        assert_staging_parent_empty(&root.0);
    }
    assert_eq!(fixture.join().unwrap().len(), 3);
}

#[test]
#[ignore = "network: transfers exact pinned Whisper tokenizer, validates SHA and ranged reuse"]
fn actual_pinned_whisper_tokenizer_transfers_and_reuses_verified_ranges() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _reset = set_endpoint("https://huggingface.co".into());
    let catalog = turbospark_catalog::AudioCatalog::embedded().unwrap();
    let profile = catalog
        .entries()
        .find(|profile| profile.identity.alias == "whisper-base")
        .unwrap();
    let mut plan = catalog.pinned_plan(&profile.identity).unwrap();
    plan.sources[0]
        .files
        .retain(|file| file.path == "tokenizer.json");
    assert_eq!(plan.sources[0].files.len(), 1);
    let file = &plan.sources[0].files[0];
    assert_eq!(file.expected_size, Some(2480466));
    let root = TempRoot::new("actual-whisper-tokenizer");
    let client = Client::new();
    let hub = hub_for(&client);
    for _ in 0..2 {
        let mut updates = Vec::new();
        let receipt = hub
            .download_pinned_artifacts(
                &plan,
                &root.0,
                &mut |update| updates.push(update),
                &CancelFlag::new(),
            )
            .unwrap();
        assert_eq!(
            receipt.sources[0].files[0].sha256,
            file.expected_sha256.as_deref().unwrap()
        );
        assert_progress(&updates, file.expected_size.unwrap(), 1);
        assert_eq!(cached_ranges(&root.0).len(), 1);
        std::fs::remove_dir_all(receipt.staging_root).unwrap();
    }
}

#[test]
fn interrupted_partial_file_reuses_completed_ranges_with_monotonic_progress() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let bytes = std::sync::Arc::new(vec![b'r'; 17 * 1024 * 1024]);
    let digest = sha256_hex(&bytes);
    let source_metadata = metadata(
        "owner/ranges",
        vec![listed_file(
            "weights.bin",
            bytes.len() as u64,
            Some(&digest),
        )],
    );
    let fail_first_range = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let responses = (0..6)
        .map(|_| {
            FixtureResponse::Source(
                bytes.clone(),
                source_metadata.clone(),
                fail_first_range.clone(),
            )
        })
        .collect();
    let (endpoint, fixture) = fixture_server(responses);
    let _reset = set_endpoint(endpoint);
    let root = TempRoot::new("partial-range-retry");
    let caller_file = root.0.join("published.bin");
    std::fs::write(&caller_file, b"existing install").unwrap();
    let client = Client::with_token(Some("hf_partial_range_fixture".into()));
    let hub = hub_for(&client);
    let plan = plan(
        "range-owner",
        vec![source_group(
            "owner/ranges",
            "weights",
            vec![source_file(
                "weights.bin",
                Some(bytes.len() as u64),
                Some(&digest),
            )],
        )],
    );
    assert!(matches!(
        hub.download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new()),
        Err(turbospark_catalog::HubError::Network(_))
    ));
    assert_eq!(
        cached_ranges(&root.0).len(),
        1,
        "interrupted file keeps only the completed tail range"
    );
    assert_eq!(
        std::fs::read_dir(&root.0)
            .unwrap()
            .filter(|entry| entry.as_ref().unwrap().file_name() != ".download-cache")
            .count(),
        1
    );
    fail_first_range.store(false, std::sync::atomic::Ordering::Release);
    let mut updates = Vec::new();
    let receipt = hub
        .download_pinned_artifacts(
            &plan,
            &root.0,
            &mut |update| updates.push(update),
            &CancelFlag::new(),
        )
        .unwrap();
    assert_progress(&updates, bytes.len() as u64, 1);
    assert_eq!(receipt.sources[0].files[0].sha256, digest);
    assert_eq!(
        std::fs::read(&receipt.sources[0].files[0].staged_path).unwrap(),
        *bytes
    );
    assert_eq!(std::fs::read(&caller_file).unwrap(), b"existing install");
    let requests = fixture.join().unwrap();
    assert_eq!(requests.len(), 5);
    assert_eq!(
        requests
            .iter()
            .filter(|request| request
                .to_ascii_lowercase()
                .contains("range: bytes=16777216-17825791"))
            .count(),
        1,
        "verified tail is not fetched again"
    );
    assert_eq!(
        requests
            .iter()
            .filter(|request| request
                .to_ascii_lowercase()
                .contains("range: bytes=0-16777215"))
            .count(),
        2,
        "only the interrupted first range is retried"
    );
}

#[test]
fn shared_control_interrupts_slow_small_file_preflight_and_restarts_it_after_pause() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let bytes = vec![b's'; 128 * 1024];
    let metadata = metadata(
        "owner/slow-small",
        vec![listed_file("small.json", bytes.len() as u64, None)],
    );
    let plan = plan(
        "slow-owner",
        vec![source_group(
            "owner/slow-small",
            "config",
            vec![source_file(
                "small.json",
                Some(bytes.len() as u64),
                Some(&sha256_hex(&bytes)),
            )],
        )],
    );
    for pause in [false, true] {
        let (ready, waiting) = std::sync::mpsc::channel();
        let sent = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut responses = vec![
            FixtureResponse::Json(metadata.clone()),
            FixtureResponse::SlowBody(bytes.clone(), ready, sent.clone()),
        ];
        if pause {
            responses.extend([
                FixtureResponse::Body(bytes.clone()),
                FixtureResponse::Body(bytes.clone()),
            ]);
        }
        let (endpoint, fixture) = fixture_server(responses);
        let _reset = set_endpoint(endpoint);
        let root = TempRoot::new(if pause { "small-pause" } else { "small-cancel" });
        let flag = CancelFlag::new();
        let worker_flag = flag.clone();
        let destination = root.0.clone();
        let plan = plan.clone();
        let worker = thread::spawn(move || {
            let client = Client::with_token(Some("hf_slow_small_fixture".into()));
            hub_for(&client).download_pinned_artifacts(
                &plan,
                &destination,
                &mut |_| {},
                &worker_flag,
            )
        });
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        let started = Instant::now();
        if pause {
            assert!(flag.pause());
            thread::sleep(Duration::from_millis(100));
            assert!(!worker.is_finished());
            assert!(
                sent.load(std::sync::atomic::Ordering::Acquire) < bytes.len() as u64,
                "pause must drop the unverified stream before its whole body finishes"
            );
            assert_staging_parent_empty(&root.0);
            assert!(cached_ranges(&root.0).is_empty());
            assert!(flag.resume());
            let receipt = worker
                .join()
                .unwrap()
                .expect("resume refetches and verifies the exact small pin");
            assert_eq!(
                std::fs::read(&receipt.sources[0].files[0].staged_path).unwrap(),
                bytes
            );
            let requests = fixture.join().unwrap();
            assert_eq!(
                requests.len(),
                4,
                "pause restarts the bounded GET before the ranged payload transfer"
            );
            assert!(!requests[1].to_ascii_lowercase().contains("range:"));
            assert!(!requests[2].to_ascii_lowercase().contains("range:"));
            assert!(requests[3].to_ascii_lowercase().contains("range:"));
        } else {
            flag.cancel();
            assert!(matches!(
                worker.join().unwrap(),
                Err(turbospark_catalog::HubError::Cancelled)
            ));
            assert!(
                started.elapsed() < Duration::from_millis(500),
                "cancel must stop between chunks rather than consume the 1.28 second body"
            );
            assert!(sent.load(std::sync::atomic::Ordering::Acquire) < bytes.len() as u64);
            assert_staging_parent_empty(&root.0);
            assert_eq!(fixture.join().unwrap().len(), 2);
        }
    }
}

#[test]
#[ignore = "network: validates a one-byte LFS range and source validators for each exact Audio pin"]
fn actual_pinned_lfs_ranges_preserve_remote_size_and_validators() {
    use repack::RangeSource;
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _reset = set_endpoint("https://huggingface.co".into());
    let catalog = turbospark_catalog::AudioCatalog::embedded().unwrap();
    let client = Client::new();
    let hub = hub_for(&client);
    for profile in catalog.entries() {
        let mut plan = catalog.pinned_plan(&profile.identity).unwrap();
        plan.sources[0].files.retain(|file| {
            file.expected_size
                .is_some_and(|size| size > 16 * 1024 * 1024)
        });
        plan.sources[0].files.truncate(1);
        assert_eq!(plan.sources[0].files.len(), 1);
        let source = &plan.sources[0];
        let resolved = hub.resolve_source_identity(source).unwrap();
        let file = &resolved.files[0];
        let url = source.repo.file_url(&file.path);
        let head_client = reqwest::blocking::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let mut head = head_client.head(&url);
        if let Some(token) = client.token() {
            head = head.bearer_auth(token);
        }
        let response = head.send().unwrap();
        assert_eq!(
            response
                .headers()
                .get("x-linked-etag")
                .unwrap()
                .to_str()
                .unwrap()
                .trim_matches('"'),
            file.source_sha256.as_deref().unwrap()
        );
        let validator = response
            .headers()
            .get("x-xet-hash")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let range = repack::HttpRangeSource::new(url)
            .with_optional_token(client.token())
            .with_pinned_identity(
                profile.identity.alias.clone(),
                file.authoritative_size.unwrap(),
                Some(validator),
            );
        assert_eq!(
            range.read_range(0, 1).unwrap().len(),
            1,
            "{} LFS source",
            profile.identity.alias
        );
    }
}

#[test]
fn lfs_head_checks_manifest_authority_and_changed_xet_validator_before_cache_reuse() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let digest = sha256_hex(b"good");
    let metadata = metadata(
        "owner/xet",
        vec![listed_file("weights.bin", 4, Some(&digest))],
    );
    let head = |size: u64, sha: &str, xet: &str| {
        FixtureResponse::Head(format!(
            "X-Linked-Size: {size}\r\nX-Linked-ETag: \"{sha}\"\r\nX-Xet-Hash: {xet}\r\n"
        ))
    };
    let ranged = |xet: &str| {
        let mut response = format!("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-3/4\r\nETag: \"{xet}\"\r\nContent-Length: 4\r\nConnection: close\r\n\r\n").into_bytes();
        response.extend_from_slice(b"good");
        FixtureResponse::Raw(response)
    };
    let first_xet = "a".repeat(64);
    let second_xet = "b".repeat(64);
    let (endpoint, fixture) = fixture_server(vec![
        FixtureResponse::Json(metadata.clone()),
        head(5, &digest, &first_xet),
        FixtureResponse::Json(metadata.clone()),
        head(4, &"0".repeat(64), &first_xet),
        FixtureResponse::Json(metadata.clone()),
        head(4, &digest, "malformed-xet"),
        FixtureResponse::Json(metadata.clone()),
        head(4, &digest, &first_xet),
        ranged(&first_xet),
        FixtureResponse::Json(metadata),
        head(4, &digest, &second_xet),
        ranged(&second_xet),
    ]);
    let _reset = set_endpoint(endpoint);
    let root = TempRoot::new("xet-authority");
    let client = Client::with_token(Some("hf_xet_fixture".into()));
    let hub = hub_for(&client);
    let plan = plan(
        "xet-owner",
        vec![source_group(
            "owner/xet",
            "weights",
            vec![source_file("weights.bin", Some(4), Some(&digest))],
        )],
    );
    for rule in [
        "remote_source_size_matches_pin",
        "remote_source_digest_matches_pin",
        "remote_validator_shape",
    ] {
        assert!(
            matches!(hub.download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new()), Err(turbospark_catalog::HubError::InvalidResponse { rule: actual, .. }) if actual == rule)
        );
        assert_staging_parent_empty(&root.0);
        assert!(cached_ranges(&root.0).is_empty());
    }
    for _ in 0..2 {
        let receipt = hub
            .download_pinned_artifacts(&plan, &root.0, &mut |_| {}, &CancelFlag::new())
            .unwrap();
        assert_eq!(
            receipt.sources[0].files[0].sha256, digest,
            "manifest SHA remains the whole-file authority, distinct from Xet ETag"
        );
    }
    let requests = fixture.join().unwrap();
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.starts_with("GET /owner/"))
            .count(),
        2,
        "changed HEAD Xet validator forces a fresh range despite unchanged manifest SHA"
    );
    assert!(requests[4]
        .to_ascii_lowercase()
        .contains(&format!("if-match: \"{first_xet}\"")));
    assert!(requests[6]
        .to_ascii_lowercase()
        .contains(&format!("if-match: \"{second_xet}\"")));
}

fn assert_shared_retry_control(ranged: bool) {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let digest = sha256_hex(b"good");
    let plan = plan(
        "retry-owner",
        vec![source_group(
            "owner/retry",
            "config",
            vec![source_file("config.json", Some(4), Some(&digest))],
        )],
    );
    // Test the two retry ladders separately: preflight must issue no requests
    // while paused, and range backoff must wake before its 30-second timeout.
    let modes = if ranged { [false, true] } else { [true, false] };
    for pause in modes {
        let (ready, waiting) = std::sync::mpsc::channel();
        let (retried, retry_notice) = std::sync::mpsc::channel();
        let mut responses = vec![FixtureResponse::Json(metadata(
            "owner/retry",
            vec![listed_file("config.json", 4, None)],
        ))];
        if ranged {
            responses.push(FixtureResponse::Body(b"good".to_vec()));
        }
        responses.push(FixtureResponse::Retry(
            if pause { 429 } else { 503 },
            if pause { 1 } else { 30 },
            ready,
        ));
        if pause {
            responses.push(FixtureResponse::ObservedBody(b"good".to_vec(), retried));
            if !ranged {
                responses.push(FixtureResponse::Body(b"good".to_vec()));
            }
        }
        let (endpoint, fixture) =
            fixture_server_with_idle_timeout(responses, Duration::from_secs(5));
        let _reset = set_endpoint(endpoint);
        let root = TempRoot::new(if ranged {
            "range-backoff"
        } else {
            "preflight-backoff"
        });
        let flag = CancelFlag::new();
        let worker_flag = flag.clone();
        let destination = root.0.clone();
        let requested = plan.clone();
        let (completed, result) = std::sync::mpsc::channel();
        let worker = thread::spawn(move || {
            let client = Client::with_token(Some("hf_retry_fixture".into()));
            let transferred = hub_for(&client).download_pinned_artifacts(
                &requested,
                &destination,
                &mut |_| {},
                &worker_flag,
            );
            let _ = completed.send(transferred);
        });
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        if pause {
            assert!(flag.pause());
            assert!(
                matches!(
                    retry_notice.recv_timeout(Duration::from_millis(1300)),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                ),
                "paused transfer must not issue a retry after Retry-After expires"
            );
            assert!(flag.resume());
            retry_notice.recv_timeout(Duration::from_secs(5)).unwrap();
            let receipt = result
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap();
            assert_eq!(receipt.sources[0].files[0].sha256, digest);
        } else {
            // Let the HTTP error reach its retry wait before cancellation.
            thread::sleep(Duration::from_millis(50));
            let started = Instant::now();
            flag.cancel();
            assert!(
                matches!(
                    result.recv_timeout(Duration::from_millis(500)),
                    Ok(Err(turbospark_catalog::HubError::Cancelled))
                ),
                "cancellation must interrupt the 30-second retry backoff"
            );
            assert!(started.elapsed() < Duration::from_millis(500));
            assert_staging_parent_empty(&root.0);
        }
        worker.join().unwrap();
        let requests = fixture.join().unwrap();
        assert_eq!(
            requests.len(),
            if pause {
                4
            } else if ranged {
                3
            } else {
                2
            }
        );
    }
}

#[test]
fn shared_control_stops_preflight_retries_while_paused_and_cancels_backoff() {
    assert_shared_retry_control(false);
}

#[test]
fn shared_control_cancels_ranged_backoff_and_stops_retries_while_paused() {
    assert_shared_retry_control(true);
}

fn assert_missing_size_head_control(pause: bool) {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    fn receive(listener: &TcpListener, timeout: Duration) -> Option<(std::net::TcpStream, String)> {
        let started = Instant::now();
        let (stream, _) = loop {
            match listener.accept() {
                Ok(accepted) => break accepted,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if started.elapsed() >= timeout {
                        return None;
                    }
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept missing-size fixture: {error}"),
            }
        };
        stream.set_nonblocking(false).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut request = String::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let done = line.is_empty() || line == "\r\n";
            request.push_str(&line);
            if done {
                break;
            }
        }
        Some((stream, request))
    }
    fn head(stream: &mut std::net::TcpStream) {
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\n")
            .unwrap();
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let flag = CancelFlag::new();
    let server_flag = flag.clone();
    let (observed, next_head) = std::sync::mpsc::channel();
    let server = thread::spawn(move || {
        let (mut stream, request) = receive(&listener, Duration::from_secs(5)).unwrap();
        assert!(request.starts_with("GET /api/"));
        let metadata = metadata(
            "owner/fallback",
            vec![
                serde_json::json!({"rfilename":"one.bin","blobId":"a".repeat(40)}),
                serde_json::json!({"rfilename":"two.bin","blobId":"b".repeat(40)}),
            ],
        );
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{metadata}",
            metadata.len()
        )
        .unwrap();
        drop(stream);
        let (mut stream, request) = receive(&listener, Duration::from_secs(5)).unwrap();
        assert!(request.starts_with("HEAD ") && request.contains("one.bin"));
        // Change control while the first HEAD is active, before its response.
        // The second file has no listing size and must require another HEAD.
        if pause {
            assert!(server_flag.pause());
        } else {
            server_flag.cancel();
        }
        head(&mut stream);
        drop(stream);
        let premature = receive(&listener, Duration::from_millis(400));
        observed.send(premature.is_some()).unwrap();
        if let Some((mut stream, request)) = premature {
            assert!(request.starts_with("HEAD ") && request.contains("two.bin"));
            head(&mut stream);
            server_flag.cancel();
            return false;
        }
        if !pause {
            return false;
        }
        let (mut stream, request) = receive(&listener, Duration::from_secs(5)).unwrap();
        assert!(request.starts_with("HEAD ") && request.contains("two.bin"));
        assert!(!server_flag.is_paused());
        head(&mut stream);
        drop(stream);
        for path in ["one.bin", "two.bin"] {
            let (mut stream, request) = receive(&listener, Duration::from_secs(5)).unwrap();
            assert!(request.starts_with("GET ") && request.contains(path));
            assert!(request.to_ascii_lowercase().contains("range: bytes=0-3"));
            stream.write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-3/4\r\nContent-Length: 4\r\nConnection: close\r\n\r\ngood").unwrap();
        }
        true
    });
    let _reset = set_endpoint(endpoint);
    let root = TempRoot::new(if pause {
        "missing-size-pause"
    } else {
        "missing-size-cancel"
    });
    let requested = plan(
        "fallback",
        vec![source_group(
            "owner/fallback",
            "config",
            vec![
                source_file("one.bin", Some(4), None),
                source_file("two.bin", Some(4), None),
            ],
        )],
    );
    let worker_flag = flag.clone();
    let destination = root.0.clone();
    let worker = thread::spawn(move || {
        let client = Client::with_token(Some("hf_fallback_fixture".into()));
        hub_for(&client).download_pinned_artifacts(
            &requested,
            &destination,
            &mut |_| {},
            &worker_flag,
        )
    });
    assert!(
        !next_head.recv_timeout(Duration::from_secs(5)).unwrap(),
        "second size-fallback HEAD must not start after pause/cancel during the first HEAD"
    );
    if pause {
        assert!(flag.resume());
        let receipt = worker.join().unwrap().unwrap();
        assert_eq!(receipt.sources[0].files.len(), 2);
        assert!(receipt.sources[0]
            .files
            .iter()
            .all(|file| file.sha256 == sha256_hex(b"good")));
        assert!(server.join().unwrap(), "second HEAD must run after resume");
    } else {
        assert!(matches!(
            worker.join().unwrap(),
            Err(turbospark_catalog::HubError::Cancelled)
        ));
        assert!(!server.join().unwrap());
        assert_staging_parent_empty(&root.0);
    }
}

#[test]
fn shared_control_pauses_missing_size_heads_before_the_next_file() {
    assert_missing_size_head_control(true);
}

#[test]
fn shared_control_cancels_missing_size_heads_before_the_next_file() {
    assert_missing_size_head_control(false);
}
