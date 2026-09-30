use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use turbospark_catalog::{
    evaluate_gguf, group_variants, install_variant, record, set_hf_endpoint_override, Catalog,
    Client, HubClient, HubGgufVariant, Machine, ProbeReport, RepoFile, RepoRef, ShardSetStatus,
    Store, VariantInstallPlan, VariantInstallability,
};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
static ENDPOINT_LOCK: Mutex<()> = Mutex::new(());

struct FixtureServer {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FixtureServer {
    fn requests(&self) -> Vec<String> {
        self.requests.lock().expect("requests mutex").clone()
    }
}

impl Drop for FixtureServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("fixture server joins");
        }
    }
}

struct EndpointReset;

impl Drop for EndpointReset {
    fn drop(&mut self) {
        set_hf_endpoint_override(None);
    }
}

fn fixture_server(files: BTreeMap<String, Vec<u8>>) -> FixtureServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("fixture bind");
    listener
        .set_nonblocking(true)
        .expect("set nonblocking fixture listener");
    let address = listener.local_addr().expect("fixture address");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let thread_requests = Arc::clone(&requests);
    let thread_stop = Arc::clone(&stop);
    let thread = thread::spawn(move || {
        while !thread_stop.load(Ordering::Acquire) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream
                        .set_nonblocking(false)
                        .expect("accepted fixture stream is blocking");
                    stream
                        .set_read_timeout(Some(Duration::from_secs(3)))
                        .expect("set fixture stream timeout");
                    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                    let mut request_line = String::new();
                    reader
                        .read_line(&mut request_line)
                        .expect("read request line");
                    let mut range = None;
                    loop {
                        let mut line = String::new();
                        reader.read_line(&mut line).expect("read request header");
                        if line == "\r\n" || line.is_empty() {
                            break;
                        }
                        if let Some((name, value)) = line.trim_end().split_once(':') {
                            if name.eq_ignore_ascii_case("range") {
                                range = Some(value.trim().to_string());
                            }
                        }
                    }

                    let mut parts = request_line.split_whitespace();
                    let method = parts.next().unwrap_or("?");
                    let target = parts.next().unwrap_or("/");
                    thread_requests
                        .lock()
                        .expect("requests mutex")
                        .push(format!("{method} {target}"));

                    let result = fixture_response(method, target, range.as_deref(), &files);
                    stream.write_all(&result).expect("write fixture response");
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(2));
                }
                Err(error) => panic!("accept fixture request: {error}"),
            }
        }
    });

    FixtureServer {
        endpoint: format!("http://{address}"),
        requests,
        stop,
        thread: Some(thread),
    }
}

fn fixture_response(
    method: &str,
    target: &str,
    range: Option<&str>,
    files: &BTreeMap<String, Vec<u8>>,
) -> Vec<u8> {
    let path = target.split('?').next().unwrap_or(target);
    if path.starts_with("/api/models/owner/model/revision/") {
        let siblings: Vec<_> = files
            .iter()
            .map(|(name, bytes)| {
                serde_json::json!({ "rfilename": name, "size": bytes.len() as u64 })
            })
            .collect();
        let body = serde_json::json!({
            "id": "owner/model",
            "sha": REVISION,
            "siblings": siblings,
        })
        .to_string()
        .into_bytes();
        return response(200, "OK", "application/json", &body, &[]);
    }

    let Some(name) = path.strip_prefix(&format!("/owner/model/resolve/{REVISION}/")) else {
        return response(404, "Not Found", "text/plain", b"not found", &[]);
    };
    let Some(bytes) = files.get(name) else {
        return response(404, "Not Found", "text/plain", b"not found", &[]);
    };

    if method.eq_ignore_ascii_case("HEAD") {
        return response(
            200,
            "OK",
            "application/octet-stream",
            b"",
            &[
                ("x-linked-size", bytes.len().to_string()),
                ("Content-Length", "0".to_string()),
            ],
        );
    }

    if let Some(range) = range {
        let requested = range.strip_prefix("bytes=").expect("range bytes prefix");
        let (start, end) = requested.split_once('-').expect("range endpoints");
        let start: usize = start.parse().expect("range start");
        let end: usize = end.parse().expect("range end");
        if start >= bytes.len() || start > end {
            return response(416, "Range Not Satisfiable", "text/plain", b"", &[]);
        }
        let end = end.min(bytes.len() - 1);
        let body = &bytes[start..=end];
        return response(
            206,
            "Partial Content",
            "application/octet-stream",
            body,
            &[(
                "Content-Range",
                format!("bytes {start}-{end}/{}", bytes.len()),
            )],
        );
    }

    response(200, "OK", "application/octet-stream", bytes, &[])
}

fn response(
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
    headers: &[(&str, String)],
) -> Vec<u8> {
    let mut wire = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        wire.push_str(&format!("{name}: {value}\r\n"));
    }
    wire.push_str("\r\n");
    wire.into_bytes()
        .into_iter()
        .chain(body.iter().copied())
        .collect()
}

fn client_for(endpoint: &str) -> (EndpointReset, Client) {
    set_hf_endpoint_override(Some(endpoint.to_string()));
    (EndpointReset, Client::with_timeout(Duration::from_secs(3)))
}

fn hub_for(client: &Client) -> HubClient<'_> {
    HubClient::new(
        client,
        Catalog::embedded().expect("embedded catalog"),
        Machine {
            physical_bytes: 32 * 1024 * 1024 * 1024,
            ..Machine::default()
        },
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
}

fn tempfile(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "turbospark-hub-install-{tag}-{}",
        std::process::id()
    ));
    std::fs::remove_dir_all(&path).ok();
    std::fs::create_dir_all(&path).expect("create test directory");
    path
}

fn explicit_probe(repo: &RepoRef, file: &str, gguf: &[u8]) -> ProbeReport {
    let header = repack::parse_gguf_header(gguf, repack::GGUF_DEFAULT_MAX_HEADER_BYTES)
        .expect("synthetic GGUF header");
    let mut report = evaluate_gguf(&header, repo, file, Some(gguf.len() as u64));
    report.sidecars_present = vec![
        "tokenizer.json".to_string(),
        "tokenizer_config.json".to_string(),
    ];
    report
}

fn selected_variant<'a>(variants: &'a [HubGgufVariant], file: &str) -> &'a HubGgufVariant {
    variants
        .iter()
        .find(|variant| variant.files.iter().any(|entry| entry.name == file))
        .expect("variant containing selected file")
}

#[test]
fn selected_variant_install_records_its_label_and_streams_no_sibling_group_files() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let repo = RepoRef::new("owner/model", REVISION);
    let selected_file = "alpha-Q8_0.gguf";
    let sibling_file = "beta-Q8_0.gguf";
    let (selected_bytes, _) =
        repack::build_synthetic_gemma4_gguf(repack::SyntheticGgufShape::default());
    let (sibling_bytes, _) =
        repack::build_synthetic_gemma4_gguf(repack::SyntheticGgufShape::k_quant());
    let tokenizer_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tokenizer/tests/fixtures/ChatMLTokenizer");
    let mut files = BTreeMap::from([
        (selected_file.to_string(), selected_bytes.clone()),
        (sibling_file.to_string(), sibling_bytes),
        (
            "tokenizer.json".to_string(),
            std::fs::read(tokenizer_dir.join("tokenizer.json")).expect("tokenizer fixture"),
        ),
        (
            "tokenizer_config.json".to_string(),
            std::fs::read(tokenizer_dir.join("tokenizer_config.json"))
                .expect("tokenizer config fixture"),
        ),
    ]);
    let server = fixture_server(std::mem::take(&mut files));
    let (_reset, client) = client_for(&server.endpoint);
    let hub = hub_for(&client);
    let probe = explicit_probe(&repo, selected_file, &selected_bytes);

    let variants = hub
        .repo_variants_with_probe(&repo, &probe)
        .expect("variant metadata");
    let selected = selected_variant(&variants, selected_file);
    assert_eq!(selected.label.0, "Q8_0");
    assert_eq!(selected.installability, VariantInstallability::Ready);

    let sibling = selected_variant(&variants, sibling_file);
    assert_eq!(sibling.label, selected.label);
    assert!(matches!(
        sibling.installability,
        VariantInstallability::SupportUnverified { .. }
    ));

    let other_group_probe = explicit_probe(&repo, sibling_file, &selected_bytes);
    assert!(
        VariantInstallPlan::from_hub_variant(
            "selected",
            selected,
            &other_group_probe,
            repo.clone()
        )
        .is_err(),
        "a same-label probe from a different file group must not bind to the selection"
    );

    let plan = VariantInstallPlan::from_hub_variant("selected", selected, &probe, repo.clone())
        .expect("bind selected variant to its explicit probe");
    let root = tempfile("selected");
    let install_dir = root.join("selected.gturbo");
    let installed = install_variant(&plan, &install_dir, &client, |_| {}).expect("install");
    assert_eq!(installed.model.variant.as_deref(), Some("Q8_0"));

    let store = Store::new(root.join("store"));
    record(&store, &installed).expect("record selected variant");
    assert_eq!(
        store.installed()["selected"].variant.as_deref(),
        Some("Q8_0")
    );

    let weight_requests: Vec<String> = server
        .requests()
        .into_iter()
        .filter(|request| request.contains("/owner/model/resolve/") && request.ends_with(".gguf"))
        .collect();
    assert!(
        !weight_requests.is_empty(),
        "selected GGUF should be streamed"
    );
    assert!(
        weight_requests
            .iter()
            .all(|request| request.contains(selected_file)),
        "only the selected variant file set may be requested: {weight_requests:?}"
    );
    assert!(
        weight_requests
            .iter()
            .all(|request| !request.contains(sibling_file)),
        "the sibling same-label group must never be streamed: {weight_requests:?}"
    );
    assert!(install_dir.join("model_weights.bin").is_file());
    std::fs::remove_dir_all(root).expect("remove test directory");
}

#[test]
fn variant_constructor_rejects_non_ready_and_unpinned_selections() {
    let repo = RepoRef::new("owner/model", REVISION);
    let (gguf, _) = repack::build_synthetic_gemma4_gguf(repack::SyntheticGgufShape::default());
    let files = vec![RepoFile {
        name: "model-Q8_0.gguf".to_string(),
        size: Some(gguf.len() as u64),
    }];
    let file_name = files[0].name.clone();
    let report = explicit_probe(&repo, &file_name, &gguf);
    let variant = HubGgufVariant {
        repo: repo.clone(),
        label: turbospark_catalog::QuantLabel("Q8_0".to_string()),
        files,
        total_bytes: Some(gguf.len() as u64),
        fit: turbospark_catalog::FitSummary {
            verdict: "unknown (probe it)".to_string(),
            is_estimate: false,
            counted_bytes: None,
            counted_source: "unknown".to_string(),
            mapped_bytes: None,
            notes: Vec::new(),
        },
        installability: VariantInstallability::SupportUnverified {
            files: vec!["model-Q8_0.gguf".to_string()],
        },
    };
    assert_eq!(
        group_variants(&variant.files)
            .first()
            .map(|group| group.shard_set.clone()),
        Some(ShardSetStatus::SingleFile)
    );
    assert!(VariantInstallPlan::from_hub_variant("model", &variant, &report, repo).is_err());

    let mut ready_variant = variant;
    ready_variant.installability = VariantInstallability::Ready;
    let same_label_other_group = explicit_probe(
        &RepoRef::new("owner/model", REVISION),
        "model-copy-Q8_0.gguf",
        &gguf,
    );
    assert!(
        VariantInstallPlan::from_hub_variant(
            "model",
            &ready_variant,
            &same_label_other_group,
            RepoRef::new("owner/model", REVISION),
        )
        .is_err(),
        "a probe for another file group must not bind solely because its label matches"
    );

    let floating_probe = explicit_probe(&RepoRef::new("owner/model", "main"), &file_name, &gguf);
    assert!(
        VariantInstallPlan::from_hub_variant(
            "model",
            &ready_variant,
            &floating_probe,
            RepoRef::new("owner/model", REVISION),
        )
        .is_err(),
        "a floating probe must not bind to an immutable variant"
    );
}
