use std::ffi::OsString;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use turbospark_catalog::{Catalog, Client, HubClient, HubError, HubQuery, Machine, Refresh, Store};

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
    fn anonymous(home: &std::path::Path) -> Self {
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
            "turbospark-hub-errors-{label}-{}-{sequence}",
            std::process::id()
        )))
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture_status_server(
    status: u16,
    responses: Vec<Option<u64>>,
) -> (String, thread::JoinHandle<usize>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture server");
    listener
        .set_nonblocking(true)
        .expect("make fixture accept bounded");
    let address = listener.local_addr().expect("fixture address");
    let handle = thread::spawn(move || {
        let mut requests = 0;
        for retry_after in responses {
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
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("set request timeout");
            let mut reader = BufReader::new(stream.try_clone().expect("clone fixture stream"));
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).expect("read request header") == 0
                    || line == "\r\n"
                    || line == "\n"
                {
                    break;
                }
            }
            let retry_header = retry_after
                .map(|seconds| format!("Retry-After: {seconds}\r\n"))
                .unwrap_or_default();
            let reason = match status {
                429 => "Too Many Requests",
                503 => "Service Unavailable",
                _ => "Fixture Status",
            };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\n{retry_header}Content-Length: 0\r\nConnection: close\r\n\r\n"
            );
            stream
                .write_all(response.as_bytes())
                .expect("write fixture response");
            requests += 1;
        }
        requests
    });
    (format!("http://{address}"), handle)
}

fn hub_client<'a>(client: &'a Client, catalog: &Catalog, root: &TempRoot) -> HubClient<'a> {
    HubClient::new(
        client,
        catalog.clone(),
        Machine::default(),
        4096,
        model_io::ExpertCacheSlots::Auto,
    )
    .with_store(Store::new(&root.0))
}

fn assert_bundled_lookup_still_works(catalog: &Catalog) {
    assert!(
        catalog
            .find("Qwen3.8")
            .iter()
            .any(|entry| entry.source.repo == "mlx-community/Qwen3.8-27B-4bit"),
        "failed live requests must leave bundled lookup available"
    );
}

#[test]
fn connection_failure_is_offline_and_bundled_lookup_remains_available() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("offline");
    std::fs::create_dir_all(root.0.join("anonymous-home")).expect("create anonymous home");
    let _auth = AuthEnvironmentReset::anonymous(&root.0.join("anonymous-home"));

    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve local endpoint");
    let address = listener.local_addr().expect("reserved endpoint address");
    drop(listener);
    turbospark_catalog::set_hf_endpoint_override(Some(format!("http://{address}")));
    let _reset = EndpointReset;

    let catalog = Catalog::embedded().expect("embedded catalog");
    let client = Client::with_timeout(Duration::from_secs(2));
    let hub = hub_client(&client, &catalog, &root);

    let error = hub
        .search(&HubQuery::new("Qwen3.8"), Refresh::Force)
        .expect_err("closed local endpoint must fail");
    assert_eq!(error, HubError::Offline);
    assert_bundled_lookup_still_works(&catalog);
}

#[test]
fn rate_limit_preserves_final_retry_after_and_bundled_lookup_remains_available() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("rate-limit");
    std::fs::create_dir_all(root.0.join("anonymous-home")).expect("create anonymous home");
    let _auth = AuthEnvironmentReset::anonymous(&root.0.join("anonymous-home"));

    let mut retry_after_values = vec![Some(0); 7];
    retry_after_values.push(Some(37));
    let (endpoint, server) = fixture_status_server(429, retry_after_values);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;

    let catalog = Catalog::embedded().expect("embedded catalog");
    let client = Client::with_timeout(Duration::from_secs(2));
    let hub = hub_client(&client, &catalog, &root);

    let error = hub
        .search(&HubQuery::new("Qwen3.8"), Refresh::Force)
        .expect_err("fixture rate limit must fail after retries");
    assert_eq!(
        error,
        HubError::RateLimited {
            retry_after_secs: Some(37)
        }
    );
    assert_eq!(server.join().expect("join fixture server"), 8);
    assert_bundled_lookup_still_works(&catalog);
}

#[test]
fn service_unavailable_remains_a_network_error() {
    let _endpoint_lock = ENDPOINT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let root = TempRoot::new("service-unavailable");
    std::fs::create_dir_all(root.0.join("anonymous-home")).expect("create anonymous home");
    let _auth = AuthEnvironmentReset::anonymous(&root.0.join("anonymous-home"));

    let (endpoint, server) = fixture_status_server(503, vec![Some(0); 8]);
    turbospark_catalog::set_hf_endpoint_override(Some(endpoint));
    let _reset = EndpointReset;

    let catalog = Catalog::embedded().expect("embedded catalog");
    let client = Client::with_timeout(Duration::from_secs(2));
    let hub = hub_client(&client, &catalog, &root);

    let error = hub
        .search(&HubQuery::new("Qwen3.8"), Refresh::Force)
        .expect_err("fixture service failure must remain an error");
    assert!(matches!(error, HubError::Network(message) if message.contains("HTTP 503")));
    assert_eq!(server.join().expect("join fixture server"), 8);
    assert_bundled_lookup_still_works(&catalog);
}
