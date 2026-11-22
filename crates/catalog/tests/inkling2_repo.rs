//! The whole-file Inkling route against a loopback Hub: detection through
//! the probe, the pinned download, resume by size, and the kept folder on a
//! failed intake. No real weights: the intake is expected to refuse the
//! fixture files, which is what proves the download step ran to the end.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use turbospark_catalog::{
    detect_inkling2_repo, install_destination, install_inkling2_repo, probe,
    set_hf_endpoint_override, Client, Inkling2RepoKind, InstallKind, RepoRef, SourceKind, Store,
    INKLING2_TOKENIZER_REPO,
};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const TOK_REVISION: &str = "89abcdef0123456789abcdef0123456789abcdef";
static ENDPOINT_LOCK: Mutex<()> = Mutex::new(());

type Repos = BTreeMap<String, (String, BTreeMap<String, Vec<u8>>)>;

struct Server {
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Server {
    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            t.join().unwrap();
        }
        set_hf_endpoint_override(None);
    }
}

/// `repos`: repo id -> (commit, files).
fn serve(repos: Repos) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let (log, halt) = (Arc::clone(&requests), Arc::clone(&stop));
    let thread = thread::spawn(move || {
        while !halt.load(Ordering::Acquire) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_nonblocking(false).unwrap();
                    stream
                        .set_read_timeout(Some(Duration::from_secs(3)))
                        .unwrap();
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let mut range = None;
                    loop {
                        let mut h = String::new();
                        reader.read_line(&mut h).unwrap();
                        if h == "\r\n" || h.is_empty() {
                            break;
                        }
                        if let Some((k, v)) = h.trim_end().split_once(':') {
                            if k.eq_ignore_ascii_case("range") {
                                range = Some(v.trim().to_string());
                            }
                        }
                    }
                    let mut parts = line.split_whitespace();
                    let method = parts.next().unwrap_or("?").to_string();
                    let target = parts.next().unwrap_or("/").to_string();
                    log.lock().unwrap().push(format!("{method} {target}"));
                    let out = respond(&method, &target, range.as_deref(), &repos);
                    stream.write_all(&out).unwrap();
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(2));
                }
                Err(e) => panic!("accept: {e}"),
            }
        }
    });
    set_hf_endpoint_override(Some(format!("http://{address}")));
    Server {
        requests,
        stop,
        thread: Some(thread),
    }
}

fn respond(method: &str, target: &str, range: Option<&str>, repos: &Repos) -> Vec<u8> {
    let path = target.split('?').next().unwrap_or(target);
    for (id, (sha, files)) in repos {
        if path.starts_with(&format!("/api/models/{id}/revision/")) {
            let siblings: Vec<_> = files
                .iter()
                .map(|(n, b)| serde_json::json!({"rfilename": n, "size": b.len() as u64}))
                .collect();
            let body = serde_json::json!({"id": id, "sha": sha, "siblings": siblings}).to_string();
            return reply(200, body.as_bytes(), &[]);
        }
        // The Hub resolves a branch name as well as a commit.
        let Some(name) = path
            .strip_prefix(&format!("/{id}/resolve/{sha}/"))
            .or_else(|| path.strip_prefix(&format!("/{id}/resolve/main/")))
        else {
            continue;
        };
        let Some(bytes) = files.get(name) else {
            return reply(404, b"", &[]);
        };
        if method.eq_ignore_ascii_case("HEAD") {
            return reply(200, b"", &[("x-linked-size", bytes.len().to_string())]);
        }
        if let Some(r) = range {
            let (a, b) = r.strip_prefix("bytes=").unwrap().split_once('-').unwrap();
            let a: usize = a.parse().unwrap();
            let b: usize = b.parse::<usize>().unwrap().min(bytes.len() - 1);
            if a >= bytes.len() {
                return reply(416, b"", &[]);
            }
            return reply(
                206,
                &bytes[a..=b],
                &[("Content-Range", format!("bytes {a}-{b}/{}", bytes.len()))],
            );
        }
        return reply(200, bytes, &[]);
    }
    reply(404, b"", &[])
}

fn reply(status: u16, body: &[u8], headers: &[(&str, String)]) -> Vec<u8> {
    let mut wire = format!(
        "HTTP/1.1 {status} X\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        wire.push_str(&format!("{k}: {v}\r\n"));
    }
    wire.push_str("\r\n");
    let mut out = wire.into_bytes();
    out.extend_from_slice(body);
    out
}

/// A GGUF v3 header with no tensors and one key: general.architecture.
fn gguf(arch: &str) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    let key = b"general.architecture";
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key);
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&(arch.len() as u64).to_le_bytes());
    b.extend_from_slice(arch.as_bytes());
    // Larger than the header reader's first range, as a real file is, so
    // the reader never asks past the end (which the server would refuse and
    // the reader retry).
    b.resize(4 << 20, 0);
    b
}

fn files(entries: &[(&str, Vec<u8>)]) -> BTreeMap<String, Vec<u8>> {
    entries
        .iter()
        .map(|(n, b)| (n.to_string(), b.clone()))
        .collect()
}

fn tokenizer_repo() -> (String, (String, BTreeMap<String, Vec<u8>>)) {
    (
        INKLING2_TOKENIZER_REPO.to_string(),
        (
            TOK_REVISION.to_string(),
            files(&[
                ("tokenizer.json", b"{}".to_vec()),
                ("chat_template.jinja", b"x".to_vec()),
                ("model.safetensors", vec![0; 8]),
            ]),
        ),
    )
}

fn lamq_release(format: &str) -> (String, (String, BTreeMap<String, Vec<u8>>)) {
    (
        "owner/lamq".to_string(),
        (
            REVISION.to_string(),
            files(&[
                (
                    "manifest.json",
                    format!(r#"{{"format": "{format}"}}"#).into_bytes(),
                ),
                ("layer_00_packed.npz", vec![7; 3000]),
                ("steering.safetensors", vec![1; 100]),
                ("tokenizer.json", b"{}".to_vec()),
                ("README.md", b"readme".to_vec()),
            ]),
        ),
    )
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbospark-ink-repo-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn the_probe_reports_a_lamq_release_as_runnable_and_sizes_only_what_it_downloads() {
    let _guard = ENDPOINT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _server = serve([lamq_release(repack::LAMQ_INKLING2_FORMAT)].into());
    let client = Client::with_timeout(Duration::from_secs(3));
    let repo = RepoRef::new("owner/lamq", "main");
    let report = probe(&client, &repo, None, None).unwrap();
    assert!(report.verdict.is_runnable(), "{:?}", report.verdict);
    assert_eq!(report.kind, SourceKind::Lamq);
    assert_eq!(report.family, Some(model_io::ModelFamily::Inkling2Loop));
    // manifest (34 bytes) + npz + steering + tokenizer; README excluded.
    let manifest_len = format!(r#"{{"format": "{}"}}"#, repack::LAMQ_INKLING2_FORMAT).len() as u64;
    assert_eq!(report.download_bytes, Some(manifest_len + 3000 + 100 + 2));
    // Pinned to the commit, not "main".
    assert_eq!(report.repo.revision, REVISION);
}

#[test]
fn a_manifest_with_another_format_is_not_this_route() {
    let _guard = ENDPOINT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _server = serve([lamq_release("opgraph-packed-1")].into());
    let client = Client::with_timeout(Duration::from_secs(3));
    let repo = RepoRef::new("owner/lamq", "main");
    assert_eq!(
        detect_inkling2_repo(&client, &repo, None, None).unwrap(),
        None
    );
    // The ordinary probe then answers, and it is not an Inkling report.
    if let Ok(report) = probe(&client, &repo, None, None) {
        assert_ne!(report.kind, SourceKind::Lamq);
    }
}

#[test]
fn an_inkling_gguf_takes_its_tokenizer_from_the_base_repo() {
    let _guard = ENDPOINT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let weights = (
        "owner/gguf".to_string(),
        (
            REVISION.to_string(),
            files(&[("q/m.gguf", gguf("inkling2_loop"))]),
        ),
    );
    let _server = serve([weights, tokenizer_repo()].into());
    let client = Client::with_timeout(Duration::from_secs(3));
    let repo = RepoRef::new("owner/gguf", "main");
    // The CLI passes the weights repo as its default sidecar: that means
    // the base repository, since this one has no tokenizer.json.
    let source = detect_inkling2_repo(&client, &repo, None, Some(&repo))
        .unwrap()
        .expect("an Inkling GGUF");
    assert_eq!(source.kind, Inkling2RepoKind::Gguf);
    assert_eq!(source.tokenizer_repo.repo, INKLING2_TOKENIZER_REPO);
    assert_eq!(source.tokenizer_repo.revision, TOK_REVISION);
    assert_eq!(
        source.tokenizer_files,
        ["tokenizer.json", "chat_template.jinja"]
    );
    let report = probe(&client, &repo, None, Some(&repo)).unwrap();
    assert!(report.verdict.is_runnable(), "{:?}", report.verdict);
    assert_eq!(report.kind, SourceKind::Gguf);
    assert_eq!(report.file.as_deref(), Some("q/m.gguf"));
}

#[test]
fn a_gguf_of_another_architecture_is_not_this_route() {
    let _guard = ENDPOINT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let weights = (
        "owner/gguf".to_string(),
        (REVISION.to_string(), files(&[("m.gguf", gguf("llama"))])),
    );
    let _server = serve([weights, tokenizer_repo()].into());
    let client = Client::with_timeout(Duration::from_secs(3));
    let repo = RepoRef::new("owner/gguf", "main");
    assert_eq!(
        detect_inkling2_repo(&client, &repo, None, None).unwrap(),
        None
    );
}

#[test]
fn a_failed_intake_keeps_the_download_and_a_retry_fetches_nothing_again() {
    let _guard = ENDPOINT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let weights = (
        "owner/gguf".to_string(),
        (
            REVISION.to_string(),
            files(&[("m.gguf", gguf("inkling2_loop"))]),
        ),
    );
    let server = serve([weights, tokenizer_repo()].into());
    let client = Client::with_timeout(Duration::from_secs(3));
    let repo = RepoRef::new("owner/gguf", "main");
    let source = detect_inkling2_repo(&client, &repo, None, None)
        .unwrap()
        .unwrap();

    let home = tmp("install");
    let store = Store::new(home.join("store"));
    let env = turbospark_catalog::library::LocationEnv::system(store.root()).unwrap();
    let target = install_destination(&store, None, InstallKind::Text, "ink-test", &env).unwrap();
    let mut bytes = 0u64;
    // A header with no tensors: the intake refuses it after the download.
    let err = install_inkling2_repo(
        &source,
        "ink-test",
        &target,
        &client,
        |_| {},
        &mut |n| bytes += n,
        None,
        &store,
        &env,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("kept in"), "{err}");
    assert_eq!(bytes, source.download_bytes());
    assert!(!target.dir.exists(), "nothing is left at the install path");
    let dl = std::fs::read_dir(store.root())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with(".inkling2-download-"))
        })
        .expect("the download folder is kept");
    assert_eq!(
        std::fs::read(dl.join("m.gguf")).unwrap(),
        gguf("inkling2_loop")
    );
    assert!(dl.join("tokenizer/tokenizer.json").is_file());

    let before = server.requests().len();
    let mut again = 0u64;
    let _ = install_inkling2_repo(
        &source,
        "ink-test",
        &target,
        &client,
        |_| {},
        &mut |n| again += n,
        None,
        &store,
        &env,
    );
    let gguf_gets = server.requests()[before..]
        .iter()
        .filter(|r| r.contains("/resolve/") && r.ends_with("m.gguf"))
        .count();
    assert_eq!(gguf_gets, 0, "a file already there at its size is skipped");
    assert_eq!(again, source.download_bytes(), "skipped bytes still count");

    let _ = std::fs::remove_dir_all(&home);
}
