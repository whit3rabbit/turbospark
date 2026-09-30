//! Shared header-only split discovery for probe and installation.
use crate::hf::{Client, RepoRef};
use repack::{ByteProgressCallback, CancelFlag, GgufSet, HttpRangeSource};
use std::path::Path;

pub(crate) fn load(
    client: &Client,
    repo: &RepoRef,
    file: &str,
    progress: Option<&ByteProgressCallback>,
    cancel: Option<&CancelFlag>,
    cache_dir: Option<&Path>,
) -> Result<GgufSet<HttpRangeSource>, String> {
    load_inner(client, repo, file, progress, cancel, cache_dir, None)
}

/// Loads exactly the selected variant's files after the first header reveals
/// the GGUF shard topology. A mismatch is rejected before any sibling shard
/// is fetched.
pub(crate) fn load_selected(
    client: &Client,
    repo: &RepoRef,
    file: &str,
    selected_files: &[String],
    progress: Option<&ByteProgressCallback>,
    cancel: Option<&CancelFlag>,
    cache_dir: Option<&Path>,
) -> Result<GgufSet<HttpRangeSource>, String> {
    load_inner(
        client,
        repo,
        file,
        progress,
        cancel,
        cache_dir,
        Some(selected_files),
    )
}

fn load_inner(
    client: &Client,
    repo: &RepoRef,
    file: &str,
    progress: Option<&ByteProgressCallback>,
    cancel: Option<&CancelFlag>,
    cache_dir: Option<&Path>,
    selected_files: Option<&[String]>,
) -> Result<GgufSet<HttpRangeSource>, String> {
    let first =
        crate::stream::range_source(repo.file_url(file), progress, client, cancel, cache_dir);
    let header = repack::fetch_gguf_header(&first).map_err(|e| format!("{file}: {e}"))?;
    let names = repack::gguf_shard_names(file, &header)?;
    if let Some(selected_files) = selected_files {
        let mut selected = selected_files.to_vec();
        selected.sort();
        selected.dedup();
        let mut discovered = names.clone();
        discovered.sort();
        if selected.len() != selected_files.len() || selected != discovered {
            return Err(format!(
                "GGUF header declares files {discovered:?}, which do not exactly match the selected variant files {selected:?}"
            ));
        }
    }
    let mut shards = Vec::with_capacity(names.len());
    let mut initial = Some((header, first));
    for name in &names {
        if cancel.is_some_and(|flag| flag.is_cancelled()) {
            return Err(crate::install::INSTALL_CANCELLED.into());
        }
        let url = repo.file_url(name);
        let (header, source) = match initial.take() {
            Some(pair) => pair,
            None => {
                let source =
                    crate::stream::range_source(url.clone(), progress, client, cancel, cache_dir);
                let header =
                    repack::fetch_gguf_header(&source).map_err(|e| format!("{name}: {e}"))?;
                (header, source)
            }
        };
        let length = client
            .content_length(&url)?
            .ok_or_else(|| format!("{name}: missing content length"))?;
        shards.push((header, source, length));
    }
    GgufSet::new(shards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
    const SHARD_BYTES: usize = 1 << 20;
    static ENDPOINT_LOCK: Mutex<()> = Mutex::new(());

    struct FixtureServer {
        endpoint: String,
        requests: Arc<Mutex<Vec<String>>>,
        stop: Arc<AtomicBool>,
        thread: Option<JoinHandle<()>>,
    }

    impl FixtureServer {
        fn requests(&self) -> Vec<String> {
            self.requests.lock().expect("request log lock").clone()
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
            crate::set_hf_endpoint_override(None);
        }
    }

    fn fixture_server(files: BTreeMap<String, Vec<u8>>) -> FixtureServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("fixture bind");
        listener
            .set_nonblocking(true)
            .expect("set listener nonblocking");
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
                            .expect("set fixture read timeout");
                        let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                        let mut request_line = String::new();
                        reader
                            .read_line(&mut request_line)
                            .expect("read fixture request line");
                        let mut range = None;
                        loop {
                            let mut line = String::new();
                            reader.read_line(&mut line).expect("read fixture header");
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
                        let path = target.split('?').next().unwrap_or(target);
                        thread_requests
                            .lock()
                            .expect("request log lock")
                            .push(format!("{method} {path}"));
                        let response = fixture_response(method, path, range.as_deref(), &files);
                        stream.write_all(&response).expect("write fixture response");
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
        path: &str,
        range: Option<&str>,
        files: &BTreeMap<String, Vec<u8>>,
    ) -> Vec<u8> {
        let Some(name) = path.strip_prefix(&format!("/owner/model/resolve/{REVISION}/")) else {
            return response(404, "Not Found", b"not found", &[]);
        };
        let Some(bytes) = files.get(name) else {
            return response(404, "Not Found", b"not found", &[]);
        };
        if method.eq_ignore_ascii_case("HEAD") {
            return response(
                200,
                "OK",
                b"",
                &[("x-linked-size", bytes.len().to_string())],
            );
        }
        if let Some(range) = range {
            let requested = range.strip_prefix("bytes=").expect("range prefix");
            let (start, end) = requested.split_once('-').expect("range endpoints");
            let start: usize = start.parse().expect("range start");
            let end: usize = end.parse().expect("range end");
            if start >= bytes.len() || start > end {
                return response(416, "Range Not Satisfiable", b"", &[]);
            }
            let end = end.min(bytes.len() - 1);
            let body = &bytes[start..=end];
            return response(
                206,
                "Partial Content",
                body,
                &[(
                    "Content-Range",
                    format!("bytes {start}-{end}/{}", bytes.len()),
                )],
            );
        }
        response(200, "OK", bytes, &[])
    }

    fn response(status: u16, reason: &str, body: &[u8], headers: &[(&str, String)]) -> Vec<u8> {
        let mut wire = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
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

    fn split_shard(split_no: u32) -> Vec<u8> {
        let (mut bytes, _) = repack::GgufBuilder::new()
            .metadata_u32("split.count", 2)
            .metadata_u32("split.no", split_no)
            .metadata_u32("split.tensors.count", 0)
            .build();
        // The header fetch starts with a fixed range; avoid testing the
        // downloader's short-read retry ladder when the header is already
        // complete.
        bytes.resize(SHARD_BYTES, 0);
        bytes
    }

    fn set_endpoint(server: &FixtureServer) -> (EndpointReset, Client) {
        crate::set_hf_endpoint_override(Some(server.endpoint.clone()));
        (EndpointReset, Client::with_timeout(Duration::from_secs(3)))
    }

    fn file_requests(server: &FixtureServer) -> Vec<String> {
        server
            .requests()
            .into_iter()
            .filter(|request| request.contains("/owner/model/resolve/"))
            .collect()
    }

    #[test]
    fn selected_split_load_requests_both_selected_shards_and_no_other_variant() {
        let _lock = ENDPOINT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first = "alpha-Q8_0-00001-of-00002.gguf";
        let second = "alpha-Q8_0-00002-of-00002.gguf";
        let unrelated = "beta-Q8_0-00001-of-00002.gguf";
        let server = fixture_server(BTreeMap::from([
            (first.to_string(), split_shard(0)),
            (second.to_string(), split_shard(1)),
            (unrelated.to_string(), split_shard(0)),
        ]));
        let (_reset, client) = set_endpoint(&server);
        let repo = RepoRef::new("owner/model", REVISION);
        let selected = vec![first.to_string(), second.to_string()];

        let loaded = load_selected(&client, &repo, first, &selected, None, None, None)
            .expect("complete selected shard set loads");

        assert_eq!(loaded.bytes, (SHARD_BYTES * 2) as u64);
        let requests = file_requests(&server);
        assert!(requests.iter().any(|request| request.ends_with(first)));
        assert!(requests.iter().any(|request| request.ends_with(second)));
        assert!(
            requests.iter().all(|request| !request.ends_with(unrelated)),
            "no request should cross into another same-label variant: {requests:?}"
        );
    }

    #[test]
    fn selected_split_mismatch_fails_before_fetching_an_unselected_sibling() {
        let _lock = ENDPOINT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first = "alpha-Q8_0-00001-of-00002.gguf";
        let unselected_sibling = "alpha-Q8_0-00002-of-00002.gguf";
        let other_variant = "beta-Q8_0-00001-of-00002.gguf";
        let server = fixture_server(BTreeMap::from([
            (first.to_string(), split_shard(0)),
            (unselected_sibling.to_string(), split_shard(1)),
            (other_variant.to_string(), split_shard(0)),
        ]));
        let (_reset, client) = set_endpoint(&server);
        let repo = RepoRef::new("owner/model", REVISION);
        let selected = vec![first.to_string(), other_variant.to_string()];

        let result = load_selected(&client, &repo, first, &selected, None, None, None);
        let requests = file_requests(&server);
        assert!(requests.iter().any(|request| request.ends_with(first)));
        assert!(
            requests
                .iter()
                .all(|request| !request.ends_with(unselected_sibling)),
            "the rejected selected-file set must stop before its unselected sibling: {requests:?}"
        );
        let error = match result {
            Ok(_) => panic!("mismatched selected files must fail"),
            Err(error) => error,
        };
        assert!(error.contains("do not exactly match"), "{error}");
        assert!(
            requests
                .iter()
                .all(|request| !request.ends_with(other_variant)),
            "the unrelated selected-looking file must never be fetched: {requests:?}"
        );
    }
}
