//! A safetensors header's `data_offsets` are attacker-controlled and are never
//! compared to the file, so `HttpRangeSource::read_range` is the last place a
//! declared range can be refused before it sizes the chunk table and the output
//! buffer. These tests drive it over a real socket because the length it checks
//! against comes from the server's `Content-Range`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use repack::{fetch_safetensors_header, DownloadError, HttpRangeSource, RangeSource};
use turbospark_repack as repack;

/// Large enough that the pre-fix `vec![0u8; len]` fails in the allocator and
/// aborts on this platform, small enough that the chunk table built before it
/// stays near 256 MiB instead of exhausting the machine if the guard regresses.
const HOSTILE_LEN: u64 = 1 << 48;

/// The client's chunk size (`ranged_download::chunks::MAX_RANGE_BYTES`, private
/// to the crate), which is also the size above which a read needs the length.
const MAX_RANGE_BYTES: u64 = 16 * 1024 * 1024;

/// Past one chunk, so these reads take the multi-chunk path.
const MULTI_CHUNK_LEN: usize = MAX_RANGE_BYTES as usize + 1024 * 1024;

#[derive(Clone, Copy, PartialEq)]
enum TotalHeader {
    Real,
    /// RFC 9110 allows `*` when the sender does not know the length.
    Star,
    Omitted,
}

/// Minimal range server. Its request log is what lets a test tell "refused
/// before any request" from "the server answered and the read then failed".
struct RangeServer {
    url: String,
    requests: Arc<RequestLog>,
}

#[derive(Default)]
struct RequestLog {
    ranges: Mutex<Vec<String>>,
    arrived: Condvar,
}

impl RangeServer {
    fn start(file: Vec<u8>, total: TotalHeader) -> Self {
        Self::start_with_overlap(file, total, false)
    }

    fn start_with_overlap(file: Vec<u8>, total: TotalHeader, require_overlap: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/weights", listener.local_addr().unwrap());
        let requests = Arc::new(RequestLog::default());
        let log = Arc::clone(&requests);
        let file = Arc::new(file);
        // Detached on purpose: it dies with the test process and every socket
        // is `Connection: close`, so nothing outlives a request.
        std::thread::spawn(move || {
            for socket in listener.incoming() {
                let Ok(socket) = socket else { return };
                let file = Arc::clone(&file);
                let log = Arc::clone(&log);
                std::thread::spawn(move || serve(socket, &file, total, &log, require_overlap));
            }
        });
        Self { url, requests }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.ranges.lock().unwrap().clone()
    }

    fn clear_requests(&self) {
        self.requests.ranges.lock().unwrap().clear();
    }
}

fn serve(
    mut socket: TcpStream,
    file: &[u8],
    total: TotalHeader,
    log: &RequestLog,
    require_overlap: bool,
) {
    let mut request = Vec::new();
    let mut byte = [0u8; 1];
    while !request.ends_with(b"\r\n\r\n") {
        if socket.read_exact(&mut byte).is_err() {
            return;
        }
        request.push(byte[0]);
    }
    let request = String::from_utf8(request).unwrap().to_ascii_lowercase();
    let range = request
        .lines()
        .find_map(|line| line.strip_prefix("range: bytes="))
        .expect("every source request is ranged")
        .to_string();
    {
        let mut ranges = log.ranges.lock().unwrap();
        ranges.push(range.clone());
        if require_overlap && range != "0-0" {
            // Hold the first payload response until the second request arrives.
            // The one-byte length probe does not count as an overlapping fetch.
            log.arrived.notify_all();
            let timed_out = log
                .arrived
                .wait_timeout_while(ranges, Duration::from_secs(3), |ranges| {
                    ranges
                        .iter()
                        .filter(|range| range.as_str() != "0-0")
                        .count()
                        < 2
                })
                .unwrap()
                .1
                .timed_out();
            if timed_out {
                let _ = socket.write_all(
                    b"HTTP/1.1 409 Conflict\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                return;
            }
        }
    }

    let (start, end) = range.split_once('-').unwrap();
    let (start, end): (u64, u64) = (start.parse().unwrap(), end.parse().unwrap());
    if start >= file.len() as u64 {
        let _ = socket.write_all(
            b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        );
        return;
    }
    // Clamped like a real server, so a range past EOF is a short body.
    let end = end.min(file.len() as u64 - 1);
    let body = &file[start as usize..=end as usize];
    let content_range = match total {
        TotalHeader::Real => format!("Content-Range: bytes {start}-{end}/{}\r\n", file.len()),
        TotalHeader::Star => format!("Content-Range: bytes {start}-{end}/*\r\n"),
        TotalHeader::Omitted => String::new(),
    };
    let head = format!(
        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n{content_range}Connection: close\r\n\r\n",
        body.len()
    );
    if socket.write_all(head.as_bytes()).is_ok() {
        let _ = socket.write_all(body);
    }
}

/// A valid safetensors file whose one tensor claims `HOSTILE_LEN` bytes while
/// the file holds only its header and four bytes of data.
fn hostile_safetensors() -> Vec<u8> {
    let json = format!(
        r#"{{"t":{{"dtype":"U8","shape":[{HOSTILE_LEN}],"data_offsets":[0,{HOSTILE_LEN}]}}}}"#
    );
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(json.as_bytes());
    file.extend_from_slice(&[1, 2, 3, 4]);
    file
}

fn patterned(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

fn scratch_dir(label: &str) -> std::path::PathBuf {
    static ID: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "turbospark-range-bounds-{label}-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ))
}

/// The finding itself: header, then the same `absolute_range` plus `read_range`
/// pair every safetensors walk uses to read a tensor.
#[test]
fn a_tensor_declared_past_the_shard_is_refused_before_it_is_allocated() {
    let file = hostile_safetensors();
    let file_len = file.len() as u64;
    let server = RangeServer::start(file, TotalHeader::Real);
    let source = HttpRangeSource::new(server.url.clone());

    let header = fetch_safetensors_header(&source).expect("the header itself is well formed");
    let (start, end) = header.absolute_range("t").expect("offsets do not overflow");
    assert_eq!(end - start, HOSTILE_LEN);

    server.clear_requests();
    assert_eq!(
        source.read_range(start, end),
        Err(DownloadError::ShortRead {
            expected: HOSTILE_LEN,
            actual: file_len - start,
        })
    );
    assert!(
        server.requests().is_empty(),
        "the header read already told the source its length: {:?}",
        server.requests()
    );
}

/// A resumed walk reads its header from the range cache, so no response has
/// told this source its length. That is the path a check based only on observed
/// responses would miss, so it must ask.
#[test]
fn a_source_whose_header_came_from_the_cache_probes_before_allocating() {
    let file = hostile_safetensors();
    let file_len = file.len() as u64;
    let cache = scratch_dir("cache");
    let server = RangeServer::start(file, TotalHeader::Real);

    let first = HttpRangeSource::new(server.url.clone()).with_cache_dir(&cache);
    fetch_safetensors_header(&first).expect("first walk populates the cache");

    server.clear_requests();
    let resumed = HttpRangeSource::new(server.url.clone()).with_cache_dir(&cache);
    let header = fetch_safetensors_header(&resumed).expect("cached header");
    assert!(
        server.requests().is_empty(),
        "the resumed header must be a cache hit or this test proves nothing: {:?}",
        server.requests()
    );

    let (start, end) = header.absolute_range("t").unwrap();
    assert_eq!(
        resumed.read_range(start, end),
        Err(DownloadError::ShortRead {
            expected: HOSTILE_LEN,
            actual: file_len - start,
        })
    );
    assert_eq!(server.requests(), ["0-0"], "one probe and no chunk reads");
    let _ = std::fs::remove_dir_all(cache);
}

/// The check must not refuse real data: a multi-chunk read ending exactly at
/// EOF is the last tensor of every real shard.
#[test]
fn a_multi_chunk_read_ending_exactly_at_eof_returns_the_file() {
    let file = patterned(MULTI_CHUNK_LEN);
    let server = RangeServer::start(file.clone(), TotalHeader::Real);
    let len = file.len() as u64;

    let fresh = HttpRangeSource::new(server.url.clone());
    assert_eq!(fresh.read_range(0, len).unwrap(), file);
    let probes = server.requests().iter().filter(|r| *r == "0-0").count();
    assert_eq!(
        probes, 1,
        "an oversized first read asks for the length once"
    );

    server.clear_requests();
    assert_eq!(fresh.read_range(0, len).unwrap(), file);
    assert!(
        !server.requests().iter().any(|r| r == "0-0"),
        "the length is pinned after the first response: {:?}",
        server.requests()
    );
}

#[test]
fn multi_chunk_reads_overlap_http_requests() {
    let file = patterned(MULTI_CHUNK_LEN);
    let server = RangeServer::start_with_overlap(file.clone(), TotalHeader::Real, true);
    let source = HttpRangeSource::new(server.url);

    let received = source
        .read_range(0, file.len() as u64)
        .expect("the server requires two overlapping requests");
    assert_eq!(received, file);
}

#[test]
fn multi_chunk_file_downloads_overlap_http_requests() {
    let file = patterned(MULTI_CHUNK_LEN);
    let server = RangeServer::start_with_overlap(file.clone(), TotalHeader::Real, true);
    let source = HttpRangeSource::new(server.url);
    let dir = scratch_dir("overlap");
    let destination = dir.join("weights");

    let downloaded = source.download_to(&destination, file.len() as u64);
    let received = std::fs::read(&destination);
    std::fs::remove_dir_all(dir).unwrap();
    assert_eq!(
        downloaded.expect("the server requires two overlapping requests"),
        file.len() as u64
    );
    assert_eq!(received.unwrap(), file);
}

/// One byte past EOF is refused, and `actual` stays usable:
/// `fetch_gguf_header` re-reads exactly `actual` bytes when a speculative read
/// overruns a small file, and that retry must still reach the whole file.
#[test]
fn one_byte_past_eof_reports_what_the_file_does_hold() {
    let file = patterned(MULTI_CHUNK_LEN);
    let server = RangeServer::start(file.clone(), TotalHeader::Real);
    let len = file.len() as u64;
    let source = HttpRangeSource::new(server.url.clone());

    assert_eq!(
        source.read_range(0, len + 1),
        Err(DownloadError::ShortRead {
            expected: len + 1,
            actual: len,
        })
    );
    assert_eq!(source.read_range(0, len).unwrap(), file);
}

/// A read of exactly one chunk has a bounded buffer whatever the header claims,
/// so it stays the single request it always was. Every small tensor and header
/// read of a walk takes this path, so a probe here would cost one per source.
#[test]
fn a_read_of_one_chunk_is_not_probed() {
    let file = patterned(MAX_RANGE_BYTES as usize);
    let server = RangeServer::start(file.clone(), TotalHeader::Real);
    let source = HttpRangeSource::new(server.url.clone());

    assert_eq!(source.read_range(0, MAX_RANGE_BYTES).unwrap(), file);
    assert_eq!(
        server.requests(),
        [format!("0-{}", MAX_RANGE_BYTES - 1)],
        "exactly the chunk, with no probe before it"
    );
}

/// A server that reports no total cannot be checked, which must not cost a real
/// tensor on such a mirror. Only a claim no real tensor reaches is refused, and
/// it is refused after the probe, without a chunk read.
#[test]
fn a_server_without_a_total_keeps_real_reads_and_refuses_absurd_ones() {
    for total in [TotalHeader::Star, TotalHeader::Omitted] {
        let file = patterned(MULTI_CHUNK_LEN);
        let server = RangeServer::start(file.clone(), total);
        let source = HttpRangeSource::new(server.url.clone());

        assert_eq!(source.read_range(0, file.len() as u64).unwrap(), file);

        server.clear_requests();
        assert_eq!(
            source.read_range(0, HOSTILE_LEN),
            Err(DownloadError::InvalidRange {
                start: 0,
                end_exclusive: HOSTILE_LEN,
            })
        );
        assert_eq!(server.requests(), ["0-0"]);
    }
}

/// A pinned source was told its authoritative size up front, so it needs no
/// response at all. Port 1 is unreachable: reaching the network would surface
/// as `Request`, not `ShortRead`.
#[test]
fn a_pinned_source_refuses_from_its_pinned_size_without_a_request() {
    let source = HttpRangeSource::new("http://127.0.0.1:1/nothing")
        .with_pinned_identity("pinned", 100, None);
    assert_eq!(
        source.read_range(90, HOSTILE_LEN),
        Err(DownloadError::ShortRead {
            expected: HOSTILE_LEN - 90,
            actual: 10,
        })
    );
    assert_eq!(
        source.read_range(HOSTILE_LEN, HOSTILE_LEN + 8),
        Err(DownloadError::ShortRead {
            expected: 8,
            actual: 0,
        })
    );
}
