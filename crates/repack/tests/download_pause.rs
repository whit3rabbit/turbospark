//! Pause must stop new HTTP ranges without restarting completed ranges, and
//! cancellation must release parked workers without sending another request.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use repack::{CancelFlag, HttpRangeSource, RangeSource};
use turbospark_repack as repack;

#[test]
fn pause_preserves_completed_ranges_and_resume_fetches_only_the_next_range() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/weights", listener.local_addr().unwrap());
    let (requests_tx, requests_rx) = mpsc::channel();
    let server = std::thread::spawn(move || {
        for expected in ["bytes=0-15", "bytes=16-31"] {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (mut socket, _) = loop {
                match listener.accept() {
                    Ok(pair) => break pair,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "missing range request");
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("accept: {e}"),
                }
            };
            socket.set_nonblocking(false).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                socket.read_exact(&mut byte).unwrap();
                request.extend(byte);
            }
            let request = String::from_utf8(request).unwrap().to_ascii_lowercase();
            assert!(
                request.contains(&format!("range: {expected}\r\n")),
                "{request}"
            );
            requests_tx.send(expected).unwrap();
            socket.write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Length: 16\r\nConnection: close\r\n\r\n").unwrap();
            let offset = if expected == "bytes=0-15" { 0 } else { 16 };
            socket
                .write_all(&(offset..offset + 16).collect::<Vec<u8>>())
                .unwrap();
        }
    });

    let flag = CancelFlag::new();
    let source = HttpRangeSource::new(url).with_cancel(flag.clone());
    assert_eq!(
        source.read_range(0, 16).unwrap(),
        (0..16).collect::<Vec<u8>>()
    );
    assert_eq!(
        requests_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        "bytes=0-15"
    );
    assert!(flag.pause());
    let (entered_tx, entered_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        entered_tx.send(()).unwrap();
        source.read_range(16, 32)
    });
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let premature = requests_rx.recv_timeout(Duration::from_millis(150));
    assert!(flag.resume());
    let output = reader.join().unwrap().unwrap();
    server.join().unwrap();
    assert!(premature.is_err(), "pause allowed a new range to start");
    assert_eq!(output, (16..32).collect::<Vec<u8>>());
    assert_eq!(
        requests_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        "bytes=16-31"
    );
}

#[test]
fn cancelling_a_paused_range_wakes_the_worker_and_cannot_be_resumed() {
    let flag = CancelFlag::new();
    flag.pause();
    let source =
        HttpRangeSource::new("http://127.0.0.1:1/never-requested").with_cancel(flag.clone());
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || tx.send(source.read_range(0, 16)).unwrap());
    let premature = rx.recv_timeout(Duration::from_millis(100));
    flag.cancel();
    let result = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("cancel must wake a paused worker");
    worker.join().unwrap();
    assert!(premature.is_err(), "paused worker unexpectedly finished");
    assert!(matches!(result, Err(repack::DownloadError::Cancelled)));
    assert!(!flag.resume());
    assert!(!flag.pause());
    assert!(
        CancelFlag::new().checkpoint().is_ok(),
        "a cancelled walk cannot poison the next one"
    );
}

#[test]
fn resume_releases_all_workers_sharing_a_flag() {
    let flag = Arc::new(CancelFlag::new());
    flag.pause();
    let (tx, rx) = mpsc::channel();
    let workers = (0..4)
        .map(|_| {
            let flag = Arc::clone(&flag);
            let tx = tx.clone();
            std::thread::spawn(move || tx.send(flag.checkpoint()).unwrap())
        })
        .collect::<Vec<_>>();
    let premature = rx.recv_timeout(Duration::from_millis(100));
    flag.resume();
    for _ in 0..4 {
        assert!(rx
            .recv_timeout(Duration::from_secs(2))
            .expect("resume must wake every worker")
            .is_ok());
    }
    for worker in workers {
        worker.join().unwrap();
    }
    assert!(premature.is_err(), "pause allowed a worker through");
}
