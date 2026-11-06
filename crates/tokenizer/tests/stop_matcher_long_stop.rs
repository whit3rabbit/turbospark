//! Stop strings come straight from HTTP `stop` fields with no length cap, so
//! the withheld-suffix scan must stay linear in the stop length. The old scan
//! rebuilt a char suffix per candidate length (quadratic per push).

use std::time::{Duration, Instant};
use turbospark_tokenizer::StreamingStopMatcher;

/// Feeds `chunks` through a matcher and returns emitted text plus stop state.
fn reference(stops: &[&str], chunks: &[&str]) -> (String, bool) {
    let mut m = StreamingStopMatcher::new(stops.iter().map(|s| s.to_string()).collect());
    let mut out = String::new();
    for c in chunks {
        out += &m.push(c);
    }
    let stopped = m.is_stopped();
    out += &m.finish();
    (out, stopped)
}

#[test]
fn withheld_suffix_semantics_hold_for_overlapping_and_multibyte_stops() {
    // Overlapping prefix: "aab" withheld as "aa" until the "b" arrives.
    assert_eq!(
        reference(&["aab"], &["xaa", "ab"]),
        ("xa".to_string(), true)
    );
    assert_eq!(
        reference(&["abab"], &["zaba", "b"]),
        ("z".to_string(), true)
    );
    assert_eq!(
        reference(&["abab"], &["zaba", "c"]),
        ("zabac".to_string(), false)
    );
    // Multibyte: partial stop withheld, released at finish without splitting chars.
    let mut m = StreamingStopMatcher::new(vec!["\u{00e9}\u{00e8}X".to_string()]);
    assert_eq!(m.push("a\u{00e9}\u{00e8}"), "a");
    assert_eq!(m.push("Y"), "\u{00e9}\u{00e8}Y");
    assert!(!m.is_stopped());
    // Two stops: the longer withheld tail wins.
    let mut m = StreamingStopMatcher::new(vec!["xy".to_string(), "abcd".to_string()]);
    assert_eq!(m.push("qabc"), "q");
    assert_eq!(m.push("d"), "");
    assert!(m.is_stopped());
}

#[test]
fn long_adversarial_stop_string_scan_is_linear() {
    // Pending is N 'a's ending in 'c'; no suffix is a stop prefix, so the old
    // scan tried every length with an O(N) rebuild each time.
    let n = 60_000;
    let stop = "a".repeat(2 * n);
    let mut m = StreamingStopMatcher::new(vec![stop]);
    let start = Instant::now();
    let first = m.push(&"a".repeat(n));
    assert_eq!(first, "");
    let out = m.push("c");
    assert_eq!(out.len(), n + 1);
    for _ in 0..20 {
        m.push(&"a".repeat(n / 20));
    }
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "stop scan took {:?}",
        start.elapsed()
    );
}
