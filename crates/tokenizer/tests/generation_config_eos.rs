//! `generation_config.json` is the authority for a checkpoint's FULL
//! end-of-sequence set: `tokenizer_config.json` only ever carries one
//! `eos_token` string, so a multi-stop checkpoint (the real Gemma 4
//! 26B-A4B declares `eos_token_id: [1, 106, 50]`) looks single-stop
//! without it. These tests copy the ChatML fixture into a temp dir and
//! vary the sidecar.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_tokenizer::MfTokenizer;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer")
}

fn temp_fixture_copy() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-genconfig-eos-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(fixture_dir().join(name), dir.join(name)).unwrap();
    }
    dir
}

#[test]
fn eos_token_id_array_extends_the_stop_set() {
    let dir = temp_fixture_copy();
    let baseline = MfTokenizer::load_from_dir(&dir).expect("loads without sidecar");
    assert!(!baseline.stop_token_ids.contains(&7));
    assert!(!baseline.stop_token_ids.contains(&9));

    std::fs::write(
        dir.join("generation_config.json"),
        r#"{"eos_token_id": [7, 9], "temperature": 1.0}"#,
    )
    .unwrap();
    let tok = MfTokenizer::load_from_dir(&dir).expect("loads with sidecar");
    assert!(tok.stop_token_ids.contains(&7));
    assert!(tok.stop_token_ids.contains(&9));
    // The dialect's own stops are kept, not replaced.
    assert!(tok.stop_token_ids.is_superset(&baseline.stop_token_ids));
}

#[test]
fn eos_token_id_single_integer_form_is_accepted() {
    let dir = temp_fixture_copy();
    std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 7}"#).unwrap();
    let tok = MfTokenizer::load_from_dir(&dir).expect("loads");
    assert!(tok.stop_token_ids.contains(&7));
}

#[test]
fn invalid_or_missing_sidecar_is_tolerated() {
    let dir = temp_fixture_copy();
    std::fs::write(dir.join("generation_config.json"), "not json").unwrap();
    let tok = MfTokenizer::load_from_dir(&dir).expect("invalid sidecar ignored");
    let dir2 = temp_fixture_copy();
    let baseline = MfTokenizer::load_from_dir(&dir2).expect("no sidecar");
    assert_eq!(tok.stop_token_ids, baseline.stop_token_ids);
    // Negative ids are dropped rather than poisoning the set.
    std::fs::write(
        dir.join("generation_config.json"),
        r#"{"eos_token_id": [-1, 7]}"#,
    )
    .unwrap();
    let tok = MfTokenizer::load_from_dir(&dir).expect("loads");
    assert!(!tok.stop_token_ids.contains(&-1));
    assert!(tok.stop_token_ids.contains(&7));
}

#[test]
fn oversized_sidecar_is_ignored() {
    let dir = temp_fixture_copy();
    let baseline = MfTokenizer::load_from_dir(&dir).expect("loads without sidecar");
    let mut sidecar = br#"{"eos_token_id": [7]}"#.to_vec();
    sidecar.resize((1 << 20) + 1, b' ');
    std::fs::write(dir.join("generation_config.json"), sidecar).unwrap();

    let tok = MfTokenizer::load_from_dir(&dir).expect("oversized sidecar ignored");
    assert_eq!(tok.stop_token_ids, baseline.stop_token_ids);
}

#[test]
fn excessive_or_out_of_vocab_eos_ids_are_ignored() {
    let dir = temp_fixture_copy();
    let baseline = MfTokenizer::load_from_dir(&dir).expect("loads without sidecar");
    let ids = std::iter::repeat_n("7", 257).collect::<Vec<_>>().join(",");
    std::fs::write(
        dir.join("generation_config.json"),
        format!(r#"{{"eos_token_id": [{ids}]}}"#),
    )
    .unwrap();
    let tok = MfTokenizer::load_from_dir(&dir).expect("excessive array ignored");
    assert_eq!(tok.stop_token_ids, baseline.stop_token_ids);

    std::fs::write(
        dir.join("generation_config.json"),
        r#"{"eos_token_id": [9, 1000000, 4294967303]}"#,
    )
    .unwrap();
    let tok = MfTokenizer::load_from_dir(&dir).expect("invalid ids ignored");
    assert!(tok.stop_token_ids.contains(&9));
    assert!(!tok.stop_token_ids.contains(&7));
    assert!(!tok.stop_token_ids.contains(&1_000_000));
    assert_eq!(tok.stop_token_ids.len(), baseline.stop_token_ids.len() + 1);
}
