//! Checkpoint sidecars (`tokenizer_config.json`, `chat_template.jinja`,
//! `chat_template.json`) come from a downloaded, untrusted model directory.
//! They are read whole into memory, so each read is bounded the way
//! `generation_config.json` already is: an over-cap file is ignored as if
//! absent rather than loaded. The cap is far above any real sidecar
//! (installed ones are under 20 KB; HF's largest known
//! `tokenizer_config.json` is about 1.2 MB).
//!
//! Each case writes a template that renders a marker. Under the cap the
//! marker must appear (the observable works); one byte over the cap it must
//! not (the file was skipped and the dialect renderer took over).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_tokenizer::{Message, MfTokenizer, Role};

const CAP: usize = 16 << 20;
const MARK: &str = "SIDECAR-MARK";

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir_with_tokenizer() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir =
        std::env::temp_dir().join(format!("turbospark-sidecar-cap-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ChatMLTokenizer");
    std::fs::copy(fixture.join("tokenizer.json"), dir.join("tokenizer.json")).unwrap();
    dir
}

/// JSON whitespace padding keeps the document valid at any length.
fn padded(mut bytes: Vec<u8>, len: usize) -> Vec<u8> {
    assert!(bytes.len() <= len);
    bytes.resize(len, b' ');
    bytes
}

fn render(dir: &std::path::Path) -> String {
    let tok = MfTokenizer::load_from_dir(dir).expect("loads");
    let out = tok
        .apply_chat_template(&[Message::new(Role::User, "hi")])
        .expect("renders");
    std::fs::remove_dir_all(dir).ok();
    out
}

fn jinja(len: usize) -> Vec<u8> {
    // The padding sits inside a Jinja comment so it renders to nothing.
    let mut body = format!("{MARK}{{#").into_bytes();
    body.resize(len - 2, b' ');
    body.extend_from_slice(b"#}");
    body
}

fn template_json(len: usize) -> Vec<u8> {
    padded(
        format!(r#"{{"chat_template": "{MARK}"}}"#).into_bytes(),
        len,
    )
}

#[test]
fn chat_template_jinja_over_cap_is_ignored() {
    let dir = temp_dir_with_tokenizer();
    std::fs::write(dir.join("chat_template.jinja"), jinja(CAP)).unwrap();
    assert!(render(&dir).contains(MARK), "at-cap template must load");

    let dir = temp_dir_with_tokenizer();
    std::fs::write(dir.join("chat_template.jinja"), jinja(CAP + 1)).unwrap();
    assert!(!render(&dir).contains(MARK), "over-cap template was loaded");
}

#[test]
fn chat_template_json_over_cap_is_ignored() {
    let dir = temp_dir_with_tokenizer();
    std::fs::write(dir.join("chat_template.json"), template_json(CAP)).unwrap();
    assert!(render(&dir).contains(MARK), "at-cap template must load");

    let dir = temp_dir_with_tokenizer();
    std::fs::write(dir.join("chat_template.json"), template_json(CAP + 1)).unwrap();
    assert!(!render(&dir).contains(MARK), "over-cap template was loaded");
}

#[test]
fn tokenizer_config_over_cap_is_ignored() {
    let dir = temp_dir_with_tokenizer();
    std::fs::write(dir.join("tokenizer_config.json"), template_json(CAP)).unwrap();
    assert!(render(&dir).contains(MARK), "at-cap config must load");

    let dir = temp_dir_with_tokenizer();
    std::fs::write(dir.join("tokenizer_config.json"), template_json(CAP + 1)).unwrap();
    assert!(!render(&dir).contains(MARK), "over-cap config was loaded");
}
