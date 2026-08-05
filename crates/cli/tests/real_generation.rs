#![cfg(target_os = "macos")]
//! Black-box test proving `mference-check` performs real generation: build
//! a `.gturbo` install with a bundled tokenizer, point `--model` at it with
//! `--prompt`, and check real generated text/token-count lines appear.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir =
        std::env::temp_dir().join(format!("mrefrust-cli-real-gen-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tokenizer_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tokenizer/tests/fixtures/ChatMLTokenizer")
}

#[test]
fn real_prompt_mode_generates_real_tokens() {
    let dir = temp_dir();
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ] {
        std::fs::copy(tokenizer_fixture_dir().join(name), dir.join(name)).unwrap();
    }
    let tok = tokenizer::MfTokenizer::load_from_dir(&dir).expect("tokenizer loads");
    repack::build_synthetic_gemma4_install(&dir, tok.vocab_size as i64, 2, "cli-real-gen")
        .expect("synthetic install writes");

    let output = Command::new(env!("CARGO_BIN_EXE_mference-check"))
        .args([
            "--model",
            dir.to_str().unwrap(),
            "--prompt",
            "hi",
            "--max-new",
            "3",
        ])
        .output()
        .expect("binary should run");

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("generating (real forward pass"));
    assert!(stdout.contains("generated, stop reason"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
}

#[test]
fn real_naming_gemma4_install_generates() {
    let dir = temp_dir();
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ] {
        std::fs::copy(tokenizer_fixture_dir().join(name), dir.join(name)).unwrap();
    }
    let tok = tokenizer::MfTokenizer::load_from_dir(&dir).expect("tokenizer loads");
    repack::build_synthetic_gemma4_real_install(
        &dir,
        tok.vocab_size as i64,
        2,
        2,
        2,
        8,
        "cli-real-naming",
    )
    .expect("real-naming install writes");

    let output = Command::new(env!("CARGO_BIN_EXE_mference-check"))
        .args([
            "--model",
            dir.to_str().unwrap(),
            "--prompt",
            "hi",
            "--max-new",
            "3",
        ])
        .output()
        .expect("binary should run");

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("generating (real forward pass"));
    assert!(stdout.contains("generated, stop reason"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
}
