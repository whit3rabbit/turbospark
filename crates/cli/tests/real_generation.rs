#![cfg(target_os = "macos")]
//! Black-box tests proving `turbospark-check` performs real generation: build
//! a `.gturbo` install with a bundled tokenizer, point `--model` at it, and
//! check real generated text/token-count lines appear. Covers all three
//! invocation modes (`--prompt`, `--messages-file`, `--chat`).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-cli-real-gen-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tokenizer_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tokenizer/tests/fixtures/ChatMLTokenizer")
}

/// A real-naming MoE install with the ChatML tokenizer bundled alongside it,
/// which is what the chat modes need: they render through the tokenizer's
/// own dialect template (ChatML here; the repo has no Gemma fixture).
fn install_with_tokenizer(label: &str) -> PathBuf {
    let dir = temp_dir();
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ] {
        std::fs::copy(tokenizer_fixture_dir().join(name), dir.join(name)).unwrap();
    }
    let tok = tokenizer::MfTokenizer::load_from_dir(&dir).expect("tokenizer loads");
    repack::build_synthetic_gemma4_real_install(&dir, tok.vocab_size as i64, 2, 2, 2, 8, label)
        .expect("real-naming install writes");
    dir
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

    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-check"))
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
    // The run summary is the shared footer now, on stderr, exactly as in
    // `--messages-file` mode.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("[stop="), "unexpected stderr: {stderr}");
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

    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-check"))
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
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("[stop="), "unexpected stderr: {stderr}");
}

/// `--kv-bits` threads all the way from the parser through
/// `open_session`'s `open_with_kv_quant` call to a real forward pass, and
/// the resolved-request block and the diagnostic line both show the parsed
/// value.
#[test]
fn kv_bits_flag_threads_through_to_a_real_forward_pass() {
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
        "cli-kv-bits",
    )
    .expect("real-naming install writes");

    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-check"))
        .args([
            "--model",
            dir.to_str().unwrap(),
            "--prompt",
            "hi",
            "--max-new",
            "3",
            "--kv-bits",
            "3.5",
        ])
        .output()
        .expect("binary should run");

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("generating (real forward pass"));
    assert!(
        stdout.contains("kv_bits: TurboQuant"),
        "resolved request must show the parsed value: {stdout}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("kv-bits: 3.5 (K3/V4)"),
        "unexpected stderr: {stderr}"
    );
    assert!(stderr.contains("[stop="), "unexpected stderr: {stderr}");
}

#[test]
fn messages_file_mode_generates_through_the_chat_template() {
    let dir = install_with_tokenizer("cli-messages-file");
    let messages = dir.join("messages.json");
    std::fs::write(
        &messages,
        r#"[{"role":"system","content":"be nice"},{"role":"user","content":"hi"}]"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-check"))
        .args([
            "--model",
            dir.to_str().unwrap(),
            "--messages-file",
            messages.to_str().unwrap(),
            "--max-new",
            "3",
        ])
        .output()
        .expect("binary should run");

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("generating (real forward pass, chat template applied)"),
        "unexpected stdout: {stdout}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("[stop="), "unexpected stderr: {stderr}");
}

#[test]
fn messages_file_mode_rejects_an_unknown_role() {
    let dir = install_with_tokenizer("cli-messages-role");
    let messages = dir.join("messages.json");
    std::fs::write(&messages, r#"[{"role":"wizard","content":"hi"}]"#).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_turbospark-check"))
        .args([
            "--model",
            dir.to_str().unwrap(),
            "--messages-file",
            messages.to_str().unwrap(),
        ])
        .output()
        .expect("binary should run");

    // Fire-and-forget discipline: a bad conversation is a note, not a crash
    // and not a different exit status.
    assert_eq!(output.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unsupported role \"wizard\""),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn chat_mode_runs_a_turn_and_quits() {
    let dir = install_with_tokenizer("cli-chat");

    let mut child = Command::new(env!("CARGO_BIN_EXE_turbospark-check"))
        .args([
            "--model",
            dir.to_str().unwrap(),
            "--chat",
            "--system",
            "be nice",
            "--max-new",
            "3",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("binary should run");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"hi\n/history\n/quit\n")
        .unwrap();
    let output = child.wait_with_output().expect("binary should exit");

    assert_eq!(output.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Interactive chat. Commands: /clear, /history, /quit."),
        "unexpected stderr: {stderr}"
    );
    assert!(stderr.contains("[stop="), "unexpected stderr: {stderr}");
    // /history ran after the turn, so it shows the system opening and the
    // committed user turn. The assistant turn is deliberately not asserted:
    // these weights are untrained, so a three-token reply can decode to no
    // visible text at all, and an empty reply is not appended to the history.
    assert!(stderr.contains("[system] be nice"), "stderr: {stderr}");
    assert!(stderr.contains("[user] hi"), "stderr: {stderr}");
}
