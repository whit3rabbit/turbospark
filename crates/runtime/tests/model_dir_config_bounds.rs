//! Model-dir config reads are capped before parsing. A downloaded checkpoint
//! controls these files, so an oversized one must be refused instead of being
//! read whole into memory.
#![cfg(target_os = "macos")]

mod common;

use common::TempDir;
use std::path::Path;

/// Over the 1 MiB config cap, but still valid JSON, so an uncapped read
/// parses it and only the cap can refuse it.
fn write_padded(path: &Path, body: &str) {
    let inner = body.strip_prefix('{').expect("object body");
    let mut text = String::from("{");
    text.push_str(&" ".repeat((1 << 20) + 16));
    text.push_str(inner);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn assert_capped(result: Result<(), String>, site: &str) {
    let err = result.expect_err(&format!("{site}: oversized config was accepted"));
    assert!(err.contains("byte limit"), "{site}: wrong refusal: {err}");
}

#[test]
fn drama3_trunk_config_is_capped() {
    let dir = TempDir::new("cfg-bounds");
    write_padded(
        &dir.join("config.json"),
        r#"{"text_config":{"vocab_size":1000}}"#,
    );
    assert_capped(
        turbospark_runtime::fish_drama3::expected_trunk_arch(&dir).map(|_| ()),
        "fish_drama3",
    );
}

#[test]
fn encoder_config_and_pooling_are_capped() {
    let dir = TempDir::new("cfg-bounds");
    write_padded(&dir.join("config.json"), r#"{"model_type":"bert"}"#);
    assert_capped(
        turbospark_runtime::encoder::check_encoder_dir(&dir).map(|_| ()),
        "encoder config",
    );

    let dir = TempDir::new("cfg-bounds");
    write_padded(
        &dir.join("1_Pooling").join("config.json"),
        r#"{"pooling_mode_cls_token":true}"#,
    );
    assert_capped(
        turbospark_runtime::encoder::check_pooling(&dir),
        "encoder pooling",
    );
}

#[test]
fn encoder_pooling_still_optional() {
    // The cap must keep the NotFound pass-through: arctic-embed ships no
    // 1_Pooling directory.
    let dir = TempDir::new("cfg-bounds");
    turbospark_runtime::encoder::check_pooling(&dir).unwrap();
}

#[test]
fn moonshine_config_is_capped() {
    let dir = TempDir::new("cfg-bounds");
    write_padded(&dir.join("config.json"), r#"{"hidden_size":288}"#);
    assert_capped(
        turbospark_runtime::moonshine::MoonshineMetalEngine::open(&dir).map(|_| ()),
        "moonshine",
    );
}

#[test]
fn qwen3_asr_config_is_capped() {
    let dir = TempDir::new("cfg-bounds");
    write_padded(&dir.join("config.json"), r#"{"model_type":"qwen3_asr"}"#);
    assert_capped(
        turbospark_runtime::qwen3_asr_metal::Qwen3AsrMetalEngine::open(&dir).map(|_| ()),
        "qwen3_asr_metal",
    );
}

#[test]
fn unlimited_ocr_coreml_config_is_capped() {
    use turbospark_runtime::coreml::unlimited_ocr::{OcrOptions, UnlimitedOcrCoreMl};
    if !turbospark_coreml::supports(turbospark_coreml::Capability::State) {
        // Open refuses before reading on macOS 14; nothing to check there.
        return;
    }
    let dir = TempDir::new("cfg-bounds");
    write_padded(&dir.join("config.json"), r#"{"vocab_size":1}"#);
    assert_capped(
        UnlimitedOcrCoreMl::open(&dir, &OcrOptions::default()).map(|_| ()),
        "unlimited_ocr coreml",
    );
}
