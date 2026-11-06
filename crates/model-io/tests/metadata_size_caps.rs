//! Small metadata sidecars in a model dir are attacker-controlled (users
//! download checkpoints), so each reader must refuse an oversized file
//! instead of buffering it whole. Each test pads otherwise-acceptable JSON
//! with whitespace past the reader's cap, so only the size check can refuse
//! it.

use std::path::PathBuf;

use turbospark_model_io::speech_receipt::{SpeechInstallReceipt, SPEECH_RECEIPT_FILENAME};
use turbospark_model_io::trunk_role::{self, TRUNK_ROLE_FILE};
use turbospark_model_io::{DashConfig, ModelError, DASH_SIDECAR};

fn tempdir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("turbospark-meta-caps-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `body` followed by `pad` bytes of JSON whitespace, still valid JSON.
fn padded(body: &str, pad: usize) -> Vec<u8> {
    let mut bytes = body.as_bytes().to_vec();
    bytes.resize(bytes.len() + pad, b' ');
    bytes
}

#[test]
fn speech_receipt_refuses_oversized_file() {
    let dir = tempdir("speech");
    let json = SpeechInstallReceipt::new("whisper", 80, 1, None, None, Default::default())
        .to_json_pretty()
        .unwrap();
    // A well-formed receipt at normal size still loads.
    std::fs::write(dir.join(SPEECH_RECEIPT_FILENAME), &json).unwrap();
    SpeechInstallReceipt::read_from_dir(&dir).expect("normal receipt loads");

    std::fs::write(
        dir.join(SPEECH_RECEIPT_FILENAME),
        padded(&json, 5 * 1024 * 1024),
    )
    .unwrap();
    let err = SpeechInstallReceipt::read_from_dir(&dir).expect_err("oversized receipt");
    let ModelError::TrustedReceiptInvalid { detail } = err else {
        panic!("expected TrustedReceiptInvalid, got {err:?}");
    };
    assert!(detail.contains("exceeds metadata cap"), "{detail}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn trunk_role_refuses_oversized_marker() {
    let dir = tempdir("trunk");
    let json = r#"{"kind":"headless_trunk","owner":"fish_drama3"}"#;
    std::fs::write(dir.join(TRUNK_ROLE_FILE), json).unwrap();
    trunk_role::read(&dir).expect("normal marker loads");

    std::fs::write(dir.join(TRUNK_ROLE_FILE), padded(json, 128 * 1024)).unwrap();
    let err = trunk_role::read(&dir).expect_err("oversized marker");
    let ModelError::IndexCorrupt { detail } = err else {
        panic!("expected IndexCorrupt, got {err:?}");
    };
    assert!(detail.contains("exceeds metadata cap"), "{detail}");
    // Presence alone still refuses a chat open, whatever the size.
    assert!(trunk_role::has_marker(&dir));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dash_sidecar_refuses_oversized_file() {
    let dir = tempdir("dash");
    std::fs::write(
        dir.join(DASH_SIDECAR),
        padded(r#"{"sparse_attention_config":{}}"#, 5 * 1024 * 1024),
    )
    .unwrap();
    let err = DashConfig::load(&dir, 2, 128).expect_err("oversized sidecar");
    assert!(err.contains("exceeds metadata cap"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}
