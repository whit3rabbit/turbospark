//! Tests for `ResidentBuffer::map`'s file-length gate: a header-declared
//! `(file_offset, resident_size)` past the end of a truncated
//! `model_weights.bin` must fail with a named error rather than mapping
//! successfully and SIGBUS-ing on first touch.

use std::io::Write;

use turbospark_model_io::{ModelError, ResidentBuffer};

fn write_file(bytes: &[u8]) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-resident-buffer-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model_weights.bin");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
    path
}

#[test]
fn map_succeeds_when_the_region_fits_the_file() {
    let path = write_file(&[0u8; 8192]);
    let buffer = ResidentBuffer::map(&path, 0, 4096).unwrap();
    assert_eq!(buffer.data().len(), 4096);
}

#[test]
fn map_refuses_a_region_one_byte_past_a_truncated_file() {
    let path = write_file(&[0u8; 4096]);
    // The file is exactly 4096 bytes; ask for one byte more than it holds.
    // `ResidentBuffer` has no `Debug` impl, so `unwrap_err` cannot be used
    // here (AGENTS.md's `expect_err` gotcha, one method over).
    let Err(err) = ResidentBuffer::map(&path, 0, 4097) else {
        panic!("expected a TensorSizeMismatch refusal");
    };
    assert!(
        matches!(err, ModelError::TensorSizeMismatch { .. }),
        "{err:?}"
    );
}

#[test]
fn map_refuses_a_nonzero_offset_past_the_file() {
    let path = write_file(&[0u8; 4096]);
    // Offset 4096 is exactly EOF; there is no byte left to map.
    let Err(err) = ResidentBuffer::map(&path, 4096, 1) else {
        panic!("expected a TensorSizeMismatch refusal");
    };
    assert!(
        matches!(err, ModelError::TensorSizeMismatch { .. }),
        "{err:?}"
    );
}
