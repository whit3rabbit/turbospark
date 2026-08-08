//! Tests for `packed_experts/layout.json` decode.

use std::io::Write;

use turbospark_model_io::{load_packed_experts_layout, ModelError};

fn write_layout(dir: &std::path::Path, json: &str) {
    let sub = dir.join("packed_experts");
    std::fs::create_dir_all(&sub).unwrap();
    let mut f = std::fs::File::create(sub.join("layout.json")).unwrap();
    f.write_all(json.as_bytes()).unwrap();
}

fn valid_layout_json() -> &'static str {
    r#"{
        "expertStride": 4096,
        "numLayers": 1,
        "expertsPerLayer": 2,
        "layers": [
            {
                "layer": 0,
                "file": "layer_00.bin",
                "experts": [
                    {
                        "expert": 0,
                        "offset": 0,
                        "size": 4096,
                        "tensors": {
                            "gate": {"offset": 0, "size": 2048, "dtype": "int4", "shape": [64, 64]}
                        }
                    },
                    {
                        "expert": 1,
                        "offset": 4096,
                        "size": 4096,
                        "tensors": {}
                    }
                ]
            }
        ]
    }"#
}

#[test]
fn load_succeeds_and_resolves_experts_by_index() {
    let dir = tempdir();
    write_layout(&dir, valid_layout_json());
    let layout = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap();
    assert_eq!(layout.expert_stride, 4096);
    let e0 = layout.expert(0, 0);
    assert_eq!(e0.offset, 0);
    assert_eq!(e0.sub_tensors["gate"].size, 2048);
    let e1 = layout.expert(0, 1);
    assert_eq!(e1.offset, 4096);
}

#[test]
fn load_rejects_missing_expert_entries() {
    let dir = tempdir();
    // expertsPerLayer says 2 but only expert 0 is present.
    let json = r#"{
        "expertStride": 4096,
        "numLayers": 1,
        "expertsPerLayer": 2,
        "layers": [
            {
                "layer": 0,
                "file": "layer_00.bin",
                "experts": [
                    {"expert": 0, "offset": 0, "size": 4096, "tensors": {}}
                ]
            }
        ]
    }"#;
    write_layout(&dir, json);
    let err = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }));
}

#[test]
fn load_rejects_missing_file() {
    let dir = tempdir();
    let err = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::MissingFile { .. }));
}

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique_counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "turbospark-model-io-layout-{}-{}-{unique_counter}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}
