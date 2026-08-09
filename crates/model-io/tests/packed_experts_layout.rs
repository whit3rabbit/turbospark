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

/// A layer without its own `expertStride` inherits the top-level one, which
/// is every install written before ROADMAP Phase S. The fixture above is
/// exactly that shape, so this is asserting the fallback rather than a new
/// field: without it, adding the field would have silently zeroed the stride
/// on every existing artifact, which reads as an empty expert blob and not as
/// a parse error.
#[test]
fn a_layer_without_its_own_stride_inherits_the_top_level_one() {
    let dir = tempdir();
    write_layout(&dir, valid_layout_json());
    let layout = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap();
    assert_eq!(layout.layers[0].expert_stride, layout.expert_stride);
}

/// Per-layer strides are read where present, and are capped by the top-level
/// value, which is what every consumer sizes a slot from.
#[test]
fn per_layer_strides_are_read_and_bounded_by_the_top_level() {
    let dir = tempdir();
    let json = |layer_stride: u64| {
        format!(
            r#"{{
            "expertStride": 4096,
            "numLayers": 1,
            "expertsPerLayer": 1,
            "layers": [
                {{
                    "layer": 0,
                    "file": "layer_00.bin",
                    "expertStride": {layer_stride},
                    "experts": [
                        {{"expert": 0, "offset": 0, "size": {layer_stride}, "tensors": {{}}}}
                    ]
                }}
            ]
        }}"#
        )
    };

    write_layout(&dir, &json(2048));
    let layout = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap();
    assert_eq!(layout.expert_stride, 4096);
    assert_eq!(layout.layers[0].expert_stride, 2048);

    // Above the ceiling is refused rather than silently trusted: a slot is
    // allocated from the top-level number, so a larger layer stride would be
    // a buffer overrun waiting at the first cache miss.
    write_layout(&dir, &json(8192));
    let err = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap_err();
    assert!(
        matches!(err, ModelError::IndexCorrupt { ref detail } if detail.contains("8192")),
        "{err:?}"
    );
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
