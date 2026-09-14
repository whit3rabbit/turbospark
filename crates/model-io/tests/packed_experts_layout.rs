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
                        "tensors": {
                            "gate": {"offset": 0, "size": 2048, "dtype": "int4", "shape": [64, 64]}
                        }
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
    assert_eq!(e1.sub_tensors["gate"].size, 2048);
}

fn one_tensor_layout(tensor: &str) -> String {
    format!(
        r#"{{
        "expertStride": 64,
        "numLayers": 1,
        "expertsPerLayer": 1,
        "layers": [{{
            "layer": 0,
            "file": "layer_00.bin",
            "experts": [{{"expert": 0, "offset": 0, "size": 64,
                "tensors": {{"gate_biases": {tensor}}}}}]
        }}]
    }}"#
    )
}

#[test]
fn sub_tensor_ranges_must_fit_the_expert_blob() {
    let dir = tempdir();
    write_layout(
        &dir,
        &one_tensor_layout(r#"{"offset": 60, "size": 8, "dtype": "f32", "shape": [2]}"#),
    );
    let err = load_packed_experts_layout(&dir, 1024).unwrap_err();
    assert!(
        err.to_string().contains("exceeds expert blob size"),
        "{err}"
    );

    let oversized = one_tensor_layout(r#"{"offset": 0, "size": 8, "dtype": "f32", "shape": [2]}"#)
        .replacen(r#""size": 64,"#, r#""size": 68,"#, 1);
    assert!(oversized.contains(r#""size": 68,"#));
    write_layout(&dir, &oversized);
    let err = load_packed_experts_layout(&dir, 1024).unwrap_err();
    assert!(err.to_string().contains("exceeds its stride"), "{err}");
}

#[test]
fn bias_layout_requires_aligned_f32_bytes_matching_its_shape() {
    for (tensor, expected) in [
        (
            r#"{"offset": 2, "size": 8, "dtype": "f32", "shape": [2]}"#,
            "4-byte aligned",
        ),
        (
            r#"{"offset": 4, "size": 4, "dtype": "f32", "shape": [2]}"#,
            "with 8 bytes",
        ),
        (
            r#"{"offset": 4, "size": 8, "dtype": "bf16", "shape": [2]}"#,
            "must be F32",
        ),
        (
            r#"{"offset": 4, "size": 8, "dtype": "f32", "shape": [1, 2]}"#,
            "one dimension",
        ),
    ] {
        let dir = tempdir();
        write_layout(&dir, &one_tensor_layout(tensor));
        let err = load_packed_experts_layout(&dir, 1024).unwrap_err();
        assert!(err.to_string().contains(expected), "{err}");
    }
}

#[test]
fn experts_in_one_layer_must_use_the_same_sub_tensor_layout() {
    let dir = tempdir();
    let json = valid_layout_json().replacen(
        r#""offset": 4096,
                        "size": 4096,
                        "tensors": {
                            "gate": {"offset": 0"#,
        r#""offset": 4096,
                        "size": 4096,
                        "tensors": {
                            "gate": {"offset": 4"#,
        1,
    );
    assert_ne!(json, valid_layout_json(), "fixture mutation must apply");
    write_layout(&dir, &json);
    let err = load_packed_experts_layout(&dir, 1024 * 1024).unwrap_err();
    assert!(err.to_string().contains("layout differs"), "{err}");
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

/// Two layers declaring the wrong ORDER (1 before 0) resolve the wrong blob
/// through `PackedExpertsLayout::expert()`'s positional indexing with no
/// error, unless the loader refuses it -- which it now does.
#[test]
fn load_rejects_layers_out_of_order() {
    let dir = tempdir();
    let json = r#"{
        "expertStride": 4096,
        "numLayers": 2,
        "expertsPerLayer": 1,
        "layers": [
            {"layer": 1, "file": "layer_01.bin", "experts": [
                {"expert": 0, "offset": 0, "size": 4096, "tensors": {}}
            ]},
            {"layer": 0, "file": "layer_00.bin", "experts": [
                {"expert": 0, "offset": 0, "size": 4096, "tensors": {}}
            ]}
        ]
    }"#;
    write_layout(&dir, json);
    let err = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }));
}

/// A gap (0, then 2, skipping 1) is the same positional-indexing hazard as
/// out-of-order layers and is refused the same way.
#[test]
fn load_rejects_a_gap_in_the_layer_sequence() {
    let dir = tempdir();
    let json = r#"{
        "expertStride": 4096,
        "numLayers": 2,
        "expertsPerLayer": 1,
        "layers": [
            {"layer": 0, "file": "layer_00.bin", "experts": [
                {"expert": 0, "offset": 0, "size": 4096, "tensors": {}}
            ]},
            {"layer": 2, "file": "layer_02.bin", "experts": [
                {"expert": 0, "offset": 0, "size": 4096, "tensors": {}}
            ]}
        ]
    }"#;
    write_layout(&dir, json);
    let err = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::IndexCorrupt { .. }));
}

/// `numLayers` disagreeing with the array's own length (here: declaring 2
/// while shipping 1) is refused rather than silently trusting either side.
#[test]
fn load_rejects_num_layers_disagreeing_with_the_array() {
    let dir = tempdir();
    let json = r#"{
        "expertStride": 4096,
        "numLayers": 2,
        "expertsPerLayer": 1,
        "layers": [
            {"layer": 0, "file": "layer_00.bin", "experts": [
                {"expert": 0, "offset": 0, "size": 4096, "tensors": {}}
            ]}
        ]
    }"#;
    write_layout(&dir, json);
    let err = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap_err();
    let ModelError::IndexCorrupt { detail } = err else {
        panic!("expected IndexCorrupt, got {err:?}");
    };
    assert!(detail.contains("numLayers"), "{detail}");
}

/// A duplicate `expert` id would otherwise silently overwrite the first
/// blob and leave the real second slot `None`, reported downstream as an
/// unrelated "missing expert entries" with no hint of the actual cause.
#[test]
fn load_rejects_a_duplicate_expert_id() {
    let dir = tempdir();
    let json = r#"{
        "expertStride": 4096,
        "numLayers": 1,
        "expertsPerLayer": 2,
        "layers": [
            {"layer": 0, "file": "layer_00.bin", "experts": [
                {"expert": 0, "offset": 0, "size": 4096, "tensors": {}},
                {"expert": 0, "offset": 4096, "size": 4096, "tensors": {}}
            ]}
        ]
    }"#;
    write_layout(&dir, json);
    let err = load_packed_experts_layout(&dir, 64 * 1024 * 1024).unwrap_err();
    let ModelError::IndexCorrupt { detail } = err else {
        panic!("expected IndexCorrupt, got {err:?}");
    };
    assert!(detail.contains("more than once"), "{detail}");
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
