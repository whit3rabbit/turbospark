//! The DeepSeek-OCR tower on a REAL install, held to the numpy transcription
//! `scripts/deepseek_ocr_tower_reference.py` stage by stage
//! (`docs/DEEPSEEK_OCR_VISION_PHASE0.md`, "The tower gate").
//!
//! The reference writes its input (`input.f32`, the normalized 1024 x 1024 x 3
//! HWC image) and one raw float32 file per stage into a directory. This test
//! feeds the SAME input to the engine's tower, reads the engine's stages back,
//! and reports and asserts the cosine similarity and the worst relative error
//! per stage. The engine is FP16 with FP32 accumulation, the reference float32
//! on the CPU from the same BF16 bytes, so the stages are close, not equal.
//!
//! ```sh
//! python3 -I scripts/deepseek_ocr_tower_reference.py \
//!   /Volumes/T7/hf-models/DeepSeek-OCR-4bit/model.safetensors /Volumes/T7/dsocr-ref
//! TURBOSPARK_DSOCR_INSTALL_DIR=/Volumes/T7/tshome-dsocr/models/text/dsocr.gturbo \
//! TURBOSPARK_DSOCR_SIDECAR_DIR=/Volumes/T7/tshome-dsocr/models/text/dsocr-vision.gturbo-vision \
//! TURBOSPARK_DSOCR_REF_DIR=/Volumes/T7/dsocr-ref \
//!   cargo test -p turbospark-runtime --test deepseek_ocr_tower_real --release -- \
//!   --ignored --nocapture
//! ```

use std::path::PathBuf;

use turbospark_runtime::RealForwardRunner;
use turbospark_vision_io::DeepseekOcrView;

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key} (see this file's doc)")))
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

/// `(cosine, worst |a-b| / max|b|, finite)` of an engine stage against a
/// reference stage.
fn compare(engine: &[f32], reference: &[f32]) -> (f64, f64, bool) {
    assert_eq!(engine.len(), reference.len(), "stage length");
    let (mut dot, mut ne, mut nr) = (0.0f64, 0.0f64, 0.0f64);
    let mut worst = 0.0f64;
    let mut peak = 0.0f64;
    let mut finite = true;
    for (&a, &b) in engine.iter().zip(reference) {
        finite &= a.is_finite();
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        ne += a * a;
        nr += b * b;
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    (
        dot / (ne.sqrt() * nr.sqrt()).max(1e-30),
        worst / peak.max(1e-30),
        finite,
    )
}

/// The stages and the cosine each must reach. Set just under the values
/// measured on the real checkpoint with the synthetic input (all at or above
/// 0.99998 except `clip_out` 0.999987, `proj` and `rows` 0.999992), not from a
/// guess. Two mutations were applied and both failed this test: dropping the
/// relative-position bias takes `sam_b0` to 0.9986 and `rows` to 0.634, and
/// padding the SAM windows with token 0 instead of zeros takes `rows` to
/// 0.889 (`docs/DEEPSEEK_OCR_VISION_PHASE0.md`).
const STAGES: [(&str, f64); 14] = [
    ("sam_patch", 0.9999),
    ("sam_b0", 0.9999),
    ("sam_b2", 0.9999),
    ("sam_b5", 0.9999),
    ("sam_b11", 0.9999),
    ("sam_neck", 0.9999),
    ("sam_n2", 0.9999),
    ("sam_out", 0.9999),
    ("clip_pre", 0.9999),
    ("clip_l0", 0.9999),
    ("clip_l11", 0.9999),
    ("clip_out", 0.9995),
    ("proj", 0.999),
    ("rows", 0.999),
];

#[test]
#[ignore = "needs the real dsocr install, its vision sidecar, the reference stages and Metal"]
fn the_real_tower_matches_the_numpy_reference_stage_by_stage() {
    let install = env_path("TURBOSPARK_DSOCR_INSTALL_DIR");
    let sidecar = env_path("TURBOSPARK_DSOCR_SIDECAR_DIR");
    let refs = env_path("TURBOSPARK_DSOCR_REF_DIR");
    let pixels = read_f32(&refs.join("input.f32"));
    assert_eq!(
        pixels.len(),
        1024 * 1024 * 3,
        "input.f32 is the 1024 x 1024 x 3 view"
    );
    let view = DeepseekOcrView { pixels };

    let arch = turbospark_repack::peek_manifest_arch(&install).expect("install manifest");
    let mut runner = RealForwardRunner::open(&install, arch).expect("open install");
    runner.attach_vision_sidecar(&sidecar).expect("attach");
    let (rows, stages) = runner
        .encode_image_deepseek_ocr_with_stages(&view)
        .expect("tower runs");

    let width = runner.vision_config().out_hidden_size as usize;
    assert_eq!(rows.len(), 273 * width, "273 prompt rows");

    println!("{:<10} {:>10} {:>12}", "stage", "cosine", "worst/peak");
    let mut failures = Vec::new();
    for (name, floor) in STAGES {
        let engine = f16_bits_to_f32(
            stages
                .get(name)
                .unwrap_or_else(|| panic!("no stage {name}")),
        );
        let reference = read_f32(&refs.join(format!("{name}.f32")));
        let (cos, rel, finite) = compare(&engine, &reference);
        println!("{name:<10} {cos:>10.6} {rel:>12.5}");
        if !finite || cos < floor {
            failures.push(format!(
                "{name}: cosine {cos:.6} (floor {floor}), finite {finite}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "stages below their floor: {failures:#?}"
    );

    // The layout rows are copies, so they match the checkpoint exactly: row 16
    // of the run is `image_newline`, the last is `view_separator`.
    let row = |i: usize| &rows[i * width..(i + 1) * width];
    assert_eq!(
        row(16),
        row(33),
        "every newline row is the same learned row"
    );
    assert_ne!(row(16), row(272), "the separator is a different row");
}
