//! The LLaVA-OneVision tower and its row packing against a numpy
//! transcription of the reference, on the real checkpoint
//! (`docs/LLAVA_ONEVISION_VISION_PHASE0.md`).
//!
//! The reference is `scripts/llava_onevision_tower_reference.py`, written from
//! mlx-vlm's and Hugging Face's sources (read, never executed). It is a
//! transcription, not the library run on the same bytes, so a convention both
//! it and the port get wrong would pass; what it does check is that the port's
//! tower, projector, anyres unpad, bilinear pooling and newline placement
//! agree with an independent implementation of the same text.
//!
//! The test has two phases, keyed on the dump directory: it first writes the
//! port's own patch rows (`pixels.f32`, `meta.json`) for the numpy script to
//! consume, then compares the port's stages and packed rows against the
//! script's `ref_*.f32` once they exist.
//!
//! # Setup
//!
//! ```sh
//! TURBOSPARK_HOME=<store> turbospark-model pull --repo \
//!   paazmaya/llava-onevision-qwen2-7b-ov-hf-gguf \
//!   --file llava-onevision-qwen2-7b-ov-hf_Q4_K_M.gguf \
//!   --sidecar-repo llava-hf/llava-onevision-qwen2-7b-ov-hf --alias llavaov-7b
//! turbospark-model pull-vision --repo \
//!   llava-hf/llava-onevision-qwen2-7b-ov-hf@0d50680527681998e456c7b78950205bedd8a068 \
//!   --alias llavaov-vision-tower --out <sidecar dir>
//!
//! export TURBOSPARK_OV_TRUNK_INSTALL_DIR=<llavaov-7b.gturbo>
//! export TURBOSPARK_OV_SIDECAR_DIR=<sidecar dir>
//! export TURBOSPARK_OV_DUMP_DIR=<dump>  TURBOSPARK_OV_IMAGE=<image>
//! cargo test -p turbospark-runtime --test onevision_vision_parity --release -- \
//!   --ignored --nocapture          # writes the dump, then fails for want of refs
//! python3 -I scripts/llava_onevision_tower_reference.py \
//!   --cache-dir <cache> --dump <dump>
//! cargo test ... (again)           # compares
//! ```
//!
//! Metal and the real installs are required, so the test is `#[ignore]`d.

use std::path::{Path, PathBuf};

use turbospark_runtime::RealForwardRunner;
use turbospark_vision_io::{clip_preprocess, decode_image_file, ClipPreprocessParams};

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().map(PathBuf::from)
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f16s(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

/// Cosine over everything, and the worst single row's cosine.
fn cosines(got: &[f32], want: &[f32], width: usize) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "length");
    let dot =
        |a: &[f32], b: &[f32]| -> f64 { a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum() };
    let whole = dot(got, want) / (dot(got, got).sqrt() * dot(want, want).sqrt());
    let worst = got
        .chunks_exact(width)
        .zip(want.chunks_exact(width))
        .map(|(g, w)| dot(g, w) / (dot(g, g).sqrt() * dot(w, w).sqrt()))
        .fold(1.0f64, f64::min);
    (whole, worst)
}

#[test]
#[ignore = "needs the real llava-onevision installs, Metal, and the numpy reference's dump"]
fn the_onevision_tower_and_packing_agree_with_the_numpy_reference() {
    let (Some(trunk), Some(sidecar), Some(dump), Ok(image_path)) = (
        env_dir("TURBOSPARK_OV_TRUNK_INSTALL_DIR"),
        env_dir("TURBOSPARK_OV_SIDECAR_DIR"),
        env_dir("TURBOSPARK_OV_DUMP_DIR"),
        std::env::var("TURBOSPARK_OV_IMAGE"),
    ) else {
        eprintln!(
            "SKIP: set TURBOSPARK_OV_{{TRUNK_INSTALL,SIDECAR,DUMP}}_DIR and TURBOSPARK_OV_IMAGE"
        );
        return;
    };
    let arch = turbospark_repack::peek_manifest_arch(&trunk).expect("trunk manifest");
    assert!(
        !arch.vision.is_active(),
        "the trunk must be TEXT-ONLY: the sidecar is what is under test"
    );
    let mut runner = RealForwardRunner::open(&trunk, arch).expect("open trunk");
    runner
        .attach_vision_sidecar(&sidecar)
        .expect("the sidecar pairs with this trunk's family and hidden size");
    assert!(
        runner.clip_image_ids().is_some(),
        "a flat-injection tower is attached"
    );

    let json = std::fs::read_to_string(runner.vision_dir().join("preprocessor_config.json"))
        .expect("preprocessor_config.json beside the sidecar");
    let params = ClipPreprocessParams::from_preprocessor_config_json(&json).expect("params");
    let decoded = decode_image_file(Path::new(&image_path)).expect("decode");
    let patches = clip_preprocess(&decoded, &params).expect("preprocess");
    let anyres = patches
        .anyres
        .expect("a OneVision image carries its tiling");

    // Phase 1: the numpy script's inputs.
    std::fs::create_dir_all(&dump).expect("dump dir");
    let bytes: Vec<u8> = patches
        .pixels
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(dump.join("pixels.f32"), bytes).expect("pixels.f32");
    std::fs::write(
        dump.join("meta.json"),
        serde_json::json!({
            "image": image_path,
            "orig_h": anyres.orig_h,
            "orig_w": anyres.orig_w,
            "tiles_h": anyres.tiles_h,
            "tiles_w": anyres.tiles_w,
            "tokens": patches.grid * patches.grid_w,
        })
        .to_string(),
    )
    .expect("meta.json");
    assert!(
        dump.join("ref_packed.f32").exists(),
        "dump written; run scripts/llava_onevision_tower_reference.py --dump {} and re-run",
        dump.display()
    );

    // Phase 2: the stages and the packed run.
    let (rows, stages) = runner
        .encode_image_clip_with_stages(&patches)
        .expect("encode");
    let mut results = Vec::new();
    for (label, got, name, width, floor) in [
        ("embed", f16s(&stages.embed), "ref_embed.f32", 1152, 0.9999),
        (
            "block 0",
            f16s(&stages.block_first),
            "ref_block_0.f32",
            1152,
            0.9999,
        ),
        (
            "block 25 (hs[-1])",
            f16s(&stages.block_last),
            "ref_block_last.f32",
            1152,
            0.998,
        ),
        ("packed run", f16s(&rows), "ref_packed.f32", 3584, 0.998),
    ] {
        let want = read_f32(&dump.join(name));
        let (whole, worst_row) = cosines(&got, &want, width);
        eprintln!("{label:>18}: cosine {whole:.8}  worst row {worst_row:.6}");
        results.push((label, whole, floor));
    }
    for (label, whole, floor) in results {
        assert!(
            whole > floor,
            "{label}: cosine {whole} is under {floor}; the port and the reference disagree"
        );
    }
}
