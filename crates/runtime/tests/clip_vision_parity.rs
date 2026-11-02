//! The CLIP tower (LLaVA-1.5) against `transformers`, on the real checkpoint.
//!
//! Stage 2 of the four-gate order `docs/VISION.md` describes, for the tower
//! `docs/LLAVA_VISION_PHASE0.md` records. Two claims:
//!
//! 1. **Preprocessing is the processor's.** `turbospark_vision_io::
//!    clip_preprocess` produces the same pixels as `CLIPImageProcessor` (PIL
//!    bicubic resize, center crop, normalize), compared element for element
//!    after un-patchifying. A resize that differs by one rounding step moves
//!    a pixel by `1/255/std`, about 0.015, so the bound below separates "same
//!    pixels" from "different pixels" by an order of magnitude.
//! 2. **The tower is the reference's function.** The residual stream at
//!    `hidden_states[0]` (after `ln_pre`), `hidden_states[1]` (after block 0)
//!    and `hidden_states[-2]` (after block 22), and the projector's output,
//!    each as a cosine against the reference run in float32 on the CPU.
//!
//! # Setup
//!
//! ```sh
//! turbospark-model pull-vision --repo \
//!   llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
//!   --alias llava15-vision-tower
//! turbospark-model pull --repo second-state/Llava-v1.5-7B-GGUF \
//!   --file llava-v1.5-7b-Q4_K_M.gguf \
//!   --sidecar-repo llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
//!   --alias llava15-7b
//!
//! uv run --python 3.12 --with torch --with numpy --with pillow \
//!   --with transformers --with safetensors -- python scripts/llava_vision_probe.py \
//!   --checkpoint <llava-hf checkout> --image <image> --out-dir <dump>
//!
//! TURBOSPARK_LLAVA_TRUNK_INSTALL_DIR=<llava15-7b.gturbo> \
//! TURBOSPARK_LLAVA_SIDECAR_DIR=<llava15-vision-tower.gturbo-vision> \
//! TURBOSPARK_LLAVA_DUMP_DIR=<dump> \
//!   cargo test -p turbospark-runtime --test clip_vision_parity --release -- \
//!   --ignored --nocapture
//! ```
//!
//! Metal and the real installs are required, so the test is `#[ignore]`d.

use std::path::{Path, PathBuf};

use turbospark_runtime::RealForwardRunner;
use turbospark_vision_io::{
    clip_preprocess, decode_image_file, ClipPreprocessParams, CLIP_PATCH, CLIP_PATCH_DIM,
};

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
    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
    };
    let whole = dot(got, want) / (dot(got, got).sqrt() * dot(want, want).sqrt());
    let worst = got
        .chunks_exact(width)
        .zip(want.chunks_exact(width))
        .map(|(g, w)| dot(g, w) / (dot(g, g).sqrt() * dot(w, w).sqrt()))
        .fold(1.0f64, f64::min);
    (whole, worst)
}

#[test]
#[ignore = "needs the real llava installs, Metal, and scripts/llava_vision_probe.py's dump"]
fn the_clip_tower_agrees_with_transformers_at_every_stage() {
    let (Some(trunk), Some(sidecar), Some(dump)) = (
        env_dir("TURBOSPARK_LLAVA_TRUNK_INSTALL_DIR"),
        env_dir("TURBOSPARK_LLAVA_SIDECAR_DIR"),
        env_dir("TURBOSPARK_LLAVA_DUMP_DIR"),
    ) else {
        eprintln!("SKIP: set TURBOSPARK_LLAVA_{{TRUNK_INSTALL,SIDECAR,DUMP}}_DIR");
        return;
    };
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dump.join("meta.json")).expect("meta.json"))
            .expect("meta json");
    let image_path = meta["image"].as_str().expect("meta.image");

    let arch = turbospark_repack::peek_manifest_arch(&trunk).expect("trunk manifest");
    assert!(
        !arch.vision.is_active(),
        "the trunk must be TEXT-ONLY: the sidecar is what is under test"
    );
    let mut runner = RealForwardRunner::open(&trunk, arch).expect("open trunk");
    runner
        .attach_vision_sidecar(&sidecar)
        .expect("the sidecar pairs with this trunk's family and hidden size");
    assert!(runner.clip_image_ids().is_some(), "a CLIP tower is attached");

    // Claim 1: the pixels.
    let json = std::fs::read_to_string(runner.vision_dir().join("preprocessor_config.json"))
        .expect("preprocessor_config.json beside the sidecar");
    let params = ClipPreprocessParams::from_preprocessor_config_json(&json).expect("params");
    let decoded = decode_image_file(Path::new(image_path)).expect("decode");
    let patches = clip_preprocess(&decoded, &params).expect("preprocess");
    let reference = read_f32(&dump.join("pixels.f32")); // [3, 336, 336]
    let side = params.size;
    let grid = patches.grid;
    let mut worst_pixel = 0f32;
    for py in 0..grid {
        for px in 0..grid {
            let row = &patches.pixels[(py * grid + px) * CLIP_PATCH_DIM..][..CLIP_PATCH_DIM];
            for c in 0..3 {
                for r in 0..CLIP_PATCH {
                    for k in 0..CLIP_PATCH {
                        let got = row[c * CLIP_PATCH * CLIP_PATCH + r * CLIP_PATCH + k];
                        let want = reference
                            [c * side * side + (py * CLIP_PATCH + r) * side + px * CLIP_PATCH + k];
                        worst_pixel = worst_pixel.max((got - want).abs());
                    }
                }
            }
        }
    }
    eprintln!("pixels: worst |port - processor| = {worst_pixel:e}");
    assert!(
        worst_pixel < 1e-3,
        "preprocessing disagrees with the processor by {worst_pixel}: a resize or crop differs"
    );

    // Claim 2: the stages.
    let (rows, stages) = runner
        .encode_image_clip_with_stages(&patches)
        .expect("encode");
    let want_hs = |name: &str| read_f32(&dump.join(name));
    let mut results = Vec::new();
    for (label, got, name, width) in [
        ("embed (hs[0])", f16s(&stages.embed), "hs0.f32", 1024),
        ("block 0 (hs[1])", f16s(&stages.block_first), "hs1.f32", 1024),
        (
            "block 22 (hs[-2])",
            f16s(&stages.block_last),
            "hs_last.f32",
            1024,
        ),
        ("projector", f16s(&rows), "features.f32", 4096),
    ] {
        let want = want_hs(name);
        let (whole, worst_row) = cosines(&got, &want, width);
        eprintln!("{label:>18}: cosine {whole:.8}  worst row {worst_row:.6}");
        results.push((label, whole, worst_row));
    }
    // WHICH rows are bad matters more than the number: CLIP-L carries a few
    // high-norm "artifact" rows in its late blocks, where FP16 storage loses
    // the most. Print the five worst of the last block against their norms.
    {
        let got = f16s(&stages.block_last);
        let want = want_hs("hs_last.f32");
        let norm = |r: &[f32]| r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let mut rows: Vec<(usize, f64, f64)> = got
            .chunks_exact(1024)
            .zip(want.chunks_exact(1024))
            .enumerate()
            .map(|(i, (g, w))| {
                let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
                (i, dot / (norm(g) * norm(w)), norm(w))
            })
            .collect();
        let mut norms: Vec<f64> = rows.iter().map(|r| r.2).collect();
        norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "block 22 reference row norm: median {:.1}, max {:.1}",
            norms[norms.len() / 2],
            norms[norms.len() - 1]
        );
        rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let below = rows.iter().filter(|r| r.1 < 0.99).count();
        eprintln!("rows below 0.99: {below} of {}", rows.len());
        // The noise floor, when the probe dumped it: the reference itself in
        // FP16, against its own float32 run, on the same rows.
        let floor = dump.join("hs_last_f16.f32");
        let half_ref = floor.exists().then(|| read_f32(&floor));
        for (i, c, n) in rows.iter().take(5) {
            let note = half_ref.as_ref().map(|h| {
                let (g, w) = (&h[i * 1024..(i + 1) * 1024], &want[i * 1024..(i + 1) * 1024]);
                let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
                dot / (norm(g) * norm(w))
            });
            match note {
                Some(f) => eprintln!(
                    "  row {i:>3}  port {c:.4}  FP16-reference {f:.4}  ref norm {n:.1}"
                ),
                None => eprintln!("  row {i:>3}  cosine {c:.4}  ref norm {n:.1}"),
            }
        }
        if let Some(h) = &half_ref {
            let (floor_whole, floor_worst) = cosines(h, &want, 1024);
            eprintln!(
                "FP16 reference vs float32 reference: cosine {floor_whole:.8}  worst row \
                 {floor_worst:.6}"
            );
            let (whole, worst) = cosines(&got, h, 1024);
            eprintln!(
                "port vs FP16 reference:              cosine {whole:.8}  worst row {worst:.6}"
            );
            // The port stores FP16, so the right yardstick is the reference
            // run at the same precision. 0.99994 here against 0.99948 for the
            // reference's OWN FP16-versus-float32 gap says the residual is
            // storage precision on a few rows, not a function difference.
            assert!(
                whole > 0.9999 && worst > 0.99,
                "port vs the FP16 reference: cosine {whole} (worst row {worst})"
            );
            let port_whole = results[2].1;
            assert!(
                port_whole >= floor_whole - 5e-4,
                "the port is farther from float32 ({port_whole}) than FP16 storage alone \
                 explains ({floor_whole})"
            );
        }
    }
    for (label, whole, worst_row) in &results {
        assert!(whole.is_finite(), "{label}: non-finite");
        // Against float32, the bound is the FP16 floor measured above (worst
        // rows near 0.78 are the reference's own), not a tight per-row one.
        assert!(
            *whole > 0.999 && *worst_row > 0.6,
            "{label}: cosine {whole} (worst row {worst_row}) is below the bound"
        );
    }
}
