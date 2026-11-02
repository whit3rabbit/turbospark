//! The LFM3-VL tower on a REAL combined install: runs the deterministic page
//! through the Metal tower and dumps every stage so
//! `scripts/lfm3_vl_tower_probe.py` can compare it with mlx-vlm's own
//! `lfm3_vl.vision` on the same patch rows (`docs/LFM3_VL_VISION_PHASE0.md`).
//!
//! ```sh
//! uv run --python 3.12 --with pillow scripts/make_vision_test_page.py \
//!   /Volumes/T7/vision-pages/page.png --size 1024 1280
//!
//! TURBOSPARK_LFM3_VL_INSTALL_DIR=/Volumes/T7/turbospark-models/lfm3-vl-2b.gturbo \
//! TURBOSPARK_LFM3_VL_PAGE=/Volumes/T7/vision-pages/page-small.png \
//! TURBOSPARK_LFM3_VL_DUMP_DIR=/Volumes/T7/lfm3-vl-dump \
//!   cargo test -p turbospark-runtime --test lfm3_vl_vision_real --release -- \
//!   --ignored --nocapture
//!
//! uv run --python 3.12 --with mlx --with numpy --with transformers -- \
//!   python scripts/lfm3_vl_tower_probe.py --vendor-root /Volumes/T7/refs/mlx-vlm \
//!     --model-dir /Volumes/T7/hf-models/LFM3-VL-2B-MLX-4bit \
//!     --dump-dir /Volumes/T7/lfm3-vl-dump
//! ```
//!
//! Opt-in like every real-install arm: it needs a GPU and the install.

use std::path::PathBuf;

use turbospark_vision_io::{decode_image_file, preprocess, PreprocessParams};

fn env_path(key: &str) -> Option<PathBuf> {
    let raw = std::env::var(key).ok().filter(|s| !s.is_empty())?;
    Some(match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(raw),
    })
}

fn bits_to_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

#[test]
#[ignore = "needs the real LFM3-VL combined install, a GPU and the deterministic page"]
fn the_real_tower_runs_and_dumps_every_stage() {
    let dir = env_path("TURBOSPARK_LFM3_VL_INSTALL_DIR").expect("TURBOSPARK_LFM3_VL_INSTALL_DIR");
    let page = env_path("TURBOSPARK_LFM3_VL_PAGE").expect("TURBOSPARK_LFM3_VL_PAGE");
    let dump = env_path("TURBOSPARK_LFM3_VL_DUMP_DIR").expect("TURBOSPARK_LFM3_VL_DUMP_DIR");
    std::fs::create_dir_all(&dump).expect("dump dir");

    let arch = turbospark_repack::peek_manifest_arch(&dir).expect("install manifest");
    assert!(arch.vision.is_active(), "the install declares no tower");
    assert_eq!(arch.vision.tower, model_io::VisionTowerKind::Lfm3Vl);
    // The install's own preprocessor config, as the CLI reads it.
    let json = std::fs::read_to_string(dir.join("preprocessor_config.json"))
        .expect("the install carries preprocessor_config.json");
    let params = PreprocessParams::from_preprocessor_config_json(&json).expect("preprocessor");
    let rgb = decode_image_file(&page).expect("decode the page");
    let image = preprocess(&rgb, &params).expect("preprocess the page");
    eprintln!(
        "grid {}x{}x{}, {} merged tokens, resized {:?}",
        image.grid.t, image.grid.h, image.grid.w, image.merged_tokens, image.resized
    );

    let mut runner = turbospark_runtime::RealForwardRunner::open(&dir, arch).expect("open install");
    let (embedding, blocks, patch_embed) = runner
        .encode_image_lfm3_stages(&image, &params)
        .expect("the tower runs");
    assert_eq!(embedding.merged_tokens, image.merged_tokens);
    assert_eq!(embedding.out_hidden, 2048);

    std::fs::write(
        dump.join("patch_rows.f32"),
        image
            .patch_rows
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    std::fs::write(
        dump.join("grid.json"),
        format!(
            "{{\"t\": {}, \"h\": {}, \"w\": {}, \"patch_dim\": {}}}",
            image.grid.t,
            image.grid.h,
            image.grid.w,
            params.patch_dim()
        ),
    )
    .unwrap();
    std::fs::write(dump.join("patch_embed.f16"), bits_to_bytes(&patch_embed)).unwrap();
    for (i, block) in blocks.iter().enumerate() {
        std::fs::write(dump.join(format!("block_{i}.f16")), bits_to_bytes(block)).unwrap();
    }
    std::fs::write(dump.join("rows.f16"), bits_to_bytes(&embedding.rows)).unwrap();
    eprintln!("dumped {} blocks to {}", blocks.len(), dump.display());

    // Not a parity claim (the probe makes that): the rows are finite and not
    // constant, which a never-run or zero-weight tower would fail.
    let rows: Vec<f32> = embedding
        .rows
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();
    assert!(rows.iter().all(|v| v.is_finite()));
    let mean = rows.iter().sum::<f32>() / rows.len() as f32;
    let var = rows.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / rows.len() as f32;
    assert!(var > 1e-6, "the output rows are constant (variance {var})");
}
