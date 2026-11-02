//! The DeepSeek-VL2 SigLIP tower on a REAL install: preprocess an image, run
//! the tower with a stage readback, and dump what a second implementation can
//! be held to (`docs/DEEPSEEK_VL2_VISION_PHASE0.md`, "The tower gate").
//!
//! Written next to `TURBOSPARK_DSVL2_OUT`:
//!
//! - `<out>.pixels.f32`: the tiles' patch rows the engine fed the tower,
//!   little endian f32, tile-major;
//! - `<out>.{embed,block_first,block_last,normed,projector}.f16`: the residual
//!   stream at five points, FP16 bits;
//! - `<out>.rows.f16`: the assembled prompt run (projector rows with the
//!   newline and separator rows interleaved).
//!
//! `scripts/deepseek_vl2_tower_reference.py` computes the same stages from the
//! checkpoint's safetensors with numpy alone and compares.
//!
//! ```sh
//! TURBOSPARK_DSVL2_INSTALL_DIR=/Volumes/T7/tshome-dsvl2/models/text/dsvl2-tiny.gturbo \
//! TURBOSPARK_DSVL2_SIDECAR_DIR=/Volumes/T7/tshome-dsvl2/models/vision/dsvl2-tiny-vision.gturbo-vision \
//! TURBOSPARK_DSVL2_IMAGE=/Volumes/T7/vision-pages/shapes.png \
//! TURBOSPARK_DSVL2_OUT=/Volumes/T7/vision-dumps/dsvl2-tower \
//!   cargo test -p turbospark-runtime --test deepseek_vl2_tower_real --release -- \
//!   --ignored --nocapture
//! ```

use std::path::PathBuf;
use turbospark_runtime::RealForwardRunner;
use turbospark_vision_io::{decode_image_file, deepseek_vl2_preprocess, DeepseekVl2Params};

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key} (see this file's doc)")))
}

fn bits_to_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

#[test]
#[ignore = "needs the real dsvl2-tiny install, its vision sidecar and Metal; see the module doc"]
fn the_real_tower_dumps_its_stages() {
    let install = env_path("TURBOSPARK_DSVL2_INSTALL_DIR");
    let sidecar = env_path("TURBOSPARK_DSVL2_SIDECAR_DIR");
    let out = env_path("TURBOSPARK_DSVL2_OUT");
    let image = decode_image_file(&env_path("TURBOSPARK_DSVL2_IMAGE")).expect("decodes");
    let params = DeepseekVl2Params::from_processor_config_json(
        &std::fs::read_to_string(sidecar.join("processor_config.json")).expect("processor config"),
    )
    .expect("parses");
    let crop = std::env::var("TURBOSPARK_DSVL2_NO_CROP").is_err();
    let tiles = deepseek_vl2_preprocess(&image, &params, crop).expect("preprocess");

    let arch = turbospark_repack::peek_manifest_arch(&install).expect("install manifest");
    let mut runner = RealForwardRunner::open(&install, arch).expect("open install");
    runner.attach_vision_sidecar(&sidecar).expect("attach");
    let (rows, stages) = runner
        .encode_image_siglip_with_stages(&tiles)
        .expect("tower runs");

    let pixels: Vec<u8> = tiles.pixels.iter().flat_map(|v| v.to_le_bytes()).collect();
    let at = |ext: &str| out.with_extension(ext);
    std::fs::write(at("pixels.f32"), pixels).expect("write pixels");
    for (name, bits) in [
        ("embed", &stages.embed),
        ("block_first", &stages.block_first),
        ("block_last", &stages.block_last),
        ("normed", &stages.normed),
        ("projector", &stages.projector),
        ("rows", &rows),
    ] {
        std::fs::write(at(&format!("{name}.f16")), bits_to_bytes(bits)).expect("write stage");
    }
    let width = runner.vision_config().out_hidden_size as usize;
    println!(
        "{} tile(s), {} prompt rows of {width}",
        tiles.tiles(),
        rows.len() / width
    );
}
