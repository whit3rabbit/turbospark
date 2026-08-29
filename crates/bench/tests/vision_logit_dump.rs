//! This port's full-vocabulary logits for a TEXT+IMAGE prompt, so mlx-vlm can
//! be handed the same ids and the same pixels and the two distributions
//! compared (ROADMAP M-V5, stage 2).
//!
//! # What separates this from the two vision gates that already exist
//!
//! `vision_tower_synthetic.rs` proves the tower composes what this repo's own
//! reference says. `vision_tower_parity.rs` proves the composition is the
//! RIGHT one, against mlx-vlm -- but it stops at the merger, and nothing
//! downstream of it was wired when that gate was written. This is the first
//! instrument that reaches the three things M-V5 adds: which rows land at
//! which positions, which rope angle each position gets, and whether the mRoPE
//! selector agrees with the reference's.
//!
//! # It replays the reference's IDS and its PIXELS, and neither is optional
//!
//! IDS, for `logit_dump.rs`'s reason: a tokenizer or template difference would
//! surface as a divergence and be misread as a numerics gap. PIXELS, for
//! `vision_tower_parity.rs`'s: `crates/vision-io`'s golden fixtures hold this
//! port's preprocessing to the reference and that parity gate holds its tower,
//! so letting either back into this comparison would make a gap unattributable
//! between four candidates instead of one.
//!
//! What is left in the gap is exactly what M-V5 added.
//!
//! # The reference runs FIRST here, unlike every sibling, and it is temporary
//!
//! `logit_dump.rs` writes `meta.json` and Python replays it. M-V6 does not
//! exist, so this port cannot build a text+image id sequence at all -- no
//! splice, no `--image`, no server arm -- and the processor is the authority on
//! the ids until it does. Flip this to match its siblings when M-V6 lands, and
//! compare the two splices rather than taking one on trust.
//!
//! # Setup
//!
//! See `scripts/kld_mlx_vlm.py`'s docstring for all three steps. In short:
//!
//! ```sh
//! uv run --python 3.12 --with mlx --with mlx-vlm --with numpy --with pillow \
//!   --with transformers -- \
//!   python scripts/kld_mlx_vlm.py prepare /tmp/vision-kld \
//!     --image ~/models/vision-probe-qwen38/imgs/page.png
//!
//! TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
//! TURBOSPARK_VISION_KLD_DIR=/tmp/vision-kld \
//!   cargo test -p turbospark-bench --test vision_logit_dump --release -- \
//!   --ignored --nocapture
//!
//! uv run --python 3.12 --with numpy -- \
//!   python scripts/kld_mlx_vlm.py compare /tmp/vision-kld
//! ```
#![cfg(target_os = "macos")]

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use foundation::LogitValue;
use runtime::LogitProducer;
use turbospark_bench::protocol::{PROTOCOL_EXPERT_CACHE_SLOTS, PROTOCOL_MAX_CONTEXT};
use turbospark_vision_io::{mrope_position_triples, GridThw, PreprocessedImage, VisionSpecialIds};

/// Tokens to generate past the prompt, for the end-to-end arm.
///
/// Enough to see whether the model is transcribing the page rather than
/// answering from the question alone, short enough that a mismatch is
/// readable side by side.
const GENERATE_TOKENS: usize = 48;

/// Greedy pick. Non-finite is REFUSED rather than ranked: NaN loses every
/// comparison, so a naive argmax over a NaN row silently returns index 0 and
/// the run reads as a model that keeps proposing the same token (AGENTS.md
/// Gotcha 59, and `dflash_select`'s exact failure).
fn argmax(logits: &[LogitValue]) -> i32 {
    assert!(
        logits.iter().all(|v| v.to_f32().is_finite()),
        "a non-finite logit reached the greedy pick"
    );
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if v.to_f32() > logits[best].to_f32() {
            best = i;
        }
    }
    best as i32
}

fn env_dir(key: &str) -> Option<PathBuf> {
    let raw = std::env::var(key).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(shellexpand(&raw)))
}

/// `~` only, matching `vision_tower_parity.rs`. A full shell expansion here
/// would be a second, worse shell.
fn shellexpand(raw: &str) -> String {
    match raw.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => raw.to_string(),
    }
}

/// The `prepare` step's sidecar.
struct Header {
    input_ids: Vec<i32>,
    grid: GridThw,
    pixel_shape: Vec<usize>,
    rows: usize,
    vocab: usize,
    image_token_id: i32,
    vision_start_token_id: i32,
    raw: serde_json::Value,
}

fn load_header(dir: &Path) -> Header {
    let raw: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("header.json"))
            .expect("header.json -- run `kld_mlx_vlm.py prepare` before this test"),
    )
    .expect("header.json parses");
    let ints = |key: &str| -> Vec<usize> {
        raw[key]
            .as_array()
            .unwrap_or_else(|| panic!("header.json is missing {key}"))
            .iter()
            .map(|v| v.as_u64().expect("a non-negative integer") as usize)
            .collect()
    };
    let g = ints("grid_thw");
    assert_eq!(g.len(), 3, "grid_thw must be [t, h, w]");
    Header {
        input_ids: raw["input_ids"]
            .as_array()
            .expect("input_ids")
            .iter()
            .map(|v| v.as_i64().expect("an integer id") as i32)
            .collect(),
        grid: GridThw::new(g[0], g[1], g[2]),
        pixel_shape: ints("pixel_values_shape"),
        rows: raw["rows"].as_u64().expect("rows") as usize,
        vocab: raw["vocab"].as_u64().expect("vocab") as usize,
        image_token_id: raw["image_token_id"].as_i64().expect("image_token_id") as i32,
        vision_start_token_id: raw["vision_start_token_id"]
            .as_i64()
            .expect("vision_start_token_id") as i32,
        raw,
    }
}

/// The reference's own `pixel_values`, as this port's patch rows.
///
/// **ALREADY PERMUTED BY `prepare`, and the two orders are NOT the same.** The
/// processor emits each row as `(C, T, P_h, P_w)`; this port emits
/// `(T, P_h, P_w, C)` so the repack can copy `patch_embed.proj.weight`
/// verbatim (`crates/vision-io` Gotcha 1). Feeding the raw rows does not fail
/// -- the GEMM keeps its shape, the tower runs, and the model reads a
/// DIFFERENT IMAGE -- which is what this script's first run measured at 18.87
/// mean nats on the image positions against a near-exact text median.
///
/// Replaying them at all, rather than preprocessing here, is what keeps a
/// pixel difference out of this comparison
/// (`vision_tower_parity.rs`'s discipline).
fn load_patch_rows(dir: &Path, shape: &[usize]) -> Vec<f32> {
    load_f32(dir, "pixel_values.f32", shape.iter().product())
}

/// A little-endian f32 sidecar, length-checked against what the header
/// declares. The check is the point: a shape that disagrees is a dump from a
/// different run, and it would otherwise present as a numerics gap.
fn load_f32(dir: &Path, name: &str, want: usize) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(name))
        .unwrap_or_else(|e| panic!("{name}: {e} -- re-run `kld_mlx_vlm.py prepare`"));
    assert_eq!(
        bytes.len(),
        want * 4,
        "{name} is {} bytes against a declared {want} values",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "needs a real vision install (TURBOSPARK_QWEN38_VISION_INSTALL_DIR) \
            and a prepared dump (TURBOSPARK_VISION_KLD_DIR)"]
fn dump_text_and_image_logits() {
    let (Some(install), Some(dir)) = (
        env_dir("TURBOSPARK_QWEN38_VISION_INSTALL_DIR"),
        env_dir("TURBOSPARK_VISION_KLD_DIR"),
    ) else {
        eprintln!(
            "vision_logit_dump: needs TURBOSPARK_QWEN38_VISION_INSTALL_DIR and \
             TURBOSPARK_VISION_KLD_DIR; skipping."
        );
        return;
    };

    let header = load_header(&dir);
    let ids = header.input_ids.clone();
    assert_eq!(
        header.rows,
        ids.len() - 1,
        "the reference dropped a different number of rows than there are ids"
    );

    // Opened DIRECTLY rather than through `open_model_runner`, which also
    // loads a tokenizer -- and this install has none. The vision install is
    // streamed for the tower and carries no sidecars, which is fine here
    // precisely because the ids are replayed: a tokenizer would be a second
    // source for a sequence the reference already fixed. When M-V6 lands and
    // this port builds its own splice, that stops being true and the sidecars
    // become a requirement.
    let arch = repack::peek_manifest_arch(&install).expect("peeks");
    let mut runner = runtime::RealForwardRunner::open_with_options(
        &install,
        arch.clone(),
        PROTOCOL_MAX_CONTEXT as usize,
        PROTOCOL_EXPERT_CACHE_SLOTS,
    )
    .expect("the install opens");
    assert!(
        runner.has_vision_tower(),
        "{} declares no vision tower; point this at the install streamed WITH \
         vision_tower.*",
        install.display()
    );
    let vocab = runner.vocab_size();
    assert_eq!(
        vocab, header.vocab,
        "this install's vocabulary is {vocab} against the reference's {}: the \
         two are not the same checkpoint",
        header.vocab
    );

    // The tower, on the reference's own patch rows.
    let params = params_from(&arch);
    let patch_rows = load_patch_rows(&dir, &header.pixel_shape);
    let image = PreprocessedImage {
        merged_tokens: header.grid.merged_tokens(params.merge_size),
        patch_rows,
        grid: header.grid,
        resized: (
            header.grid.h * params.patch_size,
            header.grid.w * params.patch_size,
        ),
    };
    // TWO ARMS, and running both is what makes a gap attributable.
    //
    // The default runs THIS PORT's tower, which is the whole pipeline and the
    // number to publish. `TURBOSPARK_VISION_REPLAY_ROWS=1` substitutes the
    // reference's OWN merger rows instead, holding the injection and the trunk
    // fixed and taking the tower out of the comparison. Where the two arms
    // disagree, the difference is the tower's FP16 error amplified through 64
    // layers; where they agree, it is not.
    //
    // `vision_tower_parity.rs`'s four-stage argument one level up: a composite
    // gap localizes only when its stages can be substituted one at a time.
    let replay_rows = std::env::var_os("TURBOSPARK_VISION_REPLAY_ROWS").is_some();
    let embedding = if replay_rows {
        let shape: Vec<usize> = header.raw["image_features_shape"]
            .as_array()
            .expect("image_features_shape -- re-run `kld_mlx_vlm.py prepare`")
            .iter()
            .map(|v| v.as_u64().expect("an integer") as usize)
            .collect();
        let values = load_f32(&dir, "image_features.f32", shape.iter().product());
        eprintln!(
            "vision_logit_dump: REPLAYING the reference's {} x {} merger rows; \
             this port's tower is NOT in this comparison",
            shape[0], shape[1]
        );
        runtime::vision::VisionEmbedding {
            rows: values
                .iter()
                .map(|&v| LogitValue::from_f32(v).to_bits())
                .collect(),
            merged_tokens: shape[0],
            out_hidden: shape[1],
            grid: header.grid,
        }
    } else {
        runner
            .encode_image(&image, &params)
            .expect("the tower runs")
    };
    eprintln!(
        "vision_logit_dump: {} merged rows x {}",
        embedding.merged_tokens, embedding.out_hidden
    );

    // The position table, from this port's own walk of the reference's ids.
    // This IS one of the things under test: the walk is a port of
    // `get_rope_index` and a disagreement with it moves every angle past the
    // image.
    let positions = mrope_position_triples(
        &ids,
        &[header.grid],
        VisionSpecialIds {
            vision_start: header.vision_start_token_id,
            image_pad: header.image_token_id,
        },
        params.merge_size,
    )
    .expect("the walk places the reference's image");
    eprintln!(
        "vision_logit_dump: {} span(s), rope_delta {}",
        positions.spans.len(),
        positions.rope_delta
    );

    runner
        .set_prompt_vision(std::slice::from_ref(&embedding), &positions, ids.len())
        .expect("the injection map validates against the reference's prompt");

    // Row i holds the next-token logits after consuming ids[i], so the last id
    // is fed to nobody -- the reference drops the same row in `prepare`.
    let path = dir.join("port.f16");
    let mut file = BufWriter::new(std::fs::File::create(&path).expect("create port.f16"));
    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    let mut written = 0usize;
    for (position, &token) in ids.iter().enumerate().take(ids.len() - 1) {
        runner
            .produce(token, position, &mut logits)
            .unwrap_or_else(|e| panic!("produce at position {position}: {e}"));
        // Finiteness at the point the measurement is TAKEN, not where it is
        // used: NaN reads as a perfect score on the top-1 instrument the
        // comparison runs (AGENTS.md Gotcha 59), and this tower's FP16 stream
        // is exactly the shape that can overflow (Gotcha 60).
        assert!(
            logits.iter().all(|v| v.to_f32().is_finite()),
            "position {position} produced a non-finite logit"
        );
        let mut bytes = Vec::with_capacity(logits.len() * 2);
        for value in &logits {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        file.write_all(&bytes).expect("write a logit row");
        written += 1;
    }
    file.flush().expect("flush port.f16");
    assert_eq!(written, header.rows, "row count must match the reference's");

    // GREEDY CONTINUATION, past the prompt. The divergence numbers above are
    // the instrument; this is the thing a reader actually wants to know, and
    // it is the only arm that exercises the DECODE side of the position rule
    // (`rope_position` past the prompt resolves to `position + rope_delta`,
    // which no prompt position reaches).
    //
    // Ids rather than text, because this install carries no tokenizer -- the
    // comparison script decodes both sides with the reference's.
    let mut generated = Vec::new();
    let mut next = argmax(&logits);
    for step in 0..GENERATE_TOKENS {
        generated.push(next);
        let position = ids.len() - 1 + step;
        runner
            .produce(next, position, &mut logits)
            .unwrap_or_else(|e| panic!("produce at decode position {position}: {e}"));
        next = argmax(&logits);
    }
    eprintln!("vision_logit_dump: generated {} ids", generated.len());

    let meta = serde_json::json!({
        "install": install.to_string_lossy(),
        "generated_ids": generated,
        "input_ids": header.input_ids,
        "rows": written,
        "vocab": vocab,
        "spans": positions.spans.iter().map(|s| [s.start, s.len]).collect::<Vec<_>>(),
        "rope_delta": positions.rope_delta,
        "merged_tokens": embedding.merged_tokens,
        "reference_header": header.raw,
        "dtype": "float16",
    });
    std::fs::write(
        dir.join("port_meta.json"),
        serde_json::to_string_pretty(&meta).expect("serializes"),
    )
    .expect("write port_meta.json");

    eprintln!(
        "vision_logit_dump: wrote {written} rows x {vocab} float16 to {}",
        path.display()
    );
}

/// Preprocessing parameters read off the INSTALL, never restated
/// (`vision_tower_parity.rs`'s rule).
fn params_from(arch: &model_io::ArchConfig) -> turbospark_vision_io::PreprocessParams {
    let v = &arch.vision;
    turbospark_vision_io::PreprocessParams {
        patch_size: v.patch_size as usize,
        temporal_patch_size: v.temporal_patch_size as usize,
        merge_size: v.spatial_merge_size as usize,
        in_channels: v.in_channels as usize,
        min_pixels: 65_536,
        max_pixels: 16_777_216,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    }
}
