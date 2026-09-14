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
//! # THIS PORT BUILDS THE IDS AND THE REFERENCE'S ARE THE ORACLE (M-V6)
//!
//! Until M-V6 this file REPLAYED the processor's id sequence, because nothing
//! here could produce one. It renders the prompt through the checkpoint's own
//! template, encodes it, and expands the placeholder run itself now, and the
//! equality against `header.input_ids` is M-V6's whole gate -- `splice_and_walk`
//! is what closed it.
//!
//! Everything downstream then runs on the port's OWN ids, so a splice that
//! agreed on length and disagreed on placement cannot hide behind a replayed
//! sequence.
//!
//! The reference still runs FIRST, which is now only about the ORDER of the two
//! commands rather than about who owns the prompt: `prepare` needs to write the
//! pixels and the question before this can read them.
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
use turbospark_vision_io::{GridThw, PreprocessedImage};

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

/// Render the prompt this port's own way, encode it, and expand the
/// placeholder run (ROADMAP M-V6).
///
/// **THE TOKENIZER COMES FROM A SEPARATE DIRECTORY BY DEFAULT**, because
/// `qwen38-27b-vision.gturbo` was streamed for its tower and carries no
/// sidecars. `TURBOSPARK_VISION_TOKENIZER_DIR` points at one that does;
/// `~/models/qwen38-27b.gturbo` is the same checkpoint
/// (`mlx-community/Qwen3.8-27B-4bit`) and its `tokenizer.json` and
/// `chat_template.jinja` are BYTE-IDENTICAL to the reference snapshot's,
/// which is what makes borrowing them correct rather than convenient.
/// Pointing it somewhere wrong is self-checking: the assertion against the
/// processor's ids fails loudly.
///
/// The message shape mirrors what `apply_chat_template(processor, config,
/// question, num_images=1)` builds -- ONE user turn whose content is
/// `[image, text]`, in that order. The order is not cosmetic: the template
/// emits the marker run where the part sits, so swapping them moves the image
/// after the question and changes every position past it.
fn build_prompt_ids(
    install: &Path,
    dir: &Path,
    header: &Header,
) -> turbospark_vision_io::SplicedPrompt {
    let tokenizer_dir =
        env_dir("TURBOSPARK_VISION_TOKENIZER_DIR").unwrap_or_else(|| install.to_path_buf());
    let tokenizer = tokenizer::MfTokenizer::load_from_dir(&tokenizer_dir).unwrap_or_else(|e| {
        panic!(
            "no tokenizer in {}: {e}\n  the vision install carries no sidecars; set \
             TURBOSPARK_VISION_TOKENIZER_DIR to an install of the same checkpoint",
            tokenizer_dir.display()
        )
    });

    let question = header.raw["question"]
        .as_str()
        .expect("header.json carries the question `prepare` rendered")
        .to_string();
    let messages = [tokenizer::Message::with_parts(
        tokenizer::Role::User,
        vec![
            tokenizer::ContentPart::Image,
            tokenizer::ContentPart::Text(question),
        ],
    )];
    let rendered = tokenizer
        .apply_chat_template(&messages)
        .expect("the checkpoint's template renders an image part");
    let encoded = tokenizer.encode(&rendered, false);

    // ONE placeholder in, `merged_tokens` out, with the count derived from the
    // GRID rather than from the reference's own placeholder run -- taking it
    // from the dump would make the splice agree with the oracle by
    // construction and test nothing.
    let params = params_from(&repack::peek_manifest_arch(install).expect("peeks"));
    let spliced = turbospark_vision_io::splice_and_walk(
        &encoded,
        &[header.grid],
        turbospark_vision_io::VisionSpecialIds {
            vision_start: header.vision_start_token_id,
            image_pad: header.image_token_id,
        },
        params.merge_size,
    )
    .expect("one placeholder expands to the grid's merged-token count");
    eprintln!(
        "vision_logit_dump: rendered {} ids, spliced to {} ({} merged tokens)",
        encoded.len(),
        spliced.ids.len(),
        header.grid.merged_tokens(params.merge_size),
    );
    let _ = dir;
    spliced
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
    assert_eq!(
        header.rows,
        header.input_ids.len() - 1,
        "the reference dropped a different number of rows than there are ids"
    );

    // THIS PORT BUILDS THE ID SEQUENCE, and the reference's is the ORACLE
    // (ROADMAP M-V6). Until M-V6 this test REPLAYED the reference's ids
    // because nothing here could produce them; the splice is what flipped
    // that, and comparing the two sequences is the milestone's own gate.
    //
    // Everything downstream then runs on the port's OWN ids, so a splice that
    // agreed on length and disagreed on placement could not hide behind a
    // replayed sequence.
    let prompt = build_prompt_ids(&install, &dir, &header);
    let ids = prompt.ids.clone();
    assert_eq!(
        ids, header.input_ids,
        "this port's rendered-and-spliced prompt differs from the processor's"
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
    let positions = prompt.positions;
    eprintln!(
        "vision_logit_dump: {} span(s), rope_delta {}",
        positions.spans.len(),
        positions.rope_delta
    );

    runner
        .set_prompt_vision(std::slice::from_ref(&embedding), &positions, ids.len())
        .expect("the injection map validates against the reference's prompt");

    // Row i holds the next-token logits after consuming ids[i]. The comparison
    // dump omits the final row because the reference drops it in `prepare`, but
    // the final id must still enter the KV cache before greedy generation.
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

    let final_prompt_position = ids.len() - 1;
    runner
        .produce(
            ids[final_prompt_position],
            final_prompt_position,
            &mut logits,
        )
        .unwrap_or_else(|e| panic!("produce at position {final_prompt_position}: {e}"));

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
        let position = ids.len() + step;
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
