//! The `unlimited_ocr2` flow on a synthetic install. Untrained weights make
//! "it decoded" worthless (`crates/runtime` Gotcha 11), so each test moves one
//! input and requires the logits to move, or not to.
//!
//! **Blind spots, stated:** nothing here compares against a reference, and
//! nothing can show V stays unrotated. Those are the real-model gates
//! (`unlimited_ocr2_real.rs`, `docs/UNLIMITED_OCR2_PHASE0.md`).
#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    build_synthetic_unlimited_ocr2_install, SYNTHETIC_UOCR2_IMAGE_TOKEN as IMAGE,
    SYNTHETIC_UOCR2_PAGE_TOKEN as PAGE,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: i64 = 512;
const HIDDEN: usize = 128;

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("uocr2-synth-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open() -> RealForwardRunner {
    let dir = tempdir();
    let arch = build_synthetic_unlimited_ocr2_install(&dir, VOCAB).unwrap();
    RealForwardRunner::open_with_max_context(&dir, arch, 1024).unwrap()
}

/// `<bos> <PAGE> images... text...`, past the point routing can reach them.
fn prompt(images: usize, text: usize) -> Vec<i32> {
    let mut ids = vec![0, PAGE];
    ids.extend(std::iter::repeat_n(IMAGE, images));
    ids.extend((0..text).map(|t| 200 + (t % 100) as i32));
    ids
}

fn rows(images: usize, seed: u32) -> Vec<f16> {
    let mut x = seed.wrapping_mul(2_654_435_761) | 1;
    (0..images * HIDDEN)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            f16::from_f32((x as f32 / u32::MAX as f32) * 2.0 - 1.0)
        })
        .collect()
}

fn run(runner: &mut RealForwardRunner, ids: &[i32], image_rows: Vec<f16>) -> Vec<u16> {
    runner.set_unlimited_ocr2_prompt(ids, image_rows).unwrap();
    LogitProducer::reset(runner);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in ids.iter().enumerate() {
        runner.produce(token, position, &mut logits).unwrap();
    }
    logits.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_used_runner_replays_a_fresh_ones_generation() {
    let ids = prompt(64, 200);
    let fresh = run(&mut open(), &ids, rows(64, 1));
    assert!(fresh.iter().any(|b| *b != 0), "all-zero logits");
    // Against a FRESH runner, and with other rows first. Stale summaries keep
    // the low block indices, so a used runner replaying its own first rows would
    // look up exactly the values it wrote and pass with no reset at all.
    let mut used = open();
    let other = run(&mut used, &ids, rows(64, 2));
    assert_ne!(fresh, other, "image rows never reached the logits");
    assert_eq!(
        fresh,
        run(&mut used, &ids, rows(64, 1)),
        "reset left router or KV state behind"
    );
}

#[test]
fn a_query_routes_to_image_blocks_past_its_local_window() {
    let mut runner = open();
    let ids = prompt(64, 200);
    run(&mut runner, &ids, rows(64, 1));
    let (positions, bias, _) = runner.unlimited_ocr2_last_route().unwrap();
    let pos = ids.len() as u32 - 1;
    let remote: Vec<usize> = (0..positions.len())
        .filter(|&i| positions[i] + 128 <= pos)
        .collect();
    assert!(!remote.is_empty(), "no key past the local window");
    // Trailing text is on no page yet (upstream's response page -1), so it may
    // route to any completed block. `candidates_restrict_a_query_to_its_page`
    // covers the restriction.
    // The whole local window is attended, unbiased, on top.
    let local: Vec<u32> = positions
        .iter()
        .copied()
        .filter(|p| p + 128 > pos)
        .collect();
    assert_eq!(local, ((pos - 127)..=pos).collect::<Vec<_>>());
    assert!(positions.len() <= 512);
    // Three blocks route, so their centred log-priors cannot all be zero unless
    // the prior was skipped. (Bias is what makes this DASH and not a plain cut.)
    assert!(
        remote.iter().any(|&i| bias[i] != 0.0),
        "the routed blocks carry no bias"
    );
    assert!(
        (0..positions.len())
            .filter(|&i| positions[i] + 128 > pos)
            .all(|i| bias[i] == 0.0),
        "the local window must be unbiased"
    );
}

#[test]
fn candidates_restrict_a_query_to_its_page() {
    let mut runner = open();
    // Page 0 is positions 2..=201, page 1 is 203..=402. Block 6 (192..=223)
    // holds both, so it is a candidate for either.
    let mut ids = vec![0, PAGE];
    ids.extend(std::iter::repeat_n(IMAGE, 200));
    ids.push(PAGE);
    ids.extend(std::iter::repeat_n(IMAGE, 200));
    let remote = |runner: &RealForwardRunner, pos: u32| -> Vec<u32> {
        let (positions, _, _) = runner.unlimited_ocr2_last_route().unwrap();
        positions.into_iter().filter(|p| p + 128 <= pos).collect()
    };
    run(&mut runner, &ids, rows(400, 1));
    let pos = ids.len() as u32 - 1;
    let page_one = remote(&runner, pos);
    assert!(!page_one.is_empty(), "no routed keys");
    assert!(
        page_one.iter().all(|&p| p >= 192),
        "a page 1 query read page 0: {page_one:?}"
    );
    // A generated `<PAGE>` starts response page 0, which reads page 0's blocks.
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.produce(PAGE, ids.len(), &mut logits).unwrap();
    let pos = ids.len() as u32;
    let response = remote(&runner, pos);
    assert!(!response.is_empty(), "no routed keys after the marker");
    assert!(
        response.iter().all(|&p| p < 224),
        "response page 0 read page 1: {response:?}"
    );
}

#[test]
fn a_position_rotates_by_its_page_coordinates() {
    let mut runner = open();
    // Two pages. Page coordinates count the markers, and the last pair's angle
    // is the page modulo 2 times an inverse frequency of 1.
    let mut ids = vec![0, PAGE];
    ids.extend(std::iter::repeat_n(IMAGE, 8));
    ids.push(PAGE);
    ids.extend(std::iter::repeat_n(IMAGE, 8));
    ids.extend([200, 201]);
    runner.set_unlimited_ocr2_prompt(&ids, rows(16, 1)).unwrap();
    LogitProducer::reset(&mut runner);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in ids.iter().enumerate() {
        runner.produce(token, position, &mut logits).unwrap();
        let (_, _, angles) = runner.unlimited_ocr2_last_route().unwrap();
        let page_angle = *angles.last().unwrap();
        let (page_mod_2, tok) = match position {
            0 => (0.0, 0.0),
            1..=9 => (1.0, position as f32 - 1.0),
            10..=18 => (0.0, position as f32 - 10.0),
            // Past the last image the pages restart at 0, and these tokens
            // continue page 0's count from BOS.
            _ => (0.0, position as f32 - 18.0),
        };
        assert_eq!(page_angle, page_mod_2, "page angle at {position}");
        assert_eq!(angles[0], tok, "token angle at {position}");
    }
}

#[test]
fn image_rows_reach_later_queries() {
    let mut runner = open();
    // Through routing or through the chain of local windows: this cannot tell
    // which, and `a_query_routes_to_image_blocks_past_its_local_window` does.
    let ids = prompt(64, 200);
    let a = run(&mut runner, &ids, rows(64, 1));
    let b = run(&mut runner, &ids, rows(64, 2));
    assert_ne!(a, b, "the image rows were not injected");
}

#[test]
fn feeding_other_ids_than_the_installed_prompt_is_refused() {
    let mut runner = open();
    let ids = prompt(4, 4);
    runner.set_unlimited_ocr2_prompt(&ids, rows(4, 1)).unwrap();
    LogitProducer::reset(&mut runner);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    let err = runner.produce(5, 0, &mut logits).unwrap_err();
    assert!(err.to_string().contains("installed prompt"), "{err}");
}

#[test]
fn a_prompt_needs_one_row_per_image_token() {
    let mut runner = open();
    let err = runner
        .set_unlimited_ocr2_prompt(&prompt(4, 4), rows(3, 1))
        .unwrap_err();
    assert!(err.to_string().contains("embedding rows"), "{err}");
}

#[test]
fn decode_allocates_no_metal_buffers() {
    let mut runner = open();
    let ids = prompt(8, 8);
    run(&mut runner, &ids, rows(8, 1));
    let before = runner.gpu_buffer_allocations();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for t in 0..40 {
        runner
            .produce(300 + t as i32, ids.len() + t, &mut logits)
            .unwrap();
    }
    assert_eq!(runner.gpu_buffer_allocations(), before);
}

#[test]
fn a_decoded_page_marker_advances_without_error() {
    let mut runner = open();
    let ids = prompt(8, 8);
    run(&mut runner, &ids, rows(8, 1));
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    // Past the last prompt page, where `advance` clamps.
    for (t, token) in [PAGE, 300, PAGE, PAGE, 301].into_iter().enumerate() {
        runner.produce(token, ids.len() + t, &mut logits).unwrap();
    }
    assert!(logits.iter().all(|v| v.to_f32().is_finite()));
}

/// Every tensor of a real install's `uocr2_tower.json`, dumped from
/// `Unlimited-OCR-2-4bit.gturbo` rather than derived from the tower's constants,
/// so the validator's table is checked against something it did not produce.
fn real_tower_layout() -> Vec<(String, Vec<u64>)> {
    let mut out: Vec<(String, Vec<u64>)> = Vec::new();
    let mut add = |name: String, shape: &[u64]| out.push((name, shape.to_vec()));
    for (name, shape) in [
        ("sam.patch_embed.proj.weight", &[768, 16, 16, 3][..]),
        ("sam.patch_embed.proj.bias", &[768]),
        ("sam.pos_embed", &[1, 64, 64, 768]),
        ("sam.neck.0.weight", &[256, 1, 1, 768]),
        ("sam.neck.1.weight", &[256]),
        ("sam.neck.1.bias", &[256]),
        ("sam.neck.2.weight", &[256, 3, 3, 256]),
        ("sam.neck.3.weight", &[256]),
        ("sam.neck.3.bias", &[256]),
        ("sam.net_2.weight", &[512, 3, 3, 256]),
        ("sam.net_3.weight", &[1024, 3, 3, 512]),
        ("clip.embeddings.class_embedding", &[1024]),
        ("clip.embeddings.patch_embedding.weight", &[1024, 14, 14, 3]),
        ("clip.embeddings.position_embedding.weight", &[257, 1024]),
        ("clip.pre_layrnorm.weight", &[1024]),
        ("clip.pre_layrnorm.bias", &[1024]),
        ("projector.weight", &[1280, 2048]),
        ("projector.bias", &[1280]),
        ("image_newline", &[1280]),
        ("view_separator", &[1280]),
    ] {
        add(name.to_string(), shape);
    }
    for block in 0..12 {
        let rel: &[u64] = if [2, 5, 8, 11].contains(&block) {
            &[127, 64]
        } else {
            &[27, 64]
        };
        for (suffix, shape) in [
            ("norm1.weight", &[768][..]),
            ("norm1.bias", &[768]),
            ("norm2.weight", &[768]),
            ("norm2.bias", &[768]),
            ("attn.qkv.weight", &[2304, 768]),
            ("attn.qkv.bias", &[2304]),
            ("attn.proj.weight", &[768, 768]),
            ("attn.proj.bias", &[768]),
            ("attn.rel_pos_h", rel),
            ("attn.rel_pos_w", rel),
            ("mlp.lin1.weight", &[3072, 768]),
            ("mlp.lin1.bias", &[3072]),
            ("mlp.lin2.weight", &[768, 3072]),
            ("mlp.lin2.bias", &[768]),
        ] {
            add(format!("sam.blocks.{block}.{suffix}"), shape);
        }
    }
    for block in 0..24 {
        for (suffix, shape) in [
            ("layer_norm1.weight", &[1024][..]),
            ("layer_norm1.bias", &[1024]),
            ("layer_norm2.weight", &[1024]),
            ("layer_norm2.bias", &[1024]),
            ("self_attn.qkv_proj.weight", &[3072, 1024]),
            ("self_attn.qkv_proj.bias", &[3072]),
            ("self_attn.out_proj.weight", &[1024, 1024]),
            ("self_attn.out_proj.bias", &[1024]),
            ("mlp.fc1.weight", &[4096, 1024]),
            ("mlp.fc1.bias", &[4096]),
            ("mlp.fc2.weight", &[1024, 4096]),
            ("mlp.fc2.bias", &[1024]),
        ] {
            add(format!("clip.transformer.blocks.{block}.{suffix}"), shape);
        }
    }
    out
}

/// The smallest view through a tower whose entries all sit at offset 0 of a bin
/// as long as the largest one: zeros are enough to reach every weight the view
/// reads, with no 800 MB file. `edit` then spoils the index. Returns the length
/// of the projected rows.
fn run_tower_with_index(
    edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) -> Result<usize, turbospark_runtime::RealForwardError> {
    let dir = tempdir();
    let arch = build_synthetic_unlimited_ocr2_install(&dir, VOCAB).unwrap();
    let mut runner = RealForwardRunner::open_with_max_context(&dir, arch, 1024).unwrap();
    let layout = real_tower_layout();
    let largest: u64 = layout
        .iter()
        .map(|(_, shape)| shape.iter().product::<u64>() * 2)
        .max()
        .unwrap();
    let mut index = serde_json::Map::new();
    for (name, shape) in layout {
        index.insert(name, serde_json::json!({ "offset": 0u64, "shape": shape }));
    }
    edit(&mut index);
    std::fs::write(dir.join("uocr2_tower.bin"), vec![0u8; largest as usize]).unwrap();
    std::fs::write(
        dir.join("uocr2_tower.json"),
        serde_json::to_vec(&serde_json::Value::Object(index)).unwrap(),
    )
    .unwrap();
    const SIDE: usize = 64;
    let ran = runner
        .unlimited_ocr2_view_stages(&vec![0.0; SIDE * SIDE * 3], SIDE)
        .map(|stages| stages.projected.len());
    let _ = std::fs::remove_dir_all(&dir);
    ran
}

fn set_entry(
    index: &mut serde_json::Map<String, serde_json::Value>,
    name: &str,
    offset: u64,
    shape: &[u64],
) {
    index.insert(
        name.to_string(),
        serde_json::json!({ "offset": offset, "shape": shape }),
    );
}

/// Refused by name, as an error, rather than by a panic or a wrong answer.
fn assert_refuses(ran: Result<usize, turbospark_runtime::RealForwardError>, name: &str) {
    let message = ran
        .expect_err("a spoiled tower index was accepted")
        .to_string();
    assert!(
        message.contains("uocr2_tower.json") && message.contains(name),
        "refused for some other reason: {message}"
    );
}

/// The control for the refusals below: the real layout is accepted, and it
/// names every weight this view reads.
#[test]
fn tower_index_with_the_real_layout_runs() {
    assert_eq!(run_tower_with_index(|_| {}).unwrap(), 1280);
}

/// 2^32 * 2^32 wraps to 0 elements in release, so `host` returns an empty table
/// where the shape claimed an impossible one.
#[test]
fn tower_index_refuses_a_shape_that_overflows() {
    let ran = run_tower_with_index(|index| {
        set_entry(index, "sam.pos_embed", 0, &[1 << 32, 1 << 32, 1]);
    });
    assert_refuses(ran, "sam.pos_embed");
}

/// Bytes, not elements: 2^63 elements are in range of a u64 and still wrap the
/// byte length to 0.
#[test]
fn tower_index_refuses_a_byte_length_that_overflows() {
    let ran = run_tower_with_index(|index| {
        set_entry(index, "sam.pos_embed", 0, &[1 << 63]);
    });
    assert_refuses(ran, "sam.pos_embed");
}

#[test]
fn tower_index_refuses_an_offset_past_the_bin() {
    let ran = run_tower_with_index(|index| {
        set_entry(index, "sam.pos_embed", 1 << 40, &[1, 64, 64, 768]);
    });
    assert_refuses(ran, "sam.pos_embed");
}

/// The GPU binds this offset as-is, so a range check on `host` alone misses it.
#[test]
fn tower_index_refuses_an_end_that_wraps() {
    let ran = run_tower_with_index(|index| {
        set_entry(index, "sam.net_3.weight", u64::MAX - 1, &[1024, 3, 3, 512]);
    });
    assert_refuses(ran, "sam.net_3.weight");
}

/// `host` floors an odd offset while the GPU binds it, so the two would read
/// different weights from one entry.
#[test]
fn tower_index_refuses_an_odd_offset() {
    let ran = run_tower_with_index(|index| {
        set_entry(index, "sam.pos_embed", 1, &[1, 64, 64, 768]);
    });
    assert_refuses(ran, "sam.pos_embed");
}

/// In the bin, and too short for the table built from it.
#[test]
fn tower_index_refuses_a_short_host_tensor() {
    let ran = run_tower_with_index(|index| {
        set_entry(index, "sam.pos_embed", 0, &[4]);
    });
    assert_refuses(ran, "sam.pos_embed");
}

/// The kernels size their reads from the tower's constants, not this shape, so
/// a short entry at the bin's tail would read past the Metal buffer. Offset 0
/// keeps the unfixed read inside the buffer: the failure is that it is accepted.
#[test]
fn tower_index_refuses_a_short_gpu_tensor() {
    let ran = run_tower_with_index(|index| {
        set_entry(index, "sam.net_3.weight", 0, &[1]);
    });
    assert_refuses(ran, "sam.net_3.weight");
}

/// Opens the tower over a `uocr2_tower.json`/`.bin` pair of the given sizes.
/// Both are sparse, so an oversized file costs no disk, only the reads.
fn open_tower_with_sizes(
    index_bytes: u64,
    bin_bytes: u64,
) -> Result<usize, turbospark_runtime::RealForwardError> {
    let dir = tempdir();
    let arch = build_synthetic_unlimited_ocr2_install(&dir, VOCAB).unwrap();
    let mut runner = RealForwardRunner::open_with_max_context(&dir, arch, 1024).unwrap();
    for (name, len) in [
        ("uocr2_tower.json", index_bytes),
        ("uocr2_tower.bin", bin_bytes),
    ] {
        std::fs::File::create(dir.join(name))
            .and_then(|file| file.set_len(len))
            .unwrap();
    }
    const SIDE: usize = 64;
    let ran = runner
        .unlimited_ocr2_view_stages(&vec![0.0; SIDE * SIDE * 3], SIDE)
        .map(|stages| stages.projected.len());
    let _ = std::fs::remove_dir_all(&dir);
    ran
}

fn assert_too_large(ran: Result<usize, turbospark_runtime::RealForwardError>, name: &str) {
    let message = ran
        .expect_err("an oversized tower file was read")
        .to_string();
    assert!(
        message.contains(name) && message.contains("byte limit"),
        "refused for some other reason: {message}"
    );
}

/// The real index is 61 KB; a crafted one must not be read whole.
#[test]
fn tower_refuses_an_oversized_index() {
    assert_too_large(open_tower_with_sizes((1 << 20) + 1, 2), "uocr2_tower.json");
}

/// The real bin is 803 MB. Before the bound this one was read whole and copied
/// twice more (Metal buffer and host table), about 6 GiB for one image.
#[test]
fn tower_refuses_an_oversized_bin() {
    assert_too_large(open_tower_with_sizes(2, (2 << 30) + 1), "uocr2_tower.bin");
}
