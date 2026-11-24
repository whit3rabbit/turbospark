#![cfg(target_os = "macos")]
//! Gemma 3 4B-and-up linear rope scaling on real Metal: a tiny Gemma 3 GGUF
//! that declares `rope.scaling.type = linear` installs through the real GGUF
//! walk, opens, decodes finite logits, and differs from the same weights
//! without scaling.
//!
//! This proves the config reaches the manifest, the open path builds the
//! global-layer table and the decode path takes the per-pair branch. It does
//! NOT prove agreement with llama.cpp on a real 4B file (none was run; see
//! `docs/GEMMA3_PHASE0.md`). Weights are untrained: assert on structure and
//! on the two runs differing, never on text (AGENTS.md Gotcha 12).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    parse_gguf_header, write_gguf_install_streamed, GgufBuilder, MemoryRangeSource,
    GGUF_DEFAULT_MAX_HEADER_BYTES,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);
const LAYERS: usize = 6;
const HIDDEN: u64 = 128;
const HEAD_DIM: u64 = 256;
const HEADS: u64 = 2;
const INTER: u64 = 256;
const VOCAB: usize = 256;
const STEPS: usize = 64;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-gemma3-linear-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn norm(base: f32, n: u64) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (base + (i % 16) as f32 / 16.0).to_le_bytes())
        .collect()
}

/// Raw Q8_0 weights (`rows` output rows of `HIDDEN` inputs) with a LARGE scale.
/// The stock fixture weights are small enough that attention is near-uniform,
/// so a rotation changes the output by less than fp16 can show and the test
/// would pass or fail on rounding. Large q and k make the scores peaked, so
/// the angle decides which position is attended and the logits move visibly.
fn peaked_q8_0(rows: u64, scale: f32, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut out = Vec::new();
    for _ in 0..rows * (HIDDEN / 32) {
        out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        for _ in 0..32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            out.push(((state >> 24) as i8) as u8);
        }
    }
    out
}

fn gemma3_gguf(linear_factor: Option<f32>) -> Vec<u8> {
    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "gemma3")
        .metadata_u32("gemma3.block_count", LAYERS as u32)
        .metadata_u32("gemma3.embedding_length", HIDDEN as u32)
        .metadata_u32("gemma3.feed_forward_length", INTER as u32)
        .metadata_u32("gemma3.attention.head_count", HEADS as u32)
        .metadata_u32("gemma3.attention.head_count_kv", 1)
        .metadata_u32("gemma3.attention.key_length", HEAD_DIM as u32)
        .metadata_u32("gemma3.attention.value_length", HEAD_DIM as u32)
        .metadata_u32("gemma3.attention.sliding_window", 512)
        .metadata_f32("gemma3.rope.freq_base", 1_000_000.0)
        .metadata_f32("gemma3.attention.layer_norm_rms_epsilon", 1e-6)
        // Distinct rows: the stock fixture's embedding repeats one block, so
        // every token is the same vector and attention cannot see position.
        .tensor(
            "token_embd.weight",
            8,
            &[HIDDEN, VOCAB as u64],
            peaked_q8_0(VOCAB as u64, 0.02, 1),
        );
    if let Some(f) = linear_factor {
        b = b
            .metadata_str("gemma3.rope.scaling.type", "linear")
            .metadata_f32("gemma3.rope.scaling.factor", f);
    }
    for l in 0..LAYERS {
        let t = |s: &str| format!("blk.{l}.{s}.weight");
        b = b
            .tensor(
                &t("attn_q"),
                8,
                &[HIDDEN, HEADS * HEAD_DIM],
                peaked_q8_0(HEADS * HEAD_DIM, 0.05, 3 + l as u32),
            )
            .tensor(
                &t("attn_k"),
                8,
                &[HIDDEN, HEAD_DIM],
                peaked_q8_0(HEAD_DIM, 0.05, 4 + l as u32),
            )
            .q8_0_tensor(&t("attn_v"), &[HIDDEN, HEAD_DIM], 5)
            .q8_0_tensor(&t("attn_output"), &[HEADS * HEAD_DIM, HIDDEN], 7)
            .q8_0_tensor(&t("ffn_gate"), &[HIDDEN, INTER], 8)
            .q8_0_tensor(&t("ffn_up"), &[HIDDEN, INTER], 9)
            .q8_0_tensor(&t("ffn_down"), &[INTER, HIDDEN], 10)
            .tensor(&t("attn_norm"), 0, &[HIDDEN], norm(1.0, HIDDEN))
            .tensor(&t("post_attention_norm"), 0, &[HIDDEN], norm(2.0, HIDDEN))
            .tensor(&t("ffn_norm"), 0, &[HIDDEN], norm(3.0, HIDDEN))
            .tensor(&t("post_ffw_norm"), 0, &[HIDDEN], norm(4.0, HIDDEN))
            .tensor(&t("attn_q_norm"), 0, &[HEAD_DIM], norm(5.0, HEAD_DIM))
            .tensor(&t("attn_k_norm"), 0, &[HEAD_DIM], norm(6.0, HEAD_DIM));
    }
    b.tensor("output_norm.weight", 0, &[HIDDEN], norm(7.0, HIDDEN))
        .build()
        .0
}

fn logits(linear_factor: Option<f32>) -> Vec<Vec<f32>> {
    let bytes = gemma3_gguf(linear_factor);
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let dir = temp_dir();
    let arch = write_gguf_install_streamed(
        &dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "tiny-gemma3",
        |_| {},
    )
    .expect("gemma3 install writes");
    assert_eq!(arch.rope_scaling.is_linear(), linear_factor.is_some());
    let mut runner = RealForwardRunner::open(&dir, arch).expect("gemma3 install opens");
    runner.reset();
    let mut out = Vec::new();
    for position in 0..STEPS {
        let token = ((position * 5 + 1) % VOCAB) as i32;
        let mut head = vec![f16::from_f32(0.0); VOCAB];
        runner.produce(token, position, &mut head).expect("produce");
        let row: Vec<f32> = head.iter().map(|v| v.to_f32()).collect();
        assert!(
            row.iter().all(|v| v.is_finite()),
            "non-finite at {position}"
        );
        out.push(row);
    }
    std::fs::remove_dir_all(&dir).ok();
    out
}

#[test]
fn linear_scaling_installs_opens_and_changes_the_global_layer_logits() {
    let plain = logits(None);
    let scaled = logits(Some(1000.0));
    let diffs: Vec<f32> = plain
        .iter()
        .zip(&scaled)
        .map(|(p, s)| {
            p.iter()
                .zip(s)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max)
        })
        .collect();
    // Position 0 is angle 0 for every pair under any scaling.
    assert_eq!(diffs[0], 0.0);
    // Attention here is peaked but the fixture is untrained, so only SOME
    // positions flip which key wins; one visible difference anywhere is the
    // proof the table reaches the kernel (an ignored table gives identical
    // rows everywhere; verified by disabling the table, which fails this assertion).
    let worst = diffs.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        worst > 1e-3,
        "scaled and unscaled logits agree at all {STEPS} positions: the global-layer table is \
         not reaching the kernel"
    );
}
