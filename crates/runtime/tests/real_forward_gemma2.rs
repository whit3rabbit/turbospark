#![cfg(target_os = "macos")]
//! Gemma 2 on the Gemma flow, on real Metal: a tiny Gemma 2 GGUF installs
//! through the real GGUF walk (no q/k norm tensors anywhere), opens, decodes
//! finite logits bounded by the final-logit cap, and a file that states a
//! different attention cap is refused instead of decoded with the family's.
//!
//! What this proves: the family wiring, the alternating local/global mask, the
//! absence of q/k norms in the flow, and the two refusals. What it does NOT
//! prove: agreement with llama.cpp. The attention-score cap's arithmetic is
//! checked against a CPU reference in `crates/gpu/tests/attention_softcap.rs`,
//! and the whole model against llama.cpp only by the real-file KL gate recorded
//! in `docs/GGUF.md`. Weights are untrained: assert on structure, never text.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    parse_gguf_header, write_gguf_install_streamed, GgufBuilder, MemoryRangeSource,
    GGUF_DEFAULT_MAX_HEADER_BYTES,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);
const LAYERS: usize = 4;
const HIDDEN: u64 = 128;
const HEAD_DIM: u64 = 256;
const HEADS: u64 = 2;
const KV_HEADS: u64 = 1;
const INTER: u64 = 256;
const VOCAB: usize = 256;
const STEPS: usize = 24;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("turbospark-gemma2-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn norm(base: f32, n: u64) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (base + (i % 16) as f32 / 16.0).to_le_bytes())
        .collect()
}

/// Raw Q8_0 weights with distinct rows (the stock fixture repeats one block).
fn q8_rows(rows: u64, scale: f32, seed: u32) -> Vec<u8> {
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

fn gemma2_gguf(attn_cap: f32) -> Vec<u8> {
    let mut b = GgufBuilder::new()
        .metadata_str("general.architecture", "gemma2")
        .metadata_u32("gemma2.block_count", LAYERS as u32)
        .metadata_u32("gemma2.context_length", 8192)
        .metadata_u32("gemma2.embedding_length", HIDDEN as u32)
        .metadata_u32("gemma2.feed_forward_length", INTER as u32)
        .metadata_u32("gemma2.attention.head_count", HEADS as u32)
        .metadata_u32("gemma2.attention.head_count_kv", KV_HEADS as u32)
        .metadata_u32("gemma2.attention.key_length", HEAD_DIM as u32)
        .metadata_u32("gemma2.attention.value_length", HEAD_DIM as u32)
        .metadata_u32("gemma2.attention.sliding_window", 4096)
        .metadata_f32("gemma2.rope.freq_base", 10_000.0)
        .metadata_f32("gemma2.attention.layer_norm_rms_epsilon", 1e-6)
        .metadata_f32("gemma2.attn_logit_softcapping", attn_cap)
        .metadata_f32("gemma2.final_logit_softcapping", 30.0)
        .tensor(
            "token_embd.weight",
            8,
            &[HIDDEN, VOCAB as u64],
            q8_rows(VOCAB as u64, 0.02, 1),
        );
    for l in 0..LAYERS {
        let t = |s: &str| format!("blk.{l}.{s}.weight");
        b = b
            .tensor(
                &t("attn_q"),
                8,
                &[HIDDEN, HEADS * HEAD_DIM],
                q8_rows(HEADS * HEAD_DIM, 0.05, 3 + l as u32),
            )
            .tensor(
                &t("attn_k"),
                8,
                &[HIDDEN, KV_HEADS * HEAD_DIM],
                q8_rows(KV_HEADS * HEAD_DIM, 0.05, 4 + l as u32),
            )
            .q8_0_tensor(&t("attn_v"), &[HIDDEN, KV_HEADS * HEAD_DIM], 5)
            .q8_0_tensor(&t("attn_output"), &[HEADS * HEAD_DIM, HIDDEN], 7)
            .q8_0_tensor(&t("ffn_gate"), &[HIDDEN, INTER], 8)
            .q8_0_tensor(&t("ffn_up"), &[HIDDEN, INTER], 9)
            .q8_0_tensor(&t("ffn_down"), &[INTER, HIDDEN], 10)
            .tensor(&t("attn_norm"), 0, &[HIDDEN], norm(1.0, HIDDEN))
            .tensor(&t("post_attention_norm"), 0, &[HIDDEN], norm(2.0, HIDDEN))
            .tensor(&t("ffn_norm"), 0, &[HIDDEN], norm(3.0, HIDDEN))
            .tensor(&t("post_ffw_norm"), 0, &[HIDDEN], norm(4.0, HIDDEN));
        // NO attn_q_norm / attn_k_norm: Gemma 2 has none.
    }
    b.tensor("output_norm.weight", 0, &[HIDDEN], norm(7.0, HIDDEN))
        .build()
        .0
}

fn install(attn_cap: f32) -> Result<(model_io::ArchConfig, PathBuf), String> {
    let bytes = gemma2_gguf(attn_cap);
    let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
    let dir = temp_dir();
    let arch = write_gguf_install_streamed(
        &dir,
        &header,
        &MemoryRangeSource::new(&bytes),
        "tiny-gemma2",
        |_| {},
    )
    .map_err(|e| e.to_string())?;
    Ok((arch, dir))
}

#[test]
fn gemma2_installs_with_alternating_layers_opens_and_keeps_logits_under_the_final_cap() {
    let (arch, dir) = install(50.0).expect("a Gemma 2 install writes");
    assert_eq!(arch.family, model_io::ModelFamily::Gemma2);
    // Even layers slide, odd layers are global.
    assert_eq!(arch.full_attention_layer_mask, vec![0, 1, 0, 1]);
    assert_eq!(arch.final_logit_softcap, 30.0);
    assert_eq!(arch.attention_scale, 0.0625);
    assert!(arch.rope_scaling == model_io::RopeScalingConfig::NONE);

    let mut runner = RealForwardRunner::open(&dir, arch).expect("gemma2 install opens");
    runner.reset();
    for position in 0..STEPS {
        let token = ((position * 5 + 1) % VOCAB) as i32;
        let mut head = vec![f16::from_f32(0.0); VOCAB];
        runner.produce(token, position, &mut head).expect("produce");
        for v in &head {
            let v = v.to_f32();
            assert!(v.is_finite(), "non-finite logit at {position}");
            // The final cap is `30 * tanh(x / 30)`, so |logit| <= 30. A flow
            // that skipped it would exceed this on this fixture.
            assert!(
                v.abs() <= 30.0 + 1e-2,
                "logit {v} escapes the final cap at {position}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_gemma2_file_with_another_attention_cap_is_refused_not_decoded() {
    let err = install(25.0).expect_err("a different attn cap must refuse");
    assert!(err.contains("attn_logit_softcapping"), "{err}");
}
