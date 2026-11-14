//! Muse Glimmer's GGUF intake, offline.
//!
//! The authoritative checks are in `muse_gguf_network.rs`, which reads the
//! real `ggml-org/Muse-Glimmer-30B-GGUF` and the MLX and official BF16
//! conversions. These cover what it cannot: the name table end to end, the
//! derivation on a small header with the real file's keys, and the refusals.

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, family_for_architecture, gguf_architecture, map_gguf_name, parse_gguf_header,
    GgufBuilder, GgufConfigError, GgufMapping, GgufValue, GGUF_DEFAULT_MAX_HEADER_BYTES,
};

const LAYERS: usize = 8;

/// A header carrying the keys the real file publishes, at a size that keeps
/// the tensor data tiny. `pattern` is the bool `sliding_window_pattern`.
fn builder(eps: f32, logit_scale: f32, pattern: Vec<bool>) -> GgufBuilder {
    GgufBuilder::new()
        .metadata_str("general.architecture", "muse-glimmer")
        .metadata_u32("muse-glimmer.block_count", LAYERS as u32)
        .metadata_u32("muse-glimmer.embedding_length", 128)
        .metadata_u32("muse-glimmer.feed_forward_length", 256)
        .metadata_u32("muse-glimmer.attention.head_count", 4)
        .metadata_u32("muse-glimmer.attention.head_count_kv", 2)
        .metadata_u32("muse-glimmer.attention.key_length", 128)
        .metadata_u32("muse-glimmer.attention.value_length", 128)
        .metadata_u32("muse-glimmer.attention.sliding_window", 2048)
        .metadata_f32("muse-glimmer.rope.freq_base", 500_000.0)
        .metadata_f32("muse-glimmer.attention.layer_norm_rms_epsilon", eps)
        .metadata_f32("muse-glimmer.final_logit_softcapping", 20.0)
        .metadata_f32("muse-glimmer.logit_scale", logit_scale)
        .metadata(
            "muse-glimmer.attention.sliding_window_pattern",
            GgufValue::Array(pattern.into_iter().map(GgufValue::Bool).collect()),
        )
        .q8_0_tensor("token_embd.weight", &[128, 256], 1)
        .q8_0_tensor("output.weight", &[128, 256], 2)
}

/// `[sliding, sliding, sliding, full]` repeated, as the real file writes it.
fn real_pattern() -> Vec<bool> {
    (0..LAYERS).map(|l| l % 4 != 3).collect()
}

fn arch(b: GgufBuilder) -> Result<model_io::ArchConfig, GgufConfigError> {
    let (bytes, _) = b.build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    arch_from_gguf(&h)
}

#[test]
fn the_architecture_string_is_hyphenated_and_round_trips() {
    assert_eq!(
        family_for_architecture("muse-glimmer"),
        Some(ModelFamily::MuseGlimmer)
    );
    // The HF spelling is NOT a GGUF architecture.
    assert_eq!(family_for_architecture("muse_glimmer"), None);
    assert_eq!(
        gguf_architecture(ModelFamily::MuseGlimmer),
        Some("muse-glimmer")
    );
}

#[test]
fn derives_the_muse_fields_the_generic_path_would_get_wrong() {
    let a = arch(builder(1e-5, 0.196_116_13, real_pattern())).expect("derives");
    assert_eq!(a.family, ModelFamily::MuseGlimmer);
    assert_eq!(a.num_layers, LAYERS as i64);
    // true = slides, so the full layers are 3 and 7: the polarity that
    // silently swaps every layer's attention kind if it is backwards.
    assert_eq!(a.full_attention_layer_mask, vec![0, 0, 0, 1, 0, 0, 0, 1]);
    // The file's one `freq_base` is the SLIDING layers' base; the full layers
    // apply no rotary embedding at all.
    assert_eq!(a.rope_theta, 500_000.0);
    assert_eq!(a.full_rope_theta, 0.0);
    assert!(!a.tie_word_embeddings, "the file ships output.weight");
    assert_eq!(a.final_logit_softcap, 20.0);
    assert_eq!(a.sliding_window, 2048);
    assert_eq!((a.head_dim, a.full_head_dim), (128, 128));
    assert_eq!((a.num_kv_heads, a.num_full_kv_heads), (2, 2));
}

#[test]
fn a_conversion_whose_scalars_disagree_with_the_flow_is_refused() {
    for (eps, scale, key) in [
        (1e-6f32, 0.196_116_13f32, "attention.layer_norm_rms_epsilon"),
        (1e-5, 1.0, "logit_scale"),
    ] {
        match arch(builder(eps, scale, real_pattern())) {
            Err(GgufConfigError::BadValue { key: k, .. }) => {
                assert_eq!(k, format!("muse-glimmer.{key}"))
            }
            other => panic!("expected BadValue for {key}, got {other:?}"),
        }
    }
}

#[test]
fn a_pattern_that_does_not_cover_every_layer_is_refused() {
    let short = real_pattern()[..LAYERS - 1].to_vec();
    assert!(matches!(
        arch(builder(1e-5, 0.196_116_13, short)),
        Err(GgufConfigError::BadValue { .. })
    ));
}

#[test]
fn every_tensor_the_real_file_carries_maps_and_nothing_else_does() {
    let m = |n: &str| map_gguf_name(n, ModelFamily::MuseGlimmer);
    let resident = |n: &str| match m(n) {
        Ok(GgufMapping::Resident(c)) => c,
        other => panic!("{n}: {other:?}"),
    };
    let p = "language_model.model.layers.7.";
    for (g, c) in [
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_gate", "self_attn.gate_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("ffn_norm", "pre_feedforward_layernorm"),
        ("post_ffw_norm", "post_feedforward_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
    ] {
        assert_eq!(
            resident(&format!("blk.7.{g}.weight")),
            format!("{p}{c}.weight")
        );
    }
    assert_eq!(
        resident("token_embd.weight"),
        "language_model.model.embed_tokens.weight"
    );
    assert_eq!(
        resident("output_norm.weight"),
        "language_model.model.norm.weight"
    );
    assert_eq!(resident("output.weight"), "language_model.lm_head.weight");
    assert_eq!(
        m("blk.0.attn_q_norm.weight"),
        Ok(GgufMapping::MuseQkNorm { is_q: true })
    );
    assert_eq!(
        m("blk.0.attn_k_norm.weight"),
        Ok(GgufMapping::MuseQkNorm { is_q: false })
    );
    // 14 per-layer suffixes and no others: a fused qkv is another family's.
    for unknown in ["blk.0.attn_qkv.weight", "blk.0.ffn_gate_exps.weight"] {
        assert!(m(unknown).is_err(), "{unknown}");
    }
    // The same name under Gemma 4 is a different mapping, not Muse's.
    assert!(map_gguf_name("blk.0.attn_gate.weight", ModelFamily::Gemma4).is_err());
}

// ---------------------------------------------------------------------------
// The whole walk on a small file: names, centered norms, dropped q/k norms,
// and the rotary row order.
// ---------------------------------------------------------------------------

use turbospark_repack::{
    orchestrate_gguf_checkpoint, MemoryRangeSource, ResidentEntrySpec, DTYPE_BF16,
};

const HIDDEN: u64 = 128;
const HEAD_DIM: u64 = 128;
const HEADS: u64 = 4;
const KV_HEADS: u64 = 2;
const INTER: u64 = 256;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The norm value at index `i`: `base` plus a step on a 1/16 grid, which BF16
/// represents exactly up to 8, so a missed or doubled `-1` shows in every
/// element and rounding cannot.
fn norm_at(base: f32, i: u64) -> f32 {
    base + (i % 16) as f32 / 16.0
}

fn norm_vec(base: f32) -> Vec<u8> {
    f32_bytes(&(0..HIDDEN).map(|i| norm_at(base, i)).collect::<Vec<_>>())
}

fn muse_file() -> Vec<u8> {
    let mut b = builder(1e-5, 0.196_116_13, real_pattern());
    for l in 0..LAYERS {
        let t = |s: &str| format!("blk.{l}.{s}.weight");
        b = b
            .q8_0_tensor(&t("attn_q"), &[HIDDEN, HEADS * HEAD_DIM], 3)
            .q8_0_tensor(&t("attn_k"), &[HIDDEN, KV_HEADS * HEAD_DIM], 4)
            .q8_0_tensor(&t("attn_v"), &[HIDDEN, KV_HEADS * HEAD_DIM], 5)
            .q8_0_tensor(&t("attn_gate"), &[HIDDEN, HEADS * HEAD_DIM], 6)
            .q8_0_tensor(&t("attn_output"), &[HEADS * HEAD_DIM, HIDDEN], 7)
            .q8_0_tensor(&t("ffn_gate"), &[HIDDEN, INTER], 8)
            .q8_0_tensor(&t("ffn_up"), &[HIDDEN, INTER], 9)
            .q8_0_tensor(&t("ffn_down"), &[INTER, HIDDEN], 10)
            // Converted values: the centered weight plus 1.
            .tensor(&t("attn_norm"), 0, &[HIDDEN], norm_vec(1.0))
            .tensor(&t("post_attention_norm"), 0, &[HIDDEN], norm_vec(2.0))
            .tensor(&t("ffn_norm"), 0, &[HIDDEN], norm_vec(3.0))
            .tensor(&t("post_ffw_norm"), 0, &[HIDDEN], norm_vec(4.0))
            // The converter's synthesized pair.
            .tensor(
                &t("attn_q_norm"),
                0,
                &[HEAD_DIM],
                f32_bytes(&vec![3.87f32; HEAD_DIM as usize]),
            )
            .tensor(
                &t("attn_k_norm"),
                0,
                &[HEAD_DIM],
                f32_bytes(&vec![1.0f32; HEAD_DIM as usize]),
            );
    }
    b.tensor("output_norm.weight", 0, &[HIDDEN], norm_vec(0.5))
        .build()
        .0
}

fn walk(
    bytes: &[u8],
) -> Result<turbospark_repack::GgufRepackOutput, turbospark_repack::GgufRepackError> {
    let h = parse_gguf_header(bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    orchestrate_gguf_checkpoint(&h, &MemoryRangeSource::new(bytes))
}

fn raw<'a>(out: &'a turbospark_repack::GgufRepackOutput, name: &str) -> &'a [u8] {
    out.resident
        .iter()
        .find_map(|e| match e {
            ResidentEntrySpec::Raw(r) if r.name == name => {
                assert_eq!(r.dtype, DTYPE_BF16, "{name}");
                Some(r.bytes.as_slice())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("{name} not a raw resident entry"))
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

#[test]
fn the_walk_installs_centered_norms_and_drops_the_synthesized_pair() {
    let out = walk(&muse_file()).expect("walks");
    assert_eq!(out.arch.family, ModelFamily::MuseGlimmer);
    let p = "language_model.model.layers.3.";
    for (resident, base) in [
        ("input_layernorm", 1.0f32),
        ("post_attention_layernorm", 2.0),
        ("pre_feedforward_layernorm", 3.0),
        ("post_feedforward_layernorm", 4.0),
    ] {
        let got = bf16s(raw(&out, &format!("{p}{resident}.weight")));
        // The file holds the converted value, so the install holds it minus one.
        let want: Vec<f32> = (0..HIDDEN).map(|i| norm_at(base, i) - 1.0).collect();
        assert_eq!(got, want, "{resident}");
    }
    // The final norm passes through unshifted.
    let got = bf16s(raw(&out, "language_model.model.norm.weight"));
    let want: Vec<f32> = (0..HIDDEN).map(|i| norm_at(0.5, i)).collect();
    assert_eq!(got, want);
    // No q/k norm tensor reaches the install, and each is reported.
    assert!(out.resident.iter().all(|e| match e {
        ResidentEntrySpec::Raw(r) => !r.name.contains("q_norm") && !r.name.contains("k_norm"),
        _ => true,
    }));
    assert_eq!(
        out.ignored
            .iter()
            .filter(|s| s.contains("synthesized by the converter"))
            .count(),
        2 * LAYERS
    );
}

#[test]
fn a_different_q_scale_refuses_the_whole_install() {
    // Rebuild with one layer's q norm at another factor.
    let mut b = builder(1e-5, 0.196_116_13, real_pattern());
    for l in 0..LAYERS {
        let t = |s: &str| format!("blk.{l}.{s}.weight");
        let q = if l == 5 { 4.0f32 } else { 3.87 };
        b = b
            .q8_0_tensor(&t("attn_q"), &[HIDDEN, HEADS * HEAD_DIM], 3)
            .q8_0_tensor(&t("attn_k"), &[HIDDEN, KV_HEADS * HEAD_DIM], 4)
            .q8_0_tensor(&t("attn_v"), &[HIDDEN, KV_HEADS * HEAD_DIM], 5)
            .q8_0_tensor(&t("attn_gate"), &[HIDDEN, HEADS * HEAD_DIM], 6)
            .q8_0_tensor(&t("attn_output"), &[HEADS * HEAD_DIM, HIDDEN], 7)
            .q8_0_tensor(&t("ffn_gate"), &[HIDDEN, INTER], 8)
            .q8_0_tensor(&t("ffn_up"), &[HIDDEN, INTER], 9)
            .q8_0_tensor(&t("ffn_down"), &[INTER, HIDDEN], 10)
            .tensor(&t("attn_norm"), 0, &[HIDDEN], norm_vec(1.0))
            .tensor(&t("post_attention_norm"), 0, &[HIDDEN], norm_vec(1.0))
            .tensor(&t("ffn_norm"), 0, &[HIDDEN], norm_vec(1.0))
            .tensor(&t("post_ffw_norm"), 0, &[HIDDEN], norm_vec(1.0))
            .tensor(&t("attn_q_norm"), 0, &[HEAD_DIM], f32_bytes(&vec![q; 128]))
            .tensor(
                &t("attn_k_norm"),
                0,
                &[HEAD_DIM],
                f32_bytes(&vec![1.0; 128]),
            );
    }
    let bytes = b
        .tensor("output_norm.weight", 0, &[HIDDEN], norm_vec(0.5))
        .build()
        .0;
    match walk(&bytes) {
        Err(turbospark_repack::GgufRepackError::ShapeMismatch { tensor, detail }) => {
            assert!(tensor.starts_with("blk.5.attn_q_norm"), "{tensor}");
            assert!(detail.contains("constant 3.87"), "{detail}");
        }
        Ok(_) => panic!("a wrong q scale must refuse the install"),
        Err(e) => panic!("unexpected error: {e:?}"),
    }
}
