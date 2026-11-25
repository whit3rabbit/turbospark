//! Gemma 3's GGUF intake, offline.
//!
//! The authoritative checks are in `gemma3_gguf_network.rs`, which reads the
//! real `ggml-org/gemma-3-1b-it-GGUF` and a public bf16 copy. These cover what
//! it cannot: the constants the file omits, every refusal, and the whole walk
//! on a small file.

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, family_for_architecture, gguf_architecture, map_gguf_name,
    orchestrate_gguf_checkpoint, parse_gguf_header, GgufBuilder, GgufConfigError, GgufMapping,
    MemoryRangeSource, ResidentEntrySpec, DTYPE_BF16, GGUF_DEFAULT_MAX_HEADER_BYTES,
};

const LAYERS: usize = 6;
const HIDDEN: u64 = 128;
const HEAD_DIM: u64 = 256;
const HEADS: u64 = 2;
const INTER: u64 = 256;

/// The keys the real 1B file publishes, shrunk. No `sliding_window_pattern`,
/// no `freq_base_swa`, no `rope.scaling.*` and no `output.weight`.
fn builder(eps: f32, key_length: u32) -> GgufBuilder {
    GgufBuilder::new()
        .metadata_str("general.architecture", "gemma3")
        .metadata_u32("gemma3.block_count", LAYERS as u32)
        .metadata_u32("gemma3.embedding_length", HIDDEN as u32)
        .metadata_u32("gemma3.feed_forward_length", INTER as u32)
        .metadata_u32("gemma3.attention.head_count", HEADS as u32)
        .metadata_u32("gemma3.attention.head_count_kv", 1)
        .metadata_u32("gemma3.attention.key_length", key_length)
        .metadata_u32("gemma3.attention.value_length", key_length)
        .metadata_u32("gemma3.attention.sliding_window", 512)
        .metadata_f32("gemma3.rope.freq_base", 1_000_000.0)
        .metadata_f32("gemma3.attention.layer_norm_rms_epsilon", eps)
        .q8_0_tensor("token_embd.weight", &[HIDDEN, 256], 1)
}

fn arch(b: GgufBuilder) -> Result<model_io::ArchConfig, GgufConfigError> {
    let (bytes, _) = b.build();
    let h = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
    arch_from_gguf(&h)
}

fn refused_key(b: GgufBuilder) -> String {
    match arch(b) {
        Err(GgufConfigError::BadValue { key, .. }) => key,
        other => panic!("expected BadValue, got {other:?}"),
    }
}

#[test]
fn the_architecture_string_round_trips() {
    assert_eq!(family_for_architecture("gemma3"), Some(ModelFamily::Gemma3));
    assert_eq!(gguf_architecture(ModelFamily::Gemma3), Some("gemma3"));
    assert_eq!(ModelFamily::parse("gemma3"), Some(ModelFamily::Gemma3));
    // Gemma 4 is a different string and a different family.
    assert_eq!(family_for_architecture("gemma4"), Some(ModelFamily::Gemma4));
}

#[test]
fn derives_the_constants_the_file_omits() {
    let a = arch(builder(1e-6, 256)).expect("derives");
    assert_eq!(a.family, ModelFamily::Gemma3);
    // 5 local : 1 global, global last, from the converter's constant.
    assert_eq!(a.full_attention_layer_mask, vec![0, 0, 0, 0, 0, 1]);
    // The file's `freq_base` is the GLOBAL base; the sliding layers' is the
    // 10000 no GGUF key carries. The generic path would hand them 1e6.
    assert_eq!(a.full_rope_theta, 1_000_000.0);
    assert_eq!(a.rope_theta, 10_000.0);
    assert_eq!(a.attention_scale, 0.0625);
    assert_eq!(a.partial_rotary_factor, 1.0);
    assert_eq!(a.final_logit_softcap, 0.0);
    assert!(a.tie_word_embeddings, "no output.weight in the file");
    assert!(a.ffn_sandwich_norms && !a.router_scaled && !a.attention_k_eq_v);
    assert_eq!((a.num_experts, a.sliding_window), (0, 512));
    assert_eq!((a.head_dim, a.full_head_dim), (256, 256));
}

#[test]
fn a_file_that_does_not_match_the_one_shape_read_off_bytes_is_refused() {
    // The flow hard-codes epsilon 1e-6.
    assert_eq!(
        refused_key(builder(1e-5, 256)),
        "gemma3.attention.layer_norm_rms_epsilon"
    );
    // 27B: key_length 128 and a different attention scale.
    assert_eq!(
        refused_key(builder(1e-6, 128)),
        "gemma3.attention.key_length"
    );
    // Another scaling mechanism is not Gemma 3's. (A YaRN file never gets this
    // far: the generic reader refuses it first for a missing original context.)
    let longrope = builder(1e-6, 256)
        .metadata_str("gemma3.rope.scaling.type", "longrope")
        .metadata_f32("gemma3.rope.scaling.factor", 8.0);
    assert_eq!(refused_key(longrope), "gemma3.rope.scaling.type");
    // A linear factor that scales nothing (or shrinks) is not a scaling.
    let unity = builder(1e-6, 256)
        .metadata_str("gemma3.rope.scaling.type", "linear")
        .metadata_f32("gemma3.rope.scaling.factor", 1.0);
    assert_eq!(refused_key(unity), "gemma3.rope.scaling.factor");
    // A factor with no type is ambiguous.
    let factor_only = builder(1e-6, 256).metadata_f32("gemma3.rope.scaling.factor", 8.0);
    assert_eq!(refused_key(factor_only), "gemma3.rope.scaling.factor");
    // A different local:global ratio.
    let other_period =
        builder(1e-6, 256).metadata_u32("gemma3.attention.sliding_window_pattern", 4);
    assert_eq!(
        refused_key(other_period),
        "gemma3.attention.sliding_window_pattern"
    );
    // MoE is another architecture.
    let moe = builder(1e-6, 256).metadata_u32("gemma3.expert_count", 8);
    assert_eq!(refused_key(moe), "gemma3.expert_count");
    // An untied head is another architecture.
    let untied = builder(1e-6, 256).q8_0_tensor("output.weight", &[HIDDEN, 256], 2);
    assert_eq!(refused_key(untied), "gemma3.output.weight");
    // A partial rotary width is another architecture.
    let partial = builder(1e-6, 256).metadata_u32("gemma3.rope.dimension_count", 64);
    assert_eq!(refused_key(partial), "gemma3.rope.dimension_count");
}

/// 4B and up: linear scaling on the global layers, carried as
/// `rope_scaling.linear_factor` (and NOT as YaRN's `factor`).
#[test]
fn linear_rope_scaling_is_read_for_the_global_layers() {
    let a = arch(
        builder(1e-6, 256)
            .metadata_str("gemma3.rope.scaling.type", "linear")
            .metadata_f32("gemma3.rope.scaling.factor", 8.0),
    )
    .expect("4B-style scaling derives");
    assert_eq!(a.rope_scaling.linear_factor, 8.0);
    assert!(a.rope_scaling.is_linear());
    assert!(!a.rope_scaling.is_active() && !a.rope_scaling.is_llama3());
    // The unscaled 1B keeps none.
    assert!(!arch(builder(1e-6, 256)).unwrap().rope_scaling.is_linear());
}

#[test]
fn a_swa_base_in_the_file_is_believed_over_the_constant() {
    let a = arch(builder(1e-6, 256).metadata_f32("gemma3.rope.freq_base_swa", 20_000.0))
        .expect("derives");
    assert_eq!(a.rope_theta, 20_000.0);
}

#[test]
fn every_tensor_the_real_file_carries_maps_and_nothing_else_does() {
    let m = |n: &str| map_gguf_name(n, ModelFamily::Gemma3);
    let resident = |n: &str| match m(n) {
        Ok(GgufMapping::Resident(c)) => c,
        other => panic!("{n}: {other:?}"),
    };
    let p = "language_model.model.layers.7.";
    for (g, c) in [
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
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
    // Gemma 4's extras are not Gemma 3's: refused, never borrowed.
    for unknown in [
        "blk.0.layer_output_scale.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_up_exps.weight",
        "blk.0.pre_ffw_norm_2.weight",
        "blk.0.attn_gate.weight",
    ] {
        assert!(m(unknown).is_err(), "{unknown}");
    }
}

// ---------------------------------------------------------------------------
// The whole walk on a small file: the unbaked norms and the untouched rows.
// ---------------------------------------------------------------------------

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `base` plus a step on a 1/16 grid, which BF16 represents exactly up to 8,
/// so a missed or doubled `-1` shows in every element and rounding cannot.
fn norm_at(base: f32, n: u64, i: u64) -> f32 {
    base + (i % n.min(16)) as f32 / 16.0
}

fn norm_vec(base: f32, n: u64) -> Vec<u8> {
    f32_bytes(&(0..n).map(|i| norm_at(base, n, i)).collect::<Vec<_>>())
}

fn gemma3_file() -> Vec<u8> {
    let mut b = builder(1e-6, 256);
    for l in 0..LAYERS {
        let t = |s: &str| format!("blk.{l}.{s}.weight");
        b = b
            .q8_0_tensor(&t("attn_q"), &[HIDDEN, HEADS * HEAD_DIM], 3)
            .q8_0_tensor(&t("attn_k"), &[HIDDEN, HEAD_DIM], 4)
            .q8_0_tensor(&t("attn_v"), &[HIDDEN, HEAD_DIM], 5)
            .q8_0_tensor(&t("attn_output"), &[HEADS * HEAD_DIM, HIDDEN], 7)
            .q8_0_tensor(&t("ffn_gate"), &[HIDDEN, INTER], 8)
            .q8_0_tensor(&t("ffn_up"), &[HIDDEN, INTER], 9)
            .q8_0_tensor(&t("ffn_down"), &[INTER, HIDDEN], 10)
            // Converted values: the weight plus 1, in every norm.
            .tensor(&t("attn_norm"), 0, &[HIDDEN], norm_vec(1.0, HIDDEN))
            .tensor(
                &t("post_attention_norm"),
                0,
                &[HIDDEN],
                norm_vec(2.0, HIDDEN),
            )
            .tensor(&t("ffn_norm"), 0, &[HIDDEN], norm_vec(3.0, HIDDEN))
            .tensor(&t("post_ffw_norm"), 0, &[HIDDEN], norm_vec(4.0, HIDDEN))
            .tensor(&t("attn_q_norm"), 0, &[HEAD_DIM], norm_vec(5.0, HEAD_DIM))
            .tensor(&t("attn_k_norm"), 0, &[HEAD_DIM], norm_vec(6.0, HEAD_DIM));
    }
    b.tensor("output_norm.weight", 0, &[HIDDEN], norm_vec(7.0, HIDDEN))
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
fn the_walk_installs_every_norm_unbaked_including_qk_and_the_final_one() {
    let out = walk(&gemma3_file()).expect("walks");
    assert_eq!(out.arch.family, ModelFamily::Gemma3);
    let p = "language_model.model.layers.3.";
    for (resident, base, n) in [
        ("input_layernorm", 1.0f32, HIDDEN),
        ("post_attention_layernorm", 2.0, HIDDEN),
        ("pre_feedforward_layernorm", 3.0, HIDDEN),
        ("post_feedforward_layernorm", 4.0, HIDDEN),
        ("self_attn.q_norm", 5.0, HEAD_DIM),
        ("self_attn.k_norm", 6.0, HEAD_DIM),
    ] {
        let got = bf16s(raw(&out, &format!("{p}{resident}.weight")));
        let want: Vec<f32> = (0..n).map(|i| norm_at(base, n, i) - 1.0).collect();
        assert_eq!(got, want, "{resident}");
    }
    // Unlike Muse Glimmer the FINAL norm carries the +1 as well.
    let got = bf16s(raw(&out, "language_model.model.norm.weight"));
    let want: Vec<f32> = (0..HIDDEN).map(|i| norm_at(7.0, HIDDEN, i) - 1.0).collect();
    assert_eq!(got, want);
}
