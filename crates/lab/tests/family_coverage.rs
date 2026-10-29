//! Every `ModelFamily` gets a coverage answer, and every bench either
//! names its cases or says why not. Synthetic architectures, no installs:
//! this runs offline in a second.
#![cfg(target_os = "macos")]

use model_io::{ArchConfig, ModelFamily};
use turbospark_lab::families::{coverage, facts};
use turbospark_lab::kernel::{int4_matmul, Subject};

/// A small architecture of each family's shape. Families without a tiny
/// builder borrow the closest one and take its family tag.
fn arch_for(f: ModelFamily) -> ArchConfig {
    use repack as r;
    let mut a = match f {
        ModelFamily::Gemma4 => r::tiny_gemma4_arch(512, 4),
        ModelFamily::QwenGdnMoe => r::tiny_qwen_gdn_moe_arch(512, 4, 8),
        ModelFamily::QwenGdnDense | ModelFamily::DeepseekV4Flash => {
            r::tiny_qwen_gdn_dense_arch(512, 4)
        }
        ModelFamily::Llama | ModelFamily::Qwen3Dense | ModelFamily::Qwen2Dense => {
            r::tiny_dense_llama_arch(512, 4)
        }
        ModelFamily::Qwen3Moe | ModelFamily::MiniMaxM2 | ModelFamily::GptOss => {
            r::tiny_llama_arch(512, 4, 8)
        }
        ModelFamily::Qwen3Vl => r::tiny_qwen3_vl_arch(512, 4),
        ModelFamily::MuseGlimmer => r::tiny_muse_glimmer_arch(512, 4),
        ModelFamily::Qwen4Exp => r::tiny_qwen4_exp_arch(512, 4),
        ModelFamily::Spark25 => r::tiny_spark_arch(512, 4),
        ModelFamily::Deepseek2 => r::tiny_deepseek2_arch(512, 4),
        ModelFamily::Flint => r::tiny_flint_arch(512),
    };
    a.family = f;
    a
}

fn subject(arch: ArchConfig) -> Subject {
    Subject {
        dir: std::env::temp_dir().join("tslab-family-coverage-no-install"),
        family: arch.family.as_str().to_string(),
        arch: Some(arch),
        audio_config: None,
        inventory: Err("synthetic architecture, no resident index".into()),
    }
}

#[test]
fn every_family_has_coverage_and_matmul_shapes() {
    for f in ModelFamily::ALL {
        let s = subject(arch_for(f));
        let c = coverage(&s);
        assert_eq!(c.family, f.as_str());
        assert!(c.has("text"), "{f:?}");
        assert!(
            c.has(&format!("flow:{}", facts(f).flow)),
            "{f:?}: {:?}",
            c.tags
        );
        assert_eq!(
            c.benches["int4_matmul"]["runs"], true,
            "{f:?}: {:?}",
            c.benches
        );
        assert!(c.benches.contains_key("attn_decode"), "{f:?}");
        // Without an index, shapes come from the config and say so.
        assert!(
            c.notes
                .iter()
                .any(|n| n.contains("fall back to ArchConfig")),
            "{f:?}"
        );
        let moe = s.arch.as_ref().unwrap().num_experts > 0;
        assert_eq!(c.has("moe"), moe, "{f:?}");
        if !moe {
            assert!(c.has("dense"));
        }
    }
}

/// Mirrors the runtime's decode-kernel match in
/// `crates/runtime/src/real_forward_types.rs`: the 2026-10-29 flip covers
/// dense Llama but not Mixtral (same family string) or Qwen3Moe.
#[test]
fn decode_attention_follows_the_runtime_default() {
    let decode = |a: ArchConfig| coverage(&subject(a)).facts.unwrap().decode_attention;
    assert_eq!(decode(arch_for(ModelFamily::Llama)), "gqa");
    let mut mixtral = repack::tiny_llama_arch(512, 4, 8);
    mixtral.family = ModelFamily::Llama;
    assert_eq!(decode(mixtral), "barrier");
    assert_eq!(decode(arch_for(ModelFamily::Qwen3Moe)), "barrier");
    assert_eq!(decode(arch_for(ModelFamily::MiniMaxM2)), "barrier");
    for f in [
        ModelFamily::QwenGdnDense,
        ModelFamily::Qwen2Dense,
        ModelFamily::Qwen3Dense,
        ModelFamily::Qwen3Vl,
        ModelFamily::Spark25,
    ] {
        assert_eq!(decode(arch_for(f)), "gqa", "{f:?}");
    }
}

#[test]
fn deepseek2_gets_latent_projections_not_generic_attention() {
    let a = arch_for(ModelFamily::Deepseek2);
    assert!(a.has_mla_layers());
    let shapes = int4_matmul::shapes(&a);
    let m = &a.mla;
    let kv_a = (
        (m.kv_lora_rank + m.rope_head_dim) as usize,
        a.hidden_size as usize,
    );
    let kv_b = (
        (a.num_heads * (m.nope_head_dim + m.v_head_dim)) as usize,
        m.kv_lora_rank as usize,
    );
    let present = |s: (usize, usize)| shapes.contains(&s) || s.1 % 64 != 0;
    assert!(present(kv_a), "{shapes:?} lacks kv_a {kv_a:?}");
    assert!(present(kv_b), "{shapes:?} lacks kv_b {kv_b:?}");
    let c = coverage(&subject(a));
    assert_eq!(c.benches["attn_decode"]["runs"], false, "{:?}", c.benches);
    assert!(c.benches["attn_decode"]["why"]
        .as_str()
        .unwrap()
        .contains("MLA"));
}

#[test]
fn drafters_and_chunking_are_qwen_only() {
    for f in ModelFamily::ALL {
        let fa = facts(f);
        let qwen = matches!(f, ModelFamily::QwenGdnDense | ModelFamily::QwenGdnMoe);
        assert_eq!(!fa.drafters.is_empty(), qwen, "{f:?}");
        if matches!(
            f,
            ModelFamily::Spark25 | ModelFamily::Deepseek2 | ModelFamily::DeepseekV4Flash
        ) {
            assert!(!fa.chunked_prefill.possible(), "{f:?}");
        }
    }
}
