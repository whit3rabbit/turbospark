//! The `gdn_delta` bench names its cases from the install's own
//! gated-DeltaNet geometry, and says why it cannot run where there is none.
//! Synthetic architectures, no installs and no GPU work.
#![cfg(target_os = "macos")]

use model_io::ArchConfig;
use turbospark_lab::kernel::{gdn_delta, BenchOpts, KernelBench, Subject};

fn subject(arch: ArchConfig) -> Subject {
    Subject {
        dir: std::env::temp_dir().join("tslab-gdn-delta-no-install"),
        family: arch.family.as_str().to_string(),
        arch: Some(arch),
        audio_config: None,
        inventory: Err("synthetic architecture, no resident index".into()),
    }
}

#[test]
fn a_gdn_install_gets_one_case_with_its_geometry_and_layer_count() {
    let arch = repack::tiny_qwen_gdn_dense_arch(512, 4);
    let layers = arch
        .full_attention_layer_mask
        .iter()
        .filter(|&&m| m == 2)
        .count();
    assert!(layers > 0, "the fixture must have GDN layers");
    let cases = gdn_delta::GdnDelta
        .cases(&subject(arch.clone()), &BenchOpts::default())
        .expect("gdn install");
    assert_eq!(cases.len(), 1);
    let p = &cases[0].params;
    assert_eq!(p["count"], layers);
    assert_eq!(p["num_v_heads"], arch.linear_attention.num_v_heads);
    assert_eq!(p["key_head_dim"], arch.linear_attention.key_head_dim);
    assert_eq!(gdn_delta::layer_count(&arch), layers);
}

#[test]
fn an_install_without_gdn_layers_says_so() {
    let arch = repack::tiny_dense_llama_arch(512, 4);
    let err = gdn_delta::GdnDelta
        .cases(&subject(arch), &BenchOpts::default())
        .expect_err("llama has no gated-DeltaNet layer");
    assert!(err.contains("gated-DeltaNet"), "{err}");
}
