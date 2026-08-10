//! The admission gate for `arch_registry`'s planned rows (ROADMAP Phase M
//! Stage 1): every key is re-read off the real published file it was taken
//! from.
//!
//! A registry of architecture strings is exactly the kind of table that
//! rots into folklore, because a wrong row costs nothing until someone
//! points a real checkpoint at it and gets "not in this port's registry"
//! for a model the port would in fact have recognized. `gguf_names.rs`
//! states the same rule for its name table; this is that rule made runnable.
//!
//! Cost is a header per row -- a few MB off files of 4 to 700 GB -- not a
//! download, and about a second each. `#[ignore]`d and opt-in like every
//! other `*_network` test.
//!
//! ```sh
//! cargo test -p turbospark-repack --test arch_registry_network --release -- --ignored --nocapture
//! ```

use turbospark_repack::{
    fetch_gguf_header, gguf_arch_support, planned_gguf_architectures, ArchSupport, HttpRangeSource,
};

/// Mixtral, which is the reason ROADMAP Phase M2 does the `llama`
/// architecture MoE-first: it reports the SAME architecture string as a
/// dense Llama 3.1 and expresses its experts through `expert_count`. If
/// this ever stops being true, the plan that rests on it is wrong.
const MIXTRAL_8X7B: &str = "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_0.gguf";

fn architecture_of(url: &str) -> String {
    let source = HttpRangeSource::new(url);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    header
        .architecture()
        .expect("general.architecture")
        .to_string()
}

#[test]
#[ignore = "network: reads a header per planned architecture"]
fn every_planned_row_still_matches_its_witness() {
    for (key, planned) in planned_gguf_architectures() {
        let found = architecture_of(planned.witness);
        println!("{key:<12} <- {found:<12} {}", planned.witness);
        assert_eq!(
            found, key,
            "registry row {key:?} disagrees with its witness {}",
            planned.witness
        );
    }
}

#[test]
#[ignore = "network: reads one header off a 26 GB remote checkpoint"]
fn mixtral_is_the_llama_architecture_with_experts() {
    let source = HttpRangeSource::new(MIXTRAL_8X7B);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    let arch = header.architecture().expect("general.architecture");
    assert_eq!(arch, "llama", "Mixtral is no longer the llama architecture");
    assert!(
        matches!(gguf_arch_support(arch), Some(ArchSupport::Supported(_))),
        "llama gained a decode flow in ROADMAP Phase M2"
    );

    // The MoE is in the metadata, not in the architecture string. Printed
    // rather than asserted on an exact count: the claim under test is that
    // an expert count EXISTS here and does not on a dense Llama.
    let experts = header
        .metadata
        .get("llama.expert_count")
        .and_then(|v| v.as_u64());
    println!("llama.expert_count = {experts:?}");
    assert!(
        experts.is_some_and(|n| n > 1),
        "a Mixtral header should carry an expert count"
    );
}
