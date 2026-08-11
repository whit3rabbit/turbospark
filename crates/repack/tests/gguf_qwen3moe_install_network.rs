//! The real bytes for `qwen3moe`: streams the published
//! `Qwen/Qwen3-30B-A3B-GGUF` Q4_K_M checkpoint into a `.gturbo` install.
//!
//! ```sh
//! TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
//!   cargo test -p turbospark-repack --test gguf_qwen3moe_install_network --release -- --ignored --nocapture
//! ```
//!
//! Same cost properties as its siblings: the 17.3 GB checkpoint is NEVER
//! materialized locally, because `write_gguf_install_streamed` reads it a
//! layer at a time through `HttpRangeSource`, so the only disk this needs is
//! the install.
//!
//! **This is the checkpoint the ROADMAP Phase M granularity finding asked
//! for**, and the contrast with the Mixtral install beside it is the point:
//! same layer graph, same decode flow, 128 experts of 2.5 MiB instead of 8 of
//! 108.9 MiB. Its slot cache at 16 slots is 1.90 GiB against Mixtral's 54.5,
//! which is what makes it the first `llama`-flow checkpoint whose memory
//! oracle and quality gate are worth running at all (AGENTS.md Gotcha 36).
//!
//! No new kernels: Q4_K and Q6_K both already have a resident GEMV, an
//! embedding lookup and the routed halves this file needs. That was asserted
//! off the header before this test was written
//! (`gguf_checkpoint_network.rs::qwen3moe_maps_every_name_and_derives_its_baseline`).

use std::path::PathBuf;

use turbospark_repack::{fetch_gguf_header, write_gguf_install_streamed, HttpRangeSource};

const QWEN3_30B_A3B_Q4_K_M: &str =
    "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf";
const MODEL_ID: &str = "Qwen/Qwen3-30B-A3B-GGUF";

/// The GGUF carries its tokenizer as llama.cpp metadata; this port loads an
/// HF `tokenizer.json`. Take the sidecars from the checkpoint the GGUF was
/// converted from, as every sibling does.
const SIDECAR_BASE: &str = "https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/main";

fn get(url: &str) -> Vec<u8> {
    let response = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .expect("client")
        .get(url)
        .send()
        .unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(
        response.status().is_success(),
        "GET {url}: HTTP {}",
        response.status()
    );
    response.bytes().expect("body").to_vec()
}

fn install_dir() -> PathBuf {
    let dir = match std::env::var_os("TURBOSPARK_QWEN3MOE_INSTALL_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("turbospark-qwen3moe-{}", std::process::id())),
    };
    for pinned in [
        "TURBOSPARK_GEMMA4_INSTALL_DIR",
        "TURBOSPARK_QWEN36_INSTALL_DIR",
    ] {
        assert_ne!(
            std::env::var_os(pinned).map(PathBuf::from),
            Some(dir.clone()),
            "refusing to overwrite {pinned}, which the standing gates measure against"
        );
    }
    std::fs::create_dir_all(&dir).expect("create install dir");
    dir
}

#[test]
#[ignore = "network: streams the real ~17 GB Qwen3-30B-A3B Q4_K_M GGUF and writes a ~18 GB install"]
fn repacks_the_real_qwen3_30b_a3b_q4_k_m_gguf() {
    let source = HttpRangeSource::new(QWEN3_30B_A3B_Q4_K_M);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    assert_eq!(header.architecture(), Some("qwen3moe"));
    // The merged expert layout, as `gguf_names.rs` requires. A pre-merge
    // conversion carries one tensor per expert per role instead.
    assert!(
        header
            .tensors
            .keys()
            .any(|n| n.ends_with("ffn_down_exps.weight")),
        "this conversion predates the expert merge; the walk refuses it"
    );
    eprintln!(
        "header: {} tensors, alignment {}, data region starts at {}",
        header.tensors.len(),
        header.alignment,
        header.data_region_start
    );

    let dir = install_dir();
    eprintln!("installing to {}", dir.display());
    let arch = write_gguf_install_streamed(&dir, &header, &source, MODEL_ID, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed GGUF install");
    assert_eq!(arch, model_io::qwen3_30b_a3b());

    for name in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::write(dir.join(name), get(&format!("{SIDECAR_BASE}/{name}")))
            .expect("tokenizer sidecar");
    }

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates: q4_k and q6_k are both executable");
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    let tag = |ggml: u32| turbospark_repack::dtype_tag_for_ggml_type(ggml).expect("tag");
    let dtype = |name: &str| resident.entries[name].dtype;
    let l0 = "language_model.model.layers.0";
    assert_eq!(
        dtype("language_model.model.embed_tokens.weight"),
        tag(12),
        "the embedding table is Q4_K"
    );
    assert_eq!(
        dtype("language_model.lm_head.weight"),
        tag(14),
        "output.weight is Q6_K, and this model does NOT tie its head"
    );
    assert_eq!(
        dtype(&format!("{l0}.self_attn.q_proj.weight")),
        tag(12),
        "attn_q is Q4_K"
    );
    // THE TWO TENSORS THAT MAKE THIS NOT MIXTRAL. GGUF ships them F32 and the
    // runtime's `norm_view` reads BF16, so the transcode has to have narrowed
    // them; carrying them verbatim would fail at open with a byte-size error,
    // and mapping them to nothing would skip a normalization silently.
    for norm in ["q_norm", "k_norm"] {
        let entry = resident
            .entries
            .get(&format!("{l0}.self_attn.{norm}.weight"))
            .unwrap_or_else(|| panic!("{norm} is missing from the install"));
        assert_eq!(
            entry.dtype,
            turbospark_repack::DTYPE_BF16,
            "{norm} should be BF16 after the transcode"
        );
        assert_eq!(
            entry.size_bytes as i64,
            arch.full_head_dim * 2,
            "{norm} is a [head_dim] vector, not a matrix"
        );
    }
    // The F32 core is transcoded rather than carried (AGENTS.md Gotcha 29):
    // the router lands as INT8 affine, so nothing F32 survives.
    assert_eq!(
        dtype(&format!("{l0}.mlp.gate.weight")),
        5,
        "the router should have been INT8-transcoded"
    );

    let layout = model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("layout");
    assert_eq!(layout.num_layers, 48);
    assert_eq!(layout.experts_per_layer, 128);

    // The same per-LAYER mixture Mixtral has, on a different tensor split:
    // `ffn_down_exps` is Q4_K on half the layers and Q6_K on the other half.
    let down_types: std::collections::BTreeSet<String> = layout
        .layers
        .iter()
        .filter_map(|layer| layer.experts.first())
        .filter_map(|expert| expert.sub_tensors.get("down"))
        .map(|sub| sub.dtype.to_lowercase())
        .collect();
    eprintln!("routed `down` dtypes across layers: {down_types:?}");
    assert!(
        down_types.len() >= 2,
        "expected a per-layer mixture in ffn_down_exps, saw {down_types:?}"
    );

    // THE GRANULARITY CLAIM, MEASURED ON THE ARTIFACT rather than predicted
    // from the header. This is the number the whole family was chosen for, so
    // it is asserted where it can no longer be arithmetic on a model card.
    let stride = layout
        .layers
        .first()
        .and_then(|layer| layer.experts.first())
        .map(|expert| expert.sub_tensors.values().map(|sub| sub.size).sum::<u64>())
        .expect("one expert");
    let mib = stride as f64 / (1024.0 * 1024.0);
    let slot_cache_gib =
        stride as f64 * 16.0 * layout.num_layers as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!("one expert blob {mib:.2} MiB, slot cache at 16 slots {slot_cache_gib:.2} GiB");
    assert!(
        mib < 8.0,
        "expert blob {mib:.2} MiB is not fine-grained; Mixtral's is 108.9 and cannot stream here"
    );

    eprintln!(
        "SUCCESS: real Qwen3-30B-A3B Q4_K_M GGUF installed to {}",
        dir.display()
    );
}
