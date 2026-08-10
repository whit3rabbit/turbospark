//! ROADMAP Phase M2: the real bytes, for the `llama` architecture. Repacks
//! the published `mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF` Q4_K_M
//! checkpoint into a `.gturbo` install.
//!
//! ```sh
//! TURBOSPARK_MIXTRAL_INSTALL_DIR=~/models/mixtral-gguf.gturbo \
//!   cargo test -p turbospark-repack --test gguf_mixtral_install_network --release -- --ignored --nocapture
//! ```
//!
//! Same cost properties as its two siblings: the 26 GB checkpoint is NEVER
//! materialized locally, since `write_gguf_install_streamed` reads it a layer
//! at a time through `HttpRangeSource`, so the only disk this needs is the
//! install.
//!
//! What is new here, and why it is a separate test rather than an env var
//! pointed at an existing one:
//!
//! 1. **A third family**, and the first whose architecture string covers two
//!    different models (dense Llama and Mixtral share `llama`).
//! 2. **Q6_K in a ROUTED expert**, on 16 of 32 layers, against Q4_K on the
//!    other 16 -- the first mixture whose two block types differ across
//!    LAYERS of one model rather than across the phases of one expert. That
//!    is what `RoutedBlobLayout` became per-layer for in Phase S, exercised
//!    here for the first time by a real file.
//! 3. **Q5_K on `attn_output`**, a type this port had no kernel for until
//!    Phase M2.

use std::path::PathBuf;

use turbospark_repack::{fetch_gguf_header, write_gguf_install_streamed, HttpRangeSource};

/// The DENSE half of the same architecture string, at 0.84 GiB instead of 26.
///
/// It exists because the Mixtral walk in this file is blocked on transport,
/// and because the half of the `llama` name table it exercises is the one
/// Mixtral never reaches:
/// `ffn_gate`/`ffn_up`/`ffn_down` without `_exps`, and no router at all. It
/// also pins the two ends of the dense story -- `arch_from_gguf` must produce
/// `num_experts = 0` rather than failing on the absent MoE keys, and
/// `RealForwardRunner::open` must then refuse the install BY NAME rather than
/// running it on a flow that was never written for it.
const TINYLLAMA_Q6_K: &str = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q6_K.gguf";
const TINYLLAMA_MODEL_ID: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";

const MIXTRAL_Q4_K_M: &str = "https://huggingface.co/mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf";
const MODEL_ID: &str = "mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF";

/// The GGUF carries its tokenizer as llama.cpp metadata; this port loads an
/// HF `tokenizer.json`. Take the sidecars from the checkpoint the GGUF was
/// converted from, as both siblings do.
const SIDECAR_BASE: &str =
    "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main";

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
    let dir = match std::env::var_os("TURBOSPARK_MIXTRAL_INSTALL_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("turbospark-mixtral-{}", std::process::id())),
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
#[ignore = "network: streams the real ~26 GB Mixtral Q4_K_M GGUF and writes a ~26 GB install"]
fn repacks_the_real_mixtral_q4_k_m_gguf() {
    let source = HttpRangeSource::new(MIXTRAL_Q4_K_M);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    assert_eq!(header.architecture(), Some("llama"));
    // The merged expert layout. A pre-merge conversion of the same model
    // carries 256 per-expert tensors per role instead and is refused BY NAME
    // by `gguf_names.rs`; asserting here says which of the two this URL is.
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
    assert_eq!(arch, model_io::mixtral_8x7b());

    for name in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::write(dir.join(name), get(&format!("{SIDECAR_BASE}/{name}")))
            .expect("tokenizer sidecar");
    }

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates: q4_k, q5_k, q6_k and q8_0 are all executable");
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    // Every block type, where the header probe said the real file puts it
    // (`gguf_checkpoint_network.rs::scopes_phase_m2_from_the_mixtral_header`).
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
    assert_eq!(
        dtype(&format!("{l0}.self_attn.o_proj.weight")),
        tag(13),
        "attn_output is Q5_K -- the type Phase M2 wrote a kernel for"
    );
    assert_eq!(
        dtype(&format!("{l0}.self_attn.k_proj.weight")),
        tag(8),
        "attn_k stays Q8_0 in a Q4_K_M"
    );
    // The F32 core is transcoded rather than carried (Gotcha 29): the router
    // lands as INT8 affine and the norms as BF16, so nothing F32 survives.
    assert_eq!(
        dtype(&format!("{l0}.mlp.gate.weight")),
        5,
        "the router should have been INT8-transcoded"
    );

    let layout = model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("layout");
    assert_eq!(layout.num_layers, 32);
    assert_eq!(layout.experts_per_layer, 8);

    // THE MIXTURE THIS INSTALL EXISTS TO EXERCISE: `ffn_down_exps` is Q4_K on
    // half the layers and Q6_K on the other half, so the routed layout has to
    // be resolved per LAYER. A model-wide answer would be wrong on 16 of 32.
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

    eprintln!(
        "SUCCESS: real Mixtral Q4_K_M GGUF installed to {}",
        dir.display()
    );
}

#[test]
#[ignore = "network: streams a 0.84 GB dense llama GGUF and writes a ~0.9 GB install"]
fn repacks_a_real_dense_llama_gguf_and_then_refuses_to_run_it() {
    let source = HttpRangeSource::new(TINYLLAMA_Q6_K);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    assert_eq!(header.architecture(), Some("llama"));
    assert!(
        !header.metadata.contains_key("llama.expert_count"),
        "this is supposed to be the DENSE half of the architecture"
    );
    // The dense FFN names, which a Mixtral file does not carry.
    for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
        assert!(
            header.tensors.contains_key(&format!("blk.0.{suffix}")),
            "expected a dense {suffix}"
        );
    }

    let dir = std::env::temp_dir().join(format!("turbospark-tinyllama-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create install dir");
    let arch = write_gguf_install_streamed(&dir, &header, &source, TINYLLAMA_MODEL_ID, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed GGUF install");

    // The MoE keys are absent, so `arch_from_gguf` has to DEFAULT them rather
    // than fail: this is the assertion that says the `llama` family covers
    // both halves of its architecture string.
    assert_eq!(arch.num_experts, 0);
    assert_eq!(arch.top_k_experts, 0);
    assert_eq!(arch.moe_intermediate_size, 0);
    assert!(arch.full_attention_layer_mask.iter().all(|&m| m == 1));

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates: q6_k is an executable block type");
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    for suffix in [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ] {
        assert!(
            resident
                .entries
                .contains_key(&format!("language_model.model.layers.0.{suffix}")),
            "the dense FFN should have been mapped to {suffix}"
        );
    }

    // What happens NEXT -- `RealForwardRunner::open` refusing this install by
    // name -- is asserted in `crates/runtime/tests/real_forward_llama.rs`
    // rather than here, because `turbospark-repack` must not depend on
    // `turbospark-runtime` (the edge already runs the other way: the runtime
    // tests build their fixtures with this crate).

    std::fs::remove_dir_all(&dir).ok();
    eprintln!("SUCCESS: dense llama GGUF walked end to end");
}
