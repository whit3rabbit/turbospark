//! ROADMAP Phase G Stage 2 item 9: the real bytes, for the K-quants.
//! Repacks the published `ggml-org/Qwen3.6-35B-A3B-GGUF` Q4_K_M checkpoint
//! into a `.gturbo` install, which is the one thing a synthetic fixture
//! cannot stand in for -- the fixture's quantized bytes come from this
//! port's own quantizers, so it can agree with a wrong decoder.
//!
//! ```sh
//! TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
//!   cargo test -p turbospark-repack --test gguf_qwen_install_network --release -- --ignored --nocapture
//! ```
//!
//! Same two cost properties as the Gemma sibling. The checkpoint is NEVER
//! materialized locally: `write_gguf_install_streamed` reads it a layer at a
//! time through `HttpRangeSource`, so the only disk this needs is the
//! install. And the destination must NOT be `TURBOSPARK_QWEN36_INSTALL_DIR`:
//! that is the MLX-derived artifact the Qwen oracle row and quality numbers
//! are measured against, and this is a different artifact of the same model.
//!
//! What is different from Gemma's file, and why this test exists separately:
//! Q4_K_M is MIXED. Its routed experts and embedding table are Q4_K, its
//! attention and shared experts Q8_0, and exactly one tensor -- `output.weight`
//! -- is Q6_K. So this is the first install where the block type has to be
//! read per tensor rather than decided once.

use std::path::PathBuf;

use turbospark_repack::{fetch_gguf_header, write_gguf_install_streamed, HttpRangeSource};

const QWEN36_Q4_K_M: &str =
    "https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf";
const MODEL_ID: &str = "ggml-org/Qwen3.6-35B-A3B-GGUF";

/// The GGUF carries its tokenizer as llama.cpp metadata; this port loads an
/// HF `tokenizer.json`. Take the sidecars from the checkpoint the GGUF was
/// converted from, as the Gemma sibling does.
const SIDECAR_BASE: &str = "https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/resolve/38740b847e4cb78f352aba30aa41c76e08e6eb46";

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
    let dir = match std::env::var_os("TURBOSPARK_QWEN36_GGUF_INSTALL_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("turbospark-qwen36-gguf-{}", std::process::id())),
    };
    assert_ne!(
        std::env::var_os("TURBOSPARK_QWEN36_INSTALL_DIR").map(PathBuf::from),
        Some(dir.clone()),
        "refusing to overwrite the MLX-derived install every gate is measured against"
    );
    std::fs::create_dir_all(&dir).expect("create install dir");
    dir
}

#[test]
#[ignore = "network: streams the real ~20 GB Qwen 3.6 Q4_K_M GGUF and writes a ~20 GB install"]
fn repacks_the_real_qwen36_q4_k_m_gguf() {
    let source = HttpRangeSource::new(QWEN36_Q4_K_M);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    // The converter's name, not the family's (AGENTS.md Gotcha 29).
    assert_eq!(header.architecture(), Some("qwen35moe"));
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
    assert_eq!(arch, model_io::qwen36_35b_a3b());

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        std::fs::write(dir.join(name), get(&format!("{SIDECAR_BASE}/{name}")))
            .expect("tokenizer sidecar");
    }

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates: q4_k, q6_k and q8_0 are all executable block types");
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    // The three block types, each where the real file puts it. This is the
    // assertion the Gemma install cannot make: that file is Q8_0 throughout,
    // so nothing in it proves the dtype is read per tensor.
    let tag = |ggml: u32| turbospark_repack::dtype_tag_for_ggml_type(ggml).expect("tag");
    assert_eq!(
        resident.entries["language_model.model.embed_tokens.weight"].dtype,
        tag(12),
        "the embedding table should carry the Q4_K tag, verbatim from the file"
    );
    assert_eq!(
        resident.entries["language_model.lm_head.weight"].dtype,
        tag(14),
        "output.weight is the file's ONLY Q6_K tensor"
    );
    // Searched rather than indexed: Qwen 3.6 is hybrid, so only 10 of its 40
    // layers carry `q_proj` at all and layer 0 is not one of them.
    let q_proj = resident
        .entries
        .iter()
        .find(|(name, _)| name.ends_with("self_attn.q_proj.weight"))
        .expect("a full-attention layer exists");
    assert_eq!(
        q_proj.1.dtype,
        tag(8),
        "attention stays Q8_0 in a Q4_K_M ({})",
        q_proj.0
    );
    // The F32 core is transcoded rather than carried (Gotcha 29): the router
    // lands as INT8 affine and the norms as BF16, so nothing F32 survives.
    assert_eq!(
        resident.entries["language_model.model.layers.0.mlp.gate.weight"].dtype, 5,
        "router should have been INT8-transcoded"
    );

    let layout = model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("layout");
    assert_eq!(layout.num_layers, 40);
    assert_eq!(layout.experts_per_layer, 256);

    eprintln!(
        "SUCCESS: real Qwen 3.6 Q4_K_M GGUF installed to {}",
        dir.display()
    );
}
