//! ROADMAP Phase G Stage 2 item 8: the real bytes. Repacks the published
//! `ggml-org/gemma-4-26B-A4B-it-GGUF` Q8_0 checkpoint into a `.gturbo`
//! install, which is the one thing in the GGUF intake a ranged read of a
//! header or two cannot stand in for.
//!
//! ```sh
//! MREFRUST_GEMMA4_GGUF_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
//!   cargo test -p mrefrust-repack --test gguf_install_network --release -- --ignored --nocapture
//! ```
//!
//! Two things about the cost, both deliberate. The 26.9 GB checkpoint is
//! NEVER materialized locally: `write_gguf_install_streamed` reads it a
//! layer at a time through `HttpRangeSource`, so the only disk this needs is
//! the install itself. And the destination must NOT be
//! `MREFRUST_GEMMA4_INSTALL_DIR`: that is the MLX-derived artifact every
//! existing gate, oracle row and quality number is measured against, and a
//! GGUF-derived install of the same model is a different artifact.

use std::path::PathBuf;

use mrefrust_repack::{fetch_gguf_header, write_gguf_install_streamed, HttpRangeSource};

const GEMMA4_Q8_0: &str = "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q8_0.gguf";
const MODEL_ID: &str = "ggml-org/gemma-4-26B-A4B-it-GGUF";

/// The GGUF carries its tokenizer as metadata, in llama.cpp's own
/// representation; this port loads an HF `tokenizer.json`. Rather than
/// convert one to the other, take the sidecars from the checkpoint the GGUF
/// was converted from -- same model, same vocabulary, few MB.
const SIDECAR_BASE: &str = "https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/resolve/0d77464eeb233a2da68ebf9d7dc4edaac7db956d";

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
    let dir = match std::env::var_os("MREFRUST_GEMMA4_GGUF_INSTALL_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("mrefrust-gemma4-gguf-{}", std::process::id())),
    };
    assert_ne!(
        std::env::var_os("MREFRUST_GEMMA4_INSTALL_DIR").map(PathBuf::from),
        Some(dir.clone()),
        "refusing to overwrite the MLX-derived install every gate is measured against"
    );
    std::fs::create_dir_all(&dir).expect("create install dir");
    dir
}

#[test]
#[ignore = "network: streams the real 26.9 GB Gemma 4 Q8_0 GGUF and writes a ~27 GB install"]
fn repacks_the_real_gemma4_q8_0_gguf() {
    let source = HttpRangeSource::new(GEMMA4_Q8_0);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    assert_eq!(header.architecture(), Some("gemma4"));
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
    assert_eq!(arch, model_io::gemma4_26b_a4b());

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        std::fs::write(dir.join(name), get(&format!("{SIDECAR_BASE}/{name}")))
            .expect("tokenizer sidecar");
    }

    // Read it back through the real loaders. `load_manifest` is also the
    // first of the two block-type gates (AGENTS.md Gotcha 29): a Q8_0
    // install has to pass it where a Q4_K one is refused here.
    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates, and q8_0 is an executable block type");
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    let embed = &resident.entries["language_model.model.embed_tokens.weight"];
    assert_eq!(
        embed.dtype,
        mrefrust_repack::dtype_tag_for_ggml_type(8).expect("q8_0 tag"),
        "the embedding table should carry the Q8_0 tag, verbatim from the file"
    );
    // The F32 core is transcoded rather than carried (Gotcha 29): the router
    // lands as INT8 affine and the norms as BF16, so nothing F32 survives.
    assert_eq!(
        resident.entries["language_model.model.layers.0.router.proj.weight"].dtype, 5,
        "router should have been INT8-transcoded"
    );

    let layout = model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("layout");
    assert_eq!(layout.num_layers, 30);
    assert_eq!(layout.experts_per_layer, 128);

    eprintln!(
        "SUCCESS: real Gemma 4 Q8_0 GGUF installed to {}",
        dir.display()
    );
}
