//! ROADMAP Phase S step 4: install the real candidate. Repacks
//! `unsloth/gemma-4-26B-A4B-it-UD-Q3_K_M` into a `.gturbo` install, which is
//! the one thing in the intake that a ranged read of a header cannot stand in
//! for and a synthetic fixture cannot either.
//!
//! ```sh
//! TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
//!   cargo test -p turbospark-repack --test gguf_iq_install_network --release -- --ignored --nocapture
//! ```
//!
//! WHAT THIS ONE CHECKS THAT ITS TWO SIBLINGS DO NOT. `gguf_install_network`
//! and `gguf_qwen_install_network` both install UNIFORM files: every layer's
//! experts are the same block type, so the install's shape is one answer
//! repeated. This file is mixed along two axes -- IQ3_XXS gate/up over an
//! IQ4_NL down in the same expert, and a layer 29 that is IQ4_XS over Q8_0 --
//! and the per-layer stride, the per-layer offsets and the per-layer
//! per-phase routed layouts all exist for that. The assertions below name
//! each one, because the failure mode of getting any of them wrong is an
//! install that loads and generates fluent, wrong text.
//!
//! THE STRIDE ASSERTION IS THE PHASE'S PREMISE. Layer 29's expert blob is
//! 1.6x the others', so padding all thirty to the model-wide maximum would
//! write 16.2 GB where 10.3 is needed: a 35% regression against the 12 GiB
//! MLX install this phase exists to shrink. That number is checked here
//! rather than eyeballed after the fact.
//!
//! Cost, both parts deliberate: the 12 GB checkpoint is NEVER materialized
//! (`write_gguf_install_streamed` reads it a layer at a time), and the
//! destination must not be `TURBOSPARK_GEMMA4_INSTALL_DIR`, the MLX-derived
//! artifact every gate and oracle row is measured against.

use std::path::PathBuf;

use turbospark_repack::{fetch_gguf_header, write_gguf_install_streamed, HttpRangeSource};

const GEMMA4_UD_Q3_K_M: &str = "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf";
const MODEL_ID: &str = "unsloth/gemma-4-26B-A4B-it-GGUF";

/// The GGUF carries its tokenizer as llama.cpp metadata; this port loads an
/// HF `tokenizer.json`. Same sidecars the Q8_0 install takes, from the
/// checkpoint both GGUFs were converted from.
const SIDECAR_BASE: &str = "https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/resolve/0d77464eeb233a2da68ebf9d7dc4edaac7db956d";

/// ggml type ids, for reading the header's own answer rather than assuming.
const GGML_Q8_0: u32 = 8;
const GGML_Q6_K: u32 = 14;
const GGML_IQ3_XXS: u32 = 18;
const GGML_IQ4_NL: u32 = 20;
const GGML_IQ4_XS: u32 = 23;

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
    let dir = match std::env::var_os("TURBOSPARK_GEMMA4_IQ_INSTALL_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("turbospark-gemma4-iq-{}", std::process::id())),
    };
    for guard in [
        "TURBOSPARK_GEMMA4_INSTALL_DIR",
        "TURBOSPARK_QWEN36_INSTALL_DIR",
    ] {
        assert_ne!(
            std::env::var_os(guard).map(PathBuf::from),
            Some(dir.clone()),
            "refusing to overwrite {guard}, which gates are measured against"
        );
    }
    std::fs::create_dir_all(&dir).expect("create install dir");
    dir
}

#[test]
#[ignore = "network: streams the real 12 GB Gemma 4 UD-Q3_K_M GGUF and writes a ~10 GB install"]
fn repacks_the_real_gemma4_iq3_gguf() {
    let source = HttpRangeSource::new(GEMMA4_UD_Q3_K_M);
    let header = fetch_gguf_header(&source).expect("fetch GGUF header");
    assert_eq!(header.architecture(), Some("gemma4"));

    // The mixture, read off the file. Asserted rather than trusted, because
    // everything below is sized for exactly this shape and a re-upload that
    // changed it should say so here rather than three assertions later.
    let type_of = |name: &str| header.tensors[name].ggml_type;
    assert_eq!(type_of("blk.0.ffn_gate_up_exps.weight"), GGML_IQ3_XXS);
    assert_eq!(type_of("blk.0.ffn_down_exps.weight"), GGML_IQ4_NL);
    assert_eq!(type_of("blk.29.ffn_gate_up_exps.weight"), GGML_IQ4_XS);
    assert_eq!(type_of("blk.29.ffn_down_exps.weight"), GGML_Q8_0);
    assert_eq!(type_of("token_embd.weight"), GGML_Q6_K);
    assert!(
        !header.tensors.contains_key("output.weight"),
        "Gemma ties its embeddings; a separate head would need its own kernel"
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

    // The manifest gate. A mixed routed slot has to declare EVERY type it
    // carries, and all four have to be executable, or `load_manifest` refuses.
    let manifest = model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates: every routed block type is executable");
    let quant = manifest.quant.as_ref().expect("production manifest");
    let mut declared: Vec<String> = quant
        .routed_expert
        .ggml_types
        .clone()
        .expect("a mixed routed slot declares ggmlTypes")
        .iter()
        .map(|t| t.to_lowercase())
        .collect();
    declared.sort();
    assert_eq!(declared, ["iq3_xxs", "iq4_nl", "iq4_xs", "q8_0"]);

    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    assert_eq!(
        resident.entries["language_model.model.embed_tokens.weight"].dtype,
        turbospark_repack::dtype_tag_for_ggml_type(GGML_Q6_K).expect("q6_k tag"),
        "the embedding table should carry the Q6_K tag, verbatim from the file"
    );
    // The F32 core is transcoded rather than carried (AGENTS.md Gotcha 29).
    assert_eq!(
        resident.entries["language_model.model.layers.0.router.proj.weight"].dtype, 5,
        "router should have been INT8-transcoded"
    );

    let layout = model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("layout");
    assert_eq!(layout.num_layers, 30);
    assert_eq!(layout.experts_per_layer, 128);

    // Per-layer dtypes: the two axes, in the artifact.
    let roles = |layer: usize| {
        let s = &layout.expert(layer, 0).sub_tensors;
        (
            s["gate"].dtype.clone(),
            s["up"].dtype.clone(),
            s["down"].dtype.clone(),
        )
    };
    assert_eq!(
        roles(0),
        ("iq3_xxs".into(), "iq3_xxs".into(), "iq4_nl".into())
    );
    assert_eq!(roles(29), ("iq4_xs".into(), "iq4_xs".into(), "q8_0".into()));

    // Per-layer stride: the phase's whole premise, in bytes on disk.
    let normal = layout.layers[0].expert_stride;
    let odd = layout.layers[29].expert_stride;
    assert!(
        odd > normal,
        "layer 29 should hold the wider blob: {odd} vs {normal}"
    );
    assert_eq!(
        layout.expert_stride, odd,
        "the top-level value should be the model-wide maximum"
    );
    let actual: u64 = layout
        .layers
        .iter()
        .map(|l| l.expert_stride * layout.experts_per_layer as u64)
        .sum();
    let uniform = odd * layout.experts_per_layer as u64 * layout.num_layers as u64;
    eprintln!(
        "expert bytes: {:.2} GB per-layer, {:.2} GB if padded to the maximum",
        actual as f64 / 1e9,
        uniform as f64 / 1e9
    );
    assert!(
        actual < uniform * 7 / 10,
        "per-layer striding should save over 30%: {actual} vs {uniform}"
    );
    // And the bytes really are on disk at that size, not merely recorded.
    for layer in [0usize, 29] {
        let path = dir.join("packed_experts").join(&layout.layers[layer].file);
        assert_eq!(
            std::fs::metadata(&path).expect("layer file").len(),
            layout.layers[layer].expert_stride * layout.experts_per_layer as u64,
            "layer {layer} on disk does not match its declared stride"
        );
    }

    eprintln!(
        "SUCCESS: real Gemma 4 UD-Q3_K_M GGUF installed to {}",
        dir.display()
    );
}
