//! Network-gated proof that the Qwen 3.6 repack pipeline works against the
//! REAL production checkpoint: `mlx-community/Qwen3.6-35B-A3B-4bit`
//! (~20.4 GB across four shards, pinned to one commit and the index's
//! SHA-256). Sibling of `gemma4_checkpoint_network.rs`; not run by default:
//!
//! ```sh
//! TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
//!   cargo test -p turbospark-repack --test qwen36_checkpoint_network --release -- --ignored --nocapture
//! ```
//!
//! This is the ONLY thing that proves the multi-shard walk on this family:
//! the synthetic fixture is a single in-memory shard, so a companion tensor
//! living in a different shard than its weight is unexercised there.
//!
//! Set `TURBOSPARK_QWEN36_INSTALL_DIR` to keep the install for manual
//! `turbospark-check` runs (the test downloads the tokenizer sidecars into it
//! so the CLI can open the directory directly); otherwise a temp directory
//! is used.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_gemma4_quantization, parse_qwen36_config,
    write_qwen36_install_streamed, Gemma4Shards, HttpRangeSource,
};

const REPO_BASE: &str = "https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/resolve/38740b847e4cb78f352aba30aa41c76e08e6eb46";
const MODEL_ID: &str = "mlx-community/Qwen3.6-35B-A3B-4bit";
/// Pinned at the commit above; the whole tensor set is reachable from it,
/// so this one hash fingerprints the entire download.
const PINNED_INDEX_SHA256: &str =
    "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea";

fn get(path: &str) -> Vec<u8> {
    let url = format!("{REPO_BASE}/{path}");
    let response = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .expect("client")
        .get(&url)
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
    match std::env::var_os("TURBOSPARK_QWEN36_INSTALL_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create install dir");
            dir
        }
        None => {
            let dir =
                std::env::temp_dir().join(format!("turbospark-qwen36-real-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }
    }
}

#[test]
#[ignore = "downloads the real ~20.4 GB Qwen 3.6 checkpoint over the network"]
fn repacks_the_real_qwen36_checkpoint() {
    // Index: pinned fingerprint, shard list.
    let index_bytes = get("model.safetensors.index.json");
    assert_eq!(
        model_io::hash_data(&index_bytes),
        PINNED_INDEX_SHA256,
        "model.safetensors.index.json does not match the pinned fingerprint"
    );
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    let shard_names: BTreeSet<String> = index["weight_map"]
        .as_object()
        .expect("weight_map")
        .values()
        .map(|v| v.as_str().expect("shard name").to_string())
        .collect();
    eprintln!("shards: {shard_names:?}");
    assert_eq!(shard_names.len(), 4, "expected a four-shard checkpoint");

    // Config: full arch + quantization overrides, cross-checked against the
    // pinned production baseline. The same assertion runs offline in
    // `tests/qwen36_config.rs`; failing HERE and passing there means the
    // upstream checkpoint moved, not that the parser broke.
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_qwen36_config(&config).expect("config parses");
    assert_eq!(
        arch,
        model_io::qwen36_35b_a3b(),
        "parsed config does not match the pinned Qwen3.6-35B-A3B baseline"
    );
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");
    assert_eq!(quant.default_bits, 4);
    // `validate_quant` accepts router 8 ONLY, so an upstream change here
    // would fail much later, at load, with a far less obvious message.
    let manifest_quant = turbospark_repack::manifest_quant(&quant, model_io::ModelFamily::Qwen36);
    assert_eq!(
        manifest_quant["router"]["weightBits"], 8,
        "router must be INT8"
    );

    // Shard headers (a few KB each; tensor bytes stream later).
    let sources: Vec<HttpRangeSource> = shard_names
        .iter()
        .map(|name| HttpRangeSource::new(format!("{REPO_BASE}/{name}")))
        .collect();
    let headers = sources
        .iter()
        .map(|s| fetch_safetensors_header(s).expect("shard header"))
        .collect::<Vec<_>>();
    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(sources.iter())
            .map(|(h, s)| (h, s as &dyn turbospark_repack::RangeSource))
            .collect(),
    );

    let dir = install_dir();
    eprintln!("installing to {}", dir.display());
    write_qwen36_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install");

    // Tokenizer sidecars so turbospark-check can open the dir directly. This
    // checkpoint is ChatML-dialect, so raw `--prompt` text babbles the same
    // way Gemma's does; use `--messages-file` or `--chat`.
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        std::fs::write(dir.join(name), get(name)).expect("tokenizer sidecar");
    }

    // Read the install back through the real loaders.
    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates against the production baseline");
    let resident_index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    for name in [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        // Layer 0 is LINEAR: it has no self_attn at all.
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.0.linear_attn.conv1d.weight",
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.model.layers.0.linear_attn.dt_bias",
        // Layer 3 is the first FULL-attention layer.
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.layers.3.self_attn.k_norm.weight",
        "language_model.model.layers.39.mlp.shared_expert.down_proj.weight",
        "language_model.model.layers.39.mlp.shared_expert_gate.weight",
    ] {
        assert!(
            resident_index.entries.contains_key(name),
            "resident index is missing {name}"
        );
    }
    // `lm_head` is untied on this family, so it must be its own entry.
    assert!(!arch.tie_word_embeddings);
    for layer in [0usize, 39] {
        assert_eq!(
            resident_index.entries[&format!("language_model.model.layers.{layer}.mlp.gate.weight")]
                .dtype,
            5,
            "layer {layer} router should carry the INT8 tag"
        );
    }
    // Routed experts must NOT be resident: a wrong routed marker classifies
    // all 256 of them as resident, which loads and runs but blows the
    // footprint up by the whole expert table.
    assert!(
        !resident_index
            .entries
            .keys()
            .any(|k| k.contains(".mlp.switch_mlp.")),
        "routed experts leaked into the resident set"
    );
    // Vision tower is excluded outright.
    assert!(
        !resident_index
            .entries
            .keys()
            .any(|k| k.starts_with("vision_tower.")),
        "vision tower leaked into the resident set"
    );

    let layout =
        model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("experts layout");
    assert_eq!(layout.num_layers, 40);
    assert_eq!(layout.experts_per_layer, 256);
    assert_eq!(layout.expert_stride % 16_384, 0);

    eprintln!(
        "SUCCESS: real Qwen3.6-35B-A3B repacked into {}, run \
         `cargo run -p turbospark-cli --bin turbospark-check --release -- --model {} --messages-file /tmp/p.json`",
        dir.display(),
        dir.display()
    );
}
