//! Network-gated proof that the Gemma 4 repack pipeline works against the
//! REAL production checkpoint: `mlx-community/gemma-4-26b-a4b-it-4bit`
//! (~14.6 GB download, pinned to the same commit and index SHA-256 the
//! Swift `SupportedModelSource.gemma4` pins). Not run by default; run
//! explicitly:
//!
//! ```sh
//! TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!   cargo test -p turbospark-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture
//! ```
//!
//! Set `TURBOSPARK_GEMMA4_INSTALL_DIR` to keep the ~14 GB install for manual
//! `turbospark-check` runs (the test also downloads the tokenizer sidecars
//! into the install dir so the CLI can load it directly); otherwise a temp
//! directory is used.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_gemma4_config, parse_gemma4_quantization,
    write_gemma4_install_streamed, Gemma4Shards, HttpRangeSource,
};

const REPO_BASE: &str = "https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/resolve/0d77464eeb233a2da68ebf9d7dc4edaac7db956d";
const MODEL_ID: &str = "mlx-community/gemma-4-26b-a4b-it-4bit";
/// The Swift `SupportedModelSource.gemma4` pin.
const PINNED_INDEX_SHA256: &str =
    "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13";

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
    match std::env::var_os("TURBOSPARK_GEMMA4_INSTALL_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create install dir");
            dir
        }
        None => {
            let dir =
                std::env::temp_dir().join(format!("turbospark-gemma4-real-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }
    }
}

#[test]
#[ignore = "downloads the real ~14.6 GB Gemma 4 checkpoint over the network"]
fn repacks_the_real_gemma4_checkpoint() {
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

    // Config: full arch + quantization overrides, cross-checked against
    // the pinned production baseline.
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_gemma4_config(&config).expect("config parses");
    assert_eq!(
        arch,
        model_io::gemma4_26b_a4b(),
        "parsed config does not match the pinned Gemma 4 26B-A4B baseline"
    );
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");
    assert_eq!(quant.default_bits, 4);

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
    )
    .expect("no tensor name collides across shards");

    let dir = install_dir();
    eprintln!("installing to {}", dir.display());
    write_gemma4_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install");

    // Tokenizer sidecars so turbospark-check can open the dir directly.
    // NOTE: this checkpoint is instruction-tuned with the Gemma 4 turn
    // markup (`<|turn>user\n...<turn|>\n<|turn>model\n`); raw text
    // prompts produce out-of-distribution babble, chat-formatted prompts
    // produce real answers.
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
    assert!(resident_index
        .entries
        .contains_key("language_model.model.embed_tokens.weight"));
    assert!(resident_index
        .entries
        .contains_key("language_model.model.layers.29.router.proj.weight"));
    assert_eq!(
        resident_index.entries["language_model.model.layers.0.router.proj.weight"].dtype, 5,
        "router should carry the INT8 tag"
    );
    let layout =
        model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("experts layout");
    assert_eq!(layout.num_layers, 30);
    assert_eq!(layout.experts_per_layer, 128);
    assert_eq!(layout.expert_stride % 16_384, 0);

    eprintln!(
        "SUCCESS: real Gemma 4 26B-A4B repacked into {} — run \
         `cargo run -p turbospark-cli --bin turbospark-check --release -- --model {} --prompt \"...\"`",
        dir.display(),
        dir.display()
    );
}
