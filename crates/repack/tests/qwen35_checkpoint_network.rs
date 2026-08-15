//! Network-gated proof that the DENSE, ONE-BIT repack path works against the
//! REAL published checkpoint: `prism-ml/Bonsai-27B-mlx-1bit` (ROADMAP's 1-bit
//! entry, step 4's second half). Sibling of `qwen36_checkpoint_network.rs`;
//! not run by default:
//!
//! ```sh
//! TURBOSPARK_QWEN35_INSTALL_DIR=~/models/bonsai27b.gturbo \
//!   cargo test -p turbospark-repack --test qwen35_checkpoint_network --release -- --ignored --nocapture
//! ```
//!
//! **The artifact is 4.78 GiB and the install is ~3.92**, because the walk
//! drops the vision tower: 333 F16 tensors under `vision_tower.`, 0.858 GiB,
//! excluded by `classify_for_family` the same way Gemma's is. TEXT-ONLY was
//! the owner's scope decision, and the checkpoint's own
//! `language_model_only` field says the model expects it to be available.
//!
//! What this covers that the synthetic fixture cannot, which is the reason it
//! exists at all (`crates/repack` Gotcha 5): the real tensor INVENTORY. The
//! fixture only ever contains names and dtypes its author already knew, and
//! the F16-versus-BF16 axis is exactly what it got wrong -- it was forked
//! from the Qwen 3.6 fixture, whose checkpoint really is BF16, so it said
//! nothing about a walk writing F16 bytes under a tag no reader honours until
//! this checkpoint's header was read.

use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_gemma4_quantization, parse_qwen_gdn_dense_config,
    write_qwen_gdn_dense_install_streamed, Gemma4Shards, HttpRangeSource,
};

const REPO_BASE: &str = "https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit/resolve/ef22f239c670078e1507f9769bcaa66657332b96";
const MODEL_ID: &str = "prism-ml/Bonsai-27B-mlx-1bit";
/// Pinned at the commit above. This checkpoint is ONE shard and still ships
/// an index, so the index is the cheapest whole-file fingerprint available:
/// it names all 2,180 tensors.
const PINNED_INDEX_SHA256: &str =
    "90cf4963944951462c8b647fdf6a76b91e337148d0a8b2f26c9bfb68ea29b15e";
/// `model.safetensors`, bytes. Asserted so a silently re-uploaded artifact
/// fails here rather than at some tensor offset half an hour in.
const MODEL_BYTES: u64 = 5_129_115_752;

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
    match std::env::var_os("TURBOSPARK_QWEN35_INSTALL_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create install dir");
            dir
        }
        None => {
            let dir =
                std::env::temp_dir().join(format!("turbospark-qwen35-real-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }
    }
}

#[test]
#[ignore = "downloads the real 4.78 GB Bonsai-27B 1-bit checkpoint over the network"]
fn repacks_the_real_bonsai27b_checkpoint() {
    let index_bytes = get("model.safetensors.index.json");
    assert_eq!(
        model_io::hash_data(&index_bytes),
        PINNED_INDEX_SHA256,
        "model.safetensors.index.json does not match the pinned fingerprint"
    );
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    let shards: std::collections::BTreeSet<String> = index["weight_map"]
        .as_object()
        .expect("weight_map")
        .values()
        .map(|v| v.as_str().expect("shard name").to_string())
        .collect();
    assert_eq!(
        shards.len(),
        1,
        "expected a single-file checkpoint, got {shards:?}"
    );

    // Config: the full arch, cross-checked against the pinned baseline. The
    // same assertion runs offline in `tests/qwen35_config.rs`; failing HERE
    // and passing there means the upstream checkpoint moved.
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_qwen_gdn_dense_config(&config).expect("config parses");
    assert_eq!(
        arch,
        model_io::qwen_gdn_dense_27b(),
        "parsed config does not match the pinned qwen3_5 baseline"
    );
    assert_eq!(arch.num_experts, 0, "this family is DENSE");
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");
    assert_eq!(quant.default_bits, 1, "the whole point of this checkpoint");
    assert_eq!(quant.group_size, 128);

    let source = HttpRangeSource::new(format!("{REPO_BASE}/model.safetensors"));
    let header = fetch_safetensors_header(&source).expect("safetensors header");
    let declared: u64 = header
        .tensors
        .values()
        .map(|t| t.data_offsets.1)
        .max()
        .expect("tensors")
        + header.data_region_start();
    assert_eq!(
        declared, MODEL_BYTES,
        "the header's last tensor does not end at the published file size"
    );
    // Read off the header rather than assumed: the vision tower is a THIRD of
    // the tensors and none of the install.
    let vision = header
        .tensors
        .keys()
        .filter(|n| n.starts_with("vision_tower."))
        .count();
    assert_eq!(vision, 333, "the vision tower moved");

    let shards = Gemma4Shards::single(&header, &source);
    let dir = install_dir();
    eprintln!("installing to {}", dir.display());
    write_qwen_gdn_dense_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install");

    // Tokenizer sidecars so turbospark-check can open the dir directly. This
    // checkpoint is ChatML-dialect, so a raw `--prompt` babbles; use
    // `--messages-file` or `--chat`. NOTE it ships no `generation_config.json`
    // (the stop set unions that file's `eos_token_id` list when present) and
    // no `merges.txt`-free tokenizer, so both text files come along.
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
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
        // Layer 3 is the first FULL-attention layer (`full_attention_interval`
        // 4, so every fourth).
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.layers.3.self_attn.k_norm.weight",
        // The DENSE FFN, on the last layer.
        "language_model.model.layers.63.mlp.gate_proj.weight",
        "language_model.model.layers.63.mlp.down_proj.weight",
    ] {
        assert!(
            resident_index.entries.contains_key(name),
            "resident index is missing {name}"
        );
    }
    // Untied: `lm_head` is its own 1-bit tensor, not the embedding again.
    assert!(!arch.tie_word_embeddings);
    for name in [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "language_model.model.layers.63.mlp.gate_proj.weight",
    ] {
        assert_eq!(
            resident_index.entries[name].dtype, 15,
            "{name} should carry the 1-bit affine tag"
        );
    }
    // EVERY unquantized tensor is narrowed to BF16. The checkpoint writes F16
    // and nothing in `crates/runtime` reads an F16 tag, so a tensor that came
    // through verbatim would be decoded as BF16 off its byte size and be
    // wrong by up to 2^112 with no error anywhere.
    for e in resident_index.entries.values() {
        if e.scale_size == 0 && e.bias_size == 0 {
            assert_eq!(e.dtype, 1, "{} was not narrowed to BF16", e.name);
        }
    }
    // A dense install: no router, no shared expert, no routed experts.
    for marker in [".mlp.gate.weight", ".mlp.shared_expert", ".switch_mlp."] {
        assert!(
            !resident_index.entries.keys().any(|k| k.contains(marker)),
            "a dense install carries {marker}"
        );
    }
    // The vision tower is excluded outright.
    assert!(
        !resident_index
            .entries
            .keys()
            .any(|k| k.starts_with("vision_tower.")),
        "vision tower leaked into the resident set"
    );

    let layout =
        model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("experts layout");
    assert_eq!(layout.layers.len(), 0, "a dense model streams nothing");

    eprintln!(
        "SUCCESS: real Bonsai-27B repacked into {}, run \
         `cargo run -p turbospark-cli --bin turbospark-check --release -- --model {} --messages-file /tmp/p.json`",
        dir.display(),
        dir.display()
    );
}
