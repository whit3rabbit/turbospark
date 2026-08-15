//! Network-gated proof that the DENSE, TWO-BIT repack path works against the
//! REAL published checkpoint: `prism-ml/Ternary-Bonsai-27B-mlx-2bit`
//! (ROADMAP's ternary entry). Sibling of `qwen35_checkpoint_network.rs`; not
//! run by default:
//!
//! ```sh
//! TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
//!   cargo test -p turbospark-repack --test ternary_checkpoint_network --release -- --ignored --nocapture
//! ```
//!
//! **THIS IS A THIRD CHECKPOINT OF ONE ARCHITECTURE, NOT A NEW FAMILY**, and
//! `tests/qwen35_config.rs` says so without the network: this file's
//! `text_config` parses to `model_io::qwen_gdn_dense_27b()` exactly, as
//! Bonsai-27B's and Qwen3.8-27B's do. So no `ArchConfig` field, no baseline
//! and no decode flow moved for it, and what is left to prove is the WALK at a
//! width it has never written.
//!
//! The artifact is 8.49 GiB and the install is ~8, because the walk drops the
//! same 333-tensor vision tower Bonsai's has.
//!
//! What this covers that the synthetic fixture cannot (`crates/repack`
//! Gotcha 5): the real tensor INVENTORY at two bits. The fixture contains only
//! names and dtypes its author already knew, and this checkpoint's own header
//! is where the packing was read from -- 320 `u32` words for 5,120 columns is
//! 16 elements a word, i.e. exactly two bits, and 40 companions a row is group
//! 128.
//!
//! **THE SIDECAR LIST IS THIS REPO'S, NOT CARRIED OVER.** It ships `merges.txt`
//! and NO `generation_config.json` -- the exact inverse of Qwen3.8-27B's --
//! and asking for a file that 404s fails the test AFTER a ~16-minute stream
//! has written a perfectly good install. Read the repo's file list; copying a
//! sidecar list between checkpoints of one family is the near-miss that costs
//! a re-stream.

use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_gemma4_quantization, parse_qwen_gdn_dense_config,
    write_qwen_gdn_dense_install_streamed, Gemma4Shards, HttpRangeSource,
};

const REPO_BASE: &str = "https://huggingface.co/prism-ml/Ternary-Bonsai-27B-mlx-2bit/resolve/70f75f3ad081ab840a42f3304c02c27e7f89bfb7";
const MODEL_ID: &str = "prism-ml/Ternary-Bonsai-27B-mlx-2bit";
/// Pinned at the commit above. Like Bonsai's, this checkpoint is ONE shard and
/// still ships an index, so the index is the cheapest whole-file fingerprint
/// available: it names all 2,180 tensors.
const PINNED_INDEX_SHA256: &str =
    "ea95a45cda323247258962e694606dea77881ffbc6565bba84bb9aa3d6031736";
/// `model.safetensors`, bytes. Asserted so a silently re-uploaded artifact
/// fails here rather than at some tensor offset a quarter of an hour in.
const MODEL_BYTES: u64 = 8_490_785_104;

/// The 2-bit-affine resident dtype tag. `resident_writer`'s constant is
/// private, so it is restated rather than imported; 15 would be Bonsai's
/// 1-bit tag, 4 the INT4 one and 1 raw BF16.
const DTYPE_INT2_AFFINE: u8 = 16;

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
    match std::env::var_os("TURBOSPARK_TERNARY_INSTALL_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create install dir");
            dir
        }
        None => {
            let dir = std::env::temp_dir()
                .join(format!("turbospark-ternary-real-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }
    }
}

#[test]
#[ignore = "streams the real 8.5 GB Ternary-Bonsai-27B 2-bit checkpoint over the network"]
fn repacks_the_real_ternary_bonsai27b_checkpoint() {
    let index_bytes = get("model.safetensors.index.json");
    assert_eq!(
        model_io::hash_data(&index_bytes),
        PINNED_INDEX_SHA256,
        "model.safetensors.index.json does not match the pinned fingerprint"
    );
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    let weight_map = index["weight_map"].as_object().expect("weight_map");
    let shards: std::collections::BTreeSet<String> = weight_map
        .values()
        .map(|v| v.as_str().expect("shard name").to_string())
        .collect();
    assert_eq!(
        shards.len(),
        1,
        "expected a single-file checkpoint, got {shards:?}"
    );
    // The inventory equals Bonsai-27B's and Qwen3.8-27B's exactly, which is
    // the cheapest independent evidence that all three are one architecture:
    // the config comparison and the tensor count share no input.
    assert_eq!(weight_map.len(), 2_180, "the tensor inventory moved");

    // Config: the full arch, cross-checked against the pinned baseline. The
    // same assertion runs offline in `tests/qwen35_config.rs` for all three
    // checkpoints; failing HERE and passing there means the upstream file
    // moved, not that the parser broke.
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_qwen_gdn_dense_config(&config).expect("config parses");
    assert_eq!(
        arch,
        model_io::qwen_gdn_dense_27b(),
        "parsed config does not match the pinned qwen3_5 baseline"
    );
    assert_eq!(arch.num_experts, 0, "this family is DENSE");

    // TWO BITS AT GROUP 128, which is what makes this walk new. Bonsai's is
    // (1, 128) and Qwen3.8's (4, 64); `is_supported_affine_shape` admits
    // exactly those three pairs.
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");
    assert_eq!(quant.default_bits, 2, "the whole point of this checkpoint");
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
    // The PACKING, off the header rather than off the config's `bits` field,
    // which is a separate statement about the same fact: 320 packed u32 words
    // for a 5,120-element row is 16 elements a word.
    let probe = &header.tensors["language_model.model.layers.0.linear_attn.in_proj_a.weight"];
    assert_eq!(probe.shape, vec![48, 320], "the probe tensor's shape moved");

    let shards = Gemma4Shards::single(&header, &source);
    let dir = install_dir();
    eprintln!("installing to {}", dir.display());
    write_qwen_gdn_dense_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install");

    // Tokenizer sidecars so turbospark-check can open the dir directly. This
    // checkpoint is ChatML-dialect, so a raw `--prompt` babbles; use
    // `--messages-file` or `--chat`. See the module header: this list is read
    // off THIS repo, and it has `merges.txt` and no `generation_config.json`.
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
    // Untied: `lm_head` is its own 2-bit tensor, not the embedding again.
    assert!(!arch.tie_word_embeddings);
    for name in [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "language_model.model.layers.63.mlp.gate_proj.weight",
    ] {
        assert_eq!(
            resident_index.entries[name].dtype, DTYPE_INT2_AFFINE,
            "{name} should carry the 2-bit affine tag"
        );
    }
    // The width, on real bytes: a 5,120-element row at two bits is 1,280
    // packed bytes with 40 FP16 scales, so the packed plane is exactly twice
    // the 1-bit install's for the same tensor and the companion planes agree.
    // A dtype tag alone cannot say this -- all four affine tags share an entry
    // shape.
    let embed = &resident_index.entries["language_model.model.embed_tokens.weight"];
    let (rows, cols) = (arch.vocab_size as u64, arch.hidden_size as u64);
    assert_eq!(embed.size_bytes, rows * cols / 4);
    assert_eq!(embed.scale_size, rows * (cols / 128) * 2);
    assert_eq!(embed.bias_size, embed.scale_size);

    // EVERY unquantized tensor is narrowed to BF16. This checkpoint writes
    // F16 for all 1,682 of them, exactly as Bonsai does, and nothing in
    // `crates/runtime` reads an F16 tag -- so a tensor that came through
    // verbatim would be decoded as BF16 off its byte size and be wrong by up
    // to 2^112 with no error anywhere (AGENTS.md Gotcha 45).
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
        "SUCCESS: real Ternary-Bonsai-27B repacked into {}, run \
         `cargo run -p turbospark-cli --bin turbospark-check --release -- --model {} --messages-file /tmp/p.json`",
        dir.display(),
        dir.display()
    );
}
