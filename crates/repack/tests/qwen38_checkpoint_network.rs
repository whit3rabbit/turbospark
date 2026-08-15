//! Network-gated proof that the `qwen3_5` repack path works against the
//! SECOND published checkpoint of that architecture:
//! `mlx-community/Qwen3.8-27B-4bit` (16.08 GB across three shards, pinned to
//! one commit and the index's SHA-256). Not run by default:
//!
//! ```sh
//! TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
//!   cargo test -p turbospark-repack --test qwen38_checkpoint_network --release -- --ignored --nocapture
//! ```
//!
//! **THIS IS A SECOND CHECKPOINT, NOT A SECOND FAMILY, and the offline test
//! is what says so.** `Qwen/Qwen3.8-27B` and `prism-ml/Bonsai-27B-mlx-1bit`
//! agree on 33 of 35 `text_config` keys and both parse to
//! `model_io::qwen_gdn_dense_27b()`; `tests/qwen35_config.rs` asserts exactly that,
//! without the network. So no `ArchConfig` field, no kernel and no decode
//! flow moved for this checkpoint, and what is left to prove is the WALK.
//!
//! Two things it covers that `qwen35_checkpoint_network.rs` structurally
//! cannot, which is the whole reason for a second target rather than an env
//! var pointed somewhere else:
//!
//! 1. **THREE SHARDS.** Bonsai is a single-file checkpoint, so it only ever
//!    exercised `Gemma4Shards::single` on this writer. A companion tensor
//!    (`.scales` / `.biases`) living in a different shard from its weight is
//!    unreachable there, and going through `Gemma4Shards`' merged registry
//!    rather than resolving inside one shard is what handles it.
//! 2. **FOUR BITS.** Bonsai is 1-bit at group 128 with FP16 companions; this
//!    is INT4 at group 64 with BF16 ones. Those are the two arms of
//!    `is_supported_affine_shape`'s conjunction, and the companion-dtype axis
//!    is the one that fails SILENTLY -- the planes are the same width, so
//!    reading the wrong one yields an install of exactly the right size whose
//!    scales are wrong by orders of magnitude (`crates/repack` Gotcha 9).
//!    Until this checkpoint the `qwen35` walk had never written a 4-bit
//!    install.
//!
//! **THE WALK IS BYTE-REPRODUCIBLE ON THIS CHECKPOINT, verified rather than
//! asserted**: two independent streams a few hours apart produced a
//! `model_weights.bin` with SHA-256
//! `a9dc35641ea71eecf7bcc096f657b7bf27a650bfb1c93ff124a5789a18f9cf0c` both
//! times, and `qwen38_quality_gate.rs`'s frozen digests held against the
//! second one. That is what licenses the pinned constants above being a
//! fingerprint of the RESULT and not just of the download.
//!
//! Note what this checkpoint does NOT bring, so nobody goes looking: the
//! official `Qwen/Qwen3.8-27B` carries 15 `mtp.*` tensors (a multi-token
//! prediction head nothing here implements), and the mlx-community
//! conversion drops them. Only `language_model.` and `vision_tower.` survive,
//! so there is no new exclusion rule -- the walk sees the same two prefixes
//! Bonsai showed it. `mtp.*` becomes live only if the bf16 checkpoint is ever
//! repacked directly.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_gemma4_quantization, parse_qwen_gdn_dense_config,
    write_qwen_gdn_dense_install_streamed, Gemma4Shards, HttpRangeSource,
};

const REPO_BASE: &str = "https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/resolve/3e6447f082e89cc7f0bc6e5441afd38dfce760ff";
const MODEL_ID: &str = "mlx-community/Qwen3.8-27B-4bit";
/// Pinned at the commit above. The index names all 2,180 tensors and every
/// shard is reachable from it, so this one hash fingerprints the download.
const PINNED_INDEX_SHA256: &str =
    "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8";
/// Per-shard published byte sizes, asserted so a silently re-uploaded
/// artifact fails HERE rather than at some tensor offset twenty minutes in.
const SHARD_BYTES: [u64; 3] = [5_343_268_662, 5_354_185_130, 5_357_087_557];

/// The INT4-affine resident dtype tag. `resident_writer`'s constant is
/// private, so it is restated rather than imported; 15 would be Bonsai's
/// 1-bit tag and 1 is raw BF16.
const DTYPE_INT4_AFFINE: u8 = 4;

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
    match std::env::var_os("TURBOSPARK_QWEN38_INSTALL_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create install dir");
            dir
        }
        None => {
            let dir =
                std::env::temp_dir().join(format!("turbospark-qwen38-real-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }
    }
}

#[test]
#[ignore = "streams the real 16 GB Qwen3.8-27B 4-bit checkpoint over the network"]
fn repacks_the_real_qwen38_27b_checkpoint() {
    let index_bytes = get("model.safetensors.index.json");
    assert_eq!(
        model_io::hash_data(&index_bytes),
        PINNED_INDEX_SHA256,
        "model.safetensors.index.json does not match the pinned fingerprint"
    );
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    let weight_map = index["weight_map"].as_object().expect("weight_map");
    let shard_names: BTreeSet<String> = weight_map
        .values()
        .map(|v| v.as_str().expect("shard name").to_string())
        .collect();
    eprintln!("shards: {shard_names:?}");
    assert_eq!(shard_names.len(), 3, "expected a three-shard checkpoint");

    // The tensor inventory, off the index rather than assumed. Both counts
    // equal Bonsai's exactly, which is the cheapest independent evidence
    // that the two checkpoints really are one architecture -- the config
    // comparison and the tensor count share no input.
    assert_eq!(weight_map.len(), 2_180, "the tensor inventory moved");
    let vision = weight_map
        .keys()
        .filter(|n| n.starts_with("vision_tower."))
        .count();
    assert_eq!(vision, 333, "the vision tower moved");
    // No multi-token-prediction head in the mlx conversion. If this ever
    // fires, the walk has an unmapped prefix to refuse rather than to drop.
    assert!(
        !weight_map.keys().any(|n| n.starts_with("mtp.")),
        "the mlx artifact grew an mtp head; the walk has no rule for it"
    );

    // Config: the full arch, cross-checked against the pinned baseline. The
    // same assertion runs offline in `tests/qwen35_config.rs` alongside
    // Bonsai's; failing HERE and passing there means the upstream checkpoint
    // moved, not that the parser broke.
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_qwen_gdn_dense_config(&config).expect("config parses");
    assert_eq!(
        arch,
        model_io::qwen_gdn_dense_27b(),
        "parsed config does not match the pinned qwen3_5 baseline"
    );
    assert_eq!(arch.num_experts, 0, "this family is DENSE");

    // FOUR BITS AT GROUP 64, which is what makes this walk new. Bonsai's is
    // (1, 128); `is_supported_affine_shape` admits exactly these two pairs.
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");
    assert_eq!(quant.default_bits, 4, "the point of this checkpoint");
    assert_eq!(quant.group_size, 64);

    // Shard headers (a few KB each; tensor bytes stream later).
    let sources: Vec<HttpRangeSource> = shard_names
        .iter()
        .map(|name| HttpRangeSource::new(format!("{REPO_BASE}/{name}")))
        .collect();
    let headers = sources
        .iter()
        .map(|s| fetch_safetensors_header(s).expect("shard header"))
        .collect::<Vec<_>>();
    for (i, (name, header)) in shard_names.iter().zip(headers.iter()).enumerate() {
        let declared: u64 = header
            .tensors
            .values()
            .map(|t| t.data_offsets.1)
            .max()
            .expect("tensors")
            + header.data_region_start();
        assert_eq!(
            declared, SHARD_BYTES[i],
            "{name}: the header's last tensor does not end at the published file size"
        );
    }
    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(sources.iter())
            .map(|(h, s)| (h, s as &dyn turbospark_repack::RangeSource))
            .collect(),
    );

    let dir = install_dir();
    eprintln!("installing to {}", dir.display());
    write_qwen_gdn_dense_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install");

    // Tokenizer sidecars so turbospark-check can open the dir directly. This
    // checkpoint is ChatML-dialect, so a raw `--prompt` babbles; use
    // `--messages-file` or `--chat`.
    //
    // TWO DIFFERENCES FROM BONSAI'S LIST, both read off the repo rather than
    // carried over. It DOES ship `generation_config.json`, whose
    // `eos_token_id` is the LIST [248046, 248044] the stop set unions --
    // leaving it out would cost the `<|endoftext|>` stop. And it ships NO
    // `merges.txt`: this conversion embeds the merges in `tokenizer.json`,
    // so asking for one is a 404 that fails the test AFTER the 20-minute
    // stream has already written a perfectly good install. Copying a sidecar
    // list between checkpoints of one family is exactly the kind of
    // near-miss that costs a re-stream to find.
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "vocab.json",
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
    // Untied: `lm_head` is its own quantized tensor, not the embedding again.
    assert!(!arch.tie_word_embeddings);
    for name in [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "language_model.model.layers.63.mlp.gate_proj.weight",
    ] {
        assert_eq!(
            resident_index.entries[name].dtype, DTYPE_INT4_AFFINE,
            "{name} should carry the INT4-affine tag"
        );
    }
    // Every unquantized tensor is BF16. This checkpoint writes BF16 already,
    // unlike Bonsai's F16, so `narrow_raw_to_bf16` is a NO-OP here and the
    // assertion is a control rather than a repeat: it says the walk did not
    // acquire a tag no reader honours on the way through (Gotcha 45).
    for e in resident_index.entries.values() {
        if e.scale_size == 0 && e.bias_size == 0 {
            assert_eq!(e.dtype, 1, "{} is not BF16", e.name);
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
        "SUCCESS: real Qwen3.8-27B repacked into {}, run \
         `cargo run -p turbospark-cli --bin turbospark-check --release -- --model {} --messages-file /tmp/p.json`",
        dir.display(),
        dir.display()
    );
}
