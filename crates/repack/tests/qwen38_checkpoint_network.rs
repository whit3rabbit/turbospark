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
//! Note what this checkpoint does NOT bring: the official `Qwen/Qwen3.8-27B`
//! carries 15 `mtp.*` tensors (a multi-token prediction head) and the
//! mlx-community conversion drops them, so only `language_model.` and
//! `vision_tower.` survive and the walk sees the same two prefixes Bonsai
//! showed it. The FIRST test below is that install and has no drafter.
//!
//! **The SECOND test builds the same trunk WITH the head**, by handing
//! `Gemma4Shards` a fourth `(header, source)` pair pointing at the official
//! repository's last shard (`docs/MTP_SPECULATIVE.md`, step 1). The two are
//! separate targets writing separate directories on purpose: the headless
//! one is the CONTROL that says the head's mere presence moves no number,
//! which no single-install run can show, and it is what the pinned
//! `model_weights.bin` SHA-256 above describes.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_gemma4_quantization, parse_qwen_gdn_dense_config,
    write_qwen_gdn_dense_install_streamed, Gemma4Shards, HttpRangeSource, RangeSource,
    SafetensorsHeader,
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
/// Raw BF16, which every unquantized tensor in a written install carries
/// (Gotcha 45: the walk narrows rather than tagging, so there is no other).
const DTYPE_BF16: u8 = 1;

// -- The official BF16 checkpoint, for the MTP head alone -------------------

/// `Qwen/Qwen3.8-27B`, pinned at a REVISION rather than `resolve/main`.
/// `tests/mtp_head_network.rs` reads this repo at `main` and that is fine for
/// a header probe; an install test writes 14 GB off these bytes and has to
/// name which ones.
const OFFICIAL_BASE: &str =
    "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
/// Its own index, which is NOT the mlx one: 1,199 tensors under
/// `model.language_model.`, `lm_head.` and `mtp.`, where the conversion
/// re-spells the first two and drops the third.
const OFFICIAL_INDEX_SHA256: &str =
    "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df";
/// The one shard the index maps every `mtp.*` tensor to.
const MTP_SHARD: &str = "model-00018-of-00018.safetensors";
/// Its published size. **3.39 GB, and only 849 MB of that is the head** --
/// the shard also holds a BF16 `lm_head.weight`, which is why the filter
/// below exists. The walk reads RANGES, so the extra 2.54 GB never moves.
const MTP_SHARD_BYTES: u64 = 3_392_197_344;
/// The 15 `mtp.*` tensors' own byte total, cross-checked against
/// `tests/mtp_head_network.rs`, which derives the same figure from shapes.
const MTP_TENSOR_BYTES: u64 = 849_398_784;

fn get(path: &str) -> Vec<u8> {
    get_from(REPO_BASE, path)
}

fn get_from(base: &str, path: &str) -> Vec<u8> {
    let url = format!("{base}/{path}");
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
    install_dir_from("TURBOSPARK_QWEN38_INSTALL_DIR", "turbospark-qwen38-real")
}

fn install_dir_from(var: &str, slug: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create install dir");
            dir
        }
        None => {
            let dir = std::env::temp_dir().join(format!("{slug}-{}", std::process::id()));
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
    )
    .expect("no tensor name collides across shards");

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

// -- The same trunk, WITH the multi-token-prediction head -------------------

/// The 15 tensors the head is made of, in the shard's own order. Restated
/// here rather than imported from `tests/mtp_head_network.rs` (integration
/// tests are separate binaries), and that duplication is deliberate: this
/// list is what the FILTER below is checked against, so a head that grew a
/// tensor has to redden both files rather than silently widen this one.
const MTP_TENSORS: [&str; 15] = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
];

/// The head's SEVEN rank-1 tensors, which `mtp::read_mtp_entries` narrows to
/// BF16 where it quantizes the eight rank-2 ones. That module splits by RANK
/// and not by a name list, so this is the list that says the split landed
/// where the published shapes put it: five `[5120]` norms plus the two
/// `[256]` per-head ones. The other eight (`fc`, q/k/v/o and the three MLP
/// projections) are matrices and are asserted as the complement below, so
/// the two lists cannot drift apart or both grow.
const MTP_NORMS: [&str; 7] = [
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
];

/// Restricts a shard header to the head's tensors, and asserts what it drops.
///
/// **`Gemma4Shards::new` merges every shard's registry, so a fourth pair
/// contributes ALL of its names**, and the official `model-00018-of-00018`
/// holds a 2.54 GB BF16 `lm_head.weight` beside the head. That name matches
/// no prefix `classify_for_family` knows -- the official repo spells its
/// trunk `model.language_model.` where the conversion spells it
/// `language_model.model.` -- so it would classify `Unknown` and the walk
/// would refuse the whole install by name, twenty minutes in.
///
/// Dropping it is correct rather than convenient: the install already
/// carries the conversion's INT4 `language_model.lm_head.weight`, and the
/// BF16 one is a second copy under a name no decode flow resolves.
///
/// Filtering the map cannot move a byte offset. `absolute_range` is
/// `data_region_start() + data_offsets`, and `data_region_start()` is
/// `8 + header_len` read off the file's length prefix, not derived from the
/// map's contents.
fn head_only(mut header: SafetensorsHeader) -> SafetensorsHeader {
    let before = header.tensors.len();
    let dropped: Vec<String> = header
        .tensors
        .keys()
        .filter(|k| !k.starts_with("mtp."))
        .cloned()
        .collect();
    // Named, not counted. A shard that grows a second non-head tensor is a
    // checkpoint change worth failing on, and `before - 15` would hide it.
    assert_eq!(
        dropped,
        vec!["lm_head.weight".to_string()],
        "{MTP_SHARD} no longer holds exactly the head plus a bare lm_head"
    );
    header.tensors.retain(|k, _| k.starts_with("mtp."));
    assert_eq!(header.tensors.len(), MTP_TENSORS.len());
    assert_eq!(before, MTP_TENSORS.len() + 1);
    header
}

#[test]
#[ignore = "streams the 16 GB mlx trunk plus the official checkpoint's 849 MB MTP head"]
fn repacks_the_real_qwen38_27b_checkpoint_with_its_mtp_head() {
    // 1. The trunk's index, at the same pins the headless test asserts.
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
    assert_eq!(shard_names.len(), 3);
    // The conversion still has no head of its own. If this fires, the trunk
    // and the official shard would both supply `mtp.*` and the merged
    // registry would silently prefer one -- so it is a precondition of the
    // fourth pair meaning anything, not a repeat of the other test.
    assert!(
        !weight_map.keys().any(|n| n.starts_with("mtp.")),
        "the mlx artifact grew an mtp head; the fourth pair would now collide"
    );

    // 2. The OFFICIAL index, which is a different repository and a different
    //    naming convention. Pinned separately for the same reason.
    let official_bytes = get_from(OFFICIAL_BASE, "model.safetensors.index.json");
    assert_eq!(
        model_io::hash_data(&official_bytes),
        OFFICIAL_INDEX_SHA256,
        "the official checkpoint's index does not match its pinned fingerprint"
    );
    let official: serde_json::Value =
        serde_json::from_slice(&official_bytes).expect("official index json");
    let official_map = official["weight_map"].as_object().expect("weight_map");
    for name in MTP_TENSORS {
        assert_eq!(
            official_map.get(name).and_then(|v| v.as_str()),
            Some(MTP_SHARD),
            "{name} is not in the shard this test reads"
        );
    }
    assert_eq!(
        official_map
            .keys()
            .filter(|n| n.starts_with("mtp."))
            .count(),
        MTP_TENSORS.len(),
        "the head's inventory moved"
    );

    // 3. Config and quantization come from the CONVERSION, not the official
    //    repo: the head is quantized here at repack time and the trunk's
    //    `quantization` block describes the trunk's own packed bytes.
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_qwen_gdn_dense_config(&config).expect("config parses");
    assert_eq!(arch, model_io::qwen_gdn_dense_27b());
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");

    // 4. Four sources: three trunk shards, then the official head shard.
    let mut sources: Vec<HttpRangeSource> = shard_names
        .iter()
        .map(|name| HttpRangeSource::new(format!("{REPO_BASE}/{name}")))
        .collect();
    sources.push(HttpRangeSource::new(format!("{OFFICIAL_BASE}/{MTP_SHARD}")));

    let mut headers: Vec<SafetensorsHeader> = sources
        .iter()
        .map(|s| fetch_safetensors_header(s).expect("shard header"))
        .collect();
    for (i, (name, header)) in shard_names.iter().zip(headers.iter()).enumerate() {
        let declared: u64 = header
            .tensors
            .values()
            .map(|t| t.data_offsets.1)
            .max()
            .expect("tensors")
            + header.data_region_start();
        assert_eq!(declared, SHARD_BYTES[i], "{name}: published size moved");
    }
    // The head shard's size is checked BEFORE the filter, because the filter
    // removes the tensor that ends the file.
    let head_header = headers.pop().expect("head shard header");
    let declared = head_header
        .tensors
        .values()
        .map(|t| t.data_offsets.1)
        .max()
        .expect("tensors")
        + head_header.data_region_start();
    assert_eq!(
        declared, MTP_SHARD_BYTES,
        "{MTP_SHARD}: published size moved"
    );
    let head_header = head_only(head_header);
    // Only 849 MB of that 3.39 GB shard is ever read, because the walk reads
    // per-tensor RANGES. Asserted so the cost claim in the module header is
    // derived rather than remembered.
    let head_bytes: u64 = head_header
        .tensors
        .values()
        .map(|t| t.data_offsets.1 - t.data_offsets.0)
        .sum();
    assert_eq!(head_bytes, MTP_TENSOR_BYTES, "the head's byte total moved");
    headers.push(head_header);

    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(sources.iter())
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect(),
    )
    .expect("no tensor name collides across shards");

    let dir = install_dir_from("TURBOSPARK_QWEN38_MTP_INSTALL_DIR", "turbospark-qwen38-mtp");
    eprintln!("installing to {}", dir.display());
    write_qwen_gdn_dense_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install");

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "vocab.json",
    ] {
        std::fs::write(dir.join(name), get(name)).expect("tokenizer sidecar");
    }

    // 5. Read it back. The head is the only thing this test asserts that the
    //    headless one does not, so everything else is a one-line control.
    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates against the production baseline");
    let resident_index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    // Every head tensor is present and nothing else `mtp.`-prefixed is.
    for name in MTP_TENSORS {
        assert!(
            resident_index.entries.contains_key(name),
            "resident index is missing {name}"
        );
    }
    assert_eq!(
        resident_index
            .entries
            .keys()
            .filter(|k| k.starts_with("mtp."))
            .count(),
        MTP_TENSORS.len(),
        "the install grew an mtp tensor the source does not have"
    );

    // **THIS IS THE WHOLE "does this install have a drafter" QUESTION.**
    // There is no manifest field and no flag, deliberately, so nothing can
    // disagree with the bytes.
    let fc = resident_index
        .entries
        .get("mtp.fc.weight")
        .expect("fc is resident");
    assert_eq!(
        fc.dtype, DTYPE_INT4_AFFINE,
        "the head's projections must be quantized: this engine dispatches no \
         unquantized GEMV, so a BF16 head fails at the first draft dispatch"
    );
    // Rank 2 quantizes, rank 1 narrows. Checked as the SPLIT rather than as
    // two independent facts, because the classifier keys on rank and a rank
    // it read wrong would move a tensor from one list to the other.
    for name in MTP_NORMS {
        let e = &resident_index.entries[name];
        assert_eq!(e.dtype, DTYPE_BF16, "{name} should be a narrowed norm");
        assert_eq!(e.scale_size, 0, "{name} carries quantization companions");
    }
    for name in MTP_TENSORS.iter().filter(|n| !MTP_NORMS.contains(n)) {
        let e = &resident_index.entries[*name];
        assert_eq!(e.dtype, DTYPE_INT4_AFFINE, "{name} should be quantized");
        assert!(e.scale_size > 0, "{name} has no scale plane");
    }

    // The bare `lm_head.weight` the filter dropped did NOT reach the install.
    assert!(
        !resident_index.entries.contains_key("lm_head.weight"),
        "the official shard's BF16 lm_head leaked in beside the INT4 one"
    );
    assert!(resident_index
        .entries
        .contains_key("language_model.lm_head.weight"));

    // The trunk is unchanged: still dense, still no vision tower.
    for marker in [".mlp.gate.weight", ".mlp.shared_expert", ".switch_mlp."] {
        assert!(!resident_index.entries.keys().any(|k| k.contains(marker)));
    }
    assert!(!resident_index
        .entries
        .keys()
        .any(|k| k.starts_with("vision_tower.")));
    let layout =
        model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("experts layout");
    assert_eq!(layout.layers.len(), 0, "a dense model streams nothing");

    eprintln!(
        "SUCCESS: Qwen3.8-27B + MTP head repacked into {}\n\
         The headless install is the CONTROL: run qwen38_quality_gate against \
         BOTH and require the same perplexity and digests.",
        dir.display()
    );
}

/// **ROADMAP M-V3's stage-2 gate: the real 333-tensor vision tower, packed.**
///
/// A SEPARATE install rather than a flag on `repacks_the_real_qwen38_27b_checkpoint`,
/// and the reason is that the headless install that test writes is the one
/// `qwen38_memory_oracle` and `qwen38_quality_gate` assert their frozen rows
/// against. Adding ~0.9 GiB of tower to it would move its footprint and force
/// a re-freeze for a component neither gate exercises. So the text-only walk
/// stays byte-identical and this writes its own directory.
///
/// It runs AFTER `crates/repack/tests/synthetic_qwen35_vision.rs` is green,
/// which is Gotcha 8's order and not a preference: every assertion below cost
/// milliseconds there and costs twenty-five minutes here.
///
/// **THE ONE THING ONLY THIS TEST CAN SEE is the dtype.** The fixture is F16
/// throughout, like the rest of the dense 1-bit install it hangs off; THIS
/// checkpoint's tower is **BF16**, so the conversion arm of
/// `convert_raw_to_fp16` runs here and its verbatim arm runs there.
/// `a_bf16_tower_converts_to_fp16_exactly_in_the_normal_range` proves the
/// arithmetic offline; what it cannot prove is that the real tower's values
/// all sit inside FP16's range, and an overflow is a hard refusal by design.
#[test]
#[ignore = "streams the real 16 GB Qwen3.8-27B 4-bit checkpoint plus its 0.9 GiB vision tower"]
fn repacks_the_real_qwen38_27b_checkpoint_with_its_vision_tower() {
    let index_bytes = get("model.safetensors.index.json");
    assert_eq!(
        model_io::hash_data(&index_bytes),
        PINNED_INDEX_SHA256,
        "model.safetensors.index.json does not match the pinned fingerprint"
    );
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    let weight_map = index["weight_map"].as_object().expect("weight_map");

    // The tower is PRESENT in this artifact. Ornith's MLX conversion declares
    // the identical `vision_config` and ships none, so "the config says 27
    // blocks" is not evidence that the bytes are here.
    let vision_names: Vec<&String> = weight_map
        .keys()
        .filter(|n| n.starts_with("vision_tower."))
        .collect();
    assert_eq!(
        vision_names.len(),
        333,
        "the vision tower's inventory moved"
    );

    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let mut arch = parse_qwen_gdn_dense_config(&config).expect("config parses");
    // The TRUNK still parses to the pinned baseline, which is what says
    // ingesting the tower changed no text field.
    assert_eq!(arch, model_io::qwen_gdn_dense_27b());

    let vision = turbospark_repack::parse_vision_config(&config).expect("vision_config parses");
    // Read off the published file, cross-checked in `docs/VISION_PHASE0.md`
    // item 1. `intermediate_size` is the one worth naming: the planning table
    // guessed 4608 by analogy with the merger's width and it is 4304.
    assert_eq!(vision.depth, 27);
    assert_eq!(vision.hidden_size, 1152);
    assert_eq!(vision.intermediate_size, 4304, "NOT 4608");
    assert_eq!(vision.num_heads, 16);
    assert_eq!(vision.head_dim(), 72);
    assert_eq!(
        vision.out_hidden_size, arch.hidden_size,
        "the merger writes into the trunk"
    );
    assert_eq!(vision.num_position_embeddings, 2304, "a 48x48 grid");
    assert_eq!(vision.mrope_section, [11, 11, 10]);
    arch.vision = vision.clone();

    // The role table accounts for the whole tower: 27 blocks x 12 roles plus
    // the 9 non-block tensors is exactly 333. Arithmetic rather than a
    // comment, and it is what would catch a publisher adding a tensor.
    assert_eq!(
        vision.depth as usize * turbospark_repack::VISION_BLOCK_ROLES.len()
            + turbospark_repack::VISION_RESIDENT_TENSORS.len(),
        vision_names.len(),
        "the role table does not account for the published tower"
    );

    let quant = parse_gemma4_quantization(&config).expect("quantization parses");
    let shard_names: BTreeSet<String> = weight_map
        .values()
        .map(|v| v.as_str().expect("shard name").to_string())
        .collect();
    let sources: Vec<HttpRangeSource> = shard_names
        .iter()
        .map(|name| HttpRangeSource::new(format!("{REPO_BASE}/{name}")))
        .collect();
    let headers = sources
        .iter()
        .map(|s| fetch_safetensors_header(s).expect("shard header"))
        .collect::<Vec<_>>();

    // THE DTYPE, off the header before a single weight byte moves. This
    // checkpoint's tower is BF16 where Bonsai's is F16, which is the whole
    // reason `convert_raw_to_fp16` has two arms.
    let tower_dtypes: BTreeSet<&str> = headers
        .iter()
        .flat_map(|h| h.tensors.iter())
        .filter(|(n, _)| n.starts_with("vision_tower."))
        .map(|(_, t)| t.dtype.as_str())
        .collect();
    assert_eq!(
        tower_dtypes,
        BTreeSet::from(["BF16"]),
        "this checkpoint's tower is BF16; a change here moves which arm runs"
    );

    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(sources.iter())
            .map(|(h, s)| (h, s as &dyn turbospark_repack::RangeSource))
            .collect(),
    )
    .expect("no tensor name collides across shards");

    let dir = install_dir_from(
        "TURBOSPARK_QWEN38_VISION_INSTALL_DIR",
        "turbospark-qwen38-vision",
    );
    eprintln!("installing to {}", dir.display());
    write_qwen_gdn_dense_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install with a vision tower");

    // The manifest declares the tower and loads with it.
    let manifest = model_io::load_manifest(&dir, &arch, 4 * 1024 * 1024)
        .expect("the vision install's manifest loads");
    for f in ["packed_vision/layout.json", "packed_vision/blobs.bin"] {
        assert!(manifest.files.contains_key(f), "the manifest omits {f}");
    }

    // The packed side: 27 blocks, addressed through the packed-experts loader.
    let layout = model_io::load_packed_layout_from(
        &dir,
        model_io::PACKED_VISION_DIR,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("packed_vision/layout.json decodes");
    assert_eq!(layout.num_layers, 1);
    assert_eq!(layout.experts_per_layer, vision.depth as usize);
    let blocks = &layout.layers[0];
    assert_eq!(blocks.experts.len(), 27);

    // The per-block stride is what the shapes predict, which is the check the
    // fixture's arithmetic stands in for. Twelve tensors at FP16: two norms
    // and their biases, a fused qkv and its bias, a proj and its bias, and the
    // two MLP matrices with theirs.
    let h = vision.hidden_size as u64;
    let i = vision.intermediate_size as u64;
    let expected_block_bytes = 2
        * (
            // norm1 w+b, norm2 w+b
            4 * h
        // qkv [3h, h] + bias [3h]
        + 3 * h * h + 3 * h
        // proj [h, h] + bias [h]
        + h * h + h
        // fc1 [i, h] + bias [i]
        + i * h + i
        // fc2 [h, i] + bias [h]
        + h * i + h
        );
    // Asserted EXACTLY rather than as a range, which is both stronger and the
    // rule the writer actually applies: the stride is that byte count rounded
    // up to `GTURBO_PAGE_BYTES`. Taken from the constant rather than a literal
    // -- this format pages to 16 KiB, not to the OS's 4 KiB, and a hardcoded
    // 4096 here reads as a plausible near-miss (measured: 30,490,624 against
    // 30,479,008, an 11,616-byte gap that is one 16 KiB page and not three
    // 4 KiB ones).
    let page = turbospark_repack::GTURBO_PAGE_BYTES;
    assert_eq!(
        blocks.expert_stride,
        expected_block_bytes.div_ceil(page) * page,
        "stride {} is not the {expected_block_bytes} bytes a block needs, \
         rounded up to the {page}-byte page",
        blocks.expert_stride
    );

    // The resident side: nine tensors, all FP16 (tag 2), beside a trunk whose
    // norms are BF16 (tag 1). Both rules in one index is the thing that could
    // quietly stop being true.
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    for suffix in turbospark_repack::VISION_RESIDENT_TENSORS {
        let name = format!("vision.{suffix}");
        let entry = resident
            .entries
            .get(&name)
            .unwrap_or_else(|| panic!("{name} is not in the resident index"));
        assert_eq!(entry.dtype, 2, "{name} should be FP16");
    }
    assert_eq!(
        resident
            .entries
            .get("language_model.model.norm.weight")
            .expect("the trunk's final norm")
            .dtype,
        DTYPE_BF16,
        "the trunk's norms must still narrow to BF16"
    );

    eprintln!(
        "vision install at {} ({} blocks, {} bytes/block)",
        dir.display(),
        blocks.experts.len(),
        blocks.expert_stride
    );
}
