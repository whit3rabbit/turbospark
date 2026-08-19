//! Network-gated proof that the DFlash2 drafter (`docs/DFLASH2.md`) streams
//! into a `qwen3_5` install beside the mlx trunk: the THIRD repository this
//! family's installs are assembled from. Not run by default:
//!
//! ```sh
//! TURBOSPARK_QWEN38_DFLASH2_INSTALL_DIR=~/models/qwen38-27b-dflash2.gturbo \
//!   cargo test -p turbospark-repack --test dflash2_checkpoint_network --release -- --ignored --nocapture
//! ```
//!
//! The trunk comes from `mlx-community/Qwen3.8-27B-4bit` at the same pins
//! `qwen38_checkpoint_network.rs` asserts; the drafter comes from
//! `incoai/Qwen3.8-27B-DFlash2` at a REVISION, because an install test
//! writes GB off these bytes and has to name which ones.
//!
//! **THE DRAFTER'S NAMES ARE RENAMED ONTO `dflash.` HERE, in the shard
//! header, before the walk sees them** (`classify::DFLASH_PREFIX`). The
//! published repository spells its tensors bare (`layers.0.*`, `fc.weight`),
//! which no classifier arm could match; renaming the header's map cannot
//! move a byte offset, for the same reason `head_only`'s filter in the MTP
//! test cannot (offsets are resolved against the file's own length prefix).
//!
//! The fixture that found every walk hole BEFORE this stream ran is
//! `synthetic_qwen35.rs`'s DFlash2 block (`crates/repack` Gotcha 8).

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_qwen_gdn_dense_config, write_qwen_gdn_dense_install_streamed,
    Gemma4Shards, HttpRangeSource, RangeSource, SafetensorsHeader, DFLASH_PREFIX,
};

const REPO_BASE: &str = "https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/resolve/3e6447f082e89cc7f0bc6e5441afd38dfce760ff";
const MODEL_ID: &str = "mlx-community/Qwen3.8-27B-4bit";
const PINNED_INDEX_SHA256: &str =
    "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8";
const SHARD_BYTES: [u64; 3] = [5_343_268_662, 5_354_185_130, 5_357_087_557];

/// `incoai/Qwen3.8-27B-DFlash2`, pinned at a revision rather than `main`.
const DFLASH_BASE: &str =
    "https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/dedf8df68adfb1afeaf7b7480c0a0243108177b4";
const DFLASH_MODEL_ID: &str = "incoai/Qwen3.8-27B-DFlash2";
/// The drafter is one file: 8 (length prefix) + 8,928 (header) +
/// 3,848,808,960 (data), asserted so a re-upload fails before a 16 GB
/// trunk stream starts, not after it.
const DFLASH_FILE_BYTES: u64 = 3_848_817_896;
const DFLASH_TENSOR_BYTES: u64 = 3_848_808_960;
/// The published inventory: 6 top-level tensors plus 15 per layer over 5
/// layers (`docs/DFLASH2.md`).
const DFLASH_TENSORS: usize = 81;

const DTYPE_INT4_AFFINE: u8 = 4;
const DTYPE_BF16: u8 = 1;

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
    match std::env::var_os("TURBOSPARK_QWEN38_DFLASH2_INSTALL_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create install dir");
            dir
        }
        None => {
            let dir = std::env::temp_dir()
                .join(format!("turbospark-qwen38-dflash2-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }
    }
}

/// Renames the drafter header's BARE names onto `dflash.`, asserting the
/// inventory while it is at it: every tensor the published file carries is
/// one this walk has a rule for, so a repo that grows a tensor fails HERE
/// with its name rather than as an `UnknownTensor` mid-stream.
fn dflash_namespaced(mut header: SafetensorsHeader) -> SafetensorsHeader {
    assert_eq!(
        header.tensors.len(),
        DFLASH_TENSORS,
        "the published drafter's inventory moved"
    );
    let mut out = std::collections::BTreeMap::new();
    for (name, info) in std::mem::take(&mut header.tensors) {
        out.insert(format!("{DFLASH_PREFIX}{name}"), info);
    }
    header.tensors = out;
    header
}

#[test]
#[ignore = "streams the 16 GB mlx trunk plus the 3.8 GB DFlash2 drafter"]
fn repacks_the_qwen38_trunk_with_the_dflash2_drafter() {
    // 1. The trunk, at the same pins the headless test asserts.
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
    assert!(
        !weight_map.keys().any(|n| n.starts_with("mtp.")),
        "the mlx artifact grew an mtp head; this install carries the DFlash2 \
         drafter instead and has no use for one"
    );
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_qwen_gdn_dense_config(&config).expect("config parses");
    assert_eq!(arch, model_io::qwen_gdn_dense_27b());
    let quant = turbospark_repack::parse_gemma4_quantization(&config).expect("quant parses");

    // 2. The drafter's own config, fetched from ITS repo and asserted on the
    //    five fields the runtime's conventions hang off. This is the
    //    cheap-headers discipline: if the checkpoint moves, this fails in
    //    KB, not after the trunk stream.
    let dflash_config_url = format!("{DFLASH_BASE}/config.json");
    let dconfig_bytes = reqwest::blocking::Client::new()
        .get(&dflash_config_url)
        .send()
        .expect("drafter config fetch")
        .bytes()
        .expect("drafter config body")
        .to_vec();
    let dconfig: serde_json::Value =
        serde_json::from_slice(&dconfig_bytes).expect("drafter config json");
    assert_eq!(
        dconfig["dflash_config"]["block_size"], 8,
        "block_size moved"
    );
    assert_eq!(
        dconfig["dflash_config"]["conv_kernel_size"], 2,
        "conv_kernel_size moved"
    );
    assert_eq!(
        dconfig["dflash_config"]["conv_group_size"], 16,
        "conv_group_size moved"
    );
    assert_eq!(
        dconfig["dflash_config"]["selector_top_k"], 16,
        "selector_top_k moved"
    );
    assert_eq!(
        dconfig["dflash_config"]["target_layer_ids"],
        serde_json::json!([5, 19, 33, 47, 61]),
        "target_layer_ids moved, and the aux-capture taps are keyed on them"
    );
    assert_eq!(
        dconfig["dflash_config"]["mask_token_id"], 248_070,
        "mask_token_id moved"
    );

    // 3. Sources: three trunk shards, then the drafter's single file.
    let mut sources: Vec<HttpRangeSource> = shard_names
        .iter()
        .map(|name| HttpRangeSource::new(format!("{REPO_BASE}/{name}")))
        .collect();
    sources.push(HttpRangeSource::new(format!(
        "{DFLASH_BASE}/model.safetensors"
    )));

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
    // The drafter's size is checked BEFORE the rename, for the filter
    // precedent's reason: the checks describe the FILE, not the map.
    let mut dheader = headers.pop().expect("drafter header");
    let declared: u64 = dheader
        .tensors
        .values()
        .map(|t| t.data_offsets.1)
        .max()
        .expect("tensors")
        + dheader.data_region_start();
    assert_eq!(
        declared, DFLASH_FILE_BYTES,
        "the drafter's published file size moved"
    );
    let tensor_bytes: u64 = dheader
        .tensors
        .values()
        .map(|t| t.data_offsets.1 - t.data_offsets.0)
        .sum();
    assert_eq!(
        tensor_bytes, DFLASH_TENSOR_BYTES,
        "the drafter's data moved"
    );
    dheader = dflash_namespaced(dheader);
    headers.push(dheader);

    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(sources.iter())
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect(),
    );

    // 4. The install. The trunk still streams a layer at a time; the
    //    drafter's 3.8 GB rides the resident region's walk.
    let dir = install_dir();
    eprintln!("installing to {}", dir.display());
    write_qwen_gdn_dense_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install");
    let _ = DFLASH_MODEL_ID;

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "vocab.json",
    ] {
        std::fs::write(dir.join(name), get(name)).expect("tokenizer sidecar");
    }

    // 5. Read it back. The drafter is the only thing this test asserts that
    //    the headless qwen38 test does not.
    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates against the production baseline");
    let resident_index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    let dflash: Vec<&String> = resident_index
        .entries
        .keys()
        .filter(|k| k.starts_with("dflash."))
        .collect();
    assert_eq!(
        dflash.len(),
        DFLASH_TENSORS,
        "the install does not carry exactly the drafter: {} tensors",
        dflash.len()
    );

    // The dtype split, as the fixture pinned it: codebooks and base kernels
    // raw BF16, projections INT4, norms BF16.
    for name in [
        "dflash.candidate_selector.predecessor_codebook",
        "dflash.candidate_selector.successor_codebook",
        "dflash.layers.0.attention_conv.base_kernel",
    ] {
        assert_eq!(resident_index.entries[name].dtype, DTYPE_BF16, "{name}");
        assert_eq!(
            resident_index.entries[name].scale_size, 0,
            "{name} carries quantization companions"
        );
    }
    for name in [
        "dflash.fc.weight",
        "dflash.candidate_selector.hidden_projection.weight",
        "dflash.layers.0.self_attn.q_proj.weight",
        "dflash.layers.0.mlp.down_proj.weight",
        "dflash.layers.0.attention_conv.kernel_projection.weight",
    ] {
        assert_eq!(
            resident_index.entries[name].dtype, DTYPE_INT4_AFFINE,
            "{name}"
        );
        assert!(
            resident_index.entries[name].scale_size > 0,
            "{name} has no scale plane"
        );
    }
    // fc's width carries the aux-state count: [5120, 5 * 5120].
    let fc = &resident_index.entries["dflash.fc.weight"];
    assert_eq!((fc.shape.0, fc.shape.1), (5120, 25_600), "fc's shape moved");

    // No MTP head beside the drafter: they are alternative drafters and
    // this install asked for one of them.
    assert!(
        !resident_index.entries.keys().any(|k| k.starts_with("mtp.")),
        "an mtp head rode in with the drafter"
    );
    let layout =
        model_io::load_packed_experts_layout(&dir, 64 * 1024 * 1024).expect("experts layout");
    assert_eq!(layout.layers.len(), 0, "a dense model streams nothing");

    eprintln!(
        "SUCCESS: Qwen3.8-27B + DFlash2 drafter repacked into {}",
        dir.display()
    );
}
