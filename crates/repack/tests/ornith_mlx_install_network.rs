//! Streams `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` into an INT4-affine
//! install -- the artifact that can SPECULATE, where the GGUF one cannot.
//!
//! ```sh
//! TURBOSPARK_ORNITH35B_INSTALL_DIR=~/models/ornith35b.gturbo \
//!   cargo test -p turbospark-repack --test ornith_mlx_install_network --release -- --ignored --nocapture
//! ```
//!
//! **THE POINT OF THIS INSTALL IS `MtpState::speculation_blocker`'S DTYPE
//! ARM.** `encode_gemm_any` has one arm, MLX affine INT4, so no GGUF install
//! of any family can ever speculate whatever head it acquires -- which is why
//! `~/models/ornith35b-gguf.gturbo` exists and is not enough. This one is the
//! same model at the same architecture in the dtype the batched verify
//! dispatches.
//!
//! **AND IT NEEDED NO NEW WALK, WHICH THE PLAN DID NOT EXPECT.** The route
//! that was scoped was the 71.90 GB BF16 repo, whose four naming differences
//! (`model.language_model.` for the trunk, a bare `lm_head.`, `model.visual.`
//! for the vision tower, `.mlp.experts.` for the routed marker) and FUSED
//! `gate_up_proj` would each have been a new code path, plus a quantizing arm
//! over every resident tensor and every expert. `ornith-ai` publish this
//! conversion themselves and it is the mlx-community shape exactly:
//! `language_model.model.layers.N.`, `.mlp.switch_mlp.` with gate/up/down
//! SPLIT, and no vision tensors at all. So it is `qwen36_checkpoint_network`'s
//! walk pointed at another repository.
//!
//! **THE HEAD IS NOT HERE.** This conversion carries ZERO `mtp.*` tensors,
//! exactly as `mlx-community/Qwen3.8-27B-4bit` drops Qwen3.8's
//! (`crates/repack/CLAUDE.md` on `gemma4_checkpoint/mtp.rs` -- the same
//! observation now holding for a second publisher). So this install runs and
//! does not draft, and the head has to be read cross-repo out of the BF16
//! repo's shard 16, which is its own step. `speculation_blocker` refuses on
//! `num_experts != 0` first regardless, so nothing here is blocked ON the
//! head.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_gemma4_quantization, parse_qwen_gdn_moe_config,
    write_qwen_gdn_moe_install_streamed, Gemma4Shards, HttpRangeSource,
};

const REPO_BASE: &str = "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit/resolve/19504d912fa8fc7622bf6b1de3db5d5d890b1f02";
const MODEL_ID: &str = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit";

/// Pinned at the commit above, so this one hash fingerprints every tensor
/// reachable from it (repack Gotcha 5).
const PINNED_INDEX_SHA256: &str =
    "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36";

/// The four published shard sizes, from `x-linked-size`. 19.51 GB in total
/// against the BF16 repo's 71.90 -- the whole reason this route is cheaper in
/// wall clock as well as in code.
const SHARD_BYTES: [u64; 4] = [5_343_699_466, 5_368_472_837, 5_368_324_245, 3_428_527_653];

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
    let dir = match std::env::var_os("TURBOSPARK_ORNITH35B_INSTALL_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join(format!("turbospark-ornith35b-{}", std::process::id())),
    };
    std::fs::create_dir_all(&dir).expect("create install dir");
    dir
}

#[test]
#[ignore = "network: streams the real 19.5 GB Ornith-1.5-35B-A3B MLX 4-bit checkpoint"]
fn repacks_the_real_ornith_35b_mlx_4bit() {
    let index_bytes = get("model.safetensors.index.json");
    assert_eq!(
        model_io::hash_data(&index_bytes),
        PINNED_INDEX_SHA256,
        "model.safetensors.index.json does not match the pinned fingerprint"
    );
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    let weight_map = index["weight_map"].as_object().expect("weight_map");

    // NO VISION AND NO HEAD, both asserted rather than assumed. The first is
    // why this walk needs no `ExcludedMultimodal` arm to fire; the second is
    // the fact that decides whether this install can draft.
    assert!(
        !weight_map.keys().any(|k| k.contains("visual")),
        "this conversion is text-only; a vision tower here means it was re-cut"
    );
    let mtp: Vec<&String> = weight_map
        .keys()
        .filter(|k| k.starts_with("mtp."))
        .collect();
    assert!(
        mtp.is_empty(),
        "this conversion drops the MTP head, like mlx-community's Qwen3.8 one; \
         {} tensors appeared, which would be a NEW and better artifact",
        mtp.len()
    );

    let shard_names: BTreeSet<String> = weight_map
        .values()
        .map(|v| v.as_str().expect("shard name").to_string())
        .collect();
    let shard_names: Vec<String> = shard_names.into_iter().collect();
    assert_eq!(shard_names.len(), 4, "four shards");
    eprintln!("index: {} tensors, 4 shards", weight_map.len());

    let arch = parse_qwen_gdn_moe_config(&String::from_utf8(get("config.json")).expect("utf8"))
        .expect("config parses");
    // The same equality `ornith_config.rs` asserts offline, now against the
    // REAL file rather than a vendored body.
    assert_eq!(
        arch,
        model_io::qwen_gdn_moe_35b_a3b(),
        "the MLX 4-bit conversion must derive the pinned qwen3_5_moe baseline"
    );

    let quant = parse_gemma4_quantization(&String::from_utf8(get("config.json")).expect("utf8"))
        .expect("quantization parses");
    assert_eq!(
        quant.default_bits, 4,
        "the batched verify dispatches INT4 alone"
    );
    assert_eq!(quant.group_size, 64);

    let sources: Vec<HttpRangeSource> = shard_names
        .iter()
        .map(|name| HttpRangeSource::new(format!("{REPO_BASE}/{name}")))
        .collect();
    let headers = sources
        .iter()
        .map(|s| fetch_safetensors_header(s).expect("shard header"))
        .collect::<Vec<_>>();
    // The published sizes, checked before 19.5 GB moves rather than after.
    for (i, header) in headers.iter().enumerate() {
        let last = header
            .tensors
            .values()
            .map(|t| t.data_offsets.1)
            .max()
            .expect("shard holds tensors");
        assert_eq!(
            header.data_region_start() + last,
            SHARD_BYTES[i],
            "shard {} does not end at its published size",
            i + 1
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
    write_qwen_gdn_moe_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install");

    // ChatML dialect, so a raw `--prompt` babbles; `--messages-file` renders.
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        std::fs::write(dir.join(name), get(name)).expect("tokenizer sidecar");
    }

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates against the production baseline");
    let resident =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");

    for name in [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        // Layer 0 is LINEAR and has no self_attn at all.
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.model.layers.0.linear_attn.dt_bias",
        // Layer 3 is the first FULL-attention layer.
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.layers.3.self_attn.k_norm.weight",
        "language_model.model.layers.39.mlp.shared_expert.down_proj.weight",
        "language_model.model.layers.39.mlp.shared_expert_gate.weight",
    ] {
        assert!(
            resident.entries.contains_key(name),
            "resident index is missing {name}"
        );
    }

    // **THE DTYPE THAT DECIDES SPECULATION**, read off a tensor the batched
    // verify really dispatches rather than off the manifest -- which is the
    // same place `speculation_blocker` reads it, and for the same reason.
    const BATCHED_GEMM_DTYPE: u8 = 4;
    let q_proj = &resident.entries["language_model.model.layers.3.self_attn.q_proj.weight"];
    assert_eq!(
        q_proj.dtype, BATCHED_GEMM_DTYPE,
        "q_proj must be MLX affine INT4 or the batched verify cannot dispatch it"
    );

    // The router is lifted to 8 bits by the checkpoint's own override table,
    // and `validate_quant` accepts 8 ONLY -- so a change upstream would fail
    // at load with a far less obvious message than this one.
    for layer in [0usize, 39] {
        let gate =
            &resident.entries[&format!("language_model.model.layers.{layer}.mlp.gate.weight")];
        assert_eq!(gate.dtype, 5, "layer {layer} router must be INT8");
    }

    // 40 routed layers, one blob file each, and NO 41st: this checkpoint's
    // config declares `mtp_num_hidden_layers: 1` and the walk must not turn
    // that claim into a layer.
    let layer_files = std::fs::read_dir(dir.join("packed_experts"))
        .expect("packed_experts")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("layer_"))
        .count();
    assert_eq!(layer_files, 40, "40 trunk layers of experts");

    // No head reached the install, which is the claim the index check above
    // makes about the SOURCE, now made about the ARTIFACT.
    assert!(
        !resident.entries.keys().any(|k| k.starts_with("mtp.")),
        "this conversion carries no head; `mtp.fc.weight` here would mean one \
         arrived from somewhere unexamined"
    );

    eprintln!("installed {} resident tensors", resident.entries.len());
}
