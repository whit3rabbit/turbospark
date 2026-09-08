//! Network-gated proof that the `muse_glimmer` repack path works against the
//! real published checkpoint, `mlx-community/Muse-Glimmer-30B-4bit`
//! (19.41 GB across four shards, pinned to one commit and the index's
//! SHA-256). Not run by default:
//!
//! ```sh
//! TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
//!   cargo test -p turbospark-repack --test museglimmer_checkpoint_network --release -- --ignored --nocapture
//! ```
//!
//! **WHAT IS LEFT TO PROVE HERE IS THE WALK, and almost nothing else**, which
//! is the point of having run `tests/museglimmer_config.rs` and
//! `tests/synthetic_muse.rs` first. Those two already assert offline, in
//! milliseconds, that the config parses to the pinned baseline and that the
//! walk writes a loadable dense install with no packed experts and the right
//! tensor set. `crates/repack` Gotcha 8 is the standing reason to do it in
//! that order: M4's dense `llama` half found three manifest-slot bugs one
//! five-minute re-stream at a time, because no fixture had ever asked what
//! the walk writes when `plan.routed` is empty.
//!
//! Two things this DOES cover that no fixture can:
//!
//! 1. **FOUR SHARDS.** The synthetic fixture is one blob, so a companion
//!    tensor (`.scales` / `.biases`) living in a different shard from its
//!    weight is unreachable there. `Gemma4Shards`' merged registry is what
//!    handles it.
//! 2. **THE EXCLUSION LIST AGAINST REAL NAMES.** This checkpoint splits its
//!    vision side THREE ways -- `vision_tower.` (806 tensors),
//!    `vision_adapter.` (6) and `vision_projection.` (3) -- where Gemma keeps
//!    it under one prefix. An unlisted prefix falls through to
//!    `Gemma4Bucket::Unknown`, which the walk refuses, and nothing sees that
//!    until the stream reaches the shard the tensor lives in. The counts
//!    below were enumerated from the published index before any bytes moved,
//!    and asserting them here is what keeps a re-upload from silently
//!    changing the shape of the text tower.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbospark_repack::{
    fetch_safetensors_header, parse_gemma4_quantization, parse_muse_glimmer_config,
    parse_muse_glimmer_scalars, write_muse_glimmer_install_streamed, Gemma4Shards, HttpRangeSource,
};

const REPO_BASE: &str = "https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit/resolve/3e7677d7a40d348a3daba263a2b1c0aa41910710";
const MODEL_ID: &str = "mlx-community/Muse-Glimmer-30B-4bit";
/// Pinned at the commit above. The index names all 2,278 tensors and every
/// shard is reachable from it, so this one hash fingerprints the download.
const PINNED_INDEX_SHA256: &str =
    "10e1f963a4dc827c303594d988c6d44711171eaa70eeb3e810c2ce321d4c0323";
/// Per-shard published byte sizes, asserted so a silently re-uploaded
/// artifact fails HERE rather than at some tensor offset twenty minutes in.
const SHARD_BYTES: [u64; 4] = [5_310_085_129, 5_366_299_665, 5_343_216_475, 3_395_202_844];
/// `metadata.total_size` from the index.
const TOTAL_SIZE: u64 = 19_414_521_856;

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
    match std::env::var_os("TURBOSPARK_MUSEGLIMMER_INSTALL_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create install dir");
            dir
        }
        None => {
            let dir =
                std::env::temp_dir().join(format!("turbospark-muse-real-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }
    }
}

#[test]
#[ignore = "streams the real 19.4 GB Muse-Glimmer-30B 4-bit checkpoint over the network"]
fn repacks_the_real_muse_glimmer_30b_checkpoint() {
    let index_bytes = get("model.safetensors.index.json");
    assert_eq!(
        model_io::hash_data(&index_bytes),
        PINNED_INDEX_SHA256,
        "model.safetensors.index.json does not match the pinned fingerprint"
    );
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    assert_eq!(
        index["metadata"]["total_size"].as_u64(),
        Some(TOTAL_SIZE),
        "the published total size moved"
    );
    let weight_map = index["weight_map"].as_object().expect("weight_map");
    let shard_names: BTreeSet<String> = weight_map
        .values()
        .map(|v| v.as_str().expect("shard name").to_string())
        .collect();
    eprintln!("shards: {shard_names:?}");
    assert_eq!(shard_names.len(), 4, "expected a four-shard checkpoint");

    // The tensor inventory, off the index rather than assumed.
    assert_eq!(weight_map.len(), 2_278, "the tensor inventory moved");
    let count = |prefix: &str| weight_map.keys().filter(|n| n.starts_with(prefix)).count();
    assert_eq!(count("language_model."), 1_463, "the text tower moved");
    // THE THREE VISION PREFIXES. Each must be in `classify_for_family`'s
    // exclusion list or the walk refuses mid-stream.
    assert_eq!(count("vision_tower."), 806);
    assert_eq!(count("vision_adapter."), 6);
    assert_eq!(count("vision_projection."), 3);
    let known = count("language_model.")
        + count("vision_tower.")
        + count("vision_adapter.")
        + count("vision_projection.");
    assert_eq!(
        known,
        weight_map.len(),
        "the checkpoint grew a prefix the walk has no rule for; it would be refused as \
         Unknown twenty minutes into the stream"
    );
    // 52 layers x 28 tensors + 7 model-level, which is the arithmetic that
    // says the text tower is the shape the baseline declares.
    assert_eq!(52 * 28 + 7, 1_463);

    // Config: the full arch, cross-checked against the pinned baseline. The
    // same assertion runs offline in `tests/museglimmer_config.rs`; failing
    // HERE and passing there means the upstream checkpoint moved, not that
    // the parser broke.
    let config = String::from_utf8(get("config.json")).expect("config utf8");
    let arch = parse_muse_glimmer_config(&config).expect("config parses");
    assert_eq!(
        arch,
        model_io::muse_glimmer_30b(),
        "parsed config does not match the pinned muse_glimmer baseline"
    );
    assert_eq!(arch.num_experts, 0, "this family is DENSE");
    assert_eq!(arch.full_rope_theta, 0.0, "the full layers are NoPE");

    // The four scalars the ArchConfig deliberately does not carry, against
    // the decode flow's constants. Offline too, and repeated here because
    // this is the run that reads the LIVE file rather than a transcription.
    let scalars = parse_muse_glimmer_scalars(&config).expect("scalars parse");
    assert_eq!(scalars.qk_scale_factor, 3.87);
    assert_eq!(scalars.output_multiplier, 0.19611613513818404);
    assert_eq!(scalars.rms_norm_eps, 1e-5);
    assert_eq!(scalars.post_norm_eps, 1e-8);

    // The ORDINARY INT4 affine at group 64. No new width, unlike the two
    // sub-4-bit checkpoints, so `is_supported_affine_shape` needed nothing.
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");
    assert_eq!(quant.default_bits, 4);
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
    write_muse_glimmer_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install");

    // Tokenizer sidecars so turbospark-check can open the dir directly.
    //
    // READ OFF THIS REPO'S OWN FILE LIST, never carried from a sibling
    // (`crates/catalog` Gotcha 7, and AGENTS.md Gotcha 47's 20-minute
    // lesson). It ships `chat_template.jinja` as a STANDALONE file -- the
    // newer of the two places HF puts a template -- and a
    // `generation_config.json` whose `eos_token_id` is the LIST
    // [200001, 200008] the stop set unions. It ships NO `merges.txt`.
    // `processor_config.json` exists and is deliberately not fetched: it
    // configures the VISION preprocessor, and this port ingests the text
    // tower only.
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        std::fs::write(dir.join(name), get(name)).expect("write sidecar");
    }

    let manifest = dir.join("manifest.json");
    assert!(manifest.exists(), "manifest.json written");
    // DENSE: zero packed-expert blobs. The routed marker never fires, and an
    // install that grew one would mean `classify_for_family` matched
    // something it should not have.
    let expert_files = std::fs::read_dir(dir.join("packed_experts"))
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| e.file_name().to_string_lossy().ends_with(".bin"))
                .count()
        })
        .unwrap_or(0);
    assert_eq!(expert_files, 0, "a dense install writes no expert blobs");

    let resident = std::fs::metadata(dir.join("model_weights.bin"))
        .expect("model_weights.bin")
        .len();
    eprintln!("resident region: {resident} bytes");
}
