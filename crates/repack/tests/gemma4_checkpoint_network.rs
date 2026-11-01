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
    classify_for_family, fetch_safetensors_header, parse_gemma4_config, parse_gemma4_quantization,
    parse_gemma4_vision_config, write_gemma4_install_streamed, Gemma4Bucket, Gemma4Shards,
    HttpRangeSource, EMBED_VISION_PREFIX, GEMMA4_BLOCK_ROLES, GEMMA4_RESIDENT_TENSORS,
    VISION_PREFIX,
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

/// Header-only: the real checkpoint's 358 vision names all map onto the
/// Gemma 4 tower's tables, with the dtypes and shapes the ingest expects.
/// Fetches `config.json`, the index, and one shard's header, no weights.
#[test]
#[ignore = "fetches the real Gemma 4 config, index and one shard header over the network"]
fn the_real_vision_tower_maps_onto_the_gemma4_tables() {
    let config = String::from_utf8(get("config.json")).expect("utf8 config");
    let vision = parse_gemma4_vision_config(&config).expect("vision_config parses");
    assert_eq!(vision.tower, model_io::VisionTowerKind::Gemma4);
    let arch = parse_gemma4_config(&config).expect("text config parses");
    assert_eq!(vision.out_hidden_size, arch.hidden_size);

    let index_bytes = get("model.safetensors.index.json");
    assert_eq!(model_io::hash_data(&index_bytes), PINNED_INDEX_SHA256);
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    let map = index["weight_map"].as_object().expect("weight_map");
    let names: Vec<&str> = map
        .keys()
        .map(String::as_str)
        .filter(|n| !n.starts_with("language_model."))
        .collect();
    assert_eq!(names.len(), 358);
    for n in &names {
        assert_eq!(
            classify_for_family(n, arch.num_layers as usize, model_io::ModelFamily::Gemma4),
            Gemma4Bucket::VisionTower,
            "{n}"
        );
    }

    let block = format!("{VISION_PREFIX}encoder.layers.");
    let mut blocks = vec![BTreeSet::new(); vision.depth as usize];
    let mut residents = BTreeSet::new();
    let mut projector = BTreeSet::new();
    for n in &names {
        if let Some(rest) = n.strip_prefix(&block) {
            let (i, suffix) = rest.split_once('.').expect("index.suffix");
            let i: usize = i.parse().expect("block index");
            assert!(GEMMA4_BLOCK_ROLES.iter().any(|(_, s)| *s == suffix), "{n}");
            blocks[i].insert(suffix.to_string());
        } else if let Some(tail) = n.strip_prefix(VISION_PREFIX) {
            assert!(GEMMA4_RESIDENT_TENSORS.contains(&tail), "{n}");
            residents.insert(tail.to_string());
        } else {
            projector.insert(
                n.strip_prefix(EMBED_VISION_PREFIX)
                    .expect("projector")
                    .to_string(),
            );
        }
    }
    assert!(blocks.iter().all(|b| b.len() == GEMMA4_BLOCK_ROLES.len()));
    assert_eq!(residents.len(), GEMMA4_RESIDENT_TENSORS.len());
    assert_eq!(projector.len(), 3);

    let shard = map[names[0]].as_str().expect("shard name");
    assert!(
        names.iter().all(|n| map[*n].as_str() == Some(shard)),
        "one shard"
    );
    let source = HttpRangeSource::new(format!("{REPO_BASE}/{shard}"));
    let header = fetch_safetensors_header(&source).expect("shard header");
    for n in &names {
        let t = &header.tensors[*n];
        let want = if n.ends_with("embedding_projection.weight") {
            "U32"
        } else {
            "BF16"
        };
        assert_eq!(t.dtype, want, "{n}");
    }
    let table = &header.tensors["vision_tower.patch_embedder.position_embedding_table"];
    assert_eq!(
        table.shape,
        vec![
            2,
            vision.num_position_embeddings as u64,
            vision.hidden_size as u64
        ]
    );
    let proj = &header.tensors["embed_vision.embedding_projection.weight"];
    assert_eq!(
        proj.shape,
        vec![vision.out_hidden_size as u64, vision.hidden_size as u64 / 8]
    );
}

/// A shard on local disk as a range source, so a checkpoint downloaded once
/// can be repacked without streaming it a second time.
struct FileRangeSource(std::fs::File);

impl turbospark_repack::RangeSource for FileRangeSource {
    fn read_range(
        &self,
        start: u64,
        end_exclusive: u64,
    ) -> Result<Vec<u8>, turbospark_repack::DownloadError> {
        use std::os::unix::fs::FileExt;
        let mut buf = vec![0u8; (end_exclusive - start) as usize];
        self.0
            .read_exact_at(&mut buf, start)
            .map_err(|e| turbospark_repack::DownloadError::Request(e.to_string()))?;
        Ok(buf)
    }
}

/// The real checkpoint repacked WITH its vision tower, from a local copy at
/// the pinned revision (`TURBOSPARK_GEMMA4_SOURCE_DIR`, e.g. from
/// `hf download ... --local-dir`), into `TURBOSPARK_GEMMA4_VISION_INSTALL_DIR`.
/// The install the trunk's image gate opens (`docs/GEMMA4_VISION_PHASE0.md`).
#[test]
#[ignore = "repacks a local copy of the real ~15 GB Gemma 4 checkpoint with its vision tower"]
fn repacks_the_real_gemma4_checkpoint_with_its_vision_tower() {
    let src = PathBuf::from(
        std::env::var_os("TURBOSPARK_GEMMA4_SOURCE_DIR").expect("set TURBOSPARK_GEMMA4_SOURCE_DIR"),
    );
    let dir = PathBuf::from(
        std::env::var_os("TURBOSPARK_GEMMA4_VISION_INSTALL_DIR")
            .expect("set TURBOSPARK_GEMMA4_VISION_INSTALL_DIR"),
    );
    std::fs::create_dir_all(&dir).expect("install dir");
    let read = |name: &str| std::fs::read(src.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));

    let index_bytes = read("model.safetensors.index.json");
    assert_eq!(
        model_io::hash_data(&index_bytes),
        PINNED_INDEX_SHA256,
        "the local copy is not the pinned revision"
    );
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("index json");
    let shard_names: BTreeSet<String> = index["weight_map"]
        .as_object()
        .expect("weight_map")
        .values()
        .map(|v| v.as_str().expect("shard name").to_string())
        .collect();

    let config = String::from_utf8(read("config.json")).expect("config utf8");
    let mut arch = parse_gemma4_config(&config).expect("config parses");
    arch.vision = parse_gemma4_vision_config(&config).expect("vision config parses");
    let quant = parse_gemma4_quantization(&config).expect("quantization parses");

    let sources: Vec<FileRangeSource> = shard_names
        .iter()
        .map(|name| FileRangeSource(std::fs::File::open(src.join(name)).expect("shard opens")))
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

    write_gemma4_install_streamed(&dir, &arch, MODEL_ID, &shards, &quant, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed install with the vision tower");
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    ] {
        std::fs::write(dir.join(name), read(name)).expect("sidecar");
    }

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("manifest validates with the gemma4 tower");
    let resident = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");
    for name in [
        "vision.patch_embedder.input_proj.weight",
        "vision.patch_embedder.position_embedding_table",
        "vision.std_bias",
        "vision.std_scale",
        "vision.embedding_projection.weight",
    ] {
        assert!(resident.entries.contains_key(name), "{name} missing");
    }
    assert!(dir.join("packed_vision/blobs.bin").exists());
    eprintln!(
        "SUCCESS: gemma4 install with its vision tower at {}",
        dir.display()
    );
}
