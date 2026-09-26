//! Real install gates for ukisai's Qwen3.8-Flash-Next GSQ-RCO tiers.
//!
//! Source shards are read by HTTP range and never assembled locally. The
//! install is retained so runtime and resource gates can use these exact
//! bytes.
//!
//! The install target caches source ranges beside the output by default, so
//! interrupted runs can reuse completed downloads. Set
//! `TURBOSPARK_QWEN4EXP_DISABLE_SOURCE_CACHE=1` on disk-constrained hosts to
//! stream directly into the install; an interrupted run then starts over.
//!
//! ```sh
//! TURBOSPARK_QWEN4EXP_GGUF_INSTALL_DIR=/tmp/qwen4exp-swift-q2-0.gturbo \
//!   cargo test -p turbospark-repack --test qwen4exp_gguf_install_network \
//!   --release -- --ignored --nocapture
//! TURBOSPARK_QWEN4EXP_DISABLE_SOURCE_CACHE=1 \
//! TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR=/tmp/qwen4exp-swift-iq2-xs.gturbo \
//!   cargo test -p turbospark-repack --test qwen4exp_gguf_install_network \
//!   installs_the_real_swift_qwen38_iq2_xs_gguf --release -- --ignored --nocapture --exact
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use turbospark_repack::{
    arch_from_gguf, fetch_gguf_header, ggml_type_name, verify_install_full_sha256,
    write_gguf_install_streamed, ByteProgressCallback, DownloadError, GgufSet, HttpRangeSource,
    RangeSource,
};

const REVISION: &str = "b22d729eae29b5796f76fb70f91aef549b9fc52c";
const BASE_REVISION: &str = "0bd4fe22431372cdad1979267d3ab45aa7e6150a";
const MODEL_ID: &str = "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF";
const REPO_BASE: &str = "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF";
const BASE_MODEL: &str = "https://huggingface.co/ukisai/Swift-Qwen3.8-Flash-Next";
const Q2_0_SHARDS: [(&str, u64); 2] = [
    (
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        39_799_117_984,
    ),
    (
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
        26_750_834_816,
    ),
];
const IQ2_XS_SHARDS: [(&str, u64); 2] = [
    (
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
        39_788_473_344,
    ),
    (
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
        28_363_693_824,
    ),
];

struct LocalFileRangeSource {
    path: PathBuf,
    length: u64,
}

impl LocalFileRangeSource {
    fn open(path: PathBuf, expected_length: u64) -> Self {
        let length = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("stat source shard {}: {e}", path.display()))
            .len();
        assert_eq!(
            length,
            expected_length,
            "source shard length: {}",
            path.display()
        );
        Self { path, length }
    }
}

impl RangeSource for LocalFileRangeSource {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        if start > end_exclusive || end_exclusive > self.length {
            return Err(DownloadError::InvalidRange {
                start,
                end_exclusive,
            });
        }
        let size = usize::try_from(end_exclusive - start)
            .map_err(|_| DownloadError::Request("local source range is too large".into()))?;
        let mut bytes = vec![0; size];
        let mut file = File::open(&self.path)
            .map_err(|e| DownloadError::Request(format!("{}: {e}", self.path.display())))?;
        file.seek(SeekFrom::Start(start))
            .map_err(|e| DownloadError::Request(format!("{}: {e}", self.path.display())))?;
        file.read_exact(&mut bytes)
            .map_err(|e| DownloadError::Request(format!("{}: {e}", self.path.display())))?;
        Ok(bytes)
    }
}

fn compare_routed_tensor<S: RangeSource>(
    install: &Path,
    source: &GgufSet<S>,
    layout: &model_io::PackedExpertsLayout,
    layer: usize,
    role: &str,
) -> u64 {
    let name = format!("blk.{layer}.ffn_{role}_exps.weight");
    let (global_start, global_end) = source
        .header
        .absolute_range(&name)
        .expect("routed source tensor exists")
        .expect("routed source range validates");
    let length = global_end - global_start;
    assert_eq!(length % layout.experts_per_layer as u64, 0);
    let per_expert = length / layout.experts_per_layer as u64;
    let dtype = ggml_type_name(source.header.tensors[&name].ggml_type)
        .expect("routed type is registered")
        .to_lowercase();
    for expert in 0..layout.experts_per_layer {
        let tensor = layout
            .expert(layer, expert)
            .sub_tensors
            .get(role)
            .unwrap_or_else(|| panic!("missing {role} for layer {layer}, expert {expert}"));
        assert_eq!(tensor.dtype, dtype, "{name} dtype at expert {expert}");
        assert_eq!(tensor.size, per_expert, "{name} size at expert {expert}");
    }

    let layer_path = install
        .join(model_io::PACKED_EXPERTS_DIR)
        .join(&layout.layers[layer].file);
    let mut layer_file = File::open(&layer_path).expect("open installed expert layer");
    let mut offset = 0u64;
    let mut installed_bytes = Vec::new();
    while offset < length {
        let chunk_end = offset + (16 * 1024 * 1024).min(length - offset);
        let source_bytes = source
            .read_range(global_start + offset, global_start + chunk_end)
            .unwrap_or_else(|error| panic!("read local {name} source: {error}"));
        let first_expert = offset / per_expert;
        let last_expert = (chunk_end - 1) / per_expert;
        for expert in first_expert..=last_expert {
            let expert = expert as usize;
            let expert_start = expert as u64 * per_expert;
            let start = offset.max(expert_start);
            let end = chunk_end.min(expert_start + per_expert);
            let tensor = &layout.expert(layer, expert).sub_tensors[role];
            let file_offset =
                layout.expert(layer, expert).offset + tensor.offset + (start - expert_start);
            let segment_bytes = usize::try_from(end - start).expect("expert segment size");
            installed_bytes.resize(segment_bytes, 0);
            layer_file
                .seek(SeekFrom::Start(file_offset))
                .and_then(|_| layer_file.read_exact(&mut installed_bytes))
                .unwrap_or_else(|error| {
                    panic!("read installed layer {layer} expert {expert} {role}: {error}")
                });
            let source_start = usize::try_from(start - offset).expect("source slice start");
            let source_end = usize::try_from(end - offset).expect("source slice end");
            assert_eq!(
                &source_bytes[source_start..source_end],
                installed_bytes,
                "source bytes differ for layer {layer}, expert {expert}, {role}"
            );
        }
        offset = chunk_end;
    }
    length
}

fn verify_local_source_fidelity<S: RangeSource>(install: &Path, source: &GgufSet<S>) {
    let arch = arch_from_gguf(&source.header).expect("derive Qwen4Exp from local source");
    let num_layers = usize::try_from(arch.num_layers).expect("positive layer count");
    let layout = model_io::load_packed_experts_layout(
        install,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("installed routed layout validates");
    assert_eq!(layout.num_layers, num_layers);
    assert_eq!(layout.experts_per_layer, arch.num_experts as usize);

    let mut routed_verified = 0u64;
    for layer in 0..num_layers {
        for role in ["gate", "up", "down"] {
            let previous = routed_verified;
            routed_verified += compare_routed_tensor(install, source, &layout, layer, role);
            if routed_verified / (1 << 30) != previous / (1 << 30) {
                eprintln!(
                    "verified {} GiB of local routed bytes",
                    routed_verified / (1 << 30)
                );
            }
        }
    }

    let table_layout = model_io::load_ngram_table_layout(install)
        .expect("installed PLE header validates")
        .expect("installed PLE table exists");
    assert_eq!(table_layout.ggml_type.as_deref(), Some("iq4_nl"));
    let (source_start, source_end) = source
        .header
        .absolute_range("per_layer_token_embd.weight")
        .expect("source PLE tensor exists")
        .expect("source PLE tensor range validates");
    let expected_bytes = table_layout.blob_bytes().expect("PLE byte length is valid");
    assert_eq!(source_end - source_start, expected_bytes);
    let table_path = install
        .join(model_io::NGRAM_TABLE_DIR)
        .join(model_io::NGRAM_TABLE_BLOB);
    let mut table = File::open(&table_path).expect("open installed PLE rows");
    assert_eq!(
        table.metadata().expect("PLE metadata").len(),
        expected_bytes
    );

    let mut row = 0u64;
    while row < table_layout.rows {
        let rows = 65_536.min(table_layout.rows - row);
        let byte_count = rows
            .checked_mul(table_layout.record_bytes)
            .expect("bounded PLE chunk size");
        let byte_count_usize = usize::try_from(byte_count).expect("PLE chunk fits memory");
        let offset = row
            .checked_mul(table_layout.record_bytes)
            .expect("PLE row offset");
        let source_bytes = source
            .read_range(source_start + offset, source_start + offset + byte_count)
            .expect("read local PLE rows");
        let mut installed_bytes = vec![0; byte_count_usize];
        table
            .read_exact(&mut installed_bytes)
            .expect("read installed PLE rows");
        assert_eq!(source_bytes, installed_bytes, "PLE differs at row {row}");
        row += rows;
    }
    eprintln!(
        "verified {} routed bytes and {} PLE rows against local pinned shards",
        routed_verified, table_layout.rows
    );
}

fn install_dir(env_var: &str, tier: &str) -> PathBuf {
    let dir = std::env::var_os(env_var)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!(
                "turbospark-qwen4exp-swift-{tier}-{}",
                std::process::id()
            ))
        });
    let existing = "TURBOSPARK_QWEN4EXP_INSTALL_DIR";
    assert_ne!(
        std::env::var_os(existing).map(PathBuf::from),
        Some(dir.clone()),
        "refusing to overwrite {existing}"
    );
    std::fs::create_dir_all(&dir).expect("create install directory");
    dir
}

fn source(url: String, cache_dir: Option<&Path>, total_bytes: &Arc<AtomicU64>) -> HttpRangeSource {
    let total_bytes = Arc::clone(total_bytes);
    let on_bytes: ByteProgressCallback = Arc::new(move |bytes| {
        let previous = total_bytes.fetch_add(bytes, Ordering::Relaxed);
        let total = previous + bytes;
        if previous / (1 << 30) < total / (1 << 30) {
            eprintln!("source ranges downloaded: {} GiB", total / (1 << 30));
        }
    });
    let source = HttpRangeSource::with_progress(url, on_bytes);
    match cache_dir {
        Some(cache_dir) => source.with_cache_dir(cache_dir),
        None => source,
    }
}

fn get(url: &str) -> Vec<u8> {
    let response = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .expect("HTTP client")
        .get(url)
        .send()
        .unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(
        response.status().is_success(),
        "GET {url}: HTTP {}",
        response.status()
    );
    response.bytes().expect("download sidecar").to_vec()
}

fn install_tier(tier: &str, install_env: &str, shards: &[(&str, u64)]) {
    let dir = install_dir(install_env, tier);
    let cache_dir =
        if std::env::var("TURBOSPARK_QWEN4EXP_DISABLE_SOURCE_CACHE").as_deref() == Ok("1") {
            None
        } else {
            Some(dir.with_extension("source-cache"))
        };
    if let Some(cache_dir) = &cache_dir {
        std::fs::create_dir_all(cache_dir).expect("create source range cache");
    }
    let downloaded_bytes = Arc::new(AtomicU64::new(0));
    let shard_sources = shards
        .iter()
        .map(|(name, bytes)| {
            let url = format!("{REPO_BASE}/resolve/{REVISION}/{name}");
            let source = source(url, cache_dir.as_deref(), &downloaded_bytes);
            let header = fetch_gguf_header(&source).expect("fetch shard header");
            (header, source, *bytes)
        })
        .collect();
    let source = GgufSet::new(shard_sources).expect("validate both split GGUF shards");
    assert_eq!(source.header.architecture(), Some("qwen4exp"));
    assert_eq!(source.header.tensors.len(), 1_224);

    eprintln!("installing {MODEL_ID} at {}", dir.display());
    match &cache_dir {
        Some(cache_dir) => eprintln!("source range cache: {}", cache_dir.display()),
        None => eprintln!("source range cache: disabled"),
    }
    let arch = write_gguf_install_streamed(&dir, &source.header, &source, MODEL_ID, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed Qwen4Exp GGUF install");
    assert_eq!(arch.family, model_io::ModelFamily::Qwen4Exp);
    assert_eq!(arch.num_layers, 48);
    assert_eq!(arch.num_experts, 512);
    assert_eq!(arch.top_k_experts, 10);

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ] {
        let url = format!("{BASE_MODEL}/resolve/{BASE_REVISION}/{name}");
        std::fs::write(dir.join(name), get(&url)).expect("write tokenizer sidecar");
    }

    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("the real manifest, resident index, and routed layout validate");
    verify_install_full_sha256(&dir, &arch).expect("all installed files pass SHA-256 verification");
    eprintln!("verified install: {}", dir.display());
}

fn install_tier_from_local_shards(
    tier: &str,
    install_env: &str,
    source_dir_env: &str,
    shards: &[(&str, u64)],
) {
    let dir = install_dir(install_env, tier);
    let source_dir = std::env::var_os(source_dir_env)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("set {source_dir_env} to the directory containing the pinned shards")
        });
    let shard_sources = shards
        .iter()
        .map(|(name, bytes)| {
            let source = LocalFileRangeSource::open(source_dir.join(name), *bytes);
            let header = fetch_gguf_header(&source).expect("fetch local shard header");
            (header, source, *bytes)
        })
        .collect();
    let source = GgufSet::new(shard_sources).expect("validate both local split GGUF shards");
    assert_eq!(source.header.architecture(), Some("qwen4exp"));
    assert_eq!(source.header.tensors.len(), 1_224);

    eprintln!(
        "installing {MODEL_ID} from local pinned shards at {}",
        dir.display()
    );
    let arch = write_gguf_install_streamed(&dir, &source.header, &source, MODEL_ID, |stage| {
        eprintln!("[repack] {stage}");
    })
    .expect("streamed Qwen4Exp GGUF install from local shards");
    assert_eq!(arch.family, model_io::ModelFamily::Qwen4Exp);
    assert_eq!(arch.num_layers, 48);
    assert_eq!(arch.num_experts, 512);
    assert_eq!(arch.top_k_experts, 10);

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ] {
        let url = format!("{BASE_MODEL}/resolve/{BASE_REVISION}/{name}");
        std::fs::write(dir.join(name), get(&url)).expect("write tokenizer sidecar");
    }
    model_io::load_manifest(&dir, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("the real manifest, resident index, and routed layout validate");
    verify_install_full_sha256(&dir, &arch).expect("all installed files pass SHA-256 verification");
    verify_local_source_fidelity(&dir, &source);
}

fn hf_tokens_by_id(tokenizer_json: &[u8]) -> Vec<Option<String>> {
    let json: serde_json::Value = serde_json::from_slice(tokenizer_json).expect("tokenizer JSON");
    let vocab = json["model"]["vocab"]
        .as_object()
        .expect("tokenizer model.vocab map");
    let added = json["added_tokens"]
        .as_array()
        .expect("tokenizer added_tokens array");
    let max_id = vocab
        .values()
        .filter_map(serde_json::Value::as_u64)
        .chain(added.iter().filter_map(|entry| entry["id"].as_u64()))
        .max()
        .expect("non-empty tokenizer vocabulary") as usize;
    let mut by_id = vec![None; max_id + 1];
    for (token, id) in vocab {
        let id = id.as_u64().expect("integer token id") as usize;
        assert!(
            by_id[id].replace(token.clone()).is_none(),
            "duplicate id {id}"
        );
    }
    for entry in added {
        let id = entry["id"].as_u64().expect("added token id") as usize;
        let token = entry["content"].as_str().expect("added token content");
        if let Some(existing) = by_id[id].as_deref() {
            assert_eq!(existing, token, "conflicting token at id {id}");
        } else {
            by_id[id] = Some(token.to_string());
        }
    }
    by_id
}

/// The model's GGUF embeds a tokenizer and chat template; the production
/// install uses HF sidecars. Compare both against the same pinned base model
/// before attributing bad generation to weight conversion or runtime math.
#[test]
#[ignore = "network: reads only the two pinned GGUF headers and HF tokenizer sidecars"]
fn pinned_swift_ggufs_match_hf_tokenizer_sidecars() {
    let cache_dir = std::env::var_os("TURBOSPARK_QWEN4EXP_SOURCE_CACHE_DIR")
        .map(PathBuf::from)
        .expect("set TURBOSPARK_QWEN4EXP_SOURCE_CACHE_DIR");
    let downloaded_bytes = Arc::new(AtomicU64::new(0));
    let tokenizer_bytes = get(&format!(
        "{BASE_MODEL}/resolve/{BASE_REVISION}/tokenizer.json"
    ));
    let hf_tokens = hf_tokens_by_id(&tokenizer_bytes);
    let hf_template = get(&format!(
        "{BASE_MODEL}/resolve/{BASE_REVISION}/chat_template.jinja"
    ));
    let hf_template = std::str::from_utf8(&hf_template).expect("UTF-8 Jinja template");

    for (tier, shards) in [("Q2_0", &Q2_0_SHARDS), ("IQ2_XS", &IQ2_XS_SHARDS)] {
        let (name, _) = shards[0];
        let url = format!("{REPO_BASE}/resolve/{REVISION}/{name}");
        let source = source(url, Some(&cache_dir), &downloaded_bytes);
        let header = fetch_gguf_header(&source).expect("fetch pinned shard header");
        let gguf_tokens = header
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|value| value.as_array())
            .expect("GGUF tokenizer.ggml.tokens array");
        assert!(gguf_tokens.len() >= hf_tokens.len());
        for (id, expected) in hf_tokens.iter().enumerate() {
            let Some(expected) = expected else { continue };
            let actual = gguf_tokens[id]
                .as_str()
                .unwrap_or_else(|| panic!("{tier}: GGUF token {id} is not a string"));
            assert_eq!(
                actual, expected,
                "{tier}: tokenizer token mismatch at id {id}"
            );
        }
        let gguf_template = header
            .metadata_str("tokenizer.chat_template")
            .unwrap_or_else(|| panic!("{tier}: GGUF has no tokenizer.chat_template"));
        assert_eq!(
            gguf_template.trim_end_matches('\n'),
            hf_template.trim_end_matches('\n'),
            "{tier}: embedded chat template differs from the HF sidecar"
        );
        eprintln!(
            "{tier}: {} HF token IDs and chat template match GGUF metadata",
            hf_tokens.iter().filter(|token| token.is_some()).count()
        );
    }
}

#[test]
#[ignore = "network: streams both Q2_0 shards (66.55 GB) into a real Qwen4Exp install"]
fn installs_the_real_swift_qwen38_q2_0_gguf() {
    install_tier("q2-0", "TURBOSPARK_QWEN4EXP_GGUF_INSTALL_DIR", &Q2_0_SHARDS);
}

#[test]
#[ignore = "network: streams both IQ2_XS shards (68.15 GB) into a real Qwen4Exp install"]
fn installs_the_real_swift_qwen38_iq2_xs_gguf() {
    install_tier(
        "iq2-xs",
        "TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR",
        &IQ2_XS_SHARDS,
    );
}

#[test]
#[ignore = "needs the pinned IQ2_XS GGUF shards already assembled locally"]
fn installs_swift_qwen38_iq2_xs_gguf_from_local_shards() {
    install_tier_from_local_shards(
        "iq2-xs",
        "TURBOSPARK_QWEN4EXP_IQ2_XS_INSTALL_DIR",
        "TURBOSPARK_QWEN4EXP_IQ2_XS_SHARD_DIR",
        &IQ2_XS_SHARDS,
    );
}
