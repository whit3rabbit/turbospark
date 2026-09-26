//! Compares installed Qwen4Exp GGUF routed experts and PLE rows with source bytes.
//!
//! This is an opt-in, read-only gate. It needs an existing Q2_0 install and
//! the complete range cache from that install, then streams bounded chunks
//! from the source cache and install without assembling another model or table
//! file.
//!
//! ```sh
//! TURBOSPARK_QWEN4EXP_GGUF_INSTALL_DIR=/tmp/turbospark-qwen4exp-swift-q2-0.gturbo \
//! TURBOSPARK_QWEN4EXP_SOURCE_CACHE_DIR=/tmp/turbospark-qwen4exp-swift-q2-0.source-cache \
//!   cargo test -p turbospark-repack --test qwen4exp_gguf_ple_fidelity_network \
//!   --release -- --ignored --nocapture
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use std::collections::BTreeMap;
use turbospark_repack::{
    arch_from_gguf, fetch_gguf_header, ggml_type_name, verify_install_full_sha256, GgufSet,
    HttpRangeSource, RangeSource,
};

const REVISION: &str = "b22d729eae29b5796f76fb70f91aef549b9fc52c";
const REPO_BASE: &str = "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF";
const SHARDS: [(&str, u64); 2] = [
    (
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        39_799_117_984,
    ),
    (
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
        26_750_834_816,
    ),
];
const TENSOR_NAME: &str = "per_layer_token_embd.weight";
const ROWS_PER_READ: u64 = 65_536;
const SOURCE_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

fn required_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name} to the existing artifact path"))
}

fn compare_routed_tensor(
    install: &std::path::Path,
    source: &GgufSet<HttpRangeSource>,
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
        let entry = layout.expert(layer, expert);
        let tensor = entry
            .sub_tensors
            .get(role)
            .unwrap_or_else(|| panic!("missing {role} for layer {layer}, expert {expert}"));
        assert_eq!(tensor.dtype, dtype, "{name} dtype at expert {expert}");
        assert_eq!(tensor.size, per_expert, "{name} size at expert {expert}");
    }

    let layer_layout = &layout.layers[layer];
    let layer_path = install
        .join(model_io::PACKED_EXPERTS_DIR)
        .join(&layer_layout.file);
    let mut layer_file = File::open(&layer_path).expect("open installed expert layer");
    let mut offset = 0u64;
    let mut installed_bytes = Vec::new();
    while offset < length {
        let chunk = SOURCE_CHUNK_BYTES.min(length - offset);
        let chunk_end = offset + chunk;
        let source_bytes = source
            .read_range(global_start + offset, global_start + chunk_end)
            .unwrap_or_else(|error| panic!("read pinned {name} source: {error}"));
        let first_expert = offset / per_expert;
        let last_expert = (chunk_end - 1) / per_expert;
        for expert in first_expert..=last_expert {
            let expert = expert as usize;
            let expert_start = expert as u64 * per_expert;
            let start = offset.max(expert_start);
            let end = chunk_end.min(expert_start + per_expert);
            let entry = layout.expert(layer, expert);
            let tensor = &entry.sub_tensors[role];
            let file_offset = entry.offset + tensor.offset + (start - expert_start);
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

#[test]
#[ignore = "network/cache-backed: compares installed Q2_0 routed and PLE bytes with the pinned source"]
fn installed_q2_0_routed_and_ple_bytes_match_all_pinned_source() {
    let install = required_path("TURBOSPARK_QWEN4EXP_GGUF_INSTALL_DIR");
    let cache = required_path("TURBOSPARK_QWEN4EXP_SOURCE_CACHE_DIR");
    assert!(install.is_dir(), "install directory must already exist");
    assert!(cache.is_dir(), "source range cache must already exist");

    let mut shard_sources = Vec::new();
    let mut source_base = 0u64;
    let mut ple_source = None;
    let mut source_ranges = BTreeMap::new();
    for (name, bytes) in &SHARDS {
        let url = format!("{REPO_BASE}/resolve/{REVISION}/{name}");
        let source = HttpRangeSource::new(url.clone()).with_cache_dir(&cache);
        let header = fetch_gguf_header(&source).expect("read pinned GGUF shard header");
        let shard_cache = cache.join(model_io::hash_data(url.as_bytes()));
        for tensor_name in header.tensors.keys() {
            let (start, end) = header
                .absolute_range(tensor_name)
                .expect("source tensor exists")
                .expect("source tensor range validates");
            source_ranges.insert(tensor_name.clone(), (start, end, shard_cache.clone()));
        }
        if header.tensors.contains_key(TENSOR_NAME) {
            assert!(
                ple_source.is_none(),
                "PLE tensor appears in one source shard"
            );
            ple_source = Some((source_base, shard_cache));
        }
        shard_sources.push((header, source, *bytes));
        source_base = source_base.checked_add(*bytes).expect("source shard base");
    }
    let (ple_source_base, ple_cache) = ple_source.expect("one shard contains the PLE tensor");
    let source = GgufSet::new(shard_sources).expect("validate pinned Q2_0 shard set");
    let (source_start, source_end) = source
        .header
        .absolute_range(TENSOR_NAME)
        .expect("PLE tensor exists")
        .expect("PLE tensor range is valid");

    let arch = arch_from_gguf(&source.header).expect("derive Qwen4Exp from pinned source");
    let num_layers = usize::try_from(arch.num_layers).expect("positive layer count");
    let num_experts = usize::try_from(arch.num_experts).expect("positive expert count");
    let expert_layout = model_io::load_packed_experts_layout(
        &install,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("existing routed layout validates");
    assert_eq!(expert_layout.num_layers, num_layers);
    assert_eq!(expert_layout.experts_per_layer, num_experts);
    let layout = model_io::load_ngram_table_layout(&install)
        .expect("installed PLE header validates")
        .expect("installed PLE table exists");
    assert_eq!(layout.ggml_type.as_deref(), Some("iq4_nl"));
    let expected_bytes = layout.blob_bytes().expect("table byte length is valid");
    assert_eq!(source_end - source_start, expected_bytes);
    let ple_local_start = source_start
        .checked_sub(ple_source_base)
        .expect("PLE source range begins inside its shard");

    // Check that every ranged chunk needed for routed tensors is already
    // cached before comparing payloads. read_tensor uses these same 16 MiB
    // boundaries, so this gate cannot fall back to a model download.
    for layer in 0..num_layers {
        for role in ["gate", "up", "down"] {
            let name = format!("blk.{layer}.ffn_{role}_exps.weight");
            let source_type = ggml_type_name(source.header.tensors[&name].ggml_type)
                .expect("routed source type is registered")
                .to_lowercase();
            let (tensor_start, tensor_end) = source
                .header
                .absolute_range(&name)
                .expect("routed source tensor exists")
                .expect("routed source range validates");
            let source_per_expert = (tensor_end - tensor_start) / num_experts as u64;
            for expert in 0..num_experts {
                let installed = expert_layout
                    .expert(layer, expert)
                    .sub_tensors
                    .get(role)
                    .unwrap_or_else(|| panic!("missing {role} for layer {layer}, expert {expert}"));
                assert_eq!(
                    installed.dtype, source_type,
                    "layer {layer} expert {expert} {role} dtype differs from pinned source"
                );
                assert_eq!(
                    installed.size, source_per_expert,
                    "layer {layer} expert {expert} {role} byte size differs from pinned source"
                );
            }
            let (local_start, local_end, shard_cache) = source_ranges
                .get(&name)
                .unwrap_or_else(|| panic!("missing source cache range for {name}"));
            let length = local_end - local_start;
            let mut offset = 0u64;
            while offset < length {
                let chunk = SOURCE_CHUNK_BYTES.min(length - offset);
                let start = local_start + offset;
                let end = start + chunk;
                assert!(
                    shard_cache.join(format!("{start}-{end}.range")).is_file(),
                    "missing cached source range for {name}: {start}-{end}; refusing network fallback"
                );
                offset += chunk;
            }
        }
    }

    model_io::load_manifest(&install, &arch, model_io::DEFAULT_MAX_BYTES)
        .expect("existing real install validates against the pinned source architecture");
    verify_install_full_sha256(&install, &arch)
        .expect("all manifest-listed install files pass SHA-256 verification");

    // Fail before reading payloads if the exact bounded ranges from the
    // original streamed install are absent. This gate must not silently turn
    // into another multi-gigabyte download.
    let mut cached_row = 0u64;
    while cached_row < layout.rows {
        let rows = ROWS_PER_READ.min(layout.rows - cached_row);
        let offset = cached_row
            .checked_mul(layout.record_bytes)
            .expect("cached PLE row offset");
        let bytes = rows
            .checked_mul(layout.record_bytes)
            .expect("cached PLE chunk size");
        let start = ple_local_start
            .checked_add(offset)
            .expect("cached PLE source offset");
        let end = start.checked_add(bytes).expect("cached PLE source end");
        assert!(
            ple_cache.join(format!("{start}-{end}.range")).is_file(),
            "missing cached source range {start}-{end}; refusing a network fallback"
        );
        cached_row += rows;
    }

    let mut routed_verified = 0u64;
    for layer in 0..num_layers {
        for role in ["gate", "up", "down"] {
            let previous = routed_verified;
            routed_verified +=
                compare_routed_tensor(&install, &source, &expert_layout, layer, role);
            if routed_verified / (1 << 30) != previous / (1 << 30) {
                eprintln!(
                    "verified {} GiB of pinned routed expert bytes",
                    routed_verified / (1 << 30)
                );
            }
        }
    }

    let table_path = install
        .join(model_io::NGRAM_TABLE_DIR)
        .join(model_io::NGRAM_TABLE_BLOB);
    let mut table = File::open(&table_path).expect("open existing installed PLE rows");
    assert_eq!(
        table.metadata().expect("installed table metadata").len(),
        expected_bytes,
        "installed PLE table has no trailing or missing bytes"
    );
    let mut row = 0u64;
    let mut verified = 0u64;
    while row < layout.rows {
        let rows = ROWS_PER_READ.min(layout.rows - row);
        let bytes = rows
            .checked_mul(layout.record_bytes)
            .and_then(|n| usize::try_from(n).ok())
            .expect("bounded PLE chunk size");
        let offset = row
            .checked_mul(layout.record_bytes)
            .expect("PLE row offset");
        let source_bytes = source
            .read_range(source_start + offset, source_start + offset + bytes as u64)
            .expect("read pinned source PLE rows");
        let mut installed_bytes = vec![0u8; bytes];
        table
            .read_exact(&mut installed_bytes)
            .expect("read installed PLE rows");
        assert_eq!(
            source_bytes, installed_bytes,
            "installed PLE rows diverge from source at row {row}"
        );
        row += rows;
        verified += bytes as u64;
        if verified / (1 << 30) != (verified - bytes as u64) / (1 << 30) {
            eprintln!("verified {} GiB of pinned PLE rows", verified / (1 << 30));
        }
    }
    assert_eq!(verified, expected_bytes);
}
