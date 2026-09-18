//! Fetches a small real tensor from every published Z-Image MLX variant.
//!
//! This intentionally reads only selected tensor ranges, writes a tiny local
//! safetensors fixture, and sends it through the production image packer and
//! decoder. It proves real payload compatibility without requiring the
//! multi-gigabyte source tree or a full image install.
//!
//! ```text
//! cargo test -p turbospark-image --test zimage_mlx_payload_network \
//!   --release -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use repack::{fetch_safetensors_header, HttpRangeSource, RangeSource, SafetensorsHeader};
use serde_json::json;
use turbospark_image::{pack_component, PackedTensorStore};

struct Variant {
    label: &'static str,
    repo: &'static str,
    revision: &'static str,
    bits: Option<u8>,
}

const VARIANTS: [Variant; 4] = [
    Variant {
        label: "2bit",
        repo: "andrevp/Z-Image-Turbo-MLX-2bit",
        revision: "32b4e9ceb3a813485027b1ea942f199608fb8200",
        bits: Some(2),
    },
    Variant {
        label: "4bit",
        repo: "andrevp/Z-Image-Turbo-MLX-4bit",
        revision: "9adc576198c9126874792d35569b53cf2f45a03c",
        bits: Some(4),
    },
    Variant {
        label: "8bit",
        repo: "andrevp/Z-Image-Turbo-MLX-8bit",
        revision: "c9f70995562299b1eda9b9145a94dd7a5a1ae0d6",
        bits: Some(8),
    },
    Variant {
        label: "fp16",
        repo: "andrevp/Z-Image-Turbo-MLX",
        revision: "e186d7d65d66883270671fcee05324178928ea03",
        bits: None,
    },
];

const SHARD: &str = "transformer/diffusion_pytorch_model-00001-of-00003.safetensors";

#[test]
#[ignore = "network: reads selected real Z-Image MLX tensor ranges"]
fn real_zimage_mlx_payloads_pack_and_decode_for_every_published_variant() {
    for variant in VARIANTS {
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            variant.repo, variant.revision, SHARD
        );
        let source = HttpRangeSource::new(url);
        let header = fetch_safetensors_header(&source)
            .unwrap_or_else(|error| panic!("{} header: {error}", variant.label));
        let root = temporary_directory(variant.label);
        let source_dir = root.join("source");
        let output_dir = root.join("packed");
        fs::create_dir_all(&source_dir).expect("create source directory");

        match variant.bits {
            Some(bits) => pack_real_mlx_tensor(&source, &header, &source_dir, &output_dir, bits),
            None => pack_real_fp16_tensor(&source, &header, &source_dir, &output_dir),
        }
        fs::remove_dir_all(root).expect("remove payload fixture");
    }
}

fn pack_real_mlx_tensor(
    source: &HttpRangeSource,
    header: &SafetensorsHeader,
    source_dir: &Path,
    output_dir: &Path,
    bits: u8,
) {
    let (weight_name, weight) = header
        .tensors
        .iter()
        .filter(|(_, tensor)| tensor.dtype == "U32" && tensor.shape.len() == 2)
        .filter_map(|(name, tensor)| {
            let base = name.strip_suffix(".weight").unwrap_or(name);
            let scales = header.tensors.get(&format!("{base}.scales"))?;
            let biases = header.tensors.get(&format!("{base}.biases"))?;
            let logical_cols = tensor.shape[1].checked_mul(32)?.checked_div(bits as u64)?;
            if logical_cols % 64 != 0 || scales.dtype != biases.dtype {
                return None;
            }
            Some((name.as_str(), tensor))
        })
        .min_by_key(|(_, tensor)| tensor.data_offsets.1 - tensor.data_offsets.0)
        .expect("published MLX shard has a supported U32 tensor");
    let base = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scale_name = format!("{base}.scales");
    let bias_name = format!("{base}.biases");
    let scale = &header.tensors[&scale_name];
    let bias = &header.tensors[&bias_name];
    let tensors = [
        (weight_name, weight),
        (scale_name.as_str(), scale),
        (bias_name.as_str(), bias),
    ];
    let payload = fetch_tensors(source, header, &tensors);
    write_source(source_dir, &tensors, &payload);
    pack_and_check(
        source_dir,
        output_dir,
        weight_name,
        bits,
        weight.shape[1] * 32 / bits as u64,
    );
}

fn pack_real_fp16_tensor(
    source: &HttpRangeSource,
    header: &SafetensorsHeader,
    source_dir: &Path,
    output_dir: &Path,
) {
    let (name, tensor) = header
        .tensors
        .iter()
        .filter(|(_, tensor)| tensor.dtype == "F16")
        .min_by_key(|(_, tensor)| tensor.data_offsets.1 - tensor.data_offsets.0)
        .expect("published fp16 shard has an F16 tensor");
    let tensors = [(name.as_str(), tensor)];
    let payload = fetch_tensors(source, header, &tensors);
    write_source(source_dir, &tensors, &payload);
    pack_component(source_dir, "model.safetensors.index.json", output_dir)
        .expect("pack real fp16 tensor");
    let store = PackedTensorStore::open(output_dir).expect("open packed fp16 tensor");
    let values = store.load_tensor(name).expect("decode real fp16 tensor");
    assert!(!values.is_empty());
    assert!(values.iter().all(|value| value.is_finite()));
}

fn fetch_tensors<'a>(
    source: &HttpRangeSource,
    header: &SafetensorsHeader,
    tensors: &[(&'a str, &'a repack::TensorInfo)],
) -> BTreeMap<&'a str, Vec<u8>> {
    tensors
        .iter()
        .map(|(name, tensor)| {
            let (start, end) = header
                .absolute_range(name)
                .expect("tensor has an addressable range");
            let bytes = source
                .read_range(start, end)
                .unwrap_or_else(|error| panic!("fetch {name}: {error}"));
            assert_eq!(
                bytes.len() as u64,
                tensor.data_offsets.1 - tensor.data_offsets.0
            );
            (*name, bytes)
        })
        .collect()
}

fn write_source(
    source_dir: &Path,
    tensors: &[(&str, &repack::TensorInfo)],
    payloads: &BTreeMap<&str, Vec<u8>>,
) {
    let mut offset = 0u64;
    let mut entries = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, tensor) in tensors {
        let bytes = &payloads[name];
        let end = offset + bytes.len() as u64;
        entries.insert(
            (*name).to_string(),
            json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, end]
            }),
        );
        payload.extend_from_slice(bytes);
        offset = end;
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(entries)).expect("source header");
    let mut shard = Vec::with_capacity(8 + header.len() + payload.len());
    shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard.extend_from_slice(&header);
    shard.extend_from_slice(&payload);
    fs::write(source_dir.join("actual.safetensors"), shard).expect("write source shard");
    write_index(
        source_dir,
        tensors.iter().map(|(name, _)| *name),
        "actual.safetensors",
    );
}

fn write_index<'a>(source_dir: &Path, names: impl IntoIterator<Item = &'a str>, shard: &str) {
    let weight_map: BTreeMap<_, _> = names
        .into_iter()
        .map(|name| (name.to_string(), shard.to_string()))
        .collect();
    let index = serde_json::json!({"weight_map": weight_map});
    fs::write(
        source_dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).expect("source index"),
    )
    .expect("write source index");
}

fn pack_and_check(source_dir: &Path, output_dir: &Path, name: &str, bits: u8, logical_cols: u64) {
    pack_component(source_dir, "model.safetensors.index.json", output_dir)
        .expect("pack real MLX tensor");
    let store = PackedTensorStore::open(output_dir).expect("open packed MLX tensor");
    let tensor = store.tensor(name).expect("packed MLX tensor descriptor");
    assert_eq!(tensor.storage_dtype, "MLX_AFFINE");
    assert_eq!(
        tensor.quantization.as_ref().expect("MLX metadata").bits,
        bits
    );
    let row = store.load_row(name, 0).expect("decode real MLX tensor row");
    assert_eq!(row.len() as u64, logical_cols);
    assert!(row.iter().all(|value| value.is_finite()));
}

fn temporary_directory(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "turbospark-zimage-mlx-payload-{label}-{}-{nanos}-{sequence}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create payload fixture root");
    root
}
