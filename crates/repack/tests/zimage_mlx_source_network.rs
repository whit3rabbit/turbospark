//! Header-only intake checks for the published Z-Image-Turbo MLX variants.
//!
//! These checks fetch safetensors headers and quantization metadata only. They
//! do not download the multi-gigabyte tensor payloads or claim runtime parity.
//!
//! ```text
//! cargo test -p turbospark-repack --test zimage_mlx_source_network \
//!   --release -- --ignored --nocapture
//! ```

use turbospark_repack::{fetch_safetensors_header, HttpRangeSource};

struct Variant {
    label: &'static str,
    repo: &'static str,
    revision: &'static str,
    bits: Option<u64>,
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

const TRANSFORMER_SHARDS: [&str; 3] = [
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
];

fn get_json(repo: &Variant, path: &str) -> serde_json::Value {
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        repo.repo, repo.revision, path
    );
    let response = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .expect("HTTP client")
        .get(&url)
        .send()
        .unwrap_or_else(|error| panic!("GET {url}: {error}"));
    assert!(
        response.status().is_success(),
        "GET {url}: {}",
        response.status()
    );
    serde_json::from_slice(&response.bytes().expect("response bytes")).expect("JSON response")
}

#[test]
#[ignore = "network: reads four pinned Z-Image MLX safetensors headers"]
fn published_zimage_mlx_variants_match_the_image_adapter_contract() {
    for variant in VARIANTS {
        match variant.bits {
            Some(bits) => {
                let quantization = get_json(&variant, "quantize_config.json");
                assert_eq!(
                    quantization["quantization"]["bits"], bits,
                    "{}",
                    variant.label
                );
                assert_eq!(
                    quantization["quantization"]["group_size"], 64,
                    "{}",
                    variant.label
                );
                let mut total_u32 = 0;
                for shard in TRANSFORMER_SHARDS {
                    let header = fetch_header(&variant, shard);
                    total_u32 += assert_mlx_header(&variant, &header, bits);
                }
                assert!(total_u32 > 0, "{} has no MLX weights", variant.label);
            }
            None => {
                for shard in TRANSFORMER_SHARDS {
                    let header = fetch_header(&variant, shard);
                    assert!(
                        header.tensors.values().all(|tensor| tensor.dtype == "F16"),
                        "{} transformer shard {shard} is not all F16",
                        variant.label
                    );
                }
            }
        }
    }
}

fn fetch_header(variant: &Variant, shard: &str) -> turbospark_repack::SafetensorsHeader {
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        variant.repo, variant.revision, shard
    );
    fetch_safetensors_header(&HttpRangeSource::new(url))
        .unwrap_or_else(|error| panic!("{} {shard} header: {error}", variant.label))
}

fn assert_mlx_header(
    variant: &Variant,
    header: &turbospark_repack::SafetensorsHeader,
    bits: u64,
) -> usize {
    let u32_weights: Vec<_> = header
        .tensors
        .iter()
        .filter(|(_, tensor)| tensor.dtype == "U32")
        .collect();
    let scale_count = header
        .tensors
        .keys()
        .filter(|name| name.ends_with(".scales"))
        .count();
    let bias_count = header
        .tensors
        .keys()
        .filter(|name| name.ends_with(".biases"))
        .count();
    assert_eq!(
        scale_count,
        u32_weights.len(),
        "{} scale count",
        variant.label
    );
    assert_eq!(
        bias_count,
        u32_weights.len(),
        "{} bias count",
        variant.label
    );
    for (name, weight) in &u32_weights {
        let logical_cols = weight.shape[1] * 32 / bits;
        assert_eq!(logical_cols % 64, 0, "{name} group alignment");
        for suffix in ["scales", "biases"] {
            let companion = format!(
                "{}.{}",
                name.strip_suffix(".weight").unwrap_or(name),
                suffix
            );
            let tensor = &header.tensors[&companion];
            assert_eq!(tensor.shape, vec![weight.shape[0], logical_cols / 64]);
            assert!(matches!(tensor.dtype.as_str(), "F16" | "BF16"));
        }
    }
    u32_weights.len()
}
