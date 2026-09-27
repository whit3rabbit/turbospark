//! Curated Diffusers image sources, separate from the text-model catalog.

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Deserialize;

use crate::{CancelFlag, Client, RepoRef, INSTALL_CANCELLED};
use repack::HttpRangeSource;

const EMBEDDED: &str = include_str!("image_models.json");

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ImageCatalogEntry {
    pub alias: String,
    pub model_id: String,
    pub revision: String,
    pub required_files: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ImageCatalog {
    entries: Vec<ImageCatalogEntry>,
}

impl ImageCatalog {
    pub fn embedded() -> Result<Self, String> {
        let entries: Vec<ImageCatalogEntry> = serde_json::from_str(EMBEDDED)
            .map_err(|e| format!("parsing embedded image catalog: {e}"))?;
        if entries.is_empty() {
            return Err("embedded image catalog is empty".to_string());
        }
        for entry in &entries {
            validate(entry)?;
        }
        Ok(Self { entries })
    }

    pub fn entries(&self) -> impl Iterator<Item = &ImageCatalogEntry> {
        self.entries.iter()
    }

    pub fn get(&self, alias: &str) -> Option<&ImageCatalogEntry> {
        self.entries.iter().find(|entry| entry.alias == alias)
    }
}

/// Materializes the bounded Diffusers source tree consumed by the image
/// packer. Large safetensors files use bounded concurrent ranges written to a
/// sibling partial file. Completed ranges are cached with SHA-256 under the
/// staging tree, so an interrupted immutable-revision install can reuse them.
pub fn materialize_image_source(
    client: &Client,
    repo: &RepoRef,
    destination: &Path,
    cancel: Option<&CancelFlag>,
    mut progress: impl FnMut(&str),
    on_bytes: Arc<dyn Fn(u64, u64) + Send + Sync>,
) -> Result<u64, String> {
    if destination.exists() && !destination.is_dir() {
        return Err(format!(
            "image source staging path is not a directory: {}",
            destination.display()
        ));
    }
    if repo.revision.len() != 40 || !repo.revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "image repository revision {:?} is not an immutable 40-hex commit",
            repo.revision
        ));
    }
    let files = client.file_list_with_sizes(repo)?;
    let selected = select_files(
        &files
            .iter()
            .map(|file| file.name.clone())
            .collect::<Vec<_>>(),
    )?;
    let selected = selected
        .into_iter()
        .map(|name| {
            let listed = files
                .iter()
                .find(|file| file.name == name)
                .ok_or_else(|| format!("image repository stopped listing {name}"))?;
            let size = match listed.size {
                Some(size) => size,
                None => client
                    .content_length(&repo.file_url(&name))?
                    .ok_or_else(|| format!("{name}: missing content length"))?,
            };
            Ok((name, size))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let total = selected.iter().try_fold(0u64, |sum, (_, size)| {
        sum.checked_add(*size)
            .ok_or_else(|| "image source download byte count overflowed u64".to_string())
    })?;
    let completed = Arc::new(AtomicU64::new(0));
    std::fs::create_dir_all(destination)
        .map_err(|e| format!("creating {}: {e}", destination.display()))?;
    let cache = destination.join(".download-cache");
    for (remote, expected_size) in selected {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(INSTALL_CANCELLED.to_string());
        }
        let local = safe_join(destination, &remote)?;
        let stage = format!("downloading {remote}");
        progress(&stage);
        // Rebuild even a size-matching source file from the verified range
        // cache. Size alone cannot prove that a source left by a crashed pack
        // is still intact; cache entries carry hashes and remain local I/O.
        if local.exists() {
            std::fs::remove_file(&local)
                .map_err(|e| format!("removing incomplete {}: {e}", local.display()))?;
        }
        let done = Arc::clone(&completed);
        let report = Arc::clone(&on_bytes);
        let range_progress: repack::ByteProgressCallback = Arc::new(move |delta| {
            let cumulative = done.fetch_add(delta, Ordering::Relaxed) + delta;
            report(cumulative, total);
        });
        let mut source = HttpRangeSource::with_progress(repo.file_url(&remote), range_progress)
            .with_optional_token(client.token())
            .with_cache_dir(&cache);
        if let Some(cancel) = cancel {
            source = source.with_cancel(cancel.clone());
        }
        source
            .download_to(&local, expected_size)
            .map_err(|e| format!("downloading {remote}: {e}"))?;
        progress(&stage);
    }
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(INSTALL_CANCELLED.to_string());
    }
    normalize_mflux_source(repo, destination)?;
    Ok(completed.load(Ordering::Relaxed))
}

/// The scheduler shift constant of the Z-Image family. The Turbo distillation
/// ships shift 3.0; the base model ships 6.0. mflux conversions carry no
/// scheduler config at all, so the value is derived from the repository id.
fn z_image_scheduler_shift(model_id: &str) -> f64 {
    if model_id.contains("Turbo") {
        3.0
    } else {
        6.0
    }
}

fn z_image_transformer_config_json() -> &'static str {
    r#"{
  "_class_name": "ZImageTransformer2DModel",
  "all_f_patch_size": [1],
  "all_patch_size": [2],
  "axes_dims": [32, 48, 48],
  "axes_lens": [1536, 512, 512],
  "cap_feat_dim": 2560,
  "dim": 3840,
  "in_channels": 16,
  "n_heads": 30,
  "n_kv_heads": 30,
  "n_layers": 30,
  "n_refiner_layers": 2,
  "norm_eps": 1e-05,
  "qk_norm": true,
  "rope_theta": 256.0,
  "t_scale": 1000.0
}
"#
}

fn z_image_text_encoder_config_json() -> &'static str {
    r#"{
  "architectures": ["Qwen3ForCausalLM"],
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "intermediate_size": 9728,
  "max_position_embeddings": 40960,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "tie_word_embeddings": true,
  "vocab_size": 151936
}
"#
}

fn z_image_vae_config_json() -> &'static str {
    r#"{
  "_class_name": "AutoencoderKL",
  "block_out_channels": [128, 256, 512, 512],
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D"
  ],
  "up_block_types": [
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D"
  ],
  "in_channels": 3,
  "latent_channels": 16,
  "layers_per_block": 2,
  "mid_block_add_attention": true,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 0.3611,
  "shift_factor": 0.1159,
  "use_post_quant_conv": false,
  "use_quant_conv": false
}
"#
}

fn z_image_model_index_json() -> &'static str {
    r#"{
  "_class_name": "ZImagePipeline",
  "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
  "text_encoder": ["transformers", "Qwen3Model"],
  "tokenizer": ["transformers", "Qwen2Tokenizer"],
  "transformer": ["diffusers", "ZImageTransformer2DModel"],
  "vae": ["diffusers", "AutoencoderKL"]
}
"#
}

fn write_if_absent(path: &Path, contents: &str) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    std::fs::write(path, contents).map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Completes an mflux-converted source tree (for example the deepsweet
/// Z-Image MLX rows) with the config JSONs its Diffusers-layout siblings
/// ship. mflux repos carry only sharded safetensors plus a per-component
/// index whose metadata names the quantization level; the scheduler,
/// component, and quantization configs are family constants. Detection is
/// bounded by the index metadata: only trees whose transformer index
/// carries `mflux_version` and a supported `quantization_level` are
/// synthesized, so a broken Diffusers repository still fails loudly.
pub fn normalize_mflux_source(repo: &RepoRef, destination: &Path) -> Result<(), String> {
    if destination
        .join("scheduler/scheduler_config.json")
        .is_file()
    {
        return Ok(());
    }
    let index_path = destination.join("transformer/model.safetensors.index.json");
    let index_data = std::fs::read(&index_path).map_err(|e| {
        format!(
            "reading {}: {e} (image repository is missing scheduler/scheduler_config.json and no mflux transformer index was found)",
            index_path.display()
        )
    })?;
    let index: serde_json::Value = serde_json::from_slice(&index_data)
        .map_err(|e| format!("parsing {}: {e}", index_path.display()))?;
    let metadata = index
        .get("metadata")
        .and_then(|value| value.as_object())
        .ok_or_else(|| format!("{} has no metadata block", index_path.display()))?;
    let mflux_version = metadata
        .get("mflux_version")
        .and_then(|value| value.as_str())
        .ok_or_else(|| format!("{} metadata has no mflux_version", index_path.display()))?;
    let bits = metadata
        .get("quantization_level")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
        })
        .ok_or_else(|| {
            format!(
                "{} metadata has no usable quantization_level",
                index_path.display()
            )
        })?;
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return Err(format!(
            "{mflux_version} quantization_level {bits} is not a supported MLX affine width"
        ));
    }

    let shift = z_image_scheduler_shift(&repo.repo);
    write_if_absent(
        &destination.join("scheduler/scheduler_config.json"),
        &format!(
            r#"{{"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": {shift:.1}, "use_dynamic_shifting": false}}
"#
        ),
    )?;
    write_if_absent(
        &destination.join("transformer/config.json"),
        z_image_transformer_config_json(),
    )?;
    write_if_absent(
        &destination.join("text_encoder/config.json"),
        z_image_text_encoder_config_json(),
    )?;
    write_if_absent(
        &destination.join("vae/config.json"),
        z_image_vae_config_json(),
    )?;
    write_if_absent(
        &destination.join("model_index.json"),
        z_image_model_index_json(),
    )?;
    write_if_absent(
        &destination.join("quantize_config.json"),
        &format!(
            r#"{{"quantization": {{"bits": {bits}, "group_size": 64, "skip_components": []}}}}
"#
        ),
    )?;
    // The vendored Swift MLX pipeline reads this manifest directly; writing it
    // here lets both the app session and the benchmark treat the retained
    // source tree like any other quantized snapshot.
    write_if_absent(
        &destination.join("quantization.json"),
        &format!(
            r#"{{"model_id": "{}", "revision": "{}", "group_size": 64, "bits": {bits}, "mode": "affine", "layers": []}}
"#,
            repo.repo, repo.revision
        ),
    )?;
    Ok(())
}

fn select_files(files: &[String]) -> Result<Vec<String>, String> {
    let mut selected = Vec::new();
    let mut components = [false; 3];
    for file in files {
        let path = Path::new(file);
        if !safe_relative(path) {
            return Err(format!("Hugging Face file path is unsafe: {file}"));
        }
        let is_metadata = file == "scheduler/scheduler_config.json"
            || file == "scheduler/config.json"
            || file.ends_with("/config.json")
            || file.ends_with(".index.json");
        let is_tokenizer = file == "tokenizer"
            || file.starts_with("tokenizer/")
            || file.starts_with("tokenizer_config.")
            || file == "special_tokens_map.json";
        let component = if file.starts_with("text_encoder/") {
            Some(0)
        } else if file.starts_with("transformer/") {
            Some(1)
        } else if file.starts_with("vae/") {
            Some(2)
        } else {
            None
        };
        let is_weight = component.is_some() && file.ends_with(".safetensors");
        if is_metadata || is_tokenizer || is_weight {
            if let Some(index) = component.filter(|_| is_weight) {
                components[index] = true;
            }
            selected.push(file.clone());
        }
    }
    // A missing scheduler config is tolerated here because mflux-converted
    // repositories ship none; `normalize_mflux_source` either synthesizes the
    // family config or fails the install after download. Components still
    // must all be present.
    for (name, present) in [
        ("text_encoder", components[0]),
        ("transformer", components[1]),
        ("vae", components[2]),
    ] {
        if !present {
            return Err(format!("image repository is missing {name} safetensors"));
        }
    }
    selected.sort();
    selected.dedup();
    Ok(selected)
}

fn safe_join(root: &Path, remote: &str) -> Result<PathBuf, String> {
    let path = Path::new(remote);
    if !safe_relative(path) {
        return Err(format!("Hugging Face file path is unsafe: {remote}"));
    }
    Ok(root.join(path))
}

fn safe_relative(path: &Path) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn validate(entry: &ImageCatalogEntry) -> Result<(), String> {
    if entry.alias.trim().is_empty() || entry.model_id.split('/').count() != 2 {
        return Err(format!("invalid image catalog identity {:?}", entry.alias));
    }
    if entry.revision.len() != 40 || !entry.revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "image catalog entry {} is not pinned to a 40-hex revision",
            entry.alias
        ));
    }
    if entry.required_files.is_empty()
        || entry
            .required_files
            .iter()
            .any(|file| file.is_empty() || file.starts_with('/') || file.contains(".."))
    {
        return Err(format!(
            "image catalog entry {} has invalid required files",
            entry.alias
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ImageCatalog, RepoRef};

    #[test]
    fn the_embedded_image_catalog_is_pinned_and_separate() {
        let catalog = ImageCatalog::embedded().expect("image catalog");
        let entry = catalog.get("z-image-turbo").expect("Z-Image row");
        assert_eq!(entry.model_id, "Tongyi-MAI/Z-Image-Turbo");
        assert_eq!(entry.revision.len(), 40);
        assert!(entry
            .required_files
            .iter()
            .all(|file| !file.contains("text-generation")));

        for (alias, model_id, revision) in [
            (
                "z-image-turbo-mlx-2bit",
                "andrevp/Z-Image-Turbo-MLX-2bit",
                "32b4e9ceb3a813485027b1ea942f199608fb8200",
            ),
            (
                "z-image-turbo-mlx-4bit",
                "andrevp/Z-Image-Turbo-MLX-4bit",
                "9adc576198c9126874792d35569b53cf2f45a03c",
            ),
            (
                "z-image-turbo-mlx-8bit",
                "andrevp/Z-Image-Turbo-MLX-8bit",
                "c9f70995562299b1eda9b9145a94dd7a5a1ae0d6",
            ),
            (
                "z-image-turbo-mlx-fp16",
                "andrevp/Z-Image-Turbo-MLX",
                "e186d7d65d66883270671fcee05324178928ea03",
            ),
        ] {
            let entry = catalog.get(alias).expect("published MLX image row");
            assert_eq!(entry.model_id, model_id);
            assert_eq!(entry.revision, revision);
            assert_eq!(entry.required_files.len(), 17);
        }

        let q4 = catalog
            .get("z-image-turbo-mlx-q4")
            .expect("mflux Turbo row");
        assert_eq!(q4.model_id, "deepsweet/Z-Image-Turbo-6B-MLX-Q4");
        assert_eq!(q4.revision, "f4ddfcec21b9aab1b0e72c3da877d392234f3fbb");
        assert_eq!(q4.required_files.len(), 11);

        let q8 = catalog.get("z-image-mlx-q8").expect("mflux base row");
        assert_eq!(q8.model_id, "deepsweet/Z-Image-6B-MLX-Q8");
        assert_eq!(q8.revision, "730ad68a2f35d8b6f6263aaa1ba7b605d6bb8d6a");
        assert_eq!(q8.required_files.len(), 13);
    }

    #[test]
    fn select_files_tolerates_the_mflux_layout_and_components_stay_required() {
        let files = vec![
            "README.md".to_string(),
            "tokenizer/chat_template.jinja".to_string(),
            "tokenizer/tokenizer.json".to_string(),
            "tokenizer/tokenizer_config.json".to_string(),
            "text_encoder/0.safetensors".to_string(),
            "text_encoder/1.safetensors".to_string(),
            "text_encoder/model.safetensors.index.json".to_string(),
            "transformer/0.safetensors".to_string(),
            "transformer/1.safetensors".to_string(),
            "transformer/model.safetensors.index.json".to_string(),
            "vae/0.safetensors".to_string(),
            "vae/model.safetensors.index.json".to_string(),
        ];
        let selected =
            super::select_files(&files).expect("mflux source selects without a scheduler config");
        assert_eq!(selected.len(), 11);
        assert!(selected.contains(&"tokenizer/chat_template.jinja".to_string()));

        let missing_vae = vec![
            "transformer/0.safetensors".to_string(),
            "transformer/model.safetensors.index.json".to_string(),
            "text_encoder/0.safetensors".to_string(),
        ];
        let error = super::select_files(&missing_vae).expect_err("components must stay required");
        assert!(error.contains("vae"), "{error}");
    }

    #[test]
    fn normalize_mflux_source_synthesizes_the_family_configs() {
        let root =
            std::env::temp_dir().join(format!("turbospark-mflux-normalize-{}", std::process::id()));
        let transformer = root.join("transformer");
        std::fs::create_dir_all(&transformer).expect("transformer dir");
        std::fs::write(
            transformer.join("model.safetensors.index.json"),
            r#"{"metadata": {"mflux_version": "0.17.5", "quantization_level": "4"}, "weight_map": {"x_pad_token": "0.safetensors"}}"#,
        )
        .expect("index");

        let repo = RepoRef::new(
            "deepsweet/Z-Image-Turbo-6B-MLX-Q4",
            "f4ddfcec21b9aab1b0e72c3da877d392234f3fbb",
        );
        super::normalize_mflux_source(&repo, &root).expect("normalize");

        let scheduler = std::fs::read_to_string(root.join("scheduler/scheduler_config.json"))
            .expect("scheduler");
        assert!(scheduler.contains("\"shift\": 3.0"), "{scheduler}");
        let quantization =
            std::fs::read_to_string(root.join("quantization.json")).expect("manifest");
        assert!(quantization.contains("\"bits\": 4"), "{quantization}");
        assert!(
            quantization.contains("\"group_size\": 64"),
            "{quantization}"
        );
        assert!(root.join("transformer/config.json").is_file());
        assert!(root.join("text_encoder/config.json").is_file());
        assert!(root.join("vae/config.json").is_file());
        assert!(root.join("model_index.json").is_file());
        assert!(root.join("quantize_config.json").is_file());

        // The base model carries the higher scheduler shift.
        let base_root = root.join("base");
        std::fs::create_dir_all(base_root.join("transformer")).expect("base transformer dir");
        std::fs::write(
            base_root.join("transformer/model.safetensors.index.json"),
            r#"{"metadata": {"mflux_version": "0.17.5", "quantization_level": 8}, "weight_map": {}}"#,
        )
        .expect("base index");
        let base_repo = RepoRef::new(
            "deepsweet/Z-Image-6B-MLX-Q8",
            "730ad68a2f35d8b6f6263aaa1ba7b605d6bb8d6a",
        );
        super::normalize_mflux_source(&base_repo, &base_root).expect("normalize base");
        let base_scheduler =
            std::fs::read_to_string(base_root.join("scheduler/scheduler_config.json"))
                .expect("base scheduler");
        assert!(
            base_scheduler.contains("\"shift\": 6.0"),
            "{base_scheduler}"
        );

        // A Diffusers tree without a scheduler config is not silently fixed.
        let diffusers_root = root.join("diffusers");
        std::fs::create_dir_all(diffusers_root.join("transformer")).expect("diffusers dir");
        std::fs::write(
            diffusers_root.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
            r#"{"weight_map": {}}"#,
        )
        .expect("diffusers index");
        let error = super::normalize_mflux_source(&repo, &diffusers_root)
            .expect_err("non-mflux tree must fail loudly");
        assert!(
            error.contains("missing scheduler/scheduler_config.json")
                || error.contains("mflux_version"),
            "{error}"
        );

        std::fs::remove_dir_all(&root).ok();
    }
}
