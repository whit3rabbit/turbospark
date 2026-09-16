//! Materialize the small Diffusers source tree needed by the image packer.

use std::path::{Component, Path, PathBuf};

use catalog::{Client, RepoRef};

/// Download a pinned Diffusers image repository into a temporary source tree.
///
/// The image packer owns conversion and validation. This module only selects
/// the tokenizer, scheduler metadata, component indexes, and safetensors
/// payloads that the packer can consume. Files are streamed by the catalog
/// client, so a shard is never held as one in-memory response body.
pub(super) fn materialize(
    client: &Client,
    repo: &RepoRef,
    destination: &Path,
) -> Result<u64, String> {
    if destination.exists() {
        return Err(format!(
            "refusing to overwrite image source staging directory {}",
            destination.display()
        ));
    }
    if repo.revision.len() != 40 || !repo.revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "image repository revision {:?} is not an immutable 40-hex commit",
            repo.revision
        ));
    }
    let files = client.file_list(repo)?;
    let selected = select_files(&files)?;
    let mut bytes: u64 = 0;
    for remote in selected {
        let local = safe_join(destination, &remote)?;
        bytes = bytes
            .checked_add(client.download_to(&repo.file_url(&remote), &local)?)
            .ok_or_else(|| "image source download byte count overflowed u64".to_string())?;
    }
    Ok(bytes)
}

fn select_files(files: &[String]) -> Result<Vec<String>, String> {
    let mut selected = Vec::new();
    let mut has_scheduler = false;
    let mut components = [false; 3];
    for file in files {
        let path = Path::new(file);
        if !safe_relative(path) {
            return Err(format!("Hugging Face file path is unsafe: {file}"));
        }
        let name = file.as_str();
        let is_metadata = name == "scheduler/scheduler_config.json"
            || name == "scheduler/config.json"
            || name.ends_with("/config.json")
            || name.ends_with(".index.json");
        let is_tokenizer = name == "tokenizer"
            || name.starts_with("tokenizer/")
            || name.starts_with("tokenizer_config.")
            || name == "special_tokens_map.json";
        let component = if name.starts_with("text_encoder/") {
            Some(0)
        } else if name.starts_with("transformer/") {
            Some(1)
        } else if name.starts_with("vae/") {
            Some(2)
        } else {
            None
        };
        let is_weight = component.is_some() && name.ends_with(".safetensors");
        if is_metadata || is_tokenizer || is_weight {
            if name == "scheduler/scheduler_config.json" || name == "scheduler/config.json" {
                has_scheduler = true;
            }
            if is_weight {
                let index = component.expect("weight paths have a component");
                components[index] = true;
            }
            selected.push(file.clone());
        }
    }
    if !has_scheduler {
        return Err("image repository is missing scheduler/scheduler_config.json".to_string());
    }
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

#[cfg(test)]
mod tests {
    use super::select_files;

    #[test]
    fn selects_the_complete_diffusers_image_source_contract() {
        let files = vec![
            "README.md".to_string(),
            "scheduler/scheduler_config.json".to_string(),
            "tokenizer/tokenizer.json".to_string(),
            "tokenizer_config.json".to_string(),
            "text_encoder/config.json".to_string(),
            "text_encoder/model.safetensors.index.json".to_string(),
            "text_encoder/model-00001.safetensors".to_string(),
            "transformer/config.json".to_string(),
            "transformer/diffusion_pytorch_model.safetensors.index.json".to_string(),
            "transformer/diffusion_pytorch_model-00001.safetensors".to_string(),
            "vae/config.json".to_string(),
            "vae/diffusion_pytorch_model.safetensors".to_string(),
        ];
        let selected = select_files(&files).expect("complete source");
        assert!(!selected.iter().any(|file| file == "README.md"));
        assert_eq!(selected.len(), 11);
    }

    #[test]
    fn rejects_missing_component_and_unsafe_paths() {
        let missing = vec![
            "scheduler/scheduler_config.json".to_string(),
            "text_encoder/model.safetensors".to_string(),
            "transformer/model.safetensors".to_string(),
        ];
        let error = select_files(&missing).expect_err("VAE must be required");
        assert!(error.contains("vae"), "{error}");

        let unsafe_file = vec![
            "scheduler/scheduler_config.json".to_string(),
            "../vae/model.safetensors".to_string(),
        ];
        let error = select_files(&unsafe_file).expect_err("parent path must be refused");
        assert!(error.contains("unsafe"), "{error}");
    }
}
