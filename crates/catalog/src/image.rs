//! Curated Diffusers image sources, separate from the text-model catalog.

use serde::Deserialize;

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
    use super::ImageCatalog;

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
    }
}
