//! Decides, before anything is loaded, whether a directory is an embedding
//! model this runtime can run, and which family it is.
//!
//! The embedding paths are the Core ML EmbeddingGemma 2 export (recognized by
//! its files, since it ships no `model_type`) and the BERT / XLM-RoBERTa
//! [`EncoderRunner`](crate::EncoderRunner). Everything else is refused by
//! family with a reason a host can show, instead of loading and returning
//! wrong vectors.

use std::path::Path;

use crate::coreml::{detect_layout, CoreMlLayout};
use crate::encoder;

/// What [`probe_embedding_dir`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingDirProbe {
    /// The family the directory declares (`embeddinggemma2`, `bert`,
    /// `xlm-roberta`, or a refused one such as `gte`), when it names one.
    pub family: Option<String>,
    /// `coreml` or `encoder`: which runner would open it.
    pub backend: Option<&'static str>,
    pub runnable: bool,
    pub refused_because: Option<String>,
}

impl EmbeddingDirProbe {
    fn ok(family: &str, backend: &'static str) -> Self {
        Self {
            family: Some(family.to_string()),
            backend: Some(backend),
            runnable: true,
            refused_because: None,
        }
    }

    fn refuse(family: Option<String>, backend: Option<&'static str>, why: String) -> Self {
        Self {
            family,
            backend,
            runnable: false,
            refused_because: Some(why),
        }
    }
}

pub fn probe_embedding_dir(dir: &Path) -> EmbeddingDirProbe {
    if !dir.is_dir() {
        return EmbeddingDirProbe::refuse(
            None,
            None,
            format!("{} is not a model directory", dir.display()),
        );
    }
    match detect_layout(dir) {
        Some(CoreMlLayout::EmbeddingGemma2) => {
            // The same predicate `EmbeddingGemma2CoreMl::open` refuses on.
            return if turbospark_coreml::supports(turbospark_coreml::Capability::MultiFunction) {
                EmbeddingDirProbe::ok(CoreMlLayout::EmbeddingGemma2.family(), "coreml")
            } else {
                EmbeddingDirProbe::refuse(
                    Some(CoreMlLayout::EmbeddingGemma2.family().into()),
                    Some("coreml"),
                    "EmbeddingGemma 2 on Core ML requires macOS 15 or newer".into(),
                )
            };
        }
        Some(layout @ CoreMlLayout::ParakeetTdt) => {
            return EmbeddingDirProbe::refuse(
                Some(layout.family().into()),
                Some("coreml"),
                "this is a speech-to-text model, not an embedding model".into(),
            );
        }
        None => {}
    }

    let config_path = dir.join("config.json");
    let config: serde_json::Value = match std::fs::read(&config_path)
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
    {
        Ok(v) => v,
        Err(e) => {
            return EmbeddingDirProbe::refuse(
                None,
                None,
                format!("no readable config.json ({e}); not a supported embedding model"),
            )
        }
    };
    let declared = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let model_type = match encoder::check_model_type(&config) {
        Ok(t) => t,
        Err(e) => return EmbeddingDirProbe::refuse(declared, None, e),
    };
    if let Err(e) = encoder::check_pooling(dir) {
        return EmbeddingDirProbe::refuse(declared, Some("encoder"), e);
    }
    // `EncoderRunner::open` needs the weights; the server and the app embed
    // text, which needs the tokenizer the runner treats as optional.
    for file in ["model.safetensors", "tokenizer.json"] {
        if !dir.join(file).is_file() {
            return EmbeddingDirProbe::refuse(
                declared,
                Some("encoder"),
                format!("{file} is missing"),
            );
        }
    }
    EmbeddingDirProbe::ok(model_type, "encoder")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct Tmp(PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tmp(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!(
            "ts-embed-probe-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    fn encoder_dir(tag: &str, model_type: Option<&str>) -> Tmp {
        let t = tmp(tag);
        let cfg = match model_type {
            Some(m) => format!(r#"{{"model_type":"{m}","hidden_size":8}}"#),
            None => r#"{"hidden_size":8}"#.to_string(),
        };
        std::fs::write(t.0.join("config.json"), cfg).unwrap();
        std::fs::write(t.0.join("model.safetensors"), b"x").unwrap();
        std::fs::write(t.0.join("tokenizer.json"), b"{}").unwrap();
        t
    }

    #[test]
    fn supported_encoder_families_are_runnable() {
        for m in ["bert", "xlm-roberta"] {
            let t = encoder_dir(m, Some(m));
            let p = probe_embedding_dir(&t.0);
            assert!(p.runnable, "{m}: {p:?}");
            assert_eq!(p.family.as_deref(), Some(m));
            assert_eq!(p.backend, Some("encoder"));
        }
    }

    #[test]
    fn other_families_are_refused_by_name() {
        for m in ["gte", "nomic_bert", "gemma3_text", "roberta"] {
            let t = encoder_dir(m, Some(m));
            let p = probe_embedding_dir(&t.0);
            assert!(!p.runnable, "{m}");
            assert_eq!(p.family.as_deref(), Some(m));
            assert!(p.refused_because.unwrap().contains(m));
        }
        let t = encoder_dir("none", None);
        let p = probe_embedding_dir(&t.0);
        assert!(!p.runnable);
        assert!(p.refused_because.unwrap().contains("no model_type"));
    }

    #[test]
    fn mean_pooled_bert_is_refused_and_cls_is_not() {
        let t = encoder_dir("mean", Some("bert"));
        std::fs::create_dir_all(t.0.join("1_Pooling")).unwrap();
        let pooling = |cls: bool, mean: bool| {
            format!(r#"{{"pooling_mode_cls_token":{cls},"pooling_mode_mean_tokens":{mean}}}"#)
        };
        std::fs::write(t.0.join("1_Pooling/config.json"), pooling(false, true)).unwrap();
        let p = probe_embedding_dir(&t.0);
        assert!(!p.runnable);
        assert!(p.refused_because.unwrap().contains("CLS"));
        std::fs::write(t.0.join("1_Pooling/config.json"), pooling(true, true)).unwrap();
        assert!(!probe_embedding_dir(&t.0).runnable);
        std::fs::write(t.0.join("1_Pooling/config.json"), pooling(true, false)).unwrap();
        assert!(probe_embedding_dir(&t.0).runnable);
    }

    #[test]
    fn missing_files_and_non_dirs_are_refused() {
        let t = encoder_dir("files", Some("bert"));
        std::fs::remove_file(t.0.join("tokenizer.json")).unwrap();
        let p = probe_embedding_dir(&t.0);
        assert!(!p.runnable);
        assert!(p.refused_because.unwrap().contains("tokenizer.json"));
        assert!(!probe_embedding_dir(&t.0.join("nope")).runnable);
        let empty = tmp("empty");
        assert!(!probe_embedding_dir(&empty.0).runnable);
    }

    #[test]
    fn core_ml_layouts_are_recognized_by_files() {
        let t = tmp("gemma2");
        std::fs::create_dir_all(t.0.join("EmbeddingGemma2Text.mlpackage")).unwrap();
        for f in ["embeddings.bf16", "tokenizer.json", "config.json"] {
            std::fs::write(t.0.join(f), b"{}").unwrap();
        }
        let p = probe_embedding_dir(&t.0);
        assert_eq!(p.family.as_deref(), Some("embeddinggemma2"));
        assert_eq!(p.backend, Some("coreml"));
        assert_eq!(
            p.runnable,
            turbospark_coreml::supports(turbospark_coreml::Capability::MultiFunction)
        );

        let t = tmp("parakeet");
        for d in crate::coreml::parakeet::REQUIRED_DIRS {
            std::fs::create_dir_all(t.0.join(d)).unwrap();
        }
        std::fs::write(t.0.join(crate::coreml::parakeet::VOCAB_FILE), b"{}").unwrap();
        let p = probe_embedding_dir(&t.0);
        assert!(!p.runnable);
        assert!(p.refused_because.unwrap().contains("speech-to-text"));
    }

    #[test]
    fn encoder_open_checks_family_before_weights() {
        let t = encoder_dir("open", Some("gte"));
        // model.safetensors is one garbage byte: reaching it would fail with
        // a safetensors error, not a model_type one.
        let err = crate::EncoderRunner::open(&t.0).err().unwrap();
        assert!(err.contains("model_type \"gte\""), "{err}");
    }
}
