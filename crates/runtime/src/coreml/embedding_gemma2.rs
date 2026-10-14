//! EmbeddingGemma 2 on Core ML (FluidInference `embeddinggemma-2-coreml`).
//!
//! The text encoder ships as one multifunction package with fixed-shape
//! functions `embed_{32,48,64,128,256,512}`. The host tokenizes, wraps the
//! ids as `<bos> ... <eos>`, gathers rows from the memory-mapped bf16 token
//! table, scales them by sqrt(512), and calls the smallest function that
//! fits. Mean pooling and L2 normalization happen inside the graph; the
//! output is a 768-d unit vector. Matryoshka sizes (512, 256, 128) keep the
//! leading values and re-normalize, as sentence-transformers does.
//!
//! Audio and image inputs (`embedding_gemma2_media`) produce soft tokens in
//! the same 512-d space; they are wrapped with their boundary tokens and go
//! through the same text functions.
//!
//! Reference: the model card and FluidUse `EmbeddingGemma2Manager`
//! (FluidInference/FluidUse at `d278e17df85e39e8c21bb64a467da81c39576fcd`).
//! `pack_256` (eight texts per call) is not used: it needs its own parity
//! gate against the single-text functions first.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use model_io::ResidentBuffer;
use turbospark_coreml::{ComputeUnits, CoreMlModel, LoadOptions, Tensor};

pub const TEXT_PACKAGE: &str = "EmbeddingGemma2Text.mlpackage";
pub const TOKEN_TABLE: &str = "embeddings.bf16";
pub const CONFIG: &str = "config.json";
pub const TOKENIZER: &str = "tokenizer.json";

const UNITS_ENV: &str = "TURBOSPARK_EMBEDDINGGEMMA2_COREML_UNITS";

/// Placement of the text functions. The export puts every op on the Neural
/// Engine (its stated target); the first load compiles each function for
/// it once, which takes minutes and is cached by the OS afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingGemma2Options {
    pub text_units: ComputeUnits,
    /// Audio and vision towers. The card measured both faster on the GPU,
    /// and the vision tower's chunked attention falls back to the CPU on
    /// the Neural Engine.
    pub media_units: ComputeUnits,
}

impl Default for EmbeddingGemma2Options {
    fn default() -> Self {
        Self {
            text_units: ComputeUnits::CpuAndNeuralEngine,
            media_units: ComputeUnits::CpuAndGpu,
        }
    }
}

impl EmbeddingGemma2Options {
    /// Defaults, with the text placement overridable through
    /// `TURBOSPARK_EMBEDDINGGEMMA2_COREML_UNITS` (`cpu`, `cpu_gpu`,
    /// `cpu_ane`, `all`).
    pub fn from_env() -> Result<Self, String> {
        let mut o = Self::default();
        if let Ok(v) = std::env::var(UNITS_ENV) {
            o.text_units = ComputeUnits::parse(v.trim()).ok_or_else(|| {
                format!("{UNITS_ENV}={v} is not one of cpu, cpu_gpu, cpu_ane, all")
            })?;
        }
        Ok(o)
    }
}

/// Ids of the boundary tokens, resolved from the tokenizer and checked
/// against `config.json`. A disagreement refuses the model: a wrong id
/// embeds a different token silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpecialIds {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    pub image_start: u32,
    pub image_end: u32,
}

/// What `config.json` declares and the host relies on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Config {
    pub hidden: usize,
    pub dimension: usize,
    pub matryoshka: Vec<usize>,
    pub embed_scale: f32,
    pub table_rows: usize,
    /// Text function lengths, ascending.
    pub lengths: Vec<usize>,
    pub prompts: BTreeMap<String, String>,
}

fn usize_at(v: &serde_json::Value, path: &[&str]) -> Result<usize, String> {
    let mut cur = v;
    for p in path {
        cur = cur
            .get(p)
            .ok_or_else(|| format!("config.json is missing {}", path.join(".")))?;
    }
    cur.as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| format!("config.json {} is not an unsigned integer", path.join(".")))
}

pub(crate) fn parse_config(v: &serde_json::Value) -> Result<Config, String> {
    let hidden = usize_at(v, &["hidden_size"])?;
    let dimension = usize_at(v, &["embedding_dim"])?;
    let embed_scale = v
        .get("embed_scale")
        .and_then(|x| x.as_f64())
        .ok_or("config.json embed_scale is missing")? as f32;
    // The scale is sqrt(hidden) in the reference; anything else means a
    // different export.
    if (embed_scale - (hidden as f32).sqrt()).abs() > 1e-4 {
        return Err(format!(
            "config.json embed_scale {embed_scale} is not sqrt(hidden_size {hidden})"
        ));
    }
    let table = v
        .get("token_table")
        .ok_or("config.json token_table is missing")?;
    if table.get("dtype").and_then(|d| d.as_str()) != Some("bfloat16")
        || table.get("file").and_then(|d| d.as_str()) != Some(TOKEN_TABLE)
    {
        return Err("config.json token_table must be bfloat16 embeddings.bf16".into());
    }
    let table_rows = table
        .get("shape")
        .and_then(|s| s.get(0))
        .and_then(|x| x.as_u64())
        .ok_or("config.json token_table.shape is missing")? as usize;
    let table_cols = table
        .get("shape")
        .and_then(|s| s.get(1))
        .and_then(|x| x.as_u64())
        .ok_or("config.json token_table.shape is missing")? as usize;
    if table_cols != hidden {
        return Err(format!(
            "token table width {table_cols} does not match hidden_size {hidden}"
        ));
    }
    let mut lengths: Vec<usize> = v
        .get("functions")
        .and_then(|f| f.as_object())
        .ok_or("config.json functions is missing")?
        .iter()
        .filter_map(|(name, spec)| {
            let n = name.strip_prefix("embed_")?.parse::<usize>().ok()?;
            (spec.get("tokens").and_then(|t| t.as_u64()) == Some(n as u64)).then_some(n)
        })
        .collect();
    lengths.sort_unstable();
    if lengths.is_empty() {
        return Err("config.json declares no embed_<S> functions".into());
    }
    let matryoshka: Vec<usize> = v
        .get("matryoshka_dims")
        .and_then(|m| m.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_else(|| vec![dimension]);
    if matryoshka.iter().any(|&d| d == 0 || d > dimension) {
        return Err("config.json matryoshka_dims are out of range".into());
    }
    let prompts = v
        .get("prompts")
        .and_then(|p| p.as_object())
        .map(|o| {
            o.iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default();
    Ok(Config {
        hidden,
        dimension,
        matryoshka,
        embed_scale,
        table_rows,
        lengths,
        prompts,
    })
}

fn special_ids(tok: &tokenizer::Tokenizer, v: &serde_json::Value) -> Result<SpecialIds, String> {
    let resolve = |token: &str, config_path: &[&str]| -> Result<u32, String> {
        let id = tok
            .token_to_id(token)
            .ok_or_else(|| format!("tokenizer.json has no {token}"))?;
        let declared = usize_at(v, config_path)?;
        if declared != id as usize {
            return Err(format!(
                "tokenizer.json resolves {token} to {id}, config.json {} says {declared}",
                config_path.join(".")
            ));
        }
        Ok(id)
    };
    Ok(SpecialIds {
        bos: resolve("<bos>", &["bos_token_id"])?,
        eos: resolve("<eos>", &["eos_token_id"])?,
        pad: resolve("<pad>", &["pad_token_id"])?,
        audio_start: resolve("<|audio>", &["audio", "boa_token_id"])?,
        audio_end: resolve("<audio|>", &["audio", "eoa_token_id"])?,
        image_start: resolve("<|image>", &["vision", "boi_token_id"])?,
        image_end: resolve("<image|>", &["vision", "eoi_token_id"])?,
    })
}

/// Row-major bf16 table `[rows, cols]`, memory-mapped.
pub(crate) struct Bf16Table {
    buf: ResidentBuffer,
    rows: usize,
    cols: usize,
}

impl Bf16Table {
    pub fn open(path: &Path, rows: usize, cols: usize) -> Result<Self, String> {
        let len = std::fs::metadata(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        let want = (rows * cols * 2) as u64;
        if len != want {
            return Err(format!(
                "{} is {len} bytes, expected {want} for [{rows}, {cols}] bf16",
                path.display()
            ));
        }
        let buf = ResidentBuffer::map(path, 0, len).map_err(|e| e.to_string())?;
        Ok(Self { buf, rows, cols })
    }

    /// Appends row `id` times `scale` to `out` as f32.
    pub fn push_row(&self, id: u32, scale: f32, out: &mut Vec<f32>) -> Result<(), String> {
        let id = id as usize;
        if id >= self.rows {
            return Err(format!("token {id} is outside the {}-row table", self.rows));
        }
        let bytes = &self.buf.data()[id * self.cols * 2..(id + 1) * self.cols * 2];
        out.extend(bytes.chunks_exact(2).map(|b| {
            // bf16 is the top half of an f32.
            f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16) * scale
        }));
        Ok(())
    }
}

/// L2-normalizes the leading `dim` values of `v`. The full-size output is
/// already unit length; this re-normalizes after a Matryoshka truncation.
pub fn truncate_normalize(v: &[f32], dim: usize) -> Vec<f32> {
    let head = &v[..dim.min(v.len())];
    let norm = head.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        head.iter().map(|x| x / norm).collect()
    } else {
        head.to_vec()
    }
}

/// The smallest function length that holds `n` tokens.
pub(crate) fn pick_length(lengths: &[usize], n: usize) -> Option<usize> {
    lengths.iter().copied().find(|&l| l >= n)
}

/// One row of input: a token id (looked up in the table) or a soft token
/// already in the model's input space (audio or image).
pub(crate) enum Row<'a> {
    Token(u32),
    Soft(&'a [f32]),
}

pub struct EmbeddingGemma2CoreMl {
    pub(crate) dir: PathBuf,
    pub(crate) config: Config,
    pub(crate) ids: SpecialIds,
    tokenizer: tokenizer::Tokenizer,
    table: Bf16Table,
    compiled_text: PathBuf,
    pub(crate) opts: EmbeddingGemma2Options,
    /// Text functions by length, loaded on first use. Each one is its own
    /// MLModel (and its own Neural Engine compile).
    functions: Mutex<HashMap<usize, Arc<CoreMlModel>>>,
    pub(crate) media: crate::coreml::embedding_gemma2_media::MediaTowers,
}

impl std::fmt::Debug for EmbeddingGemma2CoreMl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemma2CoreMl")
            .field("dir", &self.dir)
            .field("config", &self.config)
            .field("opts", &self.opts)
            .finish_non_exhaustive()
    }
}

impl EmbeddingGemma2CoreMl {
    pub fn open(dir: &Path, opts: &EmbeddingGemma2Options) -> Result<Self, String> {
        if !turbospark_coreml::supports(turbospark_coreml::Capability::MultiFunction) {
            return Err(
                "EmbeddingGemma 2 on Core ML requires macOS 15 or newer (multifunction models)"
                    .into(),
            );
        }
        let config_value: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join(CONFIG)).map_err(|e| format!("read {CONFIG}: {e}"))?,
        )
        .map_err(|e| format!("{CONFIG}: {e}"))?;
        let config = parse_config(&config_value)?;
        let mut tokenizer = tokenizer::Tokenizer::from_file(dir.join(TOKENIZER))
            .map_err(|e| format!("{TOKENIZER}: {e}"))?;
        // The host owns truncation and padding; whatever the file sets is
        // cleared so it cannot apply twice.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| format!("{TOKENIZER}: {e}"))?;
        let ids = special_ids(&tokenizer, &config_value)?;
        let table = Bf16Table::open(&dir.join(TOKEN_TABLE), config.table_rows, config.hidden)?;
        let compiled_text =
            turbospark_coreml::compile_if_needed(&dir.join(TEXT_PACKAGE), &super::cache_root(dir))
                .map_err(|e| e.to_string())?;
        let media =
            crate::coreml::embedding_gemma2_media::MediaTowers::new(dir, &config_value, opts)?;
        let model = Self {
            dir: dir.to_path_buf(),
            config,
            ids,
            tokenizer,
            table,
            compiled_text,
            opts: *opts,
            functions: Mutex::new(HashMap::new()),
            media,
        };
        // Load (and check) the smallest function now, so a broken package
        // fails at open rather than on the first request.
        model.function(model.config.lengths[0])?;
        Ok(model)
    }

    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Rows in the token table.
    pub fn vocab_size(&self) -> usize {
        self.config.table_rows
    }

    pub fn matryoshka_dims(&self) -> &[usize] {
        &self.config.matryoshka
    }

    /// Longest input in tokens, `<bos>` and `<eos>` included.
    pub fn max_tokens(&self) -> usize {
        *self.config.lengths.last().expect("checked non-empty")
    }

    /// Task names with a trained prefix (`search_query`, `document`, ...).
    pub fn prompt_names(&self) -> impl Iterator<Item = &str> {
        self.config.prompts.keys().map(String::as_str)
    }

    fn function(&self, length: usize) -> Result<Arc<CoreMlModel>, String> {
        let mut map = self.functions.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(m) = map.get(&length) {
            return Ok(m.clone());
        }
        let opts = LoadOptions::new(self.opts.text_units).with_function(format!("embed_{length}"));
        let m = CoreMlModel::load(&self.compiled_text, &opts).map_err(|e| e.to_string())?;
        let want = [
            ("inputs_embeds", vec![1, length, self.config.hidden]),
            ("attention_mask", vec![1, length]),
        ];
        for (name, shape) in want {
            let ok = m.inputs().iter().any(|s| {
                s.name == name
                    && matches!(&s.kind, turbospark_coreml::FeatureKind::MultiArray { shape: Some(got), .. } if *got == shape)
            });
            if !ok {
                return Err(format!("embed_{length}: input {name} is not {shape:?}"));
            }
        }
        let m = Arc::new(m);
        map.insert(length, m.clone());
        Ok(m)
    }

    /// `<bos>` + tokens of `prefix + text` + `<eos>`, keeping the leading
    /// tokens when the text is longer than the largest function.
    pub fn token_ids(&self, text: &str, task: Option<&str>) -> Result<Vec<u32>, String> {
        let prefix = match task {
            None | Some("") | Some("none") => "",
            Some(t) => self
                .config
                .prompts
                .get(t)
                .ok_or_else(|| {
                    format!(
                        "unknown task {t:?}; known: {}",
                        self.prompt_names().collect::<Vec<_>>().join(", ")
                    )
                })?
                .as_str(),
        };
        let enc = self
            .tokenizer
            .encode(format!("{prefix}{text}"), false)
            .map_err(|e| e.to_string())?;
        let body = enc.get_ids();
        let keep = body.len().min(self.max_tokens() - 2);
        let mut ids = Vec::with_capacity(keep + 2);
        ids.push(self.ids.bos);
        ids.extend_from_slice(&body[..keep]);
        ids.push(self.ids.eos);
        Ok(ids)
    }

    /// Embeds each text, in order. `dimension` picks a Matryoshka size
    /// (default: the full size).
    pub fn embed_texts(
        &self,
        texts: &[&str],
        task: Option<&str>,
        dimension: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let dim = self.check_dimension(dimension)?;
        texts
            .iter()
            .map(|t| {
                let ids = self.token_ids(t, task)?;
                let rows: Vec<Row> = ids.iter().map(|&id| Row::Token(id)).collect();
                Ok(truncate_normalize(&self.embed_rows(&rows)?, dim))
            })
            .collect()
    }

    pub(crate) fn check_dimension(&self, dimension: Option<usize>) -> Result<usize, String> {
        match dimension {
            None => Ok(self.config.dimension),
            Some(d) if self.config.matryoshka.contains(&d) => Ok(d),
            Some(d) => Err(format!(
                "dimension {d} is not one of {:?}",
                self.config.matryoshka
            )),
        }
    }

    /// Runs one text function over `rows` (padded with `<pad>` rows and
    /// masked out) and returns the model's unit-length embedding.
    pub(crate) fn embed_rows(&self, rows: &[Row]) -> Result<Vec<f32>, String> {
        let h = self.config.hidden;
        let length = pick_length(&self.config.lengths, rows.len()).ok_or_else(|| {
            format!(
                "{} rows exceed the longest function ({})",
                rows.len(),
                self.max_tokens()
            )
        })?;
        let mut embeds = Vec::with_capacity(length * h);
        for row in rows {
            match row {
                Row::Token(id) => self
                    .table
                    .push_row(*id, self.config.embed_scale, &mut embeds)?,
                Row::Soft(v) => {
                    if v.len() != h {
                        return Err(format!("soft token has {} values, expected {h}", v.len()));
                    }
                    embeds.extend_from_slice(v);
                }
            }
        }
        // Padding rows hold the <pad> embedding, as the reference does; the
        // mask keeps them out of attention and pooling.
        for _ in rows.len()..length {
            self.table
                .push_row(self.ids.pad, self.config.embed_scale, &mut embeds)?;
        }
        let mask: Vec<f32> = (0..length)
            .map(|i| f32::from(u8::from(i < rows.len())))
            .collect();
        let embeds = Tensor::f16_from_f32("inputs_embeds", vec![1, length, h], &embeds)
            .map_err(|e| e.to_string())?;
        let mask = Tensor::f16_from_f32("attention_mask", vec![1, length], &mask)
            .map_err(|e| e.to_string())?;
        let out = self
            .function(length)?
            .predict(&[("inputs_embeds", &embeds), ("attention_mask", &mask)])
            .map_err(|e| e.to_string())?;
        let v = out
            .get("embedding")
            .and_then(Tensor::to_f32_vec)
            .ok_or("embedding output is missing or not float")?;
        if v.len() != self.config.dimension || v.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "embedding has {} values or a non-finite value",
                v.len()
            ));
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_json() -> serde_json::Value {
        serde_json::json!({
            "hidden_size": 512,
            "embedding_dim": 768,
            "matryoshka_dims": [768, 512, 256, 128],
            "embed_scale": 22.627417,
            "token_table": {"file": "embeddings.bf16", "dtype": "bfloat16", "shape": [262144, 512]},
            "functions": {
                "embed_128": {"tokens": 128}, "embed_32": {"tokens": 32},
                "pack_256": {"tokens": 256, "slots": 8}, "embed_512": {"tokens": 512}
            },
            "prompts": {"search_query": "task: search result | query: "}
        })
    }

    #[test]
    fn config_parses_sorted_lengths_and_refuses_mismatches() {
        let c = parse_config(&config_json()).unwrap();
        assert_eq!(c.lengths, vec![32, 128, 512]);
        assert_eq!(c.table_rows, 262144);
        assert_eq!(c.prompts["search_query"], "task: search result | query: ");

        let mut bad = config_json();
        bad["embed_scale"] = serde_json::json!(16.0);
        assert!(parse_config(&bad).is_err());
        let mut bad = config_json();
        bad["token_table"]["shape"] = serde_json::json!([262144, 768]);
        assert!(parse_config(&bad).is_err());
        let mut bad = config_json();
        bad["token_table"]["dtype"] = serde_json::json!("float16");
        assert!(parse_config(&bad).is_err());
        let mut bad = config_json();
        bad["matryoshka_dims"] = serde_json::json!([1024]);
        assert!(parse_config(&bad).is_err());
    }

    #[test]
    fn pick_length_takes_the_smallest_fit() {
        let l = [32, 48, 64, 128, 256, 512];
        assert_eq!(pick_length(&l, 1), Some(32));
        assert_eq!(pick_length(&l, 32), Some(32));
        assert_eq!(pick_length(&l, 33), Some(48));
        assert_eq!(pick_length(&l, 512), Some(512));
        assert_eq!(pick_length(&l, 513), None);
    }

    #[test]
    fn truncate_normalize_renormalizes_the_head() {
        let v = truncate_normalize(&[3.0, 4.0, 12.0], 2);
        assert_eq!(v, vec![0.6, 0.8]);
        assert_eq!(truncate_normalize(&[0.0, 0.0], 2), vec![0.0, 0.0]);
    }

    #[test]
    fn bf16_rows_decode_and_scale() {
        let dir = std::env::temp_dir().join(format!(
            "ts-eg2-table-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.bf16");
        // Two rows of two: 1.0, -2.0 | 0.5, 3.0 as bf16 (top 16 bits).
        let vals = [1.0f32, -2.0, 0.5, 3.0];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let t = Bf16Table::open(&path, 2, 2).unwrap();
        let mut out = Vec::new();
        t.push_row(1, 2.0, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 6.0]);
        assert!(t.push_row(2, 1.0, &mut out).is_err());
        assert!(
            Bf16Table::open(&path, 3, 2).is_err(),
            "size mismatch must refuse"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
