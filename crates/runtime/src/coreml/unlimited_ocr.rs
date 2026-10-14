//! Unlimited OCR on Core ML (FluidInference `unlimited-ocr-mlx-coreml`, the
//! Core ML parts only).
//!
//! baidu/Unlimited-OCR (DeepSeek-OCR architecture, a 12-layer MoE decoder)
//! split into two Core ML packages:
//!
//! - Page encoder (SAM-B + CLIP-L + projector): one padded 1024 x 1024 page
//!   to 273 embeddings. GPU only; the card reports wrong output on the
//!   Neural Engine.
//! - Decoder: per prefix-capacity bucket P, a `prefill_P` function (128
//!   rows per call) and a `decode_P` function (one row), sharing weights.
//!   The K/V cache is Core ML state: P prefix slots plus a 128-slot ring
//!   for generated tokens (upstream's R-SWA keeps the whole prefix and the
//!   last 128 generated tokens). Outputs are the top 16 logits of each
//!   vocabulary half.
//!
//! The host builds the prefix (`<bos>`, page embeddings, prompt ids),
//! RoPE tables, masks and ring slots, copies the prefill cache into the
//! decode state, merges the two top-k halves, and applies upstream's
//! 35-gram no-repeat ban. Decoding is greedy over the unbanned candidates.
//!
//! Reference: FluidUse `UnlimitedOcrManager`, `UnlimitedOcrPagePreprocessor`
//! and `UnlimitedOcrTokenDecoder` (FluidInference/FluidUse at
//! `d278e17df85e39e8c21bb64a467da81c39576fcd`). Page padding follows PIL
//! `ImageOps.pad` (upstream's base mode) rather than FluidUse's CoreGraphics
//! draw, so resizing is PIL bicubic via `vision-io`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use model_io::ResidentBuffer;
use turbospark_coreml::{ComputeUnits, CoreMlModel, LoadOptions, Tensor};

pub const CONFIG: &str = "config.json";
pub const TOKENIZER: &str = "tokenizer.json";
pub const EMBEDDINGS: &str = "embeddings.f16";
pub const PAGE_SIZE: usize = 1024;
/// Fill for the padded border, as upstream pads (`ImageOps.pad`, color 127).
const PAD_GRAY: u8 = 127;
const MASKED: f32 = -1e4;

/// `config.json` of the Core ML bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrConfig {
    pub hidden: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub vocab_half: usize,
    pub rope_theta: f64,
    pub ring: usize,
    pub chunk: usize,
    pub capacities: Vec<usize>,
    pub page_tokens: usize,
    pub image_size: usize,
    pub bos: u32,
    pub eos: u32,
    pub image_token: u32,
    pub prompt_suffix_ids: BTreeMap<String, Vec<u32>>,
    pub ngram_size: usize,
    pub ngram_window: usize,
    pub max_length: usize,
    pub decoder: String,
    pub encoder: String,
}

fn get_usize(v: &serde_json::Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| format!("config.json {key} is missing or not an unsigned integer"))
}

fn get_str(v: &serde_json::Value, key: &str) -> Result<String, String> {
    v.get(key)
        .and_then(|x| x.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("config.json {key} is missing"))
}

impl OcrConfig {
    pub fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let mut capacities: Vec<usize> = v
            .get("capacities")
            .and_then(|c| c.as_array())
            .ok_or("config.json capacities is missing")?
            .iter()
            .map(|x| x.as_u64().map(|x| x as usize).ok_or("capacity is not an integer"))
            .collect::<Result<_, _>>()?;
        capacities.sort_unstable();
        let prompt_suffix_ids = v
            .get("prompt_suffix_ids")
            .and_then(|p| p.as_object())
            .ok_or("config.json prompt_suffix_ids is missing")?
            .iter()
            .map(|(k, ids)| {
                let ids = ids
                    .as_array()
                    .ok_or("prompt ids must be an array")?
                    .iter()
                    .map(|x| x.as_u64().map(|x| x as u32).ok_or("prompt id is not an integer"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((k.clone(), ids))
            })
            .collect::<Result<BTreeMap<_, _>, &str>>()?;
        let c = Self {
            hidden: get_usize(v, "hidden_size")?,
            head_dim: get_usize(v, "head_dim")?,
            vocab_size: get_usize(v, "vocab_size")?,
            vocab_half: get_usize(v, "vocab_half")?,
            rope_theta: v
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .ok_or("config.json rope_theta is missing")?,
            ring: get_usize(v, "ring")?,
            chunk: get_usize(v, "chunk")?,
            capacities,
            page_tokens: get_usize(v, "page_tokens")?,
            image_size: get_usize(v, "image_size")?,
            bos: get_usize(v, "bos_id")? as u32,
            eos: get_usize(v, "eos_id")? as u32,
            image_token: get_usize(v, "image_token_id")? as u32,
            prompt_suffix_ids,
            ngram_size: get_usize(v, "ngram_size")?,
            ngram_window: get_usize(v, "ngram_window")?,
            max_length: get_usize(v, "max_length")?,
            decoder: get_str(v, "decoder")?,
            encoder: get_str(v, "encoder")?,
        };
        if c.vocab_half * 2 != c.vocab_size {
            return Err(format!(
                "vocab_half {} is not half of vocab_size {}",
                c.vocab_half, c.vocab_size
            ));
        }
        if c.image_size != PAGE_SIZE {
            return Err(format!("image_size {} is not {PAGE_SIZE}", c.image_size));
        }
        if c.head_dim % 2 != 0 || c.chunk == 0 || c.ring == 0 || c.capacities.is_empty() {
            return Err("config.json head_dim, chunk, ring or capacities is invalid".into());
        }
        if c.ngram_size < 2 {
            return Err("config.json ngram_size must be at least 2".into());
        }
        Ok(c)
    }

    /// `<bos>` + page tokens + the prompt's suffix ids.
    pub fn prefix_length(&self, pages: usize, prompt: &str) -> Option<usize> {
        Some(1 + pages * self.page_tokens + self.prompt_suffix_ids.get(prompt)?.len())
    }

    /// Smallest bucket holding `tokens` prefix rows rounded up to whole
    /// prefill chunks.
    pub fn capacity_for(&self, tokens: usize) -> Option<usize> {
        let padded = tokens.div_ceil(self.chunk) * self.chunk;
        self.capacities.iter().copied().find(|&c| c >= padded)
    }

    /// Most pages one read takes with the `multi` prompt.
    pub fn max_pages(&self) -> usize {
        let mut pages = 0;
        while self
            .prefix_length(pages + 1, "multi")
            .and_then(|n| self.capacity_for(n))
            .is_some()
        {
            pages += 1;
        }
        pages
    }
}

/// `(cos, sin)` tables `[positions, head_dim]`, half-split layout: the
/// angle for frequency i fills columns i and i + head_dim / 2.
pub(crate) fn rope_tables(positions: std::ops::Range<usize>, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let n = positions.len();
    let mut cos = vec![0.0f32; n * head_dim];
    let mut sin = vec![0.0f32; n * head_dim];
    for (row, pos) in positions.enumerate() {
        for i in 0..half {
            let angle = pos as f64 / theta.powf((2 * i) as f64 / head_dim as f64);
            let (s, c) = angle.sin_cos();
            for col in [i, half + i] {
                cos[row * head_dim + col] = c as f32;
                sin[row * head_dim + col] = s as f32;
            }
        }
    }
    (cos, sin)
}

/// Prefill mask `[chunk, slots + chunk]` for the chunk starting at
/// `start`: every earlier prefix slot, plus causal attention inside the
/// chunk (whose keys are the last `chunk` columns).
pub(crate) fn prefill_mask(start: usize, chunk: usize, slots: usize) -> Vec<f32> {
    let width = slots + chunk;
    let mut m = vec![MASKED; chunk * width];
    for i in 0..chunk {
        let row = &mut m[i * width..(i + 1) * width];
        row[..start].fill(0.0);
        row[slots..slots + i + 1].fill(0.0);
    }
    m
}

/// Decode-step bookkeeping: the ring slot generated token `t` writes, and
/// the mask `[slots + 1]` it reads with. The mask shows the prefix, the
/// previous `ring - 1` generated tokens, and the token itself (last
/// column); the slot about to be overwritten is hidden.
#[derive(Debug, Clone)]
pub(crate) struct DecodeMask {
    pub mask: Vec<f32>,
    capacity: usize,
    ring: usize,
}

impl DecodeMask {
    pub fn new(prefix: usize, capacity: usize, ring: usize) -> Self {
        let slots = capacity + ring;
        let mut mask = vec![MASKED; slots + 1];
        mask[..prefix].fill(0.0);
        mask[slots] = 0.0;
        Self { mask, capacity, ring }
    }

    /// Updates the mask for step `t` and returns the slot to write.
    pub fn step(&mut self, t: usize) -> usize {
        let slot = self.capacity + t % self.ring;
        if t > 0 {
            self.mask[self.capacity + (t - 1) % self.ring] = 0.0;
        }
        self.mask[slot] = MASKED;
        slot
    }
}

/// Merges the per-half top-k into one candidate list, best first. Ids in
/// the second half are offset by `vocab_half`.
pub(crate) fn merge_top_k(values: &[f32], ids: &[i32], k: usize, vocab_half: usize) -> Result<Vec<(f32, u32)>, String> {
    if values.len() != 2 * k || ids.len() != 2 * k {
        return Err(format!("top-k outputs are not [2, {k}]"));
    }
    let mut merged = Vec::with_capacity(2 * k);
    for half in 0..2 {
        for j in 0..k {
            let id = usize::try_from(ids[half * k + j]).map_err(|_| "negative top-k id")?;
            if id >= vocab_half {
                return Err(format!("top-k id {id} is outside its half ({vocab_half})"));
            }
            merged.push((values[half * k + j], (id + half * vocab_half) as u32));
        }
    }
    // Stable sort: equal logits keep half-major order, as the reference.
    merged.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(merged)
}

/// Upstream `SlidingWindowNoRepeatNgramProcessor`: the tokens that would
/// repeat an `n`-gram seen in the last `window` tokens (prompt included).
pub(crate) fn banned_tokens(seq: &[u32], n: usize, window: usize) -> HashSet<u32> {
    let mut banned = HashSet::new();
    if seq.len() < n {
        return banned;
    }
    let start = seq.len().saturating_sub(window);
    let end = seq.len() - n + 1;
    if end <= start {
        return banned;
    }
    let tail = &seq[seq.len() - (n - 1)..];
    for i in start..end {
        if seq[i..i + n - 1] == *tail {
            banned.insert(seq[i + n - 1]);
        }
    }
    banned
}

/// GPT-2's reversible byte to printable code point table.
fn bytes_to_unicode() -> [u32; 256] {
    let mut bytes: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut codes = bytes.clone();
    let mut extra = 0;
    for b in 0..256u32 {
        if !bytes.contains(&b) {
            bytes.push(b);
            codes.push(256 + extra);
            extra += 1;
        }
    }
    let mut table = [0u32; 256];
    for (b, c) in bytes.into_iter().zip(codes) {
        table[b as usize] = c;
    }
    table
}

/// Byte-level BPE detokenizer that keeps added tokens (the `<|det|>` and
/// `<PAGE>` layout markup) and drops only the skipped ids.
pub(crate) struct TokenDecoder {
    pieces: HashMap<u32, Vec<u8>>,
    skip: HashSet<u32>,
}

impl TokenDecoder {
    pub fn from_tokenizer_json(v: &serde_json::Value, skip: &[u32]) -> Result<Self, String> {
        let vocab = v
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|m| m.as_object())
            .ok_or("tokenizer.json is not a byte-level BPE tokenizer")?;
        let mut byte_for: HashMap<u32, u8> = HashMap::new();
        for (b, c) in bytes_to_unicode().iter().enumerate() {
            byte_for.insert(*c, b as u8);
        }
        let mut pieces = HashMap::with_capacity(vocab.len());
        for (token, id) in vocab {
            let id = id.as_u64().ok_or("vocab id is not an integer")? as u32;
            let bytes: Vec<u8> = token
                .chars()
                .filter_map(|ch| byte_for.get(&(ch as u32)).copied())
                .collect();
            pieces.insert(id, bytes);
        }
        if let Some(added) = v.get("added_tokens").and_then(|a| a.as_array()) {
            for entry in added {
                if let (Some(content), Some(id)) = (
                    entry.get("content").and_then(|c| c.as_str()),
                    entry.get("id").and_then(|i| i.as_u64()),
                ) {
                    pieces.insert(id as u32, content.as_bytes().to_vec());
                }
            }
        }
        Ok(Self {
            pieces,
            skip: skip.iter().copied().collect(),
        })
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if !self.skip.contains(id) {
                if let Some(p) = self.pieces.get(id) {
                    bytes.extend_from_slice(p);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// A page ready for the encoder: planar RGB in `[-1, 1]`, `[3, 1024, 1024]`.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrPage {
    pixels: Vec<f32>,
}

/// Where an image lands in the padded square: PIL `ImageOps.pad` with
/// centering 0.5. Returns `(width, height, x, y)`.
pub(crate) fn pad_geometry(width: usize, height: usize) -> (usize, usize, usize, usize) {
    let round = |x: f64| turbospark_vision_io::round_half_to_even(x) as usize;
    let size = PAGE_SIZE;
    let (im, dest) = (width as f64 / height as f64, 1.0);
    if im == dest {
        (size, size, 0, 0)
    } else if im > dest {
        let h = round(height as f64 / width as f64 * size as f64).max(1);
        (size, h, 0, round((size - h) as f64 * 0.5))
    } else {
        let w = round(width as f64 / height as f64 * size as f64).max(1);
        (w, size, round((size - w) as f64 * 0.5), 0)
    }
}

impl OcrPage {
    /// Aspect-fits interleaved RGB8 into the padded 1024 x 1024 square
    /// (PIL bicubic resize, mid-gray border) and normalizes to `[-1, 1]`.
    pub fn from_rgb(rgb: &[u8], width: usize, height: usize) -> Result<Self, String> {
        if width == 0 || height == 0 || rgb.len() != width * height * 3 {
            return Err(format!("{width}x{height} RGB8 needs {} bytes", width * height * 3));
        }
        let (w, h, x0, y0) = pad_geometry(width, height);
        let resized = turbospark_vision_io::resize::resize_bicubic_pil(rgb, width, height, w, h);
        let n = PAGE_SIZE;
        let plane = n * n;
        let mut pixels = vec![f32::from(PAD_GRAY) / 127.5 - 1.0; 3 * plane];
        for y in 0..h {
            for x in 0..w {
                let src = (y * w + x) * 3;
                let dst = (y0 + y) * n + x0 + x;
                for c in 0..3 {
                    pixels[c * plane + dst] = f32::from(resized[src + c]) / 127.5 - 1.0;
                }
            }
        }
        Ok(Self { pixels })
    }

    pub fn from_file(path: &Path) -> Result<Self, String> {
        let img = turbospark_vision_io::decode_image_file(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_rgb(&img.data, img.width, img.height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OcrOptions {
    /// Placement of the decoder. The page encoder always runs on the GPU.
    pub decoder_units: ComputeUnits,
    pub max_new_tokens: Option<usize>,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            decoder_units: ComputeUnits::CpuAndGpu,
            max_new_tokens: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OcrStats {
    pub pages: usize,
    pub prompt_tokens: usize,
    pub capacity: usize,
    pub output_tokens: usize,
    pub encode_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub stopped_at_eos: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OcrResult {
    /// Raw model output, layout markup (`<PAGE>`, `<|det|>`) included.
    pub text: String,
    pub stats: OcrStats,
}

impl OcrResult {
    /// The text without layout markup; pages are separated by a blank
    /// line.
    pub fn plain_text(&self) -> String {
        plain_text(&self.text)
    }
}

/// Strips the model's layout markup: `<|det|>...<|/det|>` boxes are
/// dropped, `<PAGE>` starts a new page, and any other `<|...|>` control
/// token is removed. Lines are trimmed and empty ones dropped.
pub fn plain_text(raw: &str) -> String {
    let mut out = String::new();
    let mut rest = raw;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix("<|det|>") {
            rest = after.find("<|/det|>").map_or("", |end| &after[end + "<|/det|>".len()..]);
        } else if let Some(after) = rest.strip_prefix("<PAGE>") {
            out.push_str("\n\u{c}\n");
            rest = after;
        } else if rest.starts_with("<|") {
            rest = rest.find("|>").map_or("", |end| &rest[end + 2..]);
        } else {
            let ch = rest.chars().next().expect("non-empty");
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    let pages: Vec<String> = out
        .split('\u{c}')
        .map(|page| {
            page.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|p| !p.is_empty())
        .collect();
    pages.join("\n\n")
}

/// Progress of one read.
#[derive(Debug, Clone, PartialEq)]
pub enum OcrEvent {
    PageEncoded { index: usize },
    Prefilled { tokens: usize, capacity: usize },
    Token { emitted: usize, budget: usize },
}

/// Table `[rows, cols]` of little-endian f16, memory-mapped.
struct F16Table {
    buf: ResidentBuffer,
    rows: usize,
    cols: usize,
}

impl F16Table {
    fn open(path: &Path, rows: usize, cols: usize) -> Result<Self, String> {
        let len = std::fs::metadata(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        if len != (rows * cols * 2) as u64 {
            return Err(format!(
                "{} is {len} bytes, expected [{rows}, {cols}] float16",
                path.display()
            ));
        }
        let buf = ResidentBuffer::map(path, 0, len).map_err(|e| e.to_string())?;
        Ok(Self { buf, rows, cols })
    }

    fn push_row(&self, id: u32, out: &mut Vec<f32>) -> Result<(), String> {
        let id = id as usize;
        if id >= self.rows {
            return Err(format!("token {id} is outside the {}-row table", self.rows));
        }
        let b = &self.buf.data()[id * self.cols * 2..(id + 1) * self.cols * 2];
        out.extend(
            b.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()),
        );
        Ok(())
    }
}

struct Bucket {
    prefill: CoreMlModel,
    decode: CoreMlModel,
}

pub struct UnlimitedOcrCoreMl {
    dir: PathBuf,
    config: OcrConfig,
    opts: OcrOptions,
    encoder: CoreMlModel,
    decoder_package: PathBuf,
    embeddings: F16Table,
    detok: TokenDecoder,
    buckets: Mutex<HashMap<usize, std::sync::Arc<Bucket>>>,
}

impl std::fmt::Debug for UnlimitedOcrCoreMl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnlimitedOcrCoreMl")
            .field("dir", &self.dir)
            .field("config", &self.config)
            .field("opts", &self.opts)
            .finish_non_exhaustive()
    }
}

fn err(e: turbospark_coreml::CoreMlError) -> String {
    e.to_string()
}

impl UnlimitedOcrCoreMl {
    pub fn open(dir: &Path, opts: &OcrOptions) -> Result<Self, String> {
        if !turbospark_coreml::supports(turbospark_coreml::Capability::State) {
            return Err(
                "Unlimited OCR on Core ML requires macOS 15 or newer (stateful multifunction decoder)"
                    .into(),
            );
        }
        if opts.decoder_units == ComputeUnits::CpuAndNeuralEngine {
            return Err("the Unlimited OCR decoder is not qualified on the Neural Engine".into());
        }
        let config_value: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join(CONFIG)).map_err(|e| format!("read {CONFIG}: {e}"))?,
        )
        .map_err(|e| format!("{CONFIG}: {e}"))?;
        let config = OcrConfig::parse(&config_value)?;
        let embeddings = F16Table::open(&dir.join(EMBEDDINGS), config.vocab_size, config.hidden)?;
        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join(TOKENIZER)).map_err(|e| format!("read {TOKENIZER}: {e}"))?,
        )
        .map_err(|e| format!("{TOKENIZER}: {e}"))?;
        let detok = TokenDecoder::from_tokenizer_json(&tok, &[config.bos, config.eos])?;
        let cache = super::cache_root(dir);
        // The card measured wrong page embeddings on the Neural Engine
        // (row cosine down to 0.66), so the encoder is pinned to the GPU.
        let encoder = CoreMlModel::open(
            &dir.join(&config.encoder),
            &cache,
            &LoadOptions::new(ComputeUnits::CpuAndGpu),
        )
        .map_err(err)?;
        let want = [1, 3, PAGE_SIZE, PAGE_SIZE];
        if !encoder.inputs().iter().any(|s| {
            s.name == "pixels"
                && matches!(&s.kind, turbospark_coreml::FeatureKind::MultiArray { shape: Some(sh), .. } if sh.as_slice() == want)
        }) {
            return Err(format!("page encoder input pixels is not {want:?}"));
        }
        let decoder_package =
            turbospark_coreml::compile_if_needed(&dir.join(&config.decoder), &cache).map_err(err)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            opts: *opts,
            encoder,
            decoder_package,
            embeddings,
            detok,
            buckets: Mutex::new(HashMap::new()),
        })
    }

    pub fn config(&self) -> &OcrConfig {
        &self.config
    }

    pub fn max_pages(&self) -> usize {
        self.config.max_pages()
    }

    pub fn prompts(&self) -> impl Iterator<Item = &str> {
        self.config.prompt_suffix_ids.keys().map(String::as_str)
    }

    fn bucket(&self, capacity: usize) -> Result<std::sync::Arc<Bucket>, String> {
        let mut map = self.buckets.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(b) = map.get(&capacity) {
            return Ok(b.clone());
        }
        let load = |f: String| {
            CoreMlModel::load(
                &self.decoder_package,
                &LoadOptions::new(self.opts.decoder_units).with_function(f),
            )
            .map_err(err)
        };
        let b = std::sync::Arc::new(Bucket {
            prefill: load(format!("prefill_P{capacity}"))?,
            decode: load(format!("decode_P{capacity}"))?,
        });
        map.insert(capacity, b.clone());
        Ok(b)
    }

    fn encode_page(&self, page: &OcrPage) -> Result<Vec<f32>, String> {
        let pixels =
            Tensor::f32(vec![1, 3, PAGE_SIZE, PAGE_SIZE], page.pixels.clone()).map_err(err)?;
        let out = self.encoder.predict(&[("pixels", &pixels)]).map_err(err)?;
        let t = out.get("page_embeds").ok_or("page encoder returned no page_embeds")?;
        if t.shape() != [self.config.page_tokens, self.config.hidden] {
            return Err(format!("page_embeds shape {:?}", t.shape()));
        }
        t.to_f32_vec().ok_or_else(|| "page_embeds is not float".into())
    }

    /// One decoder call. Returns the merged candidates, best first.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        model: &CoreMlModel,
        state: &mut turbospark_coreml::CoreMlState,
        x: &[f32],
        rows: usize,
        positions: std::ops::Range<usize>,
        mask: &[f32],
        write_idx: &[i32],
        out_index: i32,
    ) -> Result<Vec<(f32, u32)>, String> {
        let c = &self.config;
        let width = mask.len() / rows;
        let (cos, sin) = rope_tables(positions, c.head_dim, c.rope_theta);
        let x = Tensor::f16_from_f32("x", vec![1, rows, c.hidden], x).map_err(err)?;
        let cos = Tensor::f16_from_f32("cos", vec![rows, c.head_dim], &cos).map_err(err)?;
        let sin = Tensor::f16_from_f32("sin", vec![rows, c.head_dim], &sin).map_err(err)?;
        let mask = Tensor::f16_from_f32("mask", vec![1, 1, rows, width], mask).map_err(err)?;
        let write_idx = Tensor::i32(vec![rows], write_idx.to_vec()).map_err(err)?;
        let out_index = Tensor::i32(vec![1], vec![out_index]).map_err(err)?;
        let out = model
            .predict_with_state(
                state,
                &[
                    ("x", &x),
                    ("cos", &cos),
                    ("sin", &sin),
                    ("mask", &mask),
                    ("write_idx", &write_idx),
                    ("out_index", &out_index),
                ],
            )
            .map_err(err)?;
        let values = out
            .get("top_values")
            .and_then(Tensor::to_f32_vec)
            .ok_or("decoder returned no top_values")?;
        let ids = out
            .get("top_ids")
            .and_then(|t| t.as_i32().map(<[i32]>::to_vec))
            .ok_or("decoder returned no top_ids")?;
        merge_top_k(&values, &ids, values.len() / 2, c.vocab_half)
    }

    /// Reads `pages` as one document with `prompt` (`multi`, or `single`
    /// for one page). `progress` returning `false` cancels.
    pub fn read(
        &self,
        pages: &[OcrPage],
        prompt: &str,
        mut progress: impl FnMut(OcrEvent) -> bool,
    ) -> Result<OcrResult, String> {
        let c = &self.config;
        if pages.is_empty() {
            return Err("no pages".into());
        }
        let suffix = c.prompt_suffix_ids.get(prompt).ok_or_else(|| {
            format!(
                "unknown prompt {prompt:?}; known: {}",
                self.prompts().collect::<Vec<_>>().join(", ")
            )
        })?;
        let n = c.prefix_length(pages.len(), prompt).expect("prompt checked");
        let capacity = c.capacity_for(n).ok_or_else(|| {
            format!(
                "{} pages need {n} prefix slots; this bundle holds {} pages",
                pages.len(),
                self.max_pages()
            )
        })?;
        let bucket = self.bucket(capacity)?;
        let cancelled = || Err::<OcrResult, _>("OCR cancelled".to_string());

        // Prefix rows: <bos>, page embeddings, the prompt's ids.
        let started = std::time::Instant::now();
        let mut prefix = Vec::with_capacity(n * c.hidden);
        let mut sequence = vec![c.bos];
        self.embeddings.push_row(c.bos, &mut prefix)?;
        for (index, page) in pages.iter().enumerate() {
            prefix.extend(self.encode_page(page)?);
            sequence.extend(std::iter::repeat_n(c.image_token, c.page_tokens));
            if !progress(OcrEvent::PageEncoded { index }) {
                return cancelled();
            }
        }
        for &id in suffix {
            self.embeddings.push_row(id, &mut prefix)?;
        }
        sequence.extend_from_slice(suffix);
        debug_assert_eq!(prefix.len(), n * c.hidden);
        let encode_ms = started.elapsed().as_secs_f64() * 1e3;

        // Prefill in chunks, then hand the cache to the decode state.
        let started = std::time::Instant::now();
        let slots = capacity + c.ring;
        let mut prefill_state = bucket.prefill.new_state().map_err(err)?;
        let mut top = Vec::new();
        for start in (0..n).step_by(c.chunk) {
            let rows = c.chunk.min(n - start);
            let mut x = vec![0.0f32; c.chunk * c.hidden];
            x[..rows * c.hidden].copy_from_slice(&prefix[start * c.hidden..(start + rows) * c.hidden]);
            let write: Vec<i32> = (start..start + c.chunk).map(|i| i as i32).collect();
            top = self.step(
                &bucket.prefill,
                &mut prefill_state,
                &x,
                c.chunk,
                start..start + c.chunk,
                &prefill_mask(start, c.chunk, slots),
                &write,
                (rows - 1) as i32,
            )?;
        }
        let mut state = bucket.decode.new_state().map_err(err)?;
        state.copy_from(&prefill_state).map_err(err)?;
        drop(prefill_state);
        let prefill_ms = started.elapsed().as_secs_f64() * 1e3;
        if !progress(OcrEvent::Prefilled {
            tokens: n,
            capacity,
        }) {
            return cancelled();
        }

        // Decode greedily over the candidates the n-gram ban allows.
        let started = std::time::Instant::now();
        let mut budget = c.max_length.saturating_sub(n);
        if let Some(m) = self.opts.max_new_tokens {
            budget = budget.min(m);
        }
        let mut mask = DecodeMask::new(n, capacity, c.ring);
        let mut generated = Vec::new();
        let mut stopped_at_eos = false;
        for t in 0..budget {
            let banned = banned_tokens(&sequence, c.ngram_size, c.ngram_window);
            let token = top
                .iter()
                .find(|(_, id)| !banned.contains(id))
                .map(|(_, id)| *id)
                .ok_or_else(|| format!("every top-k candidate is banned at step {t}"))?;
            sequence.push(token);
            generated.push(token);
            if !progress(OcrEvent::Token {
                emitted: generated.len(),
                budget,
            }) {
                return cancelled();
            }
            if token == c.eos {
                stopped_at_eos = true;
                break;
            }
            if t + 1 == budget {
                break;
            }
            let slot = mask.step(t);
            let mut x = Vec::with_capacity(c.hidden);
            self.embeddings.push_row(token, &mut x)?;
            top = self.step(
                &bucket.decode,
                &mut state,
                &x,
                1,
                n + t..n + t + 1,
                &mask.mask,
                &[slot as i32],
                0,
            )?;
        }
        let decode_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok(OcrResult {
            text: self.detok.decode(&generated),
            stats: OcrStats {
                pages: pages.len(),
                prompt_tokens: n,
                capacity,
                output_tokens: generated.len(),
                encode_ms,
                prefill_ms,
                decode_ms,
                stopped_at_eos,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> OcrConfig {
        OcrConfig::parse(&serde_json::json!({
            "hidden_size": 1280, "head_dim": 128, "layers": 12, "vocab_size": 129280,
            "vocab_half": 64640, "rope_theta": 10000.0, "ring": 128, "chunk": 128,
            "capacities": [640, 384, 2304, 1152], "page_tokens": 273, "image_size": 1024,
            "bos_id": 0, "eos_id": 1, "image_token_id": 128815,
            "prompt_suffix_ids": {"multi": [1, 2, 3, 4, 5], "single": [6, 7]},
            "ngram_size": 35, "ngram_window": 1024, "max_length": 8192,
            "decoder": "UnlimitedOcrDecoder.mlpackage", "encoder": "UnlimitedOcrPageEncoder.mlpackage"
        }))
        .unwrap()
    }

    #[test]
    fn buckets_and_page_limits_follow_the_reference() {
        let c = config();
        assert_eq!(c.capacities, vec![384, 640, 1152, 2304]);
        // One page: 1 + 273 + 5 = 279 -> 384 after rounding to 128.
        assert_eq!(c.prefix_length(1, "multi"), Some(279));
        assert_eq!(c.capacity_for(279), Some(384));
        // 384 exactly fits; one more row needs the next bucket.
        assert_eq!(c.capacity_for(384), Some(384));
        assert_eq!(c.capacity_for(385), Some(640));
        // 8 pages: 1 + 2184 + 5 = 2190 -> 2304; 9 pages: 2463 -> none.
        assert_eq!(c.capacity_for(c.prefix_length(8, "multi").unwrap()), Some(2304));
        assert_eq!(c.capacity_for(c.prefix_length(9, "multi").unwrap()), None);
        assert_eq!(c.max_pages(), 8);
        assert_eq!(c.prefix_length(1, "nope"), None);
    }

    #[test]
    fn config_refuses_inconsistent_values() {
        let mut v = serde_json::json!({
            "hidden_size": 8, "head_dim": 4, "vocab_size": 10, "vocab_half": 4,
            "rope_theta": 1e4, "ring": 4, "chunk": 4, "capacities": [8], "page_tokens": 2,
            "image_size": 1024, "bos_id": 0, "eos_id": 1, "image_token_id": 2,
            "prompt_suffix_ids": {}, "ngram_size": 3, "ngram_window": 8, "max_length": 32,
            "decoder": "d", "encoder": "e"
        });
        assert!(OcrConfig::parse(&v).is_err(), "vocab_half must be half");
        v["vocab_half"] = serde_json::json!(5);
        assert!(OcrConfig::parse(&v).is_ok());
        v["image_size"] = serde_json::json!(640);
        assert!(OcrConfig::parse(&v).is_err());
    }

    #[test]
    fn prefill_mask_shows_earlier_chunks_and_is_causal_within() {
        // chunk 4, 8 cache slots, second chunk (start 4).
        let m = prefill_mask(4, 4, 8);
        let width = 12;
        let row = |i: usize| &m[i * width..(i + 1) * width];
        // Row 0: slots 0..4 (earlier chunk) and its own key (column 8).
        assert_eq!(&row(0)[..4], &[0.0; 4]);
        assert_eq!(&row(0)[4..8], &[MASKED; 4]);
        assert_eq!(row(0)[8], 0.0);
        assert_eq!(&row(0)[9..], &[MASKED; 3]);
        // Row 3 sees all four keys of its chunk.
        assert_eq!(&row(3)[8..], &[0.0; 4]);
        // The first chunk sees no cache slots at all.
        assert_eq!(&prefill_mask(0, 4, 8)[..8], &[MASKED; 8]);
    }

    #[test]
    fn decode_mask_walks_the_ring() {
        // prefix 3, capacity 4, ring 2: slots 0..6, own key at column 6.
        let mut d = DecodeMask::new(3, 4, 2);
        let visible = |d: &DecodeMask| -> Vec<usize> {
            (0..d.mask.len()).filter(|&i| d.mask[i] == 0.0).collect()
        };
        assert_eq!(visible(&d), vec![0, 1, 2, 6]);
        assert_eq!(d.step(0), 4);
        assert_eq!(visible(&d), vec![0, 1, 2, 6]);
        // Step 1 writes slot 5 and can now see the token in slot 4.
        assert_eq!(d.step(1), 5);
        assert_eq!(visible(&d), vec![0, 1, 2, 4, 6]);
        // Step 2 wraps to slot 4, evicting it; slot 5 becomes visible.
        assert_eq!(d.step(2), 4);
        assert_eq!(visible(&d), vec![0, 1, 2, 5, 6]);
    }

    #[test]
    fn rope_uses_the_half_split_layout() {
        let (cos, sin) = rope_tables(0..2, 4, 10000.0);
        // Position 0: cos 1, sin 0 everywhere.
        assert_eq!(&cos[..4], &[1.0; 4]);
        assert_eq!(&sin[..4], &[0.0; 4]);
        // Position 1, freq 0: angle 1; freq 1: angle 1 / 100.
        assert!((cos[4] - 1f32.cos()).abs() < 1e-6 && cos[6] == cos[4]);
        assert!((sin[5] - 0.01f32.sin()).abs() < 1e-6 && sin[7] == sin[5]);
    }

    #[test]
    fn top_k_halves_merge_with_offset_and_order() {
        let merged = merge_top_k(&[5.0, 1.0, 4.0, 6.0], &[10, 11, 3, 7], 2, 100).unwrap();
        assert_eq!(merged, vec![(6.0, 107), (5.0, 10), (4.0, 103), (1.0, 11)]);
        assert!(merge_top_k(&[1.0, 2.0], &[0, 200], 1, 100).is_err());
        assert!(merge_top_k(&[1.0], &[0], 1, 100).is_err());
    }

    #[test]
    fn ngram_ban_matches_the_reference_definition() {
        // n = 3: the last two tokens are [1, 2]; [1, 2] was followed by 3
        // and by 4 earlier, so both are banned.
        let seq = [1, 2, 3, 9, 1, 2, 4, 9, 1, 2];
        let banned = banned_tokens(&seq, 3, 1024);
        assert_eq!(banned, HashSet::from([3, 4]));
        // A window of 6 only covers the last occurrence's continuation.
        let banned = banned_tokens(&seq, 3, 6);
        assert_eq!(banned, HashSet::from([4]));
        assert!(banned_tokens(&[1, 2], 3, 1024).is_empty());
    }

    #[test]
    fn pad_geometry_follows_imageops_pad() {
        assert_eq!(pad_geometry(500, 500), (1024, 1024, 0, 0));
        // Landscape 2:1 -> 1024 x 512, centred vertically at 256.
        assert_eq!(pad_geometry(800, 400), (1024, 512, 0, 256));
        // Portrait 3:4 -> 768 x 1024 at x 128.
        assert_eq!(pad_geometry(600, 800), (768, 1024, 128, 0));
        // US letter at 72 dpi: 612 x 792 -> round(791.27) = 791, x = round(116.5) = 116 (half to even).
        assert_eq!(pad_geometry(612, 792), (791, 1024, 116, 0));
    }

    #[test]
    fn page_border_is_mid_gray_and_content_normalized() {
        let rgb = vec![255u8; 4 * 2 * 3];
        let p = OcrPage::from_rgb(&rgb, 4, 2).unwrap();
        let gray = 127.0 / 127.5 - 1.0;
        assert_eq!(p.pixels.len(), 3 * PAGE_SIZE * PAGE_SIZE);
        assert_eq!(p.pixels[0], gray, "top-left is border");
        let centre = 512 * PAGE_SIZE + 512;
        assert!((p.pixels[centre] - 1.0).abs() < 1e-6, "white content maps to 1");
        assert!(OcrPage::from_rgb(&rgb, 5, 2).is_err());
    }

    #[test]
    fn plain_text_strips_markup_and_splits_pages() {
        let raw = "<PAGE><|det|>title [1, 2, 3, 4]<|/det|>Title\n<|det|>text [5]<|/det|>Body line.\n\
                   <PAGE><|det|>text [6]<|/det|>Second <|ref|>page|<|/ref|>.";
        assert_eq!(plain_text(raw), "Title\nBody line.\n\nSecond page|.");
        assert_eq!(plain_text("no markup"), "no markup");
        assert_eq!(plain_text("<|det|>unterminated"), "");
    }

    #[test]
    fn byte_level_decoder_keeps_markup_and_skips_bos_eos() {
        // "Ġ" is the byte-level form of a space.
        let tok = serde_json::json!({
            "model": {"vocab": {"Hello": 5, "\u{0120}world": 6}},
            "added_tokens": [{"id": 0, "content": "<bos>"}, {"id": 1, "content": "<eos>"},
                             {"id": 7, "content": "<|det|>"}]
        });
        let d = TokenDecoder::from_tokenizer_json(&tok, &[0, 1]).unwrap();
        assert_eq!(d.decode(&[0, 7, 5, 6, 1]), "<|det|>Hello world");
    }
}
