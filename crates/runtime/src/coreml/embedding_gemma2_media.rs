//! EmbeddingGemma 2 audio and image towers. Each turns its input into soft
//! tokens in the text model's 512-d input space; the text functions then
//! embed `<bos> <|audio|image> tokens <audio|image|> <eos>` so audio,
//! images and text share one embedding space.
//!
//! - Audio: 10 s windows of 16 kHz mono with a 160-sample left pad. The
//!   log-mel is inside the graph. 250 tokens per window; token k is valid
//!   when mel frame 4k is.
//! - Image: HF `Gemma4ImageProcessor` resize (aspect kept, sides a multiple
//!   of 48 px, at most 9 x budget patches, PIL bicubic), RGB / 255, 16 px
//!   patches row by row, then 3 x 3 average pooling into soft tokens.
//!
//! References: the model card, FluidUse `EmbeddingGemma2Audio`
//! (`d278e17df85e39e8c21bb64a467da81c39576fcd`), and transformers
//! `image_processing_pil_gemma4.py` / `Gemma4VisionPooler` at
//! `28034545c35acac05793695bd5c878f66e6b544e`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use model_io::ResidentBuffer;
use turbospark_coreml::{CoreMlModel, LoadOptions, Tensor};

use super::embedding_gemma2::{EmbeddingGemma2CoreMl, EmbeddingGemma2Options, Row};

pub const AUDIO_PACKAGE: &str = "EmbeddingGemma2Audio.mlpackage";
pub const VISION_PACKAGE: &str = "EmbeddingGemma2Vision.mlpackage";
pub const POSITION_TABLE: &str = "position_embeddings.f16";

pub const AUDIO_SAMPLE_RATE: u32 = 16_000;
const AUDIO_WINDOW: usize = 160_000;
const AUDIO_LEFT_PAD: usize = 160;
const AUDIO_FRAMES: usize = 1_000;
/// A trailing window shorter than this is dropped (FluidUse does the
/// same): it may be too short to produce a token.
const AUDIO_MIN_TAIL: usize = AUDIO_SAMPLE_RATE as usize / 2;

const PATCH: usize = 16;
const POOL: usize = 3;
const POSITION_ROWS: usize = 1024;

/// Mel frames that hold only real audio in a window of `samples` samples:
/// frame i reads padded samples `[160 i, 160 i + 321)`.
pub(crate) fn valid_frames(samples: usize) -> usize {
    let real = AUDIO_LEFT_PAD + samples.min(AUDIO_WINDOW);
    if real < 321 {
        0
    } else {
        AUDIO_FRAMES.min((real - 321) / 160 + 1)
    }
}

/// Two stride-2 convolutions: token k is valid when frame 4k is.
pub(crate) fn audio_token_count(valid_frames: usize) -> usize {
    valid_frames.div_ceil(4)
}

/// Window start samples over a recording.
pub(crate) fn audio_windows(len: usize) -> Vec<usize> {
    let mut starts: Vec<usize> = (0..len).step_by(AUDIO_WINDOW).collect();
    if starts.len() > 1 && len - starts[starts.len() - 1] < AUDIO_MIN_TAIL {
        starts.pop();
    }
    starts
}

/// `(height, width)` the image is resized to for `max_patches` patches:
/// transformers `get_aspect_ratio_preserving_size`, line for line.
pub(crate) fn gemma4_resize_dims(
    height: usize,
    width: usize,
    max_patches: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err("image has a zero dimension".into());
    }
    let total = (height * width) as f64;
    let target_px = (max_patches * PATCH * PATCH) as f64;
    let factor = (target_px / total).sqrt();
    let side = POOL * PATCH;
    let mut h = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut w = ((factor * width as f64) / side as f64).floor() as usize * side;
    if h == 0 && w == 0 {
        return Err("image would resize to 0 x 0".into());
    }
    let max_side = (max_patches / (POOL * POOL)) * side;
    if h == 0 {
        h = side;
        w = ((width / height) * side).min(max_side);
    } else if w == 0 {
        w = side;
        h = ((height / width) * side).min(max_side);
    }
    if (h * w) as f64 > target_px {
        return Err(format!(
            "resizing {height}x{width} to {h}x{w} exceeds {max_patches} patches"
        ));
    }
    Ok((h, w))
}

/// Patch inputs for one image at a given budget.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ImagePatches {
    /// `[n_patches, 768]`, each patch `[16 rows][16 cols][RGB]` in `[0, 1]`.
    pub pixels: Vec<f32>,
    /// `(x, y)` patch coordinates.
    pub positions: Vec<(usize, usize)>,
    pub grid_w: usize,
    pub grid_h: usize,
}

impl ImagePatches {
    pub fn from_rgb(rgb: &[u8], width: usize, height: usize) -> Result<Self, String> {
        if width % (POOL * PATCH) != 0 || height % (POOL * PATCH) != 0 {
            return Err(format!("{width}x{height} is not a multiple of 48 px"));
        }
        let (gw, gh) = (width / PATCH, height / PATCH);
        let mut pixels = Vec::with_capacity(gw * gh * PATCH * PATCH * 3);
        let mut positions = Vec::with_capacity(gw * gh);
        for py in 0..gh {
            for px in 0..gw {
                for r in 0..PATCH {
                    let row = (py * PATCH + r) * width + px * PATCH;
                    for c in 0..PATCH {
                        for ch in 0..3 {
                            pixels.push(f32::from(rgb[(row + c) * 3 + ch]) / 255.0);
                        }
                    }
                }
                positions.push((px, py));
            }
        }
        Ok(Self {
            pixels,
            positions,
            grid_w: gw,
            grid_h: gh,
        })
    }

    /// Soft token each patch pools into: `Gemma4VisionPooler`'s
    /// `x / 3 + (grid_w / 3) * (y / 3)`.
    pub fn pool_index(&self, i: usize) -> usize {
        let (x, y) = self.positions[i];
        x / POOL + (self.grid_w / POOL) * (y / POOL)
    }

    pub fn soft_tokens(&self) -> usize {
        self.positions.len() / (POOL * POOL)
    }
}

/// One vision function: its soft-token budget and patch count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VisionFn {
    tokens: usize,
    patches: usize,
}

fn vision_functions(config: &serde_json::Value) -> Vec<VisionFn> {
    let mut v: Vec<VisionFn> = config
        .get("vision")
        .and_then(|x| x.get("functions"))
        .and_then(|f| f.as_object())
        .map(|o| {
            o.iter()
                .filter_map(|(name, spec)| {
                    let tokens = name.strip_prefix("vision_")?.parse().ok()?;
                    let patches = spec.get("patches")?.as_u64()? as usize;
                    (spec.get("tokens")?.as_u64()? as usize == tokens
                        && patches == tokens * POOL * POOL)
                        .then_some(VisionFn { tokens, patches })
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort_by_key(|f| f.tokens);
    v
}

/// Lazily loaded audio and vision programs. Both packages are optional:
/// a text-only install still opens, and asking for a missing tower is an
/// error at that call.
pub struct MediaTowers {
    dir: PathBuf,
    units: turbospark_coreml::ComputeUnits,
    vision_fns: Vec<VisionFn>,
    audio: Mutex<Option<Arc<CoreMlModel>>>,
    vision: Mutex<Vec<(usize, Arc<CoreMlModel>)>>,
    positions: Mutex<Option<Arc<ResidentBuffer>>>,
}

impl MediaTowers {
    pub(crate) fn new(
        dir: &Path,
        config: &serde_json::Value,
        opts: &EmbeddingGemma2Options,
    ) -> Result<Self, String> {
        Ok(Self {
            dir: dir.to_path_buf(),
            units: opts.media_units,
            vision_fns: vision_functions(config),
            audio: Mutex::new(None),
            vision: Mutex::new(Vec::new()),
            positions: Mutex::new(None),
        })
    }

    fn compiled(&self, package: &str) -> Result<PathBuf, String> {
        let p = self.dir.join(package);
        if !p.is_dir() {
            return Err(format!(
                "{package} is not installed in {}",
                self.dir.display()
            ));
        }
        turbospark_coreml::compile_if_needed(&p, &super::cache_root(&self.dir))
            .map_err(|e| e.to_string())
    }

    fn audio_model(&self) -> Result<Arc<CoreMlModel>, String> {
        let mut slot = self.audio.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(m) = slot.as_ref() {
            return Ok(m.clone());
        }
        let m = CoreMlModel::load(
            &self.compiled(AUDIO_PACKAGE)?,
            &LoadOptions::new(self.units),
        )
        .map_err(|e| e.to_string())?;
        let m = Arc::new(m);
        *slot = Some(m.clone());
        Ok(m)
    }

    fn vision_model(&self, f: VisionFn) -> Result<Arc<CoreMlModel>, String> {
        let mut map = self.vision.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, m)) = map.iter().find(|(t, _)| *t == f.tokens) {
            return Ok(m.clone());
        }
        let opts = LoadOptions::new(self.units).with_function(format!("vision_{}", f.tokens));
        let m =
            CoreMlModel::load(&self.compiled(VISION_PACKAGE)?, &opts).map_err(|e| e.to_string())?;
        let m = Arc::new(m);
        map.push((f.tokens, m.clone()));
        Ok(m)
    }

    fn position_table(&self) -> Result<Arc<ResidentBuffer>, String> {
        let mut slot = self.positions.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(t) = slot.as_ref() {
            return Ok(t.clone());
        }
        let path = self.dir.join(POSITION_TABLE);
        let want = (2 * POSITION_ROWS * 768 * 2) as u64;
        let len = std::fs::metadata(&path)
            .map_err(|e| format!("{POSITION_TABLE}: {e}"))?
            .len();
        if len != want {
            return Err(format!("{POSITION_TABLE} is {len} bytes, expected {want}"));
        }
        let t = Arc::new(ResidentBuffer::map(&path, 0, len).map_err(|e| e.to_string())?);
        *slot = Some(t.clone());
        Ok(t)
    }
}

fn f16_at(bytes: &[u8], index: usize) -> f32 {
    half::f16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]).to_f32()
}

/// One embedded audio window.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioWindowEmbedding {
    pub start_seconds: f64,
    pub duration_seconds: f64,
    pub embedding: Vec<f32>,
}

impl EmbeddingGemma2CoreMl {
    /// Embeds 16 kHz mono audio as consecutive 10 s windows.
    pub fn embed_audio(
        &self,
        samples: &[f32],
        dimension: Option<usize>,
    ) -> Result<Vec<AudioWindowEmbedding>, String> {
        let dim = self.check_dimension(dimension)?;
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            return Err("audio must be nonempty finite 16 kHz mono PCM".into());
        }
        let model = self.media.audio_model()?;
        let sr = f64::from(AUDIO_SAMPLE_RATE);
        audio_windows(samples.len())
            .into_iter()
            .map(|start| {
                let chunk = &samples[start..(start + AUDIO_WINDOW).min(samples.len())];
                let v = self.embed_audio_window(&model, chunk)?;
                Ok(AudioWindowEmbedding {
                    start_seconds: start as f64 / sr,
                    duration_seconds: chunk.len() as f64 / sr,
                    embedding: super::embedding_gemma2::truncate_normalize(&v, dim),
                })
            })
            .collect()
    }

    fn embed_audio_window(&self, model: &CoreMlModel, chunk: &[f32]) -> Result<Vec<f32>, String> {
        let mut wave = vec![0.0f32; AUDIO_LEFT_PAD + AUDIO_WINDOW];
        wave[AUDIO_LEFT_PAD..AUDIO_LEFT_PAD + chunk.len()].copy_from_slice(chunk);
        let frames = valid_frames(chunk.len());
        let tokens = audio_token_count(frames);
        if tokens == 0 {
            return Err("audio window is too short to embed".into());
        }
        let mask: Vec<f32> = (0..AUDIO_FRAMES)
            .map(|i| f32::from(u8::from(i < frames)))
            .collect();
        let wave = Tensor::f32(vec![1, wave.len()], wave).map_err(|e| e.to_string())?;
        let mask = Tensor::f16_from_f32("frame_mask", vec![1, AUDIO_FRAMES], &mask)
            .map_err(|e| e.to_string())?;
        let out = model
            .predict(&[("waveform", &wave), ("frame_mask", &mask)])
            .map_err(|e| e.to_string())?;
        let soft = out
            .get("audio_tokens")
            .and_then(Tensor::to_f32_vec)
            .ok_or("audio_tokens output is missing or not float")?;
        self.embed_soft(self.ids.audio_start, self.ids.audio_end, &soft, tokens)
    }

    /// Embeds an image (interleaved RGB8). `budget` picks the soft-token
    /// count (70, 140 or 280 for this export; default the largest).
    pub fn embed_image(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        budget: Option<usize>,
        dimension: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        let dim = self.check_dimension(dimension)?;
        if rgb.len() != width * height * 3 {
            return Err(format!(
                "{width}x{height} RGB8 needs {} bytes",
                width * height * 3
            ));
        }
        let f = match budget {
            None => *self
                .media
                .vision_fns
                .last()
                .ok_or("no vision functions declared")?,
            Some(b) => *self
                .media
                .vision_fns
                .iter()
                .find(|f| f.tokens == b)
                .ok_or_else(|| format!("image budget {b} is not one of the vision functions"))?,
        };
        let (h, w) = gemma4_resize_dims(height, width, f.patches)?;
        let resized = turbospark_vision_io::resize::resize_bicubic_pil(rgb, width, height, w, h);
        let patches = ImagePatches::from_rgb(&resized, w, h)?;
        let n = patches.positions.len();
        if n > f.patches || patches.grid_w > POSITION_ROWS || patches.grid_h > POSITION_ROWS {
            return Err(format!(
                "{n} patches exceed the {} the function takes",
                f.patches
            ));
        }
        let table = self.media.position_table()?;
        let tdata = table.data();
        let width_px = PATCH * PATCH * 3;
        let mut pix = vec![0.0f32; f.patches * width_px];
        pix[..patches.pixels.len()].copy_from_slice(&patches.pixels);
        let mut pos = vec![-1.0f32; f.patches * 2];
        let mut pe = vec![0.0f32; f.patches * 768];
        let mut valid = vec![0.0f32; f.patches];
        let mut pool = vec![0.0f32; f.tokens * f.patches];
        for (i, &(x, y)) in patches.positions.iter().enumerate() {
            pos[i * 2] = x as f32;
            pos[i * 2 + 1] = y as f32;
            for c in 0..768 {
                pe[i * 768 + c] =
                    f16_at(tdata, x * 768 + c) + f16_at(tdata, (POSITION_ROWS + y) * 768 + c);
            }
            valid[i] = 1.0;
            pool[patches.pool_index(i) * f.patches + i] = 1.0 / (POOL * POOL) as f32;
        }
        let model = self.media.vision_model(f)?;
        let t = |name: &str, shape: Vec<usize>, data: &[f32]| {
            Tensor::f16_from_f32(name, shape, data).map_err(|e| e.to_string())
        };
        let inputs = [
            ("patches", t("patches", vec![1, f.patches, width_px], &pix)?),
            ("positions", t("positions", vec![1, f.patches, 2], &pos)?),
            (
                "position_embeddings",
                t("position_embeddings", vec![1, f.patches, 768], &pe)?,
            ),
            ("valid", t("valid", vec![1, f.patches], &valid)?),
            ("pool", t("pool", vec![f.tokens, f.patches], &pool)?),
        ];
        // The package declares each input's dtype; convert the f16 tensors
        // to whatever it asks for rather than assume.
        let inputs: Vec<(&str, Tensor)> = inputs
            .into_iter()
            .map(|(name, tensor)| Ok((name, match_input_dtype(&model, name, tensor)?)))
            .collect::<Result<_, String>>()?;
        let refs: Vec<(&str, &Tensor)> = inputs.iter().map(|(n, t)| (*n, t)).collect();
        let out = model.predict(&refs).map_err(|e| e.to_string())?;
        let soft = out
            .get("image_tokens")
            .and_then(Tensor::to_f32_vec)
            .ok_or("image_tokens output is missing or not float")?;
        let v = self.embed_soft(
            self.ids.image_start,
            self.ids.image_end,
            &soft,
            patches.soft_tokens(),
        )?;
        Ok(super::embedding_gemma2::truncate_normalize(&v, dim))
    }

    /// [`Self::embed_audio`] on a WAV file, mixed to mono and resampled to
    /// 16 kHz.
    pub fn embed_audio_file(
        &self,
        path: &Path,
        dimension: Option<usize>,
    ) -> Result<Vec<AudioWindowEmbedding>, String> {
        let wave = audio::wav::read_wav_f32(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mono = audio::conversion::to_mono_resampled(
            &wave,
            AUDIO_SAMPLE_RATE,
            &audio::conversion::MonoResampleStrategy::SincHann(Default::default()),
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
        self.embed_audio(&mono, dimension)
    }

    /// [`Self::embed_image`] on a PNG or JPEG file.
    pub fn embed_image_file(
        &self,
        path: &Path,
        budget: Option<usize>,
        dimension: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        let img = turbospark_vision_io::decode_image_file(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        self.embed_image(&img.data, img.width, img.height, budget, dimension)
    }

    /// `<bos> start soft[..count] end <eos>` through the text model.
    fn embed_soft(
        &self,
        start: u32,
        end: u32,
        soft: &[f32],
        count: usize,
    ) -> Result<Vec<f32>, String> {
        let h = self.config.hidden;
        if soft.len() < count * h {
            return Err(format!("{count} soft tokens need {} values", count * h));
        }
        let mut rows = Vec::with_capacity(count + 4);
        rows.push(Row::Token(self.ids.bos));
        rows.push(Row::Token(start));
        rows.extend(soft[..count * h].chunks_exact(h).map(Row::Soft));
        rows.push(Row::Token(end));
        rows.push(Row::Token(self.ids.eos));
        self.embed_rows(&rows)
    }
}

/// Re-types an f16 input to the dtype the model declares for it.
fn match_input_dtype(model: &CoreMlModel, name: &str, t: Tensor) -> Result<Tensor, String> {
    let want = model
        .inputs()
        .iter()
        .find(|s| s.name == name)
        .and_then(|s| match &s.kind {
            turbospark_coreml::FeatureKind::MultiArray { dtype, .. } => *dtype,
            _ => None,
        });
    match want {
        Some(turbospark_coreml::DType::F32) => {
            let v = t.to_f32_vec().ok_or("float input expected")?;
            Tensor::f32(t.shape().to_vec(), v).map_err(|e| e.to_string())
        }
        _ => Ok(t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_frames_and_tokens_follow_the_reference() {
        assert_eq!(valid_frames(0), 0);
        // 160 pad + 160 samples = 320 < 321: no full frame yet.
        assert_eq!(valid_frames(160), 0);
        assert_eq!(valid_frames(161), 1);
        assert_eq!(valid_frames(AUDIO_WINDOW), 999);
        assert_eq!(valid_frames(10 * AUDIO_WINDOW), 999);
        assert_eq!(audio_token_count(999), 250);
        assert_eq!(audio_token_count(1), 1);
        assert_eq!(audio_token_count(0), 0);
    }

    #[test]
    fn audio_windows_drop_a_short_tail_only_when_not_alone() {
        assert_eq!(audio_windows(100), vec![0]);
        assert_eq!(audio_windows(AUDIO_WINDOW + 7_999), vec![0]);
        assert_eq!(audio_windows(AUDIO_WINDOW + 8_000), vec![0, AUDIO_WINDOW]);
        assert_eq!(audio_windows(2 * AUDIO_WINDOW), vec![0, AUDIO_WINDOW]);
    }

    #[test]
    fn resize_matches_transformers_examples() {
        // Worked by hand from get_aspect_ratio_preserving_size, budget 280
        // (2520 patches, 645120 px): a 640x480 photo has factor ~1.4491.
        assert_eq!(gemma4_resize_dims(480, 640, 2520).unwrap(), (672, 912));
        // A square at budget 70 (630 patches): 401.6 -> 384 per side.
        assert_eq!(gemma4_resize_dims(1000, 1000, 630).unwrap(), (384, 384));
        // A very wide strip rounds the short side to zero and clamps.
        assert_eq!(gemma4_resize_dims(10, 4000, 630).unwrap(), (48, 3360));
        for (h, w, p) in [
            (480, 640, 2520),
            (1000, 1000, 630),
            (10, 4000, 630),
            (333, 777, 1260),
        ] {
            let (rh, rw) = gemma4_resize_dims(h, w, p).unwrap();
            assert_eq!(rh % 48, 0);
            assert_eq!(rw % 48, 0);
            assert!((rh / 16) * (rw / 16) <= p);
        }
    }

    #[test]
    fn patches_are_row_major_and_pool_by_3x3_blocks() {
        // A 96x48 image: a 6x3 patch grid, two soft tokens side by side.
        let (w, h) = (96, 48);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| [(i % 251) as u8, 0, 255]).collect();
        let p = ImagePatches::from_rgb(&rgb, w, h).unwrap();
        assert_eq!((p.grid_w, p.grid_h), (6, 3));
        assert_eq!(p.positions[0], (0, 0));
        assert_eq!(p.positions[1], (1, 0));
        assert_eq!(p.positions[6], (0, 1));
        assert_eq!(p.soft_tokens(), 2);
        assert_eq!(p.pool_index(0), 0);
        assert_eq!(p.pool_index(2), 0);
        assert_eq!(p.pool_index(3), 1);
        assert_eq!(p.pool_index(17), 1);
        // Patch 1 starts at pixel (0, 16); channel order is RGB.
        let patch1 = &p.pixels[768..768 + 3];
        assert_eq!(patch1, &[16.0 / 255.0, 0.0, 1.0]);
        assert!(ImagePatches::from_rgb(&rgb[..3 * 50 * 48], 50, 48).is_err());
    }
}
