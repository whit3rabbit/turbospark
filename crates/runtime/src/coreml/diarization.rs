//! Offline speaker diarization on Core ML: FluidInference
//! `speaker-diarization-coreml`, the pyannote community-1 set (CC-BY-4.0
//! scope; the legacy `pyannote_segmentation` and `wespeaker*` programs in the
//! same repository are not used).
//!
//! The export ships four programs and the PLDA eigenvalues:
//!
//! - `Segmentation`: 10 s of 16 kHz PCM to powerset log-probabilities
//!   `[B,589,7]` over three local speakers, at most two at once.
//! - `FBank`: the same 10 s to WeSpeaker fbank features `[B,1,80,998]`,
//!   already mean-subtracted over frames.
//! - `Embedding`: fbank features plus a 589-frame speaker mask to a raw
//!   256-dim embedding.
//! - `PldaRho`: embeddings to PLDA features times `sqrt(psi)` `[B,128]`.
//! - `plda-parameters.json`: `psi`, the across-class variances VBx needs.
//!
//! The host side follows pyannote.audio 4.0.0 `SpeakerDiarization` with
//! `VBxClustering` (commit `a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a`, MIT):
//! 10 s windows every 1 s, an embedding for every active local speaker
//! (overlap excluded when enough clean speech remains), centroid AHC cut at
//! 0.6 to seed VBx (Fa 0.07, Fb 0.8), constrained assignment, then the
//! exclusive reconstruction (one speaker per frame). Two details come from
//! FluidAudio at `2d6c42ca5f3accc0faf9fee9e0fff5b9b2cf6042` (Apache-2.0),
//! which made this export: the FBank re-centering on non-silent frames, and
//! the 1 s minimum of clean speech before overlap is excluded.
//!
//! One deliberate difference from FluidAudio: it feeds the `PldaRho` output
//! to VBx as the features and scales it by `sqrt(psi)` a second time. Here
//! the features are `rho / sqrt(psi)`, as pyannote's VBx expects. The JSON
//! `psi` equals the graph's `sqrt_phi` squared (checked on the pinned files).

use std::path::Path;

use turbospark_coreml::{ComputeUnits, CoreMlModel, LoadOptions, Tensor};

pub const SAMPLE_RATE: u32 = 16_000;
pub const REQUIRED_DIRS: [&str; 4] = [
    "Segmentation.mlmodelc",
    "FBank.mlmodelc",
    "Embedding.mlmodelc",
    "PldaRho.mlmodelc",
];
pub const PLDA_FILE: &str = "plda-parameters.json";
/// Persisted family string.
pub const FAMILY: &str = "pyannote_community1";

/// Samples per segmentation window: 10 s.
const WINDOW: usize = 160_000;
/// pyannote's default `segmentation_step` of 0.1 window.
const CHUNK_STEP: usize = 16_000;
/// The segmentation model's receptive field (SincNet stride 270 samples,
/// 991 samples wide). Frames of every window land on this grid.
const FRAME_STEP: usize = 270;
const FRAME_SPAN: usize = 991;
const LOCAL_SPEAKERS: usize = 3;
/// pyannote `Powerset(3, 2)` class order: {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}.
const POWERSET: [u8; 7] = [0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110];
const EMBEDDING_DIM: usize = 256;
const PLDA_DIM: usize = 128;
/// Every program accepts batches of 1 through 32.
const MAX_BATCH: usize = 32;
/// FBank frame geometry (25 ms every 10 ms), for finding silent frames.
const FBANK_FRAME: usize = 400;
const FBANK_SHIFT: usize = 160;

/// pyannote `filter_embeddings`: train only on speakers alone in at least
/// 20% of the window's frames.
const MIN_ACTIVE_RATIO: f64 = 0.2;
/// Clean speech needed before overlap is excluded from a mask. pyannote
/// uses the embedding model's minimum input length, which this export does
/// not record; FluidAudio uses 1 s for this export.
const MIN_CLEAN_SECONDS: f64 = 1.0;
const AHC_THRESHOLD: f64 = 0.6;
const VBX_FA: f64 = 0.07;
const VBX_FB: f64 = 0.8;
const VBX_MAX_ITERS: usize = 20;
const VBX_EPSILON: f64 = 1e-4;
const VBX_INIT_SMOOTHING: f64 = 7.0;
/// VBx speakers whose prior falls below this are dropped.
const VBX_MIN_PRIOR: f64 = 1e-7;

const EMBEDDING_UNITS_ENV: &str = "TURBOSPARK_DIARIZATION_COREML_EMBEDDING_UNITS";
/// Lets a subprocess probe place the embedding program on the GPU. Shared
/// with the Parakeet runner: an unmeasured placement can abort the process.
const ALLOW_UNVERIFIED_GPU_ENV: &str = "TURBOSPARK_COREML_ALLOW_QUANTIZED_GPU";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiarizationOptions {
    /// The only program with Neural Engine coverage: `MLComputePlan` puts
    /// 99.96% of its cost there. Segmentation (LSTMs), FBank and PldaRho
    /// plan entirely on the CPU whatever is requested, so they load CPU-only.
    pub embedding_units: ComputeUnits,
}

impl Default for DiarizationOptions {
    fn default() -> Self {
        Self {
            embedding_units: ComputeUnits::CpuAndNeuralEngine,
        }
    }
}

impl DiarizationOptions {
    /// Defaults, then `TURBOSPARK_DIARIZATION_COREML_EMBEDDING_UNITS`
    /// (`cpu`, `cpu_gpu`, `cpu_ane`, `all`). An unknown value is an error;
    /// a GPU choice is refused at open.
    pub fn from_env() -> Result<Self, String> {
        let mut o = Self::default();
        if let Ok(v) = std::env::var(EMBEDDING_UNITS_ENV) {
            o.embedding_units = ComputeUnits::parse(v.trim()).ok_or_else(|| {
                format!("{EMBEDDING_UNITS_ENV}={v} is not one of cpu, cpu_gpu, cpu_ane, all")
            })?;
        }
        Ok(o)
    }
}

/// The embedding program was never run on the GPU here.
fn check_embedding_units(units: ComputeUnits, allow_unverified: bool) -> Result<(), String> {
    if matches!(units, ComputeUnits::CpuAndGpu | ComputeUnits::All) && !allow_unverified {
        return Err(
            "the Core ML speaker embedding is not verified on the GPU, where Core ML can abort \
             the process; use cpu_ane or cpu"
                .into(),
        );
    }
    Ok(())
}

/// One speaker's turn on the input timeline, in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerTurn {
    /// 0-based, numbered in order of first appearance.
    pub speaker: usize,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Diarization {
    /// Non-overlapping turns sorted by start.
    pub turns: Vec<SpeakerTurn>,
    pub speakers: usize,
}

pub struct DiarizationCoreMl {
    segmentation: CoreMlModel,
    fbank: CoreMlModel,
    embedding: CoreMlModel,
    plda_rho: CoreMlModel,
    /// PLDA across-class variances (VBx `Phi`), 128 values.
    psi: Vec<f64>,
}

impl std::fmt::Debug for DiarizationCoreMl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiarizationCoreMl").finish_non_exhaustive()
    }
}

impl DiarizationCoreMl {
    pub fn open(dir: &Path, opts: &DiarizationOptions) -> Result<Self, String> {
        let (major, _, _) = turbospark_coreml::os_version();
        // The programs declare macOS 14, where FluidAudio reports Core ML's
        // CPU path crashing in BNNS during this pipeline (fixed in 15).
        if major < 15 {
            return Err("speaker diarization on Core ML requires macOS 15 or newer".into());
        }
        check_embedding_units(
            opts.embedding_units,
            std::env::var(ALLOW_UNVERIFIED_GPU_ENV).as_deref() == Ok("1"),
        )?;
        let psi = read_psi(&dir.join(PLDA_FILE))?;
        let cache = super::cache_root(dir);
        let load = |name: &str, units: ComputeUnits| {
            CoreMlModel::open(&dir.join(name), &cache, &LoadOptions::new(units))
                .map_err(|e| e.to_string())
        };
        Ok(Self {
            segmentation: load(REQUIRED_DIRS[0], ComputeUnits::CpuOnly)?,
            fbank: load(REQUIRED_DIRS[1], ComputeUnits::CpuOnly)?,
            embedding: load(REQUIRED_DIRS[2], opts.embedding_units)?,
            plda_rho: load(REQUIRED_DIRS[3], ComputeUnits::CpuOnly)?,
            psi,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Diarizes mono 16 kHz PCM. `progress(done, total)` runs before each
    /// batch of windows and stops the run by returning `false`, as
    /// `cancelled` does.
    pub fn diarize(
        &self,
        samples: &[f32],
        sample_rate: u32,
        cancelled: &dyn Fn() -> bool,
        mut progress: impl FnMut(usize, usize) -> bool,
    ) -> Result<Diarization, String> {
        if sample_rate != SAMPLE_RATE {
            return Err(format!(
                "speaker diarization expects {SAMPLE_RATE} Hz PCM, received {sample_rate} Hz"
            ));
        }
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            return Err("speaker diarization input must be nonempty finite mono PCM".into());
        }
        let starts = chunk_starts(samples.len());
        let frames = frames_per_window();
        let mut masks: Vec<Vec<u8>> = Vec::with_capacity(starts.len());
        let mut embeddings: Vec<[Option<Vec<f32>>; LOCAL_SPEAKERS]> =
            Vec::with_capacity(starts.len());
        let mut features: Vec<[Option<Vec<f64>>; LOCAL_SPEAKERS]> =
            Vec::with_capacity(starts.len());
        for (group_index, group) in starts.chunks(MAX_BATCH).enumerate() {
            if cancelled() || !progress(group_index * MAX_BATCH, starts.len()) {
                return Err("speaker diarization cancelled".into());
            }
            let mut audio = vec![0.0f32; group.len() * WINDOW];
            for (row, &start) in group.iter().enumerate() {
                let end = (start + WINDOW).min(samples.len());
                audio[row * WINDOW..row * WINDOW + end - start]
                    .copy_from_slice(&samples[start..end]);
            }
            let audio = Tensor::f32(vec![group.len(), 1, WINDOW], audio).map_err(err)?;
            let group_masks = self.segment(&audio, group.len(), frames)?;
            let fbank = self.fbank(&audio, group.len())?;
            let (group_embeddings, group_features) =
                self.embed(&fbank, &group_masks, group.len())?;
            masks.extend(group_masks);
            embeddings.extend(group_embeddings);
            features.extend(group_features);
        }
        if cancelled() || !progress(starts.len(), starts.len()) {
            return Err("speaker diarization cancelled".into());
        }
        let hard = cluster(&masks, &embeddings, &features, &self.psi, frames);
        let mut turns = reconstruct(&masks, &hard, frames);
        let duration = samples.len() as f64 / SAMPLE_RATE as f64;
        for t in &mut turns {
            t.end = t.end.min(duration);
        }
        turns.retain(|t| t.end > t.start);
        Ok(renumber(turns))
    }

    /// Segmentation for `batch` windows: each frame's local speakers as a
    /// 3-bit set.
    fn segment(&self, audio: &Tensor, batch: usize, frames: usize) -> Result<Vec<Vec<u8>>, String> {
        let out = self
            .segmentation
            .predict(&[("audio", audio)])
            .map_err(err)?;
        let logits = take(&out, "log_probs")?;
        if logits.shape() != [batch, frames, POWERSET.len()] {
            return Err(format!(
                "segmentation output {:?} is not [{batch}, {frames}, {}]",
                logits.shape(),
                POWERSET.len()
            ));
        }
        let logits = logits
            .as_f32()
            .ok_or("segmentation output is not float32")?;
        Ok(logits
            .chunks(frames * POWERSET.len())
            .map(|chunk| {
                chunk
                    .chunks(POWERSET.len())
                    .map(powerset_speakers)
                    .collect()
            })
            .collect())
    }

    /// FBank features `[batch, 80 * 998]`, re-centered on non-silent frames.
    fn fbank(&self, audio: &Tensor, batch: usize) -> Result<Fbank, String> {
        let out = self.fbank.predict(&[("audio", audio)]).map_err(err)?;
        let t = take(&out, "fbank_features")?;
        let shape = t.shape();
        if shape.len() != 4 || shape[0] != batch || shape[1] != 1 {
            return Err(format!(
                "fbank output {shape:?} is not [{batch}, 1, bands, frames]"
            ));
        }
        let (bands, frames) = (shape[2], shape[3]);
        let mut data = t.as_f32().ok_or("fbank output is not float32")?.to_vec();
        let pcm = audio.as_f32().ok_or("audio is not float32")?;
        for (row, features) in data.chunks_mut(bands * frames).enumerate() {
            let silent = silent_frames(&pcm[row * WINDOW..(row + 1) * WINDOW], frames);
            recenter(features, bands, frames, &silent);
        }
        Ok(Fbank {
            data,
            bands,
            frames,
        })
    }

    /// Raw embeddings and PLDA features for every active local speaker.
    #[allow(clippy::type_complexity)]
    fn embed(
        &self,
        fbank: &Fbank,
        masks: &[Vec<u8>],
        batch: usize,
    ) -> Result<
        (
            Vec<[Option<Vec<f32>>; LOCAL_SPEAKERS]>,
            Vec<[Option<Vec<f64>>; LOCAL_SPEAKERS]>,
        ),
        String,
    > {
        let frames = masks.first().map_or(0, Vec::len);
        let min_clean =
            (MIN_CLEAN_SECONDS * SAMPLE_RATE as f64 / FRAME_STEP as f64).ceil() as usize;
        let mut jobs: Vec<(usize, usize, Vec<f32>)> = Vec::new();
        for (row, mask) in masks.iter().enumerate().take(batch) {
            for speaker in 0..LOCAL_SPEAKERS {
                if let Some(weights) = speaker_weights(mask, speaker, min_clean) {
                    jobs.push((row, speaker, weights));
                }
            }
        }
        let mut embeddings = vec![[None, None, None]; batch];
        let mut features = vec![[None, None, None]; batch];
        let row_len = fbank.bands * fbank.frames;
        for part in jobs.chunks(MAX_BATCH) {
            let mut x = Vec::with_capacity(part.len() * row_len);
            let mut w = Vec::with_capacity(part.len() * frames);
            for (row, _, weights) in part {
                x.extend_from_slice(&fbank.data[row * row_len..(row + 1) * row_len]);
                w.extend_from_slice(weights);
            }
            let x = Tensor::f32(vec![part.len(), 1, fbank.bands, fbank.frames], x).map_err(err)?;
            let w = Tensor::f32(vec![part.len(), frames], w).map_err(err)?;
            let out = self
                .embedding
                .predict(&[("fbank_features", &x), ("weights", &w)])
                .map_err(err)?;
            let e = take(&out, "embedding")?;
            if e.shape() != [part.len(), EMBEDDING_DIM] {
                return Err(format!(
                    "embedding output {:?} is not [{}, {EMBEDDING_DIM}]",
                    e.shape(),
                    part.len()
                ));
            }
            let e = e
                .as_f32()
                .ok_or("embedding output is not float32")?
                .to_vec();
            let rho = self.plda_rho(&e, part.len())?;
            for (i, (row, speaker, _)) in part.iter().enumerate() {
                let embedding = e[i * EMBEDDING_DIM..(i + 1) * EMBEDDING_DIM].to_vec();
                // pyannote treats a non-finite embedding as missing.
                if embedding.iter().all(|v| v.is_finite()) {
                    let fea: Vec<f64> = rho[i * PLDA_DIM..(i + 1) * PLDA_DIM]
                        .iter()
                        .zip(&self.psi)
                        .map(|(&r, &p)| f64::from(r) / p.sqrt())
                        .collect();
                    if fea.iter().all(|v| v.is_finite()) {
                        features[*row][*speaker] = Some(fea);
                    }
                    embeddings[*row][*speaker] = Some(embedding);
                }
            }
        }
        Ok((embeddings, features))
    }

    fn plda_rho(&self, embeddings: &[f32], n: usize) -> Result<Vec<f32>, String> {
        let x = Tensor::f32(vec![n, EMBEDDING_DIM], embeddings.to_vec()).map_err(err)?;
        let out = self.plda_rho.predict(&[("embeddings", &x)]).map_err(err)?;
        let rho = take(&out, "rho")?;
        if rho.shape() != [n, PLDA_DIM] {
            return Err(format!(
                "PldaRho output {:?} is not [{n}, {PLDA_DIM}]",
                rho.shape()
            ));
        }
        Ok(rho
            .as_f32()
            .ok_or("PldaRho output is not float32")?
            .to_vec())
    }
}

struct Fbank {
    data: Vec<f32>,
    bands: usize,
    frames: usize,
}

fn err(e: turbospark_coreml::CoreMlError) -> String {
    e.to_string()
}

fn take<'a>(
    m: &'a std::collections::HashMap<String, Tensor>,
    name: &str,
) -> Result<&'a Tensor, String> {
    m.get(name)
        .ok_or_else(|| format!("Core ML output {name} is missing"))
}

/// `psi` from `plda-parameters.json`: 128 positive float32 values, base64.
fn read_psi(path: &Path) -> Result<Vec<f64>, String> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    // The pinned file is 89 KB; the cap keeps a foreign file from being read whole.
    file.take((1 << 20) + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() > 1 << 20 {
        return Err(format!("{}: exceeds the 1 MiB limit", path.display()));
    }
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let encoded = v
        .pointer("/tensors/psi/data_base64")
        .and_then(|d| d.as_str())
        .ok_or_else(|| format!("{}: tensors.psi.data_base64 is missing", path.display()))?;
    let raw =
        decode_base64(encoded).ok_or_else(|| format!("{}: psi is not base64", path.display()))?;
    if raw.len() != PLDA_DIM * 4 {
        return Err(format!(
            "{}: psi has {} bytes, expected {}",
            path.display(),
            raw.len(),
            PLDA_DIM * 4
        ));
    }
    let psi: Vec<f64> = raw
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        .collect();
    if psi.iter().any(|p| !p.is_finite() || *p <= 0.0) {
        return Err(format!(
            "{}: psi must be positive and finite",
            path.display()
        ));
    }
    Ok(psi)
}

/// Standard-alphabet base64 with optional padding; whitespace is skipped.
fn decode_base64(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes().filter(|c| !c.is_ascii_whitespace()) {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

/// Frames the segmentation model emits per window.
fn frames_per_window() -> usize {
    (WINDOW - FRAME_SPAN) / FRAME_STEP + 1
}

/// Window starts, pyannote `Inference.slide`: whole windows every step,
/// plus one zero-padded window when audio is left over.
fn chunk_starts(len: usize) -> Vec<usize> {
    let full = if len >= WINDOW {
        (len - WINDOW) / CHUNK_STEP + 1
    } else {
        0
    };
    let mut starts: Vec<usize> = (0..full).map(|c| c * CHUNK_STEP).collect();
    if len < WINDOW || (len - WINDOW) % CHUNK_STEP > 0 {
        starts.push(full * CHUNK_STEP);
    }
    starts
}

/// The argmax powerset class as a set of local speakers.
fn powerset_speakers(logits: &[f32]) -> u8 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    POWERSET[best]
}

/// The mask an embedding of `speaker` uses, or `None` when the speaker is
/// silent in this window. Overlapped frames are left out when at least
/// `min_clean` frames remain (pyannote `embedding_exclude_overlap`).
fn speaker_weights(mask: &[u8], speaker: usize, min_clean: usize) -> Option<Vec<f32>> {
    let bit = 1u8 << speaker;
    let active = mask.iter().filter(|&&m| m & bit != 0).count();
    if active == 0 {
        return None;
    }
    let clean = mask.iter().filter(|&&m| m == bit).count();
    let exclude_overlap = clean > min_clean;
    Some(
        mask.iter()
            .map(|&m| {
                let on = if exclude_overlap {
                    m == bit
                } else {
                    m & bit != 0
                };
                if on {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
    )
}

/// FBank frames whose samples are all exactly zero (padding, muted input).
fn silent_frames(audio: &[f32], frames: usize) -> Vec<bool> {
    let mut nonzero = vec![0usize; audio.len() + 1];
    for (i, &s) in audio.iter().enumerate() {
        nonzero[i + 1] = nonzero[i] + usize::from(s != 0.0);
    }
    (0..frames)
        .map(|f| {
            let start = (f * FBANK_SHIFT).min(audio.len());
            let end = (start + FBANK_FRAME).min(audio.len());
            nonzero[end] == nonzero[start]
        })
        .collect()
}

/// The FBank program subtracts each band's mean over all frames. Exact
/// silence sits at the log floor, so in a mostly silent window that mean
/// drowns the voice. Subtracting the output's mean over the voiced frames
/// gives the features of a mean taken over voiced frames only (FluidAudio
/// `SilenceAwareFbank`). Windows with no silent or no voiced frame are kept.
fn recenter(features: &mut [f32], bands: usize, frames: usize, silent: &[bool]) {
    let voiced: Vec<usize> = (0..frames).filter(|&f| !silent[f]).collect();
    if voiced.is_empty() || voiced.len() == frames {
        return;
    }
    for band in features.chunks_mut(frames).take(bands) {
        let mean = voiced.iter().map(|&f| f64::from(band[f])).sum::<f64>() / voiced.len() as f64;
        for v in band.iter_mut() {
            *v -= mean as f32;
        }
    }
}

/// Cluster id per (window, local speaker); `None` for silent speakers and
/// speakers left without a cluster.
fn cluster(
    masks: &[Vec<u8>],
    embeddings: &[[Option<Vec<f32>>; LOCAL_SPEAKERS]],
    features: &[[Option<Vec<f64>>; LOCAL_SPEAKERS]],
    psi: &[f64],
    frames: usize,
) -> Vec<[Option<usize>; LOCAL_SPEAKERS]> {
    let min_train = MIN_ACTIVE_RATIO * frames as f64;
    let mut train: Vec<(usize, usize)> = Vec::new();
    for (c, mask) in masks.iter().enumerate() {
        for s in 0..LOCAL_SPEAKERS {
            let clean = mask.iter().filter(|&&m| m == 1 << s).count();
            if clean as f64 >= min_train && embeddings[c][s].is_some() && features[c][s].is_some() {
                train.push((c, s));
            }
        }
    }
    let active = |c: usize, s: usize| masks[c].iter().any(|&m| m & (1 << s) != 0);
    if train.len() < 2 {
        // pyannote: too little to cluster, so every speaker is one cluster.
        return (0..masks.len())
            .map(|c| std::array::from_fn(|s| active(c, s).then_some(0)))
            .collect();
    }
    let raw: Vec<Vec<f64>> = train
        .iter()
        .map(|&(c, s)| {
            embeddings[c][s]
                .as_ref()
                .expect("filtered above")
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        })
        .collect();
    let normed: Vec<Vec<f64>> = raw.iter().map(|v| normalize(v)).collect();
    let ahc = ahc_centroid(&normed, AHC_THRESHOLD);
    let fea: Vec<Vec<f64>> = train
        .iter()
        .map(|&(c, s)| features[c][s].clone().expect("filtered above"))
        .collect();
    let (gamma, pi) = vbx(&fea, psi, &ahc, VBX_FA, VBX_FB, VBX_MAX_ITERS);
    let kept: Vec<usize> = (0..pi.len()).filter(|&k| pi[k] > VBX_MIN_PRIOR).collect();
    let centroids: Vec<Vec<f64>> = kept
        .iter()
        .map(|&k| {
            let mut sum = vec![0.0; EMBEDDING_DIM];
            let mut weight = 0.0;
            for (t, row) in raw.iter().enumerate() {
                let g = gamma[t][k];
                weight += g;
                for (a, b) in sum.iter_mut().zip(row) {
                    *a += g * b;
                }
            }
            sum.iter().map(|v| v / weight).collect()
        })
        .collect();
    let centroids: Vec<Vec<f64>> = centroids.iter().map(|c| normalize(c)).collect();
    // pyannote scores with 2 - cosine distance, silent speakers below all.
    let mut scores: Vec<[Vec<f64>; LOCAL_SPEAKERS]> = Vec::with_capacity(masks.len());
    let mut lowest = f64::INFINITY;
    for (c, row) in embeddings.iter().enumerate() {
        scores.push(std::array::from_fn(|s| match (&row[s], active(c, s)) {
            (Some(e), true) => {
                let e = normalize(&e.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
                centroids.iter().map(|k| 1.0 + dot(&e, k)).collect()
            }
            _ => Vec::new(),
        }));
        for v in scores[c].iter().flatten() {
            lowest = lowest.min(*v);
        }
    }
    let floor = if lowest.is_finite() {
        lowest - 1.0
    } else {
        0.0
    };
    scores
        .into_iter()
        .enumerate()
        .map(|(c, row)| {
            let matrix: Vec<Vec<f64>> = row
                .iter()
                .map(|r| {
                    if r.is_empty() {
                        vec![floor; centroids.len()]
                    } else {
                        r.clone()
                    }
                })
                .collect();
            let assigned = max_score_assignment(&matrix);
            std::array::from_fn(|s| if active(c, s) { assigned[s] } else { None })
        })
        .collect()
}

fn normalize(v: &[f64]) -> Vec<f64> {
    let n = dot(v, v).sqrt();
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Centroid-linkage agglomerative clustering on unit-normalized rows, cut
/// like scipy `fcluster(criterion="distance")`: a subtree is one cluster
/// when no merge inside it is higher than `threshold`. Labels are 0..k.
fn ahc_centroid(rows: &[Vec<f64>], threshold: f64) -> Vec<usize> {
    let n = rows.len();
    if n < 2 {
        return vec![0; n];
    }
    let mut centroid = rows.to_vec();
    let mut size = vec![1usize; n];
    let mut alive = vec![true; n];
    let mut node = (0..n).collect::<Vec<_>>();
    let nearest = |i: usize, centroid: &[Vec<f64>], alive: &[bool]| {
        let mut best = (usize::MAX, f64::INFINITY);
        for j in 0..centroid.len() {
            if j != i && alive[j] {
                let d = squared_distance(&centroid[i], &centroid[j]);
                if d < best.1 {
                    best = (j, d);
                }
            }
        }
        best
    };
    let mut nn: Vec<(usize, f64)> = (0..n).map(|i| nearest(i, &centroid, &alive)).collect();
    // Internal node n + m merged children[m] at height[m].
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(n - 1);
    let mut height: Vec<f64> = Vec::with_capacity(n - 1);
    for step in 0..n - 1 {
        let a = (0..n)
            .filter(|&i| alive[i])
            .min_by(|&i, &j| nn[i].1.total_cmp(&nn[j].1))
            .expect("two clusters remain");
        let b = nn[a].0;
        children.push((node[a], node[b]));
        height.push(nn[a].1.sqrt());
        let (sa, sb) = (size[a] as f64, size[b] as f64);
        let merged: Vec<f64> = centroid[a]
            .iter()
            .zip(&centroid[b])
            .map(|(x, y)| (sa * x + sb * y) / (sa + sb))
            .collect();
        centroid[a] = merged;
        size[a] += size[b];
        alive[b] = false;
        node[a] = n + step;
        if step + 2 == n {
            break;
        }
        nn[a] = nearest(a, &centroid, &alive);
        for i in 0..n {
            if !alive[i] || i == a {
                continue;
            }
            if nn[i].0 == a || nn[i].0 == b {
                nn[i] = nearest(i, &centroid, &alive);
            } else {
                let d = squared_distance(&centroid[i], &centroid[a]);
                if d < nn[i].1 {
                    nn[i] = (a, d);
                }
            }
        }
    }
    // The highest merge inside each internal node's subtree.
    let mut max_height = vec![0.0f64; n - 1];
    for m in 0..n - 1 {
        let (l, r) = children[m];
        let sub = |c: usize| if c >= n { max_height[c - n] } else { 0.0 };
        max_height[m] = height[m].max(sub(l)).max(sub(r));
    }
    let mut labels = vec![usize::MAX; n];
    let mut next = 0;
    let mut stack = vec![2 * n - 2];
    while let Some(v) = stack.pop() {
        if v < n {
            labels[v] = next;
            next += 1;
        } else if max_height[v - n] <= threshold {
            let mut leaves = vec![v];
            while let Some(u) = leaves.pop() {
                if u < n {
                    labels[u] = next;
                } else {
                    let (l, r) = children[u - n];
                    leaves.extend([l, r]);
                }
            }
            next += 1;
        } else {
            let (l, r) = children[v - n];
            stack.extend([r, l]);
        }
    }
    labels
}

/// pyannote `cluster_vbx` and `VBx`: `x` are PLDA features (T x D), `phi`
/// the across-class variances, `init` the AHC labels. Returns the
/// responsibilities (T x S) and speaker priors (S).
fn vbx(
    x: &[Vec<f64>],
    phi: &[f64],
    init: &[usize],
    fa: f64,
    fb: f64,
    max_iters: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let speakers = init.iter().max().map_or(1, |m| m + 1);
    let d = phi.len();
    let mut gamma: Vec<Vec<f64>> = init
        .iter()
        .map(|&k| {
            let row: Vec<f64> = (0..speakers)
                .map(|s| if s == k { VBX_INIT_SMOOTHING } else { 0.0 })
                .collect();
            softmax(&row)
        })
        .collect();
    let mut pi = vec![1.0 / speakers as f64; speakers];
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let g: Vec<f64> = x
        .iter()
        .map(|row| -0.5 * (dot(row, row) + d as f64 * log_2pi))
        .collect();
    let sqrt_phi: Vec<f64> = phi.iter().map(|p| p.sqrt()).collect();
    let rho: Vec<Vec<f64>> = x
        .iter()
        .map(|row| row.iter().zip(&sqrt_phi).map(|(a, b)| a * b).collect())
        .collect();
    let ratio = fa / fb;
    let mut previous = f64::NEG_INFINITY;
    for iter in 0..max_iters {
        let mut inv_l = vec![vec![0.0; d]; speakers];
        let mut alpha = vec![vec![0.0; d]; speakers];
        for s in 0..speakers {
            let total: f64 = gamma.iter().map(|r| r[s]).sum();
            for j in 0..d {
                inv_l[s][j] = 1.0 / (1.0 + ratio * total * phi[j]);
            }
            for (t, r) in rho.iter().enumerate() {
                let w = gamma[t][s];
                if w != 0.0 {
                    for j in 0..d {
                        alpha[s][j] += w * r[j];
                    }
                }
            }
            for j in 0..d {
                alpha[s][j] *= ratio * inv_l[s][j];
            }
        }
        let offset: Vec<f64> = (0..speakers)
            .map(|s| {
                -0.5 * (0..d)
                    .map(|j| (inv_l[s][j] + alpha[s][j] * alpha[s][j]) * phi[j])
                    .sum::<f64>()
            })
            .collect();
        let log_pi: Vec<f64> = pi.iter().map(|p| (p + 1e-8).ln()).collect();
        let mut log_px_total = 0.0;
        for (t, r) in rho.iter().enumerate() {
            let scores: Vec<f64> = (0..speakers)
                .map(|s| fa * (dot(r, &alpha[s]) + offset[s] + g[t]) + log_pi[s])
                .collect();
            let lse = log_sum_exp(&scores);
            log_px_total += lse;
            gamma[t] = scores.iter().map(|v| (v - lse).exp()).collect();
        }
        let mut sum = vec![0.0; speakers];
        for row in &gamma {
            for (a, b) in sum.iter_mut().zip(row) {
                *a += b;
            }
        }
        let total: f64 = sum.iter().sum();
        pi = sum.iter().map(|v| v / total).collect();
        let reg: f64 = (0..speakers)
            .flat_map(|s| (0..d).map(move |j| (s, j)))
            .map(|(s, j)| inv_l[s][j].ln() - inv_l[s][j] - alpha[s][j] * alpha[s][j] + 1.0)
            .sum();
        let elbo = log_px_total + fb * 0.5 * reg;
        if iter > 0 && elbo - previous < VBX_EPSILON {
            break;
        }
        previous = elbo;
    }
    (gamma, pi)
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let lse = log_sum_exp(v);
    v.iter().map(|x| (x - lse).exp()).collect()
}

fn log_sum_exp(v: &[f64]) -> f64 {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + v.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

/// Maximum-total-score assignment of rows to distinct columns (scipy
/// `linear_sum_assignment(maximize=True)`). Rows left over when there are
/// more rows than columns get `None`.
fn max_score_assignment(scores: &[Vec<f64>]) -> Vec<Option<usize>> {
    let rows = scores.len();
    let cols = scores.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return vec![None; rows];
    }
    let n = rows.max(cols);
    let max = scores
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let min = scores
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    // Padding cells share one cost so they never decide a real pair.
    let worst = if min.is_finite() { min - 1.0 } else { 0.0 };
    let max = if max.is_finite() { max } else { 0.0 };
    let cost = |r: usize, c: usize| -> f64 {
        if r < rows && c < cols && scores[r][c].is_finite() {
            max - scores[r][c]
        } else if r < rows && c < cols {
            max - worst
        } else {
            0.0
        }
    };
    // Kuhn-Munkres with potentials, 1-based columns; p[j] is column j's row.
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![None; rows];
    for j in 1..=n {
        let r = p[j];
        if r >= 1 && r <= rows && j <= cols {
            out[r - 1] = Some(j - 1);
        }
    }
    out
}

/// Round half to even, as numpy `rint`.
fn rint(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// pyannote's exclusive reconstruction: the per-frame speaker count from the
/// averaged segmentation, capped at one, picks the cluster with the largest
/// summed activity in each frame; frames become turns between frame centers.
fn reconstruct(
    masks: &[Vec<u8>],
    hard: &[[Option<usize>; LOCAL_SPEAKERS]],
    frames: usize,
) -> Vec<SpeakerTurn> {
    if masks.is_empty() {
        return Vec::new();
    }
    let clusters = hard.iter().flatten().flatten().max().map_or(0, |m| m + 1);
    let offset = |c: usize| (c * CHUNK_STEP + FRAME_STEP / 2) / FRAME_STEP;
    let total =
        rint((WINDOW + (masks.len() - 1) * CHUNK_STEP) as f64 / FRAME_STEP as f64) as usize + 1;
    let mut count_sum = vec![0.0f64; total];
    let mut cover = vec![0.0f64; total];
    let mut activation = vec![vec![0.0f64; clusters]; total];
    for (c, mask) in masks.iter().enumerate() {
        let start = offset(c);
        for (f, &m) in mask.iter().enumerate().take(frames) {
            let Some(g) = start.checked_add(f).filter(|&g| g < total) else {
                break;
            };
            count_sum[g] += f64::from(m.count_ones());
            cover[g] += 1.0;
            for (s, k) in hard[c].iter().enumerate() {
                if let Some(k) = k {
                    if m & (1 << s) != 0 {
                        // Several local speakers on one cluster count once (max).
                        let already = hard[c][..s]
                            .iter()
                            .enumerate()
                            .any(|(p, q)| *q == Some(*k) && m & (1 << p) != 0);
                        if !already {
                            activation[g][*k] += 1.0;
                        }
                    }
                }
            }
        }
    }
    let mut speaker_at = vec![None; total];
    for g in 0..total {
        let count = if cover[g] > 0.0 {
            rint(count_sum[g] / cover[g])
        } else {
            0.0
        };
        if count >= 1.0 && clusters > 0 {
            let mut best = 0;
            for k in 1..clusters {
                if activation[g][k] > activation[g][best] {
                    best = k;
                }
            }
            speaker_at[g] = Some(best);
        }
    }
    let middle = |g: usize| {
        (g * FRAME_STEP) as f64 / SAMPLE_RATE as f64
            + FRAME_SPAN as f64 / (2.0 * SAMPLE_RATE as f64)
    };
    let mut turns = Vec::new();
    // (speaker, first frame) of the turn being built.
    let mut open: Option<(usize, usize)> = None;
    for (g, &k) in speaker_at.iter().enumerate() {
        if open.map(|o| o.0) == k {
            continue;
        }
        if let Some((spk, start)) = open.take() {
            turns.push(SpeakerTurn {
                speaker: spk,
                start: middle(start),
                end: middle(g),
            });
        }
        open = k.map(|k| (k, g));
    }
    if let Some((spk, start)) = open {
        turns.push(SpeakerTurn {
            speaker: spk,
            start: middle(start),
            end: middle(total - 1),
        });
    }
    turns
}

/// Speakers numbered by first appearance.
fn renumber(mut turns: Vec<SpeakerTurn>) -> Diarization {
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut order: Vec<usize> = Vec::new();
    for t in &mut turns {
        let id = match order.iter().position(|&s| s == t.speaker) {
            Some(i) => i,
            None => {
                order.push(t.speaker);
                order.len() - 1
            }
        };
        t.speaker = id;
    }
    Diarization {
        speakers: order.len(),
        turns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_grid_matches_the_segmentation_output() {
        assert_eq!(frames_per_window(), 589);
    }

    #[test]
    fn chunk_starts_follow_pyannote_slide() {
        assert_eq!(chunk_starts(1), vec![0]);
        assert_eq!(chunk_starts(WINDOW), vec![0]);
        assert_eq!(chunk_starts(WINDOW + 1), vec![0, CHUNK_STEP]);
        assert_eq!(
            chunk_starts(WINDOW + 2 * CHUNK_STEP),
            vec![0, CHUNK_STEP, 2 * CHUNK_STEP]
        );
        assert_eq!(
            chunk_starts(WINDOW + 2 * CHUNK_STEP + 5),
            vec![0, CHUNK_STEP, 2 * CHUNK_STEP, 3 * CHUNK_STEP]
        );
    }

    #[test]
    fn powerset_argmax_maps_to_speaker_sets() {
        let mut logits = [-9.0f32; 7];
        logits[0] = 0.0;
        assert_eq!(powerset_speakers(&logits), 0);
        logits[5] = 1.0;
        assert_eq!(powerset_speakers(&logits), 0b101);
        logits[3] = 2.0;
        assert_eq!(powerset_speakers(&logits), 0b100);
    }

    #[test]
    fn weights_exclude_overlap_only_with_enough_clean_frames() {
        let mask = [0b001, 0b001, 0b011, 0b010, 0];
        assert_eq!(
            speaker_weights(&mask, 0, 1).unwrap(),
            vec![1.0, 1.0, 0.0, 0.0, 0.0]
        );
        // Two clean frames are not more than two: keep the overlap.
        assert_eq!(
            speaker_weights(&mask, 0, 2).unwrap(),
            vec![1.0, 1.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(speaker_weights(&mask, 2, 0), None);
    }

    #[test]
    fn recenter_uses_voiced_frames_and_skips_uniform_windows() {
        // Two bands, four frames; the last frame is silent.
        let mut f = vec![1.0, 2.0, 3.0, -6.0, 0.0, 0.0, 0.0, 0.0];
        recenter(&mut f, 2, 4, &[false, false, false, true]);
        assert_eq!(f, vec![-1.0, 0.0, 1.0, -8.0, 0.0, 0.0, 0.0, 0.0]);
        let mut g = vec![1.0, 2.0];
        recenter(&mut g, 1, 2, &[false, false]);
        assert_eq!(g, vec![1.0, 2.0]);
        let audio = [0.0f32; 400]
            .iter()
            .chain([0.5f32; 160].iter())
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(silent_frames(&audio, 3), vec![true, false, false]);
    }

    #[test]
    fn base64_round_trips_psi_bytes() {
        assert_eq!(decode_base64("AACAPw==").unwrap(), 1.0f32.to_le_bytes());
        assert_eq!(decode_base64("AQID\nBA==").unwrap(), [1, 2, 3, 4]);
        assert_eq!(decode_base64("A*"), None);
    }

    #[test]
    fn ahc_cuts_two_tight_groups() {
        let rows: Vec<Vec<f64>> = [
            [1.0, 0.0, 0.0],
            [0.99, 0.141, 0.0],
            [0.0, 1.0, 0.0],
            [0.141, 0.99, 0.0],
            [0.98, 0.0, 0.199],
        ]
        .iter()
        .map(|r| normalize(r))
        .collect();
        let labels = ahc_centroid(&rows, 0.6);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[4]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
        assert_eq!(ahc_centroid(&rows, 2.0), vec![0; 5]);
        let mut all: Vec<usize> = ahc_centroid(&rows, 0.0);
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn ahc_uses_the_subtree_maximum_like_scipy() {
        // Centroid linkage is not monotone: {0,1} merges at 1.0 (point 2 is
        // 1.07 from both), then their centroid (0.5, 0) meets point 2 at
        // 0.95. A cut at 0.97 passes the root's own height, but fcluster
        // keeps the tree apart because its highest merge is above the cut.
        let rows = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.95]];
        let labels = ahc_centroid(&rows, 0.97);
        assert_eq!(
            labels
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );
        assert_eq!(ahc_centroid(&rows, 1.0), vec![0, 0, 0]);
    }

    #[test]
    fn vbx_separates_speakers_and_forgets_its_seed() {
        // Two speakers 10 apart in both dimensions, interleaved in time.
        let phi = vec![4.0, 4.0];
        let mut x = Vec::new();
        for i in 0..20 {
            let e = (i % 5) as f64 * 0.05;
            x.push(vec![5.0 + e, 5.0 - e]);
            x.push(vec![-5.0 - e, -5.0 + e]);
        }
        let init: Vec<usize> = (0..40).map(|i| i % 2).collect();
        let (gamma, pi) = vbx(&x, &phi, &init, 0.07, 0.8, 20);
        assert!(pi.iter().all(|&p| p > 0.4), "{pi:?}");
        assert!(
            gamma[0][0] > 0.99 && gamma[1][1] > 0.99,
            "{:?}",
            &gamma[..2]
        );
        // One speaker split across two AHC seeds: after the E-step every
        // point's responsibilities follow its features, not its seed.
        let init: Vec<usize> = (0..40).map(|i| usize::from(i % 4 == 0)).collect();
        let same: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0].abs(), r[1].abs()]).collect();
        let (gamma, _) = vbx(&same, &phi, &init, 0.07, 0.8, 20);
        let spread = gamma
            .iter()
            .map(|g| (g[0] - gamma[0][0]).abs())
            .fold(0.0, f64::max);
        assert!(spread < 1e-2, "{spread}");
    }

    #[test]
    fn assignment_maximizes_total_and_leaves_extra_rows() {
        let s = vec![vec![0.9, 0.8], vec![0.85, 0.1], vec![0.0, 0.0]];
        assert_eq!(max_score_assignment(&s), vec![Some(1), Some(0), None]);
        let one = vec![vec![0.2], vec![0.7]];
        assert_eq!(max_score_assignment(&one), vec![None, Some(0)]);
        let nan = vec![vec![f64::NAN, 0.5], vec![0.4, 0.3]];
        assert_eq!(max_score_assignment(&nan), vec![Some(1), Some(0)]);
    }

    #[test]
    fn rint_rounds_half_to_even() {
        assert_eq!(rint(0.5), 0.0);
        assert_eq!(rint(1.5), 2.0);
        assert_eq!(rint(2.5), 2.0);
        assert_eq!(rint(0.6), 1.0);
        assert_eq!(rint(0.4), 0.0);
    }

    #[test]
    fn reconstruct_hands_over_between_speakers() {
        let frames = frames_per_window();
        // One window: local speaker 0 for the first half, 1 for the second.
        let mask: Vec<u8> = (0..frames)
            .map(|f| if f < frames / 2 { 0b001 } else { 0b010 })
            .collect();
        let turns = reconstruct(&[mask], &[[Some(1), Some(0), None]], frames);
        assert_eq!(turns.len(), 2);
        assert_eq!((turns[0].speaker, turns[1].speaker), (1, 0));
        let half = (frames / 2 * FRAME_STEP) as f64 / 16_000.0 + FRAME_SPAN as f64 / 32_000.0;
        assert!((turns[0].end - half).abs() < 1e-9 && (turns[1].start - half).abs() < 1e-9);
        let d = renumber(turns);
        assert_eq!(d.speakers, 2);
        assert_eq!((d.turns[0].speaker, d.turns[1].speaker), (0, 1));
    }

    #[test]
    fn reconstruct_drops_frames_most_windows_call_silent() {
        let frames = frames_per_window();
        let speech: Vec<u8> = vec![0b001; frames];
        let silence: Vec<u8> = vec![0; frames];
        // Three windows, two silent: the averaged count rounds to zero where
        // only one of three windows hears speech.
        // Windows start at frames 0, 59 and 119; only frames 0..59 are heard
        // by a majority.
        let hard = [[Some(0), None, None]; 3];
        let turns = reconstruct(&[speech, silence.clone(), silence], &hard, frames);
        let middle = |g: usize| (g * FRAME_STEP) as f64 / 16_000.0 + FRAME_SPAN as f64 / 32_000.0;
        assert_eq!(turns.len(), 1, "{turns:?}");
        assert!((turns[0].start - middle(0)).abs() < 1e-9);
        assert!((turns[0].end - middle(59)).abs() < 1e-9, "{turns:?}");
    }

    #[test]
    fn gpu_embedding_placement_is_refused_unless_probed() {
        assert!(check_embedding_units(ComputeUnits::CpuAndGpu, false).is_err());
        assert!(check_embedding_units(ComputeUnits::All, false).is_err());
        assert!(check_embedding_units(ComputeUnits::CpuAndNeuralEngine, false).is_ok());
        assert!(check_embedding_units(ComputeUnits::CpuOnly, false).is_ok());
        assert!(check_embedding_units(ComputeUnits::All, true).is_ok());
    }
}
