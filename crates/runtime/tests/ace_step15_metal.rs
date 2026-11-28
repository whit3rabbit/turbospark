#![cfg(target_os = "macos")]
//! ACE-Step 1.5 on Metal: the condition encoder, DiT, turbo sampler and
//! FSQ cover path with projections resident on `gpu::Music3Device`
//! (bf16 weights, tiled float32 GEMM) and unmasked attention on the
//! device, checked against the same reference arrays as the CPU gates.
//!
//! The backend here is test-local on purpose: it is the seam the runtime
//! runner will own once the family is wired (crates/runtime/src is not
//! touched yet). Tiny fixtures run by default; the real checkpoint and
//! the timing run are ignored and need TURBOSPARK_ACE_STEP15_DIT and
//! TURBOSPARK_ACE_STEP15_DIT_REF (see the family README).

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use audio::backend::{
    AttentionShape, ComputeBackend, ConvShape, DType, DeviceWeight, WeightData, WeightEncoding,
};
use audio::music::ace_step15::{
    cover_hints, cover_switch_step, sample_turbo, turbo_schedule, AceConfig, AceStep15,
    AudioDetokenizer, AudioTokenizer, ConditionEncoder, Conditioning, Dit, Loader, OobleckConfig,
    OobleckVae, StepCondition, Text2Music,
};
use audio::SpeechError;
use gpu::{Music3DType, Music3Device, Music3Encoding, Music3Weight};
use model_io::safetensors::SafetensorsFile;

fn metal_error(e: gpu::GpuError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("ACE-Step Metal: {e}"),
    }
}

/// Dense 16- or 32-bit weights with F32 logical compute, and F32
/// non-causal attention. Everything else is refused, not approximated.
struct MetalBackend {
    device: Music3Device,
    prof: Rc<Prof>,
}
struct MetalWeight(Music3Weight, Rc<Prof>);

/// Wall time spent inside device calls, including upload and readback.
#[derive(Default)]
struct Prof {
    linear: Cell<Duration>,
    linears: Cell<usize>,
    attention: Cell<Duration>,
    attentions: Cell<usize>,
    conv: Cell<Duration>,
    convs: Cell<usize>,
}

impl Prof {
    fn add(&self, which: &Cell<Duration>, count: &Cell<usize>, started: Instant) {
        which.set(which.get() + started.elapsed());
        count.set(count.get() + 1);
    }
    fn reset(&self) {
        self.linear.set(Duration::ZERO);
        self.linears.set(0);
        self.attention.set(Duration::ZERO);
        self.attentions.set(0);
        self.conv.set(Duration::ZERO);
        self.convs.set(0);
    }
}

fn metal() -> MetalBackend {
    MetalBackend {
        device: Music3Device::new().expect("Metal device"),
        prof: Rc::default(),
    }
}

impl ComputeBackend for MetalBackend {
    fn load_weight(&self, data: WeightData<'_>) -> audio::Result<Rc<dyn DeviceWeight>> {
        let encoding = match data.encoding {
            WeightEncoding::Bf16 => Music3Encoding::Bf16,
            WeightEncoding::F16 => Music3Encoding::F16,
            WeightEncoding::F32 => Music3Encoding::F32,
            other => {
                return Err(SpeechError::Unsupported {
                    why: format!("ACE-Step Metal loads dense weights only, got {other:?}"),
                })
            }
        };
        if data.dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "ACE-Step Metal computes in F32".into(),
            });
        }
        let w = self
            .device
            .load_weight(data.shape, data.bytes, encoding, &[], &[], &[])
            .map_err(metal_error)?;
        let w = if data.tiled_f32 { w.tiled_for_f32() } else { w };
        Ok(Rc::new(MetalWeight(w, self.prof.clone())))
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: AttentionShape,
        d: DType,
    ) -> audio::Result<Vec<f32>> {
        if d != DType::F32 || s.causal || s.kv_time_major || s.offset != 0 {
            return Err(SpeechError::Unsupported {
                why: "ACE-Step attention is non-causal F32".into(),
            });
        }
        let started = Instant::now();
        let out = self
            .device
            .attention_typed(
                q,
                k,
                v,
                s.batch,
                s.queries,
                s.keys,
                s.heads,
                s.kv_heads,
                s.dim,
                false,
                false,
                0,
                Music3DType::F32,
            )
            .map_err(metal_error);
        self.prof
            .add(&self.prof.attention, &self.prof.attentions, started);
        out
    }
}

impl DeviceWeight for MetalWeight {
    fn linear(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input: usize,
        output: usize,
        d: DType,
    ) -> audio::Result<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "ACE-Step linear is F32".into(),
            });
        }
        let started = Instant::now();
        let out = self
            .0
            .linear_typed(x, bias, rows, input, output, Music3DType::F32)
            .map_err(metal_error);
        self.1.add(&self.1.linear, &self.1.linears, started);
        out
    }
    fn embedding(&self, _ids: &[i32], _width: usize) -> audio::Result<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "ACE-Step DiT has no device embeddings".into(),
        })
    }
    fn convolution(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        d: DType,
    ) -> audio::Result<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "ACE-Step VAE convolution is F32".into(),
            });
        }
        let started = Instant::now();
        let out = self
            .0
            .convolution_f32_grouped(
                x,
                bias,
                s.input_channels,
                s.output_channels,
                s.kernel,
                s.stride,
                s.padding,
                s.dilation,
                s.transpose,
                1,
            )
            .map_err(metal_error);
        self.1.add(&self.1.conv, &self.1.convs, started);
        out
    }
}

fn read_npy(path: &Path) -> (Vec<usize>, Vec<u8>, String) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len])
        .unwrap()
        .to_string();
    let start = header.find('(').unwrap() + 1;
    let end = header[start..].find(')').unwrap() + start;
    let shape = header[start..end]
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect();
    (shape, bytes[10 + header_len..].to_vec(), header)
}

fn f32s(dir: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, raw, header) = read_npy(&dir.join(format!("{name}.npy")));
    assert!(header.contains("'<f4'"), "{name}: not f32");
    let data = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, data)
}

fn i32s(dir: &Path, name: &str) -> Vec<i32> {
    let (_, raw, header) = read_npy(&dir.join(format!("{name}.npy")));
    assert!(header.contains("'<i4'"), "{name}: not i32");
    raw.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter().all(|v| v.is_finite()),
        "non-finite device output"
    );
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale
}

fn context(src: &[f32], chunk: &[f32], frames: usize, a: usize) -> Vec<f32> {
    let mut ctx = Vec::with_capacity(frames * 2 * a);
    for f in 0..frames {
        ctx.extend_from_slice(&src[f * a..(f + 1) * a]);
        ctx.extend_from_slice(&chunk[f * a..(f + 1) * a]);
    }
    ctx
}

/// Every stage on the device path against the reference arrays in
/// `dir`, each fed the reference output of the stage before it.
fn check(config: &AceConfig, file: &SafetensorsFile, dir: &Path) -> Vec<(&'static str, f32)> {
    let backend: Rc<dyn ComputeBackend> = Rc::new(metal());
    let l = Loader::with_backend(file, backend);
    let enc_model = ConditionEncoder::load(&l, config).unwrap();
    let dit = Dit::load(&l, config).unwrap();
    let tokenizer = AudioTokenizer::load(&l, config).unwrap();
    let detokenizer = AudioDetokenizer::load(&l, config).unwrap();
    let a = config.audio_acoustic_hidden_dim;
    let (ls, lyric) = f32s(dir, "lyric");
    let (rs, refer) = f32s(dir, "refer");
    let (ts, text) = f32s(dir, "text");
    let mut out = Vec::new();
    let (enc, _) = enc_model
        .forward(&Conditioning {
            text: &text,
            text_len: ts[0],
            lyric: &lyric,
            lyric_len: ls[0],
            refer: &refer,
            refer_len: rs[0],
        })
        .unwrap();
    let (es, want_enc) = f32s(dir, "encoder");
    out.push(("encoder", rel_err(&enc, &want_enc)));
    let embedded = dit.embed_condition(&want_enc, es[0]).unwrap();
    let (xs, x) = f32s(dir, "x");
    let frames = xs[0];
    let (_, ctx_in) = f32s(dir, "context");
    let v = dit
        .forward(&x, &ctx_in, frames, 0.75, &embedded, es[0])
        .unwrap();
    out.push(("velocity", rel_err(&v, &f32s(dir, "velocity").1)));
    let (_, src) = f32s(dir, "src");
    let (_, chunk) = f32s(dir, "chunk");
    let (_, noise) = f32s(dir, "noise");
    let schedule = turbo_schedule(3.0);
    let plain = context(&src, &chunk, frames, a);
    let latents = sample_turbo(
        &dit,
        &noise,
        frames,
        &schedule,
        StepCondition {
            context: &plain,
            enc: &embedded,
            enc_seq: es[0],
        },
        None,
        None,
        |_| true,
    )
    .unwrap()
    .unwrap();
    out.push(("latents", rel_err(&latents, &f32s(dir, "latents").1)));
    let (_, silence) = f32s(dir, "silence");
    let (tokens, indices) = tokenizer.tokenize(&src, frames, &silence).unwrap();
    out.push(("tokens", rel_err(&tokens, &f32s(dir, "tokens").1)));
    let want_idx = i32s(dir, "indices");
    let mismatched = indices
        .iter()
        .zip(&want_idx)
        .filter(|(a, b)| a != b)
        .count();
    out.push(("indices", mismatched as f32));
    let (_, want_hints) = f32s(dir, "hints");
    let hints = cover_hints(&tokenizer, &detokenizer, &src, frames, &silence).unwrap();
    out.push(("hints_chained", rel_err(&hints, &want_hints)));
    let (ns, want_enc_nc) = f32s(dir, "encoder_nc");
    let embedded_nc = dit.embed_condition(&want_enc_nc, ns[0]).unwrap();
    let cover_ctx = context(&want_hints, &chunk, frames, a);
    let fallback_ctx = context(&silence[..frames * a], &chunk, frames, a);
    let cover = sample_turbo(
        &dit,
        &noise,
        frames,
        &schedule,
        StepCondition {
            context: &cover_ctx,
            enc: &embedded,
            enc_seq: es[0],
        },
        Some((
            cover_switch_step(schedule.len(), 0.5).unwrap(),
            StepCondition {
                context: &fallback_ctx,
                enc: &embedded_nc,
                enc_seq: ns[0],
            },
        )),
        None,
        |_| true,
    )
    .unwrap()
    .unwrap();
    out.push((
        "latents_cover",
        rel_err(&cover, &f32s(dir, "latents_cover").1),
    ));
    out
}

fn report(tag: &str, errs: &[(&str, f32)], tol: f32, max_index_flips: usize) {
    let line: Vec<String> = errs.iter().map(|(n, e)| format!("{n} {e:.2e}")).collect();
    eprintln!("{tag}: {}", line.join(" "));
    for (stage, e) in errs {
        if *stage == "indices" {
            assert!(
                *e as usize <= max_index_flips,
                "{tag}: {e} FSQ indices differ"
            );
        } else {
            assert!(
                *e <= tol,
                "{tag} {stage}: relative error {e:.3e} > {tol:.1e}"
            );
        }
    }
}

fn config_at(path: &Path) -> AceConfig {
    let text = std::fs::read_to_string(path).unwrap();
    AceConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap()
}

fn tiny_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../audio/testdata/ace_step15_dit")
}

/// Measured on M4 Max: worst stage 2.1e-6 (velocity). The bound keeps
/// about 5x headroom. Scaling the device attention or linear outputs by
/// 1.001 fails it at every stage that reaches the device.
const TINY_TOL: f32 = 1e-5;

#[test]
fn tiny_metal_matches_reference() {
    if Music3Device::new().is_err() {
        eprintln!("no Metal device; skipping");
        return;
    }
    let dir = tiny_dir();
    let config = config_at(&dir.join("tiny_config.json"));
    let file = SafetensorsFile::open(&dir.join("tiny_weights.safetensors")).unwrap();
    report("tiny metal", &check(&config, &file, &dir), TINY_TOL, 0);
}

fn real_dirs() -> (PathBuf, PathBuf) {
    let (Ok(dit), Ok(reference)) = (
        std::env::var("TURBOSPARK_ACE_STEP15_DIT"),
        std::env::var("TURBOSPARK_ACE_STEP15_DIT_REF"),
    ) else {
        panic!("set TURBOSPARK_ACE_STEP15_DIT and TURBOSPARK_ACE_STEP15_DIT_REF");
    };
    (PathBuf::from(dit), PathBuf::from(reference))
}

#[test]
#[ignore = "needs TURBOSPARK_ACE_STEP15_DIT and TURBOSPARK_ACE_STEP15_DIT_REF"]
fn real_metal_matches_reference() {
    let (dit_dir, ref_dir) = real_dirs();
    let config = config_at(&dit_dir.join("config.json"));
    let file = SafetensorsFile::open(&dit_dir.join("model.safetensors")).unwrap();
    // Measured on M4 Max: worst stage 5.8e-6 (text2music latents).
    report("real metal", &check(&config, &file, &ref_dir), 2e-5, 0);
}

/// Timing only, no parity: one DiT forward and the 8-step loop at
/// `ACE_STEP15_SECONDS` of latents (default 30), plus device residency.
/// Not a frozen benchmark row (see docs/BENCHMARKING.md).
#[test]
#[ignore = "timing; needs TURBOSPARK_ACE_STEP15_DIT"]
fn real_metal_timing() {
    let (dit_dir, _) = real_dirs();
    let seconds: usize = std::env::var("ACE_STEP15_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let config = config_at(&dit_dir.join("config.json"));
    let file = SafetensorsFile::open(&dit_dir.join("model.safetensors")).unwrap();
    let backend = Rc::new(metal());
    let started = Instant::now();
    let l = Loader::with_backend(&file, backend.clone());
    let dit = Dit::load(&l, &config).unwrap();
    let load = started.elapsed();
    let resident = backend.device.resident_weight_bytes();
    let a = config.audio_acoustic_hidden_dim;
    let frames = seconds * 25;
    let enc_seq = 300;
    let mut state = 0x2545_F491u32;
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let x: Vec<f32> = (0..frames * a).map(|_| noise()).collect();
    let ctx: Vec<f32> = (0..frames * 2 * a).map(|_| noise()).collect();
    let enc: Vec<f32> = (0..enc_seq * config.hidden_size)
        .map(|_| noise() * 0.1)
        .collect();
    let enc = dit.embed_condition(&enc, enc_seq).unwrap();
    // Discard one warm-up forward (pipeline compilation, cold device).
    dit.forward(&x, &ctx, frames, 0.75, &enc, enc_seq).unwrap();
    backend.prof.reset();
    let t = Instant::now();
    dit.forward(&x, &ctx, frames, 0.75, &enc, enc_seq).unwrap();
    let one = t.elapsed();
    let p = &backend.prof;
    eprintln!(
        "profile, one forward {:.3} s: device linear {:.3} s ({} calls), device attention \
         {:.3} s ({} calls), host {:.3} s",
        one.as_secs_f64(),
        p.linear.get().as_secs_f64(),
        p.linears.get(),
        p.attention.get().as_secs_f64(),
        p.attentions.get(),
        (one - p.linear.get() - p.attention.get()).as_secs_f64()
    );
    // The same weights with every layer full: what host banded attention
    // costs (not a valid model, timing only).
    let mut unwindowed = config.clone();
    unwindowed.sliding_window = None;
    let full = Dit::load(&l, &unwindowed).unwrap();
    full.forward(&x, &ctx, frames, 0.75, &enc, enc_seq).unwrap();
    backend.prof.reset();
    let t = Instant::now();
    full.forward(&x, &ctx, frames, 0.75, &enc, enc_seq).unwrap();
    let nf = t.elapsed();
    eprintln!(
        "profile, no windows {:.3} s: device linear {:.3} s, device attention {:.3} s ({} calls), \
         host {:.3} s",
        nf.as_secs_f64(),
        p.linear.get().as_secs_f64(),
        p.attention.get().as_secs_f64(),
        p.attentions.get(),
        (nf - p.linear.get() - p.attention.get()).as_secs_f64()
    );
    drop(full);
    let t = Instant::now();
    let schedule = turbo_schedule(3.0);
    sample_turbo(
        &dit,
        &x,
        frames,
        &schedule,
        StepCondition {
            context: &ctx,
            enc: &enc,
            enc_seq,
        },
        None,
        None,
        |_| true,
    )
    .unwrap()
    .unwrap();
    let eight = t.elapsed();
    eprintln!(
        "timing: {seconds} s ({frames} frames): load {:.1} s, resident {:.2} GiB, \
         one forward {:.2} s, 8 steps {:.2} s",
        load.as_secs_f64(),
        resident as f64 / (1u64 << 30) as f64,
        one.as_secs_f64(),
        eight.as_secs_f64()
    );
}

fn vae_at(dir: &Path, weights: &str, backend: Option<&MetalBackend>) -> OobleckVae {
    let text = std::fs::read_to_string(dir.join(if weights.starts_with("tiny") {
        "tiny_config.json"
    } else {
        "config.json"
    }))
    .unwrap();
    let config = OobleckConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap();
    let file = SafetensorsFile::open(&dir.join(weights)).unwrap();
    let mut vae = OobleckVae::load(config, &file).unwrap();
    if let Some(b) = backend {
        vae.attach_backend(b).unwrap();
    }
    vae
}

/// Encode, sample with the recorded noise, decode; per-stage errors.
fn vae_check(vae: &OobleckVae, dir: &Path, tag: &str) -> Vec<(&'static str, f32)> {
    let (ashape, audio) = f32s(dir, &format!("{tag}_audio"));
    let (lshape, mean) = f32s(dir, &format!("{tag}_mean"));
    let (_, std) = f32s(dir, &format!("{tag}_std"));
    let (_, noise) = f32s(dir, &format!("{tag}_noise"));
    let (_, sample) = f32s(dir, &format!("{tag}_sample"));
    let (_, decoded) = f32s(dir, &format!("{tag}_decoded"));
    let m = vae.encode(&audio, ashape[1]).unwrap();
    let (wave, _) = vae.decode(&sample, lshape[1]).unwrap();
    vec![
        ("mean", rel_err(&m.mean, &mean)),
        ("std", rel_err(&m.std, &std)),
        ("sample", rel_err(&m.sample(&noise).unwrap(), &sample)),
        ("decode", rel_err(&wave, &decoded)),
    ]
}

fn vae_tiny_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../audio/testdata/ace_step15_vae")
}

#[test]
fn tiny_metal_vae_matches_reference() {
    if Music3Device::new().is_err() {
        eprintln!("no Metal device; skipping");
        return;
    }
    let backend = metal();
    let dir = vae_tiny_dir();
    let vae = vae_at(&dir, "tiny_weights.safetensors", Some(&backend));
    for tag in ["a", "b"] {
        report(
            &format!("tiny metal vae {tag}"),
            &vae_check(&vae, &dir, tag),
            TINY_TOL,
            0,
        );
    }
    assert!(
        backend.prof.convs.get() > 0,
        "no convolution reached the device"
    );
}

fn vae_dirs() -> (PathBuf, PathBuf) {
    let (Ok(vae), Ok(reference)) = (
        std::env::var("TURBOSPARK_ACE_STEP15_VAE"),
        std::env::var("TURBOSPARK_ACE_STEP15_VAE_REF"),
    ) else {
        panic!("set TURBOSPARK_ACE_STEP15_VAE and TURBOSPARK_ACE_STEP15_VAE_REF");
    };
    (PathBuf::from(vae), PathBuf::from(reference))
}

#[test]
#[ignore = "needs TURBOSPARK_ACE_STEP15_VAE and TURBOSPARK_ACE_STEP15_VAE_REF"]
fn real_metal_vae_matches_reference() {
    let (vae_dir, ref_dir) = vae_dirs();
    let backend = metal();
    let vae = vae_at(
        &vae_dir,
        "diffusion_pytorch_model.safetensors",
        Some(&backend),
    );
    report(
        "real metal vae",
        &vae_check(&vae, &ref_dir, "real"),
        2e-5,
        0,
    );
}

/// Timing, plus tiled against untiled on the device. Decode and encode of
/// `ACE_STEP15_SECONDS` (default 30). Not a frozen benchmark row.
#[test]
#[ignore = "timing; needs TURBOSPARK_ACE_STEP15_VAE"]
fn real_metal_vae_timing() {
    let (vae_dir, _) = vae_dirs();
    let seconds: usize = std::env::var("ACE_STEP15_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let backend = metal();
    let vae = vae_at(
        &vae_dir,
        "diffusion_pytorch_model.safetensors",
        Some(&backend),
    );
    let frames = seconds * 25;
    let latents: Vec<f32> = (0..64 * frames)
        .map(|i| ((i as f32) * 0.37).sin())
        .collect();
    // Warm-up on a short window (pipeline compilation).
    vae.decode(&latents[..64 * 8], 8).ok();
    let decode_untiled = vae.decode(&latents[..], frames);
    backend.prof.reset();
    let t = Instant::now();
    let (tiled, samples) = vae.decode_tiled(&latents, frames, 512, 64).unwrap();
    let tiled_time = t.elapsed();
    let conv = backend.prof.conv.get();
    let convs = backend.prof.convs.get();
    let t = Instant::now();
    let m = vae.encode(&tiled, samples).unwrap();
    let encode = t.elapsed();
    assert_eq!(m.frames, frames);
    let seam = match &decode_untiled {
        Ok((untiled, _)) => format!("{:.2e}", rel_err(&tiled, untiled)),
        Err(e) => format!("untiled refused: {e}"),
    };
    eprintln!(
        "vae metal: {seconds} s: tiled decode (512/64) {:.2} s (device conv {:.2} s, {convs} \
         calls, host {:.2} s), encode {:.2} s, tiled vs untiled {seam}",
        tiled_time.as_secs_f64(),
        conv.as_secs_f64(),
        (tiled_time - conv).as_secs_f64(),
        encode.as_secs_f64(),
    );
}

/// End to end on the device path: the same request and replayed noise as
/// `tools/gen_ace_step15_e2e_reference.py`, decoded to a WAV beside the
/// reference. Needs TURBOSPARK_ACE_STEP15_ROOT (silence latent converted)
/// and TURBOSPARK_ACE_STEP15_E2E_REF.
fn metal_e2e(
    refdir_var: &str,
    dcw: Option<audio::music::ace_step15::Dcw>,
    wav: &str,
    audio_tol: f32,
) {
    let root = PathBuf::from(std::env::var("TURBOSPARK_ACE_STEP15_ROOT").unwrap());
    let refdir = PathBuf::from(std::env::var(refdir_var).unwrap());
    let req: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refdir.join("request.json")).unwrap())
            .unwrap();
    let (_, noise) = f32s(&refdir, "noise");
    let backend: Rc<dyn ComputeBackend> = Rc::new(metal());
    let started = Instant::now();
    let mut model = AceStep15::load(&root, Some(backend)).unwrap();
    let load = started.elapsed();
    let request = Text2Music {
        caption: req["caption"].as_str().unwrap(),
        lyrics: req["lyrics"].as_str().unwrap(),
        language: req["language"].as_str().unwrap(),
        bpm: req["bpm"].as_u64().map(|b| b as u32),
        key_scale: req["key_scale"].as_str().unwrap(),
        time_signature: req["time_signature"].as_str().unwrap(),
        duration_seconds: req["duration"].as_f64().unwrap(),
        shift: req["shift"].as_f64().unwrap() as f32,
        seed: 0,
        noise: Some(&noise),
        dcw,
    };
    assert_eq!(
        req["dcw"].as_bool().unwrap_or(false),
        dcw.is_some(),
        "reference DCW setting"
    );
    // A first run compiles pipelines on a cold device; time the second.
    model.text2music(&request, |_| true).unwrap().unwrap();
    let t = Instant::now();
    let out = model.text2music(&request, |_| true).unwrap().unwrap();
    let total = t.elapsed();
    let (le, ae) = (
        rel_err(&out.latents, &f32s(&refdir, "latents").1),
        rel_err(&out.audio, &f32s(&refdir, "audio").1),
    );
    eprintln!(
        "e2e metal ({wav}): latents {le:.2e} audio {ae:.2e}; load {:.1} s, total {:.1} s (text {:.2}, \
         condition {:.2}, sampling {:.2}, decode {:.2})",
        load.as_secs_f64(),
        total.as_secs_f64(),
        out.timings.text.as_secs_f64(),
        out.timings.condition.as_secs_f64(),
        out.timings.sampling.as_secs_f64(),
        out.timings.decode.as_secs_f64()
    );
    let mut interleaved = Vec::with_capacity(out.audio.len());
    for i in 0..out.samples {
        interleaved.push(out.audio[i]);
        interleaved.push(out.audio[out.samples + i]);
    }
    let wave = audio::waveform::Waveform::new(48_000, 2, interleaved).unwrap();
    std::fs::write(refdir.join(wav), audio::wav::write_wav_f32(&wave)).unwrap();
    assert!(le <= 4e-5, "latents: relative error {le:.3e}");
    assert!(
        ae <= audio_tol,
        "audio: relative error {ae:.3e} > {audio_tol:.1e}"
    );
}

#[test]
#[ignore = "needs TURBOSPARK_ACE_STEP15_ROOT and TURBOSPARK_ACE_STEP15_E2E_REF"]
fn real_metal_text2music_matches_reference() {
    // Audio measured 6.7e-6.
    metal_e2e(
        "TURBOSPARK_ACE_STEP15_E2E_REF",
        None,
        "rust_metal.wav",
        4e-5,
    );
}

#[test]
#[ignore = "needs TURBOSPARK_ACE_STEP15_ROOT and TURBOSPARK_ACE_STEP15_E2E_DCW_REF"]
fn real_metal_text2music_dcw_matches_reference() {
    // Audio measured 9.6e-5; see the CPU DCW gate for why the decoder
    // amplifies the latent gap tenfold at this operating point.
    metal_e2e(
        "TURBOSPARK_ACE_STEP15_E2E_DCW_REF",
        Some(audio::music::ace_step15::Dcw::TURBO),
        "rust_metal_dcw.wav",
        3e-4,
    );
}
