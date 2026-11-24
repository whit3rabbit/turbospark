//! Inworld Voice Isolator on the native audio session.
//!
//! With [`Device::Metal`] the network's projections (every linear layer, the
//! bulk of its arithmetic) and its non-causal attention (the offline time
//! attention and every band attention) and the Mamba selective scans run on
//! `gpu::Music3Device` in F32; norms, the causal stream attention, small
//! per-band projections and the STFT stay on the CPU. With [`Device::Cpu`]
//! it is the portable f32 port throughout.

use std::path::Path;
use std::rc::Rc;

use audio::backend::{
    AttentionShape, ComputeBackend, ConvShape, DType, DeviceWeight, ScanInputs, ScanShape,
    WeightData, WeightEncoding,
};
use audio::sts::VoiceIsolator;
use audio::{Result as AudioResult, SpeechError};
use gpu::{Music3Device, Music3Encoding, Music3Weight};

/// Where the network's projections and non-causal attention run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
}

/// The device the session opens on: Metal unless
/// `TURBOSPARK_VOICE_ISOLATOR_DEVICE=cpu`.
pub fn default_device() -> Device {
    match std::env::var("TURBOSPARK_VOICE_ISOLATOR_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ => Device::Metal,
    }
}

fn metal_error(e: gpu::GpuError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("voice isolator Metal: {e}"),
    }
}

/// F32 weights, tiled float GEMM and batched non-causal attention.
pub(crate) struct MetalBackend {
    device: Music3Device,
}

struct MetalWeight {
    inner: Music3Weight,
}

impl ComputeBackend for MetalBackend {
    fn load_weight(&self, data: WeightData<'_>) -> AudioResult<Rc<dyn DeviceWeight>> {
        if data.encoding != WeightEncoding::F32 || data.dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "the voice isolator loads F32 weights only".into(),
            });
        }
        let inner = self
            .device
            .load_weight(data.shape, data.bytes, Music3Encoding::F32, &[], &[], &[])
            .map_err(metal_error)?;
        Ok(Rc::new(MetalWeight { inner }))
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: AttentionShape,
        d: DType,
    ) -> AudioResult<Vec<f32>> {
        if d != DType::F32 || s.causal || s.kv_time_major || s.offset != 0 || s.queries != s.keys {
            return Err(SpeechError::Unsupported {
                why: "voice isolator device attention is non-causal F32 self attention".into(),
            });
        }
        self.device
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
                gpu::Music3DType::F32,
            )
            .map_err(metal_error)
    }

    fn selective_scan(&self, i: ScanInputs<'_>, s: ScanShape) -> AudioResult<(Vec<f32>, Vec<f32>)> {
        self.device
            .selective_scan_f32(
                i.u,
                i.delta,
                i.a,
                i.bc,
                i.d,
                i.z,
                i.carried,
                s.seqs,
                s.len,
                s.channels,
                s.states,
                (s.bc_stride, s.b_offset, s.c_offset),
                (s.z_stride, s.z_offset),
            )
            .map_err(metal_error)
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
    ) -> AudioResult<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "voice isolator linears stay F32".into(),
            });
        }
        self.inner
            .linear_f32(x, bias, rows, input, output)
            .map_err(metal_error)
    }

    fn embedding(&self, _ids: &[i32], _width: usize) -> AudioResult<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "the voice isolator has no device embeddings".into(),
        })
    }

    fn convolution(
        &self,
        _x: &[f32],
        _bias: Option<&[f32]>,
        _s: ConvShape,
        _d: DType,
    ) -> AudioResult<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "the voice isolator has no device convolutions".into(),
        })
    }
}

/// Opens a voice-isolator install (`denoiser.safetensors` or
/// `denoiser_realtime.safetensors`) on `device`.
pub fn open(path: &Path, device: Device) -> Result<VoiceIsolator, String> {
    let mut model = VoiceIsolator::open(path).map_err(|e| e.to_string())?;
    if device == Device::Metal {
        let dev = Music3Device::new().map_err(|e| e.to_string())?;
        model.set_backend(Some(Rc::new(MetalBackend { device: dev })));
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(case: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../audio/testdata/voice_isolator")
            .join(case)
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn tone(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (i as f32 * 0.07).sin() * 0.3 + ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.2
            })
            .collect()
    }

    /// The device path against the host port on the tiny fixtures: the
    /// offline render (device attention and linears) and the stream (device
    /// linears, host causal attention), with no call falling back.
    #[test]
    fn metal_matches_cpu_on_the_fixtures() {
        let _serial = crate::heavy::serial_test_lock();
        for case in ["offline", "causal"] {
            let cpu = open(&fixture(case), Device::Cpu).unwrap();
            let metal = open(&fixture(case), Device::Metal).unwrap();
            assert!(metal.on_device() && !cpu.on_device());
            let audio = tone(20_800);
            let diff = max_diff(
                &metal.isolate(&audio).unwrap(),
                &cpu.isolate(&audio).unwrap(),
            );
            eprintln!("{case} metal vs cpu isolate {diff}");
            assert!(diff < 1e-4, "{case} {diff}");
            assert_eq!(
                metal.device_fallbacks(),
                0,
                "{case}: a device call fell back to the host"
            );
        }
        let cpu = open(&fixture("causal"), Device::Cpu).unwrap();
        let metal = open(&fixture("causal"), Device::Metal).unwrap();
        let audio = tone(4_800);
        let run = |m: &VoiceIsolator| {
            let mut s = audio::sts::VoiceIsolatorStream::new(m, 4, None).unwrap();
            let mut out = Vec::new();
            for piece in audio.chunks(320) {
                out.extend(s.push(m, piece).unwrap());
            }
            out.extend(s.flush(m).unwrap());
            out
        };
        let diff = max_diff(&run(&metal), &run(&cpu));
        eprintln!("causal metal vs cpu stream {diff}");
        assert!(diff < 1e-4, "stream {diff}");
        assert_eq!(metal.device_fallbacks(), 0);
    }

    /// Real checkpoints on Metal against the torch reference, with timing.
    /// Needs `TURBOSPARK_VOICE_ISOLATOR_MODEL` (see the audio crate's gate).
    #[test]
    #[ignore = "requires TURBOSPARK_VOICE_ISOLATOR_MODEL"]
    fn real_checkpoint_on_metal() {
        let dir =
            std::path::PathBuf::from(std::env::var("TURBOSPARK_VOICE_ISOLATOR_MODEL").unwrap());
        let read = |name: &str| -> Vec<f32> {
            let bytes = std::fs::read(dir.join(name)).unwrap();
            let header = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            bytes[10 + header..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let _serial = crate::heavy::serial_test_lock();
        let clip = read("real_clip.npy");
        for (file, reference) in [
            ("denoiser.safetensors", "real_voice.npy"),
            ("denoiser_realtime.safetensors", "real_voice_realtime.npy"),
        ] {
            let only = tempdir_with(&dir.join(file));
            let model = open(&only, Device::Metal).unwrap();
            // Primer pass: the first pass on a cold device runs slow.
            model.forward(&clip[..48_000]).unwrap();
            let started = std::time::Instant::now();
            let out = model.isolate(&clip).unwrap();
            let seconds = started.elapsed().as_secs_f64();
            let diff = max_diff(&out, &read(reference));
            eprintln!(
                "{file} metal: diff {diff}, {seconds:.2}s for {:.1}s (RTF {:.3}), fallbacks {}, linear/attention/scan seconds incl. primer {:?}",
                clip.len() as f64 / 48_000.0,
                seconds / (clip.len() as f64 / 48_000.0),
                model.device_fallbacks(),
                model.op_seconds()
            );
            assert!(diff < 2e-3, "{file} {diff}");
            assert_eq!(model.device_fallbacks(), 0);
            if file.contains("realtime") {
                // The live shape: 20 ms pushes (960 samples) through the
                // stream, timing every push against its 20 ms budget.
                let mut s = audio::sts::VoiceIsolatorStream::new(&model, 4, None).unwrap();
                let (mut worst, mut late, started) = (0.0f64, 0usize, std::time::Instant::now());
                let mut out = Vec::new();
                for piece in clip.chunks(960) {
                    let t = std::time::Instant::now();
                    out.extend(s.push(&model, piece).unwrap());
                    let took = t.elapsed().as_secs_f64();
                    worst = worst.max(took);
                    late += usize::from(took > 0.020);
                }
                out.extend(s.flush(&model).unwrap());
                let total = started.elapsed().as_secs_f64();
                eprintln!(
                    "{file} stream: {total:.2}s for {:.1}s (RTF {:.3}), worst push {:.1} ms, {late} of {} pushes over 20 ms",
                    clip.len() as f64 / 48_000.0,
                    total / (clip.len() as f64 / 48_000.0),
                    worst * 1e3,
                    clip.len().div_ceil(960)
                );
                assert_eq!(out.len(), clip.len());
            }
            std::fs::remove_dir_all(&only).unwrap();
        }
    }

    /// The realtime stream alone, for profiling: 20 ms pushes on `device`
    /// (`TURBOSPARK_VOICE_ISOLATOR_DEVICE`), timing reported.
    #[test]
    #[ignore = "requires TURBOSPARK_VOICE_ISOLATOR_MODEL"]
    fn real_stream_timing() {
        let dir =
            std::path::PathBuf::from(std::env::var("TURBOSPARK_VOICE_ISOLATOR_MODEL").unwrap());
        let _serial = crate::heavy::serial_test_lock();
        let only = tempdir_with(&dir.join("denoiser_realtime.safetensors"));
        let model = open(&only, default_device()).unwrap();
        let clip: Vec<f32> = (0..10 * 48_000)
            .map(|i| ((i as f32) * 0.013).sin() * 0.1)
            .collect();
        let mut s = audio::sts::VoiceIsolatorStream::new(&model, 4, None).unwrap();
        let started = std::time::Instant::now();
        for piece in clip.chunks(960) {
            s.push(&model, piece).unwrap();
        }
        s.flush(&model).unwrap();
        eprintln!(
            "stream {:.2}s for 10 s, ops {:?}",
            started.elapsed().as_secs_f64(),
            model.op_seconds()
        );
        std::fs::remove_dir_all(&only).unwrap();
    }

    /// Kernel choice for the dominant shape (one 8 s window's feed-forward
    /// input projection): F32 weights through `linear_f32` against F16 and
    /// BF16 weights through the tiled typed path. Measurement only.
    #[test]
    #[ignore = "Metal microbenchmark"]
    fn linear_kernel_microbenchmark() {
        let _serial = crate::heavy::serial_test_lock();
        let dev = Music3Device::new().unwrap();
        let (rows, k, n) = (36_846usize, 512usize, 2048usize);
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 40503usize) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let f32_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_bytes: Vec<u8> = w
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let bf16_bytes: Vec<u8> = w
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let w32 = dev
            .load_weight(&[n, k], &f32_bytes, Music3Encoding::F32, &[], &[], &[])
            .unwrap();
        let w16 = dev
            .load_weight(&[n, k], &f16_bytes, Music3Encoding::F16, &[], &[], &[])
            .unwrap()
            .tiled_for_f32();
        let wb16 = dev
            .load_weight(&[n, k], &bf16_bytes, Music3Encoding::Bf16, &[], &[], &[])
            .unwrap()
            .tiled_for_f32();
        let flop = 2.0 * (rows * k * n) as f64;
        let time = |label: &str, f: &dyn Fn() -> Vec<f32>| {
            f();
            let t = std::time::Instant::now();
            let out = f();
            let s = t.elapsed().as_secs_f64();
            eprintln!("{label}: {:.1} ms, {:.2} TFLOPS", s * 1e3, flop / s / 1e12);
            out
        };
        let a = time("f32 linear_f32", &|| {
            w32.linear_f32(&x, None, rows, k, n).unwrap()
        });
        let b = time("f16 tiled f32 out", &|| {
            w16.linear_typed(&x, None, rows, k, n, gpu::Music3DType::F32)
                .unwrap()
        });
        let c = time("bf16 tiled f32 out", &|| {
            wb16.linear_typed(&x, None, rows, k, n, gpu::Music3DType::F32)
                .unwrap()
        });
        let diff = |p: &[f32], q: &[f32]| {
            p.iter()
                .zip(q)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        eprintln!(
            "f16 vs f32 max diff {}, bf16 vs f32 {}",
            diff(&a, &b),
            diff(&a, &c)
        );
    }

    /// A directory holding only `file` (symlinked), so `open` picks it.
    fn tempdir_with(file: &Path) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ts-vi-metal-{}-{}",
            std::process::id(),
            file.file_name().unwrap().to_string_lossy()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::os::unix::fs::symlink(file, dir.join(file.file_name().unwrap())).unwrap();
        dir
    }
}
