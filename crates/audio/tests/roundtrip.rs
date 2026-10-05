//! End-to-end checks through real files: write WAV with this crate, read it
//! back through symphonia, and convert, trim, probe and peak it. Fixtures are
//! generated per test in a temp directory; nothing binary is checked in.

use std::f32::consts::PI;
use std::path::PathBuf;

use turbospark_audio::wav::write_mono;
use turbospark_audio::{
    convert, load_speech_samples, peaks, probe, AudioError, ConvertOptions, DecodedStream,
    WavSampleFormat, WavWriter,
};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-audio-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sine(rate: u32, freq: f32, seconds: f32, amplitude: f32) -> Vec<f32> {
    let n = (rate as f32 * seconds) as usize;
    (0..n)
        .map(|i| amplitude * (2.0 * PI * freq * i as f32 / rate as f32).sin())
        .collect()
}

fn decode_all(path: &std::path::Path) -> (u32, Vec<Vec<f32>>) {
    let mut stream = DecodedStream::open(path).unwrap();
    let mut planes: Vec<Vec<f32>> = vec![Vec::new(); stream.channels()];
    while let Some(chunk) = stream.next_chunk().unwrap() {
        for (plane, channel) in planes.iter_mut().zip(chunk) {
            plane.extend(channel);
        }
    }
    (stream.sample_rate(), planes)
}

#[test]
fn int16_wav_round_trips_through_the_decoder() {
    let dir = TempDir::new("int16");
    let path = dir.path("tone.wav");
    let tone = sine(22_050, 440.0, 0.5, 0.5);
    write_mono(&path, &tone, 22_050, WavSampleFormat::Int16).unwrap();

    let (rate, planes) = decode_all(&path);
    assert_eq!(rate, 22_050);
    assert_eq!(planes.len(), 1);
    assert_eq!(planes[0].len(), tone.len());
    for (a, b) in planes[0].iter().zip(&tone) {
        assert!((a - b).abs() < 1.0 / 16_000.0, "{a} vs {b}");
    }
}

#[test]
fn float32_wav_round_trips_bit_exact() {
    let dir = TempDir::new("float32");
    let path = dir.path("tone.wav");
    let tone = sine(48_000, 1_000.0, 0.25, 0.8);
    write_mono(&path, &tone, 48_000, WavSampleFormat::Float32).unwrap();
    let (_, planes) = decode_all(&path);
    assert_eq!(planes[0], tone);
}

#[test]
fn probe_reports_rate_channels_and_duration() {
    let dir = TempDir::new("probe");
    let path = dir.path("stereo.wav");
    let mut writer = WavWriter::create(&path, 44_100, 2, WavSampleFormat::Int16).unwrap();
    let left = sine(44_100, 300.0, 2.0, 0.3);
    let right = sine(44_100, 600.0, 2.0, 0.3);
    writer.write_planar(&[left, right]).unwrap();
    writer.finish().unwrap();

    let report = probe(&path).unwrap();
    assert_eq!(report.sample_rate, 44_100);
    assert_eq!(report.channels, 2);
    assert_eq!(report.frames, 88_200);
    assert!((report.duration_seconds - 2.0).abs() < 1e-9);
    assert_eq!(report.codec, "pcm_s16le");
}

#[test]
fn speech_normalization_yields_16k_mono_of_the_same_duration() {
    let dir = TempDir::new("speech");
    let source = dir.path("source.wav");
    let target = dir.path("speech.wav");
    let mut writer = WavWriter::create(&source, 48_000, 2, WavSampleFormat::Float32).unwrap();
    let tone = sine(48_000, 440.0, 1.5, 0.5);
    writer.write_planar(&[tone.clone(), tone]).unwrap();
    writer.finish().unwrap();

    let report = convert(&source, &target, &ConvertOptions::speech()).unwrap();
    assert_eq!(report.sample_rate, 16_000);
    assert_eq!(report.channels, 1);
    assert_eq!(report.frames, 24_000);

    let (rate, planes) = decode_all(&target);
    assert_eq!(rate, 16_000);
    assert_eq!(planes.len(), 1);
    assert_eq!(planes[0].len(), 24_000);
    // Identical channels downmix to the same tone; interior amplitude holds.
    let peak = planes[0][1_000..23_000]
        .iter()
        .fold(0.0f32, |m, s| m.max(s.abs()));
    assert!((peak - 0.5).abs() < 0.01, "peak {peak}");
}

#[test]
fn trim_cuts_on_the_selected_source_frames() {
    let dir = TempDir::new("trim");
    let source = dir.path("source.wav");
    let target = dir.path("cut.wav");
    // One second of silence, then one second of tone.
    let mut samples = vec![0.0f32; 8_000];
    samples.extend(sine(8_000, 200.0, 1.0, 0.9));
    write_mono(&source, &samples, 8_000, WavSampleFormat::Float32).unwrap();

    let options = ConvertOptions {
        start_seconds: Some(1.0),
        end_seconds: Some(1.5),
        sample_format: WavSampleFormat::Float32,
        ..ConvertOptions::default()
    };
    let report = convert(&source, &target, &options).unwrap();
    assert_eq!(report.frames, 4_000);
    let (_, planes) = decode_all(&target);
    assert_eq!(planes[0], samples[8_000..12_000].to_vec());
}

#[test]
fn a_range_past_the_end_is_empty_and_leaves_no_file() {
    let dir = TempDir::new("empty");
    let source = dir.path("source.wav");
    let target = dir.path("cut.wav");
    write_mono(
        &source,
        &sine(8_000, 200.0, 1.0, 0.5),
        8_000,
        WavSampleFormat::Int16,
    )
    .unwrap();
    let options = ConvertOptions {
        start_seconds: Some(5.0),
        ..ConvertOptions::default()
    };
    assert!(matches!(
        convert(&source, &target, &options),
        Err(AudioError::EmptyRange)
    ));
    assert!(!target.exists());
}

#[test]
fn invalid_options_are_refused_before_writing() {
    let dir = TempDir::new("invalid");
    let source = dir.path("source.wav");
    write_mono(
        &source,
        &sine(8_000, 200.0, 0.1, 0.5),
        8_000,
        WavSampleFormat::Int16,
    )
    .unwrap();
    for options in [
        ConvertOptions {
            sample_rate: Some(100),
            ..ConvertOptions::default()
        },
        ConvertOptions {
            channels: Some(2),
            ..ConvertOptions::default()
        },
        ConvertOptions {
            start_seconds: Some(1.0),
            end_seconds: Some(0.5),
            ..ConvertOptions::default()
        },
    ] {
        assert!(
            convert(&source, &dir.path("out.wav"), &options).is_err(),
            "{options:?}"
        );
    }
}

#[test]
fn peaks_locate_the_loud_half() {
    let dir = TempDir::new("peaks");
    let path = dir.path("ramp.wav");
    let mut samples = sine(16_000, 300.0, 1.0, 0.05);
    samples.extend(sine(16_000, 300.0, 1.0, 0.8));
    write_mono(&path, &samples, 16_000, WavSampleFormat::Float32).unwrap();

    let report = peaks(&path, 20).unwrap();
    assert_eq!(report.peaks.len(), 20);
    assert!((report.duration_seconds - 2.0).abs() < 1e-9);
    let quiet = report.peaks[..10].iter().fold(0.0f32, |m, v| m.max(*v));
    let loud = report.peaks[10..].iter().fold(1.0f32, |m, v| m.min(*v));
    assert!(quiet < 0.1, "quiet half {quiet}");
    assert!(loud > 0.9, "loud half {loud}");
}

#[test]
fn peaks_refuse_a_zero_or_huge_bucket_count() {
    let dir = TempDir::new("buckets");
    let path = dir.path("tone.wav");
    write_mono(
        &path,
        &sine(8_000, 200.0, 0.1, 0.5),
        8_000,
        WavSampleFormat::Int16,
    )
    .unwrap();
    assert!(peaks(&path, 0).is_err());
    assert!(peaks(&path, 1_000_000).is_err());
}

#[test]
fn a_refused_extension_names_itself() {
    let dir = TempDir::new("refused");
    let path = dir.path("voice.opus");
    std::fs::write(&path, b"OggS not really").unwrap();
    match DecodedStream::open(&path) {
        Err(AudioError::Unsupported(what)) => assert_eq!(what, ".opus"),
        other => panic!("expected Unsupported, got {:?}", other.err()),
    }
}

#[test]
fn garbage_with_a_supported_extension_is_a_decode_error_not_a_panic() {
    let dir = TempDir::new("garbage");
    let path = dir.path("broken.wav");
    std::fs::write(&path, b"this is not a riff file at all").unwrap();
    assert!(DecodedStream::open(&path).is_err());
    assert!(probe(&path).is_err());
}

#[test]
fn speech_samples_load_at_16k_mono() {
    let dir = TempDir::new("load");
    let path = dir.path("source.wav");
    let mut writer = WavWriter::create(&path, 44_100, 2, WavSampleFormat::Int16).unwrap();
    let tone = sine(44_100, 500.0, 1.0, 0.4);
    writer.write_planar(&[tone.clone(), tone]).unwrap();
    writer.finish().unwrap();
    let samples = load_speech_samples(&path).unwrap();
    assert_eq!(samples.len(), 16_000);
}
