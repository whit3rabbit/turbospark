#![cfg(target_os = "macos")]
//! The Fish S1 DAC with its heavy operations on Metal against the f32 host
//! path: the tiny fixture (`crates/audio/testdata/fish_s1_dac`) always, and
//! the real codec through the runner's worker when
//! `TURBOSPARK_FISH_DRAMA3_MODEL` names a Drama 3 install.

use std::path::PathBuf;
use std::time::Instant;

use audio::codec::fish_s1_dac::{FishS1Dac, FishS1DacConfig};
use model_io::safetensors::SafetensorsFile;
use turbospark_runtime::fish_drama3_codec::{metal_dac_exec, CodecWorker};

fn tiny() -> FishS1Dac {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../audio/testdata/fish_s1_dac");
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tiny_config.json")).unwrap())
            .unwrap();
    let config = FishS1DacConfig::from_json(&config).expect("config");
    let file = SafetensorsFile::open(&dir.join("tiny_weights.safetensors")).expect("weights");
    FishS1Dac::load(config, &file).expect("tiny codec")
}

/// Valid codes `[books][frames]` from a fixed xorshift stream.
fn codes(codec: &FishS1Dac, frames: usize, seed: u32) -> Vec<Vec<i32>> {
    let q = &codec.config.quantizer;
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..codec.config.total_books())
        .map(|b| {
            let size = if b == 0 {
                q.semantic_codebook_size
            } else {
                q.codebook_size
            } as u32;
            (0..frames)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (state % size) as i32
                })
                .collect()
        })
        .collect()
}

/// Worst absolute error relative to the reference's largest magnitude.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    assert!(got.iter().all(|v| v.is_finite()));
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale
}

fn pieces(codes: &[Vec<i32>], first: usize, every: usize) -> Vec<Vec<Vec<i32>>> {
    let frames = codes[0].len();
    let mut out = Vec::new();
    let mut at = 0;
    let mut n = first;
    while at < frames {
        let end = (at + n).min(frames);
        out.push(codes.iter().map(|b| b[at..end].to_vec()).collect());
        at = end;
        n = every;
    }
    out
}

#[test]
fn metal_codec_matches_the_host_on_the_tiny_codec() {
    let codec = tiny();
    let exec = metal_dac_exec().expect("metal exec");
    let codes = codes(&codec, 20, 11);
    let host = codec.decode(&codes).unwrap();

    let whole = rel_err(&codec.decode_on(&exec, &codes).unwrap(), &host);
    let mut stream = codec.stream_on(&exec);
    let mut streamed = Vec::new();
    for piece in pieces(&codes, 4, 7) {
        streamed.extend(stream.push(&piece).unwrap());
    }
    let streamed = rel_err(&streamed, &host);
    eprintln!("decode vs host {whole:.2e}, streamed {streamed:.2e}");
    assert!(whole < 1e-4, "decode relative error {whole:.2e}");
    assert!(streamed < 1e-4, "streamed relative error {streamed:.2e}");

    let samples: Vec<f32> = (0..codec.config.frame_length() * 6)
        .map(|i| (i as f32 * 0.011).sin() * 0.4 + (i as f32 * 0.003).cos() * 0.2)
        .collect();
    let (want, want_len) = codec.encode(&samples).unwrap();
    let (got, got_len) = codec.encode_on(&exec, &samples).unwrap();
    assert_eq!(got_len, want_len);
    assert_eq!(got, want, "encoded codes on metal");
}

/// Opt-in: the real codec through [`CodecWorker`], streamed in the runner's
/// pieces, against the host decode, with both timed.
#[test]
fn real_codec_worker_matches_the_host() {
    let Ok(install) = std::env::var("TURBOSPARK_FISH_DRAMA3_MODEL") else {
        eprintln!("skipping: TURBOSPARK_FISH_DRAMA3_MODEL not set");
        return;
    };
    let weights = PathBuf::from(install).join("codec.safetensors");
    let host_codec = FishS1Dac::open_weights(&weights).expect("host codec");
    let codes = codes(&host_codec, 60, 5);
    let per_frame = |started: Instant| started.elapsed().as_secs_f64() * 1e3 / 60.0;
    let started = Instant::now();
    let host = host_codec.decode(&codes).unwrap();
    let host_ms = per_frame(started);
    let started = Instant::now();
    let mut stream = host_codec.stream();
    for piece in pieces(&codes, 4, 7) {
        stream.push(&piece).unwrap();
    }
    let host_stream_ms = per_frame(started);
    {
        let exec = metal_dac_exec().expect("metal exec");
        host_codec.decode_on(&exec, &codes).unwrap(); // uploads and warms up
        let started = Instant::now();
        let whole = host_codec.decode_on(&exec, &codes).unwrap();
        eprintln!(
            "whole decode: host {host_ms:.1} ms/frame, metal {:.1} ms/frame ({:.2e}); \
             host streamed {host_stream_ms:.1} ms/frame",
            per_frame(started),
            rel_err(&whole, &host)
        );
    }
    drop(host_codec);

    let worker = CodecWorker::open(&weights, true).expect("worker");
    for round in 0..2 {
        let (frames, pcm) = worker.stream(4, 7).unwrap();
        let started = Instant::now();
        for f in 0..codes[0].len() {
            frames
                .send(Some(codes.iter().map(|b| b[f]).collect()))
                .unwrap();
        }
        frames.send(None).unwrap();
        let mut streamed = Vec::new();
        for piece in pcm {
            streamed.extend(piece.unwrap());
        }
        let metal_ms = per_frame(started);
        let err = rel_err(&streamed, &host);
        eprintln!(
            "round {round}: worker streamed {metal_ms:.1} ms/frame, relative error {err:.2e}"
        );
        assert!(err < 1e-3, "relative error {err:.2e}");
    }
}
