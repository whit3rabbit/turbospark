//! Sortformer streaming demo: `cargo run -p turbospark-audio --example
//! sortformer_stream -- <model_dir> <npy> <feed|stream> [chunk_secs]`.
//!
//! Mirrors the Python streaming reference (`/tmp/sortformer_stream_ref.py`):
//! `feed` runs the real-time chunk loop with the state compressing between
//! chunks; `stream` runs the file-mode `generate_stream` path. Writes
//! per-chunk prediction `.npy` files to /tmp and the final streaming
//! state as raw f32 bins for diffing against the reference.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let npy = args.next().expect("npy");
    let mode = args.next().expect("mode: feed|stream");
    let chunk_secs: f64 = args.next().map(|v| v.parse().unwrap()).unwrap_or(5.0);

    let model = turbospark_audio::models::sortformer::Sortformer::load(&model_dir)?;
    let samples = read_npy_f32(&npy)?;
    let n_spk = model.config().modules.num_speakers;
    let emb_dim = model.config().fc_encoder.hidden_size;

    let mut all_segments: Vec<(f64, f64, usize)> = Vec::new();
    let mut final_state: Option<turbospark_audio::models::sortformer::StreamingState> = None;

    if mode == "feed" {
        let chunk_samples = (chunk_secs * 16000.0) as usize;
        let mut state = model.init_streaming_state();
        for (i, start) in (0..samples.len()).step_by(chunk_samples).enumerate() {
            let chunk = &samples[start..(start + chunk_samples).min(samples.len())];
            let (result, new_state) = model.feed(chunk, 16_000, &state, &Default::default())?;
            state = new_state;
            write_chunk_preds(i, &result.speaker_probs);
            for seg in &result.segments {
                all_segments.push((seg.start, seg.end, seg.speaker));
            }
            println!(
                "chunk {i}: frames {} segs {} cache {} fifo {}",
                result.speaker_probs.len() / n_spk,
                result.segments.len(),
                state.spkcache.len() / emb_dim,
                state.fifo.len() / emb_dim,
            );
        }
        final_state = Some(state);
    } else {
        let results = model.generate_stream(&samples, 16_000, chunk_secs, &Default::default())?;
        for (i, result) in results.iter().enumerate() {
            write_chunk_preds(i, &result.speaker_probs);
            for seg in &result.segments {
                all_segments.push((seg.start, seg.end, seg.speaker));
                println!(
                    "  chunk {i} seg: {:.3}-{:.3} spk {}",
                    seg.start, seg.end, seg.speaker
                );
            }
            println!(
                "chunk {i}: frames {} segs {}",
                result.speaker_probs.len() / n_spk,
                result.segments.len()
            );
        }
    }

    for (start, end, spk) in &all_segments {
        println!(
            "SPEAKER audio 1 {:.3} {:.3} <NA> <NA> speaker_{spk} <NA> <NA>",
            start,
            end - start
        );
    }
    if let Some(state) = final_state {
        write_bin("/tmp/rs_final_spkcache.bin", &state.spkcache);
        write_bin("/tmp/rs_final_spkcache_preds.bin", &state.spkcache_preds);
        write_bin("/tmp/rs_final_fifo.bin", &state.fifo);
        write_bin("/tmp/rs_final_fifo_preds.bin", &state.fifo_preds);
        write_bin("/tmp/rs_final_mean_sil.bin", &state.mean_sil_emb);
        println!(
            "final: frames_processed {} spkcache {} fifo {} n_sil {}",
            state.frames_processed,
            state.spkcache.len() / emb_dim,
            state.fifo.len() / emb_dim,
            state.n_sil_frames
        );
    }
    Ok(())
}

fn write_chunk_preds(i: usize, preds: &[f32]) {
    let path = format!("/tmp/rs_chunk{i}_preds.npy");
    let json = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        preds.len()
    );
    let header_len = {
        let mut len = json.len() + 1;
        while (10 + len) % 64 != 0 {
            len += 1;
        }
        len
    };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(header_len as u16).to_le_bytes());
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend(std::iter::repeat_n(b' ', header_len - json.len() - 1));
    bytes.push(b'\n');
    for v in preds {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write preds");
}

fn write_bin(path: &str, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("write bin");
}

fn read_npy_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if &bytes[..6] != b"\x93NUMPY" {
        return Err("not an npy file".into());
    }
    let major = bytes[6];
    let header_len = if major == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    };
    let offset = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&bytes[offset..offset + header_len])?;
    if !header.contains("<f4") {
        return Err(format!("only little-endian f4 supported, header: {header}").into());
    }
    if header.contains("fortran_order': True") {
        return Err("fortran order not supported".into());
    }
    let data_start = offset + header_len;
    let count = (bytes.len() - data_start) / 4;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = &bytes[data_start + i * 4..data_start + i * 4 + 4];
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    Ok(out)
}
