//! Multitalker Parakeet tests.
//!
//! Goldens in `testdata/multitalker_asr/` come from NeMo Speech 218d0b4 on
//! CPU with the Frozen E settings (`scripts/nemo_multitalker_trace.py`, then
//! `scripts/generate_multitalker_fixture.py`). Checkpoint-gated tests need
//! `TURBOSPEECH_MULTITALKER_MODEL` pointing at the F32 conversion
//! (`scripts/convert_multitalker_nemo.py convert`); the frontend test also
//! needs `TURBOSPEECH_FROZEN_E_TRACE`, a full trace directory whose
//! `meta.json` names the clip it traced.

use super::*;
use serde_json::json;

fn read_npy(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(header.contains("'<f4'") && header.contains("'fortran_order': False"));
    let start = header.find('(').unwrap() + 1;
    let end = header[start..].find(')').unwrap() + start;
    let shape = header[start..end]
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect();
    let data = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, data)
}

fn testdata(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/multitalker_asr")
        .join(name)
}

fn gated_model() -> Option<MultitalkerAsr> {
    let dir = std::env::var("TURBOSPEECH_MULTITALKER_MODEL").ok()?;
    let dir = std::path::PathBuf::from(dir);
    if !dir.join("config.json").exists() {
        eprintln!("skipping: no checkpoint at {}", dir.display());
        return None;
    }
    Some(MultitalkerAsr::open(&dir).unwrap())
}

/// Worst element error scaled by the reference RMS: `max |a - b| / rms(b)`.
/// f32 accumulation order differs from torch's blocked GEMM, so the error
/// grows with the tensor's scale (pre-encode values reach ~1500); a fixed
/// absolute bound would be either loose for small tensors or fail on large
/// ones. Observed values are at or below 1e-5.
fn scaled_max_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (worst, _, _, rms) = diff_stats(a, b);
    (f64::from(worst) / rms.max(f64::MIN_POSITIVE)) as f32
}

/// `(max |a - b|, |b| at that index, rms(a - b), rms(b))` for diagnostics.
fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32, f64, f64) {
    let (mut worst, mut at, mut sq, mut ref_sq) = (0.0f32, 0.0f32, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = y.abs();
        }
        sq += f64::from(d) * f64::from(d);
        ref_sq += f64::from(*y) * f64::from(*y);
    }
    let n = a.len().max(1) as f64;
    (worst, at, (sq / n).sqrt(), (ref_sq / n).sqrt())
}

fn tiny_config() -> Value {
    json!({
        "model_type": "multitalker_asr",
        "preprocessor": {
            "sample_rate": 16000, "features": 4, "n_fft": 8,
            "window_size": 0.00025, "window_stride": 0.000125, "window": "hann",
            "normalize": "NA", "preemph": 0.97,
            "log_zero_guard_value": 2f64.powi(-24)
        },
        "encoder": {
            "feat_in": 4, "n_layers": 1, "d_model": 8, "n_heads": 2,
            "ff_expansion_factor": 2, "subsampling_factor": 8,
            "subsampling_conv_channels": 2, "conv_kernel_size": 3,
            "causal_downsampling": true, "conv_context_size": "causal",
            "conv_norm_type": "layer_norm", "self_attention_model": "rel_pos",
            "att_context_style": "chunked_limited", "att_context_size": [[70, 13], [70, 6]],
            "pos_emb_max_len": 5000, "use_bias": false, "xscaling": false
        },
        "speaker_kernels": {"type": "ff", "layers": [0], "background": true, "hidden": 8},
        "decoder": {"pred_hidden": 4, "pred_rnn_layers": 1, "vocab_size": 3, "blank_as_pad": true},
        "joint": {
            "joint_hidden": 4, "activation": "relu", "encoder_hidden": 8,
            "pred_hidden": 4, "num_classes": 3
        },
        "vocabulary": ["<unk>", "\u{2581}a", "b"],
        "default_att_context_size": [70, 13],
        "max_symbols": 10
    })
}

/// The prompt-free parse accepts the Multitalker graph (no `prompt`, no
/// `default_language`), while the Nemotron parser still refuses it and the
/// speaker-kernel block must be the implemented one.
#[test]
fn prompt_free_parse_accepts_multitalker_and_from_json_still_rejects() {
    let value = tiny_config();
    let config = NemotronAsrConfig::from_json_prompt_free(&value).unwrap();
    assert_eq!(config.default_attention_context, [70, 13]);
    assert_eq!((config.num_prompts, config.prompt_hidden), (0, 0));
    assert!(config.prompt_indices.is_empty());
    assert!(NemotronAsrConfig::from_json(&value).is_err());
    let mut renamed = value.clone();
    renamed["model_type"] = json!("nemotron_asr");
    let err = NemotronAsrConfig::from_json(&renamed)
        .unwrap_err()
        .to_string();
    assert!(err.contains("prompt"), "{err}");

    check_speaker_kernels(&value, 8).unwrap();
    for (key, bad) in [
        ("layers", json!([1])),
        ("background", json!(false)),
        ("type", json!("conv2d")),
        ("hidden", json!(16)),
    ] {
        let mut v = value.clone();
        v["speaker_kernels"][key] = bad;
        assert!(check_speaker_kernels(&v, 8).is_err(), "{key} accepted");
    }
}

/// The set NeMo's `extract_punctuation_from_vocab` returns for the pinned
/// checkpoint, as the converter writes it.
const FROZEN_E_PUNCTUATION: [char; 13] = [
    '!', '"', '%', '&', '\'', '*', ',', '-', '.', '/', ':', '?', '_',
];

/// Expected outputs are NeMo 218d0b4's own `space_before_punct_pattern`
/// substitution run on these inputs with the pinned set.
#[test]
fn strip_space_before_punctuation_matches_nemo() {
    for (text, want) in [
        (
            "and 'cause we know . ok , right",
            "and'cause we know. ok, right",
        ),
        ("a  . b", "a . b"),
        ("uh 'Kay. 50 % - yes", "uh'Kay. 50%- yes"),
        ("x \t?y", "x ?y"),
        (" 'tis", "'tis"),
        ("no change here", "no change here"),
    ] {
        assert_eq!(
            strip_space_before_punctuation(text, &FROZEN_E_PUNCTUATION),
            want,
            "{text:?}"
        );
    }
}

#[test]
fn supported_punctuation_is_required_and_single_chars() {
    let mut value = tiny_config();
    assert!(supported_punctuation(&value).is_err());
    value["supported_punctuation"] = json!(["'", ","]);
    assert_eq!(supported_punctuation(&value).unwrap(), ['\'', ',']);
    for bad in [json!(["''"]), json!([""]), json!([1]), json!("',")] {
        value["supported_punctuation"] = bad.clone();
        assert!(supported_punctuation(&value).is_err(), "{bad} accepted");
    }
}

/// NeMo `solve_length_mismatch`: short masks are left-padded with the
/// default, long ones keep their last rows (the final Frozen E chunk passes
/// 14 targets for 3 rows).
#[test]
fn align_mask_left_pads_and_keeps_last() {
    assert_eq!(align_mask(&[0.0, 1.0], 4, 1.0), [1.0, 1.0, 0.0, 1.0]);
    assert_eq!(align_mask(&[0.0, 1.0], 4, 0.0), [0.0, 0.0, 0.0, 1.0]);
    let fourteen: Vec<f32> = (0..14).map(|i| i as f32).collect();
    assert_eq!(align_mask(&fourteen, 3, 1.0), [11.0, 12.0, 13.0]);
    assert_eq!(align_mask(&[5.0, 6.0], 2, 0.0), [5.0, 6.0]);
}

/// The step geometry Frozen E derives from `att_context_size = [70, 13]`
/// and x8 dw_striding: 14-row chunks, a 112-frame shift, a 9-frame
/// pre-encode cache and 2 dropped rows.
#[test]
fn frozen_e_step_geometry_known_values() {
    let Some(model) = gated_model() else { return };
    assert_eq!(
        (
            model.chunk_rows(),
            model.shift_frames(),
            model.pre_encode_cache_frames(),
            model.drop_extra_pre_encoded()
        ),
        (14, 112, 9, 2)
    );
    assert_eq!(model.punctuation, FROZEN_E_PUNCTUATION);
}

/// Speaker kernels against NeMo on a two-speaker step (step 46 of the
/// IS1004a trace): target and background masks with partial activity.
#[test]
fn speaker_kernels_match_nemo() {
    let Some(model) = gated_model() else { return };
    let (pshape, pre) = read_npy(&testdata("kernel_case_pre_encoded.npy"));
    let (sshape, spk) = read_npy(&testdata("kernel_case_spk.npy"));
    let (_, bg) = read_npy(&testdata("kernel_case_bg.npy"));
    let (lshape, golden) = read_npy(&testdata("kernel_case_layer0_input.npy"));
    let (rows, hidden, speakers) = (pshape[0], pshape[1], sshape[0]);
    assert_eq!((lshape[0], lshape[1], lshape[2]), (speakers, rows, hidden));
    let pre = PreEncoded {
        data: pre,
        rows,
        valid: rows,
    };
    let mut worst = 0.0f32;
    for s in 0..speakers {
        let got = model.apply_speaker_kernels(
            &pre,
            &spk[s * sshape[1]..(s + 1) * sshape[1]],
            &bg[s * sshape[1]..(s + 1) * sshape[1]],
        );
        let want = &golden[s * rows * hidden..(s + 1) * rows * hidden];
        worst = worst.max(scaled_max_error(&got, want));
        eprintln!("speaker {s} kernels {:?}", diff_stats(&got, want));
    }
    eprintln!("speaker kernels scaled max error {worst:e}");
    assert!(worst <= 1e-4, "speaker kernels differ from NeMo by {worst}");
}

/// The first six Frozen E steps of the IS1004a trace (one active speaker):
/// pre-encode, layer-0 input, encoder output and the RNN-T tokens with their
/// global frames, carrying caches and decoder state across steps.
#[test]
fn stream_steps_match_nemo() {
    let Some(model) = gated_model() else { return };
    let (fshape, feats) = read_npy(&testdata("features.npy"));
    let (mels, total) = (fshape[0], fshape[1]);
    let steps: Value =
        serde_json::from_str(&std::fs::read_to_string(testdata("steps.json")).unwrap()).unwrap();
    let features = Features {
        frames: total,
        mels,
        data: (0..total)
            .flat_map(|t| (0..mels).map(move |m| (t, m)))
            .map(|(t, m)| feats[m * total + t])
            .collect(),
    };
    let shift = model.shift_frames();
    let cache = model.pre_encode_cache_frames();
    let hidden = model.config().encoder_hidden;
    let mut stream = model.new_stream();
    let (mut worst_pre, mut worst_l0, mut worst_enc) = (0.0f32, 0.0f32, 0.0f32);
    for step in steps.as_array().unwrap() {
        let k = step["step"].as_u64().unwrap() as usize;
        assert_eq!(
            step["active"],
            json!([0]),
            "fixture steps are single-speaker"
        );
        let frames = step["asr_chunk_frames"].as_u64().unwrap() as usize;
        let valid = step["asr_chunk_len"].as_u64().unwrap() as usize;
        let drop = step["drop_extra_pre_encoded"].as_u64().unwrap() as usize;
        let chunk = features.frame_major((k * shift) as isize - cache as isize, frames);
        let pre = model.pre_encode(&chunk, frames, valid, drop).unwrap();
        assert_eq!(
            pre.valid,
            step["pre_encoded_len"].as_u64().unwrap() as usize
        );
        let (_, golden_pre) = read_npy(&testdata(&format!("step_{k:04}_pre_encoded.npy")));
        worst_pre = worst_pre.max(scaled_max_error(&pre.data, &golden_pre));
        eprintln!(
            "step {k} pre-encode {:?}",
            diff_stats(&pre.data, &golden_pre)
        );

        let target: Vec<f32> = step["spk_targets"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let background: Vec<f32> = step["bg_targets"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let out = model
            .step(&pre, &target, &background, &mut stream, false)
            .unwrap();
        let (_, golden_l0) = read_npy(&testdata(&format!("step_{k:04}_layer0_input.npy")));
        worst_l0 = worst_l0.max(scaled_max_error(&out.layer0_input, &golden_l0));
        eprintln!(
            "step {k} layer-0 {:?}",
            diff_stats(&out.layer0_input, &golden_l0)
        );
        // NeMo encoder output is channel-major [1, hidden, rows].
        let (eshape, golden_enc) = read_npy(&testdata(&format!("step_{k:04}_encoded.npy")));
        let rows = eshape[2];
        assert_eq!(out.encoded_len, rows);
        for t in 0..rows {
            for c in 0..hidden {
                let d = (out.encoded[t * hidden + c] - golden_enc[c * rows + t]).abs();
                worst_enc = worst_enc.max(d);
            }
        }
        let want_tokens: Vec<usize> = serde_json::from_value(step["tokens"][0].clone()).unwrap();
        let want_frames: Vec<usize> =
            serde_json::from_value(step["token_frames"][0].clone()).unwrap();
        let got_tokens: Vec<usize> = stream.tokens().iter().map(|t| t.id).collect();
        let got_frames: Vec<usize> = stream.tokens().iter().map(|t| t.frame).collect();
        assert_eq!(got_tokens, want_tokens, "step {k} tokens");
        assert_eq!(got_frames, want_frames, "step {k} token frames");
        assert_eq!(
            model.detokenize(&got_tokens),
            step["texts"][0].as_str().unwrap(),
            "step {k} text"
        );
        assert_eq!(
            stream.decoded_frames(),
            step["decoded_length"][0].as_u64().unwrap() as usize
        );
        assert_eq!(
            stream.cache_len(),
            step["cache_last_channel_len"][0].as_u64().unwrap() as usize
        );
    }
    eprintln!(
        "pre-encode scaled {worst_pre:e}, layer-0 scaled {worst_l0:e}, encoder abs {worst_enc:e}"
    );
    assert!(worst_pre <= 1e-4, "pre-encode differs by {worst_pre}");
    assert!(worst_l0 <= 1e-4, "layer-0 input differs by {worst_l0}");
    assert!(worst_enc <= 1e-4, "encoder output differs by {worst_enc}");
}

/// Whole-clip log-mel against NeMo's streaming buffer for the traced clip:
/// constant STFT padding, no normalization, the final centered frame zeroed.
#[test]
fn frontend_matches_nemo() {
    let Some(model) = gated_model() else { return };
    let Ok(trace) = std::env::var("TURBOSPEECH_FROZEN_E_TRACE") else {
        eprintln!("skipping: TURBOSPEECH_FROZEN_E_TRACE unset");
        return;
    };
    let trace = std::path::PathBuf::from(trace);
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(trace.join("meta.json")).unwrap()).unwrap();
    let wav = crate::wav::read_wav_f32(Path::new(meta["wav"].as_str().unwrap())).unwrap();
    assert_eq!((wav.sample_rate, wav.channels), (16_000, 1));
    let got = model.features(&wav.samples, 16_000).unwrap();
    let (shape, want) = read_npy(&trace.join("features.npy"));
    assert_eq!((got.mels, got.frames), (shape[0], shape[1]));
    let mut worst = 0.0f32;
    for t in 0..got.frames {
        for m in 0..got.mels {
            worst = worst.max((got.data[t * got.mels + m] - want[m * got.frames + t]).abs());
        }
    }
    assert!(got.data[(got.frames - 1) * got.mels..]
        .iter()
        .all(|&v| v == 0.0));
    let transposed: Vec<f32> = (0..got.frames)
        .flat_map(|t| (0..got.mels).map(move |m| (t, m)))
        .map(|(t, m)| want[m * got.frames + t])
        .collect();
    eprintln!("frontend {:?}", diff_stats(&got.data, &transposed));
    let scaled = scaled_max_error(&got.data, &transposed);
    eprintln!("frontend max |d| {worst:e}, scaled {scaled:e}");
    assert!(
        scaled <= 1e-4,
        "frontend differs from NeMo by {scaled} (scaled)"
    );
}

/// Runs one full traced clip through the coupled scheduler and requires every
/// step (active speakers, binarized targets, tokens, token frames, decoded
/// and cache lengths) and the final SegLST to equal the NeMo trace.
fn assert_full_clip_matches_nemo<D: scheduler::DiarizationSource>(
    model: &MultitalkerAsr,
    diar: D,
    trace: &Path,
) {
    let (wav, session_id) = trace_wav(trace);
    let features = model.features(&wav.samples, 16_000).unwrap();
    let mut session =
        scheduler::CoupledSession::new(model, diar, scheduler::CoupledConfig::frozen_e());
    let reports = session.run(&features).unwrap();
    assert_reports_match_nemo(&reports, &session.seglst(), trace, &session_id);
}

/// The traced clip's samples and session id.
fn trace_wav(trace: &Path) -> (crate::waveform::Waveform, String) {
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(trace.join("meta.json")).unwrap()).unwrap();
    let wav_path = std::path::PathBuf::from(meta["wav"].as_str().unwrap());
    let session_id = wav_path.file_stem().unwrap().to_string_lossy().into_owned();
    (crate::wav::read_wav_f32(&wav_path).unwrap(), session_id)
}

/// Every step and the final SegLST of a run against NeMo's trace.
fn assert_reports_match_nemo(
    reports: &[scheduler::StepReport],
    seglst: &[seglst::SeglstEntry],
    trace: &Path,
    session_id: &str,
) {
    let nemo: Vec<Value> = std::fs::read_to_string(trace.join("trace.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(reports.len(), nemo.len(), "step count");
    for (r, n) in reports.iter().zip(&nemo) {
        let k = r.step;
        assert_eq!(json!(r.active), n["active"], "step {k} active speakers");
        if r.active.is_empty() {
            continue;
        }
        assert_eq!(json!(r.spk_targets), n["spk_targets"], "step {k} targets");
        assert_eq!(json!(r.bg_targets), n["bg_targets"], "step {k} background");
        assert_eq!(json!(r.tokens), n["tokens"], "step {k} tokens");
        assert_eq!(
            json!(r.token_frames),
            n["token_frames"],
            "step {k} token frames"
        );
        assert_eq!(
            json!(r.decoded_frames),
            n["decoded_length"],
            "step {k} decoded"
        );
        assert_eq!(
            json!(r.cache_len),
            n["cache_last_channel_len"],
            "step {k} cache"
        );
    }
    let got: Vec<Value> = seglst
        .iter()
        .map(|e| {
            json!({"speaker": e.speaker, "start_time": e.start_time, "end_time": e.end_time,
                   "words": e.words, "session_id": session_id})
        })
        .collect();
    let want: Value =
        serde_json::from_str(&std::fs::read_to_string(trace.join("seglst.json")).unwrap()).unwrap();
    assert_eq!(json!(got), want, "SegLST");
}

fn gated_trace() -> Option<std::path::PathBuf> {
    let Ok(trace) = std::env::var("TURBOSPEECH_FROZEN_E_TRACE") else {
        eprintln!("skipping: TURBOSPEECH_FROZEN_E_TRACE unset");
        return None;
    };
    Some(std::path::PathBuf::from(trace))
}

/// ASR side of the full clip: NeMo's recorded diarizer predictions replayed,
/// so any mismatch is the Multitalker port's.
#[test]
fn scheduler_with_replayed_diarization_matches_nemo_full_clip() {
    let Some(model) = gated_model() else { return };
    let Some(trace) = gated_trace() else { return };
    let steps: Vec<Vec<f32>> = std::fs::read_to_string(trace.join("trace.jsonl"))
        .unwrap()
        .lines()
        .map(|l| {
            let row: Value = serde_json::from_str(l).unwrap();
            row["chunk_preds"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| r.as_array().unwrap().iter())
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();
    assert_full_clip_matches_nemo(
        &model,
        scheduler::ReplayedDiarization::new(steps, 8),
        &trace,
    );
}

/// The whole Frozen E pipeline on the full clip: the Rust coupled Nemotron-3
/// diarizer (`TURBOSPEECH_NEMOTRON_MODEL`, the f667ed73 conversion) feeding
/// the Rust Multitalker model.
#[test]
fn scheduler_with_live_diarization_matches_nemo_full_clip() {
    let Some(model) = gated_model() else { return };
    let Some(trace) = gated_trace() else { return };
    let Ok(diar_dir) = std::env::var("TURBOSPEECH_NEMOTRON_MODEL") else {
        eprintln!("skipping: TURBOSPEECH_NEMOTRON_MODEL unset");
        return;
    };
    let diar =
        crate::models::vad::nemotron_diarization::NemotronDiarization::load(Path::new(&diar_dir))
            .unwrap();
    let coupled = diar
        .coupled(
            diar.config()
                .modules
                .multitalker_coupled(model.chunk_rows()),
        )
        .unwrap();
    assert_full_clip_matches_nemo(&model, coupled, &trace);
}

/// Streams the traced clip through [`scheduler::LiveSession`] in uneven PCM
/// pieces (smaller than a hop, odd, and larger than a chunk) and requires the
/// same steps and SegLST as the whole-file run.
fn assert_live_clip_matches_nemo<D: scheduler::DiarizationSource>(
    model: &MultitalkerAsr,
    diar: D,
    trace: &Path,
) {
    let (wav, session_id) = trace_wav(trace);
    let mut session =
        scheduler::LiveSession::new(model, diar, scheduler::CoupledConfig::frozen_e()).unwrap();
    let mut reports = Vec::new();
    let mut held = std::collections::BTreeMap::new();
    let (mut at, mut sizes) = (0, [3_331usize, 97, 16_000, 160, 40_001].iter().cycle());
    while at < wav.samples.len() {
        let n = (*sizes.next().unwrap()).min(wav.samples.len() - at);
        reports.extend(session.append(&wav.samples[at..at + n]).unwrap());
        for u in session.take_segment_updates() {
            held.insert((u.speaker, u.index), (u.start_time, u.end_time, u.words));
        }
        at += n;
    }
    reports.extend(session.finish().unwrap());
    for u in session.take_segment_updates() {
        held.insert((u.speaker, u.index), (u.start_time, u.end_time, u.words));
    }
    // The deltas, applied as a consumer would, are the final SegLST.
    let mut from_updates: Vec<_> = held
        .into_iter()
        .map(|((spk, _), (start, end, words))| (spk, start, end, words))
        .collect();
    let mut from_entries: Vec<_> = session
        .seglst()
        .into_iter()
        .map(|e| (e.speaker, e.start_time, e.end_time, e.words))
        .collect();
    from_updates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    from_entries.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        from_updates, from_entries,
        "segment updates replay to the SegLST"
    );
    assert_reports_match_nemo(&reports, &session.seglst(), trace, &session_id);
    assert!(
        session.append(&[0.0]).is_err(),
        "a finished session is closed"
    );
}

/// Live ASR side with NeMo's recorded diarizer predictions replayed.
#[test]
fn live_session_with_replayed_diarization_matches_nemo_full_clip() {
    let Some(model) = gated_model() else { return };
    let Some(trace) = gated_trace() else { return };
    let steps: Vec<Vec<f32>> = std::fs::read_to_string(trace.join("trace.jsonl"))
        .unwrap()
        .lines()
        .map(|l| {
            let row: Value = serde_json::from_str(l).unwrap();
            row["chunk_preds"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| r.as_array().unwrap().iter())
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();
    assert_live_clip_matches_nemo(
        &model,
        scheduler::ReplayedDiarization::new(steps, 8),
        &trace,
    );
}

/// The whole live pipeline: incremental frontend, coupled diarizer and
/// Multitalker, against NeMo's trace.
#[test]
fn live_session_with_live_diarization_matches_nemo_full_clip() {
    let Some(model) = gated_model() else { return };
    let Some(trace) = gated_trace() else { return };
    let Ok(diar_dir) = std::env::var("TURBOSPEECH_NEMOTRON_MODEL") else {
        eprintln!("skipping: TURBOSPEECH_NEMOTRON_MODEL unset");
        return;
    };
    let diar =
        crate::models::vad::nemotron_diarization::NemotronDiarization::load(Path::new(&diar_dir))
            .unwrap();
    let coupled = diar
        .coupled(
            diar.config()
                .modules
                .multitalker_coupled(model.chunk_rows()),
        )
        .unwrap();
    assert_live_clip_matches_nemo(&model, coupled, &trace);
}

/// A failed step poisons the session, and a non-finite chunk does not.
#[test]
fn live_session_refuses_work_after_a_failed_step_but_not_a_bad_chunk() {
    let Some(model) = gated_model() else { return };
    let mut session = scheduler::LiveSession::new(
        &model,
        scheduler::ReplayedDiarization::new(Vec::new(), 8),
        scheduler::CoupledConfig::frozen_e(),
    )
    .unwrap();
    assert!(session.append(&[f32::NAN]).is_err());
    // No replayed step exists, so the first full chunk fails the diarizer.
    let silence = vec![0.01f32; 16_000 * 3];
    assert!(session.append(&silence).is_err());
    let again = session.append(&[0.0; 160]).unwrap_err().to_string();
    assert!(again.contains("failed on an earlier step"), "{again}");
}

/// A cancel hook stops a call before its next step and closes the session.
#[test]
fn live_session_cancel_stops_before_the_next_step() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    let Some(model) = gated_model() else { return };
    let stop = Arc::new(AtomicBool::new(false));
    let mut session = scheduler::LiveSession::new(
        &model,
        scheduler::ReplayedDiarization::new(Vec::new(), 8),
        scheduler::CoupledConfig::frozen_e(),
    )
    .unwrap()
    .with_cancel(Box::new({
        let stop = Arc::clone(&stop);
        move || stop.load(Ordering::Acquire)
    }));
    stop.store(true, Ordering::Release);
    let err = session.append(&vec![0.01f32; 16_000 * 3]).unwrap_err();
    assert!(err.to_string().contains("cancelled"), "{err}");
    assert!(session.append(&[0.0; 160]).is_err());
}
