//! Frozen E meeting transcription: Nemotron-3 diarization coupled to
//! Multitalker Parakeet, as NeMo Speech 218d0b4's multitalker driver runs it.
//!
//! ```text
//! cargo run -p turbospark-audio --release --example multitalker_meeting -- \
//!   --model ~/models/multitalker-parakeet-streaming-0.6b-v1-f32 \
//!   --diar ~/models/Nemotron-3-Diarization-f667ed73-turbospark \
//!   --clips ~/models/frozen-e/clips --out ~/models/frozen-e/rust-run \
//!   [--ids ID,ID] [--replay-trace NEMO_TRACE_DIR]
//! ```
//!
//! Writes `<out>/seglst.json` (every clip, NeMo SegLST layout) and
//! `<out>/<id>.trace.jsonl` (one line per step, field names matching
//! `scripts/nemo_multitalker_trace.py`). Score with
//! `meeting_eval score --seglst <out>/seglst.json` or
//! `crates/bench/scripts/frozen_e_score.py`. `--replay-trace` feeds NeMo's
//! recorded per-step diarizer predictions instead of running the diarizer
//! (single clip), which isolates ASR divergence.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value};
use turbospark_audio::models::stt::multitalker_asr::scheduler::{
    CoupledConfig, CoupledSession, DiarizationSource, ReplayedDiarization, StepReport,
};
use turbospark_audio::models::stt::multitalker_asr::MultitalkerAsr;
use turbospark_audio::models::vad::nemotron_diarization::NemotronDiarization;

struct Args {
    model: PathBuf,
    diar: Option<PathBuf>,
    clips: PathBuf,
    out: PathBuf,
    ids: Vec<String>,
    replay: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let (mut model, mut diar, mut clips, mut out, mut ids, mut replay) =
        (None, None, None, None, Vec::new(), None);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--diar" => diar = Some(PathBuf::from(value)),
            "--clips" => clips = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--ids" => {
                ids = value
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect()
            }
            "--replay-trace" => replay = Some(PathBuf::from(value)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        diar,
        clips: clips.ok_or("--clips is required")?,
        out: out.ok_or("--out is required")?,
        ids,
        replay,
    })
}

fn report_json(r: &StepReport) -> Value {
    json!({
        "step": r.step,
        "offset": r.offset,
        "asr_chunk_frames": r.asr_frames,
        "diar_input_frames": r.diar_frames,
        "chunk_preds": r.chunk_preds.chunks(8).map(|c| c.to_vec()).collect::<Vec<_>>(),
        "gate_window_max": r.gate_max,
        "active": r.active,
        "spk_targets": r.spk_targets,
        "bg_targets": r.bg_targets,
        "pre_encoded_len": r.pre_encoded_valid,
        "tokens": r.tokens,
        "token_frames": r.token_frames,
        "texts": r.texts,
        "decoded_length": r.decoded_frames,
        "cache_last_channel_len": r.cache_len,
        "is_buffer_empty": r.last,
    })
}

fn replayed(trace: &Path) -> Result<ReplayedDiarization, Box<dyn std::error::Error>> {
    let mut steps = Vec::new();
    for line in std::fs::read_to_string(trace.join("trace.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let preds: Vec<f32> = row["chunk_preds"]
            .as_array()
            .ok_or("trace row without chunk_preds")?
            .iter()
            .flat_map(|r| r.as_array().into_iter().flatten())
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        steps.push(preds);
    }
    Ok(ReplayedDiarization::new(steps, 8))
}

fn run_clip<D: DiarizationSource>(
    asr: &MultitalkerAsr,
    diar: D,
    wav: &Path,
    id: &str,
    out: &Path,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let w = turbospark_audio::read_wav_f32(wav)?;
    if w.channels != 1 || w.sample_rate != 16_000 {
        return Err(format!("{}: expected mono 16 kHz", wav.display()).into());
    }
    let features = asr.features(&w.samples, w.sample_rate)?;
    let mut session = CoupledSession::new(asr, diar, CoupledConfig::frozen_e());
    let start = Instant::now();
    let reports = session.run(&features)?;
    let seconds = w.samples.len() as f64 / 16_000.0;
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "{id}: {} steps, {:.1} s audio in {elapsed:.1} s (RTF {:.2})",
        reports.len(),
        seconds,
        elapsed / seconds
    );
    let mut trace = String::new();
    for r in &reports {
        trace.push_str(&report_json(r).to_string());
        trace.push('\n');
    }
    std::fs::write(out.join(format!("{id}.trace.jsonl")), trace)?;
    Ok(session
        .seglst()
        .into_iter()
        .map(|e| {
            json!({
                "speaker": e.speaker,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "words": e.words,
                "session_id": id,
            })
        })
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("{e}\nsee the module docs for usage"))?;
    std::fs::create_dir_all(&args.out)?;
    let asr = MultitalkerAsr::open(&args.model)?;
    let ids: Vec<String> = if args.ids.is_empty() {
        let mut ids: Vec<String> = std::fs::read_dir(&args.clips)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let p = e.path();
                (p.extension().is_some_and(|x| x == "wav"))
                    .then(|| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
                    .flatten()
            })
            .collect();
        ids.sort();
        ids
    } else {
        args.ids.clone()
    };
    let mut seglst = Vec::new();
    if let Some(trace) = &args.replay {
        if ids.len() != 1 {
            return Err("--replay-trace takes exactly one clip (--ids ID)".into());
        }
        let wav = args.clips.join(format!("{}.wav", ids[0]));
        seglst.extend(run_clip(&asr, replayed(trace)?, &wav, &ids[0], &args.out)?);
    } else {
        let diar_dir = args
            .diar
            .as_ref()
            .ok_or("--diar is required without --replay-trace")?;
        let diar = NemotronDiarization::load(diar_dir)?;
        for id in &ids {
            let wav = args.clips.join(format!("{id}.wav"));
            let coupled =
                diar.coupled(diar.config().modules.multitalker_coupled(asr.chunk_rows()))?;
            seglst.extend(run_clip(&asr, coupled, &wav, id, &args.out)?);
        }
    }
    std::fs::write(
        args.out.join("seglst.json"),
        serde_json::to_string_pretty(&seglst)?,
    )?;
    eprintln!(
        "wrote {} segments to {}",
        seglst.len(),
        args.out.join("seglst.json").display()
    );
    Ok(())
}
