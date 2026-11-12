//! Frozen E meeting transcription with the Nemotron-3 diarization encoder
//! (`nemotron_diar_metal`) and the Multitalker ASR encoder layers
//! (`multitalker_metal`) each on Metal or on the CPU, for paired runs and
//! the panel cpWER gate. Speaker kernels and the RNN-T decode stay on the
//! CPU.
//!
//! ```text
//! cargo run -p turbospark-runtime --release --example frozen_e_metal -- \
//!   --model ~/models/multitalker-parakeet-streaming-0.6b-v1-f32 \
//!   --diar ~/models/Nemotron-3-Diarization-f667ed73-turbospark \
//!   --clips ~/models/frozen-e/clips --out ~/models/frozen-e/metal-run \
//!   [--ids ID,ID] [--diar-device metal|cpu] [--asr-device metal|cpu]
//! ```
//!
//! Writes `<out>/seglst.json` in the layout of `crates/audio`'s
//! `multitalker_meeting` example; score it with `meeting_eval score`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use audio::models::stt::multitalker_asr::scheduler::{CoupledConfig, CoupledSession};
use audio::models::stt::multitalker_asr::MultitalkerAsr;
use audio::models::vad::nemotron_diarization::NemotronDiarization;
use serde_json::{json, Value};
use turbospark_runtime::multitalker_metal::MultitalkerEncoderMetal;
use turbospark_runtime::nemotron_diar_metal::NemotronDiarMetal;

struct Args {
    model: PathBuf,
    diar: PathBuf,
    clips: PathBuf,
    out: PathBuf,
    ids: Vec<String>,
    metal: bool,
    asr_metal: bool,
}

fn device(flag: &str, value: &str) -> Result<bool, String> {
    match value {
        "metal" => Ok(true),
        "cpu" => Ok(false),
        other => Err(format!("{flag} {other}: expected metal or cpu")),
    }
}

fn parse_args() -> Result<Args, String> {
    let (mut model, mut diar, mut clips, mut out, mut ids) = (None, None, None, None, Vec::new());
    let (mut metal, mut asr_metal) = (true, true);
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
            "--diar-device" => metal = device(&flag, &value)?,
            "--asr-device" => asr_metal = device(&flag, &value)?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        diar: diar.ok_or("--diar is required")?,
        clips: clips.ok_or("--clips is required")?,
        out: out.ok_or("--out is required")?,
        ids,
        metal,
        asr_metal,
    })
}

fn clip_ids(clips: &Path, ids: &[String]) -> std::io::Result<Vec<String>> {
    if !ids.is_empty() {
        return Ok(ids.to_vec());
    }
    let mut found: Vec<String> = std::fs::read_dir(clips)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            (p.extension().is_some_and(|x| x == "wav"))
                .then(|| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
                .flatten()
        })
        .collect();
    found.sort();
    Ok(found)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("{e}\nsee the module docs for usage"))?;
    std::fs::create_dir_all(&args.out)?;
    let asr = MultitalkerAsr::open(&args.model)?;
    let diar = NemotronDiarization::load(&args.diar)?;
    let name = |metal: bool| if metal { "metal" } else { "cpu" };
    let mut asr_encoder = if args.asr_metal {
        Some(MultitalkerEncoderMetal::new(&asr)?)
    } else {
        None
    };
    let mut seglst: Vec<Value> = Vec::new();
    for id in clip_ids(&args.clips, &args.ids)? {
        let w = audio::read_wav_f32(&args.clips.join(format!("{id}.wav")))?;
        if w.channels != 1 || w.sample_rate != 16_000 {
            return Err(format!("{id}: expected mono 16 kHz").into());
        }
        let features = asr.features(&w.samples, w.sample_rate)?;
        let mut coupled =
            diar.coupled(diar.config().modules.multitalker_coupled(asr.chunk_rows()))?;
        if args.metal {
            coupled = coupled.with_executor(Box::new(NemotronDiarMetal::new(&diar)?));
        }
        let mut session = CoupledSession::new(&asr, coupled, CoupledConfig::frozen_e());
        if let Some(encoder) = asr_encoder.as_mut() {
            encoder.reset();
            session = session.with_encoder(Box::new(encoder));
        }
        let start = Instant::now();
        let reports = session.run(&features)?;
        let seconds = w.samples.len() as f64 / 16_000.0;
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!(
            "{id}: {} steps, {seconds:.1} s audio in {elapsed:.1} s (RTF {:.2}, diarizer {}, \
             asr {})",
            reports.len(),
            elapsed / seconds,
            name(args.metal),
            name(args.asr_metal)
        );
        seglst.extend(session.seglst().into_iter().map(|e| {
            json!({
                "speaker": e.speaker,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "words": e.words,
                "session_id": id,
            })
        }));
    }
    let path = args.out.join("seglst.json");
    std::fs::write(&path, serde_json::to_string_pretty(&seglst)?)?;
    eprintln!("wrote {} segments to {}", seglst.len(), path.display());
    Ok(())
}
