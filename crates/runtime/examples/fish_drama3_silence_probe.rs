//! Fish Audio Drama 3 silence probe: which main tokens decode to silence, and
//! how the semantic head behaves through pauses and pause traps.
//!
//! ```sh
//! cargo run --release -p turbospark-runtime --example fish_drama3_silence_probe -- \
//!     <install-dir> <corpus.txt> <out.jsonl> [seeds=0,1] [gate=on|off]
//! ```
//!
//! Speaks every corpus line at every seed (a line starting `clone|` in a
//! cloned voice, see [`REFERENCE_TEXT`]), then (with `gate=on`) the three
//! texts and seeds of `crates/bench/tests/fish_drama3_pause_trap_gate.rs`.
//! The pause guard is off, so every draw is the raw model's. Otherwise the
//! takes run on the runner's path: default options, [`MAX_CONTEXT`], 16
//! streamed slots, the Metal tower and codec, the codec streamed in 4- then
//! 7-frame pieces. A corpus line may write a character as `\u{hex}`, which
//! keeps a corpus of other scripts ASCII.
//! Writes one JSON line per main draw: the take, the draw's chunk and index,
//! the main code `c0` (`-1` for EOS), the decoded RMS of that frame in dBFS,
//! and the head's EOS probability and top 30 `(index, p)` at temperature 1.0
//! over the whole row (the product recipe's temperature).
//!
//! The decoded audio is not cut per chunk: frame `i` of a take is samples
//! `[2048 i, 2048 (i + 1))` of one codec stream fed every chunk's frames, as
//! the runner feeds it. Measurement only: it changes nothing in the loop.

#![cfg(target_os = "macos")]

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use audio::codec::fish_s1_dac::{DacExec, FishS1Dac};
use audio::tts::fish_drama3::generate::{
    Backbone, ChunkStop, Drama3, Flow, FrameEvent, GenerateOptions, CODEC_FILE, TRUNK_DIR,
};
use audio::tts::fish_drama3::prompt::Utterance;
use audio::tts::fish_drama3::tower::{TowerConfig, TowerEmbeddings, TowerModel};
use turbospark_runtime::fish_drama3::{
    expected_trunk_arch, open_drama, TrunkBackbone, DEFAULT_EXPERT_SLOTS, MAX_CONTEXT,
};
use turbospark_runtime::{ExpertCacheSlots, ExpertResidency, HiddenTrunk};

const FRAME_SAMPLES: usize = 2048;
const TOP: usize = 30;

/// The gate's texts and seeds (`fish_drama3_pause_trap_gate.rs`).
const GATE: [(&str, std::ops::Range<u64>); 3] = [
    (
        "Welcome back. This is a short voiceover test with no reference voice.",
        0..12,
    ),
    (
        "<|speaker:0|> Are you coming tonight? <|speaker:1|> I don't know yet. \
         <|speaker:0|> Let me know by six.",
        0..6,
    ),
    (
        "[laughs] Oh, that is the best thing I have heard all week. [pause] Seriously.",
        0..6,
    ),
];

/// One semantic-head reading: EOS probability and the top [`TOP`] candidates.
struct Head {
    p_eos: f32,
    top: Vec<(usize, f32)>,
}

/// Delegates to the real tower and records every semantic-head reading.
struct Recorder {
    inner: Box<dyn TowerModel + Send>,
    heads: Arc<Mutex<Vec<Head>>>,
}

impl TowerModel for Recorder {
    fn config(&self) -> &TowerConfig {
        self.inner.config()
    }
    fn embeddings(&self) -> &TowerEmbeddings {
        self.inner.embeddings()
    }
    fn position(&self) -> usize {
        self.inner.position()
    }
    fn truncate(&mut self, position: usize) -> audio::Result<()> {
        self.inner.truncate(position)
    }
    fn semantic_logits(&mut self, hidden: &[f32]) -> audio::Result<Vec<f32>> {
        let logits = self.inner.semantic_logits(hidden)?;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|v| (v - max).exp()).sum();
        let p: Vec<f32> = logits.iter().map(|v| (v - max).exp() / z).collect();
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
        order.truncate(TOP);
        self.heads.lock().unwrap().push(Head {
            p_eos: *p.last().unwrap_or(&0.0),
            top: order.into_iter().map(|i| (i, p[i])).collect(),
        });
        Ok(logits)
    }
    fn prefill(
        &mut self,
        hidden: &[f32],
        frames: &[&[i32]],
        labels: &[&[i32]],
    ) -> audio::Result<()> {
        self.inner.prefill(hidden, frames, labels)
    }
    fn begin_residual_codes(
        &mut self,
        hidden: &[f32],
        main: i32,
        frame: &[i32],
        sample: &mut dyn FnMut(usize, &[f32]) -> i32,
    ) -> audio::Result<[i32; 2]> {
        self.inner.begin_residual_codes(hidden, main, frame, sample)
    }
    fn finish_residual_codes(
        &mut self,
        sample: &mut dyn FnMut(usize, &[f32]) -> i32,
    ) -> audio::Result<Vec<i32>> {
        self.inner.finish_residual_codes(sample)
    }
}

/// The cloned voice for `clone|` lines: this text spoken at this seed with no
/// reference, decoded, and encoded back into codes, as the memory oracle
/// builds its reference. Its voice is the model's own, so it stands for "a
/// reference voice" in code coverage, not for any real speaker.
const REFERENCE_TEXT: &str = "Every summer we rent a small cabin by the lake and read until dark.";
const REFERENCE_SEED: u64 = 7;

type BoxError = Box<dyn std::error::Error>;

/// `line` with every `\u{hex}` escape replaced by its character.
fn unescape(line: &str) -> Result<String, BoxError> {
    let mut out = String::new();
    let mut rest = line;
    while let Some(at) = rest.find("\\u{") {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 3..];
        let end = tail.find('}').ok_or("unclosed \\u{ escape")?;
        let code = u32::from_str_radix(&tail[..end], 16)?;
        out.push(char::from_u32(code).ok_or("escape is not a character")?);
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// One take: every frame's codes by book, and every draw's `(chunk, c0)` with
/// `-1` for EOS, and each chunk's end.
struct Take {
    codes: Vec<Vec<i32>>,
    draws: Vec<(usize, i32)>,
    ends: Vec<String>,
}

fn speak(
    drama: &mut Drama3,
    backbone: &mut TrunkBackbone,
    references: &[Utterance],
    text: &str,
    seed: u64,
) -> Result<Take, BoxError> {
    let books = drama.config.tower.num_codebooks;
    let mut take = Take {
        codes: vec![Vec::new(); books],
        draws: Vec::new(),
        ends: Vec::new(),
    };
    let options = GenerateOptions {
        seed,
        // The raw model: a guard rollback would redraw frames the head
        // record has already counted.
        pause_guard: None,
        ..GenerateOptions::default()
    };
    drama.generate(backbone, references, text, options, |event| {
        match event {
            FrameEvent::Frame {
                chunk,
                codes: frame,
            } => {
                take.draws.push((chunk, frame[0]));
                for (book, code) in take.codes.iter_mut().zip(frame) {
                    book.push(code);
                }
            }
            FrameEvent::Held { .. } => {}
            FrameEvent::ChunkEnd {
                chunk,
                frames,
                limit,
                stop,
                ..
            } => {
                if stop == ChunkStop::Eos {
                    take.draws.push((chunk, -1));
                }
                take.ends.push(format!("{frames}/{limit} {stop:?}"));
            }
        }
        Flow::Continue
    })?;
    Ok(take)
}

/// Streams `codes` through the codec in 4- then 7-frame pieces, as the
/// runner does.
fn decode(codec: &FishS1Dac, exec: &DacExec, codes: &[Vec<i32>]) -> Result<Vec<f32>, BoxError> {
    let mut samples = Vec::new();
    let mut stream = codec.stream_on(exec);
    let total = codes[0].len();
    let (mut start, mut piece) = (0, 4);
    while start < total {
        let end = (start + piece).min(total);
        let chunk: Vec<Vec<i32>> = codes.iter().map(|b| b[start..end].to_vec()).collect();
        samples.extend(stream.push(&chunk)?);
        start = end;
        piece = 7;
    }
    Ok(samples)
}

fn main() -> Result<(), BoxError> {
    let mut args = std::env::args().skip(1);
    let install = PathBuf::from(args.next().ok_or("install dir")?);
    let corpus_path = args.next().ok_or("corpus.txt")?;
    let out_path = PathBuf::from(args.next().ok_or("out.jsonl")?);
    let seeds: Vec<u64> = args
        .next()
        .unwrap_or_else(|| "0,1".into())
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let gate = match args.next().as_deref() {
        None | Some("on") => true,
        Some("off") => false,
        Some(other) => return Err(format!("gate {other:?}: on or off").into()),
    };

    // A line starting `clone|` is spoken in the cloned voice.
    let corpus: Vec<(String, bool)> = std::fs::read_to_string(&corpus_path)?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            Ok(match l.strip_prefix("clone|") {
                Some(text) => (unescape(text.trim())?, true),
                None => (unescape(l)?, false),
            })
        })
        .collect::<Result<_, BoxError>>()?;
    let mut takes: Vec<(String, String, u64, bool)> = Vec::new();
    for (i, (text, clone)) in corpus.iter().enumerate() {
        for &seed in &seeds {
            let case = format!("{}{i:02}", if *clone { "clone" } else { "corpus" });
            takes.push((case, text.clone(), seed, *clone));
        }
    }
    if gate {
        for (g, (text, range)) in GATE.iter().enumerate() {
            for seed in range.clone() {
                takes.push((format!("gate{g}"), text.to_string(), seed, false));
            }
        }
    }

    let heads = Arc::new(Mutex::new(Vec::new()));
    let Drama3 {
        config,
        tokenizer,
        tower,
    } = open_drama(&install, MAX_CONTEXT)?;
    let mut drama = Drama3 {
        config,
        tokenizer,
        tower: Box::new(Recorder {
            inner: tower,
            heads: Arc::clone(&heads),
        }),
    };
    let trunk_dir = install.join(TRUNK_DIR);
    let arch = expected_trunk_arch(&install)?;
    let trunk = HiddenTrunk::open(
        &trunk_dir,
        arch,
        MAX_CONTEXT,
        ExpertCacheSlots::Fixed(DEFAULT_EXPERT_SLOTS),
        ExpertResidency::Streamed,
    )?;
    let mut backbone = TrunkBackbone::new(trunk);
    let codec = FishS1Dac::open_weights(&install.join(CODEC_FILE))?;
    let exec = turbospark_runtime::fish_drama3_codec::metal_dac_exec()?;

    let reference = if takes.iter().any(|t| t.3) {
        let take = speak(
            &mut drama,
            &mut backbone,
            &[],
            REFERENCE_TEXT,
            REFERENCE_SEED,
        )?;
        let pcm = decode(&codec, &exec, &take.codes)?;
        let (codes, frames) = codec.encode_on(&exec, &pcm)?;
        eprintln!(
            "reference: {} frames spoken, {frames} encoded, {:?}",
            take.codes[0].len(),
            take.ends
        );
        vec![Utterance {
            text: REFERENCE_TEXT.to_string(),
            codes: codes.into_iter().map(|b| b[..frames].to_vec()).collect(),
        }]
    } else {
        Vec::new()
    };

    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    for (case, text, seed, clone) in &takes {
        heads.lock().unwrap().clear();
        let refs: &[Utterance] = if *clone { &reference } else { &[] };
        let Take { codes, draws, ends } = speak(&mut drama, &mut backbone, refs, text, *seed)?;
        let heads = std::mem::take(&mut *heads.lock().unwrap());
        if heads.len() != draws.len() {
            return Err(format!(
                "{case} seed {seed}: {} head readings for {} draws",
                heads.len(),
                draws.len()
            )
            .into());
        }
        let samples = decode(&codec, &exec, &codes)?;
        let total = codes[0].len();
        let rms_db = |frame: usize| -> Option<f32> {
            let s = samples.get(frame * FRAME_SAMPLES..(frame + 1) * FRAME_SAMPLES)?;
            let power = s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32;
            Some(10.0 * power.max(1e-20).log10())
        };
        let (mut run, mut longest, mut frame) = (0usize, 0usize, 0usize);
        for ((chunk, c0), head) in draws.iter().zip(&heads) {
            let rms = if *c0 >= 0 { rms_db(frame) } else { None };
            if let Some(db) = rms {
                run = if db < -45.0 { run + 1 } else { 0 };
                longest = longest.max(run);
            }
            let top: Vec<String> = head
                .top
                .iter()
                .map(|(i, p)| format!("[{i},{p:.5}]"))
                .collect();
            writeln!(
                out,
                "{{\"case\":\"{case}\",\"seed\":{seed},\"chunk\":{chunk},\"frame\":{},\"c0\":{c0},\
                 \"rms_db\":{},\"p_eos\":{:.6},\"top\":[{}]}}",
                if *c0 >= 0 { frame as i64 } else { -1 },
                rms.map_or("null".into(), |v| format!("{v:.2}")),
                head.p_eos,
                top.join(",")
            )?;
            if *c0 >= 0 {
                frame += 1;
            }
        }
        out.flush()?;
        eprintln!(
            "{case} seed {seed}: {total} frames {ends:?}, longest silent run {longest}: {text}"
        );
    }

    // What a mid-chunk restore point costs on the trunk: the recurrent-state
    // copy and its wait, then the restore's copy back and its wait.
    let (mut snap_ms, mut restore_ms) = (Vec::new(), Vec::new());
    for _ in 0..40 {
        let t = Instant::now();
        let snap = backbone.snapshot()?;
        snap_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        backbone.restore(&snap)?;
        restore_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    for (name, v) in [("snapshot", &mut snap_ms), ("restore", &mut restore_ms)] {
        v.sort_by(f64::total_cmp);
        eprintln!(
            "{name}: median {:.3} ms, max {:.3} ms over {} at position {}",
            v[v.len() / 2],
            v[v.len() - 1],
            v.len(),
            backbone.trunk().position()
        );
    }
    Ok(())
}
