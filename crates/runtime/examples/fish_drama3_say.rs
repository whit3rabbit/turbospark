//! Fish Audio Drama 3 bring-up: text to a 44.1 kHz WAV on the headless Metal
//! trunk (streamed experts), the Metal or f32 CPU tower, and the Metal or
//! CPU codec.
//!
//! ```sh
//! cargo run --release -p turbospark-runtime --example fish_drama3_say -- \
//!     <install-dir> "<text>" <out.wav> [expert-slots=16] [seed=0] [max-frames=1024] \
//!     [residency=streamed|mapped] [tower=metal|cpu] [codec=metal|cpu] [text-cap=on|off]
//! ```
//!
//! `text-cap=off` leaves only `max-frames` as the chunk cap, as the reference
//! does, to measure how long chunks run on their own.
//!
//! `<install-dir>` holds `config.json`, `tokenizer.json`, `codec.safetensors`,
//! `tts_side.safetensors` and `trunk/` (`crates/repack/examples/fish_drama3_intake.rs`).
//! Prints where the time goes, per decoded frame: backbone steps, the rest of
//! the step (tower and sampling), and the codec. A frame is 46.4 ms of audio.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use audio::codec::fish_s1_dac::FishS1Dac;
use audio::tts::fish_drama3::generate::{
    Backbone, BackboneRow, Drama3, Flow, FrameEvent, GenerateOptions, TextCap, CODEC_FILE,
    TRUNK_DIR,
};
use turbospark_runtime::fish_drama3::TrunkBackbone;
use turbospark_runtime::{ExpertCacheSlots, ExpertResidency, HiddenTrunk};

/// Accumulates the time spent inside the backbone.
struct Timed<B> {
    inner: B,
    prefill: Duration,
    prefill_rows: usize,
    steps: Duration,
    step_count: usize,
}

impl<B: Backbone> Backbone for Timed<B> {
    type Snapshot = B::Snapshot;

    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    fn embedding_row(&mut self, token: i32, out: &mut [f32]) -> audio::Result<()> {
        self.inner.embedding_row(token, out)
    }

    fn prefill(&mut self, rows: &[BackboneRow], hidden_out: &mut [f32]) -> audio::Result<()> {
        let t = Instant::now();
        let r = self.inner.prefill(rows, hidden_out);
        self.prefill += t.elapsed();
        self.prefill_rows += rows.len();
        r
    }

    fn step(&mut self, row: &BackboneRow, hidden_out: &mut [f32]) -> audio::Result<()> {
        let t = Instant::now();
        let r = self.inner.step(row, hidden_out);
        self.steps += t.elapsed();
        self.step_count += 1;
        r
    }

    fn snapshot(&mut self) -> audio::Result<Self::Snapshot> {
        self.inner.snapshot()
    }

    fn restore(&mut self, snapshot: &Self::Snapshot) -> audio::Result<()> {
        self.inner.restore(snapshot)
    }

    fn reset(&mut self) {
        self.inner.reset()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let install = PathBuf::from(args.next().ok_or("install dir")?);
    let text = args.next().ok_or("text")?;
    let out = PathBuf::from(args.next().ok_or("out.wav")?);
    let slots: usize = args.next().map_or(Ok(16), |s| s.parse())?;
    let seed: u64 = args.next().map_or(Ok(0), |s| s.parse())?;
    let max_frames: usize = args.next().map_or(Ok(1024), |s| s.parse())?;
    let residency = match args.next().as_deref() {
        None | Some("streamed") => ExpertResidency::Streamed,
        Some("mapped") => ExpertResidency::Mapped,
        Some(other) => return Err(format!("residency {other:?}: streamed or mapped").into()),
    };
    let metal_tower = match args.next().as_deref() {
        None | Some("metal") => true,
        Some("cpu") => false,
        Some(other) => return Err(format!("tower {other:?}: metal or cpu").into()),
    };
    let metal_codec = match args.next().as_deref() {
        None | Some("metal") => true,
        Some("cpu") => false,
        Some(other) => return Err(format!("codec {other:?}: metal or cpu").into()),
    };
    let text_cap = match args.next().as_deref() {
        None | Some("on") => Some(TextCap::MEASURED),
        Some("off") => None,
        Some(other) => return Err(format!("text-cap {other:?}: on or off").into()),
    };

    let open = Instant::now();
    let context = 4096;
    let mut drama = if metal_tower {
        turbospark_runtime::fish_drama3::open_drama(&install, context)?
    } else {
        Drama3::open(&install)?
    };
    let trunk_dir = install.join(TRUNK_DIR);
    let arch = turbospark_repack::peek_manifest_arch(&trunk_dir)?;
    let trunk = HiddenTrunk::open(
        &trunk_dir,
        arch,
        context,
        ExpertCacheSlots::Fixed(slots),
        residency,
    )?;
    let mut backbone = Timed {
        inner: TrunkBackbone::new(trunk),
        prefill: Duration::ZERO,
        prefill_rows: 0,
        steps: Duration::ZERO,
        step_count: 0,
    };
    eprintln!(
        "open {:.2}s ({slots} expert slots, {residency:?} residency, {} tower)",
        open.elapsed().as_secs_f64(),
        if metal_tower { "metal" } else { "cpu" }
    );

    let books = drama.config.tower.num_codebooks;
    let mut codes: Vec<Vec<i32>> = vec![Vec::new(); books];
    let generate = Instant::now();
    let mut first_frame: Option<Duration> = None;
    // `TURBOSPARK_DRAMA3_SERIAL=1` runs the backbone step after the whole
    // tower frame instead of beside its tail, for A/B timing.
    let serial = std::env::var("TURBOSPARK_DRAMA3_SERIAL").as_deref() == Ok("1");
    let options = GenerateOptions {
        seed,
        max_frames,
        text_cap,
        overlap_tower_tail: !serial,
        ..GenerateOptions::default()
    };
    eprintln!(
        "tower tail {}",
        if serial { "serial" } else { "overlapped" }
    );
    let stats = drama.generate(&mut backbone, &[], &text, options, |event| {
        match event {
            FrameEvent::Frame { codes: frame, .. } => {
                first_frame.get_or_insert_with(|| generate.elapsed());
                for (book, code) in codes.iter_mut().zip(frame) {
                    book.push(code);
                }
            }
            FrameEvent::Held { .. } => {}
            FrameEvent::ChunkEnd {
                chunk,
                frames,
                limit,
                stop,
                retries,
            } => {
                eprintln!(
                    "chunk {chunk}: {frames} of {limit} frames, stop {stop:?}, {retries} pause retries"
                );
            }
        }
        Flow::Continue
    })?;
    let total = generate.elapsed();
    let frames = stats.frames.max(1) as f64;
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let rest = total.saturating_sub(backbone.prefill + backbone.steps);
    eprintln!(
        "prompt rows {}: backbone prefill {:.0} ms ({:.1} ms/row)",
        backbone.prefill_rows,
        ms(backbone.prefill),
        ms(backbone.prefill) / backbone.prefill_rows.max(1) as f64
    );
    eprintln!(
        "{} frames ({:.2}s of audio) in {:.2}s, first frame after {:.0} ms",
        stats.frames,
        stats.frames as f64 * 2048.0 / 44_100.0,
        total.as_secs_f64(),
        first_frame.map_or(0.0, ms)
    );
    eprintln!(
        "per frame: backbone {:.1} ms, tower+sampling {:.1} ms (budget 46.4 ms)",
        ms(backbone.steps) / frames,
        ms(rest) / frames
    );

    if stats.frames == 0 {
        return Err("no frames generated".into());
    }
    let codec_open = Instant::now();
    let codec = FishS1Dac::open_weights(&install.join(CODEC_FILE))?;
    let exec = if metal_codec {
        turbospark_runtime::fish_drama3_codec::metal_dac_exec()?
    } else {
        audio::codec::fish_s1_dac::DacExec::host()
    };
    // Streamed the way a live request would be: 4 frames, then 7-frame
    // pieces (0.32 s, the reference's default interval). The first piece
    // also uploads the device weights.
    let decode = Instant::now();
    let mut stream = codec.stream_on(&exec);
    let mut samples = Vec::new();
    let total_frames = codes[0].len();
    let mut start = 0;
    let mut piece = 4;
    while start < total_frames {
        let end = (start + piece).min(total_frames);
        let chunk: Vec<Vec<i32>> = codes.iter().map(|b| b[start..end].to_vec()).collect();
        samples.extend(stream.push(&chunk)?);
        start = end;
        piece = 7;
    }
    eprintln!(
        "codec ({}): open {:.2}s, streamed decode {:.2}s ({:.1} ms/frame)",
        if metal_codec { "metal" } else { "cpu" },
        (decode - codec_open).as_secs_f64(),
        decode.elapsed().as_secs_f64(),
        ms(decode.elapsed()) / frames
    );
    let wave = audio::Waveform {
        sample_rate: codec.config.sample_rate,
        channels: 1,
        samples,
    };
    std::fs::write(&out, audio::write_wav_f32(&wave))?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
