//! Streaming windows and greedy RNNT for Parakeet Unified, with no Core ML
//! in it so every rule here is testable on any platform.
//!
//! The Unified encoder is stateless: each step re-encodes a
//! `[left | chunk | right]` window whose chunked-attention mask was baked in
//! at conversion. Only the RNNT prediction state and the last emitted token
//! carry across steps, and the right context is held back from decoding
//! until a later step (or the final flush) re-encodes it with more future
//! audio. Reference: FluidAudio `UnifiedStreamingWindower` and
//! `UnifiedRnntDecoder` at d31145a9476256afbcc7ec4ead33d883795bef78
//! (Apache-2.0), which mirror NeMo's `StreamingBatchedAudioBuffer` loop.

use std::ops::Range;

/// One encoder frame: hop 160 times subsampling 8, 80 ms at 16 kHz.
pub const FRAME_SAMPLES: usize = 1280;
pub const SAMPLE_RATE: usize = 16_000;
/// Upper bound on one append, matching `ts_audio_job_append`.
pub const MAX_APPEND_SAMPLES: usize = 32_000;

/// Context sizes in encoder frames, as named by the encoder bundle
/// (`..._streaming_70_7_7`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub left: usize,
    pub chunk: usize,
    pub right: usize,
}

impl Geometry {
    pub fn window_samples(self) -> usize {
        (self.left + self.chunk + self.right) * FRAME_SAMPLES
    }
    pub fn chunk_samples(self) -> usize {
        self.chunk * FRAME_SAMPLES
    }
    pub fn right_samples(self) -> usize {
        self.right * FRAME_SAMPLES
    }
    /// Seconds from a sample arriving to its text being committed, ignoring
    /// compute: the chunk plus the right-context look-ahead.
    pub fn latency_seconds(self) -> f64 {
        ((self.chunk + self.right) * FRAME_SAMPLES) as f64 / SAMPLE_RATE as f64
    }
}

/// One encoder step: which global samples fill the window (zero-padded to
/// the window length) and how many frames at its end are held back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepPlan {
    pub buffer_start: usize,
    pub buffer_end: usize,
    /// Global encoder frame of `buffer_start`.
    pub start_frame: usize,
    pub holdback: usize,
}

/// Window and frame bookkeeping. The first step waits for chunk plus right
/// samples; each later step feeds one chunk. Every encoder frame is decoded
/// exactly once.
#[derive(Debug, Clone)]
pub struct Windower {
    geometry: Geometry,
    consumed: usize,
    decoded_frames: usize,
    final_flushed: bool,
}

impl Windower {
    pub fn new(geometry: Geometry) -> Self {
        Self {
            geometry,
            consumed: 0,
            decoded_frames: 0,
            final_flushed: false,
        }
    }

    pub fn consumed_samples(&self) -> usize {
        self.consumed
    }

    pub fn decoded_frames(&self) -> usize {
        self.decoded_frames
    }

    /// Plans the next window, or `None` until enough audio has arrived (or,
    /// when `is_final`, once nothing is left). The final window is planned
    /// at most once, so termination never depends on the encoder's exact
    /// output length.
    pub fn next_window(&mut self, total: usize, is_final: bool) -> Option<StepPlan> {
        if self.final_flushed {
            return None;
        }
        let g = self.geometry;
        let feed = if self.consumed == 0 {
            g.chunk_samples() + g.right_samples()
        } else {
            g.chunk_samples()
        };
        let consumed = if self.consumed + feed <= total {
            self.consumed + feed
        } else if is_final && total > 0 && total >= self.consumed {
            // Either a short tail, or the stream ended on a chunk boundary
            // and the held-back right context still needs decoding.
            total
        } else {
            return None;
        };
        let last = is_final && consumed >= total;
        if last {
            self.final_flushed = true;
        }
        let mut buffer_start = consumed.saturating_sub(g.window_samples());
        // Align up so the window never exceeds its fixed length.
        buffer_start += (FRAME_SAMPLES - buffer_start % FRAME_SAMPLES) % FRAME_SAMPLES;
        self.consumed = consumed;
        Some(StepPlan {
            buffer_start,
            buffer_end: consumed,
            start_frame: buffer_start / FRAME_SAMPLES,
            holdback: if last { 0 } else { g.right },
        })
    }

    /// Local frame range of this window to decode, given the encoder's
    /// valid output length; advances the decoded position.
    pub fn decode_range(&mut self, encoder_frames: usize, plan: &StepPlan) -> Option<Range<usize>> {
        let start = self.decoded_frames.checked_sub(plan.start_frame)?;
        let end = encoder_frames.checked_sub(plan.holdback)?;
        if end <= start {
            return None;
        }
        self.decoded_frames += end - start;
        Some(start..end)
    }
}

/// The RNNT prediction and joint networks, one step at a time.
pub trait RnntStepper {
    type State: Clone;
    fn initial(&self) -> Result<Self::State, String>;
    /// Prediction network output after consuming `token` from `state`, and
    /// the state after it.
    fn predict(&self, token: usize, state: &Self::State)
        -> Result<(Vec<f32>, Self::State), String>;
    /// `(token, probability)` for one encoder frame and prediction output.
    fn joint(&self, encoder_step: &[f32], prediction: &[f32]) -> Result<(usize, f32), String>;
}

/// Decoder state carried across streaming steps: the prediction state
/// before `last_token` was consumed, as FluidAudio keeps it.
#[derive(Debug, Clone)]
pub struct Carried<S> {
    pub state: S,
    pub last_token: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Emission {
    pub token: usize,
    /// Global encoder frame.
    pub frame: usize,
    pub probability: f32,
}

/// Encoder output for one window, `[hidden, frames]` channel-major.
#[derive(Debug, Clone)]
pub struct EncodedWindow {
    pub hidden: usize,
    pub frames: usize,
    /// Frames the encoder reports as valid.
    pub valid: usize,
    pub data: Vec<f32>,
}

impl EncodedWindow {
    fn step(&self, t: usize) -> Vec<f32> {
        (0..self.hidden)
            .map(|c| self.data[c * self.frames + t])
            .collect()
    }
}

/// Greedy plain RNNT (no durations): a blank ends the frame, at most
/// `max_symbols` tokens per frame. `carried` is updated only after the whole
/// pass succeeds, so a failed step leaves the stream where it was.
pub fn rnnt_greedy<S: RnntStepper>(
    stepper: &S,
    carried: &mut Carried<S::State>,
    encoded: &EncodedWindow,
    frames: Range<usize>,
    global_offset: usize,
    blank: usize,
    max_symbols: usize,
) -> Result<Vec<Emission>, String> {
    let mut token = carried.last_token;
    let mut state = carried.state.clone();
    let (mut prediction, mut next_state) = stepper.predict(token, &state)?;
    let mut out = Vec::new();
    for t in frames {
        let step = encoded.step(t);
        for _ in 0..max_symbols {
            let (emitted, probability) = stepper.joint(&step, &prediction)?;
            if emitted == blank {
                break;
            }
            out.push(Emission {
                token: emitted,
                frame: global_offset + t,
                probability,
            });
            token = emitted;
            state = next_state;
            (prediction, next_state) = stepper.predict(token, &state)?;
        }
    }
    carried.last_token = token;
    carried.state = state;
    Ok(out)
}

/// What a streaming model provides: window encoding (mel included), the
/// RNNT steps, and the vocabulary.
pub trait StreamingModel: RnntStepper {
    fn geometry(&self) -> Geometry;
    fn blank(&self) -> usize;
    /// Encodes one zero-padded window of `geometry().window_samples()`
    /// samples whose first `valid` samples are audio.
    fn encode(&self, window: &[f32], valid: usize) -> Result<EncodedWindow, String>;
    /// SentencePiece text of a token id, `None` for special pieces.
    fn piece(&self, token: usize) -> Option<&str>;
}

/// A finished word with its span in seconds from the stream start.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StreamWord {
    pub text: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
}

/// What one append or finish committed. Joining `appended_text` over a
/// stream's updates gives its whole transcript: committed text only grows.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct StreamUpdate {
    pub appended_text: String,
    /// Words completed by this update.
    pub closed_words: Vec<StreamWord>,
    /// Committed pieces of a word that later tokens may extend.
    pub open_word: Option<String>,
    pub received_seconds: f64,
    pub committed_through_seconds: f64,
    pub is_final: bool,
}

#[derive(Debug, Clone)]
struct OpenWord {
    text: String,
    start_frame: usize,
    end_frame: usize,
}

const MAX_SYMBOLS_PER_FRAME: usize = 10;
const SECONDS_PER_FRAME: f64 = FRAME_SAMPLES as f64 / SAMPLE_RATE as f64;

/// One streaming transcription's state. The model is passed to each call
/// rather than held, so a session can own both. Audio older than one window
/// behind the consumed position is dropped, so memory stays bounded for
/// recordings of any length.
pub struct UnifiedStream<S> {
    windower: Windower,
    samples: Vec<f32>,
    /// Global index of `samples[0]`.
    samples_start: usize,
    received: usize,
    carried: Option<Carried<S>>,
    open: Option<OpenWord>,
    emitted_text: bool,
    /// Set when a step failed or was cancelled after the windower advanced:
    /// the stream is then partway through a window and refuses more work.
    failed: bool,
}

impl<S: Clone> UnifiedStream<S> {
    pub fn new(geometry: Geometry) -> Self {
        Self {
            windower: Windower::new(geometry),
            samples: Vec::new(),
            samples_start: 0,
            received: 0,
            carried: None,
            open: None,
            emitted_text: false,
            failed: false,
        }
    }

    /// Buffered samples, for the boundedness check.
    pub fn buffered_samples(&self) -> usize {
        self.samples.len()
    }

    /// Adds mono 16 kHz audio and decodes every window it completes.
    /// `cancelled` is checked before each encoder step.
    pub fn append<M: StreamingModel<State = S>>(
        &mut self,
        model: &M,
        pcm: &[f32],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<StreamUpdate, String> {
        if pcm.is_empty() || pcm.len() > MAX_APPEND_SAMPLES {
            return Err(format!(
                "append carries {} samples; it must be 1 to {MAX_APPEND_SAMPLES}",
                pcm.len()
            ));
        }
        if pcm.iter().any(|s| !s.is_finite()) {
            return Err("append carries a non-finite sample".into());
        }
        if self.failed {
            return Err("the stream stopped after an earlier error; open a new one".into());
        }
        self.samples.extend_from_slice(pcm);
        self.received += pcm.len();
        self.run(model, false, cancelled)
    }

    /// Decodes the held-back tail, closes the last word, and resets the
    /// stream to time zero for the next segment.
    pub fn finish<M: StreamingModel<State = S>>(
        &mut self,
        model: &M,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<StreamUpdate, String> {
        let mut update = self.run(model, true, cancelled)?;
        if let Some(word) = self.open.take() {
            update.closed_words.push(Self::word(word));
        }
        update.open_word = None;
        update.is_final = true;
        *self = Self::new(model.geometry());
        Ok(update)
    }

    fn run<M: StreamingModel<State = S>>(
        &mut self,
        model: &M,
        is_final: bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<StreamUpdate, String> {
        if self.failed {
            return Err("the stream stopped after an earlier error; open a new one".into());
        }
        let mut update = StreamUpdate::default();
        while let Some(plan) = self.windower.next_window(self.received, is_final) {
            if cancelled() {
                self.failed = true;
                return Err("cancelled".into());
            }
            if let Err(e) = self.step(model, &plan, &mut update) {
                self.failed = true;
                return Err(e);
            }
            self.trim(model.geometry());
        }
        update.open_word = self.open.as_ref().map(|w| w.text.clone());
        update.received_seconds = self.received as f64 / SAMPLE_RATE as f64;
        update.committed_through_seconds =
            self.windower.decoded_frames() as f64 * SECONDS_PER_FRAME;
        Ok(update)
    }

    fn step<M: StreamingModel<State = S>>(
        &mut self,
        model: &M,
        plan: &StepPlan,
        update: &mut StreamUpdate,
    ) -> Result<(), String> {
        let window_samples = model.geometry().window_samples();
        let local = plan.buffer_start - self.samples_start..plan.buffer_end - self.samples_start;
        let valid = local.len();
        // The NeMo front end needs two hop frames for its statistics; a
        // stream shorter than that has nothing to decode.
        if valid < 2 * 160 {
            return Ok(());
        }
        let mut window = vec![0.0f32; window_samples];
        window[..valid].copy_from_slice(&self.samples[local]);
        let encoded = model.encode(&window, valid)?;
        let frames = encoded.valid.min(encoded.frames);
        let Some(range) = self.windower.decode_range(frames, plan) else {
            return Ok(());
        };
        if self.carried.is_none() {
            self.carried = Some(Carried {
                state: model.initial()?,
                last_token: model.blank(),
            });
        }
        let carried = self.carried.as_mut().expect("set above");
        let emissions = rnnt_greedy(
            model,
            carried,
            &encoded,
            range,
            plan.start_frame,
            model.blank(),
            MAX_SYMBOLS_PER_FRAME,
        )?;
        for e in emissions {
            let Some(piece) = model.piece(e.token) else {
                continue;
            };
            let mut starts_word = piece.starts_with('\u{2581}');
            let mut text = piece.replace('\u{2581}', " ");
            if !starts_word {
                let prev = self.open.as_ref().map(|w| w.text.as_str()).unwrap_or("");
                let prev_len = text.len();
                audio::stt::parakeet::adjust_sentencepiece_digit_spacing(&mut text, prev);
                if text.len() > prev_len {
                    starts_word = true;
                }
            }
            if !self.emitted_text {
                text = text.trim_start().to_string();
                if text.is_empty() {
                    continue;
                }
                self.emitted_text = true;
            }
            update.appended_text.push_str(&text);
            match &mut self.open {
                Some(word) if !starts_word => {
                    word.text.push_str(&text);
                    word.end_frame = e.frame;
                }
                _ => {
                    if let Some(done) = self.open.take() {
                        update.closed_words.push(Self::word(done));
                    }
                    self.open = Some(OpenWord {
                        text: text.trim_start().to_string(),
                        start_frame: e.frame,
                        end_frame: e.frame,
                    });
                }
            }
        }
        Ok(())
    }

    fn word(w: OpenWord) -> StreamWord {
        StreamWord {
            text: w.text,
            start_seconds: w.start_frame as f64 * SECONDS_PER_FRAME,
            end_seconds: (w.end_frame + 1) as f64 * SECONDS_PER_FRAME,
        }
    }

    /// Drops audio no future window can read.
    fn trim(&mut self, geometry: Geometry) {
        let keep_from = self
            .windower
            .consumed_samples()
            .saturating_sub(geometry.window_samples());
        if keep_from > self.samples_start {
            let drop = (keep_from - self.samples_start).min(self.samples.len());
            self.samples.drain(..drop);
            self.samples_start += drop;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    const G: Geometry = Geometry {
        left: 70,
        chunk: 7,
        right: 7,
    };

    /// Drives a windower over `total` samples arriving in `step`-sized
    /// appends and returns every decoded global frame range.
    fn ranges(total: usize, step: usize) -> Vec<Range<usize>> {
        fn drive(w: &mut Windower, received: usize, is_final: bool, out: &mut Vec<Range<usize>>) {
            while let Some(plan) = w.next_window(received, is_final) {
                let frames = (plan.buffer_end - plan.buffer_start).div_ceil(FRAME_SAMPLES);
                if let Some(r) = w.decode_range(frames, &plan) {
                    out.push(plan.start_frame + r.start..plan.start_frame + r.end);
                }
            }
        }
        let mut w = Windower::new(G);
        let mut out = Vec::new();
        let mut received = 0;
        while received < total {
            received = (received + step).min(total);
            drive(&mut w, received, false, &mut out);
        }
        drive(&mut w, received, true, &mut out);
        out
    }

    #[test]
    fn first_step_waits_for_chunk_and_right_context() {
        let mut w = Windower::new(G);
        let need = G.chunk_samples() + G.right_samples();
        assert_eq!(w.next_window(need - 1, false), None);
        let plan = w.next_window(need, false).unwrap();
        assert_eq!(
            (plan.buffer_start, plan.buffer_end, plan.holdback),
            (0, need, G.right)
        );
        // Later steps wait for one more chunk only.
        assert_eq!(w.next_window(need + G.chunk_samples() - 1, false), None);
        assert!(w.next_window(need + G.chunk_samples(), false).is_some());
    }

    #[test]
    fn every_frame_is_decoded_exactly_once() {
        for total in [100, 9_000, 107_520, 200_000, 1_000_000 + 37] {
            for step in [1_600, 16_000, 32_000] {
                let got = ranges(total, step);
                let mut next = 0;
                for r in &got {
                    assert_eq!(r.start, next, "gap or overlap at {total}/{step}: {got:?}");
                    next = r.end;
                }
                assert_eq!(next, total.div_ceil(FRAME_SAMPLES), "{total}/{step}");
            }
        }
    }

    #[test]
    fn right_context_is_held_back_until_a_later_step() {
        let mut w = Windower::new(G);
        let plan = w
            .next_window(G.chunk_samples() + G.right_samples(), false)
            .unwrap();
        let r = w.decode_range(G.chunk + G.right, &plan).unwrap();
        assert_eq!(r, 0..G.chunk, "right context must wait for more audio");
        // The final flush releases it.
        let plan = w
            .next_window(G.chunk_samples() + G.right_samples(), true)
            .unwrap();
        assert_eq!(plan.holdback, 0);
        let r = w.decode_range(G.chunk + G.right, &plan).unwrap();
        assert_eq!(plan.start_frame + r.start, G.chunk);
        assert_eq!(plan.start_frame + r.end, G.chunk + G.right);
        assert_eq!(
            w.next_window(G.chunk_samples() + G.right_samples(), true),
            None
        );
    }

    #[test]
    fn window_never_exceeds_its_length_and_is_frame_aligned() {
        let mut w = Windower::new(G);
        let mut total = 0;
        for _ in 0..200 {
            total += 1_000;
            while let Some(plan) = w.next_window(total, false) {
                assert!(plan.buffer_end - plan.buffer_start <= G.window_samples());
                assert_eq!(plan.buffer_start % FRAME_SAMPLES, 0);
            }
        }
    }

    /// A fake model: the encoder emits one frame per 1280 samples whose
    /// single feature is the mean sample value rounded to a token id, so a
    /// constant tone of value `k` reads as token `k`; zero is blank.
    struct Fake {
        predicts: Cell<usize>,
        states: RefCell<Vec<usize>>,
        geometry: Geometry,
    }

    impl Fake {
        fn new() -> Self {
            Self {
                predicts: Cell::new(0),
                states: RefCell::new(Vec::new()),
                geometry: Geometry {
                    left: 4,
                    chunk: 2,
                    right: 2,
                },
            }
        }
    }

    const PIECES: [&str; 6] = [
        "<blank>",
        "\u{2581}he",
        "llo",
        "\u{2581}world",
        "!",
        "<unk>",
    ];

    impl RnntStepper for Fake {
        type State = usize;
        fn initial(&self) -> Result<usize, String> {
            Ok(0)
        }
        fn predict(&self, token: usize, state: &usize) -> Result<(Vec<f32>, usize), String> {
            self.predicts.set(self.predicts.get() + 1);
            self.states.borrow_mut().push(*state);
            // The prediction remembers the last token so the joint emits a
            // frame's token once, then blank.
            Ok((vec![token as f32], state + 1))
        }
        fn joint(&self, step: &[f32], prediction: &[f32]) -> Result<(usize, f32), String> {
            let token = step[0].round() as usize;
            if token == prediction[0] as usize {
                Ok((0, 1.0))
            } else {
                Ok((token, 0.9))
            }
        }
    }

    impl StreamingModel for Fake {
        fn geometry(&self) -> Geometry {
            self.geometry
        }
        fn blank(&self) -> usize {
            0
        }
        fn encode(&self, window: &[f32], valid: usize) -> Result<EncodedWindow, String> {
            let frames = window.len() / FRAME_SAMPLES;
            let data = (0..frames)
                .map(|f| {
                    let s = &window[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES];
                    s.iter().sum::<f32>() / s.len() as f32
                })
                .collect();
            Ok(EncodedWindow {
                hidden: 1,
                frames,
                valid: valid.div_ceil(FRAME_SAMPLES),
                data,
            })
        }
        fn piece(&self, token: usize) -> Option<&str> {
            PIECES.get(token).copied().filter(|p| !p.starts_with('<'))
        }
    }

    /// Frames of token ids, each held for one encoder frame, with blanks
    /// (silence) between them.
    fn tone(frames: &[usize]) -> Vec<f32> {
        frames
            .iter()
            .flat_map(|&t| std::iter::repeat_n(t as f32, FRAME_SAMPLES))
            .collect()
    }

    fn transcribe(model: &Fake, pcm: &[f32], step: usize) -> (String, Vec<StreamUpdate>) {
        let mut stream = UnifiedStream::new(model.geometry());
        let mut updates = Vec::new();
        for part in pcm.chunks(step) {
            updates.push(stream.append(model, part, &|| false).unwrap());
        }
        updates.push(stream.finish(model, &|| false).unwrap());
        let text = updates.iter().map(|u| u.appended_text.as_str()).collect();
        (text, updates)
    }

    #[test]
    fn append_size_does_not_change_the_transcript() {
        let pcm = tone(&[0, 1, 0, 2, 0, 0, 3, 0, 4, 0, 5, 0, 1, 2, 0, 0, 0, 3, 0]);
        let model = Fake::new();
        let (whole, _) = transcribe(&model, &pcm, MAX_APPEND_SAMPLES);
        assert_eq!(whole, "hello world! hello world");
        for step in [160, 1_600, 4_000] {
            assert_eq!(transcribe(&model, &pcm, step).0, whole, "step {step}");
        }
    }

    #[test]
    fn updates_join_to_the_transcript_and_close_words_with_times() {
        let pcm = tone(&[0, 1, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0]);
        let model = Fake::new();
        let (text, updates) = transcribe(&model, &pcm, 1_280);
        assert_eq!(text, "hello world!");
        let words: Vec<_> = updates
            .iter()
            .flat_map(|u| u.closed_words.clone())
            .collect();
        assert_eq!(
            words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            ["hello", "world!"]
        );
        assert!((words[0].start_seconds - 0.08).abs() < 1e-9);
        assert!((words[0].end_seconds - 0.24).abs() < 1e-9);
        assert!(updates.last().unwrap().is_final);
        let committed: Vec<f64> = updates
            .iter()
            .map(|u| u.committed_through_seconds)
            .collect();
        assert!(committed.windows(2).all(|p| p[0] <= p[1]), "{committed:?}");
        assert!(
            (committed.last().unwrap() - 12.0 * 0.08).abs() < 1e-9,
            "the flush commits all"
        );
    }

    #[test]
    fn decoder_state_carries_across_steps() {
        // With the state reset each step, every step's first prediction
        // would start from 0 again.
        let pcm = tone(&[1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0]);
        let model = Fake::new();
        transcribe(&model, &pcm, 1_280);
        let states = model.states.borrow();
        assert!(states.windows(2).all(|p| p[1] >= p[0]), "{states:?}");
        assert_eq!(*states.last().unwrap(), 3, "three tokens were accepted");
    }

    #[test]
    fn symbols_per_frame_are_capped() {
        struct Stuck;
        impl RnntStepper for Stuck {
            type State = ();
            fn initial(&self) -> Result<(), String> {
                Ok(())
            }
            fn predict(&self, _: usize, _: &()) -> Result<(Vec<f32>, ()), String> {
                Ok((vec![0.0], ()))
            }
            fn joint(&self, _: &[f32], _: &[f32]) -> Result<(usize, f32), String> {
                Ok((7, 1.0))
            }
        }
        let encoded = EncodedWindow {
            hidden: 1,
            frames: 2,
            valid: 2,
            data: vec![0.0, 0.0],
        };
        let mut carried = Carried {
            state: (),
            last_token: 0,
        };
        let out = rnnt_greedy(&Stuck, &mut carried, &encoded, 0..2, 0, 0, 10).unwrap();
        assert_eq!(out.len(), 20);
        assert_eq!(carried.last_token, 7);
    }

    #[test]
    fn buffered_audio_stays_bounded() {
        let model = Fake::new();
        let mut stream = UnifiedStream::new(model.geometry());
        let bound = model.geometry.window_samples() + MAX_APPEND_SAMPLES + 2 * FRAME_SAMPLES;
        let silence = vec![0.0f32; 16_000];
        for _ in 0..3_600 {
            stream.append(&model, &silence, &|| false).unwrap();
            assert!(
                stream.buffered_samples() <= bound,
                "{}",
                stream.buffered_samples()
            );
        }
    }

    #[test]
    fn cancel_stops_before_the_next_step_and_bad_appends_are_refused() {
        let model = Fake::new();
        let mut stream = UnifiedStream::new(model.geometry());
        let pcm = tone(&[1, 0, 0, 0, 0, 0, 0, 0]);
        assert!(stream.append(&model, &[], &|| false).is_err());
        assert!(stream.append(&model, &[f32::NAN], &|| false).is_err());
        assert!(stream
            .append(&model, &vec![0.0; MAX_APPEND_SAMPLES + 1], &|| false)
            .is_err());
        // Refused input changes nothing, so the stream still works.
        assert!(stream.append(&model, &pcm[..100], &|| false).is_ok());
        assert_eq!(
            stream.append(&model, &pcm, &|| true).unwrap_err(),
            "cancelled"
        );
        // A cancelled step left the stream mid-window: it refuses more work.
        let err = stream.append(&model, &pcm, &|| false).unwrap_err();
        assert!(err.contains("earlier error"), "{err}");
        assert!(stream.finish(&model, &|| false).is_err());
    }

    #[test]
    fn finish_resets_to_time_zero() {
        let model = Fake::new();
        let mut stream = UnifiedStream::new(model.geometry());
        let first = stream
            .append(&model, &tone(&[1, 2, 0, 0]), &|| false)
            .unwrap();
        let flushed = stream.finish(&model, &|| false).unwrap();
        assert_eq!(first.appended_text + &flushed.appended_text, "hello");
        let next = stream
            .append(&model, &tone(&[3, 0, 0, 0, 0]), &|| false)
            .unwrap();
        let last = stream.finish(&model, &|| false).unwrap();
        let text = next.appended_text + &last.appended_text;
        assert_eq!(
            text, "world",
            "a new segment starts without a leading space"
        );
        assert!(last.received_seconds < 1.0);
    }
}
