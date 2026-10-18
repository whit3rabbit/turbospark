//! Speaker-wise sentence assembly, ported from NeMo Speech 218d0b4
//! `MultiTalkerInstanceManager.ASRState.update_sessionwise_seglsts_for_parallel`.
//!
//! After every streaming step each speaker's full hypothesis text is diffed
//! against the previous one. New text either extends the speaker's last
//! sentence or starts a new one when the gap since it exceeds
//! `sent_break_sec`. Times come from the new tokens' global frames relative
//! to the frames that speaker had decoded before the step, offset by the
//! chunk start time. The rules are ported as written (including the word
//! continuation split), because a word split across two segments scores as
//! two words in cpWER.

/// One SegLST segment.
#[derive(Debug, Clone, PartialEq)]
pub struct SeglstEntry {
    pub speaker: String,
    pub start_time: f64,
    pub end_time: f64,
    pub words: String,
}

/// A speaker's hypothesis after a step, as NeMo's `Hypothesis` exposes it.
#[derive(Debug, Clone, PartialEq)]
pub struct HypothesisView<'a> {
    /// Full decoded text of every token so far.
    pub text: &'a str,
    /// Global encoder frame of every token so far.
    pub frames: &'a [usize],
    /// Frames decoded so far (`dec_state.decoded_length`).
    pub decoded_frames: usize,
}

#[derive(Debug, Clone, Default)]
struct SpeakerSentences {
    sentences: Vec<SeglstEntry>,
    prev_text: String,
    prev_tokens: usize,
    prev_decoded: usize,
    /// NeMo skips speakers whose hypothesis is still `None` (never decoded).
    seen: bool,
}

/// A segment that is new, or changed, since the last
/// [`SentenceBuilder::changes`] call. `index` counts the speaker's sentences,
/// so a consumer keys on `(speaker, index)` and replaces what it holds.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentUpdate {
    pub speaker: String,
    pub index: usize,
    pub start_time: f64,
    pub end_time: f64,
    pub words: String,
}

/// What [`SentenceBuilder::changes`] already reported, per speaker: the
/// sentence count and the last sentence's end time and words.
#[derive(Debug, Clone, Default)]
pub struct SentenceCursor {
    seen: Vec<(usize, Option<(f64, String)>)>,
}

/// Per-session sentence state for every speaker id.
#[derive(Debug, Clone)]
pub struct SentenceBuilder {
    frame_seconds: f64,
    sentence_break: f64,
    speakers: Vec<SpeakerSentences>,
}

/// Python `round(x, 3)` for the magnitudes here (non-negative, well below
/// 2^40): round half to even on the scaled value, as CPython does once the
/// decimal value is exact.
fn round3(x: f64) -> f64 {
    let scaled = x * 1000.0;
    let r = scaled.round();
    let r = if (scaled - scaled.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - scaled.signum()
    } else {
        r
    };
    r / 1000.0
}

/// Python `str.capitalize` on one word: first character upper case, the
/// rest lower case.
fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first
            .to_uppercase()
            .chain(chars.as_str().to_lowercase().chars())
            .collect(),
        None => String::new(),
    }
}

impl SentenceBuilder {
    /// `frame_seconds` is the encoder frame (0.08 s); `sentence_break` is the
    /// driver's `sent_break_sec` (1.0 s for Frozen E).
    pub fn new(frame_seconds: f64, sentence_break: f64) -> Self {
        Self {
            frame_seconds,
            sentence_break,
            speakers: Vec::new(),
        }
    }

    fn update_last(&mut self, spk: usize, end: Option<f64>, diff: &str) {
        let frame = self.frame_seconds;
        if let Some(last) = self.speakers[spk].sentences.last_mut() {
            if let Some(end) = end {
                last.end_time = end.max(last.start_time + frame);
            }
            last.words = format!("{}{diff}", last.words).trim().to_string();
        }
    }

    /// Applies one step's hypotheses. `hyps[spk]` is `None` for a speaker id
    /// that exists but has never been decoded.
    pub fn update(&mut self, offset: f64, hyps: &[Option<HypothesisView<'_>>]) {
        if self.speakers.len() < hyps.len() {
            self.speakers
                .resize_with(hyps.len(), SpeakerSentences::default);
        }
        for (spk, hyp) in hyps.iter().enumerate() {
            let Some(hyp) = hyp else { continue };
            self.speakers[spk].seen = true;
            let prev_text = self.speakers[spk].prev_text.clone();
            // NeMo `_is_new_text`.
            let diff = if hyp.text == prev_text {
                None
            } else if let Some(rest) = hyp.text.strip_prefix(prev_text.as_str()) {
                Some(rest.to_string())
            } else {
                Some(hyp.text.trim().to_string())
            };
            if let Some(mut diff) = diff {
                let (start, end, sep_flag) = self.timestamps(spk, hyp, offset);
                let last_end = self.speakers[spk]
                    .sentences
                    .last()
                    .map_or(0.0, |s| s.end_time);
                let has_gap = last_end != 0.0 && start > last_end + self.sentence_break;
                let is_prefix_extension = !prev_text.is_empty() && hyp.text.starts_with(&prev_text);
                let mut continues_word = is_prefix_extension
                    && !diff.is_empty()
                    && !prev_text.ends_with(char::is_whitespace)
                    && !diff.starts_with(char::is_whitespace)
                    && !self.speakers[spk].sentences.is_empty();
                if continues_word && has_gap {
                    let (suffix, rest) = match diff.split_once(' ') {
                        Some((suffix, rest)) => (suffix.to_string(), format!(" {rest}")),
                        None => (diff.clone(), String::new()),
                    };
                    self.update_last(spk, None, &suffix);
                    diff = rest;
                    continues_word = false;
                }
                if continues_word {
                    self.update_last(spk, Some(end), &diff);
                } else if sep_flag || last_end == 0.0 || has_gap {
                    let stripped = diff.trim();
                    if let Some(first) = stripped.chars().next().filter(|c| ".,?!".contains(*c)) {
                        let rest = stripped[first.len_utf8()..].to_string();
                        self.update_last(spk, None, &first.to_string());
                        diff = rest;
                    }
                    let words = diff.trim_start();
                    if !words.trim().is_empty() {
                        let mut split: Vec<String> =
                            words.split_whitespace().map(str::to_owned).collect();
                        split[0] = capitalize(&split[0]);
                        self.speakers[spk].sentences.push(SeglstEntry {
                            speaker: format!("speaker_{spk}"),
                            start_time: round3(start),
                            end_time: round3(end),
                            words: split.join(" "),
                        });
                    }
                } else {
                    self.update_last(spk, Some(end), &diff);
                }
            }
            let s = &mut self.speakers[spk];
            s.prev_text = hyp.text.to_string();
            s.prev_decoded = hyp.decoded_frames;
            s.prev_tokens = hyp.frames.len();
        }
    }

    /// Segments new or changed since `cursor` last saw this builder, then
    /// advances `cursor`. Only a speaker's last sentence can change after it
    /// is written (a step extends it or attaches punctuation), so that one is
    /// resent when its end time or words moved. Values are as
    /// [`entries`](Self::entries) reports them: times rounded, leading space
    /// stripped, empty sentences skipped.
    pub fn changes(&self, cursor: &mut SentenceCursor) -> Vec<SegmentUpdate> {
        cursor
            .seen
            .resize_with(self.speakers.len(), Default::default);
        let mut out = Vec::new();
        for (spk, s) in self.speakers.iter().enumerate().filter(|(_, s)| s.seen) {
            let (count, last) = &mut cursor.seen[spk];
            let snapshot = |e: &SeglstEntry| (e.end_time, e.words.clone());
            let unchanged =
                *count > 0 && s.sentences.get(*count - 1).map(snapshot).as_ref() == last.as_ref();
            let from = if unchanged {
                *count
            } else {
                count.saturating_sub(1)
            };
            for (index, e) in s.sentences.iter().enumerate().skip(from) {
                let words = e.words.trim_start();
                if words.trim().is_empty() {
                    continue;
                }
                out.push(SegmentUpdate {
                    speaker: e.speaker.clone(),
                    index,
                    start_time: round3(e.start_time),
                    end_time: round3(e.end_time),
                    words: words.to_string(),
                });
            }
            *count = s.sentences.len();
            *last = s.sentences.last().map(snapshot);
        }
        out
    }

    /// NeMo `_compute_hypothesis_timestamps`.
    fn timestamps(&self, spk: usize, hyp: &HypothesisView<'_>, offset: f64) -> (f64, f64, bool) {
        let s = &self.speakers[spk];
        let frame = self.frame_seconds;
        if hyp.frames.len() > s.prev_tokens {
            let start_local = hyp.frames[s.prev_tokens] as f64 - s.prev_decoded as f64;
            let end_local = *hyp.frames.last().expect("new tokens") as f64 - s.prev_decoded as f64;
            (
                offset + start_local * frame,
                offset + (end_local + 1.0) * frame,
                false,
            )
        } else {
            let length = hyp.decoded_frames.saturating_sub(s.prev_tokens) as f64;
            (offset, offset + length * frame, true)
        }
    }

    /// The session's SegLST: every decoded speaker's sentences with nonempty
    /// words, sorted by start time (stable, speakers in id order). As NeMo's
    /// `generate_seglst_dicts_from_parallel_streaming` does, times are
    /// rounded and words left-stripped again on output, because sentence
    /// updates store unrounded end times.
    pub fn entries(&self) -> Vec<SeglstEntry> {
        let mut out: Vec<SeglstEntry> = self
            .speakers
            .iter()
            .filter(|s| s.seen)
            .flat_map(|s| s.sentences.iter())
            .map(|e| SeglstEntry {
                speaker: e.speaker.clone(),
                start_time: round3(e.start_time),
                end_time: round3(e.end_time),
                words: e.words.trim_start().to_string(),
            })
            .filter(|e| !e.words.trim().is_empty())
            .collect();
        out.sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(text: &'a str, frames: &'a [usize], decoded: usize) -> Option<HypothesisView<'a>> {
        Some(HypothesisView {
            text,
            frames,
            decoded_frames: decoded,
        })
    }

    /// Known answer produced by NeMo 218d0b4's own
    /// `ASRState.update_sessionwise_seglsts_for_parallel` on these exact
    /// synthetic steps: text extends the open sentence, a gap above
    /// `sent_break_sec` starts a new capitalized one, a word continued across
    /// the gap is finished in the old sentence first, and leading punctuation
    /// attaches to the previous sentence. Times are each speaker's new token
    /// frames relative to the frames it had decoded, plus the chunk offset.
    #[test]
    fn seglst_diff_and_sentence_break_known_answer() {
        let mut b = SentenceBuilder::new(0.08, 1.0);
        // Step 0 (offset 0): "and during" at frames 3..5.
        b.update(0.0, &[view("and during", &[3, 4, 5], 14)]);
        // Step 1 (offset 1.12): extends the sentence.
        b.update(1.12, &[view("and during each", &[3, 4, 5, 16], 28)]);
        // Step 2: no new text for speaker 0; speaker 1 appears.
        b.update(
            2.24,
            &[
                view("and during each", &[3, 4, 5, 16], 42),
                view("okay", &[30], 42),
            ],
        );
        // Step 5 (offset 5.6): speaker 0 continues "each" -> "eachother word"
        // after a gap: the suffix stays in the old sentence.
        b.update(
            5.6,
            &[
                view("and during eachother word", &[3, 4, 5, 16, 72, 73], 84),
                view("okay", &[30], 84),
            ],
        );
        // Step 9: speaker 1 resumes after a gap with leading punctuation.
        b.update(
            10.08,
            &[
                view("and during eachother word", &[3, 4, 5, 16, 72, 73], 140),
                view("okay. right then", &[30, 130, 131, 132], 140),
            ],
        );
        let e = b.entries();
        let summary: Vec<(String, f64, f64, String)> = e
            .iter()
            .map(|s| (s.speaker.clone(), s.start_time, s.end_time, s.words.clone()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (
                    "speaker_0".into(),
                    0.24,
                    1.36,
                    "And during eachother".into()
                ),
                ("speaker_1".into(), 4.64, 4.72, "Okay.".into()),
                ("speaker_0".into(), 8.0, 8.16, "Word".into()),
                ("speaker_1".into(), 13.76, 14.0, "Right then".into()),
            ]
        );
    }

    /// Applying `changes` after every update reproduces `entries`, including
    /// a sentence extended in place and punctuation attached to the previous
    /// one.
    #[test]
    fn changes_replay_to_the_same_segments_as_entries() {
        use std::collections::BTreeMap;
        let steps: Vec<(f64, Vec<Option<HypothesisView<'static>>>)> = vec![
            (0.0, vec![view("and during", &[3, 4, 5], 14)]),
            (1.12, vec![view("and during each", &[3, 4, 5, 16], 28)]),
            (
                2.24,
                vec![
                    view("and during each", &[3, 4, 5, 16], 42),
                    view("okay", &[30], 42),
                ],
            ),
            (
                5.6,
                vec![
                    view("and during eachother word", &[3, 4, 5, 16, 72, 73], 84),
                    view("okay", &[30], 84),
                ],
            ),
            (
                10.08,
                vec![
                    view("and during eachother word", &[3, 4, 5, 16, 72, 73], 140),
                    view("okay. right then", &[30, 130, 131, 132], 140),
                ],
            ),
        ];
        let mut b = SentenceBuilder::new(0.08, 1.0);
        let mut cursor = SentenceCursor::default();
        let mut held: BTreeMap<(String, usize), (f64, f64, String)> = BTreeMap::new();
        let mut resent = 0;
        for (offset, hyps) in &steps {
            b.update(*offset, hyps);
            let updates = b.changes(&mut cursor);
            resent += updates.len();
            for u in updates {
                held.insert((u.speaker, u.index), (u.start_time, u.end_time, u.words));
            }
            let mut want: Vec<_> = b
                .entries()
                .into_iter()
                .map(|e| (e.speaker, e.start_time, e.end_time, e.words))
                .collect();
            let mut got: Vec<_> = held
                .iter()
                .map(|((spk, _), (s, e, w))| (spk.clone(), *s, *e, w.clone()))
                .collect();
            want.sort_by(|a, b| a.partial_cmp(b).unwrap());
            got.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(got, want, "after the update at offset {offset}");
        }
        assert!(b.changes(&mut cursor).is_empty(), "nothing changed since");
        // Nothing is resent when a step adds no text (step 2 for speaker 0).
        assert!(resent < steps.len() * 4, "{resent}");
    }

    /// Speakers that were never decoded produce nothing, and empty text never
    /// opens a sentence.
    #[test]
    fn undecoded_and_empty_speakers_emit_nothing() {
        let mut b = SentenceBuilder::new(0.08, 1.0);
        b.update(0.0, &[view("", &[], 14), None, view("hi", &[2], 14)]);
        let e = b.entries();
        assert_eq!(e.len(), 1);
        assert_eq!(
            (e[0].speaker.as_str(), e[0].words.as_str()),
            ("speaker_2", "Hi")
        );
    }

    #[test]
    fn round3_and_capitalize_follow_python() {
        // NeMo accumulates the chunk offset by repeated addition, which drifts
        // off the 3-decimal grid (1.12 * 3 is 3.3600000000000003).
        let offset = (0..3).fold(0.0f64, |t, _| t + 14.0 * 0.08);
        assert_ne!(offset, 3.36);
        assert_eq!(round3(offset), 3.36);
        assert_eq!(round3(3.3600000000000003), 3.36);
        assert_eq!(capitalize("iPhone"), "Iphone");
        assert_eq!(capitalize("i'm"), "I'm");
    }
}
