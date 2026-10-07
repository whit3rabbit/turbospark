//! Greedy CTC collapse state shared by the CTC families.
//!
//! Callers pick each frame's token with their own argmax (the granite
//! family seeds it with the blank id) and feed it here, which drops blanks
//! and immediate repeats. `previous` advances on every frame, blank or
//! not, so a repeat separated by a blank is kept.

pub(crate) struct CtcCollapse {
    blank: usize,
    previous: Option<usize>,
    ids: Vec<usize>,
}

impl CtcCollapse {
    pub(crate) fn new(blank: usize) -> Self {
        Self {
            blank,
            previous: None,
            ids: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, token: usize) {
        if Some(token) != self.previous && token != self.blank {
            self.ids.push(token);
        }
        self.previous = Some(token);
    }

    pub(crate) fn finish(self) -> Vec<usize> {
        self.ids
    }
}

#[cfg(test)]
mod tests {
    use super::CtcCollapse;

    // ---- retired per-family collapses (verbatim) ----

    /// wav2vec / mms `decode_ctc` loop, returning the kept ids.
    fn wav2vec_ids(logits: &[f32], vocab_size: usize) -> Vec<usize> {
        let mut pieces = Vec::new();
        let mut previous = None;
        for row in logits.chunks_exact(vocab_size) {
            let mut token = 0;
            let mut best = f32::NEG_INFINITY;
            for (index, &score) in row.iter().enumerate() {
                if score > best {
                    token = index;
                    best = score;
                }
            }
            if token != previous.unwrap_or(usize::MAX) && token != 0 {
                pieces.push(token);
            }
            previous = Some(token);
        }
        pieces
    }

    /// sensevoice `greedy_ctc` with its loop argmax.
    fn sensevoice_ids(logits: &[f32], frames: usize, vocab: usize) -> Vec<usize> {
        fn argmax(row: &[f32]) -> usize {
            let mut best_index = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for (index, &value) in row.iter().enumerate() {
                if value > best_value {
                    best_value = value;
                    best_index = index;
                }
            }
            best_index
        }
        let mut token_ids = Vec::new();
        let mut previous = None;
        for frame in 0..frames {
            let token = argmax(&logits[frame * vocab..(frame + 1) * vocab]);
            if Some(token) != previous && token != 0 {
                token_ids.push(token);
            }
            previous = Some(token);
        }
        token_ids
    }

    /// granite_speech5_ctc `ctc_collapse` (blank-seeded argmax).
    fn granite_ids(logits: &[f32], rows: usize, vocab_size: usize, blank_id: usize) -> Vec<usize> {
        let mut output = Vec::new();
        let mut previous = None;
        for row in logits.chunks_exact(vocab_size).take(rows) {
            let token = row
                .iter()
                .enumerate()
                .fold(
                    (blank_id, f32::NEG_INFINITY),
                    |(best_id, best), (id, &value)| {
                        if value > best {
                            (id, value)
                        } else {
                            (best_id, best)
                        }
                    },
                )
                .0;
            if Some(token) != previous && token != blank_id {
                output.push(token);
            }
            previous = Some(token);
        }
        output
    }

    #[test]
    fn collapse_matches_the_three_retired_loops() {
        let mut state = 0xC0FF_EE11u32;
        for vocab in [2usize, 3, 5, 9] {
            for frames in [0usize, 1, 2, 7, 40] {
                // A coarse grid with NaN and -inf sprinkled in so ties and
                // degenerate rows both occur.
                let logits: Vec<f32> = (0..frames * vocab)
                    .map(|_| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        match (state >> 24) % 11 {
                            0 => f32::NAN,
                            1 => f32::NEG_INFINITY,
                            v => (v % 4) as f32,
                        }
                    })
                    .collect();
                let wav2vec = wav2vec_ids(&logits, vocab);
                assert_eq!(wav2vec, sensevoice_ids(&logits, frames, vocab));

                let mut collapse = CtcCollapse::new(0);
                for row in logits.chunks_exact(vocab) {
                    collapse.push(crate::nn::argmax(row));
                }
                assert_eq!(collapse.finish(), wav2vec, "vocab {vocab} frames {frames}");

                // Blank-seeded argmax differs from the zero-seeded one on
                // all-NaN rows, which is why the granite caller keeps its own
                // argmax and only shares the collapse.
                for blank in [0usize, vocab - 1] {
                    let mut collapse = CtcCollapse::new(blank);
                    for row in logits.chunks_exact(vocab) {
                        let token = row
                            .iter()
                            .enumerate()
                            .fold(
                                (blank, f32::NEG_INFINITY),
                                |(best_id, best), (id, &value)| {
                                    if value > best {
                                        (id, value)
                                    } else {
                                        (best_id, best)
                                    }
                                },
                            )
                            .0;
                        collapse.push(token);
                    }
                    assert_eq!(
                        collapse.finish(),
                        granite_ids(&logits, frames, vocab, blank)
                    );
                }
            }
        }
    }

    #[test]
    fn a_repeat_split_by_a_blank_is_kept() {
        let mut collapse = CtcCollapse::new(0);
        for token in [1, 1, 0, 1, 2, 2] {
            collapse.push(token);
        }
        assert_eq!(collapse.finish(), vec![1, 1, 2]);
    }
}
