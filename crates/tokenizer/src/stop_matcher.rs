//! Incremental stop-string matcher across streaming text chunks. Ported from
//! `Runtime/Generation/StreamingStopMatcher.swift`.

/// Holds back the longest suffix of accumulated text that could still be the
/// start of one of `stops`, releasing it once it stops being a viable
/// prefix, so a stop string split across two `push` calls is still caught.
pub struct StreamingStopMatcher {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl StreamingStopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            pending: String::new(),
            stopped: false,
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Feed the next chunk of generated text. Returns the portion safe to
    /// emit immediately (text before any withheld possible-stop suffix, or
    /// text before a confirmed stop match).
    pub fn push(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        self.pending.push_str(text);
        if let Some(match_start) = self.earliest_match() {
            let output = self.pending[..match_start].to_string();
            self.pending.clear();
            self.stopped = true;
            return output;
        }
        let retained = self.longest_possible_suffix();
        let boundary = self.pending.len() - retained;
        let output = self.pending[..boundary].to_string();
        self.pending = self.pending[boundary..].to_string();
        output
    }

    /// End of stream: releases any withheld text that never completed a stop
    /// match.
    pub fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }

    fn earliest_match(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min()
    }

    /// Longest suffix of `pending` (in bytes) that is a proper prefix of any
    /// stop string, i.e. the tail that must be withheld pending more input.
    fn longest_possible_suffix(&self) -> usize {
        let mut best = 0usize;
        let pending_len = self.pending.chars().count();
        for stop in &self.stops {
            let stop_len = stop.chars().count();
            let maximum = pending_len.min(stop_len.saturating_sub(1));
            for length in (1..=maximum).rev() {
                let suffix = char_suffix(&self.pending, length);
                let prefix = char_prefix(stop, length);
                if suffix == prefix {
                    best = best.max(byte_len_of_char_suffix(&self.pending, length));
                    break;
                }
            }
        }
        best
    }
}

fn char_suffix(s: &str, n_chars: usize) -> String {
    let total = s.chars().count();
    let skip = total.saturating_sub(n_chars);
    s.chars().skip(skip).collect()
}

fn char_prefix(s: &str, n_chars: usize) -> String {
    s.chars().take(n_chars).collect()
}

fn byte_len_of_char_suffix(s: &str, n_chars: usize) -> usize {
    char_suffix(s, n_chars).len()
}
