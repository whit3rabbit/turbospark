//! Incremental stop-string matcher across streaming text chunks. Ported from
//! `Runtime/Generation/StreamingStopMatcher.swift`.

/// Holds back the longest suffix of accumulated text that could still be the
/// start of one of `stops`, releasing it once it stops being a viable
/// prefix, so a stop string split across two `push` calls is still caught.
pub struct StreamingStopMatcher {
    stops: Vec<String>,
    /// KMP failure table per stop, so the withheld-suffix scan is linear in
    /// pending plus stop length. Stops arrive uncapped from HTTP `stop`
    /// fields, and the old per-length rebuild was quadratic per push.
    failures: Vec<Vec<usize>>,
    pending: String,
    stopped: bool,
}

impl StreamingStopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let failures = stops.iter().map(|s| failure_table(s.as_bytes())).collect();
        Self {
            stops,
            failures,
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
        let text = self.pending.as_bytes();
        let mut best = 0usize;
        for (stop, fail) in self.stops.iter().zip(&self.failures) {
            let pat = stop.as_bytes();
            // Matched length of the longest stop prefix ending at each byte.
            // A byte match starting at the stop's first byte starts on a char
            // boundary of pending, so the byte count is a valid split point.
            let mut k = 0usize;
            for &byte in text {
                while k > 0 && (k == pat.len() || pat[k] != byte) {
                    k = fail[k - 1];
                }
                if pat[k] == byte {
                    k += 1;
                }
            }
            if k == pat.len() {
                k = fail[k - 1];
            }
            best = best.max(k);
        }
        best
    }
}

/// Standard KMP prefix function: `fail[i]` is the length of the longest
/// proper prefix of `pat[..=i]` that is also its suffix.
fn failure_table(pat: &[u8]) -> Vec<usize> {
    let mut fail = vec![0usize; pat.len()];
    let mut k = 0usize;
    for i in 1..pat.len() {
        while k > 0 && pat[i] != pat[k] {
            k = fail[k - 1];
        }
        if pat[i] == pat[k] {
            k += 1;
        }
        fail[i] = k;
    }
    fail
}
