//! Bounded batch-local detail, separate from the legacy unbounded trace.
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct BatchCapture {
    version: u32,
    pub(super) truncated: bool,
    pub(super) rows: Vec<BatchRow>,
    #[serde(skip)]
    limit: usize,
    #[serde(skip)]
    used: usize,
    #[serde(skip)]
    request: u64,
    #[serde(skip)]
    batch: u64,
    #[serde(skip)]
    scope: Option<(&'static str, usize, usize)>,
}

#[derive(Serialize)]
pub(super) struct BatchRow {
    request: u64,
    batch: u64,
    phase: &'static str,
    layer: usize,
    token_start: usize,
    token_count: usize,
    trace_start: usize,
    trace_end: usize,
    routes: Vec<Vec<usize>>,
    completed: bool,
    cache_hits: usize,
    cache_misses: usize,
    bytes_requested: u64,
    io_samples: usize,
}

impl BatchCapture {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            version: 1,
            truncated: false,
            rows: Vec::new(),
            limit,
            used: 0,
            request: 0,
            batch: 0,
            scope: None,
        }
    }

    pub(super) fn begin(&mut self, phase: &'static str, start: usize, count: usize) {
        self.end(false);
        self.batch += 1;
        self.scope = Some((phase, start, count));
    }

    pub(super) fn end(&mut self, completed: bool) {
        if self.scope.take().is_none() {
            return;
        }
        for row in self
            .rows
            .iter_mut()
            .rev()
            .take_while(|r| r.batch == self.batch)
        {
            row.completed = completed && row.routes.len() == row.token_count;
        }
        self.scope = None;
    }

    pub(super) fn reset(&mut self) {
        self.end(false);
        self.request += 1;
    }

    pub(super) fn record(&mut self, layer: usize, offset: usize, routes: &[usize]) {
        let Some((phase, start, count)) = self.scope else {
            return;
        };
        if self.used.saturating_add(routes.len()) > self.limit || routes.is_empty() {
            self.truncated = true;
            return;
        }
        self.used += routes.len();
        // Layers are visited in order within a batch. Avoid allocating a
        // second route table when only the legacy histogram was requested.
        let needs_row = match self.rows.last() {
            Some(row) => row.batch != self.batch || row.layer != layer,
            None => true,
        };
        if needs_row {
            self.rows.push(BatchRow {
                request: self.request,
                batch: self.batch,
                phase,
                layer,
                token_start: start,
                token_count: count,
                trace_start: offset,
                trace_end: offset,
                routes: Vec::new(),
                completed: false,
                cache_hits: 0,
                cache_misses: 0,
                bytes_requested: 0,
                io_samples: 0,
            });
        }
        let row = self.rows.last_mut().expect("created row");
        row.routes.push(routes.to_vec());
        row.trace_end = offset + routes.len();
    }

    pub(super) fn io(&mut self, layer: usize, hits: usize, misses: usize, bytes: u64) {
        if let Some(row) = self
            .rows
            .last_mut()
            .filter(|r| r.batch == self.batch && r.layer == layer)
        {
            row.cache_hits += hits;
            row.cache_misses += misses;
            row.bytes_requested += bytes;
            row.io_samples += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundaries_resets_tails_and_io_are_explicit() {
        let mut c = BatchCapture::new(100);
        c.begin("prefill", 0, 2);
        c.record(3, 0, &[2, 1]);
        c.io(3, 1, 1, 32);
        c.record(3, 2, &[1, 0]);
        c.io(3, 2, 0, 0);
        c.end(true);
        c.begin("prefill", 2, 1);
        c.record(3, 4, &[0, 2]);
        c.end(true);
        c.begin("decode", 3, 1);
        c.record(3, 6, &[1, 0]);
        c.end(true);
        c.reset();
        c.begin("prefill", 0, 1);
        c.record(3, 8, &[0, 1]);
        c.end(false);
        assert_eq!(c.rows.len(), 4);
        assert!(c.rows[0].completed);
        assert_eq!(c.rows[0].trace_end, 4);
        assert_eq!(c.rows[0].cache_hits, 3);
        assert_eq!(c.rows[0].bytes_requested, 32);
        assert_eq!(c.rows[1].token_start, 2);
        assert_eq!(c.rows[2].phase, "decode");
        assert_eq!(c.rows[3].request, 1);
        assert!(!c.rows[3].completed);
    }
    #[test]
    fn limit_never_marks_partial_batches_complete() {
        let mut c = BatchCapture::new(3);
        c.begin("prefill", 0, 2);
        c.record(0, 0, &[1, 2]);
        c.record(0, 2, &[0, 1]);
        c.end(true);
        assert!(c.truncated);
        assert_eq!(c.used, 2);
        assert!(!c.rows[0].completed);
        c.begin("decode", 2, 1);
        c.record(0, 4, &[0, 1]);
        c.end(true);
        assert_eq!(c.rows.len(), 1);
    }
}
