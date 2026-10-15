//! Opt-in per-layer timing of the decode `cb1` command buffers
//! (`TURBOSPARK_CB1_TRACE=<path>`).
//!
//! `TURBOSPARK_PHASES=1` reports `cb1` as two totals: wall time spent in
//! `commit()` plus `wait` (`gpu_wait_nanos`) and the GPU-reported busy time
//! (`GPUEndTime - GPUStartTime`). Over 48 layers a total cannot tell the
//! candidate causes of a slow `cb1` apart, so this writes one TSV row per
//! layer buffer with the pieces separated:
//!
//! - `commit_ns` is the time inside `commit()` alone. A driver stall while it
//!   wires a no-copy mapping lands here and NOT in `gpu_busy_ns`.
//! - `wait_ns` is the time from `commit()` returning until the wait returns.
//! - `gpu_busy_ns` is the GPU-reported busy time of that buffer.
//! - `idle_before_ns` is the time since the previous layer's wait returned,
//!   which bounds how long the GPU sat idle while the CPU read experts and
//!   encoded. It is 0 on the first layer of a pass: the gap there includes
//!   sampling and the logits readback, which is a different thing.
//! - `expert_io_ns` is the expert `pread` time added since the previous row.
//! - `ple_encode_ns` is the host-side n-gram lookup and upload for the pass,
//!   written on the PLE layer's row and 0 elsewhere.
//!
//! WHAT A `cb1` HOLDS: this family does not commit the routed work of layer
//! L separately. It is encoded into the same pass, which becomes layer L+1's
//! `cb1`. So row L+1's `gpu_busy_ns` is the previous layer's routed expert
//! compute plus this layer's attention, hyper-connections and router. Read
//! per-layer comparisons with that in mind.
//!
//! Only the decode path (`produce.rs`) writes rows. Chunked prefill does not.
//! Disabled, the cost is one `OnceLock` read per pass.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// The environment variable that names the trace file.
const TRACE_ENV: &str = "TURBOSPARK_CB1_TRACE";

/// Column names, in the order [`format_row`] writes them.
const COLUMNS: [&str; 9] = [
    "pass",
    "position",
    "layer",
    "commit_ns",
    "wait_ns",
    "gpu_busy_ns",
    "idle_before_ns",
    "expert_io_ns",
    "ple_encode_ns",
];

/// One layer's `cb1` timing. All durations are nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Cb1Row {
    pub pass: u64,
    pub position: usize,
    pub layer: usize,
    pub commit_ns: u64,
    pub wait_ns: u64,
    pub gpu_busy_ns: u64,
    pub idle_before_ns: u64,
    pub expert_io_ns: u64,
    pub ple_encode_ns: u64,
}

fn header() -> String {
    let mut line = COLUMNS.join("\t");
    line.push('\n');
    line
}

fn format_row(row: &Cb1Row) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        row.pass,
        row.position,
        row.layer,
        row.commit_ns,
        row.wait_ns,
        row.gpu_busy_ns,
        row.idle_before_ns,
        row.expert_io_ns,
        row.ple_encode_ns,
    )
}

/// The trace path, or `None` when the variable is unset or empty. An empty
/// value is treated as unset so `TURBOSPARK_CB1_TRACE= cmd` turns it off.
fn trace_path(value: Option<std::ffi::OsString>) -> Option<PathBuf> {
    value.filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// Opens the trace file once per process. A file that cannot be opened is
/// reported once on stderr and then treated as disabled, so a bad path never
/// fails a decode.
fn sink() -> Option<&'static Mutex<File>> {
    static SINK: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    SINK.get_or_init(|| {
        let path = trace_path(std::env::var_os(TRACE_ENV))?;
        let opened = OpenOptions::new().create(true).append(true).open(&path);
        let mut file = match opened {
            Ok(file) => file,
            Err(err) => {
                eprintln!("{TRACE_ENV}: cannot open {}: {err}", path.display());
                return None;
            }
        };
        // Append mode lets several runs share a file. Only the run that finds
        // it empty writes the header.
        if file.metadata().map(|m| m.len() == 0).unwrap_or(false) {
            if let Err(err) = file.write_all(header().as_bytes()) {
                eprintln!("{TRACE_ENV}: cannot write {}: {err}", path.display());
                return None;
            }
        }
        Some(Mutex::new(file))
    })
    .as_ref()
}

/// True when `TURBOSPARK_CB1_TRACE` names a file that opened.
pub(super) fn enabled() -> bool {
    sink().is_some()
}

fn write_row(row: &Cb1Row) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    let Some(sink) = sink() else { return };
    let Ok(mut file) = sink.lock() else { return };
    // A failed write must not pass as a complete trace, so say so once.
    if file.write_all(format_row(row).as_bytes()).is_err() && !WARNED.swap(true, Ordering::Relaxed)
    {
        eprintln!("{TRACE_ENV}: a row write failed; the trace is incomplete");
    }
}

/// Per-pass state: the clock of the previous layer's wait and the
/// cumulative expert-io counter at the previous row.
pub(super) struct PassTrace {
    enabled: bool,
    pass: u64,
    position: usize,
    prev_wait_end: Option<Instant>,
    prev_expert_io_ns: u64,
    ple_encode_ns: u64,
}

impl PassTrace {
    /// `pass` is the 0-based forward-pass index, `expert_io_ns` the cumulative
    /// `PhaseCounters::expert_io_nanos` when the pass starts.
    pub(super) fn start(pass: u64, position: usize, expert_io_ns: u64) -> Self {
        Self {
            enabled: enabled(),
            pass,
            position,
            prev_wait_end: None,
            prev_expert_io_ns: expert_io_ns,
            ple_encode_ns: 0,
        }
    }

    /// Starts a timer for the PLE encode, or `None` when tracing is off.
    pub(super) fn ple_timer(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    /// Records the PLE encode time started by [`Self::ple_timer`].
    pub(super) fn note_ple(&mut self, started: Option<Instant>) {
        if let Some(started) = started {
            self.ple_encode_ns = started.elapsed().as_nanos() as u64;
        }
    }

    /// Writes the row for one layer. `t_start` is taken before `commit()`,
    /// `t_committed` right after it returns, `t_end` right after the wait.
    pub(super) fn record_layer(
        &mut self,
        layer: usize,
        t_start: Instant,
        t_committed: Instant,
        t_end: Instant,
        gpu_busy_secs: f64,
        expert_io_ns: u64,
    ) {
        if !self.enabled {
            return;
        }
        let row = self.next_row(
            layer,
            t_start,
            t_committed,
            t_end,
            gpu_busy_secs,
            expert_io_ns,
        );
        write_row(&row);
    }

    /// Builds one layer's row and advances the per-pass state.
    fn next_row(
        &mut self,
        layer: usize,
        t_start: Instant,
        t_committed: Instant,
        t_end: Instant,
        gpu_busy_secs: f64,
        expert_io_ns: u64,
    ) -> Cb1Row {
        let idle_before_ns = self
            .prev_wait_end
            .map(|prev| t_start.saturating_duration_since(prev).as_nanos() as u64)
            .unwrap_or(0);
        let row = Cb1Row {
            pass: self.pass,
            position: self.position,
            layer,
            commit_ns: t_committed.saturating_duration_since(t_start).as_nanos() as u64,
            wait_ns: t_end.saturating_duration_since(t_committed).as_nanos() as u64,
            gpu_busy_ns: (gpu_busy_secs * 1e9) as u64,
            idle_before_ns,
            expert_io_ns: expert_io_ns.saturating_sub(self.prev_expert_io_ns),
            // The PLE encode runs inside the PLE layer's iteration, before
            // that layer's commit, so the next row is the one it belongs to.
            // Taking it clears it for every later layer.
            ple_encode_ns: std::mem::take(&mut self.ple_encode_ns),
        };
        self.prev_wait_end = Some(t_end);
        self.prev_expert_io_ns = expert_io_ns;
        row
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::time::Duration;

    use super::*;

    fn row() -> Cb1Row {
        Cb1Row {
            pass: 7,
            position: 61,
            layer: 3,
            commit_ns: 11,
            wait_ns: 22,
            gpu_busy_ns: 33,
            idle_before_ns: 44,
            expert_io_ns: 55,
            ple_encode_ns: 66,
        }
    }

    #[test]
    fn header_and_row_have_the_same_columns_in_order() {
        let head = header();
        let line = format_row(&row());
        assert_eq!(head.trim_end().split('\t').count(), COLUMNS.len());
        assert_eq!(line.trim_end().split('\t').count(), COLUMNS.len());
        // Distinct values pin each field to its column, so a dropped or
        // swapped column cannot pass on the count alone.
        assert_eq!(line, "7\t61\t3\t11\t22\t33\t44\t55\t66\n");
        assert!(head.starts_with("pass\tposition\tlayer\tcommit_ns\t"));
        assert!(head.ends_with("\tple_encode_ns\n"));
    }

    #[test]
    fn empty_or_unset_variable_disables_the_trace() {
        assert_eq!(trace_path(None), None);
        assert_eq!(trace_path(Some(OsString::new())), None);
        assert_eq!(
            trace_path(Some(OsString::from("/tmp/x.tsv"))),
            Some(PathBuf::from("/tmp/x.tsv"))
        );
    }

    /// Builds a pass trace with tracing forced on, without the process-wide
    /// sink, to check the row arithmetic.
    fn forced(pass: u64, io: u64) -> PassTrace {
        PassTrace {
            enabled: true,
            pass,
            position: 5,
            prev_wait_end: None,
            prev_expert_io_ns: io,
            ple_encode_ns: 9,
        }
    }

    #[test]
    fn rows_carry_split_times_idle_and_io_deltas() {
        let base = Instant::now();
        let ms = |n: u64| base + Duration::from_millis(n);
        let mut trace = forced(2, 1_000);

        // First layer: no previous wait, so idle is 0. The io delta is taken
        // against the counter at pass start. The pending PLE time (9) lands
        // on this row because it is the next one written.
        let first = trace.next_row(0, ms(0), ms(1), ms(4), 0.002, 1_500);
        assert_eq!(
            first,
            Cb1Row {
                pass: 2,
                position: 5,
                layer: 0,
                commit_ns: 1_000_000,
                wait_ns: 3_000_000,
                gpu_busy_ns: 2_000_000,
                idle_before_ns: 0,
                expert_io_ns: 500,
                ple_encode_ns: 9,
            }
        );

        // Second layer: idle is start minus the previous wait return, io is
        // the delta from the previous row, and PLE is cleared.
        let second = trace.next_row(1, ms(10), ms(10), ms(13), 0.001, 2_000);
        assert_eq!(second.idle_before_ns, 6_000_000);
        assert_eq!(second.expert_io_ns, 500);
        assert_eq!(second.commit_ns, 0);
        assert_eq!(second.ple_encode_ns, 0);

        // A new pass starts from zero idle again.
        assert_eq!(PassTrace::start(3, 6, 2_000).prev_wait_end, None);
    }

    #[test]
    fn disabled_trace_records_nothing_and_keeps_no_state() {
        let base = Instant::now();
        let mut trace = forced(0, 0);
        trace.enabled = false;
        trace.record_layer(0, base, base, base, 0.5, 99);
        assert_eq!(trace.prev_wait_end, None);
        assert_eq!(trace.prev_expert_io_ns, 0);
        assert!(trace.ple_timer().is_none());
    }
}
