//! Bounded keep-warm for resident weight buffers (KERN-05 Step 3,
//! `docs/designs/speedup-kern.md`). Default off: nothing starts one unless a
//! caller opts in.
//!
//! After about 1 to 2 s with no command buffer naming them, macOS drops the
//! GPU mappings of an install's large buffers, and the next command buffer
//! that names them waits in the kernel while they are remapped (150 to 260 ms
//! on Inkling 2 mlx4, about 47 ms on a dense 4.7 GB install). The cost is per
//! buffer named, not per page read, and naming every buffer in one tiny
//! dispatch more often than that keeps them mapped for about 1 ms per tick
//! (Step 2, 2026-11-27).
//!
//! [`KeepWarm`] runs that dispatch on its own thread and its own queue, so it
//! never touches the owner's queue order or scratch:
//!
//! - it fires only once a full interval has passed since the buffers were
//!   last named (by a pass on the owner's context or by its own tick), so a
//!   busy decode costs nothing;
//! - it stops once `window` has passed since the owner's last pass, which
//!   bounds the idle power to one window after each use;
//! - it skips every tick while the kernel reports memory pressure other than
//!   normal (or no reading), so a mapping the system wants back is given
//!   back.
//!
//! Each dispatch reads one 32-bit word of a buffer and writes only its own
//! scratch, so it cannot change decode output. The caller chooses the
//! buffers: never a mapped or streamed expert buffer, whose pages the
//! streaming path expects the system to be free to drop.

use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use metal::{ComputePipelineState, MTLSize};

use crate::context::{GpuError, MetalContext};

/// Accepted interval, ms. Step 2 measured 250 and 750 ms with the same
/// residual; past about 1 s the mappings drop between ticks.
pub const KEEP_WARM_EVERY_MS_RANGE: RangeInclusive<u64> = 50..=5000;
/// Accepted window, s.
pub const KEEP_WARM_WINDOW_S_RANGE: RangeInclusive<u64> = 1..=3600;
/// The window when none is given: what the PolyStrata audit ships (120 s
/// after the last command buffer).
pub const KEEP_WARM_DEFAULT_WINDOW_S: u64 = 120;

const SOURCE: &str = "
    #include <metal_stdlib>
    using namespace metal;
    // One thread. The compare keeps the load alive; the sentinel is never
    // expected to match, so `out` is normally left untouched.
    kernel void keep_warm_name(device const uint* data [[buffer(0)]],
                               device uint* out [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
        if (id == 0 && data[0] == 0x9E3779B9u) { out[0] = 1; }
    }
";

/// Milliseconds since a process-wide epoch. Monotonic; only differences mean
/// anything.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Interval and window, both in ms once parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepWarmConfig {
    pub every_ms: u64,
    pub window_ms: u64,
}

impl KeepWarmConfig {
    /// Parses the two knob values. `every` unset, empty, `0` or `off` means
    /// off (`Ok(None)`). Anything else must be an integer in
    /// [`KEEP_WARM_EVERY_MS_RANGE`]; `window_s`, when set, an integer in
    /// [`KEEP_WARM_WINDOW_S_RANGE`]. A window without an interval is refused,
    /// since it would silently do nothing.
    pub fn parse(every: Option<&str>, window_s: Option<&str>) -> Result<Option<Self>, String> {
        let every = every.map(str::trim).filter(|v| !v.is_empty());
        let window_s = window_s.map(str::trim).filter(|v| !v.is_empty());
        let every_ms = match every {
            None | Some("0") | Some("off") => {
                return match window_s {
                    None => Ok(None),
                    Some(w) => Err(format!(
                        "keep-warm window {w:?} given without an interval; set the interval in ms \
                         or unset the window"
                    )),
                };
            }
            Some(v) => v
                .parse::<u64>()
                .ok()
                .filter(|ms| KEEP_WARM_EVERY_MS_RANGE.contains(ms))
                .ok_or_else(|| {
                    format!(
                        "keep-warm interval {v:?} is not an integer in {}..={} ms",
                        KEEP_WARM_EVERY_MS_RANGE.start(),
                        KEEP_WARM_EVERY_MS_RANGE.end()
                    )
                })?,
        };
        let window_s = match window_s {
            None => KEEP_WARM_DEFAULT_WINDOW_S,
            Some(v) => v
                .parse::<u64>()
                .ok()
                .filter(|s| KEEP_WARM_WINDOW_S_RANGE.contains(s))
                .ok_or_else(|| {
                    format!(
                        "keep-warm window {v:?} is not an integer in {}..={} s",
                        KEEP_WARM_WINDOW_S_RANGE.start(),
                        KEEP_WARM_WINDOW_S_RANGE.end()
                    )
                })?,
        };
        Ok(Some(Self {
            every_ms,
            window_ms: window_s * 1000,
        }))
    }
}

/// What one poll of the keep-warm thread does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepWarmTick {
    /// The buffers were named less than one interval ago.
    Wait,
    /// The window since the owner's last pass is over.
    Expired,
    /// Memory pressure is not normal, or there is no reading.
    Paused,
    /// Name every buffer now.
    Fire,
}

/// The poll's decision. `since_pass_ms` is the time since the owner's last
/// pass, `since_named_ms` the time since the buffers were last named by
/// either side, `pressure_raw` a [`crate::memory_pressure_raw`] reading
/// (1 is normal).
pub fn keep_warm_tick(
    cfg: &KeepWarmConfig,
    since_pass_ms: u64,
    since_named_ms: u64,
    pressure_raw: i64,
) -> KeepWarmTick {
    if since_named_ms < cfg.every_ms {
        KeepWarmTick::Wait
    } else if since_pass_ms > cfg.window_ms {
        KeepWarmTick::Expired
    } else if pressure_raw != 1 {
        KeepWarmTick::Paused
    } else {
        KeepWarmTick::Fire
    }
}

/// Counters since start.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeepWarmStats {
    /// Committed and completed ticks.
    pub fired: u64,
    /// Polls skipped for memory pressure.
    pub paused: u64,
    /// Ticks whose command buffer did not complete.
    pub failed: u64,
}

struct Shared {
    stop: Mutex<bool>,
    wake: Condvar,
    fired: AtomicU64,
    paused: AtomicU64,
    failed: AtomicU64,
}

/// A running keep-warm thread. Dropping it stops and joins the thread, which
/// then releases its retained buffers.
pub struct KeepWarm {
    shared: Arc<Shared>,
    cfg: KeepWarmConfig,
    thread: Option<JoinHandle<()>>,
}

impl KeepWarm {
    /// Starts naming `buffers` under `cfg`, with `context`'s passes as the
    /// owner's activity. Each buffer must hold at least one 32-bit word.
    /// Compiles its one kernel here, before the thread starts.
    pub fn start(
        context: &MetalContext,
        buffers: Vec<metal::Buffer>,
        cfg: KeepWarmConfig,
    ) -> Result<Self, GpuError> {
        if buffers.is_empty() {
            return Err(GpuError::InvalidInput(
                "keep-warm needs at least one buffer".into(),
            ));
        }
        if let Some(b) = buffers.iter().find(|b| b.length() < 4) {
            return Err(GpuError::InvalidInput(format!(
                "keep-warm buffer of {} bytes holds no 32-bit word",
                b.length()
            )));
        }
        let device = context.device();
        let library = device
            .new_library_with_source(SOURCE, &metal::CompileOptions::new())
            .map_err(GpuError::LibraryCompile)?;
        let function = library
            .get_function("keep_warm_name", None)
            .map_err(|_| GpuError::FunctionNotFound("keep_warm_name".into()))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(GpuError::PipelineCreate)?;
        let queue = device.new_command_queue();
        let scratch = device.new_buffer(16, metal::MTLResourceOptions::StorageModeShared);
        let activity = context.activity_clock();
        let shared = Arc::new(Shared {
            stop: Mutex::new(false),
            wake: Condvar::new(),
            fired: AtomicU64::new(0),
            paused: AtomicU64::new(0),
            failed: AtomicU64::new(0),
        });
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("turbospark-keep-warm".into())
            .spawn(move || {
                run(
                    &worker, &activity, &queue, &pipeline, &scratch, &buffers, cfg,
                )
            })
            .map_err(|e| GpuError::InvalidInput(format!("keep-warm thread: {e}")))?;
        Ok(Self {
            shared,
            cfg,
            thread: Some(thread),
        })
    }

    pub fn config(&self) -> KeepWarmConfig {
        self.cfg
    }

    pub fn stats(&self) -> KeepWarmStats {
        KeepWarmStats {
            fired: self.shared.fired.load(Ordering::Relaxed),
            paused: self.shared.paused.load(Ordering::Relaxed),
            failed: self.shared.failed.load(Ordering::Relaxed),
        }
    }
}

impl Drop for KeepWarm {
    fn drop(&mut self) {
        if let Ok(mut stop) = self.shared.stop.lock() {
            *stop = true;
        }
        self.shared.wake.notify_all();
        if let Some(thread) = self.thread.take() {
            // A panicked worker has nothing left to release; the drop must
            // not panic in turn.
            let _ = thread.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    shared: &Shared,
    activity: &AtomicU64,
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    scratch: &metal::Buffer,
    buffers: &[metal::Buffer],
    cfg: KeepWarmConfig,
) {
    // Polling at a quarter interval keeps the longest unnamed stretch at
    // about 1.25 intervals; polling once per interval could leave two.
    let poll = Duration::from_millis((cfg.every_ms / 4).max(10));
    let mut last_fire = now_ms();
    let Ok(mut stop) = shared.stop.lock() else {
        return;
    };
    loop {
        // Checked before every wait too: a drop that lands before this thread
        // first waits sends its notify to no one.
        if *stop {
            return;
        }
        stop = match shared.wake.wait_timeout(stop, poll) {
            Ok((guard, _)) => guard,
            Err(_) => return,
        };
        if *stop {
            return;
        }
        let now = now_ms();
        let last_pass = activity.load(Ordering::Relaxed);
        let since_pass = now.saturating_sub(last_pass);
        let since_named = now.saturating_sub(last_pass.max(last_fire));
        // Only sample the sysctl when the answer matters.
        let pressure = if since_named >= cfg.every_ms && since_pass <= cfg.window_ms {
            crate::memory_pressure_raw()
        } else {
            1
        };
        match keep_warm_tick(&cfg, since_pass, since_named, pressure) {
            KeepWarmTick::Wait | KeepWarmTick::Expired => {}
            KeepWarmTick::Paused => {
                shared.paused.fetch_add(1, Ordering::Relaxed);
            }
            KeepWarmTick::Fire => {
                let ok = name_all(queue, pipeline, scratch, buffers);
                last_fire = now_ms();
                let counter = if ok { &shared.fired } else { &shared.failed };
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// One command buffer, one single-thread dispatch per buffer, waited on.
/// Returns whether it completed.
fn name_all(
    queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    scratch: &metal::Buffer,
    buffers: &[metal::Buffer],
) -> bool {
    crate::autorelease_pool(|| {
        let command_buffer = queue.new_command_buffer();
        command_buffer.set_label("keep-warm");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(1, Some(scratch), 0);
        let one = MTLSize::new(1, 1, 1);
        for b in buffers {
            encoder.set_buffer(0, Some(b), 0);
            encoder.dispatch_thread_groups(one, one);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        command_buffer.status() == metal::MTLCommandBufferStatus::Completed
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: KeepWarmConfig = KeepWarmConfig {
        every_ms: 750,
        window_ms: 120_000,
    };

    #[test]
    fn parse_is_off_unless_an_interval_is_given() {
        for v in [None, Some(""), Some(" "), Some("0"), Some("off")] {
            assert_eq!(KeepWarmConfig::parse(v, None), Ok(None), "{v:?}");
        }
    }

    #[test]
    fn parse_takes_a_bounded_interval_and_window() {
        assert_eq!(KeepWarmConfig::parse(Some("750"), None), Ok(Some(CFG)));
        assert_eq!(
            KeepWarmConfig::parse(Some("50"), Some("3")),
            Ok(Some(KeepWarmConfig {
                every_ms: 50,
                window_ms: 3000
            }))
        );
        assert!(KeepWarmConfig::parse(Some("5000"), Some("3600"))
            .unwrap()
            .is_some());
        for bad in ["49", "5001", "-1", "1.5", "on", "750ms"] {
            assert!(KeepWarmConfig::parse(Some(bad), None).is_err(), "{bad}");
        }
        for bad in ["0", "3601", "x"] {
            assert!(
                KeepWarmConfig::parse(Some("750"), Some(bad)).is_err(),
                "{bad}"
            );
        }
        assert!(KeepWarmConfig::parse(None, Some("60")).is_err());
        assert!(KeepWarmConfig::parse(Some("off"), Some("60")).is_err());
    }

    #[test]
    fn tick_waits_inside_an_interval_and_fires_only_in_window_without_pressure() {
        assert_eq!(keep_warm_tick(&CFG, 0, 749, 1), KeepWarmTick::Wait);
        assert_eq!(keep_warm_tick(&CFG, 750, 750, 1), KeepWarmTick::Fire);
        assert_eq!(keep_warm_tick(&CFG, 120_000, 750, 1), KeepWarmTick::Fire);
        assert_eq!(keep_warm_tick(&CFG, 120_001, 750, 1), KeepWarmTick::Expired);
        // An expired window never samples pressure, so it never counts as paused.
        assert_eq!(keep_warm_tick(&CFG, 120_001, 750, 2), KeepWarmTick::Expired);
        for p in [0, 2, 4] {
            assert_eq!(
                keep_warm_tick(&CFG, 1000, 1000, p),
                KeepWarmTick::Paused,
                "{p}"
            );
        }
        // Inside the interval nothing else is consulted.
        assert_eq!(keep_warm_tick(&CFG, 999_999, 10, 4), KeepWarmTick::Wait);
    }
}
