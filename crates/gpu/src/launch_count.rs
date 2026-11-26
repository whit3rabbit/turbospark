//! Static launch counting, off unless `TURBOSPARK_LAUNCH_COUNT=1`.
//!
//! Counts what the host SUBMITS, so a launch-count reduction can be priced
//! before it is built (count times the dispatch probe's cost per dispatch).
//! Unlike `dispatch_profile` it samples no GPU counters, opens no extra
//! encoder and never waits, so the stream it counts is the one that ships.
//! It does take a lock per dispatch: a counted run is not a timing run.
//!
//! The counters are process-wide (every `MetalContext` and thread), so a
//! caller brackets the work it wants with two [`take`] calls on an otherwise
//! idle process. `MetalContext::wait_until_idle`'s empty fence buffer is not
//! counted: it is a host synchronization, not part of any encoded step.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// What was submitted since the previous [`take`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LaunchCounts {
    /// Compute dispatches encoded.
    pub dispatches: u64,
    /// Command buffers created for compute passes, one-shot dispatches and
    /// state blits.
    pub command_buffers: u64,
    /// Distinct pipelines dispatched, `function#hash`, sorted. The hash is of
    /// the function-constant fingerprint, stable across processes.
    pub routes: Vec<String>,
}

/// `TURBOSPARK_LAUNCH_COUNT=1`, read once.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_LAUNCH_COUNT").as_deref() == Ok("1"))
}

#[derive(Default)]
struct State {
    /// Pipeline object address to route name. A cached pipeline keeps its
    /// address, and a reused address is re-registered at creation.
    names: HashMap<usize, String>,
    dispatches: u64,
    command_buffers: u64,
    routes: BTreeSet<String>,
}

fn state() -> MutexGuard<'static, State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE
        .get_or_init(Mutex::default)
        .lock()
        .expect("launch count state")
}

fn key(pipeline: &metal::ComputePipelineStateRef) -> usize {
    pipeline as *const metal::ComputePipelineStateRef as usize
}

/// 32-bit FNV-1a: short, and the same in every build, unlike `DefaultHasher`.
fn short_hash(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |h, &b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    })
}

/// Returns the counts since the previous call and resets them. All zero when
/// counting is off.
pub fn take() -> LaunchCounts {
    if !enabled() {
        return LaunchCounts::default();
    }
    let mut s = state();
    LaunchCounts {
        dispatches: std::mem::take(&mut s.dispatches),
        command_buffers: std::mem::take(&mut s.command_buffers),
        routes: std::mem::take(&mut s.routes).into_iter().collect(),
    }
}

pub(crate) fn register_pipeline(
    pipeline: &metal::ComputePipelineStateRef,
    function_name: &str,
    constants_key: &[u8],
) {
    if !enabled() {
        return;
    }
    let route = format!("{function_name}#{:08x}", short_hash(constants_key));
    state().names.insert(key(pipeline), route);
}

pub(crate) fn dispatch(pipeline: &metal::ComputePipelineStateRef) {
    if !enabled() {
        return;
    }
    let mut guard = state();
    let s = &mut *guard;
    s.dispatches += 1;
    let route = s
        .names
        .get(&key(pipeline))
        .map_or("(unregistered)", String::as_str);
    if !s.routes.contains(route) {
        s.routes.insert(route.to_string());
    }
}

pub(crate) fn command_buffer() {
    if !enabled() {
        return;
    }
    state().command_buffers += 1;
}
