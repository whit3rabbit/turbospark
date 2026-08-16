//! Per-dispatch GPU timing, opt-in via `MFERENCE_DISPATCH_PROFILE=1`.
//!
//! `MFERENCE_PHASES=1` attributes GPU busy time per COMMAND BUFFER
//! (`GPUStartTime`/`GPUEndTime`). That granularity is what once pointed
//! this port's attention work at the wrong kernel: cb1 is ~25 dispatches
//! and a per-buffer number says nothing about which of them owns the time
//! (see AGENTS.md Gotcha 20 and DEVIATIONS.md's split-KV entry). This
//! module gives the next level down: one `(command buffer, kernel)` row
//! per dispatch kind, ranked.
//!
//! How, and what it costs: this device (and every Apple GPU) reports
//! `supportsCounterSampling(.atStageBoundary) == true` and
//! `.atDispatchBoundary == false`, so counters cannot be sampled BETWEEN
//! dispatches inside one encoder. The only way down to per-dispatch
//! resolution is one compute encoder per dispatch, each carrying
//! start-of-encoder and end-of-encoder timestamps. That is exactly what
//! profiling mode does, and it is why the mode is opt-in:
//!
//! - encoder boundaries add real per-dispatch overhead, so the absolute
//!   numbers are INFLATED against `MFERENCE_PHASES`'s buffer totals. The
//!   report is for RANKING dispatches against each other, never for
//!   claiming a kernel costs N ms in production.
//! - profiling waits on every command buffer at commit so its samples can
//!   be resolved, which serializes the host/GPU overlap the decode path is
//!   built around. Throughput under profiling is not a throughput number.
//!
//! Timestamps come back in GPU ticks; `scale_ns_per_tick` calibrates them
//! against the CPU clock once, at the first profiled pass.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use metal::{
    CounterSampleBuffer, CounterSampleBufferDescriptor, Device, MTLStorageMode, NSRange, NSUInteger,
};

/// Samples per profiled command buffer (two per dispatch). cb1 encodes
/// ~25 dispatches on Gemma 4; anything past this stays unsampled and is
/// counted under `UNSAMPLED` so an overflow shows up in the report rather
/// than silently truncating the ranking.
const MAX_SAMPLES: u64 = 256;

/// Sentinel Metal writes for a sample it could not resolve.
const COUNTER_ERROR_VALUE: u64 = u64::MAX;

const UNSAMPLED: &str = "(over sample capacity)";

// Counter sample buffers are a scarce device resource, not ordinary
// memory: allocating a fresh one per command buffer fails almost
// immediately (measured on an M4 Max: ~2000 failures inside 24 forward
// passes, i.e. nearly every pass silently unprofiled). They are recycled
// through this free list instead. Decode is single-threaded, so a
// thread-local needs no locking and sidesteps whether Metal's handles
// are `Send`.
thread_local! {
    static POOL: std::cell::RefCell<Vec<CounterSampleBuffer>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Passes that could not get a sample buffer at all, so a run that only
/// partly profiled says so instead of quietly under-counting.
static UNPROFILED_PASSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Returns true if GPU dispatch profiling is enabled via `MFERENCE_DISPATCH_PROFILE=1`.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MFERENCE_DISPATCH_PROFILE").as_deref() == Ok("1"))
}

/// `pipeline pointer -> kernel function name`, so a dispatch can be named
/// without threading a label through all ~25 `encode_*` call sites.
/// Pipelines are cached and handed out as retained clones, so the pointer
/// is stable for the life of the `MetalContext`.
fn names() -> &'static Mutex<BTreeMap<usize, &'static str>> {
    static NAMES: OnceLock<Mutex<BTreeMap<usize, &'static str>>> = OnceLock::new();
    NAMES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

type Totals = BTreeMap<(&'static str, &'static str), (u64, u64)>;

/// `(command buffer label, kernel) -> (dispatch count, GPU ticks)`.
fn totals() -> &'static Mutex<Totals> {
    static TOTALS: OnceLock<Mutex<Totals>> = OnceLock::new();
    TOTALS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Nanoseconds per GPU timestamp tick. Calibrated once by sampling the
/// paired CPU/GPU clocks around a short sleep. On Apple silicon the CPU
/// timestamp is already nanoseconds and the ratio comes out ~1.0; the
/// report prints it so a device where that does not hold is visible
/// rather than silently mis-scaled.
fn scale_ns_per_tick(device: &Device) -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        let (mut cpu0, mut gpu0) = (0u64, 0u64);
        let (mut cpu1, mut gpu1) = (0u64, 0u64);
        device.sample_timestamps(&mut cpu0, &mut gpu0);
        std::thread::sleep(std::time::Duration::from_millis(20));
        device.sample_timestamps(&mut cpu1, &mut gpu1);
        let gpu_delta = gpu1.saturating_sub(gpu0);
        if gpu_delta == 0 {
            return 1.0;
        }
        cpu1.saturating_sub(cpu0) as f64 / gpu_delta as f64
    })
}

/// The Objective-C object address behind a pipeline handle, which is what
/// identifies a cached pipeline across its retained clones.
fn pipeline_key(pipeline: &metal::ComputePipelineStateRef) -> usize {
    pipeline as *const metal::ComputePipelineStateRef as usize
}

/// Registers the human-readable function name for a compute pipeline state.
pub fn register_pipeline(pipeline: &metal::ComputePipelineState, function_name: &'static str) {
    if !enabled() {
        return;
    }
    names()
        .lock()
        .expect("names")
        .insert(pipeline_key(pipeline), function_name);
}

fn pipeline_name(pipeline: &metal::ComputePipelineStateRef) -> &'static str {
    names()
        .lock()
        .expect("names")
        .get(&pipeline_key(pipeline))
        .copied()
        .unwrap_or("(unregistered pipeline)")
}

/// The sampling state of one profiled command buffer.
pub struct PassProfile {
    cb_label: &'static str,
    sample_buffer: CounterSampleBuffer,
    /// One `(command buffer label, kernel)` per sampled dispatch, in
    /// encode order; index `i` owns samples `2i` and `2i + 1`. The label
    /// is captured per dispatch, not per pass, because a pass can be
    /// relabelled partway (see `relabel`).
    kernels: Vec<(&'static str, &'static str)>,
    overflow: u64,
    scale: f64,
    /// Buffers declared via `use_resource`. Each new per-dispatch encoder
    /// starts with empty resource state, so these get re-declared on every
    /// one of them (the MoE expert blobs reach the kernel only through an
    /// argument buffer and are invisible to Metal otherwise).
    used_reads: Vec<metal::Buffer>,
}

fn new_sample_buffer(device: &Device) -> Option<CounterSampleBuffer> {
    let counter_set = device
        .counter_sets()
        .into_iter()
        .find(|set| set.name() == "timestamp")?;
    let descriptor = CounterSampleBufferDescriptor::new();
    descriptor.set_counter_set(&counter_set);
    descriptor.set_sample_count(MAX_SAMPLES);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    device
        .new_counter_sample_buffer_with_descriptor(&descriptor)
        .ok()
}

impl PassProfile {
    /// Returns `None` if the device exposes no timestamp counter set or
    /// has no sample buffer to spare, which is counted and reported:
    /// profiling is a debugging aid, never a reason to fail a decode.
    pub fn new(device: &Device, cb_label: &'static str) -> Option<Self> {
        let Some(sample_buffer) = POOL
            .with_borrow_mut(|pool| pool.pop())
            .or_else(|| new_sample_buffer(device))
        else {
            UNPROFILED_PASSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        };
        Some(Self {
            cb_label,
            sample_buffer,
            kernels: Vec::new(),
            overflow: 0,
            scale: scale_ns_per_tick(device),
            used_reads: Vec::new(),
        })
    }

    /// Renames the buffer for dispatches encoded from here on; ones
    /// already sampled keep the label they were encoded under.
    pub fn relabel(&mut self, cb_label: &'static str) {
        self.cb_label = cb_label;
    }

    /// Records a buffer resource that must be declared as used for reading on each per-dispatch encoder.
    pub fn note_used_read(&mut self, buffer: &metal::Buffer) {
        self.used_reads.push(buffer.clone());
    }

    /// Ends the encoder currently in `slot` and installs a fresh one that
    /// timestamps the single dispatch about to be encoded into it.
    pub fn begin_dispatch(
        &mut self,
        command_buffer: &metal::CommandBufferRef,
        slot: &mut metal::ComputeCommandEncoder,
        pipeline: &metal::ComputePipelineStateRef,
    ) {
        slot.end_encoding();
        let next = self.kernels.len() as u64 * 2;
        let encoder = if next + 2 <= MAX_SAMPLES {
            let descriptor = metal::ComputePassDescriptor::new();
            let attachment = descriptor
                .sample_buffer_attachments()
                .object_at(0)
                .expect("compute pass sample buffer attachment 0");
            attachment.set_sample_buffer(&self.sample_buffer);
            attachment.set_start_of_encoder_sample_index(next as NSUInteger);
            attachment.set_end_of_encoder_sample_index((next + 1) as NSUInteger);
            self.kernels.push((self.cb_label, pipeline_name(pipeline)));
            command_buffer
                .compute_command_encoder_with_descriptor(descriptor)
                .to_owned()
        } else {
            self.overflow += 1;
            command_buffer.new_compute_command_encoder().to_owned()
        };
        for buffer in &self.used_reads {
            encoder.use_resource(buffer, metal::MTLResourceUsage::Read);
        }
        *slot = encoder;
    }

    /// Resolves this buffer's timestamps into the process-wide totals.
    /// Only valid once the command buffer has completed.
    pub fn resolve(self) {
        let samples = resolve_timestamps(&self.sample_buffer, self.kernels.len() * 2);
        let mut totals = totals().lock().expect("totals");
        for (index, &(cb_label, kernel)) in self.kernels.iter().enumerate() {
            let (start, end) = (samples[index * 2], samples[index * 2 + 1]);
            if start == COUNTER_ERROR_VALUE || end == COUNTER_ERROR_VALUE || end < start {
                continue;
            }
            let nanos = ((end - start) as f64 * self.scale) as u64;
            let entry = totals.entry((cb_label, kernel)).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += nanos;
        }
        if self.overflow > 0 {
            let entry = totals.entry((self.cb_label, UNSAMPLED)).or_insert((0, 0));
            entry.0 += self.overflow;
        }
        POOL.with_borrow_mut(|pool| pool.push(self.sample_buffer));
    }
}

/// `resolveCounterRange:` into a `Vec` of `MTLCounterResultTimestamp`
/// (one `u64` each). metal-rs 0.33 binds this, but its binding computes
/// the copy length from an empty `Vec`, so it copies zero bytes and then
/// `set_len`s over uninitialized memory; hence the raw message send.
// The allow is for objc's `sel_impl!`, whose expansion carries a
// `cfg(feature = "cargo-clippy")` this crate does not declare.
#[allow(unexpected_cfgs)]
fn resolve_timestamps(sample_buffer: &CounterSampleBuffer, count: usize) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }
    let mut out = vec![COUNTER_ERROR_VALUE; count];
    let range = NSRange::new(0, count as NSUInteger);
    // SAFETY: `resolveCounterRange:` returns an autoreleased `NSData`
    // holding `count` timestamps; `getBytes:length:` copies at most the
    // byte length asked for into our fully initialized buffer.
    #[allow(unsafe_code)]
    unsafe {
        use metal::objc::runtime::Object;
        use metal::objc::{msg_send, sel, sel_impl};
        let sample_buffer: &metal::CounterSampleBufferRef = sample_buffer;
        let data: *mut Object = msg_send![sample_buffer, resolveCounterRange: range];
        if data.is_null() {
            return out;
        }
        let available: NSUInteger = msg_send![data, length];
        let wanted = (count * std::mem::size_of::<u64>()) as NSUInteger;
        let bytes = available.min(wanted);
        let () = msg_send![data, getBytes: out.as_mut_ptr() length: bytes];
    }
    out
}

/// The report, or `None` when profiling is off or nothing was sampled.
/// `calls` is the number of forward passes the totals cover, so every row
/// reads per token.
pub fn report(calls: u64) -> Option<String> {
    if !enabled() || calls == 0 {
        return None;
    }
    let totals = totals().lock().expect("totals");
    if totals.is_empty() {
        return None;
    }
    let mut per_cb: BTreeMap<&'static str, Vec<(&'static str, u64, u64)>> = BTreeMap::new();
    for (&(cb_label, kernel), &(count, nanos)) in totals.iter() {
        per_cb
            .entry(cb_label)
            .or_default()
            .push((kernel, count, nanos));
    }
    let unprofiled = UNPROFILED_PASSES.load(std::sync::atomic::Ordering::Relaxed);
    let mut out = String::new();
    out.push_str(&format!(
        "[dispatch profile over {calls} forward passes -- one encoder per dispatch,\n\
         \x20every buffer waited on: absolute times are INFLATED, rank only]\n"
    ));
    if unprofiled > 0 {
        out.push_str(&format!(
            "  WARNING: {unprofiled} command buffers ran unprofiled (no sample\n             \x20buffer available); the rows below undercount by that much\n"
        ));
    }
    for (cb_label, mut rows) in per_cb {
        rows.sort_by_key(|row| std::cmp::Reverse(row.2));
        let cb_total: u64 = rows.iter().map(|row| row.2).sum();
        out.push_str(&format!(
            "  {cb_label}: {:.3} ms/token over {:.1} dispatches/token\n",
            cb_total as f64 / 1e6 / calls as f64,
            rows.iter().map(|row| row.1).sum::<u64>() as f64 / calls as f64,
        ));
        for (kernel, count, nanos) in rows {
            out.push_str(&format!(
                "    {kernel:<34} {:>5.1} x  {:>7.3} ms/token  {:>5.1}%\n",
                count as f64 / calls as f64,
                nanos as f64 / 1e6 / calls as f64,
                100.0 * nanos as f64 / cb_total.max(1) as f64,
            ));
        }
    }
    Some(out)
}

/// Drops every accumulated sample. Tests use it to isolate one pass.
pub fn reset() {
    totals().lock().expect("totals").clear();
    UNPROFILED_PASSES.store(0, std::sync::atomic::Ordering::Relaxed);
}
