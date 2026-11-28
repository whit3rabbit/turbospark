//! Pass encoder and committed pass types (`PassEncoder`, `CommittedPass`).

use std::cell::RefCell;

use metal::{ComputePipelineState, MTLSize};

use crate::dispatch_profile::PassProfile;
use crate::launch_count;

/// Describes a dispatch that asks for a larger threadgroup than the compiled
/// pipeline allows on this device, or `None` when it fits. Metal reports
/// nothing for that: it skips the dispatch, the destination keeps its old
/// contents and decode turns into fluent garbage (a 448-thread limit against
/// a 512-thread request on an M2 Max, upstream report). The limit is per chip
/// and per compiled kernel, so only a run-time check sees it.
fn threadgroup_violation(
    pipeline: &metal::ComputePipelineStateRef,
    threads_per_group: (u64, u64, u64),
) -> Option<String> {
    let limit = pipeline.max_total_threads_per_threadgroup();
    let asked = threadgroup_threads(threads_per_group);
    (asked > limit).then(|| {
        format!(
            "dispatch of '{}' needs {asked} threads per threadgroup but this device allows {limit}",
            crate::dispatch_profile::pipeline_name(pipeline)
        )
    })
}

/// Panicking form of [`threadgroup_violation`] for the standalone dispatch
/// helpers, which return nothing. A pass records the violation instead (see
/// [`PassEncoder::commit_and_wait_checked`]).
pub(crate) fn assert_threadgroup_fits(
    pipeline: &metal::ComputePipelineStateRef,
    threads_per_group: (u64, u64, u64),
) {
    if let Some(violation) = threadgroup_violation(pipeline, threads_per_group) {
        panic!("{violation}");
    }
}

fn threadgroup_threads((x, y, z): (u64, u64, u64)) -> u64 {
    x.saturating_mul(y).saturating_mul(z)
}

/// An open command buffer + serial compute encoder; see
/// [`MetalContext::begin_pass`](super::MetalContext::begin_pass). Every `encode_*` call appends one
/// dispatch; nothing runs until [`PassEncoder::commit_and_wait`] (or
/// [`PassEncoder::commit`]).
pub struct PassEncoder {
    command_buffer: metal::CommandBuffer,
    /// Rebound per dispatch under `TURBOSPARK_DISPATCH_PROFILE=1` (this
    /// device can only sample counters at encoder boundaries), hence the
    /// cell; exactly one encoder for the whole pass otherwise.
    encoder: RefCell<metal::ComputeCommandEncoder>,
    profile: Option<RefCell<PassProfile>>,
    /// Whether `endEncoding` has been sent. Read by the `Drop` impl below,
    /// which is what stops an error path aborting the process.
    ended: std::cell::Cell<bool>,
    /// The first over-limit dispatch, which was skipped rather than encoded.
    /// Encode methods return nothing, so the pass carries it to the commit:
    /// the checked commit turns it into `Err` without committing, and the
    /// unchecked ones panic, because their callers would read stale output.
    violation: RefCell<Option<String>>,
}

impl PassEncoder {
    pub(crate) fn new(
        command_buffer: metal::CommandBuffer,
        encoder: RefCell<metal::ComputeCommandEncoder>,
        profile: Option<RefCell<PassProfile>>,
    ) -> Self {
        Self {
            command_buffer,
            encoder,
            profile,
            ended: std::cell::Cell::new(false),
            violation: RefCell::new(None),
        }
    }

    /// Records `threads_per_group` against the pipeline limit and reports
    /// whether the dispatch may be encoded. Only the first violation is kept,
    /// since it is the one that names the kernel at fault.
    fn admit(&self, pipeline: &ComputePipelineState, threads_per_group: (u64, u64, u64)) -> bool {
        let Some(violation) = threadgroup_violation(pipeline, threads_per_group) else {
            return true;
        };
        self.violation.borrow_mut().get_or_insert(violation);
        false
    }

    /// The unchecked commits' half of the guard: there is no `Result` to
    /// carry the violation, so stop before committing work with a hole in it.
    fn panic_on_violation(&self) {
        if let Some(violation) = self.violation.borrow().as_deref() {
            panic!("{violation}");
        }
    }

    /// Renames this command buffer mid-pass, for the case where one
    /// buffer's role changes partway (the final head rides whatever
    /// buffer the last layer left open). Affects labelling only.
    pub fn relabel(&self, label: &'static str) {
        self.command_buffer.set_label(label);
        if let Some(profile) = &self.profile {
            profile.borrow_mut().relabel(label);
        }
    }

    /// Counts the dispatch about to be encoded (`launch_count`) and, under
    /// profiling, closes the running encoder and opens a fresh timestamped
    /// one for it. Both are no-ops on the production path.
    fn begin_dispatch(&self, pipeline: &ComputePipelineState) {
        launch_count::dispatch(pipeline);
        let Some(profile) = &self.profile else {
            return;
        };
        profile.borrow_mut().begin_dispatch(
            &self.command_buffer,
            &mut self.encoder.borrow_mut(),
            pipeline,
        );
    }

    /// Appends one `dispatchThreadgroups` call. Buffer bindings are
    /// `(buffer, argument index, byte offset)`.
    pub fn encode_threadgroups(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[(&metal::Buffer, u64, u64)],
        bytes: &[(&[u8], u64)],
        threadgroups: u64,
        threads_per_group: u64,
    ) {
        if !self.admit(pipeline, (threads_per_group, 1, 1)) {
            return;
        }
        self.begin_dispatch(pipeline);
        let encoder = self.encoder.borrow();
        encoder.set_compute_pipeline_state(pipeline);
        for &(buffer, index, offset) in buffers {
            encoder.set_buffer(index, Some(buffer), offset);
        }
        for &(data, index) in bytes {
            encoder.set_bytes(index, data.len() as u64, data.as_ptr().cast());
        }
        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
    }

    /// [`Self::encode_threadgroups`] with 3D threadgroup and threadgroup-size
    /// counts. The GDN kernels need it: their grids are `(head, row)` and
    /// their threadgroups `(32, 4)` or `(128, 1)`, which the 1D form cannot
    /// express.
    pub fn encode_threadgroups_3d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[(&metal::Buffer, u64, u64)],
        bytes: &[(&[u8], u64)],
        threadgroups: (u64, u64, u64),
        threads_per_group: (u64, u64, u64),
    ) {
        if !self.admit(pipeline, threads_per_group) {
            return;
        }
        self.begin_dispatch(pipeline);
        let encoder = self.encoder.borrow();
        encoder.set_compute_pipeline_state(pipeline);
        for &(buffer, index, offset) in buffers {
            encoder.set_buffer(index, Some(buffer), offset);
        }
        for &(data, index) in bytes {
            encoder.set_bytes(index, data.len() as u64, data.as_ptr().cast());
        }
        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups.0, threadgroups.1, threadgroups.2),
            MTLSize::new(
                threads_per_group.0,
                threads_per_group.1,
                threads_per_group.2,
            ),
        );
    }

    /// Appends one `dispatchThreads` call (non-threadgroup-quantized grid).
    pub fn encode_threads_3d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[(&metal::Buffer, u64, u64)],
        bytes: &[(&[u8], u64)],
        grid: (u64, u64, u64),
        threadgroup: (u64, u64, u64),
    ) {
        if !self.admit(pipeline, threadgroup) {
            return;
        }
        self.begin_dispatch(pipeline);
        let encoder = self.encoder.borrow();
        encoder.set_compute_pipeline_state(pipeline);
        for &(buffer, index, offset) in buffers {
            encoder.set_buffer(index, Some(buffer), offset);
        }
        for &(data, index) in bytes {
            encoder.set_bytes(index, data.len() as u64, data.as_ptr().cast());
        }
        encoder.dispatch_threads(
            MTLSize::new(grid.0, grid.1, grid.2),
            MTLSize::new(threadgroup.0, threadgroup.1, threadgroup.2),
        );
    }

    /// Declares a buffer referenced only indirectly (through an argument
    /// buffer's pointer array) as read by the pass -- Metal's
    /// `useResource(_:usage:.read)`, required for the MoE expert blobs.
    pub fn use_read_buffer(&self, buffer: &metal::Buffer) {
        self.encoder
            .borrow()
            .use_resource(buffer, metal::MTLResourceUsage::Read);
        // Resource state is per encoder, and profiling opens one per
        // dispatch, so it has to re-declare this on each of them.
        if let Some(profile) = &self.profile {
            profile.borrow_mut().note_used_read(buffer);
        }
    }

    /// Encodes a compute-pass memory barrier for the given buffers. Use this
    /// when a later dispatch in the same pass reads data written by an
    /// earlier dispatch.
    pub fn memory_barrier_with_buffers(&self, buffers: &[&metal::Buffer]) {
        let resources: Vec<&metal::ResourceRef> = buffers
            .iter()
            .map(|buffer| {
                let resource: &metal::ResourceRef = buffer;
                resource
            })
            .collect();
        self.encoder
            .borrow()
            .memory_barrier_with_resources(&resources);
    }

    /// Ends encoding, commits, and blocks until the GPU finishes. Shared-
    /// storage outputs are CPU-readable after this returns.
    pub fn commit_and_wait(self) {
        self.commit().wait();
    }

    /// Return a device failure before a caller reads shared output memory.
    /// A dispatch skipped for exceeding the thread limit is an error too, and
    /// then nothing is committed: the pass's outputs keep their old contents.
    pub fn commit_and_wait_checked(self) -> Result<(), super::GpuError> {
        if let Some(violation) = self.violation.borrow_mut().take() {
            // Dropping the pass ends encoding without committing.
            return Err(super::GpuError::InvalidInput(violation));
        }
        let pass = self.commit();
        pass.command_buffer.wait_until_completed();
        crate::submit_probe::completed(&pass.command_buffer);
        let status = pass.command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            warn_on_command_buffer_error(&pass.command_buffer);
            return Err(super::GpuError::InvalidInput(format!(
                "command buffer failed with status {status:?}"
            )));
        }
        Ok(())
    }

    /// [`Self::commit`] for callers that pipeline buffers and still need the
    /// thread-limit guard as an error: returns `Err` without committing when
    /// a dispatch was skipped. Device faults surface later, at the wait.
    pub fn commit_checked(self) -> Result<CommittedPass, super::GpuError> {
        if let Some(violation) = self.violation.borrow_mut().take() {
            return Err(super::GpuError::InvalidInput(violation));
        }
        Ok(self.commit())
    }

    /// [`Self::commit_and_wait`] that also reports the buffer's GPU-side
    /// busy interval in seconds. See [`CommittedPass::wait_with_gpu_time`].
    pub fn commit_and_wait_with_gpu_time(self) -> f64 {
        self.commit().wait_with_gpu_time()
    }

    /// Ends encoding and commits without waiting, handing back something
    /// the caller can wait on later. Buffers committed to one queue execute
    /// in commit order, so a caller can queue follow-on GPU work and then
    /// wait on an earlier buffer to be woken as soon as *its* results are
    /// ready, with the later work still running.
    pub fn commit(mut self) -> CommittedPass {
        self.panic_on_violation();
        // Already ended when `commit_signaling` encoded its signal.
        if !self.ended.get() {
            self.encoder.borrow().end_encoding();
            self.ended.set(true);
        }
        self.command_buffer.commit();
        // Timestamps can only be resolved once the buffer has completed,
        // and a pass may be committed and never waited on (the shared and
        // hit-expert buffers are), so profiling waits here rather than
        // losing those dispatches. This is the serialization the module
        // doc warns about: profiled runs are not throughput runs.
        if let Some(profile) = self.profile.take() {
            self.command_buffer.wait_until_completed();
            profile.into_inner().resolve();
        }
        CommittedPass {
            command_buffer: self.command_buffer.clone(),
            wait_on_drop: false,
        }
    }
}

/// Ends encoding on a pass that is dropped without being committed, which
/// is what every `?` inside an encode sequence does.
///
/// Without this, Metal aborts the process from `-[_MTLCommandEncoder
/// dealloc]` with "Command encoder released without endEncoding", and that
/// assertion is all anyone sees: the real error is still travelling up the
/// stack when the encoder is deallocated, so it never reaches a caller that
/// could print it. Found while opening the first mixed-block-type GGUF
/// install, where a one-line dispatch refusal presented as a Metal crash.
impl Drop for PassEncoder {
    fn drop(&mut self) {
        if !self.ended.get() {
            self.encoder.borrow().end_encoding();
            self.ended.set(true);
        }
    }
}

/// A committed, not-yet-waited-on command buffer.
pub struct CommittedPass {
    command_buffer: metal::CommandBuffer,
    wait_on_drop: bool,
}

/// `waitUntilCompleted` returns NORMALLY on a buffer whose work the GPU
/// aborted (fault, timeout, or a device-wide restart another process
/// triggered), leaving whatever was written before the fault in the output
/// buffers -- which decodes as fluent wrong tokens with no error anywhere.
/// Observed 2026-08-28 as a greedy divergence under concurrent Metal test
/// load (AGENTS.md Gotcha 27's class, but the hidden state was the GPU's,
/// not the cache's). Warn loudly rather than panic: the check must not be
/// able to change tokens, only to make a corrupted run attributable.
// The allow is for objc's `sel_impl!`, whose expansion carries a
// `cfg(feature = "cargo-clippy")` this crate does not declare.
#[allow(unexpected_cfgs)]
pub(crate) fn warn_on_command_buffer_error(command_buffer: &metal::CommandBufferRef) {
    use metal::objc::{msg_send, sel, sel_impl};
    let status = command_buffer.status();
    if status == metal::MTLCommandBufferStatus::Completed {
        return;
    }
    // SAFETY: `error` is a documented `NSError*` accessor on
    // `MTLCommandBuffer`, called on a live, waited-on buffer; `code` is a
    // documented NSInteger accessor on NSError, called only when non-nil.
    #[allow(unsafe_code)]
    let code: i64 = unsafe {
        let err: *mut metal::objc::runtime::Object = msg_send![command_buffer, error];
        if err.is_null() {
            0
        } else {
            msg_send![err, code]
        }
    };
    eprintln!(
        "[gpu] command buffer '{}' finished with status {:?} (error code {}); \
         its outputs are not trustworthy and this generation may be corrupt",
        command_buffer.label(),
        status,
        code
    );
}

impl CommittedPass {
    /// Makes dropping this handle wait for completion. Use this whenever
    /// resources referenced by the pass can otherwise be released after an
    /// error; Metal retaining its buffer objects does not retain external
    /// bytes-no-copy backing allocations.
    pub fn waiting_on_drop(mut self) -> Self {
        self.wait_on_drop = true;
        self
    }

    /// Blocks until this buffer finishes without consuming the readiness
    /// handle. Image tensors retain the handle until their CPU read seam,
    /// while later image command buffers continue in queue order.
    pub fn wait_ref(&self) {
        self.command_buffer.wait_until_completed();
        crate::submit_probe::completed(&self.command_buffer);
        warn_on_command_buffer_error(&self.command_buffer);
    }

    /// Blocks until this buffer finishes. Its shared-storage outputs are
    /// CPU-readable after this returns; buffers committed after it may
    /// still be running.
    pub fn wait(mut self) {
        self.command_buffer.wait_until_completed();
        crate::submit_probe::completed(&self.command_buffer);
        warn_on_command_buffer_error(&self.command_buffer);
        self.wait_on_drop = false;
    }

    /// [`Self::wait`] that also reports the buffer's GPU-side busy
    /// interval (`GPUEndTime - GPUStartTime`) in seconds, the per-buffer
    /// attribution the wall-clock wait cannot give (a wait on a buffer
    /// queued behind others pays for all of them). The accessors are only
    /// valid after completion, hence wait-then-read; metal-rs 0.33 does
    /// not bind them, hence the raw `msg_send`. Returns 0.0 if the device
    /// reports nothing.
    // The allow is for objc's `sel_impl!`, whose expansion carries a
    // `cfg(feature = "cargo-clippy")` this crate does not declare.
    #[allow(unexpected_cfgs)]
    pub fn wait_with_gpu_time(mut self) -> f64 {
        use metal::objc::{msg_send, sel, sel_impl};
        self.command_buffer.wait_until_completed();
        crate::submit_probe::completed(&self.command_buffer);
        warn_on_command_buffer_error(&self.command_buffer);
        // SAFETY: `GPUStartTime`/`GPUEndTime` are documented
        // `CFTimeInterval` (f64) accessors on `MTLCommandBuffer`, called
        // on a live, completed buffer.
        #[allow(unsafe_code)]
        let (start, end): (f64, f64) = unsafe {
            let cb: &metal::CommandBufferRef = &self.command_buffer;
            (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime])
        };
        self.wait_on_drop = false;
        (end - start).max(0.0)
    }
}

#[cfg(feature = "ane-prefill")]
impl PassEncoder {
    /// [`Self::commit`] after encoding a signal of `event` to `value`, so
    /// the host can watch the event instead of waiting on the buffer. The
    /// signal goes after the compute encoder ends, as Metal requires.
    pub(crate) fn commit_signaling(self, event: &metal::EventRef, value: u64) -> CommittedPass {
        self.panic_on_violation();
        self.encoder.borrow().end_encoding();
        self.ended.set(true);
        self.command_buffer.encode_signal_event(event, value);
        self.commit()
    }
}

#[cfg(feature = "ane-prefill")]
impl CommittedPass {
    /// Blocks until this buffer finishes and returns its `(GPUStartTime,
    /// GPUEndTime)` in host seconds, so the ANE prefill split (feature
    /// `ane-prefill`) can place several buffers on one timeline.
    #[allow(unexpected_cfgs)]
    pub fn wait_gpu_start_end(&self) -> (f64, f64) {
        use metal::objc::{msg_send, sel, sel_impl};
        self.command_buffer.wait_until_completed();
        crate::submit_probe::completed(&self.command_buffer);
        warn_on_command_buffer_error(&self.command_buffer);
        // SAFETY: documented `CFTimeInterval` accessors on a live,
        // completed buffer (as in `wait_with_gpu_time`).
        #[allow(unsafe_code)]
        unsafe {
            let cb: &metal::CommandBufferRef = &self.command_buffer;
            (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime])
        }
    }
}

impl Drop for CommittedPass {
    fn drop(&mut self) {
        if self.wait_on_drop {
            self.command_buffer.wait_until_completed();
            warn_on_command_buffer_error(&self.command_buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{threadgroup_threads, PassEncoder};
    use crate::context::MetalContext;

    const SOURCE: &str = "
        #include <metal_stdlib>
        using namespace metal;
        kernel void guard_probe(device uint* out [[buffer(0)]],
                                uint id [[thread_position_in_grid]]) {
            out[0] = id;
        }
    ";

    #[test]
    fn threadgroup_size_is_the_product_and_never_wraps() {
        assert_eq!(threadgroup_threads((32, 4, 1)), 128);
        assert_eq!(threadgroup_threads((u64::MAX, 2, 1)), u64::MAX);
    }

    const SENTINEL: u32 = 0xDEAD_BEEF;

    /// Encodes one `guard_probe` dispatch `extra` threads past the device
    /// limit and hands the pass, the output buffer and the limit to `finish`.
    /// The output word starts at `SENTINEL`, so a skipped dispatch is visible.
    fn probe<R>(extra: u64, finish: impl FnOnce(PassEncoder, &metal::Buffer, u64) -> R) -> R {
        let context = MetalContext::new().expect("Metal device");
        let device = context.device();
        let library = device
            .new_library_with_source(SOURCE, &metal::CompileOptions::new())
            .expect("compile");
        let function = library.get_function("guard_probe", None).expect("function");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");
        let out = device.new_buffer(64, metal::MTLResourceOptions::StorageModeShared);
        // SAFETY: a live 64-byte shared buffer, written before any GPU use.
        #[allow(unsafe_code)]
        unsafe {
            *out.contents().cast::<u32>() = SENTINEL;
        }
        let queue = context.queue();
        let buffer = queue.new_command_buffer().to_owned();
        let encoder = buffer.new_compute_command_encoder().to_owned();
        let pass = PassEncoder::new(buffer, std::cell::RefCell::new(encoder), None);
        let limit = pipeline.max_total_threads_per_threadgroup();
        pass.encode_threadgroups(&pipeline, &[(&out, 0, 0)], &[], 1, limit + extra);
        finish(pass, &out, limit)
    }

    fn first_word(out: &metal::Buffer) -> u32 {
        // SAFETY: a live shared buffer the GPU is no longer using.
        #[allow(unsafe_code)]
        unsafe {
            *out.contents().cast::<u32>()
        }
    }

    #[test]
    fn a_threadgroup_at_the_device_limit_dispatches() {
        probe(0, |pass, out, _| {
            pass.commit_and_wait_checked()
                .expect("dispatch at the limit");
            assert_ne!(first_word(out), SENTINEL, "the dispatch did not run");
        });
    }

    #[test]
    fn a_threadgroup_past_the_limit_is_an_error_on_the_checked_commit() {
        let (result, word, limit) = probe(1, |pass, out, limit| {
            let result = pass.commit_and_wait_checked();
            (result, first_word(out), limit)
        });
        let message = result
            .expect_err("over-limit dispatch must fail")
            .to_string();
        assert!(message.contains("dispatch of '"), "{message}");
        assert!(
            message.contains(&format!("needs {}", limit + 1)),
            "{message}"
        );
        assert!(message.contains(&format!("allows {limit}")), "{message}");
        // Nothing was committed, so the output keeps its old contents.
        assert_eq!(word, SENTINEL);
    }

    #[test]
    fn the_non_waiting_checked_commit_errors_or_commits() {
        let err = probe(1, |pass, _, _| pass.commit_checked().err());
        assert!(err.is_some_and(|e| e.to_string().contains("threads per threadgroup")));
        let word = probe(0, |pass, out, _| {
            pass.commit_checked().expect("dispatch at the limit").wait();
            first_word(out)
        });
        assert_ne!(word, SENTINEL, "the dispatch did not run");
    }

    #[test]
    #[should_panic(expected = "threads per threadgroup")]
    fn a_threadgroup_past_the_limit_panics_on_the_unchecked_commit() {
        probe(1, |pass, _, _| pass.commit_and_wait());
    }

    #[test]
    #[should_panic(expected = "threads per threadgroup")]
    fn a_threadgroup_past_the_limit_panics_on_a_bare_commit() {
        probe(1, |pass, _, _| drop(pass.commit()));
    }
}
