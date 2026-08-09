//! Pass encoder and committed pass types (`PassEncoder`, `CommittedPass`).

use std::cell::RefCell;

use metal::{ComputePipelineState, MTLSize};

use crate::dispatch_profile::PassProfile;

/// An open command buffer + serial compute encoder; see
/// [`MetalContext::begin_pass`](super::MetalContext::begin_pass). Every `encode_*` call appends one
/// dispatch; nothing runs until [`PassEncoder::commit_and_wait`] (or
/// [`PassEncoder::commit`]).
pub struct PassEncoder {
    command_buffer: metal::CommandBuffer,
    /// Rebound per dispatch under `MFERENCE_DISPATCH_PROFILE=1` (this
    /// device can only sample counters at encoder boundaries), hence the
    /// cell; exactly one encoder for the whole pass otherwise.
    encoder: RefCell<metal::ComputeCommandEncoder>,
    profile: Option<RefCell<PassProfile>>,
    /// Whether `endEncoding` has been sent. Read by the `Drop` impl below,
    /// which is what stops an error path aborting the process.
    ended: std::cell::Cell<bool>,
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

    /// Under profiling, closes the running encoder and opens a fresh
    /// timestamped one for the dispatch about to be encoded. A no-op
    /// otherwise, which is the production path.
    fn begin_dispatch(&self, pipeline: &ComputePipelineState) {
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

    /// Ends encoding, commits, and blocks until the GPU finishes. Shared-
    /// storage outputs are CPU-readable after this returns.
    pub fn commit_and_wait(self) {
        self.commit().wait();
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
        self.encoder.borrow().end_encoding();
        self.ended.set(true);
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

/// A committed, not-yet-waited-on command buffer. Dropping it without
/// waiting is fine: the queue keeps the buffer alive until it completes.
pub struct CommittedPass {
    command_buffer: metal::CommandBuffer,
}

impl CommittedPass {
    /// Blocks until this buffer finishes. Its shared-storage outputs are
    /// CPU-readable after this returns; buffers committed after it may
    /// still be running.
    pub fn wait(self) {
        self.command_buffer.wait_until_completed();
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
    pub fn wait_with_gpu_time(self) -> f64 {
        use metal::objc::{msg_send, sel, sel_impl};
        self.command_buffer.wait_until_completed();
        // SAFETY: `GPUStartTime`/`GPUEndTime` are documented
        // `CFTimeInterval` (f64) accessors on `MTLCommandBuffer`, called
        // on a live, completed buffer.
        #[allow(unsafe_code)]
        let (start, end): (f64, f64) = unsafe {
            let cb: &metal::CommandBufferRef = &self.command_buffer;
            (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime])
        };
        (end - start).max(0.0)
    }
}
