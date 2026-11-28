//! Lab-only probes for the idle stall (KERN-05, `docs/designs/speedup-kern.md`
//! Step 1a): after 2 s or more of GPU idle, the first command buffer waits
//! about 200 ms before it starts. Two cheap actions separate the candidate
//! causes before anything is built to fix it:
//!
//! - [`IdleProbe::wake`] commits one empty dispatch. If that absorbs the
//!   stall, the delay is GPU power or scheduling state, not memory.
//! - [`IdleProbe::touch`] reads one word from every page of a buffer. If only
//!   this absorbs it, paging of resident buffers is the likely cause.
//!
//! Both only READ the probed buffer (the design's "small write" would put a
//! store into the resident weights), so neither can change decode output.
//! Nothing in the engine calls this; tslab's idle-gap check does.

use metal::ComputePipelineState;

use crate::context::{GpuError, MetalContext};

/// The VM page on Apple Silicon. One read per page faults in the whole page,
/// which is all the touch arm needs.
pub const TOUCH_PAGE_BYTES: u64 = 16 * 1024;

const SOURCE: &str = "
    #include <metal_stdlib>
    using namespace metal;
    kernel void idle_wake(device uint* out [[buffer(0)]],
                          uint id [[thread_position_in_grid]]) {
        if (id == 0) { out[0] = 1; }
    }
    // One thread per page. The compare keeps the load alive; the sentinel
    // is never expected to match, so `out` is normally left untouched.
    kernel void idle_touch(device const uint* data [[buffer(0)]],
                           device uint* out [[buffer(1)]],
                           constant ulong& words_per_page [[buffer(2)]],
                           constant ulong& pages [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
        if ((ulong)id >= pages) { return; }
        if (data[(ulong)id * words_per_page] == 0x9E3779B9u) { out[1] = id; }
    }
";

/// The two probe pipelines and their scratch word, built once before the
/// timed region so compiling never lands inside a measurement.
pub struct IdleProbe {
    wake: ComputePipelineState,
    touch: ComputePipelineState,
    out: metal::Buffer,
}

impl IdleProbe {
    pub fn new(context: &MetalContext) -> Result<Self, GpuError> {
        let device = context.device();
        let library = device
            .new_library_with_source(SOURCE, &metal::CompileOptions::new())
            .map_err(GpuError::LibraryCompile)?;
        let pipeline = |name: &str| -> Result<ComputePipelineState, GpuError> {
            let function = library
                .get_function(name, None)
                .map_err(|_| GpuError::FunctionNotFound(name.into()))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(GpuError::PipelineCreate)
        };
        Ok(Self {
            wake: pipeline("idle_wake")?,
            touch: pipeline("idle_touch")?,
            out: device.new_buffer(8, metal::MTLResourceOptions::StorageModeShared),
        })
    }

    /// Commits one single-thread dispatch and waits for it.
    pub fn wake(&self, context: &MetalContext) -> Result<(), GpuError> {
        crate::autorelease_pool(|| {
            let pass = context.begin_pass();
            pass.encode_threadgroups(&self.wake, &[(&self.out, 0, 0)], &[], 1, 1);
            pass.commit_checked()?.wait();
            Ok(())
        })
    }

    /// Reads the first word of every [`TOUCH_PAGE_BYTES`] page of `buffer`
    /// and waits. Returns the page count.
    pub fn touch(&self, context: &MetalContext, buffer: &metal::Buffer) -> Result<u64, GpuError> {
        let pages = touch_pages(buffer.length());
        if pages == 0 {
            return Ok(0);
        }
        let words_per_page = TOUCH_PAGE_BYTES / 4;
        let width = self.touch.max_total_threads_per_threadgroup().min(256);
        crate::autorelease_pool(|| {
            let pass = context.begin_pass();
            pass.encode_threads_3d(
                &self.touch,
                &[(buffer, 0, 0), (&self.out, 1, 0)],
                &[
                    (&words_per_page.to_ne_bytes(), 2),
                    (&pages.to_ne_bytes(), 3),
                ],
                (pages, 1, 1),
                (width, 1, 1),
            );
            pass.commit_checked()?.wait();
            Ok(pages)
        })
    }
}

/// Pages whose first word lies inside a buffer of `len` bytes. A partial last
/// page counts when its first word fits.
pub fn touch_pages(len: u64) -> u64 {
    if len < 4 {
        0
    } else {
        (len - 4) / TOUCH_PAGE_BYTES + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_count_covers_every_first_word() {
        assert_eq!(touch_pages(0), 0);
        assert_eq!(touch_pages(3), 0);
        assert_eq!(touch_pages(4), 1);
        assert_eq!(touch_pages(TOUCH_PAGE_BYTES), 1);
        assert_eq!(touch_pages(TOUCH_PAGE_BYTES + 3), 1);
        assert_eq!(touch_pages(TOUCH_PAGE_BYTES + 4), 2);
    }

    #[test]
    fn wake_and_touch_run_and_leave_the_buffer_unchanged() {
        let context = MetalContext::new().expect("Metal device");
        let probe = IdleProbe::new(&context).expect("probe pipelines");
        probe.wake(&context).expect("wake");
        // 3.5 pages of a known pattern; the touch must read, never write.
        let len = TOUCH_PAGE_BYTES * 7 / 2;
        let words: Vec<u32> = (0..len as u32 / 4).collect();
        let buffer = context.device().new_buffer_with_data(
            words.as_ptr().cast(),
            len,
            metal::MTLResourceOptions::StorageModeShared,
        );
        assert_eq!(probe.touch(&context, &buffer).expect("touch"), 4);
        // SAFETY: a live shared buffer of `len` bytes the GPU has finished with.
        #[allow(unsafe_code)]
        let after =
            unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u32>(), words.len()) };
        assert_eq!(after, &words[..]);
    }

    #[test]
    fn touch_sees_every_page() {
        // Plant the sentinel at the first word of the LAST page only: the
        // kernel reports it in out[1], proving the last page was read.
        let context = MetalContext::new().expect("Metal device");
        let probe = IdleProbe::new(&context).expect("probe pipelines");
        let len = TOUCH_PAGE_BYTES * 3;
        let mut words = vec![0u32; len as usize / 4];
        words[(2 * TOUCH_PAGE_BYTES / 4) as usize] = 0x9E37_79B9;
        let buffer = context.device().new_buffer_with_data(
            words.as_ptr().cast(),
            len,
            metal::MTLResourceOptions::StorageModeShared,
        );
        probe.touch(&context, &buffer).expect("touch");
        // SAFETY: the probe's 8-byte shared scratch, idle after the wait.
        #[allow(unsafe_code)]
        let hit = unsafe { *probe.out.contents().cast::<u32>().add(1) };
        assert_eq!(hit, 2);
    }
}
