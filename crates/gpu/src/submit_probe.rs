//! How long a finished command buffer sat between the kernel scheduling it
//! and the GPU starting it (`GPUStartTime - kernelStartTime`), for the lab's
//! idle-gap probe (`tslab idle-gap`, LAB-24).
//!
//! Thread-local and installed only by a direct call: with nothing installed
//! the waits cost one thread-local check, and no environment variable turns
//! it on.

use std::cell::RefCell;

thread_local! {
    static DELAYS: RefCell<Option<Vec<f64>>> = const { RefCell::new(None) };
}

/// Start collecting on this thread, dropping anything not yet taken.
pub fn install() {
    DELAYS.with_borrow_mut(|d| *d = Some(Vec::new()));
}

/// The delays in milliseconds since the previous call, in wait order. Empty
/// when nothing is installed.
pub fn take() -> Vec<f64> {
    DELAYS.with_borrow_mut(|d| d.as_mut().map(std::mem::take).unwrap_or_default())
}

/// Stop collecting.
pub fn remove() {
    DELAYS.with_borrow_mut(|d| *d = None);
}

/// Record a COMPLETED buffer. The two accessors are only valid then.
pub(crate) fn completed(buffer: &metal::CommandBufferRef) {
    DELAYS.with_borrow_mut(|d| {
        let Some(delays) = d else {
            return;
        };
        if let Some(ms) = delay_ms(buffer) {
            delays.push(ms);
        }
    });
}

#[allow(unexpected_cfgs)]
fn delay_ms(buffer: &metal::CommandBufferRef) -> Option<f64> {
    use metal::objc::{msg_send, sel, sel_impl};
    // SAFETY: documented `CFTimeInterval` accessors on a live, completed
    // command buffer, both on the host clock.
    #[allow(unsafe_code)]
    let (kernel, gpu): (f64, f64) = unsafe {
        (
            msg_send![buffer, kernelStartTime],
            msg_send![buffer, GPUStartTime],
        )
    };
    // A buffer the GPU never ran reports zeros.
    (kernel > 0.0 && gpu >= kernel).then_some((gpu - kernel) * 1000.0)
}
