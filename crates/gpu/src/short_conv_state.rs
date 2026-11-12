//! Conv-tail state for LFM3's gated short-conv layers. Like
//! [`crate::GdnStateManager`]'s conv tail, it does not grow with context:
//! each conv layer owns one FP16 `[conv_kernel_size - 1, hidden]` buffer
//! holding the last rows of `B * xx` (`shaders/short_conv.metal`).
//!
//! Takes the conv-layer mask explicitly rather than an `ArchConfig`, so the
//! runtime decides which layers are conv layers from the family's own
//! config.

use metal::{Device, MTLResourceOptions};

const FP16_SIZE: usize = 2;

/// One conv-tail buffer per short-conv layer.
pub struct ShortConvStateManager {
    /// `Some` only at conv-layer indices, so the index is the layer index.
    tail_buffers: Vec<Option<metal::Buffer>>,
    /// Bytes per conv layer's tail: `(taps - 1) * hidden * 2`.
    pub tail_bytes_per_layer: usize,
}

impl ShortConvStateManager {
    /// Allocates a zeroed tail for every `true` in `conv_layers`.
    ///
    /// Panics when a conv layer is present but `taps < 2` or `hidden == 0`:
    /// a zero-byte tail cannot be bound and would mean the config is wrong.
    pub fn new(device: &Device, conv_layers: &[bool], hidden: usize, taps: usize) -> Self {
        let tail_bytes = taps.saturating_sub(1) * hidden * FP16_SIZE;
        let tail_buffers = conv_layers
            .iter()
            .map(|&is_conv| {
                is_conv.then(|| {
                    assert!(
                        tail_bytes > 0,
                        "short-conv layer present but hidden {hidden} / taps {taps} give an empty tail"
                    );
                    device.new_buffer(tail_bytes as u64, MTLResourceOptions::StorageModeShared)
                })
            })
            .collect();
        let mut manager = Self {
            tail_buffers,
            tail_bytes_per_layer: tail_bytes,
        };
        manager.reset();
        manager
    }

    /// The tail for a conv layer.
    pub fn tail_buffer(&self, layer: usize) -> &metal::Buffer {
        self.tail_buffers[layer]
            .as_ref()
            .expect("layer is not a short-conv layer")
    }

    /// Returns true if `layer` is a short-conv layer.
    pub fn is_conv(&self, layer: usize) -> bool {
        self.tail_buffers[layer].is_some()
    }

    /// Resets every tail to the empty-context value (zeros). A session
    /// reset must call this alongside the KV rewind, or the previous
    /// generation leaks into every conv layer (docs/NEW_MODEL.md Phase 3).
    /// Host write: only call once no committed pass that reads or writes a
    /// tail is still in flight.
    pub fn reset(&mut self) {
        for buffer in self.tail_buffers.iter().flatten() {
            let len = buffer.length() as usize;
            // SAFETY: `buffer` is a live, CPU-visible (`StorageModeShared`)
            // `MTLBuffer` allocated here with exactly `len` bytes; writing
            // zeros over its full extent cannot go out of bounds.
            #[allow(unsafe_code)]
            unsafe {
                std::ptr::write_bytes(buffer.contents() as *mut u8, 0, len);
            }
        }
    }
}
