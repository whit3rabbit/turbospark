//! Fixed-size recurrent state for gated-DeltaNet linear-attention layers
//! (Qwen 3.6). Ported from `Runtime/KVCache/GDNStateManager.swift`. Unlike
//! KV rows, this state does not grow with context: each linear layer owns
//! a delta-rule state `S` (FP32 `[num_v_heads, value_head_dim,
//! key_head_dim]`) and a causal-conv tail of the last `conv_kernel_size -
//! 1` pre-activation rows (FP16 `[conv_kernel_size - 1, qkv_dim]`).

use metal::{Device, MTLResourceOptions};
use model_io::ArchConfig;

const FP32_SIZE: usize = 4;
const FP16_SIZE: usize = 2;

pub struct GdnStateManager {
    /// `Some` only at indices whose layer mask is 2 (linear attention).
    state_buffers: Vec<Option<metal::Buffer>>,
    conv_tail_buffers: Vec<Option<metal::Buffer>>,
    pub state_bytes_per_layer: usize,
    pub conv_tail_bytes_per_layer: usize,
}

impl GdnStateManager {
    pub fn new(device: &Device, config: &ArchConfig) -> Self {
        let la = &config.linear_attention;
        let state_bytes = la.num_v_heads as usize
            * la.value_head_dim as usize
            * la.key_head_dim as usize
            * FP32_SIZE;
        let conv_tail_bytes =
            (la.conv_kernel_size.max(1) - 1) as usize * la.qkv_dim() as usize * FP16_SIZE;

        let num_layers = config.num_layers as usize;
        let mut states = Vec::with_capacity(num_layers);
        let mut tails = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            if !config.layer_is_linear(layer) {
                states.push(None);
                tails.push(None);
                continue;
            }
            assert!(
                state_bytes > 0 && conv_tail_bytes > 0,
                "linear layer present but linear_attention config is empty"
            );
            states.push(Some(device.new_buffer(
                state_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            )));
            tails.push(Some(device.new_buffer(
                conv_tail_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            )));
        }

        let mut manager = Self {
            state_buffers: states,
            conv_tail_buffers: tails,
            state_bytes_per_layer: state_bytes,
            conv_tail_bytes_per_layer: conv_tail_bytes,
        };
        manager.reset();
        manager
    }

    /// Delta-rule state `S` for a linear layer.
    pub fn state_buffer(&self, layer: usize) -> &metal::Buffer {
        self.state_buffers[layer]
            .as_ref()
            .expect("layer is not a linear-attention layer")
    }

    /// Rolling window of the last `conv_kernel_size - 1` mixed_qkv rows.
    pub fn conv_tail_buffer(&self, layer: usize) -> &metal::Buffer {
        self.conv_tail_buffers[layer]
            .as_ref()
            .expect("layer is not a linear-attention layer")
    }

    pub fn is_linear(&self, layer: usize) -> bool {
        self.state_buffers[layer].is_some()
    }

    /// Resets all recurrent state to the empty-context value (zeros): both
    /// the recurrence and the conv define the empty-context state as zero.
    pub fn reset(&mut self) {
        for buffer in self.state_buffers.iter().flatten() {
            zero_buffer(buffer);
        }
        for buffer in self.conv_tail_buffers.iter().flatten() {
            zero_buffer(buffer);
        }
    }
}

fn zero_buffer(buffer: &metal::Buffer) {
    let len = buffer.length() as usize;
    if len == 0 {
        return;
    }
    // SAFETY: `buffer` is a live, CPU-visible (`storageModeShared`) `MTLBuffer`
    // allocated by this module with exactly `len` bytes; writing zeros over
    // its full extent cannot go out of bounds.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::write_bytes(buffer.contents() as *mut u8, 0, len);
    }
}
