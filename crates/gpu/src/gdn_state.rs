//! Fixed-size recurrent state for gated-DeltaNet linear-attention layers
//! (Qwen 3.6). Ported from `Runtime/KVCache/GDNStateManager.swift`. Unlike
//! KV rows, this state does not grow with context: each linear layer owns
//! a delta-rule state `S` (FP32 `[num_v_heads, value_head_dim,
//! key_head_dim]`) and a causal-conv tail of the last `conv_kernel_size -
//! 1` pre-activation rows (FP16 `[conv_kernel_size - 1, qkv_dim]`).

use std::sync::{Arc, Mutex};

use metal::{CommandQueue, Device, MTLResourceOptions};
use model_io::ArchConfig;

const FP32_SIZE: usize = 4;
const FP16_SIZE: usize = 2;

/// GPU-private copies of every linear layer's (state, conv tail), `None`
/// at indices that are not linear layers, so the index is the layer index.
type Mirror = Vec<Option<(metal::Buffer, metal::Buffer)>>;

/// A copy of every linear layer's recurrent state, taken by
/// [`GdnStateManager::snapshot`] into GPU-private buffers. Dropping it
/// returns its buffers to the manager's pool, so a speculative loop that
/// takes one per round allocates once rather than every round.
pub struct GdnSnapshot {
    layers: Mirror,
    pool: Arc<Mutex<Vec<Mirror>>>,
}

impl Drop for GdnSnapshot {
    fn drop(&mut self) {
        let layers = std::mem::take(&mut self.layers);
        if let Ok(mut pool) = self.pool.lock() {
            pool.push(layers);
        }
    }
}

/// Manages fixed-size recurrent state buffers for gated-DeltaNet linear attention layers.
pub struct GdnStateManager {
    /// `Some` only at indices whose layer mask is 2 (linear attention).
    state_buffers: Vec<Option<metal::Buffer>>,
    conv_tail_buffers: Vec<Option<metal::Buffer>>,
    /// Number of bytes allocated for the delta-rule state per linear layer.
    pub state_bytes_per_layer: usize,
    /// Number of bytes allocated for the conv tail buffer per linear layer.
    pub conv_tail_bytes_per_layer: usize,
    /// Snapshot buffer sets no live [`GdnSnapshot`] holds. Grows to the
    /// most snapshots ever alive at once (one in the speculative loop).
    snapshot_pool: Arc<Mutex<Vec<Mirror>>>,
}

impl GdnStateManager {
    /// Allocates and initializes recurrent state and conv tail buffers for
    /// all linear layers, at `dilation` 1 -- the width every production
    /// caller of THIS manager uses today (Qwen 3.6's and `qwen4_exp`'s own
    /// GDN mask-2 layers; `qwen4_exp`'s dilated PLE conv binds its own
    /// separately-sized tail rather than going through here).
    pub fn new(device: &Device, config: &ArchConfig) -> Self {
        Self::new_with_dilation(device, config, 1)
    }

    /// [`Self::new`] at an explicit `dilation` (AGENTS.md/CLAUDE.md S9):
    /// `encode_gdn_conv_decode`'s own contract is a `(taps - 1) * dilation`
    /// row tail, which this manager's sizing had never carried past 1 --
    /// latent rather than reachable today, since the one dilated caller
    /// (`families/qwen4/ple.rs`) manages its own tail buffer instead of
    /// this manager's.
    pub fn new_with_dilation(device: &Device, config: &ArchConfig, dilation: usize) -> Self {
        let la = &config.linear_attention;
        let state_bytes = la.num_v_heads as usize
            * la.value_head_dim as usize
            * la.key_head_dim as usize
            * FP32_SIZE;
        let conv_tail_bytes = (la.conv_kernel_size.max(1) - 1) as usize
            * dilation.max(1)
            * la.qkv_dim() as usize
            * FP16_SIZE;

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
            snapshot_pool: Arc::new(Mutex::new(Vec::new())),
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

    /// Returns true if the layer at `layer` index is a linear-attention layer.
    pub fn is_linear(&self, layer: usize) -> bool {
        self.state_buffers[layer].is_some()
    }

    /// Copies every linear layer's recurrent state into a snapshot.
    ///
    /// A KV cache can be rolled back by moving a cursor, because it keeps
    /// one row per position. This cannot: `S` and the conv tail are the
    /// whole history folded into a fixed-size accumulator, and the
    /// delta-rule update is not invertible. So a speculative rollback has
    /// to keep a copy from before the block and replay the accepted prefix
    /// over it.
    ///
    /// The copy is a GPU blit on `queue`, committed WITHOUT waiting. It was
    /// a host memcpy into fresh `Vec`s: on Qwen3.8-27B (48 linear layers,
    /// 147 MiB) that measured ~3.7 ms per verify round with the GPU idle,
    /// ~6% of a block-2 MTP round. Same-queue command buffers execute in
    /// commit order (`crates/gpu` Gotcha 8), so the blit reads the state
    /// before any later pass on `queue` writes it; the caller's next pass
    /// (the verify) is what the copy overlaps with.
    ///
    /// Only valid once the pass that last advanced the state has been
    /// committed to `queue`. Host writes to the live state ([`Self::reset`])
    /// must wait until a command buffer committed after this one has
    /// completed, or they race the pending copy.
    pub fn snapshot(&self, queue: &CommandQueue) -> GdnSnapshot {
        let mirror = self
            .snapshot_pool
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_else(|| self.new_mirror(queue.device()));
        crate::autorelease_pool(|| {
            let command_buffer = queue.new_command_buffer();
            let blit = command_buffer.new_blit_command_encoder();
            for (layer, saved) in mirror.iter().enumerate() {
                let Some((state, conv)) = saved else { continue };
                let state_bytes = self.state_bytes_per_layer as u64;
                let conv_bytes = self.conv_tail_bytes_per_layer as u64;
                blit.copy_from_buffer(self.state_buffer(layer), 0, state, 0, state_bytes);
                blit.copy_from_buffer(self.conv_tail_buffer(layer), 0, conv, 0, conv_bytes);
            }
            blit.end_encoding();
            command_buffer.commit();
        });
        GdnSnapshot {
            layers: mirror,
            pool: Arc::clone(&self.snapshot_pool),
        }
    }

    /// Writes a [`Self::snapshot`] back with a GPU blit on `queue`, and
    /// waits for it, so the state is restored when this returns. The caller
    /// then replays whatever tokens it decided to keep; this restores the
    /// state as of BEFORE the snapshot's block, not as of any position
    /// inside it.
    pub fn restore(&mut self, snapshot: &GdnSnapshot, queue: &CommandQueue) {
        assert_eq!(
            snapshot.layers.len(),
            self.state_buffers.len(),
            "snapshot is from a different model"
        );
        crate::autorelease_pool(|| {
            let command_buffer = queue.new_command_buffer();
            let blit = command_buffer.new_blit_command_encoder();
            for (layer, saved) in snapshot.layers.iter().enumerate() {
                let Some((state, conv)) = saved else { continue };
                let state_bytes = self.state_bytes_per_layer as u64;
                let conv_bytes = self.conv_tail_bytes_per_layer as u64;
                blit.copy_from_buffer(state, 0, self.state_buffer(layer), 0, state_bytes);
                blit.copy_from_buffer(conv, 0, self.conv_tail_buffer(layer), 0, conv_bytes);
            }
            blit.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
    }

    /// One snapshot's worth of GPU-private buffers, matching the live
    /// layout. Private, because only blits ever touch them.
    fn new_mirror(&self, device: &metal::DeviceRef) -> Mirror {
        self.state_buffers
            .iter()
            .map(|state| {
                state.as_ref().map(|_| {
                    (
                        device.new_buffer(
                            self.state_bytes_per_layer as u64,
                            MTLResourceOptions::StorageModePrivate,
                        ),
                        device.new_buffer(
                            self.conv_tail_bytes_per_layer as u64,
                            MTLResourceOptions::StorageModePrivate,
                        ),
                    )
                })
            })
            .collect()
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
