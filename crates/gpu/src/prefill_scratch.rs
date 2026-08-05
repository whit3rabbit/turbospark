//! Chunked-prefill scratch-buffer sizing and allocation. Ported from
//! `Runtime/Prefill/PrefillChunkScratch.swift`: `PrefillChunkScratchLayout`
//! computes every intermediate buffer's element count from an `ArchConfig`
//! and a chunk size (pure arithmetic, no GPU access); `PrefillChunkScratchBuffers`
//! allocates one real `MTLBuffer` per field from that layout. Neither is
//! wired to a dispatch yet — no chunked-prefill tile-pipeline kernel is
//! vendored (see `DEVIATIONS.md`) — but the scratch-space accounting this
//! provides is what such a kernel's host-side driver would allocate against.

use metal::{Device, MTLResourceOptions};
use model_io::ArchConfig;

const FP16_SIZE: usize = 2;
const U32_SIZE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefillChunkScratchLayout {
    pub chunk_tokens: usize,
    pub hidden_size: usize,
    pub max_q_elements_per_token: usize,
    pub max_kv_elements_per_token: usize,
    pub shared_intermediate: usize,
    pub routed_intermediate: usize,
    pub top_k: usize,
    pub routed_pair_microbatch_rows: usize,
    /// Rows of the widest per-token projection written into `q`: the packed
    /// `[query; gate]` q_proj when `attn_output_gate`, and the gated-
    /// DeltaNet `in_proj_qkv` rows on architectures with linear-attention
    /// layers.
    pub q_proj_elements_per_token: usize,
    /// Non-zero when the architecture gates attention output (Qwen).
    pub attn_gate_elements_per_token: usize,
    pub gdn_qkv_dim: usize,
    pub gdn_value_dim: usize,
    pub gdn_v_heads: usize,
    /// Non-zero when the shared expert output is scalar-gated (Qwen).
    pub shared_scalar_gate_elements: usize,
}

impl PrefillChunkScratchLayout {
    pub const MAX_CHUNK_TOKENS: usize = foundation::PrefillRuntimeConfig::MAX_CHUNK_TOKENS;

    pub fn new(
        config: &ArchConfig,
        chunk_tokens: usize,
        routed_pair_microbatch_rows: usize,
    ) -> Self {
        let chunk_tokens = chunk_tokens.clamp(1, Self::MAX_CHUNK_TOKENS);
        let q_dim = config.num_heads as usize * config.head_dim.max(config.full_head_dim) as usize;
        let has_linear = config.has_linear_attention_layers();
        let la = &config.linear_attention;
        Self {
            chunk_tokens,
            hidden_size: config.hidden_size as usize,
            max_q_elements_per_token: q_dim,
            max_kv_elements_per_token: (config.num_kv_heads as usize * config.head_dim as usize)
                .max(config.num_full_kv_heads as usize * config.full_head_dim as usize),
            shared_intermediate: config.intermediate_size as usize,
            routed_intermediate: config.moe_intermediate_size as usize,
            top_k: config.top_k_experts as usize,
            routed_pair_microbatch_rows: routed_pair_microbatch_rows.clamp(1, 128),
            q_proj_elements_per_token: (if config.attn_output_gate {
                2 * q_dim
            } else {
                q_dim
            })
            .max(if has_linear { la.qkv_dim() as usize } else { 0 }),
            attn_gate_elements_per_token: if config.attn_output_gate { q_dim } else { 0 },
            gdn_qkv_dim: if has_linear { la.qkv_dim() as usize } else { 0 },
            gdn_value_dim: if has_linear {
                la.value_dim() as usize
            } else {
                0
            },
            gdn_v_heads: if has_linear {
                la.num_v_heads as usize
            } else {
                0
            },
            shared_scalar_gate_elements: if config.shared_expert_gated { 1 } else { 0 },
        }
    }

    pub fn hidden_elements(&self) -> usize {
        self.chunk_tokens * self.hidden_size
    }
    pub fn normed_elements(&self) -> usize {
        self.hidden_elements()
    }
    pub fn q_elements(&self) -> usize {
        self.chunk_tokens * self.q_proj_elements_per_token
    }
    pub fn attn_elements(&self) -> usize {
        self.chunk_tokens * self.max_q_elements_per_token
    }
    pub fn attn_q_elements(&self) -> usize {
        self.chunk_tokens * self.attn_gate_elements_per_token
    }
    pub fn attn_gate_elements(&self) -> usize {
        self.attn_q_elements()
    }
    pub fn gdn_conv_out_elements(&self) -> usize {
        self.chunk_tokens * self.gdn_qkv_dim
    }
    pub fn gdn_z_elements(&self) -> usize {
        self.chunk_tokens * self.gdn_value_dim
    }
    pub fn gdn_a_elements(&self) -> usize {
        self.chunk_tokens * self.gdn_v_heads
    }
    pub fn gdn_b_elements(&self) -> usize {
        self.gdn_a_elements()
    }
    pub fn gdn_y_elements(&self) -> usize {
        self.gdn_z_elements()
    }
    pub fn shared_scalar_gate_buffer_elements(&self) -> usize {
        self.chunk_tokens * self.shared_scalar_gate_elements
    }
    pub fn k_stage_elements(&self) -> usize {
        self.chunk_tokens * self.max_kv_elements_per_token
    }
    pub fn v_stage_elements(&self) -> usize {
        self.k_stage_elements()
    }
    pub fn attention_output_elements(&self) -> usize {
        self.attn_elements()
    }
    pub fn dense_x_elements(&self) -> usize {
        self.hidden_elements()
    }
    pub fn routed_x_elements(&self) -> usize {
        self.hidden_elements()
    }
    pub fn router_x_elements(&self) -> usize {
        self.hidden_elements()
    }
    pub fn h1_elements(&self) -> usize {
        self.hidden_elements()
    }
    pub fn h2_elements(&self) -> usize {
        self.hidden_elements()
    }
    pub fn route_partial_elements(&self) -> usize {
        self.chunk_tokens * self.top_k * self.hidden_size
    }
    pub fn route_id_elements(&self) -> usize {
        self.chunk_tokens * self.top_k
    }
    pub fn route_weight_elements(&self) -> usize {
        self.route_id_elements()
    }
    pub fn shared_expert_scratch_elements(&self) -> usize {
        self.shared_intermediate
    }
    pub fn routed_gate_up_act_elements(&self) -> usize {
        3 * self.routed_pair_microbatch_rows * self.routed_intermediate
    }
    pub fn routed_down_output_elements(&self) -> usize {
        self.routed_pair_microbatch_rows * self.hidden_size
    }

    pub fn device_private_bytes(&self) -> usize {
        let fp16_elements = self.hidden_elements()
            + self.normed_elements()
            + self.q_elements()
            + self.k_stage_elements()
            + self.v_stage_elements()
            + self.attention_output_elements()
            + self.dense_x_elements()
            + self.routed_x_elements()
            + self.router_x_elements()
            + self.h1_elements()
            + self.h2_elements()
            + self.route_partial_elements()
            + 3 * self.shared_expert_scratch_elements()
            + self.routed_gate_up_act_elements()
            + self.routed_down_output_elements()
            + self.attn_q_elements()
            + self.attn_gate_elements()
            + self.gdn_conv_out_elements()
            + self.gdn_z_elements()
            + self.gdn_a_elements()
            + self.gdn_b_elements()
            + self.gdn_y_elements()
            + self.shared_scalar_gate_buffer_elements();
        fp16_elements * FP16_SIZE
    }

    pub fn shared_metadata_bytes(&self) -> usize {
        self.route_id_elements() * U32_SIZE + self.route_weight_elements() * FP16_SIZE
    }

    pub fn total_persistent_bytes(&self) -> usize {
        self.device_private_bytes() + self.shared_metadata_bytes()
    }
}

/// One real `MTLBuffer` per scratch field in `layout`. `attn_q`, `attn_gate`,
/// `gdn_*`, and `shared_scalar_gate` are placeholder-sized (their element
/// count is `0` on architectures that don't use them; buffer allocation
/// floors that at one element) so the struct stays non-optional, matching
/// the Swift original.
pub struct PrefillChunkScratchBuffers {
    pub layout: PrefillChunkScratchLayout,
    pub hidden: metal::Buffer,
    pub normed: metal::Buffer,
    pub q: metal::Buffer,
    pub k_stage: metal::Buffer,
    pub v_stage: metal::Buffer,
    pub attention_output: metal::Buffer,
    pub dense_x: metal::Buffer,
    pub routed_x: metal::Buffer,
    pub router_x: metal::Buffer,
    pub h1: metal::Buffer,
    pub h2: metal::Buffer,
    pub route_partials: metal::Buffer,
    pub route_ids: metal::Buffer,
    pub route_weights: metal::Buffer,
    pub shared_gate_scratch: metal::Buffer,
    pub shared_up_scratch: metal::Buffer,
    pub shared_act_scratch: metal::Buffer,
    pub routed_gate_up_act_scratch: metal::Buffer,
    pub routed_down_scratch: metal::Buffer,
    pub attn_q: metal::Buffer,
    pub attn_gate: metal::Buffer,
    pub gdn_conv_out: metal::Buffer,
    pub gdn_z: metal::Buffer,
    pub gdn_a: metal::Buffer,
    pub gdn_b: metal::Buffer,
    pub gdn_y: metal::Buffer,
    pub shared_scalar_gate: metal::Buffer,
}

impl PrefillChunkScratchBuffers {
    pub fn allocate(device: &Device, layout: PrefillChunkScratchLayout) -> Self {
        let private = |elements: usize| -> metal::Buffer {
            device.new_buffer(
                (elements.max(1) * FP16_SIZE) as u64,
                MTLResourceOptions::StorageModePrivate,
            )
        };
        let shared = |bytes: usize| -> metal::Buffer {
            device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared)
        };

        Self {
            hidden: private(layout.hidden_elements()),
            normed: private(layout.normed_elements()),
            q: private(layout.q_elements()),
            k_stage: private(layout.k_stage_elements()),
            v_stage: private(layout.v_stage_elements()),
            attention_output: private(layout.attention_output_elements()),
            dense_x: private(layout.dense_x_elements()),
            routed_x: private(layout.routed_x_elements()),
            router_x: private(layout.router_x_elements()),
            h1: private(layout.h1_elements()),
            h2: private(layout.h2_elements()),
            route_partials: private(layout.route_partial_elements()),
            route_ids: shared(layout.route_id_elements() * U32_SIZE),
            route_weights: shared(layout.route_weight_elements() * FP16_SIZE),
            shared_gate_scratch: private(layout.shared_expert_scratch_elements()),
            shared_up_scratch: private(layout.shared_expert_scratch_elements()),
            shared_act_scratch: private(layout.shared_expert_scratch_elements()),
            routed_gate_up_act_scratch: private(layout.routed_gate_up_act_elements()),
            routed_down_scratch: private(layout.routed_down_output_elements()),
            attn_q: private(layout.attn_q_elements()),
            attn_gate: private(layout.attn_gate_elements()),
            gdn_conv_out: private(layout.gdn_conv_out_elements()),
            gdn_z: private(layout.gdn_z_elements()),
            gdn_a: private(layout.gdn_a_elements()),
            gdn_b: private(layout.gdn_b_elements()),
            gdn_y: private(layout.gdn_y_elements()),
            shared_scalar_gate: private(layout.shared_scalar_gate_buffer_elements()),
            layout,
        }
    }
}
