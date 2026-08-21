//! GDN head geometry and validation preconditions.

use crate::context::GpuError;

/// GDN head geometry, the dispatch-side mirror of
/// `model_io::LinearAttentionConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnShape {
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel_size: u32,
}

impl GdnShape {
    /// Conv channels, and the row width of `mixed_qkv` / `conv_out`.
    pub fn qkv_dim(&self) -> u32 {
        2 * self.num_k_heads * self.key_head_dim + self.num_v_heads * self.value_head_dim
    }

    /// `Hv * Dv`: z-gate width, `y` width, out_proj columns.
    pub fn value_dim(&self) -> u32 {
        self.num_v_heads * self.value_head_dim
    }

    /// The kernels' structural preconditions, duplicated from the Swift
    /// `GDN.init`. Each one is a silent-wrong-answer trap, not a crash:
    /// the delta kernel tiles `Dk` into 32 lanes of `float s[8]`, and its
    /// `dv` grid steps by 4.
    pub fn validate(&self) -> Result<(), GpuError> {
        let bad = |detail: String| Err(GpuError::PipelineCreate(detail));
        if self.key_head_dim == 0 || self.key_head_dim % 32 != 0 {
            return bad(format!(
                "key_head_dim {} must be a positive multiple of 32",
                self.key_head_dim
            ));
        }
        if self.key_head_dim / 32 > 8 {
            return bad(format!(
                "key_head_dim {} exceeds the kernel's 8-register lane tile",
                self.key_head_dim
            ));
        }
        if self.value_head_dim == 0 || self.value_head_dim % 4 != 0 {
            return bad(format!(
                "value_head_dim {} must be a positive multiple of 4",
                self.value_head_dim
            ));
        }
        if self.num_k_heads == 0 || self.num_v_heads % self.num_k_heads != 0 {
            return bad(format!(
                "num_v_heads {} must be a multiple of num_k_heads {}",
                self.num_v_heads, self.num_k_heads
            ));
        }
        if self.conv_kernel_size < 2 {
            return bad(format!(
                "conv_kernel_size {} must be at least 2",
                self.conv_kernel_size
            ));
        }
        Ok(())
    }

    pub(crate) fn head_dim_bytes(&self) -> [([u8; 4], u64); 4] {
        [
            (self.num_k_heads.to_le_bytes(), 7),
            (self.num_v_heads.to_le_bytes(), 8),
            (self.key_head_dim.to_le_bytes(), 9),
            (self.value_head_dim.to_le_bytes(), 10),
        ]
    }
}
