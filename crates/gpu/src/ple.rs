//! Host-side dispatch for `shaders/ple.metal` (`qwen4_exp`'s PLE gate;
//! PORT-LOCAL, `docs/QWEN4_PHASE0.md` section 4).

use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/ple.metal");
const THREADS_PER_GROUP: u64 = 256;

/// `gv[c*H+h] = sigmoid(gate[c]) * value[h]`, `gate[c]` the signed-sqrt of
/// `key[c,:] . query[c,:] / sqrt(H)`. `key`/`query` are both
/// `[groups * group_dim]`, ALREADY grouped-centered-normed
/// (`encode_rms_norm_bf16w_grouped_centered`); `value` is `[group_dim]`,
/// shared across every group. One threadgroup per group.
#[allow(clippy::too_many_arguments)]
pub fn encode_ple_gate(
    context: &mut MetalContext,
    pass: &PassEncoder,
    key: (&metal::Buffer, u64),
    query: (&metal::Buffer, u64),
    value: (&metal::Buffer, u64),
    gv: (&metal::Buffer, u64),
    groups: u32,
    group_dim: u32,
) -> Result<(), GpuError> {
    let pipeline =
        context.pipeline(SOURCE, "ple_gate_fp16", &FunctionConstantValues::new(), b"")?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            (key.0, 0, key.1),
            (query.0, 1, query.1),
            (value.0, 2, value.1),
            (gv.0, 3, gv.1),
        ],
        &[(u32_bytes(&group_dim), 4)],
        groups as u64,
        THREADS_PER_GROUP.min(group_dim.max(1) as u64),
    );
    Ok(())
}
