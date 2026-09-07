//! Host-side dispatch for `shaders/hyper.metal` (`qwen4_exp`'s hyper-connection
//! mix; PORT-LOCAL, `docs/QWEN4_PHASE0.md` section 3). NOT
//! `HyperConnectionConfig`'s Sinkhorn-normalised mHC (DeepSeek-V4-Flash's, a
//! different mechanism with no kernel here).

use metal::FunctionConstantValues;

use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/hyper.metal");
const THREADS_PER_GROUP: u64 = 256;

fn grid_for(count: u32) -> u64 {
    // AGENTS.md/CLAUDE.md S12: floored at 1, matching rope.rs's own
    // convention -- a `count == 0` caller still dispatches one
    // (harmless, bounds-checked-empty) threadgroup rather than zero.
    (count as u64).div_ceil(THREADS_PER_GROUP).max(1) * THREADS_PER_GROUP
}

/// `mixed[h] = mean over c in [0, C) of w[c*H+h] * normed[c*H+h]`.
///
/// `w` and `normed` are both `[groups * group_dim]` (`C * H`), viewed as
/// `groups` streams of `group_dim` each; `mixed` is `[group_dim]`. One
/// thread per output element, no threadgroup reduction -- see the shader for
/// why `groups` (typically 4) is too small to warrant one.
pub fn encode_hc_mix(
    context: &mut MetalContext,
    pass: &PassEncoder,
    w: (&metal::Buffer, u64),
    normed: (&metal::Buffer, u64),
    mixed: (&metal::Buffer, u64),
    groups: u32,
    group_dim: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(SOURCE, "hc_mix_fp16", &FunctionConstantValues::new(), b"")?;
    pass.encode_threads_3d(
        &pipeline,
        &[
            (w.0, 0, w.1),
            (normed.0, 1, normed.1),
            (mixed.0, 2, mixed.1),
        ],
        &[(u32_bytes(&groups), 3), (u32_bytes(&group_dim), 4)],
        (grid_for(group_dim), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}

/// `hidden[c*H+h] += out[h] * inject_w[c]`, in place -- `hidden` must
/// already hold `raw` (the UN-normalized hyper-connection residual, from
/// before `hc_norm` ran) on entry. Broadcasts the sublayer's `[group_dim]`
/// output against a `[groups]` per-stream gate and scatters it back across
/// the `[groups * group_dim]`-wide stream. No reduction, one thread per
/// output element.
pub fn encode_hc_inject_add(
    context: &mut MetalContext,
    pass: &PassEncoder,
    hidden: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    inject_w: (&metal::Buffer, u64),
    groups: u32,
    group_dim: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "hc_inject_add_fp16",
        &FunctionConstantValues::new(),
        b"",
    )?;
    let total = groups * group_dim;
    pass.encode_threads_3d(
        &pipeline,
        &[
            (hidden.0, 0, hidden.1),
            (out.0, 1, out.1),
            (inject_w.0, 2, inject_w.1),
        ],
        &[(u32_bytes(&groups), 3), (u32_bytes(&group_dim), 4)],
        (grid_for(total), 1, 1),
        (THREADS_PER_GROUP, 1, 1),
    );
    Ok(())
}
