//! Host-side dispatch for the `logit_softcap_softmax` kernel in
//! `shaders/logit.metal` (vendored verbatim from `Metal/Sampling/logit.metal`
//! — the whole file, since the kernel shares helper functions with the rest
//! of that translation unit; only `logit_softcap_softmax` itself is
//! dispatched here). Matches `mrefrust_compute::logit_softcap_softmax`
//! exactly: `softmax(softcap * tanh(logit / softcap))`.
//!
//! The rest of `logit.metal` (the `sample` kernel and the fused lm_head
//! GEMV kernels) is not dispatched; those declare function constants
//! (`FC_HEAD_D`/`FC_HEAD_V`/`FC_HEAD_USE_FC`) this kernel does not
//! reference, so no specialization is needed to build this one pipeline.

use half::f16;
use metal::FunctionConstantValues;

use crate::bytes::{f32_bytes, half_slice_to_le_bytes, read_half_buffer, u32_bytes};
use crate::context::{dispatch_one_threadgroup_per_row, GpuError, MetalContext, PassEncoder};

const SOURCE: &str = include_str!("shaders/logit.metal");
const THREADS_PER_GROUP: u64 = 256;

/// Encoder-level variant of [`logit_softcap_softmax`]: `V` halfs read at
/// `logits`, probabilities written at `probs`, appended to `pass`.
pub fn encode_logit_softcap_softmax(
    context: &mut MetalContext,
    pass: &PassEncoder,
    logits: (&metal::Buffer, u64),
    probs: (&metal::Buffer, u64),
    v: u32,
    softcap: f32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "logit_softcap_softmax",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(logits.0, 0, logits.1), (probs.0, 1, probs.1)],
        &[(u32_bytes(&v), 2), (f32_bytes(&softcap), 3)],
        1,
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// `softmax(softcap * tanh(logit / softcap))`, dispatched on the GPU via
/// `logit_softcap_softmax`. `logits.len()` is the vocab size `V`.
pub fn logit_softcap_softmax(
    context: &mut MetalContext,
    logits: &[f16],
    softcap: f32,
) -> Result<Vec<f16>, GpuError> {
    let v = logits.len() as u32;
    let logits_bytes = half_slice_to_le_bytes(logits);
    let logits_buffer = context.new_buffer_with_data(&logits_bytes);
    let probs_buffer =
        context.new_output_buffer((logits.len() * std::mem::size_of::<u16>()) as u64);

    let pipeline = context.pipeline(
        SOURCE,
        "logit_softcap_softmax",
        &FunctionConstantValues::new(),
        b"",
    )?;
    dispatch_one_threadgroup_per_row(
        context,
        &pipeline,
        &[(&logits_buffer, 0), (&probs_buffer, 1)],
        &[(u32_bytes(&v), 2), (f32_bytes(&softcap), 3)],
        1,
        THREADS_PER_GROUP,
    );

    Ok(read_half_buffer(&probs_buffer, logits.len()))
}
