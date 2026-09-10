//! MiniMax keeps routing logits in FP32; no extra weight quantization here.
use crate::bytes::u32_bytes;
use crate::context::{GpuError, MetalContext, PassEncoder};
use metal::FunctionConstantValues;

const SOURCE: &str = include_str!("shaders/minimax_router.metal");

#[allow(clippy::too_many_arguments)]
pub fn encode_minimax_router(
    context: &mut MetalContext,
    pass: &PassEncoder,
    weights: (&metal::Buffer, u64),
    x: (&metal::Buffer, u64),
    out: (&metal::Buffer, u64),
    experts: u32,
    hidden: u32,
) -> Result<(), GpuError> {
    let pipeline = context.pipeline(
        SOURCE,
        "minimax_router",
        &FunctionConstantValues::new(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[(weights.0, 0, weights.1), (x.0, 1, x.1), (out.0, 2, out.1)],
        &[(u32_bytes(&hidden), 3)],
        experts as u64,
        32,
    );
    Ok(())
}
