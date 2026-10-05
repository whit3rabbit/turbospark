//! Buffer-level Moonshine frontend kernels. The runtime owns the buffers and
//! can encode all three conv stages in one command pass.

use crate::bytes::{f32_bytes, u32_bytes};
use crate::context::{GpuError, MetalContext, PassEncoder};
use crate::rms_norm::unused_function_constants;
use crate::whisper_encoder::F32View;

const SOURCE: &str = include_str!("shaders/moonshine_frontend.metal");
const THREADS: u64 = 256;

#[allow(clippy::too_many_arguments)]
pub fn encode_moonshine_conv1d(
    context: &mut MetalContext,
    pass: &PassEncoder,
    input: F32View,
    weight: F32View,
    bias: Option<F32View>,
    output: F32View,
    input_len: u32,
    input_channels: u32,
    output_channels: u32,
    kernel_size: u32,
    stride: u32,
) -> Result<u32, GpuError> {
    if input_channels == 0
        || output_channels == 0
        || kernel_size == 0
        || stride == 0
        || input_len < kernel_size
    {
        return Err(GpuError::InvalidInput(
            "Moonshine conv dimensions must be positive and cover the kernel".into(),
        ));
    }
    let output_len = (input_len - kernel_size) / stride + 1;
    let count = output_len.checked_mul(output_channels).ok_or_else(|| {
        GpuError::InvalidInput("Moonshine conv output dimensions overflow".into())
    })?;
    let pipeline = context.pipeline(
        SOURCE,
        "moonshine_conv1d_f32",
        &unused_function_constants(),
        b"",
    )?;
    let has_bias = u32::from(bias.is_some());
    pass.encode_threadgroups(
        &pipeline,
        &[
            input.binding(0),
            weight.binding(1),
            bias.unwrap_or(input).binding(2),
            output.binding(3),
        ],
        &[
            (u32_bytes(&input_len), 4),
            (u32_bytes(&output_len), 5),
            (u32_bytes(&input_channels), 6),
            (u32_bytes(&output_channels), 7),
            (u32_bytes(&kernel_size), 8),
            (u32_bytes(&stride), 9),
            (u32_bytes(&has_bias), 10),
        ],
        (u64::from(count)).div_ceil(THREADS),
        THREADS,
    );
    Ok(output_len)
}

pub fn encode_moonshine_tanh(
    context: &mut MetalContext,
    pass: &PassEncoder,
    input: F32View,
    output: F32View,
    count: u32,
) -> Result<(), GpuError> {
    if count == 0 {
        return Err(GpuError::InvalidInput(
            "Moonshine tanh count is zero".into(),
        ));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "moonshine_tanh_f32",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[input.binding(0), output.binding(1)],
        &[(u32_bytes(&count), 2)],
        (u64::from(count)).div_ceil(THREADS),
        THREADS,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_moonshine_groupnorm(
    context: &mut MetalContext,
    pass: &PassEncoder,
    input: F32View,
    weight: F32View,
    bias: F32View,
    output: F32View,
    channels: u32,
    sequence: u32,
    epsilon: f32,
) -> Result<(), GpuError> {
    if channels == 0 || sequence == 0 || channels.checked_mul(sequence).is_none() {
        return Err(GpuError::InvalidInput(
            "Moonshine GroupNorm dimensions must be positive and bounded".into(),
        ));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "moonshine_groupnorm_f32",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[
            input.binding(0),
            weight.binding(1),
            bias.binding(2),
            output.binding(3),
        ],
        &[
            (u32_bytes(&channels), 4),
            (u32_bytes(&sequence), 5),
            (f32_bytes(&epsilon), 6),
        ],
        1,
        THREADS,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_moonshine_rope(
    context: &mut MetalContext,
    pass: &PassEncoder,
    values: F32View,
    cos: F32View,
    sin: F32View,
    rows: u32,
    heads: u32,
    head_dim: u32,
    rotary: u32,
    position_offset: u32,
) -> Result<(), GpuError> {
    if rows == 0
        || heads == 0
        || head_dim == 0
        || rotary == 0
        || rotary > head_dim
        || rotary % 2 != 0
    {
        return Err(GpuError::InvalidInput(
            "invalid Moonshine RoPE dimensions".into(),
        ));
    }
    let count = rows
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(rotary / 2))
        .ok_or_else(|| GpuError::InvalidInput("Moonshine RoPE dimensions overflow".into()))?;
    let pipeline = context.pipeline(
        SOURCE,
        "moonshine_rope_f32",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[values.binding(0), cos.binding(1), sin.binding(2)],
        &[
            (u32_bytes(&rows), 3),
            (u32_bytes(&heads), 4),
            (u32_bytes(&head_dim), 5),
            (u32_bytes(&rotary), 6),
            (u32_bytes(&position_offset), 7),
        ],
        (u64::from(count)).div_ceil(THREADS),
        THREADS,
    );
    Ok(())
}

pub fn encode_moonshine_embed(
    context: &mut MetalContext,
    pass: &PassEncoder,
    embedding: F32View,
    output: F32View,
    token: u32,
    hidden: u32,
) -> Result<(), GpuError> {
    if hidden == 0 {
        return Err(GpuError::InvalidInput(
            "Moonshine hidden width is zero".into(),
        ));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "moonshine_embed_f32",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[embedding.binding(0), output.binding(1)],
        &[(u32_bytes(&token), 2), (u32_bytes(&hidden), 3)],
        (u64::from(hidden)).div_ceil(THREADS),
        THREADS,
    );
    Ok(())
}

pub fn encode_moonshine_swiglu(
    context: &mut MetalContext,
    pass: &PassEncoder,
    input: F32View,
    output: F32View,
    intermediate: u32,
) -> Result<(), GpuError> {
    if intermediate == 0 {
        return Err(GpuError::InvalidInput(
            "Moonshine intermediate width is zero".into(),
        ));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "moonshine_swiglu_f32",
        &unused_function_constants(),
        b"",
    )?;
    pass.encode_threadgroups(
        &pipeline,
        &[input.binding(0), output.binding(1)],
        &[(u32_bytes(&intermediate), 2)],
        (u64::from(intermediate)).div_ceil(THREADS),
        THREADS,
    );
    Ok(())
}
