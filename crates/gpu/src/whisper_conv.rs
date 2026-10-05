//! Host-side dispatch for the `whisper_conv1d3_gelu` kernel in
//! `shaders/whisper_conv.metal` -- the whisper front end's conv1/conv2
//! (kernel 3, symmetric padding, stride 1 or 2, exact-erf GELU), the first
//! Metal kernel on the speech path.
//!
//! The CPU reference is `compute::whisper::conv1d3_gelu`; the parity test
//! in `tests/whisper_conv_parity.rs` runs both on real hardware and
//! bounds the difference. Inputs and outputs are f32, matching the mel
//! frontend's output and the encoder stack's expectations; the shapes are
//! small (worst case 384 x 1500) so the kernel maps one thread per output
//! element with no tiling.

use crate::bytes::read_f32_buffer;
use crate::context::{dispatch_one_threadgroup_per_row, GpuError, MetalContext, PassEncoder};
use crate::whisper_encoder::F32View;

pub(crate) static SOURCE: &str = include_str!("shaders/whisper_conv.metal");
const THREADS_PER_GROUP: u64 = 256;

/// Encodes one conv1d(k3, stride, pad) + bias + exact-erf GELU dispatch on
/// the caller's pass: `input` is `[in_ch * t_len]` band-major, `weight` is
/// `[out_ch * in_ch * 3]` flat, `out` receives `[out_ch * out_t]` with
/// `out_t = (t_len + 2*pad - 3)/stride + 1`. The buffer-level form behind
/// [`whisper_conv1d3_gelu`]: the runtime's Metal engine keeps the weights
/// and activations resident and sequences the two convs inside one pass,
/// so nothing here copies host data.
#[allow(clippy::too_many_arguments)]
pub fn encode_whisper_conv1d3_gelu(
    context: &mut MetalContext,
    pass: &PassEncoder,
    input: F32View<'_>,
    weight: F32View<'_>,
    bias: F32View<'_>,
    out: F32View<'_>,
    in_ch: u32,
    t_len: u32,
    out_ch: u32,
    stride: u32,
    pad: u32,
) -> Result<(), GpuError> {
    if in_ch == 0 || t_len == 0 || out_ch == 0 || stride == 0 {
        return Err(GpuError::InvalidInput(
            "whisper conv dims must be positive".to_string(),
        ));
    }
    let padded = t_len + 2 * pad;
    if padded < 3 {
        return Err(GpuError::InvalidInput(
            "whisper conv kernel 3 needs at least 3 frames".to_string(),
        ));
    }
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_conv1d3_gelu",
        &crate::rms_norm::unused_function_constants(),
        b"",
    )?;
    let threads = (out_ch as u64) * ((padded - 3) / stride + 1) as u64;
    // Explicit argument indices: F32View::binding() always reports 0, and
    // every view here must land on its own declared slot.
    pass.encode_threadgroups(
        &pipeline,
        &[
            (input.buffer, 0, input.byte_offset()),
            (weight.buffer, 1, weight.byte_offset()),
            (bias.buffer, 2, bias.byte_offset()),
            (out.buffer, 3, out.byte_offset()),
        ],
        &[
            (crate::bytes::u32_bytes(&in_ch), 4),
            (crate::bytes::u32_bytes(&t_len), 5),
            (crate::bytes::u32_bytes(&out_ch), 6),
            (crate::bytes::u32_bytes(&stride), 7),
            (crate::bytes::u32_bytes(&pad), 8),
        ],
        threads.div_ceil(THREADS_PER_GROUP),
        THREADS_PER_GROUP,
    );
    Ok(())
}

/// Runs one conv1d(k3, stride, pad) + bias + exact-erf GELU on Metal:
/// `input` is `[in_ch * t_len]` band-major, `weight` is
/// `[out_ch * in_ch * 3]` flat, and the returned vector is
/// `[out_ch * out_t]` with `out_t = (t_len + 2*pad - 3)/stride + 1`.
#[allow(clippy::too_many_arguments)]
pub fn whisper_conv1d3_gelu(
    context: &mut MetalContext,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_ch: usize,
    t_len: usize,
    out_ch: usize,
    stride: usize,
    pad: usize,
) -> Result<Vec<f32>, GpuError> {
    let out_t = validate_shape(
        input.len(),
        weight.len(),
        bias.len(),
        in_ch,
        t_len,
        out_ch,
        stride,
        pad,
    )?;
    let input_buffer = context.new_buffer_with_data(&f32_le_bytes(input));
    let weight_buffer = context.new_buffer_with_data(&f32_le_bytes(weight));
    let bias_buffer = context.new_buffer_with_data(&f32_le_bytes(bias));
    let out_buffer = context.new_output_buffer((out_ch * out_t * 4) as u64);

    // Scalar params bound one per buffer index, exactly as the shader
    // declares them (4 = in_ch ... 8 = pad).
    let in_ch_u32 = in_ch as u32;
    let t_len_u32 = t_len as u32;
    let out_ch_u32 = out_ch as u32;
    let stride_u32 = stride as u32;
    let pad_u32 = pad as u32;

    // No function constants: the kernel has one fixed shape per dispatch.
    let pipeline = context.pipeline(
        SOURCE,
        "whisper_conv1d3_gelu",
        &crate::rms_norm::unused_function_constants(),
        b"",
    )?;
    let threads = (out_ch * out_t) as u64;
    let threadgroups = threads.div_ceil(THREADS_PER_GROUP);
    dispatch_one_threadgroup_per_row(
        context,
        &pipeline,
        &[
            (&input_buffer, 0),
            (&weight_buffer, 1),
            (&bias_buffer, 2),
            (&out_buffer, 3),
        ],
        &[
            (crate::bytes::u32_bytes(&in_ch_u32), 4),
            (crate::bytes::u32_bytes(&t_len_u32), 5),
            (crate::bytes::u32_bytes(&out_ch_u32), 6),
            (crate::bytes::u32_bytes(&stride_u32), 7),
            (crate::bytes::u32_bytes(&pad_u32), 8),
        ],
        threadgroups,
        THREADS_PER_GROUP,
    );
    Ok(read_f32_buffer(&out_buffer, out_ch * out_t))
}

/// f32 slice as little-endian bytes. Buffers here are small (the whisper
/// front end's largest input is 80 x 3000 floats), so the copy is cheaper
/// than the unsafe reinterpret the hot path would want.
fn f32_le_bytes(slice: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(slice));
    for v in slice {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn validate_shape(
    input_len: usize,
    weight_len: usize,
    bias_len: usize,
    in_ch: usize,
    t_len: usize,
    out_ch: usize,
    stride: usize,
    pad: usize,
) -> Result<usize, GpuError> {
    let invalid = || {
        GpuError::InvalidInput(
            "Whisper convolution dimensions or buffer lengths are invalid".to_string(),
        )
    };
    if [in_ch, t_len, out_ch, stride].contains(&0)
        || [in_ch, t_len, out_ch, stride, pad]
            .iter()
            .any(|&n| n > i32::MAX as usize)
    {
        return Err(invalid());
    }
    let padded = pad
        .checked_mul(2)
        .and_then(|n| t_len.checked_add(n))
        .filter(|&n| n >= 3 && n <= i32::MAX as usize)
        .ok_or_else(invalid)?;
    let out_t = (padded - 3) / stride + 1;
    let expected_input = in_ch
        .checked_mul(t_len)
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(invalid)?;
    let expected_weight = out_ch
        .checked_mul(in_ch)
        .and_then(|n| n.checked_mul(3))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(invalid)?;
    out_ch
        .checked_mul(out_t)
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(invalid)?;
    if input_len != expected_input || weight_len != expected_weight || bias_len != out_ch {
        return Err(invalid());
    }
    Ok(out_t)
}

#[cfg(test)]
mod bounds_regression {
    use super::*;

    #[test]
    fn rejects_invalid_shader_shapes_before_dispatch() {
        assert_eq!(validate_shape(10, 6, 1, 2, 5, 1, 1, 1).unwrap(), 5);
        assert!(validate_shape(9, 6, 1, 2, 5, 1, 1, 1).is_err());
        assert!(validate_shape(10, 5, 1, 2, 5, 1, 1, 1).is_err());
        assert!(validate_shape(10, 6, 0, 2, 5, 1, 1, 1).is_err());
        assert!(validate_shape(2, 3, 1, 1, 2, 1, 1, 0).is_err());
        assert!(validate_shape(10, 6, 1, 2, 5, 1, 0, 1).is_err());
        assert!(validate_shape(10, 6, 1, 2, usize::MAX, 1, 1, 1).is_err());
    }
}
