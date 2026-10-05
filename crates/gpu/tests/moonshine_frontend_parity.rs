//! Real-Metal parity for Moonshine's valid conv frontend against the
//! portable speech implementation at small and checkpoint dimensions.
#![cfg(target_os = "macos")]

use turbospark_gpu::{
    encode_moonshine_conv1d, encode_moonshine_embed, encode_moonshine_groupnorm,
    encode_moonshine_rope, encode_moonshine_swiglu, encode_moonshine_tanh, read_f32_buffer,
    F32View, MetalContext,
};

fn unit(index: usize, salt: u64) -> f32 {
    let mut value = (index as u64)
        .wrapping_add(salt)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 40) as f32 / (1u64 << 24) as f32) * 0.2 - 0.1
}

fn run_case(
    in_ch: usize,
    out_ch: usize,
    input_len: usize,
    kernel: usize,
    stride: usize,
    bias: bool,
) {
    let input: Vec<f32> = (0..in_ch * input_len).map(|i| unit(i, 11)).collect();
    let weight: Vec<f32> = (0..out_ch * in_ch * kernel).map(|i| unit(i, 22)).collect();
    let bias_values: Vec<f32> = (0..out_ch).map(|i| unit(i, 33)).collect();
    let bias_slice = bias.then_some(bias_values.as_slice());
    let expected = turbospark_audio::ops::conv1d(
        &input, &weight, bias_slice, in_ch, out_ch, kernel, stride, 0, 1, 1,
    );
    let mut context = MetalContext::new().expect("Metal device");
    let input_buf = context.new_buffer_with_data(&input);
    let weight_buf = context.new_buffer_with_data(&weight);
    let bias_buf = context.new_buffer_with_data(&bias_values);
    let output_buf = context.new_output_buffer((expected.len() * 4) as u64);
    let tanh_buf = context.new_output_buffer((expected.len() * 4) as u64);
    let pass = context.begin_pass();
    let out_len = encode_moonshine_conv1d(
        &mut context,
        &pass,
        F32View::new(&input_buf),
        F32View::new(&weight_buf),
        bias.then_some(F32View::new(&bias_buf)),
        F32View::new(&output_buf),
        input_len as u32,
        in_ch as u32,
        out_ch as u32,
        kernel as u32,
        stride as u32,
    )
    .expect("conv dispatch");
    assert_eq!(out_len as usize * out_ch, expected.len());
    encode_moonshine_tanh(
        &mut context,
        &pass,
        F32View::new(&output_buf),
        F32View::new(&tanh_buf),
        expected.len() as u32,
    )
    .expect("tanh dispatch");
    pass.commit_and_wait();
    let actual = read_f32_buffer(&output_buf, expected.len());
    let tanh = read_f32_buffer(&tanh_buf, expected.len());
    let max_conv = actual
        .iter()
        .zip(&expected)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    let max_tanh = tanh
        .iter()
        .zip(&expected)
        .map(|(got, want)| (got - want.tanh()).abs())
        .fold(0.0f32, f32::max);
    assert!(max_conv < 2e-4, "conv max absolute error {max_conv}");
    assert!(max_tanh < 2e-4, "tanh max absolute error {max_tanh}");
}

#[test]
fn small_hand_shape_and_odd_stride() {
    run_case(2, 3, 11, 3, 2, true);
    run_case(1, 2, 7, 5, 1, false);
}

#[test]
fn moonshine_tiny_conv_shapes() {
    run_case(1, 288, 8_000, 127, 64, false);
    run_case(288, 576, 124, 7, 3, true);
    run_case(576, 288, 40, 3, 2, true);
}

#[test]
fn groupnorm_channel_affine_matches_reference() {
    for (channels, sequence) in [(2, 3), (288, 124)] {
        let count = channels * sequence;
        let input: Vec<f32> = (0..count).map(|i| unit(i, 44)).collect();
        let weights: Vec<f32> = (0..channels).map(|i| 1.0 + unit(i, 55)).collect();
        let biases: Vec<f32> = (0..channels).map(|i| unit(i, 66)).collect();
        let mean = input.iter().sum::<f32>() / count as f32;
        let variance = input.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / count as f32;
        let expected: Vec<f32> = input
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (v - mean) / (variance + 1e-5).sqrt() * weights[i / sequence] + biases[i / sequence]
            })
            .collect();
        let mut context = MetalContext::new().expect("Metal device");
        let input_buf = context.new_buffer_with_data(&input);
        let weight_buf = context.new_buffer_with_data(&weights);
        let bias_buf = context.new_buffer_with_data(&biases);
        let output_buf = context.new_output_buffer((count * 4) as u64);
        let pass = context.begin_pass();
        encode_moonshine_groupnorm(
            &mut context,
            &pass,
            F32View::new(&input_buf),
            F32View::new(&weight_buf),
            F32View::new(&bias_buf),
            F32View::new(&output_buf),
            channels as u32,
            sequence as u32,
            1e-5,
        )
        .expect("groupnorm dispatch");
        pass.commit_and_wait();
        let actual = read_f32_buffer(&output_buf, count);
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 2e-4, "GroupNorm max abs error {max_error}");
    }
}

#[test]
fn decoder_primitives_match_small_known_values() {
    let mut context = MetalContext::new().expect("Metal device");
    let embedding = context.new_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let embedded = context.new_output_buffer(3 * 4);
    let input = context.new_buffer_with_data(&[2.0f32, 3.0, -1.0, 1.0]);
    let gated = context.new_output_buffer(2 * 4);
    let rope_values = context.new_buffer_with_data(&[1.0f32, 0.0, 9.0, 8.0]);
    let cos = context.new_buffer_with_data(&[1.0f32, 0.0]);
    let sin = context.new_buffer_with_data(&[0.0f32, 1.0]);
    let pass = context.begin_pass();
    encode_moonshine_embed(
        &mut context,
        &pass,
        F32View::new(&embedding),
        F32View::new(&embedded),
        1,
        3,
    )
    .unwrap();
    encode_moonshine_swiglu(
        &mut context,
        &pass,
        F32View::new(&input),
        F32View::new(&gated),
        2,
    )
    .unwrap();
    encode_moonshine_rope(
        &mut context,
        &pass,
        F32View::new(&rope_values),
        F32View::new(&cos),
        F32View::new(&sin),
        1,
        1,
        4,
        2,
        1,
    )
    .unwrap();
    pass.commit_and_wait();
    assert_eq!(read_f32_buffer(&embedded, 3), vec![4.0, 5.0, 6.0]);
    let actual = read_f32_buffer(&gated, 2);
    let expected = [
        2.0 * -1.0 / (1.0 + 1.0f32.exp()),
        3.0 / (1.0 + (-1.0f32).exp()),
    ];
    for (got, want) in actual.iter().zip(expected) {
        assert!((got - want).abs() < 1e-6);
    }
    assert_eq!(read_f32_buffer(&rope_values, 4), vec![0.0, 1.0, 9.0, 8.0]);
}

#[test]
fn decoder_primitives_match_tiny_dimensions() {
    let hidden = 288;
    let heads = 8;
    let head_dim = 36;
    let rotary = 32;
    let rows = 45;
    let offset = 7;
    let intermediate = 1152;
    let embedding: Vec<f32> = (0..16 * hidden).map(|i| unit(i, 77)).collect();
    let gated: Vec<f32> = (0..2 * intermediate).map(|i| unit(i, 88)).collect();
    let rotated: Vec<f32> = (0..rows * hidden).map(|i| unit(i, 99)).collect();
    let (cos, sin) = turbospark_audio::ops::rope_tables(rows + offset, rotary, 10_000.0);
    let mut expected_rotated = rotated.clone();
    for row in 0..rows {
        for head in 0..heads {
            for pair in 0..rotary / 2 {
                let index = row * hidden + head * head_dim + pair * 2;
                let angle = (offset + row) * (rotary / 2) + pair;
                let a = expected_rotated[index];
                let b = expected_rotated[index + 1];
                expected_rotated[index] = a * cos[angle] - b * sin[angle];
                expected_rotated[index + 1] = b * cos[angle] + a * sin[angle];
            }
        }
    }
    let expected_gate: Vec<f32> = (0..intermediate)
        .map(|i| gated[i] * gated[intermediate + i] / (1.0 + (-gated[intermediate + i]).exp()))
        .collect();
    let mut context = MetalContext::new().unwrap();
    let embedding_buf = context.new_buffer_with_data(&embedding);
    let embedded = context.new_output_buffer((hidden * 4) as u64);
    let gate_buf = context.new_buffer_with_data(&gated);
    let gate_output = context.new_output_buffer((intermediate * 4) as u64);
    let rotated_buf = context.new_buffer_with_data(&rotated);
    let cos_buf = context.new_buffer_with_data(&cos);
    let sin_buf = context.new_buffer_with_data(&sin);
    let pass = context.begin_pass();
    encode_moonshine_embed(
        &mut context,
        &pass,
        F32View::new(&embedding_buf),
        F32View::new(&embedded),
        13,
        hidden as u32,
    )
    .unwrap();
    encode_moonshine_swiglu(
        &mut context,
        &pass,
        F32View::new(&gate_buf),
        F32View::new(&gate_output),
        intermediate as u32,
    )
    .unwrap();
    encode_moonshine_rope(
        &mut context,
        &pass,
        F32View::new(&rotated_buf),
        F32View::new(&cos_buf),
        F32View::new(&sin_buf),
        rows as u32,
        heads as u32,
        head_dim as u32,
        rotary as u32,
        offset as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    assert_eq!(
        read_f32_buffer(&embedded, hidden),
        embedding[13 * hidden..14 * hidden]
    );
    let gate_max = read_f32_buffer(&gate_output, intermediate)
        .iter()
        .zip(&expected_gate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rope_max = read_f32_buffer(&rotated_buf, rotated.len())
        .iter()
        .zip(&expected_rotated)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(gate_max < 1e-6, "SwiGLU max absolute error {gate_max}");
    assert!(rope_max < 1e-5, "RoPE max absolute error {rope_max}");
}
